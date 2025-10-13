# -*- coding: utf-8 -*-
"""
Monthly momentum backtest — optimized, metrics-rich, reproducible.

Improvements:
  • One-shot OHLC fetch per ticker over the whole backtest window.
  • ThreadPool + retry/backoff; optional on-disk Parquet cache.
  • Full metrics (CAGR, AnnVol, Sharpe, Sortino, MaxDD, hit rate, skew, kurt, turnover).
  • Rolling-12M Sharpe, drawdown, and leg-level cumulative plots.

Classic parity:
  • Same 12-1 window, same daily rounding, same SPY merge, same monthly forward windows.
"""

import os, time, math, pathlib
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlalchemy
import mysql.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

# Optional advanced features
from custom_relative_market_features import add_entropy_info_features, ewm_moments, latest_per_ticker

# -------------------- CONFIG --------------------
polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

SCORING_MODE = os.getenv("SCORING_MODE", "classic")  # "classic","vol_scaled","ewma","ewma_vol_scaled"
USE_ENTROPY_FILTER = bool(int(os.getenv("USE_ENTROPY_FILTER", "0")))
ENTROPY_CFG = dict(pe_window=126, pe_m=4, pe_tau=1, se_window=128, cs_bins=21)

EWM_HALFLIFE = int(os.getenv("EWM_HALFLIFE", "63"))

MAX_WORKERS = int(os.getenv("MAX_WORKERS_BACKTEST", "12"))
TIMEOUT = (5, 60)
USER_AGENT = "mtum-backtest/2.0"

ENABLE_DISK_CACHE = bool(int(os.getenv("ENABLE_DISK_CACHE", "1")))
CACHE_DIR = pathlib.Path(os.getenv("CACHE_DIR", "./cache_polygon"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLOT_FIGSIZE = (10, 6)
# ------------------------------------------------

def _build_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=MAX_WORKERS * 2))
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s
SESSION = _build_session()

def _get_json(url: str, params: dict | None = None) -> dict:
    params = params or {}
    try:
        r = SESSION.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                time.sleep(float(ra)); r = SESSION.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"HTTP error: {e} | URL={url} | params={params}"); return {}

# Calendar & months
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date="2019-01-01", end_date=(datetime.today() - timedelta(days=1))).index
months = trading_dates.to_series().dt.tz_localize(None).dt.normalize().drop_duplicates(keep="first").dt.strftime("%Y-%m-%d").values

# Universe PIT (exactly your table)
universe = pd.read_sql("historical_liquid_tickers_polygon", con=engine).drop_duplicates(subset=["date", "ticker"])
all_months = np.sort(universe["date"].drop_duplicates().values)
if all_months.size < 2:
    raise RuntimeError("Universe table has <2 months of data; run PIT builder first.")

# Global fetch window: earliest start needed to latest end needed
first_month = pd.to_datetime(all_months[0])
last_month  = pd.to_datetime(all_months[-1])
global_start = (first_month - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
global_end   = (last_month + timedelta(days=35)).strftime("%Y-%m-%d")

# SPY benchmark once
spy = _get_json(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{last_month.strftime('%Y-%m-%d')}",
    {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key}
)
benchmark_data = pd.json_normalize(spy.get("results", []))
if benchmark_data.empty:
    raise RuntimeError("Failed to fetch SPY benchmark.")
benchmark_data = benchmark_data.set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# Union of tickers across the backtest
all_tickers = universe["ticker"].drop_duplicates().astype(str).tolist()

def _cache_path(ticker: str) -> pathlib.Path:
    return CACHE_DIR / f"stocks_{ticker}_{global_start}_{global_end}.parquet"

def _fetch_ticker_bars(ticker: str) -> pd.DataFrame:
    """Fetch 1D adjusted bars over the entire backtest window for one ticker (with cache)."""
    pth = _cache_path(ticker)
    if ENABLE_DISK_CACHE and pth.exists():
        try:
            df = pd.read_parquet(pth)
            if not df.empty:
                return df
        except Exception:
            pass

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{global_start}/{global_end}"
    payload = _get_json(url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key})
    rows = payload.get("results", [])
    # If there's pagination, follow next_url
    next_url = payload.get("next_url")
    while next_url:
        pay2 = _get_json(next_url, {"apiKey": polygon_api_key})
        rows.extend(pay2.get("results", []))
        next_url = pay2.get("next_url")

    if not rows:
        return pd.DataFrame([])

    df = pd.json_normalize(rows).set_index("t")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")
    if ENABLE_DISK_CACHE:
        try:
            df.to_parquet(pth, index=True)
        except Exception:
            pass
    return df

# Prefetch all tickers once (concurrently)
print(f"Fetching {len(all_tickers)} tickers once for {global_start}..{global_end}")
bars_map: dict[str, pd.DataFrame] = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(_fetch_ticker_bars, t): t for t in all_tickers}
    for i, f in enumerate(as_completed(futs)):
        tkr = futs[f]
        df = f.result()
        bars_map[tkr] = df
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_tickers)} fetched")

def _score_month(month: str):
    """Return (full_period_ticker_data, top_decile, bot_decile) for a single month."""
    month_dt = pd.to_datetime(month)
    next_month = np.sort(all_months[all_months > month])[0] if np.any(all_months > month) else (month_dt + timedelta(days=31)).strftime("%Y-%m-%d")
    last_month_date = (month_dt - timedelta(days=30)).strftime("%Y-%m-%d")

    pit_date = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)[-1]
    month_universe = universe[universe["date"] == pit_date].drop_duplicates(subset=["ticker"], keep="last")["ticker"].astype(str).values

    rows = []
    # (Optional) panel parts for entropy filter
    entropy_panel_parts = []

    for ticker in month_universe:
        df = bars_map.get(ticker)
        if df is None or df.empty:
            continue

        # trailing 12-1 slice
        tmo = df[df["date"] < last_month_date].copy().tail(252)
        if tmo.shape[0] < 252:
            continue

        # Merge with SPY; QG's exact rounding points
        bmerge = pd.merge(benchmark_data[["c", "date"]], tmo[["c", "date"]], on="date")
        bmerge["benchmark_pct_change"] = np.round(bmerge["c_x"].pct_change() * 100, 2).fillna(0)
        bmerge["ticker_pct_change"]    = np.round(bmerge["c_y"].pct_change() * 100, 2).fillna(0)

        cov = np.cov(bmerge["ticker_pct_change"], bmerge["benchmark_pct_change"])
        var_b = np.var(bmerge["benchmark_pct_change"])
        beta = (cov[0, 1] / var_b) if var_b != 0 else 0.0

        tkr_ret_period = np.round(((tmo["c"].iloc[-1] - tmo["c"].iloc[0]) / tmo["c"].iloc[0]) * 100, 2)
        std_daily = bmerge["ticker_pct_change"].std() * np.sqrt(252)
        sharpe = (tkr_ret_period / std_daily) if std_daily != 0 else 0.0

        if SCORING_MODE == "classic":
            mom_score = beta * sharpe
        else:
            if SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
                mi = tmo[["c", "date"]].copy(); mi["ticker"] = ticker
                mi = mi.set_index(["date", "ticker"]).sort_index()
                mi["returns"] = mi.groupby(level="ticker")["c"].pct_change()
                em = ewm_moments(mi, ret_col="returns", halflife=EWM_HALFLIFE, use_lagged_data=True).groupby(level="ticker").tail(1)
                ewm_mean = float(em["ewm_mean"].iloc[0]) * 100.0
                ewm_vol  = float(em["ewm_vol_ann"].iloc[0])
                mom_score = 0.0 if ewm_vol == 0 else beta * (ewm_mean / ewm_vol)
            else:
                mom_score = beta * (tkr_ret_period / (std_daily if std_daily != 0 else 1e-12))

        # forward window [month, next_month]
        next_window = df[(df["date"] >= month) & (df["date"] <= next_month)].copy().sort_index()
        fwd_ret = np.round(((next_window["c"].iloc[-1] - next_window["c"].iloc[0]) / next_window["c"].iloc[0]) * 100, 2) if next_window.shape[0] >= 2 else np.nan

        rows.append({"entry_date": month, "ticker": ticker, "beta": beta, "sharpe": sharpe,
                     "12-1_return": tkr_ret_period, "mom_score": mom_score, "forward_returns": fwd_ret, "exit_date": next_month})

        if USE_ENTROPY_FILTER:
            tmp = df[df["date"] < last_month_date][["c", "date"]].copy(); tmp["ticker"] = ticker
            entropy_panel_parts.append(tmp)

    if not rows:
        return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    full_period_ticker_data = pd.DataFrame(rows)

    if USE_ENTROPY_FILTER and entropy_panel_parts:
        panel = pd.concat(entropy_panel_parts).set_index(["date", "ticker"]).sort_index()
        panel["returns"] = panel.groupby(level="ticker")["c"].pct_change()

        feats = add_entropy_info_features(
            panel[["returns"]], price_col="closeadj", use_lagged_data=True,
            pe_window=ENTROPY_CFG["pe_window"], pe_m=ENTROPY_CFG["pe_m"], pe_tau=ENTROPY_CFG["pe_tau"],
            se_window=ENTROPY_CFG["se_window"], cs_bins=ENTROPY_CFG["cs_bins"]
        )
        last_feats = latest_per_ticker(
            feats, pd.Timestamp(month),
            [
                f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}",
                f"spectral_entropy_w{ENTROPY_CFG['se_window']}",
                f"peer_surprisal_cs_bins{ENTROPY_CFG['cs_bins']}",
                "peer_surprisal_z", f"peer_jsd_bins{ENTROPY_CFG['cs_bins']}", "peer_jsd_z"
            ]
        )
        pe_col = f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}"
        se_col = f"spectral_entropy_w{ENTROPY_CFG['se_window']}"
        ok = (last_feats[pe_col].between(0.15, 0.98)) & (last_feats[se_col].between(0.15, 0.98))
        keep = set(last_feats[ok].index.astype(str))
        full_period_ticker_data = full_period_ticker_data[full_period_ticker_data["ticker"].isin(keep)]

    top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(10)
    bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(10)
    return full_period_ticker_data, top_decile, bot_decile

# ---------- Run backtest over months[:-1] ----------
full_data_list, top_decile_list, bot_decile_list = [], [], []
times = []
for i, month in enumerate(all_months[:-1]):
    start = time.time()
    fp, td, bd = _score_month(month)
    if not fp.empty:
        full_data_list.append(fp); top_decile_list.append(td); bot_decile_list.append(bd)
    dt = time.time() - start; times.append(dt)
    remaining = len(all_months[:-1]) - (i + 1); avg = np.mean(times)
    eta = datetime.now() + timedelta(seconds=int(avg * remaining))
    print(f"{round(((i+1)/len(all_months[:-1]))*100,2)}% complete | {fp.shape[0]} rows | ETA ~ {eta}")

full_dataset = pd.concat(full_data_list) if full_data_list else pd.DataFrame([])
top_decile_dataset = pd.concat(top_decile_list) if top_decile_list else pd.DataFrame([])
bot_decile_dataset = pd.concat(bot_decile_list) if bot_decile_list else pd.DataFrame([])

# ---------- Compute P&L series ----------
if full_dataset.empty:
    raise SystemExit("No data to backtest.")

covered_dates = np.sort(full_dataset["entry_date"].drop_duplicates().values)
trade_rows = []
for d in covered_dates:
    L = top_decile_dataset[top_decile_dataset["entry_date"] == d]["forward_returns"].mean()
    S = -bot_decile_dataset[bot_decile_dataset["entry_date"] == d]["forward_returns"].mean()
    trade_rows.append({"date": d, "long": L, "short": S, "ls": L + S})
returns = pd.DataFrame(trade_rows).sort_values("date")
returns["date"] = pd.to_datetime(returns["date"])
returns = returns.set_index("date").sort_index()

# ---------- Metrics ----------
def _ann_stats(rets: pd.Series) -> dict:
    r = (rets / 100.0).dropna()          # convert % to decimal
    if r.empty:
        return {}
    n_months = r.shape[0]
    CAGR = (1.0 + r).prod() ** (12.0 / n_months) - 1.0
    vol_ann = r.std(ddof=1) * np.sqrt(12.0) if n_months > 1 else np.nan
    sharpe = (CAGR / vol_ann) if (vol_ann and vol_ann > 0) else np.nan
    downside = r[r < 0]
    dd_std = downside.std(ddof=1) * np.sqrt(12.0) if downside.shape[0] > 1 else np.nan
    sortino = (CAGR / dd_std) if (dd_std and dd_std > 0) else np.nan

    cum = (1.0 + r).cumprod()
    dd = cum / cum.cummax() - 1.0
    max_dd = dd.min()

    hit = (r > 0).mean()
    skew = r.skew()
    kurt = r.kurtosis()

    return dict(
        n_months=int(n_months),
        CAGR=CAGR, AnnVol=vol_ann, Sharpe=sharpe, Sortino=sortino,
        MaxDD=float(max_dd), HitRate=float(hit), Skew=float(skew), Kurtosis=float(kurt)
    )

def _turnover(holdings_df: pd.DataFrame) -> float:
    """Approximate monthly turnover for a fixed-size portfolio."""
    # holdings_df: concatenated top (or bottom) decile with columns ['entry_date','ticker']
    if holdings_df.empty:
        return np.nan
    h = holdings_df[["entry_date", "ticker"]].drop_duplicates()
    h["entry_date"] = pd.to_datetime(h["entry_date"])
    h = h.sort_values(["entry_date", "ticker"])
    months = np.sort(h["entry_date"].unique())
    total_changes = 0; total_slots = 0
    for i in range(1, len(months)):
        prev_set = set(h[h["entry_date"] == months[i-1]]["ticker"])
        curr_set = set(h[h["entry_date"] == months[i]]["ticker"])
        changes = len(curr_set - prev_set) + len(prev_set - curr_set)
        total_changes += changes
        total_slots += max(len(curr_set), 1)  # ~10
    return total_changes / total_slots

metrics_long  = _ann_stats(returns["long"])
metrics_short = _ann_stats(returns["short"])
metrics_ls    = _ann_stats(returns["ls"])
turnover_top  = _turnover(top_decile_dataset)
turnover_bot  = _turnover(bot_decile_dataset)

print("\n=== BACKTEST METRICS (monthly) ===")
print("Long:",  {k: (round(v,4) if isinstance(v,float) else v) for k,v in metrics_long.items()})
print("Short:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in metrics_short.items()})
print("L-S:",   {k: (round(v,4) if isinstance(v,float) else v) for k,v in metrics_ls.items()})
print(f"Turnover TopDecile ~ {round(turnover_top,4)} | Turnover BottomDecile ~ {round(turnover_bot,4)}")

# ---------- Plots ----------
cum_long  = (1 + (returns["long"]/100.0)).cumprod() - 1
cum_short = (1 + (returns["short"]/100.0)).cumprod() - 1
cum_ls    = (1 + (returns["ls"]/100.0)).cumprod() - 1

plt.figure(figsize=PLOT_FIGSIZE, dpi=200)
plt.title("Cumulative Returns (Gross)")
plt.plot(cum_long.index, cum_long.values)
plt.plot(cum_short.index, cum_short.values)
plt.plot(cum_ls.index, cum_ls.values)
plt.legend(["Top Decile", "Bottom Decile", "Long-Short"])
plt.xlabel("Date"); plt.ylabel("Cumulative Return")
plt.xticks(rotation=45); plt.tight_layout(); plt.show(); plt.close()

# Drawdown of L-S
cum = (1 + (returns["ls"]/100.0)).cumprod()
dd = cum / cum.cummax() - 1.0
plt.figure(figsize=PLOT_FIGSIZE, dpi=200)
plt.title("Drawdown (Long-Short)")
plt.plot(dd.index, dd.values)
plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.xticks(rotation=45); plt.tight_layout(); plt.show(); plt.close()

# Rolling 12M Sharpe for L-S
roll = returns["ls"].rolling(12)
roll_sharpe = (np.sqrt(12) * (roll.mean()/roll.std())).dropna()
plt.figure(figsize=PLOT_FIGSIZE, dpi=200)
plt.title("Rolling 12M Sharpe (Long-Short)")
plt.plot(roll_sharpe.index, roll_sharpe.values)
plt.xlabel("Date"); plt.ylabel("Sharpe")
plt.xticks(rotation=45); plt.tight_layout(); plt.show(); plt.close()

# ---------- Save outputs ----------
outdir = pathlib.Path("./mtum_backtest_out"); outdir.mkdir(parents=True, exist_ok=True)
returns.to_csv(outdir / "monthly_returns.csv")
top_decile_dataset.to_csv(outdir / "top_decile_selections.csv", index=False)
bot_decile_dataset.to_csv(outdir / "bottom_decile_selections.csv", index=False)
full_dataset.to_parquet(outdir / "per_ticker_per_month.parquet", index=False)
print(f"Outputs saved to {outdir.resolve()}")
