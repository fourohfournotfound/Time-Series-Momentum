# -*- coding: utf-8 -*-
"""
Monthly momentum backtest — Colab-friendly (SQLite), optimized, metrics-rich, reproducible.

Fixes in this version:
  • BUGFIX: argparse attribute uses underscores. Replaced args.use-entropy-filter → args.use_entropy_filter
    and set dest="use_entropy_filter" in add_argument.
  • Respects --database-url (defaults to sqlite:///universe.db); no MySQL required.
  • One-shot OHLC fetch per ticker over the whole backtest window (prefetch) with retry/backoff.
  • Full backtest metrics + plots (CAGR, AnnVol, Sharpe, Sortino, MaxDD, hit rate, rolling Sharpe).
  • Exact 'classic' parity with original QG logic (same 12-1 window, daily rounding, SPY merge).

Quant guardrails:
  • Strict t-1 usage (12-1 window) and forward returns only in the next month (no look-ahead).
  • Shape checks on key merges.
"""

import os
import time
import math
import pathlib
import argparse
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

# Optional advanced features (present in your repo)
# These keep (date,ticker) MultiIndex invariants & t-1 causality.
# If these modules are not present, you can comment them out safely.
try:
    from custom_relative_market_features import add_entropy_info_features, ewm_moments, latest_per_ticker
except Exception:
    add_entropy_info_features = None
    ewm_moments = None
    latest_per_ticker = None


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Monthly momentum backtest")
    p.add_argument("--provider", default="polygon", choices=["polygon"], help="Data provider")
    p.add_argument("--database-url", default="sqlite:///universe.db", help="SQLAlchemy URL, e.g. sqlite:///universe.db")
    p.add_argument("--universe-table", default="historical_liquid_tickers_polygon", help="Universe table name")
    p.add_argument("--start-date", default="2019-01-01", help="Backtest start (YYYY-MM-DD)")
    p.add_argument("--cache-dir", default="cache", help="Local Parquet cache for bars")
    p.add_argument("--output-dir", default="output", help="Where to store CSV/Parquet outputs")
    p.add_argument("--momentum-type", default="classic",
                   choices=["classic", "vol_scaled", "ewma", "ewma_vol_scaled"],
                   help="Scoring mode")
    p.add_argument("--vol-estimator", default="t", help="Accepted for compatibility; not used in classic")
    p.add_argument("--prefetch", action="store_true", help="Prefetch all bars once for entire backtest window")
    # BUGFIX: explicitly set dest to ensure attribute name
    p.add_argument("--use-entropy-filter", dest="use_entropy_filter", action="store_true",
                   help="Optional entropy/peer information filter (t-1)")
    p.add_argument("--max-workers", type=int, default=12, help="Concurrency for network fetches")
    p.add_argument("--ewm-halflife", type=int, default=63, help="EWM halflife for EWMA modes")
    return p.parse_args()


# -------------------- Helpers --------------------
def _build_session(max_workers: int, user_agent: str) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=max_workers * 2))
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def _get_json(session: requests.Session, url: str, params: dict | None = None, timeout=(5, 60)) -> dict:
    params = params or {}
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                time.sleep(float(ra))
                r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"HTTP error: {e} | URL={url} | params={params}")
        return {}


def _iter_aggs(session: requests.Session, url: str, params: dict) -> list[dict]:
    """Follow Polygon v2 aggregates pagination via next_url when present."""
    out = []
    payload = _get_json(session, url, params)
    if not payload or "results" not in payload:
        return out
    out.extend(payload.get("results", []))
    next_url = payload.get("next_url")
    while next_url:
        payload = _get_json(session, next_url, {"apiKey": params.get("apiKey")})
        if not payload or "results" not in payload:
            break
        out.extend(payload.get("results", []))
        next_url = payload.get("next_url")
    return out


def _cache_path(cache_dir: pathlib.Path, ticker: str, global_start: str, global_end: str) -> pathlib.Path:
    return cache_dir / f"stocks_{ticker}_{global_start}_{global_end}.parquet"


def _fetch_ticker_bars(session: requests.Session, ticker: str, global_start: str, global_end: str,
                       api_key: str, cache_dir: pathlib.Path, enable_cache: bool = True) -> pd.DataFrame:
    """Fetch 1D adjusted bars for a ticker for the entire backtest span."""
    pth = _cache_path(cache_dir, ticker, global_start, global_end)
    if enable_cache and pth.exists():
        try:
            return pd.read_parquet(pth)
        except Exception:
            pass

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{global_start}/{global_end}"
    payload = _iter_aggs(session, url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key})
    if not payload:
        return pd.DataFrame([])

    df = pd.json_normalize(payload).set_index("t")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")
    if enable_cache:
        try:
            df.to_parquet(pth, index=True)
        except Exception:
            pass
    return df


# -------------------- Main --------------------
def main():
    args = parse_args()

    if args.provider != "polygon":
        raise SystemExit("Only --provider polygon is supported in this script.")

    # Engine selection respects --database-url (SQLite works out of the box in Colab)
    engine = create_engine(args.database_url)  # e.g., sqlite:///universe.db

    # Environment keys
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        raise SystemExit("Set POLYGON_API_KEY in environment (Colab: from google.colab import userdata; os.environ['POLYGON_API_KEY']=userdata.get('POLYGON_API_KEY'))")

    # Calendar & month list (NYSE)
    calendar = get_calendar("NYSE")
    trading_dates = calendar.schedule(start_date=args.start_date, end_date=(datetime.today() - timedelta(days=1))).index
    months_all = trading_dates.to_series().dt.tz_localize(None).dt.normalize().drop_duplicates(keep="first").dt.strftime("%Y-%m-%d").values

    # Universe PIT (from your table)
    try:
        universe = pd.read_sql(args.universe_table, con=engine).drop_duplicates(subset=["date", "ticker"])
    except Exception as e:
        raise SystemExit(f"Failed to read '{args.universe_table}' from {args.database_url} : {e}\n"
                         f"Confirm the table exists. In Colab, create it via pandas.to_sql as you showed.")

    # Ensure column types
    if "date" not in universe.columns or "ticker" not in universe.columns:
        raise SystemExit("Universe must have columns 'date' and 'ticker'.")
    universe["date"] = universe["date"].astype(str)
    universe["ticker"] = universe["ticker"].astype(str)

    # Backtest months bounded by universe coverage & --start-date
    all_months = np.sort(universe["date"].drop_duplicates().values)
    all_months = all_months[all_months >= args.start_date]
    if all_months.size < 2:
        raise SystemExit("Universe table has <2 months at/after start-date; run PIT builder or adjust --start-date.")

    # Global fetch window: earliest needed start -> latest needed end
    first_month = pd.to_datetime(all_months[0])
    last_month = pd.to_datetime(all_months[-1])
    global_start = (first_month - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
    global_end = (last_month + timedelta(days=35)).strftime("%Y-%m-%d")

    # Network session (retry/backoff; polite concurrency)
    max_workers = max(1, int(args.max_workers))
    session = _build_session(max_workers, user_agent="mtum-backtest/3.0")

    # SPY benchmark once (v2 aggregates; follow pagination)
    spy_rows = _iter_aggs(
        session,
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{last_month.strftime('%Y-%m-%d')}",
        {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key},
    )
    if not spy_rows:
        raise SystemExit("Failed to fetch SPY benchmark.")
    benchmark_data = pd.json_normalize(spy_rows).set_index("t")
    benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
    benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

    # Union of tickers across the backtest
    all_tickers = universe["ticker"].drop_duplicates().astype(str).tolist()

    # Optional local cache
    cache_dir = pathlib.Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    enable_disk_cache = True

    # Prefetch (major speedup)
    bars_map: dict[str, pd.DataFrame] = {}
    if args.prefetch:
        print(f"Prefetching {len(all_tickers)} tickers once for {global_start}..{global_end}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_ticker_bars, session, t, global_start, global_end, polygon_api_key, cache_dir, enable_disk_cache): t
                    for t in all_tickers}
            for i, f in enumerate(as_completed(futs)):
                tkr = futs[f]
                df = f.result()
                bars_map[tkr] = df
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(all_tickers)} fetched")

    # ---------- Month scoring ----------
    def _score_month(month: str):
        """Return (full_period_ticker_data, top_decile, bot_decile) for a single month."""
        month_dt = pd.to_datetime(month)
        # Next month in the PIT calendar
        next_mask = all_months > month
        if not np.any(next_mask):
            return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        next_month = np.sort(all_months[next_mask])[0]
        last_month_date = (month_dt - timedelta(days=30)).strftime("%Y-%m-%d")

        # PIT selection (same as original)
        pit_date = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)[-1]
        month_universe = universe[universe["date"] == pit_date].drop_duplicates(subset=["ticker"], keep="last")["ticker"].astype(str).values

        rows = []
        # (Optional) panel parts for entropy filter (causal)
        entropy_panel_parts = []

        for ticker in month_universe:
            # Fetch bars (prefetched or on-demand)
            if ticker in bars_map:
                df = bars_map[ticker]
            else:
                df = _fetch_ticker_bars(session, ticker, global_start, global_end, polygon_api_key, cache_dir, enable_disk_cache)
            if df is None or df.empty:
                continue

            # trailing 12-1 slice (t-1)
            tmo = df[df["date"] < last_month_date].copy().tail(252)
            if tmo.shape[0] < 252:
                continue

            # QG 'classic' rounding points: merge with SPY then round daily pct changes to 2dp before cov/var
            bmerge = pd.merge(benchmark_data[["c", "date"]], tmo[["c", "date"]], on="date", how="inner", validate="one_to_one")
            if bmerge.empty:
                continue
            bmerge["benchmark_pct_change"] = np.round(bmerge["c_x"].pct_change() * 100, 2).fillna(0)
            bmerge["ticker_pct_change"]    = np.round(bmerge["c_y"].pct_change() * 100, 2).fillna(0)

            cov = np.cov(bmerge["ticker_pct_change"], bmerge["benchmark_pct_change"])
            var_b = np.var(bmerge["benchmark_pct_change"])
            beta = (cov[0, 1] / var_b) if var_b != 0 else 0.0

            tkr_ret_period = np.round(((tmo["c"].iloc[-1] - tmo["c"].iloc[0]) / tmo["c"].iloc[0]) * 100, 2)
            std_daily = bmerge["ticker_pct_change"].std() * np.sqrt(252)
            sharpe = (tkr_ret_period / std_daily) if std_daily != 0 else 0.0

            mode = args.momentum_type
            if mode == "classic":
                mom_score = beta * sharpe
            elif mode == "vol_scaled":
                mom_score = beta * (tkr_ret_period / (std_daily if std_daily != 0 else 1e-12))
            elif mode in {"ewma", "ewma_vol_scaled"} and ewm_moments is not None:
                mi = tmo[["c", "date"]].copy(); mi["ticker"] = ticker
                mi = mi.set_index(["date", "ticker"]).sort_index()
                mi["returns"] = mi.groupby(level="ticker")["c"].pct_change()
                em = ewm_moments(mi, ret_col="returns", halflife=int(args.ewm_halflife), use_lagged_data=True).groupby(level="ticker").tail(1)
                ewm_mean = float(em["ewm_mean"].iloc[0]) * 100.0
                ewm_vol  = float(em["ewm_vol_ann"].iloc[0])
                mom_score = 0.0 if ewm_vol == 0 else beta * (ewm_mean / ewm_vol)
            else:
                mom_score = beta * sharpe  # fallback

            # forward window [month, next_month]
            nxt = df[(df["date"] >= month) & (df["date"] <= next_month)].copy().sort_index()
            fwd_ret = np.round(((nxt["c"].iloc[-1] - nxt["c"].iloc[0]) / nxt["c"].iloc[0]) * 100, 2) if nxt.shape[0] >= 2 else np.nan

            rows.append({
                "entry_date": month, "ticker": ticker, "beta": beta, "sharpe": sharpe,
                "12-1_return": tkr_ret_period, "mom_score": mom_score, "forward_returns": fwd_ret, "exit_date": next_month
            })

            # BUGFIX LINE: use the underscore form produced by argparse
            if args.use_entropy_filter and add_entropy_info_features is not None:
                tmp = df[df["date"] < last_month_date][["c", "date"]].copy(); tmp["ticker"] = ticker
                entropy_panel_parts.append(tmp)

        if not rows:
            return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

        full_period_ticker_data = pd.DataFrame(rows)

        if args.use_entropy_filter and add_entropy_info_features is not None and entropy_panel_parts:
            panel = pd.concat(entropy_panel_parts).set_index(["date", "ticker"]).sort_index()
            panel["returns"] = panel.groupby(level="ticker")["c"].pct_change()
            feats = add_entropy_info_features(panel[["returns"]], price_col="closeadj", use_lagged_data=True)
            last_feats = latest_per_ticker(
                feats, pd.Timestamp(month),
                [
                    "perm_entropy_w126_m4_tau1", "spectral_entropy_w128",
                    "peer_surprisal_cs_bins21", "peer_surprisal_z",
                    "peer_jsd_bins21", "peer_jsd_z"
                ]
            )
            pe_col = "perm_entropy_w126_m4_tau1"; se_col = "spectral_entropy_w128"
            ok = (last_feats[pe_col].between(0.15, 0.98)) & (last_feats[se_col].between(0.15, 0.98))
            keep = set(last_feats[ok].index.astype(str))
            full_period_ticker_data = full_period_ticker_data[full_period_ticker_data["ticker"].isin(keep)]

        top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(10)
        bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(10)
        return full_period_ticker_data, top_decile, bot_decile

    # ---------- Run backtest ----------
    full_data_list, top_decile_list, bot_decile_list = [], [], []
    times = []
    run_months = all_months[:-1]  # no forward window for last month

    for i, month in enumerate(run_months):
        start = time.time()
        fp, td, bd = _score_month(month)
        if not fp.empty:
            full_data_list.append(fp); top_decile_list.append(td); bot_decile_list.append(bd)
        dt = time.time() - start; times.append(dt)
        remaining = len(run_months) - (i + 1); avg = np.mean(times)
        eta = datetime.now() + timedelta(seconds=int(avg * remaining))
        print(f"{round(((i+1)/len(run_months))*100,2)}% complete | {fp.shape[0]} rows | ETA ~ {eta}")

    full_dataset = pd.concat(full_data_list) if full_data_list else pd.DataFrame([])
    top_decile_dataset = pd.concat(top_decile_list) if top_decile_list else pd.DataFrame([])
    bot_decile_dataset = pd.concat(bot_decile_list) if bot_decile_list else pd.DataFrame([])

    if full_dataset.empty:
        raise SystemExit("No data to backtest.")

    # ---------- Compute P&L series ----------
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
        r = (rets / 100.0).dropna()  # convert % to decimal
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
        if holdings_df.empty:
            return np.nan
        h = holdings_df[["entry_date", "ticker"]].drop_duplicates()
        h["entry_date"] = pd.to_datetime(h["entry_date"])
        h = h.sort_values(["entry_date", "ticker"])
        m = np.sort(h["entry_date"].unique())
        total_changes = 0; total_slots = 0
        for i in range(1, len(m)):
            prev_set = set(h[h["entry_date"] == m[i-1]]["ticker"])
            curr_set = set(h[h["entry_date"] == m[i]]["ticker"])
            changes = len(curr_set - prev_set) + len(prev_set - curr_set)
            total_changes += changes
            total_slots += max(len(curr_set), 1)  # ~10 names
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

    # Ensure output directory exists for plots and tabular outputs
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Plots ----------
    PLOT_FIGSIZE = (10, 6)

    cum_long  = (1 + (returns["long"]/100.0)).cumprod() - 1
    cum_short = (1 + (returns["short"]/100.0)).cumprod() - 1
    cum_ls    = (1 + (returns["ls"]/100.0)).cumprod() - 1

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=200)
    ax.set_title("Cumulative Returns (Gross)")
    ax.plot(cum_long.index, cum_long.values)
    ax.plot(cum_short.index, cum_short.values)
    ax.plot(cum_ls.index, cum_ls.values)
    ax.legend(["Top Decile", "Bottom Decile", "Long-Short"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    fig.autofmt_xdate()
    fig.tight_layout()
    cum_returns_path = outdir / "cumulative_returns.png"
    fig.savefig(cum_returns_path, bbox_inches="tight")
    print(f"Saved plot: {cum_returns_path}")
    plt.close(fig)

    # Drawdown of L-S
    cum = (1 + (returns["ls"]/100.0)).cumprod()
    dd = cum / cum.cummax() - 1.0
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=200)
    ax.set_title("Drawdown (Long-Short)")
    ax.plot(dd.index, dd.values)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    drawdown_path = outdir / "drawdown_long_short.png"
    fig.savefig(drawdown_path, bbox_inches="tight")
    print(f"Saved plot: {drawdown_path}")
    plt.close(fig)

    # Rolling 12M Sharpe for L-S
    roll = returns["ls"].rolling(12)
    roll_sharpe = (np.sqrt(12) * (roll.mean()/roll.std())).dropna()
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=200)
    ax.set_title("Rolling 12M Sharpe (Long-Short)")
    ax.plot(roll_sharpe.index, roll_sharpe.values)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    fig.autofmt_xdate()
    fig.tight_layout()
    rolling_sharpe_path = outdir / "rolling_sharpe_long_short.png"
    fig.savefig(rolling_sharpe_path, bbox_inches="tight")
    print(f"Saved plot: {rolling_sharpe_path}")
    plt.close(fig)

    # ---------- Save outputs ----------
    returns.to_csv(outdir / "monthly_returns.csv")
    if not top_decile_dataset.empty:
        top_decile_dataset.to_csv(outdir / "top_decile_selections.csv", index=False)
    if not bot_decile_dataset.empty:
        bot_decile_dataset.to_csv(outdir / "bottom_decile_selections.csv", index=False)
    if not full_dataset.empty:
        full_dataset.to_parquet(outdir / "per_ticker_per_month.parquet", index=False)
    print(f"Outputs saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
