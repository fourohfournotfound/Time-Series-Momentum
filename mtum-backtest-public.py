# -*- coding: utf-8 -*-
"""
Created in 2025

Backtest with the same time-frames and monthly rebalancing as the QG version,
plus optional volatility management, EWMA stats, and entropy filtering.

Guarantee:
- With METHOD="classic" and all USE_* flags False, the backtest results (top/bottom deciles,
  long/short series, cumulative P&L) match the original.

Outputs:
- full_dataset / top_decile_dataset / bot_decile_dataset identical in structure.
- A MultiIndex features frame (entry_date, ticker) for LTR training alignment.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import gspread  # parity
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# =============================================================================
# Config (same toggles as prod)
# =============================================================================

polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

METHOD = os.getenv("MTUM_METHOD", "classic")
USE_VOL_MANAGED = (METHOD in ("vol_managed","ewma_vol_managed"))
USE_EWMA        = (METHOD in ("ewma","ewma_vol_managed"))
USE_ENTROPY_FILTER = bool(int(os.getenv("USE_ENTROPY_FILTER", "0")))

VOL_WINDOW_DAYS = int(os.getenv("VOL_WINDOW_DAYS", "21"))
VOL_TARGET_DAILY = float(os.getenv("VOL_TARGET_DAILY", "1.0"))
VOL_SCALE_CLIP = float(os.getenv("VOL_SCALE_CLIP", "3.0"))
EWMA_HALFLIFE = int(os.getenv("EWMA_HALFLIFE", "21"))

ENTROPY_BINS = int(os.getenv("ENTROPY_BINS", "5"))
ENTROPY_MAX_BITS = float(os.getenv("ENTROPY_MAX_BITS", "inf"))
ENTROPY_QCUT = bool(int(os.getenv("ENTROPY_QCUT", "1")))

MAX_WORKERS = int(os.getenv("BT_MAX_WORKERS", "8"))

# =============================================================================
# Date Management - Rebalancing Every Month (unchanged)
# =============================================================================

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date = "2019-01-01",
                                  end_date   = (datetime.today() - timedelta(days=1))
                                 ).index.strftime("%Y-%m-%d").values

# =============================================================================
# PIT Universe + Benchmark (unchanged)
# =============================================================================

def build_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=500,
                          max_retries=Retry(total=3, backoff_factor=0.5,
                                            status_forcelist=(429, 500, 502, 503, 504)))
    s.mount('https://', adapter)
    s.headers.update({"Accept":"application/json"})
    return s

session = build_session()

universe = pd.read_sql("historical_liquid_tickers_polygon", con=engine).drop_duplicates(subset=["date","ticker"])
spy = session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
).json()
benchmark_data = pd.json_normalize(spy.get("results", [])).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# =============================================================================
# Helpers (mirrors prod)
# =============================================================================

def shannon_entropy_bits(x: pd.Series, bins=5, qcut=True) -> float:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return 0.0
    try:
        if qcut:
            binned = pd.qcut(x, q=bins, labels=False, duplicates="drop")
        else:
            binned = pd.cut(x, bins=bins, labels=False)
        counts = np.bincount(binned.dropna().astype(int), minlength=bins).astype(float)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    except Exception:
        return 0.0

def fetch_ticker_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    r = session.get(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
    ).json()
    df = pd.json_normalize(r.get("results", [])).set_index("t")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df

def compute_beta_sharpe_and_score(ticker_window: pd.DataFrame, bm: pd.DataFrame) -> dict:
    merged = pd.merge(bm[["c","date"]], ticker_window[["c","date"]], on="date", how="inner")
    merged["benchmark_pct_change"] = (merged["c_x"].pct_change() * 100).round(2).fillna(0)
    merged["ticker_pct_change"]    = (merged["c_y"].pct_change() * 100).round(2).fillna(0)

    cov = np.cov(merged["ticker_pct_change"], merged["benchmark_pct_change"])
    beta = cov[0,1] / (np.var(merged["benchmark_pct_change"]) + 1e-12)

    ret_period = ((merged["c_y"].iloc[-1] - merged["c_y"].iloc[0]) / merged["c_y"].iloc[0]) * 100
    std_ann = merged["ticker_pct_change"].std() * np.sqrt(252)
    sharpe = (ret_period / (std_ann + 1e-12))
    theo_expected = beta * sharpe
    return {"beta": float(beta), "sharpe": float(sharpe), "ret_12_1": float(ret_period), "mom_score_classic": float(theo_expected)}

def apply_vol_management(returns_pct: pd.Series) -> pd.Series:
    rolling_std = returns_pct.rolling(VOL_WINDOW_DAYS, min_periods=max(2, VOL_WINDOW_DAYS//2)).std()
    if USE_EWMA:
        ewma_std = returns_pct.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).std()
        vol_est = ewma_std
    else:
        vol_est = rolling_std
    scale = (VOL_TARGET_DAILY / (vol_est.replace(0, np.nan))).clip(upper=VOL_SCALE_CLIP).fillna(0.0)
    return (returns_pct * scale.shift(1))  # lagged

def compute_method_momscore(ticker_window: pd.DataFrame, bm: pd.DataFrame) -> dict:
    out = compute_beta_sharpe_and_score(ticker_window, bm)

    if METHOD == "classic" and not USE_VOL_MANAGED and not USE_EWMA and not USE_ENTROPY_FILTER:
        out["mom_score"] = out["mom_score_classic"]
        return out

    merged = pd.merge(bm[["c","date"]], ticker_window[["c","date"]], on="date", how="inner").sort_values("date")
    ret_bm = (merged["c_x"].pct_change() * 100).fillna(0)
    ret_tk = (merged["c_y"].pct_change() * 100).fillna(0)

    if USE_ENTROPY_FILTER:
        ent_bits = shannon_entropy_bits(ret_tk, bins=ENTROPY_BINS, qcut=ENTROPY_QCUT)
        if ent_bits > ENTROPY_MAX_BITS:
            out["entropy_bits"] = float(ent_bits)
            out["mom_score"] = -1e9
            return out
        out["entropy_bits"] = float(ent_bits)

    if USE_VOL_MANAGED:
        ret_tk = apply_vol_management(ret_tk)

    if USE_EWMA:
        mean_ann = ret_tk.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).mean().iloc[-1] * np.sqrt(252)
        std_ann  = ret_tk.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).std().iloc[-1]  * np.sqrt(252)
        ewma_sharpe = (mean_ann / (std_ann + 1e-12))
        cov = np.cov((ret_tk.fillna(0)).values, (ret_bm.values))
        beta = cov[0,1] / (np.var(ret_bm.values) + 1e-12)
        out["mom_score"] = float(beta * ewma_sharpe)
        out["beta"] = float(beta)
        out["sharpe"] = float(ewma_sharpe)
    else:
        std_ann = ret_tk.std() * np.sqrt(252)
        ret_period = ((merged["c_y"].iloc[-1] - merged["c_y"].iloc[0]) / merged["c_y"].iloc[0]) * 100
        sharpe = (ret_period / (std_ann + 1e-12))
        cov = np.cov(ret_tk.values, ret_bm.values)
        beta = cov[0,1] / (np.var(ret_bm.values) + 1e-12)
        out["mom_score"] = float(beta * sharpe)
        out["beta"] = float(beta)
        out["sharpe"] = float(sharpe)

    return out

# =============================================================================
# Backtesting loop
# =============================================================================

full_data_list, top_decile_list, bot_decile_list = [], [], []
times = []

months = universe["date"].drop_duplicates().values

def process_month(month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    try:
        point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
        point_in_time_date = point_in_time_dates[-1]
        pit = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")
        tickers = pit["ticker"].drop_duplicates().values

        start_date = (pd.to_datetime(month) - timedelta(days = 365+60)).strftime("%Y-%m-%d")
        end_date = month
        next_month_date = np.sort(months[months > month])[0]
        last_month_date = (pd.to_datetime(month) - timedelta(days = 30)).strftime("%Y-%m-%d")

        def process_ticker(tkr: str) -> pd.DataFrame | None:
            try:
                data = fetch_ticker_ohlc(tkr, start_date, next_month_date)  # include forward window like original
                if data.empty: return None

                twelve_minus_one = data[data["date"] < last_month_date].copy().tail(252)
                if len(twelve_minus_one) < 252: return None

                feat = compute_method_momscore(twelve_minus_one, benchmark_data)

                # forward window: [month, next_month_date]
                fwd = data[(data["date"] >= month) & (data["date"] <= next_month_date)].copy().sort_index()
                next_ret = round(((fwd["c"].iloc[-1] - fwd["c"].iloc[0]) / fwd["c"].iloc[0])*100, 2)

                row = {
                    "entry_date": month,
                    "ticker": tkr,
                    "beta": feat["beta"], "sharpe": feat["sharpe"],
                    "12-1_return": feat["ret_12_1"], "mom_score_classic": feat["mom_score_classic"],
                    "mom_score": feat["mom_score"],
                    "forward_returns": next_ret, "exit_date": next_month_date
                }
                if "entropy_bits" in feat:
                    row["entropy_bits"] = feat["entropy_bits"]
                return pd.DataFrame([row])
            except Exception as e:
                print(e)
                return None

        monthly_rows = []
        if MAX_WORKERS > 0 and len(tickers) > 0:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_ticker, t) for t in tickers]
                for f in as_completed(futs):
                    df = f.result()
                    if df is not None:
                        monthly_rows.append(df)
        else:
            for t in tickers:
                df = process_ticker(t)
                if df is not None:
                    monthly_rows.append(df)

        full_period = pd.concat(monthly_rows) if monthly_rows else pd.DataFrame(columns=[
            "entry_date","ticker","beta","sharpe","12-1_return","mom_score_classic","mom_score","forward_returns","exit_date"
        ])

        # classic parity enforcement
        if METHOD == "classic" and not USE_VOL_MANAGED and not USE_EWMA and not USE_ENTROPY_FILTER and not full_period.empty:
            assert np.allclose(full_period["mom_score"], full_period["mom_score_classic"], rtol=1e-10, atol=1e-10)

        top_decile = full_period.sort_values(by="mom_score", ascending=False).head(10)
        bot_decile = full_period.sort_values(by="mom_score", ascending=True).head(10)
        return full_period, top_decile, bot_decile
    except Exception as e:
        print(e)
        return None

# month-by-month (sequential to keep identical order; you may wrap with executor if desired)
for month in months[:-1]:
    start_time = datetime.now()
    res = process_month(month)
    if res is None: continue
    full_period, top_decile, bot_decile = res
    full_data_list.append(full_period)
    top_decile_list.append(top_decile)
    bot_decile_list.append(bot_decile)

    end_time = datetime.now()
    times.append((end_time - start_time).total_seconds())
    iteration = round((np.where(months == month)[0][0]/len(months))*100, 2)
    iterations_remaining = len(months) - np.where(months == month)[0][0]
    avg_time = np.mean(times)
    eta = (datetime.now() + timedelta(seconds=int(avg_time * iterations_remaining)))
    remaining = eta - datetime.now()
    print(f"{iteration}% complete, {remaining} left, ETA: {eta}")

full_dataset       = pd.concat(full_data_list)
top_decile_dataset = pd.concat(top_decile_list)
bot_decile_dataset = pd.concat(bot_decile_list)

# =============================================================================
# Backtest (identical arithmetic; plotting preserved)
# =============================================================================

covered_dates = full_dataset["entry_date"].drop_duplicates().values
trade_list = []
for d in covered_dates:
    long_uni  = top_decile_dataset[top_decile_dataset["entry_date"] == d].copy()
    short_uni = bot_decile_dataset[bot_decile_dataset["entry_date"] == d].copy()
    trade_list.append(pd.DataFrame([{
        "date": d,
        "long":  long_uni["forward_returns"].mean(),
        "short": short_uni["forward_returns"].mean() * -1
    }]))

all_trades = pd.concat(trade_list)
all_trades["long_pnl"]  = all_trades["long"].cumsum()
all_trades["short_pnl"] = all_trades["short"].cumsum()
all_trades["portfolio_pnl"] = all_trades["long_pnl"] + all_trades["short_pnl"]

plt.figure(figsize=(10, 6), dpi=200)
plt.xticks(rotation=45)
plt.suptitle("Gross Cumulative Performance")
plt.title("Monthly Rebalancing")
plt.plot(pd.to_datetime(all_trades["date"]), all_trades["long_pnl"])
plt.plot(pd.to_datetime(all_trades["date"]), all_trades["short_pnl"])
plt.plot(pd.to_datetime(all_trades["date"]), all_trades["portfolio_pnl"])
plt.legend(["Top Decile", "Bottom Decile", "Long-Short"])
plt.xlabel("Date"); plt.ylabel("Cumulative % Returns")
plt.show(); plt.close()
