# -*- coding: utf-8 -*-
"""
Created in 2025

Updated: strictly no forward-looking data in production scoring.
- Uses the latest available monthly rebalance_date (t) and associated as_of_date (t-1) from PIT universe.
- Momentum (12-1), residual momentum (CAPM residuals), and volatility metrics computed up to as_of_date.
- Outputs MultiIndex (entry_date, ticker) feature frame. No forward returns are computed here (prod safety).
"""

import os
import pandas as pd
import numpy as np
import requests
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta, timezone
from pandas_market_calendars import get_calendar

polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

# =============================================================================
# Dates & universe
# =============================================================================
nyse = get_calendar("NYSE")

def trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    return sched.index.tz_convert("America/New_York")

today_local = datetime.now(timezone.utc).astimezone()
tds = trading_days((today_local - timedelta(days=60)).strftime("%Y-%m-%d"),
                   today_local.strftime("%Y-%m-%d"))

# PIT universe table with as_of_date (t-1) and rebalance_date (t)
universe = pd.read_sql("SELECT * FROM historical_liquid_tickers_polygon", con=engine)  # table created by script #1
assert {"as_of_date", "rebalance_date", "ticker"}.issubset(universe.columns), "Universe missing required columns"

# Use the latest rebalance_date that is not in the future
universe["rebalance_date"] = pd.to_datetime(universe["rebalance_date"])
valid_rebals = np.sort(universe[universe["rebalance_date"] <= tds.max()]["rebalance_date"].unique())
if len(valid_rebals) == 0:
    raise RuntimeError("No valid rebalance dates available in universe table.")
rebalance_date = pd.to_datetime(valid_rebals[-1])
as_of_date = pd.to_datetime(universe[universe["rebalance_date"] == rebalance_date]["as_of_date"].iloc[0])
rebalance_date_str = rebalance_date.strftime("%Y-%m-%d")
as_of_date_str = as_of_date.strftime("%Y-%m-%d")

# Universe tickers for this rebalance date (formed at t-1)
ptu = (universe[universe["rebalance_date"] == rebalance_date_str]
       .drop_duplicates(subset=["ticker"], keep="last")
       .copy())
tickers = ptu["ticker"].drop_duplicates().to_numpy()
print("Universe size at", rebalance_date_str, ":", len(tickers))

# =============================================================================
# Benchmark (SPY) up to as_of_date only
# =============================================================================
def _get(url, params=None):
    params = {} if params is None else dict(params)
    params["apiKey"] = polygon_api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

bench = _get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{as_of_date_str}",
             params={"adjusted": "true", "sort": "asc", "limit": 50000})
bench_df = pd.json_normalize(bench.get("results", [])).set_index("t")
bench_df.index = pd.to_datetime(bench_df.index, unit="ms", utc=True).tz_convert("America/New_York")
bench_df["date"] = bench_df.index.strftime("%Y-%m-%d")

# =============================================================================
# Feature generation (all info <= as_of_date, skip most recent ~21 trading days for 12-1)
# =============================================================================
def compute_features_for_ticker(tk: str) -> pd.DataFrame:
    # History range
    start_date = (as_of_date - timedelta(days=365 + 60)).strftime("%Y-%m-%d")  # generous buffer for gaps
    # Pull only up to as_of_date
    ud = _get(f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/day/{start_date}/{as_of_date_str}",
              params={"adjusted": "true", "sort": "asc", "limit": 50000})
    ud_df = pd.json_normalize(ud.get("results", [])).set_index("t")
    if ud_df.empty:
        return pd.DataFrame()
    ud_df.index = pd.to_datetime(ud_df.index, unit="ms", utc=True).tz_convert("America/New_York")
    ud_df["date"] = ud_df.index.strftime("%Y-%m-%d")

    # Ensure at least 252 obs for 12-1
    # Exclude last ~30 calendar days (skip-month) -> approximate 21 trading days
    last_month_cut = (as_of_date - timedelta(days=30)).strftime("%Y-%m-%d")
    tmo = ud_df[ud_df["date"] < last_month_cut].tail(252).copy()
    if len(tmo) < 252:
        return pd.DataFrame()

    # Daily pct changes for joint regressions with SPY
    merged = pd.merge(bench_df[["c", "date"]], tmo[["c", "date"]], on="date", how="inner", suffixes=("_m", "_s"))
    if merged.shape[0] < 200:
        return pd.DataFrame()
    merged["ret_m"] = merged["c_m"].pct_change().fillna(0.0)
    merged["ret_s"] = merged["c_s"].pct_change().fillna(0.0)

    # CAPM beta via covariance ratio over this lookback
    cov = np.cov(merged["ret_s"], merged["ret_m"])
    beta = 0.0 if cov[1, 1] == 0 else cov[0, 1] / cov[1, 1]

    # 12-1 momentum (close-to-close) over tmo window
    mom_12_1 = (tmo["c"].iloc[-1] / tmo["c"].iloc[0]) - 1.0

    # Residual momentum: regress ret_s on ret_m, take residuals, cumulate over window
    # Simple OLS with intercept
    X = np.vstack([np.ones(len(merged["ret_m"])), merged["ret_m"].values]).T
    y = merged["ret_s"].values
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ coef
        resid_mom = (np.prod(1.0 + resid) - 1.0)  # compounding residuals
    except Exception:
        resid_mom = np.nan

    # Volatility (annualized stdev) and a risk-managed momentum score
    sigma_ann = merged["ret_s"].std() * np.sqrt(252)
    vol_neutral_mom = mom_12_1 / (sigma_ann + 1e-8)

    return pd.DataFrame([{
        "entry_date": rebalance_date_str,
        "as_of_date": as_of_date_str,
        "ticker": tk,
        "beta_capm": float(beta),
        "mom_12_1": float(mom_12_1),
        "resid_mom": float(resid_mom) if np.isfinite(resid_mom) else np.nan,
        "sigma_ann": float(sigma_ann),
        "mom_vol_scaled": float(vol_neutral_mom),
    }])

# Compute features
rows = []
for i, tk in enumerate(tickers):
    try:
        f = compute_features_for_ticker(tk)
        if not f.empty:
            rows.append(f)
    except Exception as e:
        print(f"[WARN] Feature gen failed for {tk}: {e}")
        continue
    if (i+1) % 100 == 0:
        print(f"{i+1}/{len(tickers)} tickers processed")

if len(rows) == 0:
    raise RuntimeError("No features computed for the current universe (insufficient history?).")

feat = pd.concat(rows, ignore_index=True)
# Dimension checks
assert feat["entry_date"].nunique() == 1
assert feat.set_index(["entry_date", "ticker"]).index.is_unique
print("Feature frame shape (rows, cols):", feat.shape)

# MultiIndex output for downstream LTR
features_multi = feat.set_index(["entry_date", "ticker"]).sort_index()
# (Optional) persist to DB or file as needed; left in-memory for prod safety.
# features_multi.to_sql("mtum_features", con=engine, if_exists="append")

# Ranking snapshots (no forward info)
ranked = features_multi.copy()
ranked["score"] = ranked["mom_vol_scaled"].fillna(ranked["mom_12_1"])  # default to 12-1 if vol_scaled NaN
ranked = ranked.sort_values(by="score", ascending=False)

top_decile = ranked.groupby(level=0).head(10)
bot_decile = ranked.groupby(level=0).tail(10)

print("Top decile tickers:", list(top_decile.reset_index()["ticker"]))
print("Bottom decile tickers:", list(bot_decile.reset_index()["ticker"]))
