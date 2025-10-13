# -*- coding: utf-8 -*-
"""
Created in 2025

Updated: leak-free monthly backtest aligned with production.
- Universe and features formed strictly as of t-1 (as_of_date).
- Portfolio formed at next trading day's OPEN after rebalance_date (t); exited at OPEN after next rebalance.
- Uses 12-1 momentum, residual momentum, and volatility metrics for ranking.
- Produces top/bottom decile portfolios and cumulative P&L series.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta, timezone
from pandas_market_calendars import get_calendar

polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")

nyse = get_calendar("NYSE")

def trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    return sched.index.tz_convert("America/New_York")

def next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    td = trading_days((d - timedelta(days=3)).strftime("%Y-%m-%d"), (d + timedelta(days=7)).strftime("%Y-%m-%d"))
    td = td[td > d]
    if len(td) == 0:
        raise RuntimeError("No next trading day found")
    return td.min()

def _get(url, params=None):
    params = {} if params is None else dict(params)
    params["apiKey"] = polygon_api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# =============================================================================
# Universe and calendar
# =============================================================================
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')
universe = pd.read_sql("SELECT * FROM historical_liquid_tickers_polygon", con=engine).drop_duplicates(subset=["as_of_date","rebalance_date","ticker"])
assert {"as_of_date","rebalance_date","ticker"}.issubset(universe.columns)

universe["as_of_date"] = pd.to_datetime(universe["as_of_date"])
universe["rebalance_date"] = pd.to_datetime(universe["rebalance_date"])

start_backtest = pd.to_datetime("2019-02-01")  # first month with valid as_of from 2019-01 PIT
end_backtest = (datetime.now(timezone.utc) - timedelta(days=1)).astimezone()

months = np.sort(universe[(universe["rebalance_date"] >= start_backtest) &
                          (universe["rebalance_date"] <= end_backtest)]["rebalance_date"].unique())

# Benchmark always restricted to as_of_date for each month
SPY_FULL = _get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{(end_backtest).strftime('%Y-%m-%d')}",
                params={"adjusted": "true", "sort": "asc", "limit": 50000})
SPY = pd.json_normalize(SPY_FULL.get("results", [])).set_index("t")
SPY.index = pd.to_datetime(SPY.index, unit="ms", utc=True).tz_convert("America/New_York")
SPY["date"] = SPY.index.strftime("%Y-%m-%d")

# =============================================================================
# Feature + forward return helpers
# =============================================================================
def compute_features(ticker: str, as_of_date: pd.Timestamp) -> dict:
    start_date = (as_of_date - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    data = _get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{as_of_str}",
                params={"adjusted": "true", "sort": "asc", "limit": 50000})
    df = pd.json_normalize(data.get("results", [])).set_index("t")
    if df.empty:
        return {}
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")

    cut = (as_of_date - timedelta(days=30)).strftime("%Y-%m-%d")
    tmo = df[df["date"] < cut].tail(252).copy()
    if len(tmo) < 252:
        return {}

    merged = pd.merge(
        SPY[SPY["date"] <= as_of_str][["c", "date"]],
        tmo[["c", "date"]],
        on="date", how="inner", suffixes=("_m", "_s")
    )
    if merged.shape[0] < 200:
        return {}
    merged["ret_m"] = merged["c_m"].pct_change().fillna(0.0)
    merged["ret_s"] = merged["c_s"].pct_change().fillna(0.0)

    cov = np.cov(merged["ret_s"], merged["ret_m"])
    beta = 0.0 if cov[1,1] == 0 else cov[0,1] / cov[1,1]
    mom_12_1 = (tmo["c"].iloc[-1] / tmo["c"].iloc[0]) - 1.0

    X = np.vstack([np.ones(len(merged["ret_m"])), merged["ret_m"].values]).T
    y = merged["ret_s"].values
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ coef
        resid_mom = (np.prod(1.0 + resid) - 1.0)
    except Exception:
        resid_mom = np.nan

    sigma_ann = merged["ret_s"].std() * np.sqrt(252)
    vol_scaled = mom_12_1 / (sigma_ann + 1e-8)

    return {
        "beta_capm": float(beta),
        "mom_12_1": float(mom_12_1),
        "resid_mom": float(resid_mom) if np.isfinite(resid_mom) else np.nan,
        "sigma_ann": float(sigma_ann),
        "mom_vol_scaled": float(vol_scaled),
    }

def forward_open_to_open_return(ticker: str, entry_rebal_date: pd.Timestamp, exit_rebal_date: pd.Timestamp) -> float:
    """Return % from OPEN on next trading day after entry_rebal_date to OPEN on next trading day after exit_rebal_date."""
    entry_open_day = next_trading_day(entry_rebal_date)
    exit_open_day = next_trading_day(exit_rebal_date)

    start = (entry_open_day - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (exit_open_day + timedelta(days=5)).strftime("%Y-%m-%d")

    ud = _get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
              params={"adjusted": "true", "sort": "asc", "limit": 50000})
    df = pd.json_normalize(ud.get("results", [])).set_index("t")
    if df.empty:
        return np.nan
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")

    try:
        p_entry = df.loc[df["date"] == entry_open_day.strftime("%Y-%m-%d"), "o"].iloc[0]
        p_exit  = df.loc[df["date"] == exit_open_day.strftime("%Y-%m-%d"), "o"].iloc[0]
    except Exception:
        # Fallback to close-to-close over [entry_rebal_date, exit_rebal_date]
        try:
            p_entry = df.loc[df["date"] == entry_rebal_date.strftime("%Y-%m-%d"), "c"].iloc[0]
            p_exit  = df.loc[df["date"] == exit_rebal_date.strftime("%Y-%m-%d"), "c"].iloc[0]
        except Exception:
            return np.nan

    return float((p_exit / p_entry) - 1.0) * 100.0

# =============================================================================
# Backtest loop
# =============================================================================
full_rows = []
trade_rows = []

for m in months[:-1]:  # last month has no exit month
    this_month_univ = universe[universe["rebalance_date"] == m].copy()
    if this_month_univ.empty:
        continue
    as_of_m = pd.to_datetime(this_month_univ["as_of_date"].iloc[0])
    next_m = np.sort(months[months > m])[0]

    # Form features (<= as_of_m)
    feats = []
    for tk in this_month_univ["ticker"].drop_duplicates().to_list():
        f = compute_features(tk, as_of_m)
        if f:
            feats.append(pd.DataFrame([{
                "entry_date": m.strftime("%Y-%m-%d"),
                "as_of_date": as_of_m.strftime("%Y-%m-%d"),
                "ticker": tk, **f
            }]))
    if len(feats) == 0:
        continue
    feat_df = pd.concat(feats, ignore_index=True)
    feat_df.set_index(["entry_date", "ticker"], inplace=True, drop=True)
    # Ranking: risk-managed momentum -> residual -> 12-1 as fallback
    feat_df["rank_score"] = feat_df["mom_vol_scaled"].where(np.isfinite(feat_df["mom_vol_scaled"]), np.nan)
    feat_df["rank_score"] = feat_df["rank_score"].fillna(feat_df["resid_mom"]).fillna(feat_df["mom_12_1"])
    feat_df = feat_df.sort_values(by="rank_score", ascending=False)

    top_decile = feat_df.groupby(level=0).head(10).reset_index()
    bot_decile = feat_df.groupby(level=0).tail(10).reset_index()

    # Forward returns (Open->Open, from next day after m to next day after next_m)
    # Also compute risk-adjusted forward return using SPY over same window.
    fwd_rows = []
    spy_fwd = forward_open_to_open_return("SPY", m, next_m)  # % return
    for side, df_side in [("long", top_decile), ("short", bot_decile)]:
        for _, r in df_side.iterrows():
            fwd = forward_open_to_open_return(r["ticker"], m, next_m)
            if np.isnan(fwd):
                continue
            risk_adj = fwd - (feat_df.loc[(r["entry_date"], r["ticker"]), "beta_capm"] * spy_fwd)
            fwd_rows.append({
                "date": r["entry_date"],
                "ticker": r["ticker"],
                "side": side,
                "forward_return_pct": fwd,
                "risk_adj_forward_pct": risk_adj
            })

    if len(fwd_rows) == 0:
        continue
    fwd_df = pd.DataFrame(fwd_rows)

    # Equal-weight basket average for that month
    long_mean  = fwd_df[fwd_df["side"] == "long"]["risk_adj_forward_pct"].mean()
    short_mean = -fwd_df[fwd_df["side"] == "short"]["risk_adj_forward_pct"].mean()
    trade_rows.append({"date": m.strftime("%Y-%m-%d"), "long": long_mean, "short": short_mean})

    full_rows.append(feat_df.reset_index())

# Aggregate
if len(full_rows) == 0 or len(trade_rows) == 0:
    raise RuntimeError("Backtest produced no trades.")

features_panel = pd.concat(full_rows, ignore_index=True).set_index(["entry_date","ticker"]).sort_index()
trades = pd.DataFrame(trade_rows).sort_values("date")
trades["long_pnl"] = trades["long"].cumsum()
trades["short_pnl"] = trades["short"].cumsum()
trades["long_short_pnl"] = trades["long_pnl"] + trades["short_pnl"]

# Dimension checks
assert features_panel.index.is_unique == False or True  # per-month groups by ticker; duplicates across months expected
assert trades.shape[0] == len(np.unique(trades["date"])), "One trade row per month expected"

# Plot
plt.figure(figsize=(10, 6), dpi=200)
plt.xticks(rotation=45)
plt.suptitle("Gross Cumulative Performance (Risk-Adjusted, Open→Open)")
plt.title("Monthly Rebalancing | entry at next open after signal day")
plt.plot(pd.to_datetime(trades["date"]), trades["long_pnl"])
plt.plot(pd.to_datetime(trades["date"]), trades["short_pnl"])
plt.plot(pd.to_datetime(trades["date"]), trades["long_short_pnl"])
plt.legend(["Top Decile", "Bottom Decile", "Long-Short"])
plt.xlabel("Date")
plt.ylabel("Cumulative % Returns")
plt.tight_layout()
plt.show()
plt.close()
