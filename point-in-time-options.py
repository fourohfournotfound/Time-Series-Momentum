# -*- coding: utf-8 -*-
"""
Created in 2025

Updated: Enforce true point-in-time (PIT) universe with as-of (t-1) and rebalance_date (t).
- Liquidity screen and options-eligibility evaluated strictly as of t-1 (previous trading day).
- Stores both as_of_date and rebalance_date to enable leak-free prod/backtest selection.
- Adds shape checks and safer API handling.

Economic rationale: use prior-day liquidity to ensure tradeable universe at the next day's open.
"""

import os
import time
import math
import pandas as pd
import numpy as np
import requests
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta, timezone
from pandas_market_calendars import get_calendar

# Prefer env var (safer in prod); fall back to previous value if not set.
polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")

# ---- DB engine ----
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

# =============================================================================
# Calendar utilities
# =============================================================================
nyse = get_calendar("NYSE")

def trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    return sched.index.tz_convert("America/New_York")

# Build first trading day of each month (rebalance_date = t) and its prior trading day (as_of_date = t-1)
end_dt = (datetime.now(timezone.utc) - timedelta(days=1)).astimezone()  # yesterday
start_dt = datetime(2019, 1, 1).astimezone()
tds = trading_days(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

all_dates = pd.DataFrame({"date": pd.to_datetime(tds)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month
first_of_month_t = (all_dates
                    .drop_duplicates(subset=["year", "month"], keep="first")
                    .copy()
                    .reset_index(drop=True))
rebalance_dates = first_of_month_t["date"].dt.tz_convert("America/New_York")
rebalance_dates = rebalance_dates[rebalance_dates > rebalance_dates.min()]  # ensure there is a prior day

# Map each rebalance_date (t) to the previous trading day (t-1)
td_list = list(tds)
td_pos = {d: i for i, d in enumerate(td_list)}
as_of_dates = []
for t in rebalance_dates:
    idx = td_pos[t]
    if idx == 0:
        continue  # safety
    as_of_dates.append(td_list[idx - 1])

rebalance_frame = pd.DataFrame({
    "rebalance_date": pd.to_datetime(rebalance_dates[:len(as_of_dates)]).tz_convert("America/New_York"),
    "as_of_date": pd.to_datetime(as_of_dates).tz_convert("America/New_York"),
})
rebalance_frame["rebalance_date_str"] = rebalance_frame["rebalance_date"].dt.strftime("%Y-%m-%d")
rebalance_frame["as_of_date_str"] = rebalance_frame["as_of_date"].dt.strftime("%Y-%m-%d")

# =============================================================================
# API session
# =============================================================================
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=500, max_retries=3)
session.mount('https://', adapter)
session.headers.update({"Accept": "application/json"})

def _get(url, params=None):
    params = params or {}
    params["apiKey"] = polygon_api_key
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# =============================================================================
# Build PIT universe as of t-1 for trade at t
# =============================================================================
weekly_ticker_data_list = []
times = []

# Parameters
NOTIONAL_VOL_THRESHOLD = 5e7  # $50mm
MAX_PAGES = 20  # pagination guard

for _, row in rebalance_frame.iterrows():
    start_time = datetime.now()
    as_of = row["as_of_date_str"]
    reb_t = row["rebalance_date_str"]

    # --- Point-in-time equity list as of t-1 (common stock, active that day) ---
    # Pagination
    pit_url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "date": as_of,
        "type": "CS",
        "market": "stocks",
        "active": "true",
        "order": "asc",
        "limit": 1000,
        "sort": "ticker",
    }

    pit_data_pages = []
    next_url = pit_url
    seen = 0
    for _ in range(MAX_PAGES):
        if next_url == pit_url:
            payload = _get(pit_url, params=params)
        else:
            payload = _get(next_url)
        results = payload.get("results", [])
        if not results:
            break
        pit_data_pages.append(pd.json_normalize(results))
        next_url = payload.get("next_url")
        if not next_url:
            break
        seen += 1
        if seen >= MAX_PAGES:
            break

    if not pit_data_pages:
        continue
    tickers_on_as_of = pd.concat(pit_data_pages, ignore_index=True)
    assert tickers_on_as_of.shape[0] > 0, "No PIT tickers retrieved"

    # --- Liquidity screen using grouped market summary on t-1 (EOD) ---
    grouped = _get(f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{as_of}",
                   params={"adjusted": "true"})
    grouped_df = pd.json_normalize(grouped.get("results", []))
    if grouped_df.empty:
        continue
    grouped_df = grouped_df.rename(columns={"T": "ticker"})
    grouped_df["date"] = pd.to_datetime(grouped_df["t"], unit="ms", utc=True)\
                               .dt.tz_convert("America/New_York")\
                               .dt.strftime("%Y-%m-%d")
    # notional = VWAP * shares traded
    grouped_df["notional_volume"] = grouped_df["vw"] * grouped_df["v"]

    merged = pd.merge(
        grouped_df[["date", "c", "notional_volume", "ticker"]],
        tickers_on_as_of[["ticker", "name"]],
        on="ticker",
        how="inner"
    )
    liquid = merged.loc[merged["notional_volume"] >= NOTIONAL_VOL_THRESHOLD]\
                   .copy()\
                   .sort_values("notional_volume", ascending=False)

    tickers = liquid["ticker"].drop_duplicates().to_numpy()
    if len(tickers) == 0:
        continue

    # --- Options weekly expiration check (as of t-1) ---
    for tk in tickers:
        try:
            # Using as_of for strict PIT; we don't rely on same-day discovery at t.
            # Note: Including both calls & puts; expiration_date dedup handles overlap.
            contracts = _get("https://api.polygon.io/v3/reference/options/contracts",
                             params={"underlying_ticker": tk, "as_of": as_of, "limit": 1000})
            res = pd.json_normalize(contracts.get("results", []))
            if res.empty or "expiration_date" not in res.columns:
                continue

            exps = np.sort(pd.to_datetime(res["expiration_date"]).unique())
            if len(exps) < 5:
                continue
            # compute average spacing between first 5 expirations
            d = pd.DataFrame({"date": pd.to_datetime(exps[:5])})
            d["days_between"] = d["date"].diff().dt.days
            avg_gap = d["days_between"].iloc[1:5].mean()
            if math.isnan(avg_gap) or avg_gap >= 9:
                continue  # not a weekly chain

            weekly_ticker_data_list.append(pd.DataFrame([{
                "as_of_date": as_of,
                "rebalance_date": reb_t,
                "ticker": tk,
                "avg_days_between": float(avg_gap),
            }]))
        except Exception as e:
            # Keep going; a single symbol failure shouldn't stop universe formation
            print(f"[WARN] {tk} @ {as_of}: {e}")
            continue

    # progress report
    dt = datetime.now() - start_time
    times.append(dt.total_seconds())
    pct = round((rebalance_frame.index.get_loc(_) / len(rebalance_frame)) * 100, 2)
    avg = np.mean(times)
    remaining = len(rebalance_frame) - rebalance_frame.index.get_loc(_)
    print(f"{pct}% complete | avg {avg:.1f}s/it | est ~{avg*remaining/60:.1f} min left")

# =============================================================================
# Final storage
# =============================================================================
if weekly_ticker_data_list:
    weekly_universe = pd.concat(weekly_ticker_data_list, ignore_index=True).drop_duplicates(
        subset=["as_of_date", "rebalance_date", "ticker"], keep="last"
    )

    # Sanity checks
    assert {"as_of_date", "rebalance_date", "ticker"}.issubset(weekly_universe.columns)
    assert weekly_universe[["as_of_date", "rebalance_date", "ticker"]].isna().sum().sum() == 0
    print("Weekly universe shape:", weekly_universe.shape)

    weekly_universe.to_sql("historical_liquid_tickers_polygon", con=engine, if_exists="append", index=False, method="multi")
else:
    print("No symbols passed filters. Nothing to write.")
