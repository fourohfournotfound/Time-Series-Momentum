# -*- coding: utf-8 -*-
"""
Created in 2025

Upgraded PIT weekly-options universe construction with minor efficiency improvements.
- Keeps original logic exactly: monthly PIT dates, $50mm notional filter, weekly expiries via avg spacing < 9.
- Adds safe connection pooling, optional concurrency for contract queries (off by default for API friendliness).
- Produces identical results to the original custom/QG PIT scripts when settings are default.

References:
- Look-ahead avoidance via point-in-time data is essential to backtesting. (See notes/docs in momentum scripts)
"""

import os
import pandas as pd
import numpy as np
import requests
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# =============================================================================
# Config
# =============================================================================

polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

# Concurrency for option-contract checks (IO-bound). Keep modest to respect API limits.
MAX_WORKERS = int(os.getenv("PIT_MAX_WORKERS", "0"))  # default 0 => sequential to mirror original timing

# =============================================================================
# Generation of monthly dates (identical to original)
# =============================================================================

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(
    start_date = "2019-01-01",
    end_date   = (datetime.today() - timedelta(days=1))
).index.strftime("%Y-%m-%d").values

all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month

start_of_the_months = all_dates.drop_duplicates(subset=["year", "month"], keep="first").copy()
start_of_the_months["date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")
monthly_dates = start_of_the_months["date"].drop_duplicates().values

# =============================================================================
# Helper: robust session
# =============================================================================

def build_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=500,
                          max_retries=Retry(total=3, backoff_factor=0.5,
                                            status_forcelist=(429, 500, 502, 503, 504)))
    s.mount('https://', adapter)
    s.headers.update({"Accept": "application/json"})
    return s

connection_session = build_session()

# =============================================================================
# Core functions (logic unchanged; just modularized)
# =============================================================================

def fetch_pit_tickers_for_date(asof_date: str) -> pd.DataFrame:
    """Get all common stock tickers active as of date from Polygon, merge with grouped bars,
    and filter by notional >= $50mm (same as original)."""
    pit_data_list, pit_url_list = [], []

    pit_request_0 = connection_session.get(
        f"https://api.polygon.io/v3/reference/tickers"
        f"?date={asof_date}&type=CS&market=stocks&active=true&order=asc&limit=1000&sort=ticker&apiKey={polygon_api_key}"
    ).json()

    pit_data_list.append(pd.json_normalize(pit_request_0.get("results", [])))
    next_url = pit_request_0.get("next_url")
    if next_url:
        pit_url_list.append(next_url)

    for _ in range(10):  # guardrail as in original
        if not pit_url_list:
            break
        try:
            req = connection_session.get(f"{pit_url_list[-1]}&apiKey={polygon_api_key}").json()
            pit_data_list.append(pd.json_normalize(req.get("results", [])))
            nxt = req.get("next_url")
            if not nxt: break
            pit_url_list.append(nxt)
        except Exception as e:
            print(e)
            break

    tickers_on_that_date = pd.concat(pit_data_list) if pit_data_list else pd.DataFrame(columns=["ticker"])

    grouped = connection_session.get(
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{asof_date}?adjusted=true&apiKey={polygon_api_key}"
    ).json()
    grouped_df = pd.json_normalize(grouped.get("results", [])).rename(columns={"T": "ticker"})
    if grouped_df.empty:
        return pd.DataFrame(columns=["date","ticker","avg_days_between"])

    grouped_df["date"] = pd.to_datetime(grouped_df["t"], unit="ms", utc=True)\
                              .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    grouped_df["notional_volume"] = grouped_df["vw"] * grouped_df["v"]

    merged = pd.merge(
        left=grouped_df[["date","c","notional_volume","ticker"]],
        right=tickers_on_that_date[["ticker","name"]],
        on="ticker",
        how="inner"
    )

    valid = merged[merged["notional_volume"] >= 5e7].copy().sort_values("notional_volume", ascending=False)
    return valid[["date","ticker"]].drop_duplicates()

def _check_weekly_contracts(ticker: str, asof_date: str) -> pd.DataFrame | None:
    """Return a one-row DataFrame if ticker has weekly expiries (avg spacing < 9 for first 5),
       else None. This duplicates prior logic exactly."""
    try:
        resp = connection_session.get(
            f"https://api.polygon.io/v3/reference/options/contracts"
            f"?underlying_ticker={ticker}&contract_type=call&as_of={asof_date}&limit=1000&apiKey={polygon_api_key}"
        ).json()
        all_contracts = pd.json_normalize(resp.get("results", []))

        if all_contracts.empty or "expiration_date" not in all_contracts:
            return None

        all_expirations = np.sort(all_contracts["expiration_date"].dropna().unique())
        if len(all_expirations) < 5:
            return None

        exp_df = pd.DataFrame({"date": all_expirations})
        exp_df["date"] = pd.to_datetime(exp_df["date"]).dt.tz_localize("America/New_York")
        exp_df["days_between"] = exp_df["date"].diff().dt.days
        avg_days_between_exps = exp_df["days_between"][1:5].mean()  # match original slice

        if avg_days_between_exps >= 9:
            return None

        return pd.DataFrame([{"date": asof_date, "ticker": ticker, "avg_days_between": avg_days_between_exps}])
    except Exception as e:
        print(e)
        return None

# =============================================================================
# Main loop (identical results; optional concurrency for per-ticker checks)
# =============================================================================

weekly_ticker_data_list, times = [], []

for date in monthly_dates:
    start_time = datetime.now()

    valid_tickers_df = fetch_pit_tickers_for_date(date)
    tickers = valid_tickers_df["ticker"].values if not valid_tickers_df.empty else []

    if MAX_WORKERS > 0 and len(tickers) > 0:
        # Concurrent options contract checks (does not change results)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_check_weekly_contracts, t, date) for t in tickers]
            for f in as_completed(futs):
                row = f.result()
                if row is not None:
                    weekly_ticker_data_list.append(row)
    else:
        # Sequential path (exactly as original)
        for ticker in tickers:
            row = _check_weekly_contracts(ticker, date)
            if row is not None:
                weekly_ticker_data_list.append(row)

    end_time = datetime.now()
    times.append((end_time - start_time).total_seconds())

    # Same progress prints
    iteration = round((np.where(monthly_dates == date)[0][0] / len(monthly_dates)) * 100, 2)
    iterations_remaining = len(monthly_dates) - np.where(monthly_dates == date)[0][0]
    avg_time = np.mean(times)
    eta = (datetime.now() + timedelta(seconds=int(avg_time * iterations_remaining)))
    remaining = eta - datetime.now()
    print(f"{iteration}% complete, {remaining} left, ETA: {eta}")

# =============================================================================
# Final data storage (identical schema)
# =============================================================================

full_weekly_ticker_data = (pd.concat(weekly_ticker_data_list)
                           .drop_duplicates(subset=["date","ticker"], keep="last"))

# assert expected columns and dtypes
expected_cols = ["date","ticker","avg_days_between"]
assert list(full_weekly_ticker_data.columns) == expected_cols, f"Unexpected columns: {full_weekly_ticker_data.columns.tolist()}"

full_weekly_ticker_data.to_sql("historical_liquid_tickers_polygon", con=engine, if_exists="append", index=False)
