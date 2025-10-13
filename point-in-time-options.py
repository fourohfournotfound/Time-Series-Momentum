# -*- coding: utf-8 -*-
"""Point-in-time liquid weekly-options universe builder.

This implementation forms the tradable universe for a monthly rebalance
at the **first trading day of the month** (``rebalance_date``) using only the
information that was observable at the **prior trading day**
(``as_of_date``).  Liquidity and options-eligibility checks therefore reflect
what would have been known when trading decisions are made, eliminating the
look-ahead present in the previous version that used same-day EOD data.

The resulting table contains ``as_of_date``, ``rebalance_date`` and ``ticker``
columns along with the average spacing between the first five option
expirations (``avg_days_between``).  These values mirror the schema expected by
the production and backtest scripts.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

import mysql.connector
import numpy as np
import pandas as pd
import requests
import sqlalchemy
from pandas_market_calendars import get_calendar


POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
NOTIONAL_VOLUME_THRESHOLD = float(os.getenv("PIT_NOTIONAL_THRESHOLD", "5e7"))
MAX_PAGES = 20

DB_ENGINE = sqlalchemy.create_engine(
    "mysql+mysqlconnector://user:pass@localhost:3306/my_database"
)

# Optional local CSV fallback when DB access is unavailable
CSV_FALLBACK_PATH = os.getenv(
    "PIT_UNIVERSE_CSV", "historical_liquid_tickers.csv"
)


# =============================================================================
# Trading calendar helpers
# =============================================================================

nyse = get_calendar("NYSE")


def trading_days(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    """Return NYSE trading days in New York timezone."""

    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.tz_convert("America/New_York")


today_local = datetime.now(timezone.utc).astimezone()
start_dt = datetime(2019, 1, 1, tzinfo=today_local.tzinfo)
tds = trading_days(start_dt, today_local - timedelta(days=1))

td_df = pd.DataFrame({"date": pd.to_datetime(tds)})
td_df["year"] = td_df["date"].dt.year
td_df["month"] = td_df["date"].dt.month

first_trading_of_month = (
    td_df.drop_duplicates(subset=["year", "month"], keep="first")
    .copy()
    .reset_index(drop=True)
)
rebalance_dates = first_trading_of_month["date"].dt.tz_convert(
    "America/New_York"
)

# Align each rebalance date (t) with the previous trading day (t-1)
td_list = list(tds)
td_pos = {d: i for i, d in enumerate(td_list)}
as_of_dates = []
rebalance_dates_aligned = []
for t in rebalance_dates:
    idx = td_pos.get(t)
    if idx is None or idx == 0:
        continue
    rebalance_dates_aligned.append(t)
    as_of_dates.append(td_list[idx - 1])

rebalance_frame = pd.DataFrame(
    {
        "rebalance_date": pd.to_datetime(rebalance_dates_aligned).tz_convert(
            "America/New_York"
        ),
        "as_of_date": pd.to_datetime(as_of_dates).tz_convert(
            "America/New_York"
        ),
    }
)
rebalance_frame["rebalance_date_str"] = rebalance_frame["rebalance_date"].dt.strftime(
    "%Y-%m-%d"
)
rebalance_frame["as_of_date_str"] = rebalance_frame["as_of_date"].dt.strftime(
    "%Y-%m-%d"
)


# =============================================================================
# HTTP session
# =============================================================================

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=500, max_retries=3)
session.mount("https://", adapter)
session.headers.update({"Accept": "application/json"})


def _get(url: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# =============================================================================
# Data retrieval helpers
# =============================================================================

def _pit_tickers(as_of: str) -> pd.DataFrame:
    """Fetch all common stock tickers active as of ``as_of``."""

    base_url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "date": as_of,
        "type": "CS",
        "market": "stocks",
        "active": "true",
        "order": "asc",
        "limit": 1000,
        "sort": "ticker",
    }

    frames: list[pd.DataFrame] = []
    next_url: str | None = base_url
    for _ in range(MAX_PAGES):
        if not next_url:
            break
        payload = _get(next_url, params=params if next_url == base_url else None)
        results = payload.get("results", [])
        if results:
            frames.append(pd.json_normalize(results))
        next_url = payload.get("next_url")

    if not frames:
        return pd.DataFrame(columns=["ticker"])

    data = pd.concat(frames, ignore_index=True)
    return data


def _liquid_equities(as_of: str, tickers: pd.DataFrame) -> pd.DataFrame:
    """Filter tickers by notional volume using grouped EOD aggregates."""

    grouped = _get(
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{as_of}",
        params={"adjusted": "true"},
    )
    grouped_df = pd.json_normalize(grouped.get("results", [])).rename(columns={"T": "ticker"})
    if grouped_df.empty:
        return pd.DataFrame(columns=["ticker"])

    grouped_df["date"] = (
        pd.to_datetime(grouped_df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    grouped_df["notional_volume"] = grouped_df["vw"] * grouped_df["v"]

    merged = pd.merge(
        grouped_df[["date", "c", "notional_volume", "ticker"]],
        tickers[["ticker", "name"]],
        on="ticker",
        how="inner",
    )

    liquid = (
        merged.loc[merged["notional_volume"] >= NOTIONAL_VOLUME_THRESHOLD]
        .copy()
        .sort_values("notional_volume", ascending=False)
    )
    return liquid[["date", "ticker"]].drop_duplicates()


def _has_weekly_options(ticker: str, as_of: str) -> float | None:
    """Return the average spacing (in days) between the first five expiries if ticker has weekly options."""

    contracts = _get(
        "https://api.polygon.io/v3/reference/options/contracts",
        params={"underlying_ticker": ticker, "as_of": as_of, "limit": 1000},
    )
    contract_df = pd.json_normalize(contracts.get("results", []))
    if contract_df.empty or "expiration_date" not in contract_df:
        return None

    expirations = np.sort(pd.to_datetime(contract_df["expiration_date"]).dropna().unique())
    if len(expirations) < 5:
        return None

    exp_df = pd.DataFrame({"date": pd.to_datetime(expirations[:5])})
    exp_df["days_between"] = exp_df["date"].diff().dt.days
    avg_gap = exp_df["days_between"].iloc[1:5].mean()
    if math.isnan(avg_gap) or avg_gap >= 9:
        return None
    return float(avg_gap)


# =============================================================================
# Universe construction
# =============================================================================

rows: list[pd.DataFrame] = []

for _, row in rebalance_frame.iterrows():
    as_of = row["as_of_date_str"]
    rebalance = row["rebalance_date_str"]

    pit_tickers = _pit_tickers(as_of)
    if pit_tickers.empty:
        continue

    liquid = _liquid_equities(as_of, pit_tickers)
    if liquid.empty:
        continue

    for ticker in liquid["ticker"].drop_duplicates():
        try:
            avg_gap = _has_weekly_options(ticker, as_of)
        except Exception as exc:  # pragma: no cover - network guardrail
            print(f"[WARN] Failed options lookup for {ticker} @ {as_of}: {exc}")
            continue
        if avg_gap is None:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "as_of_date": [as_of],
                    "rebalance_date": [rebalance],
                    "ticker": [ticker],
                    "avg_days_between": [avg_gap],
                }
            )
        )


if not rows:
    raise RuntimeError("No securities passed the PIT weekly options filters.")

universe = (
    pd.concat(rows, ignore_index=True)
    .drop_duplicates(subset=["as_of_date", "rebalance_date", "ticker"], keep="last")
)

required_cols = ["as_of_date", "rebalance_date", "ticker", "avg_days_between"]
if sorted(universe.columns.tolist()) != sorted(required_cols):
    raise AssertionError(f"Unexpected columns: {universe.columns.tolist()}")

if universe[required_cols].isna().sum().sum() != 0:
    raise AssertionError("Null values detected in the PIT universe.")

print("Weekly universe shape:", universe.shape)


def _persist_to_db(df: pd.DataFrame) -> None:
    try:
        df.to_sql(
            "historical_liquid_tickers_polygon",
            con=DB_ENGINE,
            if_exists="append",
            index=False,
            method="multi",
        )
        print("Universe written to database table 'historical_liquid_tickers_polygon'.")
    except Exception as exc:  # pragma: no cover - db guardrail
        print(f"[WARN] Failed to write universe to DB: {exc}")


def _persist_to_csv(df: pd.DataFrame, path: str) -> None:
    try:
        if os.path.exists(path):
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["as_of_date", "rebalance_date", "ticker"], keep="last")
        df.to_csv(path, index=False)
        print(f"Universe cached to CSV at {path}.")
    except Exception as exc:  # pragma: no cover - file guardrail
        print(f"[WARN] Failed to write universe CSV: {exc}")


_persist_to_db(universe)
_persist_to_csv(universe, CSV_FALLBACK_PATH)

