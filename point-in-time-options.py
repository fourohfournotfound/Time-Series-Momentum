# -*- coding: utf-8 -*-
"""
Point-in-time weekly-options universe builder (Polygon) — optimized.

Key speed-ups:
  • Grouped Daily stocks (one call per date) to compute notional volume.
  • v3 options/contracts with proper `order=asc&sort=expiration_date`, pagination via `next_url`.
  • ThreadPool with conservative max_workers; robust retries honoring Retry-After.
  • Single persistent Session with HTTPAdapter(retries).

Universe logic unchanged (≥ $50mm notional; weekly expirations <9-day avg across next 4 exps).
"""

import os, time, math
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlalchemy
import mysql.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

# -------------------- CONFIG --------------------
polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

MAX_WORKERS = int(os.getenv("MAX_WORKERS_PIT", "8"))   # stay polite re: plan
TIMEOUT = (5, 30)  # (connect, read)
USER_AGENT = "mtum-pit/2.0"
# ------------------------------------------------

# Trading months
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date="2019-01-01", end_date=(datetime.today() - timedelta(days=1))).index
start_of_months = trading_dates.to_series().dt.tz_localize(None).to_series().dt.normalize().drop_duplicates(keep="first")
monthly_dates = start_of_months.dt.strftime("%Y-%m-%d").values

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=MAX_WORKERS * 2))
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s

SESSION = _build_session()

def _get_json(url: str, params: dict | None = None) -> dict:
    """GET wrapper that honors Retry-After and returns {} on hard failure."""
    params = params or {}
    try:
        r = SESSION.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                time.sleep(float(ra))
                r = SESSION.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"HTTP error: {e} | URL={url} | params={params}")
        return {}

def _iter_paginated(first_url: str, first_params: dict) -> list[dict]:
    """Yield v3 pages following next_url if present."""
    out = []
    payload = _get_json(first_url, first_params)
    if not payload or "results" not in payload:
        return out
    out.extend(payload.get("results", []))
    next_url = payload.get("next_url")
    while next_url:
        payload = _get_json(next_url, {"apiKey": polygon_api_key})
        if not payload or "results" not in payload:
            break
        out.extend(payload.get("results", []))
        next_url = payload.get("next_url")
    return out

def _contracts_for_ticker_asof(ticker: str, date: str) -> pd.DataFrame:
    """Fetch options contracts for an underlying at point-in-time date (paginated)."""
    base = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "contract_type": "call",     # as in your baseline
        "as_of": date,
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
        "apiKey": polygon_api_key,
    }
    rows = _iter_paginated(base, params)
    if not rows:
        return pd.DataFrame(columns=["expiration_date"])
    df = pd.json_normalize(rows)
    return df

def _qualifies_weekly(df: pd.DataFrame) -> float | None:
    if df.empty or "expiration_date" not in df.columns:
        return None
    exp = np.sort(pd.to_datetime(df["expiration_date"]).values)
    if exp.size < 5:
        return None
    d = pd.Series(exp).diff().dt.days
    if d.iloc[1:5].size < 4:
        return None
    return float(d.iloc[1:5].mean())

def _process_date(date: str) -> pd.DataFrame:
    """Compute PIT universe for one month start date."""
    # v3 reference tickers (stocks) as-of date (paginated)
    tickers = _iter_paginated(
        "https://api.polygon.io/v3/reference/tickers",
        {
            "date": date, "type": "CS", "market": "stocks", "active": "true",
            "order": "asc", "limit": 1000, "sort": "ticker", "apiKey": polygon_api_key
        }
    )
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "avg_days_between"])

    tickers_df = pd.json_normalize(tickers)[["ticker", "name"]]

    # Grouped Daily stocks for that date (bulk coverage)
    grouped = _get_json(
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}",
        {"adjusted": "true", "apiKey": polygon_api_key}
    )
    if "results" not in grouped:
        return pd.DataFrame(columns=["date", "ticker", "avg_days_between"])

    gdf = pd.json_normalize(grouped["results"]).rename(columns={"T": "ticker"})
    gdf["date"] = pd.to_datetime(gdf["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    gdf["notional_volume"] = gdf["vw"] * gdf["v"]

    merged = pd.merge(
        gdf[["date", "c", "notional_volume", "ticker"]],
        tickers_df[["ticker", "name"]],
        on="ticker",
        how="inner"
    )

    valid = merged[merged["notional_volume"] >= 5e7].copy().sort_values("notional_volume", ascending=False)
    symbols = valid["ticker"].drop_duplicates().values

    out_rows = []

    def _worker(tkr: str):
        try:
            df = _contracts_for_ticker_asof(tkr, date)
            avg_days = _qualifies_weekly(df)
            if avg_days is not None and avg_days < 9:
                return {"date": date, "ticker": tkr, "avg_days_between": avg_days}
        except Exception as e:
            print(f"contracts err {tkr} @ {date}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_worker, t) for t in symbols]
        for f in as_completed(futs):
            row = f.result()
            if row:
                out_rows.append(row)

    return pd.DataFrame(out_rows)

# -------------------- RUN --------------------
all_rows = []
times = []
for i, date in enumerate(monthly_dates):
    start = time.time()
    df = _process_date(date)
    if not df.empty:
        all_rows.append(df)
    dt = time.time() - start
    times.append(dt)
    remaining = len(monthly_dates) - (i + 1)
    avg = np.mean(times)
    eta = datetime.now() + timedelta(seconds=int(avg * remaining))
    print(f"{round(((i+1)/len(monthly_dates))*100,2)}% complete | {len(df)} tickers | ETA ~ {eta}")

if all_rows:
    full = pd.concat(all_rows).drop_duplicates(subset=["date", "ticker"], keep="last")
    full.to_sql("historical_liquid_tickers_polygon", con=engine, if_exists="append", index=False)
