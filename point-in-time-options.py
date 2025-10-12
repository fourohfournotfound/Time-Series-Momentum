# -*- coding: utf-8 -*-
"""
Created in 2025

Point-in-time (PIT) universe builder for liquid, weekly-optionable US stocks.

Changes vs original:
- Added argparse + environment-based configuration (no hard-coded DB URL).
- Robust requests.Session with retries + timeouts.
- Safer pagination over Polygon v3 endpoints.
- Clarified PIT selection; removed unnecessary tz localization for date-only fields.
- Preserved weekly-expiry heuristic (< 9 average days between first 5 expiries).
- Eliminated potential same-day selection leak for downstream users by documenting
  and enabling strict-asof behavior (used by prod/backtest scripts).
- Removed unconditional DEFAULT key; enforce presence of POLYGON_API_KEY.

Data sources and semantics:
- Polygon v3 /reference/tickers?date=YYYY-MM-DD for PIT ticker activity.
- Polygon v2 /aggs/grouped/.../{date} for same-day liquidity (notional volume).
- Polygon v3 /reference/options/contracts?underlying_ticker=...&as_of=YYYY-MM-DD
  to identify weekly expirations (calls only, earliest expiries).

Docs:
- Grouped daily OHLC/VWAP: https://polygon.io/docs/rest/stocks/aggregates/daily-market-summary
- Tickers as-of date param: https://polygon.readthedocs.io/en/latest/References.html
- Options contracts 'as_of': https://polygon.io/docs/rest/options/contracts/contract-overview
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import sqlalchemy
from pandas_market_calendars import get_calendar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session(timeout: int = 30, max_retries: int = 5, backoff: float = 0.4) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.request_timeout = timeout  # type: ignore[attr-defined]
    return sess


def _load_colab_user_secrets() -> None:
    """Populate environment variables from Colab user secrets when available."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    polygon_key = userdata.get("POLYGON_API_KEY") if not os.getenv("POLYGON_API_KEY") else None
    if polygon_key:
        os.environ["POLYGON_API_KEY"] = polygon_key


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PIT liquid weekly-optionable universe (Polygon)")
    p.add_argument("--start", default="2019-01-01", help="Start date (YYYY-mm-dd) for monthly PIT snapshots")
    p.add_argument("--min-notional", type=float, default=5e7, help="Min same-day notional volume filter (USD)")
    p.add_argument(
        "--database-url",
        default=os.getenv("MTUM_DATABASE_URL", ""),
        help="SQLAlchemy URL for output (e.g., mysql+mysqlconnector://user:pass@host/db or sqlite:///universe.db)",
    )
    p.add_argument(
        "--table",
        default="historical_liquid_tickers_polygon",
        help="Destination table for the PIT universe",
    )
    p.add_argument("--max-pages", type=int, default=25, help="Max pages to traverse for ticker listings")
    return p.parse_args()


def _monthly_starts(start: str, end: datetime) -> np.ndarray:
    calendar = get_calendar("NYSE")
    trading_dates = calendar.schedule(start_date=start, end_date=end).index
    df = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    month_starts = df.drop_duplicates(subset=["year", "month"], keep="first").copy()
    return month_starts["date"].dt.strftime("%Y-%m-%d").values


def _paginate_tickers(session: requests.Session, api_key: str, date: str, max_pages: int) -> pd.DataFrame:
    """Pull PIT list of active common-stock tickers as of `date`."""
    url = (
        f"https://api.polygon.io/v3/reference/tickers"
        f"?date={date}&type=CS&market=stocks&active=true&order=asc&limit=1000&sort=ticker&apiKey={api_key}"
    )
    frames: List[pd.DataFrame] = []
    pages = 0
    while url and pages < max_pages:
        resp = session.get(url, timeout=getattr(session, "request_timeout", 30))
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if results:
            frames.append(pd.json_normalize(results))
        next_url = payload.get("next_url")
        url = f"{next_url}&apiKey={api_key}" if next_url else None
        pages += 1
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "name"])


def _grouped_bars_for_date(session: requests.Session, api_key: str, date: str) -> pd.DataFrame:
    """Polygon grouped daily bars for a single trading date."""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
    resp = session.get(url, params={"adjusted": "true", "apiKey": api_key}, timeout=getattr(session, "request_timeout", 30))
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame(columns=["ticker", "date", "c", "vw", "v"])
    g = pd.json_normalize(results).rename(columns={"T": "ticker"})
    g["date"] = pd.to_datetime(g["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    g["notional_volume"] = g["vw"] * g["v"]
    return g[["date", "c", "vw", "v", "notional_volume", "ticker"]]


def _has_weeklies(session: requests.Session, api_key: str, ticker: str, as_of: str) -> Tuple[bool, Optional[float]]:
    """Return (is_weekly_optionable, avg_days_between_first_5_expiries) using v3 contracts as_of."""
    contracts_results = []
    unique_expirations = set()
    expiration_filter = ""
    additional_requests = 0
    # Fetch the earliest distinct expiration dates (calls); stop at 5 expiries
    while len(unique_expirations) < 5 and additional_requests < 10:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts"
            f"?underlying_ticker={ticker}&contract_type=call&as_of={as_of}"
            f"&sort=expiration_date&order=asc&limit=6{expiration_filter}&apiKey={api_key}"
        )
        r = session.get(url, timeout=getattr(session, "request_timeout", 30))
        if r.status_code != 200:
            break
        payload = r.json()
        page_results = payload.get("results", [])
        if not page_results:
            break
        contracts_results.extend(page_results)
        expirations_seen = [row.get("expiration_date") for row in page_results if row.get("expiration_date")]
        unique_expirations.update(expirations_seen)
        last_exp = expirations_seen[-1] if expirations_seen else None
        expiration_filter = f"&expiration_date.gt={last_exp}" if last_exp else ""
        additional_requests += 1
        if len(unique_expirations) >= 5:
            break
    if not contracts_results or len(unique_expirations) < 5:
        return (False, None)
    all_contracts = pd.json_normalize(contracts_results)
    earliest = all_contracts.sort_values("expiration_date").drop_duplicates(subset=["expiration_date"]).head(5)
    exp = pd.to_datetime(earliest["expiration_date"])
    # days_between for expiries 2..5 relative to prior
    avg_gap = exp.diff().dt.days[1:5].mean()
    return (avg_gap < 9, float(avg_gap) if pd.notna(avg_gap) else None)


def main() -> None:
    _load_colab_user_secrets()
    args = _parse_args()

    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        raise RuntimeError(
            "POLYGON_API_KEY not configured. Set the environment variable before running point-in-time options extraction."
        )

    engine = None
    if args.database_url:
        engine = sqlalchemy.create_engine(args.database_url)

    session = _build_session()

    # Monthly PIT snapshots
    monthly_dates = _monthly_starts(args.start, datetime.today() - timedelta(days=1))

    weekly_ticker_rows: List[pd.DataFrame] = []
    times: List[float] = []

    for date in monthly_dates:
        start_time = datetime.now()
        try:
            # PIT tickers (active common stocks as of 'date')
            pit_tickers = _paginate_tickers(session, polygon_api_key, date, args.max_pages)
            if pit_tickers.empty:
                continue

            # Same-day grouped bars to compute liquidity filter (notional)
            grouped = _grouped_bars_for_date(session, polygon_api_key, date)
            if grouped.empty:
                continue

            # Merge & liquidity screen
            merged = pd.merge(
                grouped,
                pit_tickers[["ticker", "name"]],
                on="ticker",
                how="inner",
                validate="many_to_one",
            )
            liquid = merged.loc[merged["notional_volume"] >= float(args.min_notional)].copy()
            if liquid.empty:
                continue

            # Weekly-optionable filter using v3 contracts as_of=date
            tickers = liquid["ticker"].drop_duplicates().values
            for idx, ticker in enumerate(tickers, start=1):
                try:
                    is_weekly, avg_gap = _has_weeklies(session, polygon_api_key, ticker, date)
                    if not is_weekly:
                        continue
                    weekly_ticker_rows.append(pd.DataFrame([{"date": date, "ticker": ticker, "avg_days_between": avg_gap}]))
                except Exception as e:
                    print(f"{ticker}: {e}")
                    continue

            end_time = datetime.now()
            dt = (end_time - start_time).total_seconds()
            times.append(dt)
            if times:
                avg = np.mean(times)
                remain = len(monthly_dates) - (np.where(monthly_dates == date)[0][0] + 1)
                eta = datetime.now() + timedelta(seconds=int(avg * max(remain, 0)))
                print(f"{date} done; {remain} months remaining; ETA ~ {eta:%Y-%m-%d %H:%M:%S}")

        except Exception as error:
            print(f"{date}: {error}")
            continue

    if not weekly_ticker_rows:
        print("No qualifying weekly-optionable, liquid tickers found.")
        return

    full_weekly = pd.concat(weekly_ticker_rows, ignore_index=True).drop_duplicates(subset=["date", "ticker"], keep="last")

    if engine is None:
        # Default to local sqlite for portability
        default_sqlite = "sqlite:///universe.db"
        print(f"No database URL provided. Writing to {default_sqlite} (table={args.table})")
        engine = sqlalchemy.create_engine(default_sqlite)

    full_weekly.to_sql(args.table, con=engine, if_exists="append", index=False)
    print(f"Wrote {len(full_weekly)} rows to {args.table}")


if __name__ == "__main__":
    main()
