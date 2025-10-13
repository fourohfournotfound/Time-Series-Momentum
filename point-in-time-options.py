# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Quant Galore

Point-in-time universe generation. Adds --strict-weeklies to match
the article's "at least 6 consecutive weekly listings". Default behavior
(average days < 9) is unchanged unless you pass --strict-weeklies.

Output schema (per row): date (YYYY-MM-DD), ticker, avg_days_between (float)

Notes:
- Uses Polygon's reference endpoints.
- Avoid look-ahead: all queries use `as_of=<date>`.
"""

import os
import argparse

import pandas as pd
import numpy as np
import requests
import sqlalchemy
import mysql.connector

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("point-in-time universe builder")
    p.add_argument("--database-url",
                   default='mysql+mysqlconnector://user:pass@localhost:3306/my_database',
                   help="SQLAlchemy URL for output DB (use sqlite:///universe.db for SQLite)")
    p.add_argument("--table", default="historical_liquid_tickers_polygon",
                   help="Output table name")
    p.add_argument("--start-date", default="2019-01-01",
                   help="Earliest date to consider (YYYY-MM-DD)")
    p.add_argument("--min-notional", type=float, default=5e7,
                   help="Min notional volume filter (default 50,000,000)")
    p.add_argument("--strict-weeklies", action="store_true",
                   help="Require >=6 consecutive expirations with ~weekly spacing (5-9 days)")
    p.add_argument("--min-weekly-expirations", type=int, default=6,
                   help="Minimum consecutive expirations for strict weeklies")
    return p.parse_args()


def _load_colab_user_secrets() -> None:
    """Populate environment variables from Colab user secrets when available."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    polygon_key = userdata.get("POLYGON_API_KEY") if not os.getenv("POLYGON_API_KEY") else None
    if polygon_key:
        os.environ["POLYGON_API_KEY"] = polygon_key


_load_colab_user_secrets()

DEFAULT_POLYGON_KEY = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
polygon_api_key = os.getenv("POLYGON_API_KEY")
if not polygon_api_key or polygon_api_key == DEFAULT_POLYGON_KEY:
    raise RuntimeError(
        "POLYGON_API_KEY not configured. Set the environment variable before running point-in-time options extraction."
    )


def _strict_weekly_ok(expirations: list[str], *, min_consecutive: int) -> bool:
    """Return True if we have >=min_consecutive expirations spaced about weekly (5-9 days)."""
    if len(expirations) < min_consecutive:
        return False
    dates = pd.to_datetime(expirations).sort_values().to_numpy()
    diffs = np.diff(dates).astype("timedelta64[D]").astype(int)
    if diffs.size < (min_consecutive - 1):
        return False
    # Sliding window: any run of (min_consecutive-1) gaps all within [5, 9]
    for i in range(0, diffs.size - (min_consecutive - 2)):
        window = diffs[i:i + (min_consecutive - 1)]
        if np.all((window >= 5) & (window <= 9)):
            return True
    return False


def main() -> None:
    args = _parse_args()
    engine = sqlalchemy.create_engine(args.database-url if hasattr(args, "database-url") else args.database_url)

    # =============================================================================
    # Generation of monthly dates.
    # =============================================================================

    calendar = get_calendar("NYSE")
    trading_dates = calendar.schedule(
        start_date=args.start_date,
        end_date=(datetime.today() - timedelta(days=1))
    ).index.strftime("%Y-%m-%d").values

    all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    all_dates["year"] = all_dates["date"].dt.year
    all_dates["month"] = all_dates["date"].dt.month

    start_of_the_months = all_dates.drop_duplicates(subset=["year", "month"], keep="first").copy()
    start_of_the_months["date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")
    monthly_dates = start_of_the_months["date"].drop_duplicates().values

    # =============================================================================
    # Retrieval of optionable stocks
    # =============================================================================

    weekly_ticker_data_list = []
    times = []

    for date in monthly_dates:
        pit_data_list = []
        pit_url_list = []

        pit_request_0 = requests.get(
            f"https://api.polygon.io/v3/reference/tickers?date={date}&type=CS&market=stocks&active=true&order=asc&limit=1000&sort=ticker&apiKey={polygon_api_key}"
        ).json()
        pit_next_url = pit_request_0.get("next_url", None)
        pit_data = pd.json_normalize(pit_request_0.get("results", []))

        if pit_next_url:
            pit_url_list.append(pit_next_url)
        if not pit_data.empty:
            pit_data_list.append(pit_data)

        for _ in range(0, 10):
            try:
                if not pit_url_list:
                    break
                pit_request_n = requests.get(f"{pit_url_list[-1]}&apiKey={polygon_api_key}").json()
                pit_data_n = pd.json_normalize(pit_request_n.get("results", []))
                if not pit_data_n.empty:
                    pit_data_list.append(pit_data_n)
                pit_request_n_next_url = pit_request_n.get("next_url", None)
                if not pit_request_n_next_url:
                    break
                pit_url_list.append(pit_request_n_next_url)
            except Exception as error:
                print(error)
                break

        tickers_on_that_date = pd.concat(pit_data_list) if pit_data_list else pd.DataFrame(columns=["ticker", "name"])

        grouped_payload = requests.get(
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true&apiKey={polygon_api_key}"
        ).json()
        grouped_results = grouped_payload.get("results", [])
        if not grouped_results:
            continue
        grouped_ticker_data = pd.json_normalize(grouped_results).rename(columns={"T": "ticker"})
        grouped_ticker_data["date"] = pd.to_datetime(grouped_ticker_data["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        grouped_ticker_data["notional_volume"] = grouped_ticker_data["vw"] * grouped_ticker_data["v"]

        grouped_valid_tickers = pd.merge(
            left=grouped_ticker_data[["date", "c", "notional_volume", "ticker"]],
            right=tickers_on_that_date[["ticker", "name"]],
            on="ticker",
        )

        # at least $50mm in volume that day (configurable)
        valid_monthly_universe = grouped_valid_tickers[grouped_valid_tickers["notional_volume"] >= float(args.min_notional)].copy().sort_values(by="notional_volume", ascending=False)
        tickers = valid_monthly_universe["ticker"].drop_duplicates().values

        # ticker loop
        for ticker in tickers:
            start_time = datetime.now()
            try:
                contracts_results = []
                unique_expirations = set()
                expiration_filter = ""
                additional_requests = 0

                while len(unique_expirations) < max(6, args.min_weekly_expirations) and additional_requests < 12:
                    contracts_response = requests.get(
                        f"https://api.polygon.io/v3/reference/options/contracts?"
                        f"underlying_ticker={ticker}&contract_type=call&as_of={date}"
                        f"&sort=expiration_date&order=asc&limit=8{expiration_filter}"
                        f"&apiKey={polygon_api_key}"
                    )
                    if contracts_response.status_code != 200:
                        break

                    page_payload = contracts_response.json()
                    page_results = page_payload.get("results", [])
                    if not page_results:
                        break

                    contracts_results.extend(page_results)
                    expirations_seen = [
                        result.get("expiration_date")
                        for result in page_results
                        if result.get("expiration_date")
                    ]
                    unique_expirations.update(expirations_seen)

                    last_expiration = expirations_seen[-1] if expirations_seen else None
                    expiration_filter = f"&expiration_date.gt={last_expiration}" if last_expiration else ""
                    additional_requests += 1

                    if len(unique_expirations) >= max(6, args.min_weekly_expirations):
                        break

                if not contracts_results:
                    continue

                all_contracts = pd.json_normalize(contracts_results)
                earliest_contracts = (
                    all_contracts.sort_values("expiration_date")
                    .drop_duplicates(subset=["expiration_date"])
                    .head(max(6, args.min_weekly_expirations))
                )

                if earliest_contracts.shape[0] < max(6, args.min_weekly_expirations):
                    continue

                exp_date_data = earliest_contracts[["expiration_date"]].rename(columns={"expiration_date": "date"})
                # --- strict weeklies check (optional)
                if args.strict_weeklies:
                    exp_list = exp_date_data["date"].astype(str).tolist()
                    if not _strict_weekly_ok(exp_list, min_consecutive=max(6, args.min_weekly_expirations)):
                        continue

                exp_date_data["date"] = pd.to_datetime(exp_date_data["date"]).dt.tz_localize("America/New_York")
                exp_date_data["days_between"] = exp_date_data["date"].diff().dt.days
                avg_days_between_exps = exp_date_data["days_between"][1:].mean()

                # default heuristic (unchanged): average gap < 9 days
                if not args.strict_weeklies and avg_days_between_exps >= 9:
                    continue

                weekly_ticker_data = pd.DataFrame([{"date": date, "ticker": ticker, "avg_days_between": avg_days_between_exps}])
                weekly_ticker_data_list.append(weekly_ticker_data)

            except Exception as error:
                print(error)
                continue

            end_time = datetime.now()
            seconds_to_complete = (end_time - start_time).total_seconds()
            times.append(seconds_to_complete)
            iteration = round((np.where(tickers == ticker)[0][0] / len(tickers)) * 100, 2)
            iterations_remaining = len(tickers) - np.where(tickers == ticker)[0][0]
            average_time_to_complete = np.mean(times)
            estimated_completion_time = (datetime.now() + timedelta(seconds=int(average_time_to_complete * iterations_remaining)))
            time_remaining = estimated_completion_time - datetime.now()
            print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

    # =============================================================================
    # Final data storage.
    # =============================================================================

    if not weekly_ticker_data_list:
        raise RuntimeError("No weekly optionable tickers found. Check API key/quota and date range.")

    full_weekly_ticker_data = pd.concat(weekly_ticker_data_list).drop_duplicates(subset=["date", "ticker"], keep="last")
    full_weekly_ticker_data.to_sql(args.table, con=engine, if_exists="append", index=False)


if __name__ == "__main__":
    main()
