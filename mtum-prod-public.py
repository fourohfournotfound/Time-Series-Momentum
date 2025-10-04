# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Quant Galore
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import gspread
import sqlalchemy
import mysql.connector
from typing import List

from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from pandas_market_calendars import get_calendar


def _parse_args() -> argparse.Namespace:
    """Parse command line options for configuring data sources."""

    parser = argparse.ArgumentParser(description="Run the live momentum basket builder")
    parser.add_argument(
        "--provider",
        choices=["polygon"],
        default="polygon",
        help="Historical data provider (only polygon is supported in the prod script)",
    )
    parser.add_argument(
        "--polygon-api-key",
        default=None,
        help="Polygon.io API key (overrides the POLYGON_API_KEY environment variable)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy URL for the point-in-time universe database",
    )
    parser.add_argument(
        "--universe-table",
        default="historical_liquid_tickers_polygon",
        help="Database table containing the historical liquid tickers universe",
    )
    parser.add_argument(
        "--universe-csv",
        default=None,
        help=(
            "CSV file containing the universe data. When provided the CSV is used in "
            "place of the database connection."
        ),
    )
    return parser.parse_args()

def _load_local_env_file():
    """Populate os.environ with keys from a `.env` file if present.

    This keeps the script lightweight (no python-dotenv dependency) while
    allowing notebook users to drop a credentials file next to the script.
    Environment variables that are already set take precedence over any
    values declared in the file.
    """

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

def _load_colab_user_secrets() -> None:
    """Populate environment variables from Colab user secrets when available."""

    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return

    secret_names = {
        "POLYGON_API_KEY": "POLYGON_API_KEY",
        "ALPACA_API_KEY": "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY": "ALPACA_SECRET_KEY",
    }

    for env_name, secret_name in secret_names.items():
        if os.getenv(env_name):
            continue
        try:
            value = userdata.get(secret_name)
        except Exception:
            continue
        if value:
            os.environ[env_name] = value


_load_local_env_file()
_load_colab_user_secrets()

ARGS = _parse_args()

DEFAULT_POLYGON_KEY = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
polygon_api_key = ARGS.polygon_api_key or os.getenv("POLYGON_API_KEY")

if not polygon_api_key or polygon_api_key == DEFAULT_POLYGON_KEY:
    raise RuntimeError(
        "POLYGON_API_KEY not configured. Provide a valid key via --polygon-api-key or the POLYGON_API_KEY environment variable."
    )

database_url = ARGS.database_url or os.getenv("MTUM_DATABASE_URL")


def _candidate_paths(filename: str) -> List[Path]:
    """Return plausible locations for a data file."""

    path = Path(filename)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(Path(__file__).with_name(filename))
    # Remove duplicates while preserving order
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)
    return unique_candidates

engine = None

if ARGS.universe_csv:
    csv_candidates = _candidate_paths(ARGS.universe_csv)
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        raise RuntimeError(
            f"Universe CSV {ARGS.universe_csv} was requested but could not be found in:"
            f" {[str(path) for path in csv_candidates]}"
        )
    ARGS.universe_csv = str(csv_path)
    print(f"Using CSV universe from {csv_path}")
else:
    if not database_url:
        for sqlite_path in _candidate_paths("universe.db"):
            if sqlite_path.exists():
                database_url = f"sqlite:///{sqlite_path}"
                print(
                    "MTUM_DATABASE_URL not set; using the local universe.db SQLite "
                    f"file at {sqlite_path}."
                )
                break

    if database_url:
        engine = sqlalchemy.create_engine(database_url)
    else:
        default_csv = "historical_liquid_tickers.csv"
        csv_candidates = _candidate_paths(default_csv)
        csv_path = next((path for path in csv_candidates if path.exists()), None)
        if csv_path is None:
            raise RuntimeError(
                "MTUM_DATABASE_URL is not set; the default historical_liquid_tickers.csv "
                "file was not found either. Provide --universe-csv or a database URL."
            )
        ARGS.universe_csv = str(csv_path)
        print(
            "MTUM_DATABASE_URL is not set; falling back to the bundled "
            f"{csv_path.name} file at {csv_path}."
        )


# =============================================================================
# Date Management
# =============================================================================

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(
    start_date=(datetime.today() - timedelta(days=45)),
    end_date=datetime.today(),
).index.strftime("%Y-%m-%d").values

all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month

start_of_the_months = all_dates.drop_duplicates(subset = ["year", "month"], keep = "first").copy()
start_of_the_months["str_date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")

months = start_of_the_months["str_date"].values
month = months[-1]

start_date = (pd.to_datetime(month) - BDay(252 + 42)).strftime("%Y-%m-%d")
end_date = month

last_month_date = (pd.to_datetime(month) - BDay(21)).strftime("%Y-%m-%d")
next_month_date = (pd.to_datetime(month) + BDay(21)).strftime("%Y-%m-%d")

# =============================================================================
# Base Point-in-Time Universe + Benchmark Generation
# =============================================================================

def _load_universe() -> pd.DataFrame:
    if ARGS.universe_csv:
        csv_candidates = _candidate_paths(ARGS.universe_csv)
        csv_path = next((path for path in csv_candidates if path.exists()), None)
        if csv_path is None:
            raise RuntimeError(
                f"Requested universe CSV {ARGS.universe_csv} could not be located in:"
                f" {[str(path) for path in csv_candidates]}"
            )
        frame = pd.read_csv(csv_path)
    elif engine is not None:
        frame = pd.read_sql(ARGS.universe_table, con=engine)
    else:
        raise RuntimeError(
            "No universe data source configured. Provide --universe-csv or a database URL."
        )

    if frame.columns.size:
        first_col = str(frame.columns[0]).strip().lower()
        if first_col in {"", "index", "unnamed: 0"}:
            frame = frame.drop(columns=frame.columns[0])

    if "date" not in frame.columns or "ticker" not in frame.columns:
        raise RuntimeError("The universe data must include 'date' and 'ticker' columns.")

    if "avg_days_between" in frame.columns:
        avg_days = pd.to_numeric(frame["avg_days_between"], errors="coerce")
        frame = frame.loc[avg_days < 9].copy()

    frame = frame.drop_duplicates(subset=["date", "ticker"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


universe = _load_universe()


def _fetch_polygon_aggregates(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily aggregate bars from Polygon."""

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": polygon_api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame()

    frame = pd.json_normalize(results).set_index("t")
    frame.index = (
        pd.to_datetime(frame.index, unit="ms", utc=True)
        .tz_convert("America/New_York")
    )
    frame["date"] = frame.index.strftime("%Y-%m-%d")
    return frame


benchmark_data = _fetch_polygon_aggregates("SPY", "2017-01-01", trading_dates[-1])
if benchmark_data.empty:
    raise RuntimeError("Unable to load SPY benchmark history from Polygon")

# =============================================================================
# Real-Time Basket Construction
# =============================================================================

point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]

point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep = "last")

tickers = point_in_time_universe["ticker"].drop_duplicates().values

times = []

monthly_ticker_list = []
    
# ticker = tickers[np.random.randint(0, len(tickers))]
for idx, ticker in enumerate(tickers, start=1):
    
    try:
        
        start_time = datetime.now()
        
        underlying_data = _fetch_polygon_aggregates(ticker, start_date, end_date)
        if underlying_data.empty:
            continue

        underlying_data["year"] = underlying_data.index.year
        underlying_data["month"] = underlying_data.index.month
        
        twelve_minus_one_data = underlying_data[underlying_data["date"] < last_month_date].copy().tail(252)

        if len(twelve_minus_one_data) < 252:
            continue
        
        twelve_minus_one_data["year"] = twelve_minus_one_data.index.year
        twelve_minus_one_data["month"] = twelve_minus_one_data.index.month
        
        eom_ticker_data = twelve_minus_one_data.drop_duplicates(subset=["year", "month"], keep = "last").copy()
        eom_ticker_data["monthly_return"] = round(eom_ticker_data["c"].pct_change() * 100, 2)
        
        twelve_minus_one_return = round(((twelve_minus_one_data["c"].iloc[-1] - twelve_minus_one_data["c"].iloc[0]) / twelve_minus_one_data["c"].iloc[0]) * 100 , 2)
        avg_monthly_return = eom_ticker_data["monthly_return"].mean()
        
        benchmark_and_underlying = pd.merge(left=benchmark_data[["c", "date"]], right = twelve_minus_one_data[["c","date"]], on = "date")
        benchmark_and_underlying["benchmark_pct_change"] = round(benchmark_and_underlying["c_x"].pct_change() * 100, 2).fillna(0)
        benchmark_and_underlying["ticker_pct_change"] = round(benchmark_and_underlying["c_y"].pct_change() * 100, 2).fillna(0)
    
        covariance_matrix = np.cov(
            benchmark_and_underlying["ticker_pct_change"],
            benchmark_and_underlying["benchmark_pct_change"],
        )
        covariance_ticker_benchmark = covariance_matrix[0, 1]
        variance_benchmark = np.var(
            benchmark_and_underlying["benchmark_pct_change"]
        )
        if variance_benchmark == 0 or np.isnan(variance_benchmark):
            continue
        beta = covariance_ticker_benchmark / variance_benchmark
        
        ticker_return_over_period = round(((benchmark_and_underlying["c_y"].iloc[-1] - benchmark_and_underlying["c_y"].iloc[0]) / benchmark_and_underlying["c_y"].iloc[0]) * 100, 2)    
        std_of_returns = (
            benchmark_and_underlying["ticker_pct_change"].std() * np.sqrt(252)
        )
        if std_of_returns == 0 or np.isnan(std_of_returns):
            continue

        sharpe = ticker_return_over_period / std_of_returns
        
        theo_expected = beta * sharpe
        
        forward_data = _fetch_polygon_aggregates(ticker, end_date, next_month_date)
        if forward_data.empty or forward_data.shape[0] < 2:
            continue

        forward_return = round(((forward_data["c"].iloc[-1] - forward_data["c"].iloc[0]) / forward_data["c"].iloc[0]) * 100, 2)
        
        ticker_data = pd.DataFrame([{"entry_date": month, "ticker": ticker, "beta": beta, "sharpe": sharpe, "12-1_return": ticker_return_over_period, "avg_monthly_return": avg_monthly_return, "mom_score": theo_expected, "forward_returns": forward_return}])
                
        monthly_ticker_list.append(ticker_data)
        
        end_time = datetime.now()    
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((idx / len(tickers)) * 100, 2)
        iterations_remaining = len(tickers) - idx
        average_time_to_complete = np.mean(times)
        estimated_completion_time = datetime.now() + timedelta(
            seconds=int(average_time_to_complete * max(iterations_remaining, 0))
        )
        time_remaining = estimated_completion_time - datetime.now()
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
        
    except Exception as error:
        print(f"{ticker}: {error}")
        continue
    
# =============================================================================
# Separated Deciles
# =============================================================================

if not monthly_ticker_list:
    raise RuntimeError("No qualifying tickers were found for the current rebalance window")

full_period_ticker_data = pd.concat(monthly_ticker_list)
top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending = False).head(10)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending = True).head(10)

display_columns = [
    "ticker",
    "mom_score",
    "beta",
    "sharpe",
    "12-1_return",
    "avg_monthly_return",
    "forward_returns",
]

print("\nTop Decile Momentum Basket (long candidates):")
top_display_columns = [col for col in display_columns if col in top_decile.columns]
print(top_decile[top_display_columns].to_string(index=False))

print("\nBottom Decile Momentum Basket (short candidates):")
bot_display_columns = [col for col in display_columns if col in bot_decile.columns]
print(bot_decile[bot_display_columns].to_string(index=False))

rebalance_output_dir = Path(__file__).resolve().parent / "rebalance_outputs"
rebalance_output_dir.mkdir(exist_ok=True)

rebalance_output_path = rebalance_output_dir / f"rebalance_{month}.csv"

rebalance_export = pd.concat(
    [
        top_decile.assign(decile="top"),
        bot_decile.assign(decile="bottom"),
    ],
    ignore_index=True,
)

rebalance_export.to_csv(rebalance_output_path, index=False)

print(f"\nRebalance baskets saved to {rebalance_output_path}")

