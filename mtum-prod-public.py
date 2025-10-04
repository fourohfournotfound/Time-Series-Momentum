# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Quant Galore
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import gspread
import sqlalchemy
import mysql.connector

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

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


_load_local_env_file()

DEFAULT_POLYGON_KEY = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
polygon_api_key = os.getenv("POLYGON_API_KEY", DEFAULT_POLYGON_KEY)

if polygon_api_key == DEFAULT_POLYGON_KEY:
    print(
        "Warning: POLYGON_API_KEY not set; using the placeholder key from the "
        "repository. Set your own API key via the POLYGON_API_KEY environment "
        "variable before running in production."
    )

database_url = os.getenv("MTUM_DATABASE_URL")

if not database_url:
    sqlite_path = Path(__file__).with_name("universe.db")
    if sqlite_path.exists():
        database_url = f"sqlite:///{sqlite_path}"
        print(
            "MTUM_DATABASE_URL not set; using the local universe.db SQLite file."
        )

engine = None

if database_url:
    engine = sqlalchemy.create_engine(database_url)
else:
    print(
        "MTUM_DATABASE_URL is not set; falling back to the bundled "
        "historical_liquid_tickers.csv file."
    )

# =============================================================================
# Date Management
# =============================================================================

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date = (datetime.today() - timedelta(days=45)), end_date = (datetime.today())).index.strftime("%Y-%m-%d").values

all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month

start_of_the_months = all_dates.drop_duplicates(subset = ["year", "month"], keep = "first").copy()
start_of_the_months["str_date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")

months = start_of_the_months["str_date"].values
month = months[-1]

start_date = (pd.to_datetime(month) - timedelta(days = 365+60)).strftime("%Y-%m-%d")
end_date = month

last_month_date = (pd.to_datetime(month) - timedelta(days = 30)).strftime("%Y-%m-%d")
next_month_date = (pd.to_datetime(month) + timedelta(days = 30)).strftime("%Y-%m-%d")

# =============================================================================
# Base Point-in-Time Universe + Benchmark Generation
# =============================================================================

if engine is not None:
    universe = pd.read_sql("historical_liquid_tickers_polygon", con=engine)
else:
    csv_path = Path(__file__).with_name("historical_liquid_tickers.csv")
    if not csv_path.exists():
        raise RuntimeError(
            "historical_liquid_tickers.csv is not available and MTUM_DATABASE_URL "
            "was not provided. Upload the CSV alongside this script or set a "
            "database URL."
        )

    universe = pd.read_csv(csv_path)

if "date" not in universe.columns:
    raise RuntimeError("The universe data must include a 'date' column.")

universe["date"] = pd.to_datetime(universe["date"]).dt.strftime("%Y-%m-%d")

benchmark_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# =============================================================================
# Real-Time Basket Construction
# =============================================================================

point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]

point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep = "last")

tickers = point_in_time_universe["ticker"].drop_duplicates().values

full_data_list = []
top_decile_list = []
bot_decile_list = []

times = []

monthly_ticker_list = []
    
# ticker = tickers[np.random.randint(0, len(tickers))]
for ticker in tickers:
    
    try:
        
        start_time = datetime.now()
        
        underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying_data.index = pd.to_datetime(underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        underlying_data["date"] = underlying_data.index.strftime("%Y-%m-%d")
        
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
    
        covariance_matrix = np.cov(benchmark_and_underlying["ticker_pct_change"], benchmark_and_underlying["benchmark_pct_change"])
        covariance_ticker_benchmark = covariance_matrix[0, 1]
        variance_benchmark = np.var(benchmark_and_underlying["benchmark_pct_change"])
        beta = covariance_ticker_benchmark / variance_benchmark
        
        ticker_return_over_period = round(((benchmark_and_underlying["c_y"].iloc[-1] - benchmark_and_underlying["c_y"].iloc[0]) / benchmark_and_underlying["c_y"].iloc[0]) * 100, 2)    
        std_of_returns = benchmark_and_underlying["ticker_pct_change"].std() * np.sqrt(252)
        
        sharpe = ticker_return_over_period / std_of_returns
        
        theo_expected = beta * sharpe
        
        forward_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{end_date}/{next_month_date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        forward_data.index = pd.to_datetime(forward_data.index, unit="ms", utc=True).tz_convert("America/New_York")

        forward_return = round(((forward_data["c"].iloc[-1] - forward_data["c"].iloc[0]) / forward_data["c"].iloc[0]) * 100, 2)    
        
        ticker_data = pd.DataFrame([{"entry_date": month, "ticker": ticker, "beta": beta, "sharpe": sharpe, "12-1_return": ticker_return_over_period, "avg_monthly_return": avg_monthly_return, "mom_score": theo_expected, "forward_returns": forward_return}])
                
        monthly_ticker_list.append(ticker_data)
        
        end_time = datetime.now()    
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(tickers==ticker)[0][0]/len(tickers))*100,2)
        iterations_remaining = len(tickers) - np.where(tickers==ticker)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
        
    except Exception as error:
        print(error)
        continue
    
# =============================================================================
# Separated Deciles    
# =============================================================================

full_period_ticker_data = pd.concat(monthly_ticker_list)
top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending = False).head(10)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending = True).head(10)

