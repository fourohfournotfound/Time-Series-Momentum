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

from momentum_signals import (
    ema_crossover_price,
    ewm_momentum_returns,
    ewm_volatility,
    make_log_returns_from_prices,
    parse_halflife_string,
    rolling_volatility,
    sharpe_like_trend,
    trailing_return,
    yang_zhang_vol,
)


ANNUALIZATION = 252
FEATURE_WINDOW = 252

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
    parser.add_argument(
        "--momentum-style",
        choices=["classic", "ewma", "ewma_sr", "ema_crossover", "composite"],
        default="classic",
        help="Momentum signal to use for live ranking",
    )
    parser.add_argument(
        "--ewm-half-lives",
        default="63,126,252",
        help="Comma separated half-lives (trading days) for EWMA momentum components",
    )
    parser.add_argument(
        "--ema-fast-halflife",
        type=int,
        default=32,
        help="Half-life for the fast EMA in the crossover",
    )
    parser.add_argument(
        "--ema-slow-halflife",
        type=int,
        default=128,
        help="Half-life for the slow EMA in the crossover",
    )
    parser.add_argument(
        "--vol-estimator",
        choices=["rolling", "ewm", "yang-zhang"],
        default="ewm",
        help="Volatility estimator used for risk scaling",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=20,
        help="Window for rolling/Yang-Zhang volatility estimators",
    )
    parser.add_argument(
        "--vol-ewm-halflife",
        type=int,
        default=20,
        help="Half-life for the EWMA volatility estimator",
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


def _compute_volatility_series(returns: pd.Series, prices: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    if args.vol_estimator == "rolling":
        vol = rolling_volatility(returns, window=args.vol_window)
        return vol, "vol_rolling_ann"
    if args.vol_estimator == "ewm":
        vol = ewm_volatility(returns, hl=args.vol_ewm_halflife)
        return vol, "vol_ewm_ann"
    vol = yang_zhang_vol(prices, window=args.vol_window)
    return vol, "vol_yz_ann"


def _compute_momentum_score(metrics: pd.DataFrame, style: str) -> pd.Series:
    if style == "classic":
        if {"beta", "sharpe"}.issubset(metrics.columns):
            return metrics["beta"] * metrics["sharpe"]
        return metrics.get("mom_score", pd.Series(index=metrics.index, dtype=float))

    if style == "ewma":
        ewma_cols = [col for col in metrics.columns if col.startswith("mom_ewm_hl") and not col.endswith("_sr")]
        if ewma_cols:
            return metrics[ewma_cols].mean(axis=1)

    if style == "ewma_sr":
        sr_cols = [col for col in metrics.columns if col.endswith("_sr")]
        if sr_cols:
            return metrics[sr_cols].mean(axis=1)

    if style == "ema_crossover":
        if "ema_xover_norm" in metrics.columns:
            return metrics["ema_xover_norm"]

    if style == "composite":
        components = []
        sr_cols = [col for col in metrics.columns if col.endswith("_sr")]
        if sr_cols:
            components.append(metrics[sr_cols].mean(axis=1))
        if "ema_xover_norm" in metrics.columns:
            components.append(metrics["ema_xover_norm"])
        ewma_cols = [col for col in metrics.columns if col.startswith("mom_ewm_hl") and not col.endswith("_sr")]
        if ewma_cols:
            components.append(metrics[ewma_cols].mean(axis=1))
        if components:
            stacked = pd.concat(components, axis=1)
            return stacked.mean(axis=1)

    return pd.Series(index=metrics.index, dtype=float)


ARGS = _parse_args()

try:
    EWM_HALF_LIVES = parse_halflife_string(ARGS.ewm_half_lives)
except ValueError as exc:
    raise RuntimeError(f"Invalid --ewm-half-lives '{ARGS.ewm_half_lives}': {exc}") from exc

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

        price_frame = underlying_data.copy()
        price_frame.index = pd.to_datetime(price_frame["date"])
        price_frame = price_frame.sort_index()
        price_frame = price_frame.loc[~price_frame.index.duplicated(keep="last")]
        if "c" not in price_frame.columns:
            continue

        price_components = {"close": price_frame["c"]}
        if "o" in price_frame.columns:
            price_components["open"] = price_frame["o"]
        if "h" in price_frame.columns:
            price_components["high"] = price_frame["h"]
        if "l" in price_frame.columns:
            price_components["low"] = price_frame["l"]

        ohlc_frame = pd.DataFrame(price_components)
        ohlc_frame.index.name = "date"
        ohlc_frame["ticker"] = ticker
        multi_price = (
            ohlc_frame.set_index("ticker", append=True)
            .reorder_levels(["ticker", "date"])
            .sort_index()
        )
        multi_price.index.set_names(["ticker", "date"], inplace=True)

        returns_series = make_log_returns_from_prices(multi_price, price_col="close")

        try:
            vol_series, vol_column = _compute_volatility_series(returns_series, multi_price, ARGS)
        except KeyError as err:
            if ARGS.vol_estimator == "yang-zhang":
                print(f"{ticker}: {err}. Falling back to EWMA volatility.")
                vol_series = ewm_volatility(returns_series, hl=ARGS.vol_ewm_halflife)
                vol_column = "vol_ewm_ann"
            else:
                raise

        anchor_date = pd.to_datetime(twelve_minus_one_data["date"].iloc[-1])
        feature_index = pd.DatetimeIndex([anchor_date])

        metrics = pd.DataFrame(
            {
                "beta": [beta],
                "sharpe": [sharpe],
                "12-1_return": [ticker_return_over_period],
            },
            index=feature_index,
        )

        feature_block = pd.DataFrame(index=feature_index, dtype=float)
        vol_single = vol_series.xs(ticker, level="ticker")
        feature_block[vol_column] = vol_single.reindex(feature_index, method="ffill")
        feature_block["volatility_ann"] = feature_block[vol_column]

        trailing_series = trailing_return(multi_price, window=FEATURE_WINDOW, price_col="close")
        trailing_single = trailing_series.xs(ticker, level="ticker") * 100.0
        feature_block[f"return_{FEATURE_WINDOW}d"] = trailing_single.reindex(feature_index, method="ffill")

        mom_ewm_df = None
        if ARGS.momentum_style in {"ewma", "ewma_sr", "composite"}:
            mom_ewm_df = ewm_momentum_returns(returns_series, half_lives=EWM_HALF_LIVES)
            mom_ewm_single = mom_ewm_df.xs(ticker, level="ticker")
            for column in mom_ewm_single.columns:
                feature_block[column] = mom_ewm_single[column].reindex(feature_index, method="ffill")

        if ARGS.momentum_style in {"ewma_sr", "composite"} and mom_ewm_df is not None:
            sr_df = sharpe_like_trend(mom_ewm_df, vol_series, annualize=ANNUALIZATION)
            sr_single = sr_df.xs(ticker, level="ticker")
            for column in sr_single.columns:
                feature_block[column] = sr_single[column].reindex(feature_index, method="ffill")

        if ARGS.momentum_style in {"ema_crossover", "composite"}:
            ema_series = ema_crossover_price(
                multi_price,
                fast_hl=ARGS.ema_fast_halflife,
                slow_hl=ARGS.ema_slow_halflife,
                price_col="close",
            )
            ema_single = ema_series.xs(ticker, level="ticker")
            feature_block["ema_xover_norm"] = ema_single.reindex(feature_index, method="ffill")

        metrics = metrics.join(feature_block, how="left")
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        metrics["mom_score"] = _compute_momentum_score(metrics, ARGS.momentum_style)

        if metrics["mom_score"].isna().all() or metrics["volatility_ann"].isna().all():
            continue

        forward_data = _fetch_polygon_aggregates(ticker, end_date, next_month_date)
        if forward_data.empty or forward_data.shape[0] < 2:
            continue

        forward_return = round(((forward_data["c"].iloc[-1] - forward_data["c"].iloc[0]) / forward_data["c"].iloc[0]) * 100, 2)
        forward_days = max(forward_data.shape[0] - 1, 1)

        vol_value = metrics["volatility_ann"].iloc[0]
        entry_vol_daily = vol_value / np.sqrt(ANNUALIZATION) if pd.notna(vol_value) else np.nan
        if pd.isna(metrics["mom_score"].iloc[0]) or pd.isna(vol_value):
            continue

        forward_return_sr = np.nan
        if entry_vol_daily and entry_vol_daily > 0:
            forward_return_sr = (forward_return / 100.0) / (entry_vol_daily * np.sqrt(forward_days))

        record = {
            "entry_date": month,
            "anchor_date": anchor_date.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "beta": float(metrics["beta"].iloc[0]),
            "sharpe": float(metrics["sharpe"].iloc[0]),
            "12-1_return": float(metrics["12-1_return"].iloc[0]),
            "avg_monthly_return": avg_monthly_return,
            "mom_score": float(metrics["mom_score"].iloc[0]),
            "forward_returns": forward_return,
            "forward_return_sr": forward_return_sr,
            "volatility_ann": vol_value,
            "vol_column": vol_column,
            "momentum_style": ARGS.momentum_style,
            "vol_estimator": ARGS.vol_estimator,
            "ewm_half_lives": ",".join(str(v) for v in EWM_HALF_LIVES),
            "ema_fast_halflife": ARGS.ema_fast_halflife,
            "ema_slow_halflife": ARGS.ema_slow_halflife,
            "holding_period_days": forward_days,
        }

        signal_columns = [
            col
            for col in metrics.columns
            if col.startswith("mom_ewm_hl")
            or col.endswith("_sr")
            or col == "ema_xover_norm"
            or col.startswith("return_")
        ]
        for column in signal_columns:
            value = metrics[column].iloc[0]
            record[column] = float(value) if pd.notna(value) else np.nan

        ticker_data = pd.DataFrame([record])

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
    "forward_return_sr",
    "volatility_ann",
    "momentum_style",
    "vol_estimator",
    "vol_column",
    "anchor_date",
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

