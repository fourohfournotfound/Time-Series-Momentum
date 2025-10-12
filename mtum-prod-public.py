# -*- coding: utf-8 -*-
"""
Created in 2025

Live momentum basket builder (Polygon).

Key changes:
- Strictly avoid same-day selection bias: the point-in-time universe is taken from
  the last monthly snapshot strictly *before* the current rebalance month (fallback
  to <= if the strict set is empty).
- Added explicit shape checks/guards so MultiIndex LTR datasets won’t have misaligned
  arrays (beta/sharpe/mom_score/forward_returns).
- Retained your 12-1 style anchoring and feature shifts to avoid look-ahead.
- Safer DB discovery & CSV fallback; robust comments on adjusted data semantics.

Docs:
- Polygon agg bars (split-adjusted): see polygon Python refs (adjusted by default) and FAQ.
- Alpaca bars API reference maintained in data_providers. 
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import gspread
import sqlalchemy
import mysql.connector
from typing import Iterable, List, Optional

from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from pandas_market_calendars import get_calendar


def _parse_positive_int_list(raw: str, *, min_value: int = 1) -> List[int]:
    values: List[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value < min_value:
            raise argparse.ArgumentTypeError(
                f"All integers must be >= {min_value}; received {value} in '{raw}'"
            )
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one integer value is required")
    return values


def _compute_ewm_momentum(
    prices: pd.Series,
    half_lives: Iterable[int],
    *,
    min_period_fraction: float = 0.5,
) -> pd.DataFrame:
    # All momentum features are shifted by 1 bar to avoid look-ahead.
    returns = prices.pct_change()
    features: dict[str, pd.Series] = {}
    for hl in half_lives:
        min_periods = max(int(math.ceil(hl * min_period_fraction)), 1)
        ewm_mean = returns.ewm(halflife=hl, adjust=False, min_periods=min_periods).mean()
        features[f"ewm_ret_hl_{hl}"] = ewm_mean.shift(1)
    return pd.DataFrame(features)


def _ema_crossover_signal(
    prices: pd.Series,
    fast_hl: int,
    slow_hl: int,
    *,
    min_period_fraction: float = 0.5,
) -> pd.Series:
    if fast_hl >= slow_hl:
        raise ValueError("fast_hl must be strictly less than slow_hl")

    fast_min_periods = max(int(math.ceil(fast_hl * min_period_fraction)), 1)
    slow_min_periods = max(int(math.ceil(slow_hl * min_period_fraction)), 1)

    fast_ma = prices.ewm(halflife=fast_hl, adjust=False, min_periods=fast_min_periods).mean()
    slow_ma = prices.ewm(halflife=slow_hl, adjust=False, min_periods=slow_min_periods).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = fast_ma.divide(slow_ma) - 1.0
    return signal.replace([np.inf, -np.inf], np.nan).shift(1)


def _ewm_volatility(
    returns: pd.Series,
    half_life: int,
    *,
    annualize: int,
    min_period_fraction: float = 0.5,
) -> pd.Series:
    min_periods = max(int(math.ceil(half_life * min_period_fraction)), 1)
    ewm_std = returns.ewm(halflife=half_life, adjust=False, min_periods=min_periods).std()
    return ewm_std.shift(1) * math.sqrt(annualize)


def _yang_zhang_volatility(
    frame: pd.DataFrame,
    window: int,
    *,
    annualize: int,
) -> pd.Series:
    # Yang–Zhang volatility estimator; shifted by 1 to remain causal.
    log_open = np.log(frame["o_ticker"])
    log_high = np.log(frame["h_ticker"])
    log_low = np.log(frame["l_ticker"])
    log_close = np.log(frame["c_ticker"])

    log_prev_close = log_close.shift(1)
    log_prev_open = log_open.shift(1)

    close_vol = (log_close - log_prev_close) ** 2
    open_vol = (log_open - log_prev_close) ** 2
    rs = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (log_low - log_open)

    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))

    close_var = close_vol.rolling(window=window, min_periods=window).sum() * (1.0 / max(window - 1, 1))
    open_var = open_vol.rolling(window=window, min_periods=window).sum() * (1.0 / max(window - 1, 1))
    rs_var = rs.rolling(window=window, min_periods=window).sum() * (1.0 / max(window - 1, 1))

    yang_zhang_var = open_var + k * close_var + (1.0 - k) * rs_var
    yang_zhang_var = yang_zhang_var.clip(lower=0.0)
    return np.sqrt(yang_zhang_var).shift(1) * math.sqrt(annualize)

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
        "--annualization-factor",
        type=int,
        default=252,
        help="Trading periods per year used to annualise volatility",
    )
    parser.add_argument(
        "--momentum-type",
        choices=["classic", "ewm"],
        default="classic",
        help="Momentum flavour: classic beta*Sharpe or EWMA-based",
    )
    parser.add_argument(
        "--ewm-half-lives",
        default="63,126,252",
        help="Comma-separated half-lives (days) for EWMA momentum",
    )
    parser.add_argument(
        "--ema-crossover",
        action="store_true",
        help="Add an EMA crossover signal when using EWMA momentum",
    )
    parser.add_argument(
        "--ema-fast-hl",
        type=int,
        default=32,
        help="Half-life for the fast EMA in the crossover signal",
    )
    parser.add_argument(
        "--ema-slow-hl",
        type=int,
        default=128,
        help="Half-life for the slow EMA in the crossover signal",
    )
    parser.add_argument(
        "--vol-estimator",
        choices=["none", "ewm", "yang-zhang"],
        default="none",
        help="Volatility estimator for risk-adjusting the momentum score",
    )
    parser.add_argument(
        "--vol-ewm-halflife",
        type=int,
        default=20,
        help="Half-life for the EWM volatility estimator",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=20,
        help="Window length for the Yang–Zhang volatility estimator",
    )
    return parser.parse_args()

def _load_local_env_file():
    """Populate os.environ with keys from a `.env` file if present."""
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
ARGS.ewm_half_lives = _parse_positive_int_list(ARGS.ewm_half_lives)
if ARGS.ema_crossover and ARGS.ema_fast_hl >= ARGS.ema_slow_hl:
    raise ValueError("--ema-fast-hl must be smaller than --ema-slow-hl for the crossover signal")

polygon_api_key = ARGS.polygon_api_key or os.getenv("POLYGON_API_KEY")
if not polygon_api_key:
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

start_of_the_months = all_dates.drop_duplicates(subset=["year", "month"], keep="first").copy()
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
    """Fetch daily aggregate bars from Polygon (split-adjusted by default)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame()
    frame = pd.json_normalize(results).set_index("t")
    frame.index = pd.to_datetime(frame.index, unit="ms", utc=True).tz_convert("America/New_York")
    frame["date"] = frame.index.strftime("%Y-%m-%d")
    return frame

benchmark_data = _fetch_polygon_aggregates("SPY", "2017-01-01", trading_dates[-1])
if benchmark_data.empty:
    raise RuntimeError("Unable to load SPY benchmark history from Polygon")

benchmark_data = benchmark_data.sort_values("t").drop_duplicates(subset="date", keep="last")
benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
benchmark_data = benchmark_data.sort_values("date").reset_index(drop=True)
benchmark_pct_change = benchmark_data["c"].pct_change()
if not benchmark_pct_change.empty:
    benchmark_pct_change.iloc[0] = 0.0
benchmark_data["benchmark_pct_change"] = benchmark_pct_change * 100

# =============================================================================
# Real-Time Basket Construction
# =============================================================================

# Use the monthly snapshot STRICTLY prior to 'month' to avoid same-day selection bias.
prior_snapshot_mask = universe["date"] < month
if not prior_snapshot_mask.any():
    # fallback to <= month if strictly prior doesn't exist (earliest history)
    prior_snapshot_mask = universe["date"] <= month

point_in_time_dates = np.sort(universe[prior_snapshot_mask]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]

point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")
tickers = point_in_time_universe["ticker"].drop_duplicates().values

times: List[float] = []
monthly_ticker_list: List[pd.DataFrame] = []

for idx, ticker in enumerate(tickers, start=1):
    try:
        start_time = datetime.now()

        underlying_data = _fetch_polygon_aggregates(ticker, start_date, end_date)
        if underlying_data.empty:
            continue

        underlying_data["year"] = underlying_data.index.year
        underlying_data["month"] = underlying_data.index.month

        history_window = max(
            252,
            max(ARGS.ewm_half_lives, default=252),
            ARGS.vol_window if ARGS.vol_estimator == "yang-zhang" else 0,
        )

        # Exclude the most recent month (12-1 style skip) to prevent reversal leakage
        twelve_minus_one_data = underlying_data[underlying_data["date"] < last_month_date].copy().tail(history_window)

        if len(twelve_minus_one_data) < history_window:
            # Not enough history to compute robust metrics
            continue

        twelve_minus_one_data["year"] = twelve_minus_one_data.index.year
        twelve_minus_one_data["month"] = twelve_minus_one_data.index.month

        eom_ticker_data = twelve_minus_one_data.drop_duplicates(subset=["year", "month"], keep="last").copy()
        eom_ticker_data["monthly_return"] = round(eom_ticker_data["c"].pct_change() * 100, 2)
        avg_monthly_return = float(eom_ticker_data["monthly_return"].mean()) if len(eom_ticker_data) > 0 else np.nan

        ohlc_frame = (
            twelve_minus_one_data.copy()
            .assign(date=pd.to_datetime(twelve_minus_one_data["date"]))
            .set_index("date")[["o", "h", "l", "c"]]
            .sort_index()
        )
        close_series = ohlc_frame["c"]

        feature_window = min(len(close_series), 252)
        if feature_window < 2:
            continue

        merged = pd.merge(
            ohlc_frame.reset_index().rename(
                columns={"o": "o_ticker", "h": "h_ticker", "l": "l_ticker", "c": "c_ticker"}
            ),
            benchmark_data[["date", "c", "benchmark_pct_change"]].rename(columns={"c": "c_benchmark"}),
            on="date",
            how="inner",
            sort=True,
        )
        if merged.shape[0] < feature_window:
            continue

        merged = merged.sort_values("date").reset_index(drop=True)
        ticker_returns = merged["c_ticker"].pct_change()
        if not ticker_returns.empty:
            ticker_returns.iloc[0] = 0.0
        merged["ticker_pct_change"] = ticker_returns * 100

        rolling_cov = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).cov(
            merged["benchmark_pct_change"]
        )
        benchmark_var = merged["benchmark_pct_change"].rolling(window=feature_window, min_periods=feature_window).var()
        beta_series = rolling_cov.divide(benchmark_var).replace([np.inf, -np.inf], np.nan)

        total_return = merged["c_ticker"].pct_change(periods=feature_window - 1) * 100
        rolling_std = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).std()
        rolling_std = rolling_std * np.sqrt(ARGS.annualization_factor)
        rolling_std = rolling_std.replace(0, np.nan)
        sharpe_series = total_return.divide(rolling_std).replace([np.inf, -np.inf], np.nan)

        metrics = (
            pd.DataFrame(
                {
                    "date": merged["date"],
                    "beta": beta_series,
                    "sharpe": sharpe_series,
                    "12-1_return": total_return,
                }
            )
            .set_index("date")
            .sort_index()
        )

        mom_components: List[pd.Series] = []
        if ARGS.momentum_type == "classic":
            classic_score = beta_series * sharpe_series
            metrics["classic_mom"] = classic_score
            mom_components.append(classic_score)
        else:
            ewm_features = _compute_ewm_momentum(close_series, ARGS.ewm_half_lives)
            for column in ewm_features.columns:
                metrics[column] = ewm_features[column].reindex(metrics.index)
                mom_components.append(metrics[column])
            if ARGS.ema_crossover:
                ema_signal = _ema_crossover_signal(close_series, ARGS.ema_fast_hl, ARGS.ema_slow_hl)
                metrics["ema_crossover"] = ema_signal.reindex(metrics.index)
                mom_components.append(metrics["ema_crossover"])

        composite = pd.concat(mom_components, axis=1).mean(axis=1) if mom_components else pd.Series(index=metrics.index, dtype=float)

        # Optional volatility normalization (shifted internally)
        if ARGS.vol_estimator != "none":
            if ARGS.vol_estimator == "ewm":
                volatility = _ewm_volatility(close_series.pct_change(), ARGS.vol_ewm_halflife, annualize=ARGS.annualization_factor)
            else:
                volatility = _yang_zhang_volatility(
                    ohlc_frame.rename(columns={"o": "o_ticker", "h": "h_ticker", "l": "l_ticker", "c": "c_ticker"}),
                    ARGS.vol_window,
                    annualize=ARGS.annualization_factor,
                )
            metrics["volatility"] = volatility.reindex(metrics.index)
            with np.errstate(divide="ignore", invalid="ignore"):
                composite = composite.divide(metrics["volatility"])
            composite = composite.replace([np.inf, -np.inf], np.nan)

        metrics["mom_score"] = composite
        metrics = metrics.dropna(subset=["mom_score"])
        if metrics.empty:
            continue

        # Use last available metric value BEFORE the rebalance month
        current_metrics = metrics.iloc[-1]

        # Forward return from first bar in 'month' to ~21bd later
        forward_data = _fetch_polygon_aggregates(ticker, end_date, next_month_date)
        if forward_data.empty or forward_data.shape[0] < 2:
            continue
        forward_return = round(((forward_data["c"].iloc[-1] - forward_data["c"].iloc[0]) / forward_data["c"].iloc[0]) * 100, 2)

        record = {
            "entry_date": month,
            "ticker": ticker,
            "beta": float(current_metrics.get("beta", np.nan)),
            "sharpe": float(current_metrics.get("sharpe", np.nan)),
            "12-1_return": float(current_metrics.get("12-1_return", np.nan)),
            "avg_monthly_return": float(avg_monthly_return) if pd.notna(avg_monthly_return) else np.nan,
            "mom_score": float(current_metrics.get("mom_score", np.nan)),
            "forward_returns": forward_return,
        }

        if "volatility" in metrics.columns:
            record["volatility"] = float(current_metrics.get("volatility", np.nan))
        if "classic_mom" in metrics.columns:
            record["classic_mom"] = float(current_metrics.get("classic_mom", np.nan))
        for column in metrics.columns:
            if isinstance(column, str) and column.startswith("ewm_ret_hl_"):
                record[column] = float(current_metrics.get(column, np.nan))
        if "ema_crossover" in metrics.columns:
            record["ema_crossover"] = float(current_metrics.get("ema_crossover", np.nan))

        monthly_ticker_list.append(pd.DataFrame([record]))

        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((idx / len(tickers)) * 100, 2)
        iterations_remaining = len(tickers) - idx
        average_time_to_complete = np.mean(times) if times else 0.0
        estimated_completion_time = datetime.now() + timedelta(seconds=int(average_time_to_complete * max(iterations_remaining, 0)))
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

full_period_ticker_data = pd.concat(monthly_ticker_list, ignore_index=True)
top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(10)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(10)

display_columns = [
    "ticker",
    "mom_score",
    "beta",
    "sharpe",
    "12-1_return",
    "avg_monthly_return",
]
if "volatility" in full_period_ticker_data.columns:
    display_columns.append("volatility")
ewm_columns = sorted([col for col in full_period_ticker_data.columns if col.startswith("ewm_ret_hl_")])
display_columns.extend(ewm_columns)
if "ema_crossover" in full_period_ticker_data.columns:
    display_columns.append("ema_crossover")
if "classic_mom" in full_period_ticker_data.columns:
    display_columns.append("classic_mom")
display_columns.append("forward_returns")

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
    [top_decile.assign(decile="top"), bot_decile.assign(decile="bottom")],
    ignore_index=True,
)
rebalance_export.to_csv(rebalance_output_path, index=False)

print(f"\nRebalance baskets saved to {rebalance_output_path}")
