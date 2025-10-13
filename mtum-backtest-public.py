# -*- coding: utf-8 -*-
"""Momentum backtest with configurable data sources, caching, and optional entropy."""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy

from pandas.tseries.offsets import BDay
from pandas_market_calendars import get_calendar

from data_providers import AlpacaBarsClient, PolygonAggsClient
import relative_market_features as rmf  # NEW: entropy + peer metrics module

# ---------------------- existing helpers (unchanged) ----------------------

def _parse_positive_int_list(raw: str, *, min_value: int = 1) -> List[int]:
    """Parse a comma-separated list of positive integers."""
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
    """Return exponentially weighted momentum features for ``prices``.

    The output columns are named ``ewm_ret_hl_{half_life}`` and are shifted by one
    period to preserve causality.
    """
    returns = prices.pct_change()
    features: Dict[str, pd.Series] = {}
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
    """Difference between fast and slow exponential moving averages (lagged)."""
    if fast_hl >= slow_hl:
        raise ValueError("fast_hl must be strictly less than slow_hl")
    fast_min_periods = max(int(math.ceil(fast_hl * min_period_fraction)), 1)
    slow_min_periods = max(int(math.ceil(slow_hl * min_period_fraction)), 1)
    fast_ma = prices.ewm(halflife=fast_hl, adjust=False, min_periods=fast_min_periods).mean()
    slow_ma = prices.ewm(halflife=slow_hl, adjust=False, min_periods=slow_min_periods).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        crossover = fast_ma.divide(slow_ma) - 1.0
    crossover = crossover.replace([np.inf, -np.inf], np.nan)
    return crossover.shift(1)


def _ewm_volatility(
    returns: pd.Series,
    half_life: int,
    *,
    annualize: int,
    min_period_fraction: float = 0.5,
) -> pd.Series:
    """Exponentially weighted volatility estimate (annualised)."""
    min_periods = max(int(math.ceil(half_life * min_period_fraction)), 1)
    ewm_std = returns.ewm(halflife=half_life, adjust=False, min_periods=min_periods).std()
    return ewm_std.shift(1) * math.sqrt(annualize)


def _yang_zhang_volatility(
    frame: pd.DataFrame,
    window: int,
    *,
    annualize: int,
) -> pd.Series:
    """Compute Yang–Zhang volatility using OHLC data and annualise it."""
    if window < 2:
        raise ValueError("Yang–Zhang estimator requires a window of at least 2 observations")
    log_open = np.log(frame["o_ticker"])
    log_high = np.log(frame["h_ticker"])
    log_low = np.log(frame["l_ticker"])
    log_close = np.log(frame["c_ticker"])
    log_prev_close = log_close.shift(1)
    log_prev_open = log_open.shift(1)
    close_vol = (log_close - log_prev_close) ** 2
    open_vol = (log_open - log_prev_close) ** 2
    rs = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (log_low - log_open)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    close_var = close_vol.rolling(window=window, min_periods=window).sum() * (1.0 / (window - 1))
    open_var = open_vol.rolling(window=window, min_periods=window).sum() * (1.0 / (window - 1))
    rs_var = rs.rolling(window=window, min_periods=window).sum() * (1.0 / (window - 1))
    yang_zhang_var = open_var + k * close_var + (1.0 - k) * rs_var
    yang_zhang_var = yang_zhang_var.clip(lower=0.0)
    return np.sqrt(yang_zhang_var).shift(1) * math.sqrt(annualize)


def _vol_overlay_leverage(
    returns: pd.Series,
    *,
    halflife: int,
    target_vol: float,
    cap: float,
    periods_per_year: int,
) -> pd.Series:
    """Compute volatility-managed leverage series for portfolio returns."""
    if halflife <= 0:
        raise ValueError("halflife must be positive for the volatility overlay")
    min_periods = max(int(math.ceil(halflife * 0.5)), 1)
    ewm_std = returns.ewm(halflife=halflife, adjust=False, min_periods=min_periods).std()
    annualised = ewm_std.shift(1) * math.sqrt(periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = target_vol / annualised
    leverage = leverage.clip(lower=0.0)
    leverage = leverage.replace([np.inf, -np.inf], np.nan)
    leverage = leverage.fillna(0.0)
    return leverage.clip(upper=cap)


def compute_performance_metrics(returns: pd.Series, *, periods_per_year: float) -> Dict[str, float]:
    """Compute common performance statistics for a return series."""
    cleaned = returns.dropna()
    if cleaned.empty:
        return {
            "cumulative_return": float("nan"),
            "annualized_return": float("nan"),
            "sortino_ratio": float("nan"),
            "omega_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    cumulative = (1.0 + cleaned).prod() - 1.0
    periods = len(cleaned)
    annualized = (1.0 + cleaned).prod() ** (periods_per_year / periods) - 1.0 if periods else float("nan")

    downside = np.minimum(cleaned, 0.0)
    downside_std = math.sqrt(np.mean(np.square(downside))) if len(downside) else 0.0
    sortino = float("nan")
    if downside_std > 0:
        sortino = cleaned.mean() / downside_std * math.sqrt(periods_per_year)

    gains = np.where(cleaned > 0.0, cleaned, 0.0)
    losses = np.where(cleaned < 0.0, -cleaned, 0.0)
    omega = float("nan")
    loss_sum = losses.sum()
    if loss_sum > 0:
        omega = gains.sum() / loss_sum

    nav = (1.0 + cleaned).cumprod()
    running_max = nav.cummax()
    drawdowns = nav.divide(running_max) - 1.0
    max_drawdown = drawdowns.min()

    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "sortino_ratio": sortino,
        "omega_ratio": omega,
        "max_drawdown": max_drawdown,
    }


def _load_colab_user_secrets() -> None:
    """Populate environment variables from Colab user secrets when available."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    secret_names = {"POLYGON_API_KEY": "POLYGON_API_KEY",
                    "ALPACA_API_KEY": "ALPACA_API_KEY",
                    "ALPACA_SECRET_KEY": "ALPACA_SECRET_KEY"}
    for env_name, secret_name in secret_names.items():
        if os.getenv(env_name):
            continue
        try:
            value = userdata.get(secret_name)
        except Exception:
            continue
        if value:
            os.environ[env_name] = value


_load_colab_user_secrets()

# ---------------------- CLI (extended with entropy) ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the time-series momentum backtest")
    parser.add_argument("--provider", choices=["polygon", "alpaca"], default="polygon", help="Historical data provider")
    parser.add_argument("--polygon-api-key", help="Polygon.io API key (defaults to POLYGON_API_KEY env var)")
    parser.add_argument("--alpaca-api-key", help="Alpaca API key (defaults to ALPACA_API_KEY env var)")
    parser.add_argument("--alpaca-secret-key", help="Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)")
    parser.add_argument("--alpaca-feed", default="us", help="Alpaca data feed (us or sip)")
    parser.add_argument("--cache-dir", default="data-cache", help="Directory used to persist downloaded bars")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached data and redownload from the API")
    parser.add_argument("--prefetch", action="store_true", help="Download all ticker histories up-front")
    parser.add_argument("--max-workers", type=int, default=4, help="Workers for optional prefetching")
    parser.add_argument("--lookback-window-days", type=int, default=425, help="Lookback window used for features")
    parser.add_argument("--rebalance-frequency-days", type=int, default=30, help="Gap between rebalances")
    parser.add_argument("--forward-buffer-days", type=int, default=30, help="Forward buffer for return calculation")
    parser.add_argument("--calendar", default="NYSE", help="Market calendar to use for trading dates")
    parser.add_argument("--start-date", default="2019-01-01", help="Backtest start date for calendar generation")
    parser.add_argument("--output-dir", default="backtest-output", help="Directory where reports and plots are written")
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false", help="Skip displaying the performance figure (still saved)")
    parser.set_defaults(show_plot=True)
    parser.add_argument(
        "--database-url",
        default="sqlite:///universe.db",
        help="SQLAlchemy URL for the universe database (defaults to local SQLite)",
    )
    parser.add_argument("--universe-table", default="historical_liquid_tickers_polygon",
                        help="Table containing the point-in-time universe")
    parser.add_argument("--universe-csv", default=None, help="CSV file containing the point-in-time universe (overrides DB)")
    parser.add_argument("--annualization-factor", type=int, default=252, help="Trading periods for annualising volatility")
    parser.add_argument("--momentum-type", choices=["classic", "ewm"], default="classic",
                        help="Momentum definition: classic beta*Sharpe or EWMA-based")
    parser.add_argument("--ewm-half-lives", default="63,126,252", help="Comma-separated half-lives (days) for EWMA momentum")
    parser.add_argument("--ema-crossover", action="store_true", help="Augment EWMA momentum with an EMA crossover signal")
    parser.add_argument("--ema-fast-hl", type=int, default=32, help="Half-life for the fast EMA when --ema-crossover is enabled")
    parser.add_argument("--ema-slow-hl", type=int, default=128, help="Half-life for the slow EMA when --ema-crossover is enabled")
    parser.add_argument("--vol-estimator", choices=["none", "ewm", "yang-zhang"], default="none",
                        help="Volatility estimator to use for risk-adjusting momentum")
    parser.add_argument("--vol-ewm-halflife", type=int, default=20, help="Half-life for the EWM volatility estimator")
    parser.add_argument("--vol-window", type=int, default=20, help="Window length for the Yang–Zhang volatility estimator")
    parser.add_argument("--vol-overlay", action="store_true", help="Apply a volatility-managed overlay to the long-short returns")
    parser.add_argument("--vol-overlay-halflife", type=int, default=63, help="Half-life (rebalances) for overlay volatility estimate")
    parser.add_argument("--vol-overlay-target", type=float, default=0.10, help="Annualised volatility target for overlay")
    parser.add_argument("--vol-overlay-cap", type=float, default=5.0, help="Max leverage multiple applied by overlay")

    # ---------- NEW: entropy switches ----------
    parser.add_argument("--entropy-mode", choices=["off", "pe", "se", "both", "peer", "all"], default="off",
                        help="Which entropy metrics to compute/use")
    parser.add_argument("--entropy-role", choices=["feature", "filter", "standalone", "hybrid"], default="feature",
                        help="How entropy influences ranking/gating")
    parser.add_argument("--pe-window", type=int, default=126, help="Window for permutation entropy")
    parser.add_argument("--pe-m", type=int, default=4, help="Embedding dimension m for permutation entropy")
    parser.add_argument("--pe-tau", type=int, default=1, help="Delay τ for permutation entropy")
    parser.add_argument("--se-window", type=int, default=128, help="Window for spectral entropy")
    parser.add_argument("--se-update-freq", type=int, default=5, help="Spectral entropy recompute frequency")
    parser.add_argument("--cs-bins", type=int, default=21, help="Bins for cross-sectional peer surprisal/JSD")
    parser.add_argument("--entropy-filter-quantile", type=float, default=0.8, help="Quantile for entropy gating")
    parser.add_argument("--entropy-filter-side", choices=["low", "high"], default="low",
                        help="'low' keeps low-entropy (more structured) names; 'high' keeps high-entropy")
    return parser.parse_args()


def build_provider(args: argparse.Namespace):
    if args.provider == "polygon":
        api_key = args.polygon_api_key or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("Set --polygon-api-key or POLYGON_API_KEY before running the backtest")
        return PolygonAggsClient(api_key)
    api_key = args.alpaca_api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = args.alpaca_secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set Alpaca credentials via flags or environment variables")
    return AlpacaBarsClient(api_key, secret_key, data_feed=args.alpaca_feed)


def load_universe(args: argparse.Namespace) -> pd.DataFrame:
    if args.universe_csv:
        csv_path = Path(args.universe_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Universe CSV {csv_path} does not exist")
        universe = pd.read_csv(csv_path)
    else:
        try:
            engine = sqlalchemy.create_engine(args.database_url)
            with engine.connect() as connection:
                universe = pd.read_sql(args.universe_table, con=connection)
        except sqlalchemy.exc.SQLAlchemyError as exc:
            raise RuntimeError(
                "Failed to load universe from the database. "
                "Pass --database-url/--universe-table or provide --universe-csv."
            ) from exc

    if not isinstance(universe, pd.DataFrame):
        raise TypeError("Universe data must be a pandas DataFrame")

    if universe.columns.size:
        first_col = str(universe.columns[0]).strip().lower()
        if first_col in {"", "index", "unnamed: 0"}:
            universe = universe.drop(columns=universe.columns[0])

    required_columns = {"date", "ticker"}
    if not required_columns.issubset(universe.columns):
        missing = ", ".join(sorted(required_columns - set(universe.columns)))
        raise ValueError(f"Universe data is missing required columns: {missing}")

    if "avg_days_between" in universe.columns:
        avg_days = pd.to_numeric(universe["avg_days_between"], errors="coerce")
        universe = universe.loc[avg_days < 9].copy()

    universe = universe.drop_duplicates(subset=["date", "ticker"])
    universe["date"] = pd.to_datetime(universe["date"])
    return universe.sort_values("date")


def get_trading_dates(args: argparse.Namespace) -> np.ndarray:
    calendar = get_calendar(args.calendar)
    end_date = datetime.today() - timedelta(days=1)
    trading_dates = calendar.schedule(start_date=args.start_date, end_date=end_date).index
    return trading_dates.strftime("%Y-%m-%d").values


def cache_key(provider_name: str, ticker: str, start: str, end: str) -> str:
    safe_ticker = ticker.replace("/", "-")
    return f"{provider_name}_{safe_ticker}_{start}_{end}.pkl"


def history_loader(
    provider_name: str,
    provider,
    cache_dir: Path,
    start: str,
    end: str,
    force_refresh: bool,
) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}

    def load_history(ticker: str) -> pd.DataFrame:
        if ticker in cache and not force_refresh:
            return cache[ticker]
        cache_path = cache_dir / cache_key(provider_name, ticker, start, end)
        if cache_path.exists() and not force_refresh:
            cache[ticker] = pd.read_pickle(cache_path)
            return cache[ticker]
        data = provider.get_daily_bars(ticker, start, end)
        if not data.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_pickle(cache_path)
        cache[ticker] = data
        return data

    return cache, load_history


def fetch_benchmark(
    provider,
    cache_dir: Path,
    provider_name: str,
    force_refresh: bool,
    start: str,
    end: str,
) -> pd.DataFrame:
    cache_path = cache_dir / cache_key(provider_name, "SPY", start, end)
    if cache_path.exists() and not force_refresh:
        return pd.read_pickle(cache_path)
    benchmark = provider.get_daily_bars("SPY", start, end)
    if not benchmark.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_pickle(cache_path)
    return benchmark


# ---------------------- main ----------------------

def main() -> None:
    args = parse_args()
    args.ewm_half_lives = _parse_positive_int_list(args.ewm_half_lives)
    if args.ema_crossover and args.ema_fast_hl >= args.ema_slow_hl:
        raise ValueError("--ema-fast-hl must be smaller than --ema-slow-hl for the crossover signal")
    if args.vol_overlay and args.vol_overlay_target <= 0:
        raise ValueError("--vol-overlay-target must be positive")

    provider = build_provider(args)
    try:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trading_dates = get_trading_dates(args)
        if len(trading_dates) == 0:
            raise ValueError("No trading dates available for the selected calendar/time range")

        universe = load_universe(args)
        months_index = pd.Index(universe["date"].drop_duplicates().sort_values())
        if len(months_index) < 2:
            raise ValueError("Not enough history in the universe to run the backtest")

        analysis_months = months_index[:-1]
        last_month = months_index[-1]

        lookback_days = int(args.lookback_window_days)
        forward_buffer_days = int(args.forward_buffer_days)

        global_start_dt = analysis_months[0] - timedelta(days=lookback_days)
        global_end_dt = last_month + timedelta(days=forward_buffer_days)
        global_start = global_start_dt.strftime("%Y-%m-%d")
        global_end = global_end_dt.strftime("%Y-%m-%d")

        provider_name = args.provider
        cache, load_history = history_loader(provider_name, provider, cache_dir, global_start, global_end, args.force_refresh)

        benchmark_start = min(pd.Timestamp("2017-01-01"), global_start_dt).strftime("%Y-%m-%d")
        benchmark_data = fetch_benchmark(provider, cache_dir, provider_name, args.force_refresh, benchmark_start, global_end)
        if benchmark_data.empty:
            raise ValueError("Benchmark data is empty; verify API credentials and connectivity")

        benchmark_data = benchmark_data.drop_duplicates(subset="date")

        all_tickers = universe["ticker"].drop_duplicates().values
        if args.prefetch:
            print(f"Prefetching {len(all_tickers)} tickers with {args.max_workers} workers...")

            def _prefetch(symbol: str) -> None:
                try:
                    load_history(symbol)
                except Exception as exc:  # pragma: no cover - diagnostic only
                    print(f"Failed to prefetch {symbol}: {exc}")

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                list(executor.map(_prefetch, all_tickers))

        FEATURE_WINDOW = 252
        approx_skip_days = int(args.rebalance_frequency_days)
        skip_trading_days = max(int(round(approx_skip_days * 252 / 365)), 1)
        SKIP_BDAYS = skip_trading_days

        def _shift_by_bdays(trading_days: np.ndarray, dates: pd.Index, n_bdays: int) -> pd.Index:
            trading_index = pd.to_datetime(pd.Index(trading_days))
            target_dates = pd.to_datetime(dates)
            insertion_points = np.searchsorted(trading_index.values, target_dates.values, side="left")
            anchor_idx = np.clip(insertion_points - n_bdays, 0, len(trading_index) - 1)
            return pd.Index(trading_index[anchor_idx])

        feature_window = FEATURE_WINDOW

        benchmark_data = benchmark_data.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
        benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
        benchmark_data = benchmark_data.sort_values("date").reset_index(drop=True)
        benchmark_pct_change = benchmark_data["c"].pct_change()
        if not benchmark_pct_change.empty:
            benchmark_pct_change.iloc[0] = 0.0
        benchmark_data["benchmark_pct_change"] = benchmark_pct_change * 100

        universe_by_month = {
            month: set(universe.loc[universe["date"] == month, "ticker"].drop_duplicates().values)
            for month in analysis_months
        }

        analysis_months_index = analysis_months
        exit_months_index = months_index[1:]
        feature_anchor_index = _shift_by_bdays(trading_dates, analysis_months_index, SKIP_BDAYS)
        if len(feature_anchor_index) != len(analysis_months_index):
            raise ValueError("Feature anchor index misaligned with analysis months")
        analysis_months_array = analysis_months_index.to_numpy()
        exit_months_array = exit_months_index.to_numpy()

        all_monthly_results = []

        # Track t-1 returns for cross-sectional peer metrics at anchor dates (computed after loop)
        returns_by_ticker: Dict[str, pd.Series] = {}
        close_prices_by_ticker: Dict[str, pd.Series] = {}

        def _format_timedelta(delta: timedelta) -> str:
            total_seconds = int(delta.total_seconds())
            if total_seconds <= 0:
                return "0s"
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            if minutes:
                parts.append(f"{minutes}m")
            if seconds or not parts:
                parts.append(f"{seconds}s")
            return " ".join(parts)

        ticker_count = len(all_tickers)
        completed = 0
        durations: list[float] = []
        loop_start_time = datetime.now()

        for ticker in all_tickers:
            try:
                ticker_start_time = datetime.now()
                print(f"[{completed + 1}/{ticker_count}] Downloading history for {ticker}...", flush=True)
                history = load_history(ticker)
            except Exception as err:
                print(f"{ticker} download failed: {err}")
                continue

            if history.empty:
                completed += 1
                durations.append((datetime.now() - ticker_start_time).total_seconds())
                if durations:
                    average_seconds = sum(durations) / len(durations)
                    remaining = ticker_count - completed
                    eta = datetime.now() + timedelta(seconds=average_seconds * remaining)
                    remaining_td = timedelta(seconds=int(average_seconds * remaining))
                    print(f"    No data for {ticker}. Progress: {completed}/{ticker_count}. "
                          f"ETA {eta:%Y-%m-%d %H:%M:%S} ({_format_timedelta(remaining_td)} remaining)",
                          flush=True)
                continue

            history = history.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
            history["date"] = pd.to_datetime(history["date"])

            history_columns = ["date", "o", "h", "l", "c"]
            missing_cols = sorted(set(history_columns) - set(history.columns))
            if missing_cols:
                raise ValueError(f"History for {ticker} is missing required columns: {missing_cols}")

            close_series = history.set_index("date")["c"].sort_index()
            ohlc_frame = history.set_index("date")[["o", "h", "l", "c"]].sort_index()
            returns_by_ticker[ticker] = close_series.pct_change()

            merged = pd.merge(
                ohlc_frame.reset_index().rename(
                    columns={"o": "o_ticker", "h": "h_ticker", "l": "l_ticker", "c": "c_ticker"}
                ),
                benchmark_data[["date", "c", "benchmark_pct_change"]].rename(columns={"c": "c_benchmark"}),
                on="date",
                how="inner",
                sort=True,
            )

            if len(merged) < feature_window:
                completed += 1
                durations.append((datetime.now() - ticker_start_time).total_seconds())
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
            beta = rolling_cov.divide(benchmark_var).replace([np.inf, -np.inf], np.nan)

            total_return = merged["c_ticker"].pct_change(periods=feature_window - 1) * 100
            rolling_std = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).std()
            rolling_std = rolling_std * np.sqrt(args.annualization_factor)
            rolling_std = rolling_std.replace(0, np.nan)
            sharpe = total_return.divide(rolling_std)
            sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

            metrics = (
                pd.DataFrame({"date": merged["date"], "beta": beta, "sharpe": sharpe, "12-1_return": total_return})
                .set_index("date").sort_index()
            )

            # --- Momentum features ---
            mom_components: List[pd.Series] = []
            if args.momentum_type == "classic":
                classic_score = metrics["beta"] * metrics["sharpe"]
                metrics["classic_mom"] = classic_score
                mom_components.append(metrics["classic_mom"])
            else:
                ewm_features = _compute_ewm_momentum(close_series, args.ewm_half_lives)
                for column in ewm_features.columns:
                    metrics[column] = ewm_features[column].reindex(metrics.index)
                    mom_components.append(metrics[column])
                if args.ema_crossover:
                    ema_signal = _ema_crossover_signal(close_series, args.ema_fast_hl, args.ema_slow_hl)
                    metrics["ema_crossover"] = ema_signal.reindex(metrics.index)
                    mom_components.append(metrics["ema_crossover"])

            if mom_components:
                mom_matrix = pd.concat(mom_components, axis=1)
                composite = mom_matrix.mean(axis=1)
            else:
                composite = pd.Series(index=metrics.index, dtype=float)

            # --- Optional volatility estimates (risk adjust) ---
            volatility: Optional[pd.Series] = None
            if args.vol_estimator != "none":
                try:
                    if args.vol_estimator == "ewm":
                        volatility = _ewm_volatility(close_series.pct_change(),
                                                     args.vol_ewm_halflife,
                                                     annualize=args.annualization_factor)
                    else:
                        volatility = _yang_zhang_volatility(
                            ohlc_frame.rename(columns={"o": "o_ticker", "h": "h_ticker", "l": "l_ticker", "c": "c_ticker"}),
                            args.vol_window,
                            annualize=args.annualization_factor,
                        )
                except Exception as vol_err:
                    raise RuntimeError(f"Failed to compute volatility for {ticker}: {vol_err}") from vol_err

                if volatility is not None:
                    metrics["volatility"] = volatility.reindex(metrics.index)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        composite = composite.divide(metrics["volatility"])
                    composite = composite.replace([np.inf, -np.inf], np.nan)

            metrics["mom_score"] = composite

            # Align to feature anchors (strictly <= anchor date)
            metrics = metrics.reindex(feature_anchor_index, method="ffill")

            price_series = history.set_index("date")["c"].sort_index()
            price_dates = price_series.index.to_numpy()
            price_values = price_series.to_numpy()
            if len(price_dates) == 0:
                completed += 1
                durations.append((datetime.now() - ticker_start_time).total_seconds())
                continue

            entry_idx = np.searchsorted(price_dates, analysis_months_array, side="left")
            exit_idx = np.searchsorted(price_dates, exit_months_array, side="right") - 1

            valid_mask = (
                (entry_idx < len(price_dates))
                & (exit_idx >= 0)
                & (exit_idx < len(price_dates))
                & (exit_idx > entry_idx)
            )

            forward_returns = np.full(len(analysis_months_array), np.nan)
            if valid_mask.any():
                valid_entry_idx = entry_idx[valid_mask]
                valid_exit_idx = exit_idx[valid_mask]
                entry_subset = price_values[valid_entry_idx]
                exit_subset = price_values[valid_exit_idx]
                nonzero_entry = entry_subset != 0
                result_returns = np.full(valid_entry_idx.shape, np.nan)
                result_returns[nonzero_entry] = (
                    (exit_subset[nonzero_entry] - entry_subset[nonzero_entry]) / entry_subset[nonzero_entry]
                ) * 100
                forward_returns[valid_mask] = result_returns

            monthly_results = pd.DataFrame(
                {
                    "entry_date": analysis_months_array,
                    "exit_date": exit_months_array,
                    "ticker": ticker,
                    "beta": metrics["beta"].to_numpy(),
                    "sharpe": metrics["sharpe"].to_numpy(),
                    "12-1_return": metrics["12-1_return"].to_numpy(),
                    "mom_score": metrics["mom_score"].to_numpy(),
                    "forward_returns": forward_returns,
                    "anchor_date": pd.to_datetime(feature_anchor_index).strftime("%Y-%m-%d").to_numpy(),
                }
            )

            if "volatility" in metrics.columns:
                monthly_results["volatility"] = metrics["volatility"].to_numpy()
            if "classic_mom" in metrics.columns:
                monthly_results["classic_mom"] = metrics["classic_mom"].to_numpy()
            for column in metrics.columns:
                if column.startswith("ewm_ret_hl_"):
                    monthly_results[column] = metrics[column].to_numpy()
            if "ema_crossover" in metrics.columns:
                monthly_results["ema_crossover"] = metrics["ema_crossover"].to_numpy()
            in_universe = np.array([ticker in universe_by_month.get(month, set()) for month in analysis_months])
            monthly_results = monthly_results[in_universe]
            monthly_results = monthly_results.dropna(
                subset=["beta", "sharpe", "12-1_return", "mom_score", "forward_returns"]
            )

            if not monthly_results.empty:
                monthly_results["entry_date"] = pd.to_datetime(monthly_results["entry_date"]).dt.strftime("%Y-%m-%d")
                monthly_results["exit_date"] = pd.to_datetime(monthly_results["exit_date"]).dt.strftime("%Y-%m-%d")
                close_prices_by_ticker[ticker] = close_series
                all_monthly_results.append(monthly_results)

            completed += 1
            ticker_duration = (datetime.now() - ticker_start_time).total_seconds()
            durations.append(ticker_duration)
            average_seconds = sum(durations) / len(durations)
            remaining = ticker_count - completed
            eta = datetime.now() + timedelta(seconds=average_seconds * remaining)
            remaining_td = timedelta(seconds=int(average_seconds * remaining))
            elapsed_td = datetime.now() - loop_start_time
            print(f"    Finished {ticker} in {ticker_duration:.2f}s. Progress: {completed}/{ticker_count}. "
                  f"Elapsed {_format_timedelta(elapsed_td)}. ETA {eta:%Y-%m-%d %H:%M:%S} "
                  f"({_format_timedelta(remaining_td)} remaining)",
                  flush=True)

        if not all_monthly_results:
            raise ValueError("No data collected for any month; confirm ticker universe and data availability")

        full_dataset = pd.concat(all_monthly_results, ignore_index=True)
        full_dataset = full_dataset.sort_values(["entry_date", "mom_score"], ascending=[True, False])

        # Compute per-ticker entropy measures once across the full panel (if requested)
        if args.entropy_mode in {"pe", "se", "both", "all"} and close_prices_by_ticker:
            print("Computing entropy features across tickers...", flush=True)

            pe_name = f"perm_entropy_w{args.pe_window}_m{args.pe_m}_tau{args.pe_tau}"
            se_name = f"spectral_entropy_w{args.se_window}"

            entropy_frames: List[pd.DataFrame] = []
            for tk, prices in close_prices_by_ticker.items():
                frame = pd.DataFrame({"date": prices.index, "closeadj": prices.values, "ticker": tk})
                entropy_frames.append(frame)

            if entropy_frames:
                entropy_panel = pd.concat(entropy_frames, ignore_index=True)
                entropy_panel["date"] = pd.to_datetime(entropy_panel["date"])
                entropy_panel = entropy_panel.set_index(["date", "ticker"]).sort_index()

                ent_features = rmf.add_entropy_info_features(
                    entropy_panel,
                    price_col="closeadj",
                    use_lagged_data=True,
                    pe_window=args.pe_window,
                    pe_m=args.pe_m,
                    pe_tau=args.pe_tau,
                    se_window=args.se_window,
                    se_update_freq=args.se_update_freq,
                    cs_bins=args.cs_bins,
                    pca_window=None,
                    verbose=False,
                ).sort_index()

                components: List[pd.Series] = []
                if args.entropy_mode in {"pe", "both", "all"} and pe_name in ent_features.columns:
                    components.append(1.0 - ent_features[pe_name])
                if args.entropy_mode in {"se", "both", "all"} and se_name in ent_features.columns:
                    components.append(1.0 - ent_features[se_name])
                if components:
                    ent_quality = pd.concat(components, axis=1).mean(axis=1).clip(0.0, 1.0)
                    ent_features["entropy_quality"] = ent_quality
                    ent_features["entropy_score"] = ent_quality

                merge_columns = [col for col in [pe_name, se_name, "entropy_quality", "entropy_score"] if col in ent_features.columns]
                if merge_columns:
                    anchor_dates = pd.Index(
                        pd.to_datetime(full_dataset["anchor_date"].dropna().unique()),
                        name="date",
                    ).sort_values()
                    entropy_merge_frames: List[pd.DataFrame] = []
                    for tk in ent_features.index.get_level_values("ticker").unique():
                        ticker_slice = ent_features.xs(tk, level="ticker")
                        ticker_slice = ticker_slice.reindex(anchor_dates)
                        subset = ticker_slice[merge_columns]
                        subset = subset.reset_index().rename(columns={"date": "anchor_dt"})
                        subset["anchor_date"] = subset["anchor_dt"].dt.strftime("%Y-%m-%d")
                        subset["ticker"] = tk
                        entropy_merge_frames.append(subset.drop(columns=["anchor_dt"]))

                    if entropy_merge_frames:
                        entropy_aug = pd.concat(entropy_merge_frames, ignore_index=True)
                        full_dataset = full_dataset.merge(
                            entropy_aug,
                            on=["anchor_date", "ticker"],
                            how="left",
                        )

                        if args.entropy_role == "standalone" and "entropy_score" in full_dataset.columns:
                            full_dataset["mom_score"] = np.where(
                                full_dataset["entropy_score"].notna(),
                                full_dataset["entropy_score"],
                                full_dataset["mom_score"],
                            )
                        elif args.entropy_role == "hybrid" and "entropy_quality" in full_dataset.columns:
                            full_dataset["mom_score"] = full_dataset["mom_score"] * full_dataset["entropy_quality"].fillna(1.0)

                print("    Entropy feature computation complete.", flush=True)

        # ---------- NEW: optional cross-sectional peer metrics (at anchor dates, from t-1 returns) ----------
        if args.entropy_mode in {"peer", "all"}:
            # Build panel of r[t-1] at anchor dates for all tickers
            anchor_dates = pd.to_datetime(full_dataset["anchor_date"].drop_duplicates().sort_values())
            panel = pd.DataFrame(index=anchor_dates)
            for tk, ret in returns_by_ticker.items():
                rlag = ret.shift(1)
                panel[tk] = rlag.reindex(anchor_dates).values
            # MultiIndex series for rmf peer functions
            rlag_series = panel.stack(dropna=False)
            rlag_series.index = rlag_series.index.set_names(["date", "ticker"])
            rlag_series = rlag_series.astype(float)
            # Compute peer surprisal & JSD (causal at t-1)
            cs_surpr, _p, _ = rmf._cross_sectional_surprisal_from_bins(rlag_series.dropna(), n_bins=args.cs_bins)
            cs_jsd = rmf._peer_jsd_pointmass(rlag_series.dropna(), n_bins=args.cs_bins)
            peer_df = pd.concat([cs_surpr, cs_jsd], axis=1)
            peer_df.columns = ["peer_surprisal", "peer_jsd"]
            # Robust per-date z for JSD
            peer_df = peer_df.to_frame() if isinstance(peer_df, pd.Series) else peer_df
            peer_df["peer_jsd_z"] = peer_df.groupby(level="date")["peer_jsd"].apply(
                lambda s: ((s - s.median()) / ((s.quantile(0.75) - s.quantile(0.25)) / 1.349 + 1e-12)).clip(-8, 8)
            )
            pf = peer_df.reset_index()
            pf["entry_date"] = pf["date"].dt.strftime("%Y-%m-%d")
            full_dataset = full_dataset.merge(
                pf[["entry_date", "ticker", "peer_surprisal", "peer_jsd", "peer_jsd_z"]],
                on=["entry_date", "ticker"],
                how="left",
            )

        # ---------- NEW: entropy gating (filter role) ----------
        if args.entropy_role == "filter":
            if args.entropy_mode in {"peer", "all"} and "peer_jsd" in full_dataset.columns:
                # By default, drop the most idiosyncratic extreme (high JSD) names
                q = full_dataset.groupby("entry_date")["peer_jsd"].transform(
                    lambda x: x.quantile(args.entropy_filter_quantile)
                )
                if args.entropy_filter_side == "low":
                    mask = full_dataset["peer_jsd"] <= q
                else:
                    mask = full_dataset["peer_jsd"] >= q
                full_dataset = full_dataset[mask]
            elif "entropy_quality" in full_dataset.columns:
                # Keep low-entropy (high quality) names by default
                q = full_dataset.groupby("entry_date")["entropy_quality"].transform(
                    lambda x: x.quantile(args.entropy_filter_quantile)
                )
                if args.entropy_filter_side == "low":
                    mask = full_dataset["entropy_quality"] >= q
                else:
                    # keep high-entropy (low-quality) names if user requests
                    q_low = full_dataset.groupby("entry_date")["entropy_quality"].transform(
                        lambda x: x.quantile(1.0 - args.entropy_filter_quantile)
                    )
                    mask = full_dataset["entropy_quality"] <= q_low
                full_dataset = full_dataset[mask]

        # Choose score column (standalone entropy vs momentum)
        score_col = "mom_score"
        if args.entropy_role == "standalone" and ("entropy_score" in full_dataset.columns):
            score_col = "entropy_score"

        top_decile_dataset = (
            full_dataset.sort_values(score_col, ascending=False)
            .groupby("entry_date", group_keys=False)
            .head(10)
        )
        bot_decile_dataset = (
            full_dataset.sort_values(score_col, ascending=True)
            .groupby("entry_date", group_keys=False)
            .head(10)
        )

        covered_dates = full_dataset["entry_date"].drop_duplicates().values

        trades = []
        for covered_date in covered_dates:
            long_uni = top_decile_dataset[top_decile_dataset["entry_date"] == covered_date]
            short_uni = bot_decile_dataset[bot_decile_dataset["entry_date"] == covered_date]
            if long_uni.empty or short_uni.empty:
                continue
            long_mean = long_uni["forward_returns"].mean()
            short_mean = short_uni["forward_returns"].mean()
            if pd.isna(long_mean) or pd.isna(short_mean):
                continue
            trade = pd.DataFrame(
                [
                    {
                        "date": covered_date,
                        "long_return": long_mean / 100.0,
                        "short_return": -short_mean / 100.0,
                    }
                ]
            )
            trades.append(trade)

        if not trades:
            raise ValueError("No valid trades generated; investigate data quality or parameters")

        all_trades = pd.concat(trades, ignore_index=True)
        if all_trades.empty:
            raise ValueError("Trades table unexpectedly empty after concatenation")

        GROSS_LONG = 1.0
        GROSS_SHORT = 1.0
        PER_SIDE_BPS = 2.0

        def _basket_turnover_per_month(baskets: pd.DataFrame) -> pd.Series:
            if baskets.empty:
                return pd.Series(dtype=float)
            grouped = baskets.groupby("entry_date")["ticker"].apply(lambda s: set(s)).sort_index()
            turnover_values: list[float] = []
            previous = None
            for current in grouped.index:
                current_set = grouped.loc[current]
                if previous is None:
                    turnover_values.append(1.0)
                else:
                    previous_set = grouped.loc[previous]
                    union_size = max(len(current_set | previous_set), 1)
                    overlap = len(current_set & previous_set) / union_size
                    turnover_values.append(1.0 - overlap)
                previous = current
            return pd.Series(turnover_values, index=pd.to_datetime(grouped.index))

        long_turnover = _basket_turnover_per_month(top_decile_dataset[["entry_date", "ticker"]])
        short_turnover = _basket_turnover_per_month(bot_decile_dataset[["entry_date", "ticker"]])

        trades_dates = pd.to_datetime(all_trades["date"])
        turnover_frame = pd.DataFrame(index=trades_dates)
        turnover_frame.index.name = "date"
        turnover_frame["long_turnover"] = long_turnover.reindex(turnover_frame.index).fillna(1.0)
        turnover_frame["short_turnover"] = short_turnover.reindex(turnover_frame.index).fillna(1.0)

        long_returns = all_trades["long_return"].to_numpy() * GROSS_LONG
        short_returns = all_trades["short_return"].to_numpy() * GROSS_SHORT
        ls_returns = long_returns + short_returns

        cost_decimals = (
            (PER_SIDE_BPS / 1e4)
            * (
                turnover_frame["long_turnover"].to_numpy() * GROSS_LONG
                + turnover_frame["short_turnover"].to_numpy() * GROSS_SHORT
            )
        )
        ls_returns_net = ls_returns - cost_decimals

        overlay_returns = None
        if args.vol_overlay:
            periods_per_year = max(int(round(args.annualization_factor / max(SKIP_BDAYS, 1))), 1)
            leverage_series = _vol_overlay_leverage(
                pd.Series(ls_returns, index=trades_dates, dtype=float),
                halflife=args.vol_overlay_halflife,
                target_vol=args.vol_overlay_target,
                cap=args.vol_overlay_cap,
                periods_per_year=periods_per_year,
            ).reindex(trades_dates).fillna(0.0)
            overlay_returns = ls_returns * leverage_series.to_numpy() - cost_decimals * leverage_series.to_numpy()
            all_trades["vol_overlay_leverage"] = leverage_series.to_numpy()

        all_trades["long_nav"] = np.cumprod(1.0 + long_returns)
        all_trades["short_nav"] = np.cumprod(1.0 + short_returns)
        all_trades["ls_nav"] = np.cumprod(1.0 + ls_returns)
        all_trades["ls_nav_net"] = np.cumprod(1.0 + ls_returns_net)
        if overlay_returns is not None:
            all_trades["ls_nav_vol_managed"] = np.cumprod(1.0 + overlay_returns)

        all_trades["ls_return"] = ls_returns
        all_trades["ls_return_net"] = ls_returns_net
        if overlay_returns is not None:
            all_trades["ls_return_vol_managed"] = overlay_returns

        all_trades["long_pct"] = all_trades["long_return"] * 100.0
        all_trades["short_pct"] = all_trades["short_return"] * 100.0
        all_trades["ls_pct"] = all_trades["ls_return"] * 100.0
        all_trades["ls_pct_net"] = all_trades["ls_return_net"] * 100.0
        if overlay_returns is not None:
            all_trades["ls_pct_vol_managed"] = all_trades["ls_return_vol_managed"] * 100.0

        # Backwards-compatible column names for exported CSV (percentage values)
        all_trades["long"] = all_trades["long_pct"]
        all_trades["short"] = all_trades["short_pct"]

        all_trades["long_cum_%"] = (all_trades["long_nav"] - 1.0) * 100.0
        all_trades["short_cum_%"] = (all_trades["short_nav"] - 1.0) * 100.0
        all_trades["ls_cum_%"] = (all_trades["ls_nav"] - 1.0) * 100.0
        all_trades["ls_cum_%_net"] = (all_trades["ls_nav_net"] - 1.0) * 100.0
        if overlay_returns is not None:
            all_trades["ls_cum_%_vol_managed"] = (all_trades["ls_nav_vol_managed"] - 1.0) * 100.0

        # Performance diagnostics
        periods_per_year_float = float(args.annualization_factor) / max(SKIP_BDAYS, 1)
        ls_net_series = pd.Series(ls_returns_net, index=trades_dates, dtype=float)
        metrics_records = []

        def _format_metric(name: str, value: float) -> str:
            if pd.isna(value):
                return "n/a"
            if "ratio" in name:
                return f"{value:.4f}"
            return f"{value:.2%}"

        net_metrics = compute_performance_metrics(ls_net_series, periods_per_year=periods_per_year_float)
        metrics_records.append({"series": "long_short_net", **net_metrics})
        print("\nPerformance metrics (long-short, net of costs):")
        for metric_name, metric_value in net_metrics.items():
            print(f"  {metric_name.replace('_', ' ').title()}: {_format_metric(metric_name, metric_value)}")

        if overlay_returns is not None:
            overlay_series = pd.Series(overlay_returns, index=trades_dates, dtype=float)
            overlay_metrics = compute_performance_metrics(
                overlay_series, periods_per_year=periods_per_year_float
            )
            metrics_records.append({"series": "long_short_vol_managed", **overlay_metrics})
            print("\nPerformance metrics (long-short with volatility overlay):")
            for metric_name, metric_value in overlay_metrics.items():
                print(f"  {metric_name.replace('_', ' ').title()}: {_format_metric(metric_name, metric_value)}")

        metrics_path = output_dir / "performance_metrics.csv"
        pd.DataFrame(metrics_records).to_csv(metrics_path, index=False)

        plt.figure(figsize=(10, 6), dpi=200)
        plt.xticks(rotation=45)
        plt.suptitle("Cumulative Performance")
        plt.title("Monthly Rebalancing")
        plt.plot(trades_dates, all_trades["long_cum_%"], label="Top Decile (gross)")
        plt.plot(trades_dates, all_trades["short_cum_%"], label="Bottom Decile (gross)")
        plt.plot(trades_dates, all_trades["ls_cum_%"], label="Long-Short (gross)")
        plt.plot(trades_dates, all_trades["ls_cum_%_net"], label="Long-Short (net est. costs)")
        if overlay_returns is not None:
            plt.plot(trades_dates, all_trades["ls_cum_%_vol_managed"], label="Long-Short (vol managed)")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative % Return")
        plt.tight_layout()

        plot_path = output_dir / "cumulative_performance.png"
        plt.savefig(plot_path, bbox_inches="tight")

        trades_path = output_dir / "monthly_trades.csv"
        all_trades.to_csv(trades_path, index=False)

        if args.show_plot:
            plt.show()
        plt.close()
    finally:
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    main()
