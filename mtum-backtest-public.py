# -*- coding: utf-8 -*-
"""Momentum backtest with configurable data sources and caching."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy

from pandas.tseries.offsets import BDay
from pandas_market_calendars import get_calendar

from data_providers import AlpacaBarsClient, PolygonAggsClient
from momentum_signals import (
    ema_crossover_price,
    ewm_momentum_returns,
    ewm_volatility,
    make_log_returns_from_prices,
    parse_halflife_string,
    rolling_volatility,
    sharpe_like_trend,
    trailing_return,
    vol_managed_scaler,
    yang_zhang_vol,
)


ANNUALIZATION = 252


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


_load_colab_user_secrets()


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
    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help="Skip displaying the performance figure (still saved to disk)",
    )
    parser.set_defaults(show_plot=True)
    parser.add_argument(
        "--database-url",
        default="mysql+mysqlconnector://user:pass@localhost:3306/my_database",
        help="SQLAlchemy URL for the universe database",
    )
    parser.add_argument(
        "--universe-table",
        default="historical_liquid_tickers_polygon",
        help="Table containing the point-in-time universe",
    )
    parser.add_argument(
        "--universe-csv",
        default=None,
        help="CSV file containing the point-in-time universe (overrides the database)",
    )
    parser.add_argument(
        "--momentum-style",
        choices=["classic", "ewma", "ewma_sr", "ema_crossover", "composite"],
        default="classic",
        help="Momentum signal to rank tickers. 'composite' blends EWMA Sharpe and EMA crossover",
    )
    parser.add_argument(
        "--ewm-half-lives",
        default="63,126,252",
        help="Comma separated half-lives (in trading days) for EWMA momentum components",
    )
    parser.add_argument(
        "--ema-fast-halflife",
        type=int,
        default=32,
        help="Half-life (trading days) for the fast EMA in the crossover signal",
    )
    parser.add_argument(
        "--ema-slow-halflife",
        type=int,
        default=128,
        help="Half-life (trading days) for the slow EMA in the crossover signal",
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
        help="Window length for rolling and Yang-Zhang volatility estimators",
    )
    parser.add_argument(
        "--vol-ewm-halflife",
        type=int,
        default=20,
        help="Half-life for the EWMA volatility estimator",
    )
    parser.add_argument(
        "--overlay-target-vol",
        type=float,
        default=0.15,
        help="Target annualized volatility for the optional long/short overlay",
    )
    parser.add_argument(
        "--overlay-halflife",
        type=int,
        default=63,
        help="Half-life for the volatility managed overlay (trading days)",
    )
    parser.add_argument(
        "--overlay-max-leverage",
        type=float,
        default=5.0,
        help="Maximum leverage multiple for the volatility managed overlay",
    )
    parser.add_argument(
        "--apply-vol-overlay",
        action="store_true",
        help="Apply a Moreira-Muir volatility managed overlay to the long/short leg",
    )
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
        engine = sqlalchemy.create_engine(args.database_url)
        universe = pd.read_sql(args.universe_table, con=engine)

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


def compute_volatility_series(
    returns: pd.Series,
    prices: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.Series, str]:
    """Dispatch helper for per-ticker volatility estimation."""

    if args.vol_estimator == "rolling":
        vol = rolling_volatility(returns, window=args.vol_window)
        return vol, "vol_rolling_ann"
    if args.vol_estimator == "ewm":
        vol = ewm_volatility(returns, hl=args.vol_ewm_halflife)
        return vol, "vol_ewm_ann"
    vol = yang_zhang_vol(prices, window=args.vol_window)
    return vol, "vol_yz_ann"


def compute_momentum_score(metrics: pd.DataFrame, style: str) -> pd.Series:
    """Derive the ranking signal based on the configured style."""

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

    # Fallback to NaNs with aligned index
    return pd.Series(index=metrics.index, dtype=float)


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


def main() -> None:
    args = parse_args()
    try:
        ewm_half_lives = parse_halflife_string(args.ewm_half_lives)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid --ewm-half-lives '{args.ewm_half_lives}': {exc}") from exc
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
                except Exception as exc:  # pragma: no cover - diagnostic output only
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

        def _align_frame_to_length(frame: pd.DataFrame, target_length: int) -> pd.DataFrame:
            """Ensure ``frame`` has ``target_length`` rows by padding or trimming."""

            if target_length < 0:
                raise ValueError("target_length must be non-negative")

            frame_reset = frame.reset_index(drop=True)

            if len(frame_reset) == target_length:
                return frame_reset

            if len(frame_reset) > target_length:
                return frame_reset.iloc[-target_length:].reset_index(drop=True)

            pad_length = target_length - len(frame_reset)
            if pad_length == 0:
                return frame_reset

            pad = pd.DataFrame(np.nan, index=range(pad_length), columns=frame_reset.columns)
            return pd.concat([pad, frame_reset], ignore_index=True)

        feature_window = FEATURE_WINDOW

        benchmark_data = benchmark_data.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
        benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
        benchmark_data = benchmark_data.sort_values("date").reset_index(drop=True)
        benchmark_pct_change = benchmark_data["c"].pct_change()
        if not benchmark_pct_change.empty:
            benchmark_pct_change.iloc[0] = 0.0
        benchmark_data["benchmark_pct_change"] = benchmark_pct_change * 100

        universe_by_month = {
            month: set(
                universe.loc[universe["date"] == month, "ticker"].drop_duplicates().values
            )
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
                print(
                    f"[{completed + 1}/{ticker_count}] Downloading history for {ticker}...",
                    flush=True,
                )
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
                    print(
                        f"    No data for {ticker}. Progress: {completed}/{ticker_count}. "
                        f"ETA {eta:%Y-%m-%d %H:%M:%S} ({_format_timedelta(remaining_td)} remaining)",
                        flush=True,
                    )
                continue

            history = history.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
            history["date"] = pd.to_datetime(history["date"])

            merged = pd.merge(
                history[["date", "c"]].rename(columns={"c": "c_ticker"}),
                benchmark_data[["date", "c", "benchmark_pct_change"]].rename(
                    columns={"c": "c_benchmark"}
                ),
                on="date",
                how="inner",
                sort=True,
            )

            if len(merged) < feature_window:
                continue

            merged = merged.sort_values("date").reset_index(drop=True)
            ticker_pct_change = merged["c_ticker"].pct_change()
            if not ticker_pct_change.empty:
                ticker_pct_change.iloc[0] = 0.0
            merged["ticker_pct_change"] = ticker_pct_change * 100

            rolling_cov = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).cov(
                merged["benchmark_pct_change"]
            )
            benchmark_var = merged["benchmark_pct_change"].rolling(window=feature_window, min_periods=feature_window).var()
            beta = rolling_cov.divide(benchmark_var).replace([np.inf, -np.inf], np.nan)

            total_return = merged["c_ticker"].pct_change(periods=feature_window - 1) * 100
            rolling_std = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).std()
            rolling_std = rolling_std * np.sqrt(ANNUALIZATION)
            rolling_std = rolling_std.replace(0, np.nan)
            sharpe = total_return.divide(rolling_std)
            sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

            metrics = (
                pd.DataFrame(
                    {
                        "date": merged["date"],
                        "beta": beta,
                        "sharpe": sharpe,
                        "12-1_return": total_return,
                    }
                )
                .set_index("date")
                .sort_index()
            )

            metrics = metrics.reindex(feature_anchor_index, method="ffill")

            price_frame = history.set_index("date").sort_index()
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

            ohlc_frame = pd.DataFrame(price_components).sort_index()
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
                vol_series, vol_column = compute_volatility_series(returns_series, multi_price, args)
            except KeyError as err:
                if args.vol_estimator == "yang-zhang":
                    print(f"    {ticker}: {err}. Falling back to EWMA volatility.")
                    vol_series = ewm_volatility(returns_series, hl=args.vol_ewm_halflife)
                    vol_column = "vol_ewm_ann"
                else:
                    raise

            vol_single = vol_series.xs(ticker, level="ticker")
            feature_block = pd.DataFrame(index=feature_anchor_index, dtype=float)
            feature_block[vol_column] = vol_single.reindex(feature_block.index, method="ffill")
            feature_block["volatility_ann"] = feature_block[vol_column]

            trailing_series = trailing_return(multi_price, window=FEATURE_WINDOW, price_col="close")
            trailing_single = trailing_series.xs(ticker, level="ticker") * 100.0
            feature_block[f"return_{FEATURE_WINDOW}d"] = trailing_single.reindex(feature_block.index, method="ffill")

            mom_ewm_df = None
            if args.momentum_style in {"ewma", "ewma_sr", "composite"}:
                mom_ewm_df = ewm_momentum_returns(returns_series, half_lives=ewm_half_lives)
                mom_ewm_single = mom_ewm_df.xs(ticker, level="ticker")
                for column in mom_ewm_single.columns:
                    feature_block[column] = mom_ewm_single[column].reindex(feature_block.index, method="ffill")

            if args.momentum_style in {"ewma_sr", "composite"} and mom_ewm_df is not None:
                sr_df = sharpe_like_trend(mom_ewm_df, vol_series, annualize=ANNUALIZATION)
                sr_single = sr_df.xs(ticker, level="ticker")
                for column in sr_single.columns:
                    feature_block[column] = sr_single[column].reindex(feature_block.index, method="ffill")

            if args.momentum_style in {"ema_crossover", "composite"}:
                ema_series = ema_crossover_price(
                    multi_price,
                    fast_hl=args.ema_fast_halflife,
                    slow_hl=args.ema_slow_halflife,
                    price_col="close",
                )
                ema_single = ema_series.xs(ticker, level="ticker")
                feature_block["ema_xover_norm"] = ema_single.reindex(feature_block.index, method="ffill")

            metrics = metrics.join(feature_block, how="left")
            metrics["mom_score"] = compute_momentum_score(metrics, args.momentum_style)

            price_series = price_frame["c"].sort_index()
            price_dates = price_series.index.to_numpy()
            price_values = price_series.to_numpy()

            if len(price_dates) == 0:
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
            holding_period_days = np.full(len(analysis_months_array), np.nan)

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
                holding_period_days[valid_mask] = (valid_exit_idx - valid_entry_idx).astype(float)

            # Ensure we maintain a consistent number of rows for every component that will
            # be concatenated.  The metrics frame is already aligned with
            # ``feature_anchor_index`` (one row per analysis month); however, the anchor
            # dates can repeat when multiple analysis months share the same lookback
            # trading session (e.g. holidays or week-ends).  Reindexing a DataFrame whose
            # axis already contains duplicate labels raises ``ValueError`` in pandas, so we
            # normalise the index once here by switching to the unique analysis-month
            # labels before attaching the derived arrays below.
            aligned_metrics = _align_frame_to_length(metrics, len(feature_anchor_index))
            aligned_metrics.index = analysis_months_index
            monthly_results = aligned_metrics.copy()
            monthly_results["anchor_date"] = feature_anchor_index.to_numpy()

            monthly_results["entry_date"] = pd.Series(analysis_months_array, index=monthly_results.index)
            monthly_results["exit_date"] = pd.Series(exit_months_array, index=monthly_results.index)
            monthly_results["ticker"] = ticker
            monthly_results["forward_returns"] = pd.Series(forward_returns, index=monthly_results.index)
            monthly_results["holding_period_days"] = pd.Series(
                holding_period_days, index=monthly_results.index
            )

            if "volatility_ann" not in monthly_results.columns:
                monthly_results["volatility_ann"] = np.nan

            signal_columns = [
                col
                for col in metrics.columns
                if col.startswith("mom_ewm_hl")
                or col.endswith("_sr")
                or col == "ema_xover_norm"
                or col.startswith("return_")
            ]
            for column in signal_columns:
                if column in aligned_metrics.columns:
                    monthly_results[column] = aligned_metrics[column].to_numpy()

            entry_vol_daily = monthly_results["volatility_ann"].to_numpy() / np.sqrt(ANNUALIZATION)
            horizon = np.maximum(monthly_results["holding_period_days"].to_numpy(), 1.0)
            denom = entry_vol_daily * np.sqrt(horizon)
            forward_return_sr = np.divide(
                forward_returns / 100.0,
                denom,
                out=np.full_like(forward_returns, np.nan),
                where=denom > 0,
            )
            monthly_results["forward_return_sr"] = forward_return_sr
            monthly_results["momentum_style"] = args.momentum_style
            monthly_results["vol_estimator"] = args.vol_estimator
            monthly_results["vol_column"] = vol_column
            monthly_results["ewm_half_lives"] = ",".join(str(value) for value in ewm_half_lives)
            monthly_results["ema_fast_halflife"] = args.ema_fast_halflife
            monthly_results["ema_slow_halflife"] = args.ema_slow_halflife

            in_universe = np.array([ticker in universe_by_month.get(month, set()) for month in analysis_months])
            monthly_results = monthly_results[in_universe]
            monthly_results = monthly_results.dropna(
                subset=["beta", "sharpe", "12-1_return", "mom_score", "forward_returns", "volatility_ann"]
            )

            if monthly_results.empty:
                completed += 1
                durations.append((datetime.now() - ticker_start_time).total_seconds())
                if durations:
                    average_seconds = sum(durations) / len(durations)
                    remaining = ticker_count - completed
                    eta = datetime.now() + timedelta(seconds=average_seconds * remaining)
                    remaining_td = timedelta(seconds=int(average_seconds * remaining))
                    print(
                        f"    Filtered out {ticker}. Progress: {completed}/{ticker_count}. "
                        f"ETA {eta:%Y-%m-%d %H:%M:%S} ({_format_timedelta(remaining_td)} remaining)",
                        flush=True,
                    )
                continue

            monthly_results["entry_date"] = pd.to_datetime(monthly_results["entry_date"]).dt.strftime("%Y-%m-%d")
            monthly_results["exit_date"] = pd.to_datetime(monthly_results["exit_date"]).dt.strftime("%Y-%m-%d")

            all_monthly_results.append(monthly_results)

            completed += 1
            ticker_duration = (datetime.now() - ticker_start_time).total_seconds()
            durations.append(ticker_duration)
            average_seconds = sum(durations) / len(durations)
            remaining = ticker_count - completed
            eta = datetime.now() + timedelta(seconds=average_seconds * remaining)
            remaining_td = timedelta(seconds=int(average_seconds * remaining))
            elapsed_td = datetime.now() - loop_start_time
            print(
                f"    Finished {ticker} in {ticker_duration:.2f}s. Progress: {completed}/{ticker_count}. "
                f"Elapsed {_format_timedelta(elapsed_td)}. ETA {eta:%Y-%m-%d %H:%M:%S} "
                f"({_format_timedelta(remaining_td)} remaining)",
                flush=True,
            )

        if not all_monthly_results:
            raise ValueError("No data collected for any month; confirm ticker universe and data availability")

        full_dataset = pd.concat(all_monthly_results, ignore_index=True)
        full_dataset = full_dataset.sort_values(["entry_date", "mom_score"], ascending=[True, False])
        top_decile_dataset = (
            full_dataset.sort_values("mom_score", ascending=False)
            .groupby("entry_date", group_keys=False)
            .head(10)
        )
        bot_decile_dataset = (
            full_dataset.sort_values("mom_score", ascending=True)
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
            trade = pd.DataFrame(
                [
                    {
                        "date": covered_date,
                        "long": long_uni["forward_returns"].mean(),
                        "short": short_uni["forward_returns"].mean() * -1,
                        "long_sr": long_uni["forward_return_sr"].mean(),
                        "short_sr": short_uni["forward_return_sr"].mean() * -1,
                        "long_count": len(long_uni),
                        "short_count": len(short_uni),
                    }
                ]
            )
            trade["ls_sr"] = trade["long_sr"] + trade["short_sr"]
            trades.append(trade)

        if not trades:
            raise ValueError("No valid trades generated; investigate data quality or parameters")

        all_trades = pd.concat(trades, ignore_index=True)
        if all_trades.empty:
            raise ValueError("Trades table unexpectedly empty after concatenation")

        assert {"long", "short"}.issubset(all_trades.columns)

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

        long_returns = (all_trades["long"].to_numpy() / 100.0) * GROSS_LONG
        short_returns = (all_trades["short"].to_numpy() / 100.0) * GROSS_SHORT
        ls_returns = long_returns + short_returns

        cost_decimals = (
            (PER_SIDE_BPS / 1e4)
            * (
                turnover_frame["long_turnover"].to_numpy() * GROSS_LONG
                + turnover_frame["short_turnover"].to_numpy() * GROSS_SHORT
            )
        )
        ls_returns_net = ls_returns - cost_decimals

        overlay_scaler = None
        if args.apply_vol_overlay:
            ls_series = pd.Series(ls_returns, index=trades_dates, name="ls_return")
            overlay_scaler = vol_managed_scaler(
                ls_series,
                hl=args.overlay_halflife,
                target_vol_ann=args.overlay_target_vol,
                annualize=ANNUALIZATION,
            )
            overlay_scaler = overlay_scaler.reindex(trades_dates).fillna(1.0)
            overlay_scaler = overlay_scaler.clip(lower=0.0, upper=args.overlay_max_leverage)
            ls_returns_overlay = ls_returns * overlay_scaler.to_numpy()
            ls_returns_net_overlay = ls_returns_net * overlay_scaler.to_numpy()
            all_trades["ls_leverage"] = overlay_scaler.to_numpy()
        else:
            ls_returns_overlay = None
            ls_returns_net_overlay = None

        all_trades["long_nav"] = np.cumprod(1.0 + long_returns)
        all_trades["short_nav"] = np.cumprod(1.0 + short_returns)
        all_trades["ls_nav"] = np.cumprod(1.0 + ls_returns)
        all_trades["ls_nav_net"] = np.cumprod(1.0 + ls_returns_net)
        if ls_returns_overlay is not None and ls_returns_net_overlay is not None:
            all_trades["ls_nav_overlay"] = np.cumprod(1.0 + ls_returns_overlay)
            all_trades["ls_nav_net_overlay"] = np.cumprod(1.0 + ls_returns_net_overlay)
        all_trades["long_cum_%"] = (all_trades["long_nav"] - 1.0) * 100.0
        all_trades["short_cum_%"] = (all_trades["short_nav"] - 1.0) * 100.0
        all_trades["ls_cum_%"] = (all_trades["ls_nav"] - 1.0) * 100.0
        all_trades["ls_cum_%_net"] = (all_trades["ls_nav_net"] - 1.0) * 100.0
        if "ls_nav_overlay" in all_trades.columns:
            all_trades["ls_cum_%_overlay"] = (all_trades["ls_nav_overlay"] - 1.0) * 100.0
        if "ls_nav_net_overlay" in all_trades.columns:
            all_trades["ls_cum_%_net_overlay"] = (all_trades["ls_nav_net_overlay"] - 1.0) * 100.0

        plt.figure(figsize=(10, 6), dpi=200)
        plt.xticks(rotation=45)
        plt.suptitle("Cumulative Performance")
        plt.title("Monthly Rebalancing")
        plt.plot(trades_dates, all_trades["long_cum_%"], label="Top Decile (gross)")
        plt.plot(trades_dates, all_trades["short_cum_%"], label="Bottom Decile (gross)")
        plt.plot(trades_dates, all_trades["ls_cum_%"], label="Long-Short (gross)")
        plt.plot(trades_dates, all_trades["ls_cum_%_net"], label="Long-Short (net est. costs)")
        if "ls_cum_%_overlay" in all_trades.columns:
            plt.plot(trades_dates, all_trades["ls_cum_%_overlay"], label="Long-Short (overlay gross)")
        if "ls_cum_%_net_overlay" in all_trades.columns:
            plt.plot(trades_dates, all_trades["ls_cum_%_net_overlay"], label="Long-Short (overlay net)")
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
