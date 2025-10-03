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

from pandas_market_calendars import get_calendar

from data_providers import AlpacaBarsClient, PolygonAggsClient


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
    engine = sqlalchemy.create_engine(args.database_url)
    universe = pd.read_sql(args.universe_table, con=engine).drop_duplicates(subset=["date", "ticker"])
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


def main() -> None:
    args = parse_args()
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

        rebalance_gap = timedelta(days=int(args.rebalance_frequency_days))
        feature_window = 252

        benchmark_data = benchmark_data.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
        benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
        benchmark_data = benchmark_data.sort_values("date").reset_index(drop=True)
        benchmark_data["benchmark_pct_change"] = benchmark_data["c"].pct_change().fillna(0) * 100

        universe_by_month = {
            month: set(
                universe.loc[universe["date"] == month, "ticker"].drop_duplicates().values
            )
            for month in analysis_months
        }

        analysis_months_index = analysis_months
        exit_months_index = months_index[1:]
        feature_anchor_index = analysis_months - rebalance_gap
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
            merged["ticker_pct_change"] = merged["c_ticker"].pct_change().fillna(0) * 100

            rolling_cov = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).cov(
                merged["benchmark_pct_change"]
            )
            benchmark_var = merged["benchmark_pct_change"].rolling(window=feature_window, min_periods=feature_window).var()
            beta = rolling_cov.divide(benchmark_var).replace([np.inf, -np.inf], np.nan)

            total_return = merged["c_ticker"].pct_change(periods=feature_window - 1) * 100
            rolling_std = merged["ticker_pct_change"].rolling(window=feature_window, min_periods=feature_window).std()
            rolling_std = rolling_std * np.sqrt(252)
            rolling_std = rolling_std.replace(0, np.nan)
            sharpe = total_return.divide(rolling_std)
            sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

            mom_score = beta * sharpe

            metrics = (
                pd.DataFrame(
                    {
                        "date": merged["date"],
                        "beta": beta,
                        "sharpe": sharpe,
                        "12-1_return": total_return,
                        "mom_score": mom_score,
                    }
                )
                .set_index("date")
                .sort_index()
            )

            metrics = metrics.reindex(feature_anchor_index, method="ffill")

            price_series = history.set_index("date")["c"].sort_index()
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
                }
            )

            in_universe = np.array([ticker in universe_by_month.get(month, set()) for month in analysis_months])
            monthly_results = monthly_results[in_universe]
            monthly_results = monthly_results.dropna(
                subset=["beta", "sharpe", "12-1_return", "mom_score", "forward_returns"]
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
                    }
                ]
            )
            trades.append(trade)

        if not trades:
            raise ValueError("No valid trades generated; investigate data quality or parameters")

        all_trades = pd.concat(trades, ignore_index=True)
        all_trades["long_pnl"] = all_trades["long"].cumsum()
        all_trades["short_pnl"] = all_trades["short"].cumsum()
        all_trades["portfolio_pnl"] = all_trades["long_pnl"] + all_trades["short_pnl"]

        plt.figure(figsize=(10, 6), dpi=200)
        plt.xticks(rotation=45)
        plt.suptitle("Gross Cumulative Performance")
        plt.title("Monthly Rebalancing")
        plt.plot(pd.to_datetime(all_trades["date"]), all_trades["long_pnl"])
        plt.plot(pd.to_datetime(all_trades["date"]), all_trades["short_pnl"])
        plt.plot(pd.to_datetime(all_trades["date"]), all_trades["portfolio_pnl"])
        plt.legend(["Top Decile", "Bottom Decile", "Long-Short"])
        plt.xlabel("Date")
        plt.ylabel("Cumulative % Returns")
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
