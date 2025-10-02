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

        full_data_list = []
        top_decile_list = []
        bot_decile_list = []
        times = []

        rebalance_gap = timedelta(days=int(args.rebalance_frequency_days))

        for idx, month in enumerate(analysis_months):
            month_str = month.strftime("%Y-%m-%d")
            next_month = months_index[idx + 1]
            next_month_str = next_month.strftime("%Y-%m-%d")
            start_date_str = (month - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            last_month_str = (month - rebalance_gap).strftime("%Y-%m-%d")

            monthly_universe = universe[universe["date"] == month].drop_duplicates(subset=["ticker"], keep="last")
            tickers = monthly_universe["ticker"].drop_duplicates().values

            if len(tickers) == 0:
                continue

            monthly_ticker_data = []
            start_time = datetime.now()

            for ticker in tickers:
                try:
                    history = load_history(ticker)
                except Exception as err:
                    print(f"{ticker} download failed: {err}")
                    continue

                if history.empty:
                    continue

                history = history.sort_values("timestamp")

                lookback_mask = (history["date"] >= start_date_str) & (history["date"] < last_month_str)
                twelve_minus_one = history.loc[lookback_mask].copy().tail(252)

                if len(twelve_minus_one) < 252:
                    continue

                twelve_minus_one["year"] = twelve_minus_one["timestamp"].dt.year
                twelve_minus_one["month"] = twelve_minus_one["timestamp"].dt.month

                twelve_minus_one_return = round(
                    ((twelve_minus_one["c"].iloc[-1] - twelve_minus_one["c"].iloc[0]) / twelve_minus_one["c"].iloc[0]) * 100,
                    2,
                )

                benchmark_slice = benchmark_data[(benchmark_data["date"] >= twelve_minus_one["date"].iloc[0]) & (benchmark_data["date"] <= twelve_minus_one["date"].iloc[-1])].copy()
                benchmark_and_underlying = pd.merge(
                    benchmark_slice[["c", "date"]],
                    twelve_minus_one[["c", "date"]],
                    on="date",
                    how="inner",
                    suffixes=("_benchmark", "_ticker"),
                )

                if len(benchmark_and_underlying) < 2:
                    continue

                benchmark_and_underlying["benchmark_pct_change"] = (
                    benchmark_and_underlying["c_benchmark"].pct_change().fillna(0) * 100
                )
                benchmark_and_underlying["ticker_pct_change"] = (
                    benchmark_and_underlying["c_ticker"].pct_change().fillna(0) * 100
                )

                variance_benchmark = np.var(benchmark_and_underlying["benchmark_pct_change"])
                if variance_benchmark == 0:
                    continue

                covariance_matrix = np.cov(
                    benchmark_and_underlying["ticker_pct_change"],
                    benchmark_and_underlying["benchmark_pct_change"],
                )
                covariance_ticker_benchmark = covariance_matrix[0, 1]
                beta = covariance_ticker_benchmark / variance_benchmark

                ticker_return_over_period = round(
                    (
                        (
                            benchmark_and_underlying["c_ticker"].iloc[-1]
                            - benchmark_and_underlying["c_ticker"].iloc[0]
                        )
                        / benchmark_and_underlying["c_ticker"].iloc[0]
                    )
                    * 100,
                    2,
                )

                std_of_returns = benchmark_and_underlying["ticker_pct_change"].std() * np.sqrt(252)
                if std_of_returns == 0:
                    continue

                sharpe = ticker_return_over_period / std_of_returns
                theo_expected = beta * sharpe

                forward_mask = (history["date"] >= month_str) & (history["date"] <= next_month_str)
                next_period = history.loc[forward_mask].copy()
                if len(next_period) < 2:
                    continue

                next_period_returns = round(
                    ((next_period["c"].iloc[-1] - next_period["c"].iloc[0]) / next_period["c"].iloc[0]) * 100,
                    2,
                )

                ticker_frame = pd.DataFrame(
                    [
                        {
                            "entry_date": month_str,
                            "exit_date": next_month_str,
                            "ticker": ticker,
                            "beta": beta,
                            "sharpe": sharpe,
                            "12-1_return": ticker_return_over_period,
                            "mom_score": theo_expected,
                            "forward_returns": next_period_returns,
                        }
                    ]
                )
                monthly_ticker_data.append(ticker_frame)

            if not monthly_ticker_data:
                continue

            monthly_data = pd.concat(monthly_ticker_data, ignore_index=True)
            top_decile = monthly_data.sort_values(by="mom_score", ascending=False).head(10)
            bot_decile = monthly_data.sort_values(by="mom_score", ascending=True).head(10)

            full_data_list.append(monthly_data)
            top_decile_list.append(top_decile)
            bot_decile_list.append(bot_decile)

            end_time = datetime.now()
            seconds_to_complete = (end_time - start_time).total_seconds()
            times.append(seconds_to_complete)

            iteration = round(((idx + 1) / len(analysis_months)) * 100, 2)
            iterations_remaining = len(analysis_months) - (idx + 1)
            average_time = np.mean(times)
            eta = datetime.now() + timedelta(seconds=int(average_time * iterations_remaining)) if times else datetime.now()
            time_remaining = eta - datetime.now()
            print(f"{iteration}% complete, {time_remaining} left, ETA: {eta}")

        # end for loop

        if not full_data_list:
            raise ValueError("No data collected for any month; confirm ticker universe and data availability")

        full_dataset = pd.concat(full_data_list, ignore_index=True)
        top_decile_dataset = pd.concat(top_decile_list, ignore_index=True)
        bot_decile_dataset = pd.concat(bot_decile_list, ignore_index=True)

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
        plt.show()

        plt.close()
    finally:
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    main()
