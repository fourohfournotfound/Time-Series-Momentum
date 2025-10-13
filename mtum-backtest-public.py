# -*- coding: utf-8 -*-
"""
Created in 2025

Monthly-rebalanced cross-sectional momentum backtest with optional
volatility/EWMA/entropy filters. Classic mode reproduces QG results.

Notes:
  • Timeframe, PIT universe, SPY merge, rounding, and forward return windows
    all match the QG reference backtest.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sqlalchemy
from pandas_market_calendars import get_calendar

from data_providers import PolygonAggsClient
from relative_market_features import (
    add_entropy_info_features,
    ewm_moments,
    latest_per_ticker,
)

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"

# =============================================================================
# CONFIG
# =============================================================================
SCORING_MODE = "classic"   # "classic", "vol_scaled", "ewma", "ewma_vol_scaled"
USE_ENTROPY_FILTER = False
ENTROPY_CFG = dict(pe_window=126, pe_m=4, pe_tau=1, se_window=128, cs_bins=21)
EWM_HALFLIFE = 63
VOL_TARGET = 0.20
SCALE_CAP = 3.0
N_LONG = 10
N_SHORT = 10

# =============================================================================
# Date Management - Rebalancing Every Month
# =============================================================================
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(
    start_date="2019-01-01",
    end_date=(datetime.today() - timedelta(days=1))
).index.strftime("%Y-%m-%d").values

# =============================================================================
# Base Point-in-Time Universe + Benchmark
# =============================================================================
engine = None
universe: Optional[pd.DataFrame] = None
try:
    engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')
    with engine.connect() as connection:
        universe = pd.read_sql(
            "historical_liquid_tickers_polygon",
            con=connection
        ).drop_duplicates(subset=["date", "ticker"])
except (ModuleNotFoundError, sqlalchemy.exc.SQLAlchemyError) as db_error:
    warnings.warn(
        f"Warning: unable to connect to MySQL ({db_error}). Falling back to local CSV extract.",
        RuntimeWarning,
        stacklevel=2,
    )
    universe = pd.read_csv("historical_liquid_tickers.csv")
    redundant_cols = [col for col in ("Unnamed: 0", "index") if col in universe.columns]
    if redundant_cols:
        universe = universe.drop(columns=redundant_cols)
finally:
    if engine is not None:
        engine.dispose()

if universe is None:
    raise RuntimeError("Failed to load point-in-time universe from both MySQL and CSV sources.")

if "date" in universe.columns:
    universe["date"] = pd.to_datetime(universe["date"], utc=False)

universe = universe.drop_duplicates(subset=["date", "ticker"])
universe = universe.sort_values(["date", "ticker"])

connection_session = requests.Session()
connection_session.headers.update({"User-Agent": "mtum-backtest/1.0"})

polygon_client = PolygonAggsClient(
    polygon_api_key,
    session=connection_session,
)

POLYGON_TIMEOUT = polygon_client.config.timeout

bench_response = connection_session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}",
    params={
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": polygon_api_key,
    },
    timeout=POLYGON_TIMEOUT,
)
bench_response.raise_for_status()
bench = bench_response.json()
benchmark_data = pd.json_normalize(bench["results"]).set_index("t")
benchmark_index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data.index = benchmark_index
benchmark_data["date"] = benchmark_index.tz_localize(None).normalize()
benchmark_panel = benchmark_data.loc[:, ["c", "date"]]

# =============================================================================
# Backtesting - Feature Generation + Forward Return Calc
# =============================================================================
full_data_list = []
top_decile_list = []
bot_decile_list = []
times = []

months_index = universe["date"].drop_duplicates().sort_values()
months = months_index.to_list()

if len(months) < 2:
    raise RuntimeError("Universe must contain at least two monthly observations for the backtest.")

month_strings = pd.Index(months).strftime("%Y-%m-%d").to_numpy()

point_in_time_snapshots: Dict[pd.Timestamp, pd.DataFrame] = {
    date: snapshot.copy()
    for date, snapshot in universe.groupby("date", sort=True)
}

start_buffer = pd.Timedelta(days=365 + 60)
end_buffer = pd.Timedelta(days=35)

global_price_start = (months[0] - start_buffer).strftime("%Y-%m-%d")
global_price_end = (months[-1] + end_buffer).strftime("%Y-%m-%d")

cache_dir = Path(".cache/polygon")
cache_dir.mkdir(parents=True, exist_ok=True)


def _cache_key(ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "-").replace(".", "-")
    return cache_dir / f"{safe_ticker}_{global_price_start}_{global_price_end}.parquet"


def _load_polygon_history(ticker: str) -> pd.DataFrame:
    cache_path = _cache_key(ticker)
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached["date"] = pd.to_datetime(cached["date"], utc=False)
            return cached
        except Exception:
            cache_path.unlink(missing_ok=True)

    data = polygon_client.get_daily_bars(ticker, global_price_start, global_price_end)
    if data.empty:
        cache_path.touch(exist_ok=True)
        return data

    data["date"] = pd.to_datetime(data["date"], utc=False)
    data.to_parquet(cache_path, index=False)
    return data


def _fetch_histories(tickers: np.ndarray) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}

    def _wrapped(symbol: str) -> Tuple[str, pd.DataFrame]:
        frame = _load_polygon_history(symbol)
        frame = frame.rename(columns={"c": "close"})
        return symbol, frame

    max_workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_wrapped, tkr): tkr for tkr in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                sym, frame = future.result()
                results[sym] = frame
            except Exception as fetch_error:
                warnings.warn(f"Failed to load Polygon history for {ticker}: {fetch_error}")
                results[ticker] = pd.DataFrame(columns=["date", "close"])

    return results


all_tickers = universe["ticker"].dropna().unique()
price_histories = _fetch_histories(all_tickers)


def _build_price_panel(histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, frame in histories.items():
        if frame.empty:
            continue
        sub = frame[["date", "close"]].copy()
        sub["ticker"] = symbol
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    panel = panel.drop_duplicates(subset=["date", "ticker"])
    panel = panel.sort_values(["date", "ticker"])
    panel = panel.set_index(["date", "ticker"]).sort_index()
    panel["return_pct"] = panel.groupby(level="ticker")["close"].pct_change() * 100.0
    panel["return_dec"] = panel["return_pct"] / 100.0
    return panel


price_panel = _build_price_panel(price_histories)

if price_panel.empty:
    raise RuntimeError("No Polygon price history available for the requested universe.")

benchmark_panel = benchmark_panel.copy()
benchmark_panel["return_pct"] = benchmark_panel["c"].pct_change() * 100.0
benchmark_panel["return_dec"] = benchmark_panel["return_pct"] / 100.0
benchmark_panel = benchmark_panel.dropna(subset=["date"])
benchmark_panel = benchmark_panel.sort_values("date")
benchmark_panel.set_index("date", inplace=True)

bench_var = benchmark_panel["return_pct"].rolling(window=252, min_periods=252).var()

price_panel = price_panel.join(benchmark_panel["return_pct"].rename("bench_return_pct"), on="date")
price_panel = price_panel.join(bench_var.rename("bench_var"), on="date")


def _rolling_cov(group: pd.DataFrame) -> pd.Series:
    return group["return_pct"].rolling(window=252, min_periods=252).cov(group["bench_return_pct"])


price_panel["cov_with_bench"] = price_panel.groupby(level="ticker", group_keys=False).apply(_rolling_cov)
price_panel["beta"] = price_panel["cov_with_bench"] / price_panel["bench_var"].replace(0.0, np.nan)
price_panel["beta"] = price_panel["beta"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

price_panel["close_shift_252"] = price_panel.groupby(level="ticker")["close"].shift(252)
price_panel["12-1_return"] = ((price_panel["close"] - price_panel["close_shift_252"]) / price_panel["close_shift_252"]) * 100.0

rolling_std = price_panel.groupby(level="ticker")["return_pct"].rolling(window=252, min_periods=252).std().droplevel(0)
price_panel["std_daily"] = rolling_std * np.sqrt(252)

price_panel["sharpe"] = np.where(
    price_panel["std_daily"].replace(0.0, np.nan).notna(),
    price_panel["12-1_return"] / price_panel["std_daily"],
    0.0,
)

if SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
    ewm_inputs = price_panel[["return_dec"]].rename(columns={"return_dec": "returns"})
    ewm_features = ewm_moments(ewm_inputs, ret_col="returns", halflife=EWM_HALFLIFE, use_lagged_data=True)
    price_panel = price_panel.join(ewm_features)
    price_panel["ewm_mean_pct"] = price_panel["ewm_mean"] * 100.0
else:
    price_panel["ewm_mean_pct"] = np.nan
    price_panel["ewm_vol_ann"] = np.nan

if SCORING_MODE == "classic":
    price_panel["mom_score"] = price_panel["beta"] * price_panel["sharpe"]
elif SCORING_MODE == "vol_scaled":
    price_panel["mom_score"] = price_panel["beta"] * (
        price_panel["12-1_return"] / price_panel["std_daily"].replace(0.0, np.nan)
    )
elif SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
    price_panel["mom_score"] = price_panel["beta"] * (
        price_panel["ewm_mean_pct"] / price_panel["ewm_vol_ann"].replace(0.0, np.nan)
    )
else:
    price_panel["mom_score"] = price_panel["beta"] * price_panel["sharpe"]

price_panel.replace([np.inf, -np.inf], np.nan, inplace=True)
price_panel = price_panel.dropna(subset=["mom_score", "12-1_return", "sharpe"], how="any")

price_frame = price_panel.reset_index().copy()


def _score_month(index: int):
    month = months[index]
    next_month = months[index + 1]
    point_in_time_universe = point_in_time_snapshots.get(month)
    if point_in_time_universe is None or point_in_time_universe.empty:
        return pd.DataFrame([])

    tickers = point_in_time_universe["ticker"].to_numpy()

    feature_mask = (
        (price_frame["date"] <= month)
        & (price_frame["ticker"].isin(tickers))
    )
    features = price_frame.loc[feature_mask]
    if features.empty:
        return pd.DataFrame([])

    latest_features = (
        features.sort_values("date")
        .drop_duplicates(subset="ticker", keep="last")
        .copy()
    )

    window_mask = (
        (price_frame["date"] >= month)
        & (price_frame["date"] <= next_month)
        & (price_frame["ticker"].isin(tickers))
    )
    window = price_frame.loc[window_mask, ["date", "ticker", "close"]]
    if window.empty:
        return pd.DataFrame([])

    window = window.sort_values(["ticker", "date"])
    valid_forward = window.groupby("ticker").filter(lambda g: g.shape[0] >= 2)
    if valid_forward.empty:
        return pd.DataFrame([])

    entry_prices = (
        valid_forward.groupby("ticker").head(1).set_index("ticker")["close"]
    )
    exit_prices = (
        valid_forward.groupby("ticker").tail(1).set_index("ticker")["close"]
    )
    forward_returns = ((exit_prices - entry_prices) / entry_prices) * 100.0

    latest_features = latest_features.merge(
        forward_returns.rename("forward_returns"),
        on="ticker",
        how="inner",
    )
    if latest_features.empty:
        return pd.DataFrame([])

    latest_features["entry_date"] = month.strftime("%Y-%m-%d")
    latest_features["exit_date"] = next_month.strftime("%Y-%m-%d")

    selection = latest_features[[
        "entry_date",
        "exit_date",
        "ticker",
        "beta",
        "sharpe",
        "12-1_return",
        "mom_score",
        "forward_returns",
    ]]

    if USE_ENTROPY_FILTER:
        history_slice = price_panel.loc[
            (price_panel.index.get_level_values("date") <= month)
            & (price_panel.index.get_level_values("ticker").isin(tickers)),
            ["close"],
        ].dropna().reset_index()
        if not history_slice.empty:
            history_slice = history_slice.set_index(["date", "ticker"]).sort_index()
            history_slice["returns"] = history_slice.groupby(level="ticker")["close"].pct_change()
            feats = add_entropy_info_features(
                history_slice[["returns"]].copy(),
                price_col="closeadj",
                use_lagged_data=True,
                pe_window=ENTROPY_CFG["pe_window"],
                pe_m=ENTROPY_CFG["pe_m"],
                pe_tau=ENTROPY_CFG["pe_tau"],
                se_window=ENTROPY_CFG["se_window"],
                cs_bins=ENTROPY_CFG["cs_bins"],
            )
            last_feats = latest_per_ticker(
                feats,
                month,
                [
                    f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}",
                    f"spectral_entropy_w{ENTROPY_CFG['se_window']}",
                    f"peer_surprisal_cs_bins{ENTROPY_CFG['cs_bins']}",
                    "peer_surprisal_z",
                    f"peer_jsd_bins{ENTROPY_CFG['cs_bins']}",
                    "peer_jsd_z",
                ],
            )
            pe_col = f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}"
            se_col = f"spectral_entropy_w{ENTROPY_CFG['se_window']}"
            ok = (
                last_feats[pe_col].between(0.15, 0.98)
                & last_feats[se_col].between(0.15, 0.98)
            )
            keep = set(last_feats[ok].index.astype(str))
            selection = selection[selection["ticker"].isin(keep)]

    if selection.empty:
        return pd.DataFrame([])

    top_decile = selection.sort_values(by="mom_score", ascending=False).head(N_LONG)
    bot_decile = selection.sort_values(by="mom_score", ascending=True).head(N_SHORT)

    return selection, top_decile, bot_decile


# Run over months[:-1] exactly like QG
total_iterations = len(months) - 1
for idx in range(total_iterations):
    month_label = month_strings[idx]
    try:
        start_time = datetime.now()
        out = _score_month(idx)
        if isinstance(out, tuple):
            full_period_ticker_data, top_decile, bot_decile = out
            full_data_list.append(full_period_ticker_data)
            top_decile_list.append(top_decile)
            bot_decile_list.append(bot_decile)

        seconds_to_complete = (datetime.now() - start_time).total_seconds()
        times.append(seconds_to_complete)
        completed = idx + 1
        iteration = round((completed / total_iterations) * 100, 2)
        iterations_remaining = total_iterations - completed
        avg_time = float(np.mean(times)) if times else 0.0
        eta_seconds = avg_time * iterations_remaining
        eta_delta = timedelta(seconds=int(round(eta_seconds))) if eta_seconds else timedelta(0)
        print(f"{iteration}% complete after {month_label}, ETA in ~ {eta_delta}")

    except Exception as macro_error:
        print(macro_error)
        continue

full_dataset = pd.concat(full_data_list) if full_data_list else pd.DataFrame([])
top_decile_dataset = pd.concat(top_decile_list) if top_decile_list else pd.DataFrame([])
bot_decile_dataset = pd.concat(bot_decile_list) if bot_decile_list else pd.DataFrame([])

# =============================================================================
# Backtest (unchanged math) + Reporting
# =============================================================================
covered_dates = full_dataset["entry_date"].drop_duplicates().values if not full_dataset.empty else []

trade_list = []
for covered_date in covered_dates:
    long_uni = top_decile_dataset[top_decile_dataset["entry_date"] == covered_date].copy()
    short_uni = bot_decile_dataset[bot_decile_dataset["entry_date"] == covered_date].copy()
    if long_uni.empty or short_uni.empty:
        continue

    trade_data = pd.DataFrame(
        {
            "date": [covered_date],
            "long_return": [long_uni["forward_returns"].mean() / 100.0],
            "short_return": [(-short_uni["forward_returns"].mean()) / 100.0],
        }
    )
    trade_data["portfolio_return"] = trade_data["long_return"] + trade_data["short_return"]
    trade_list.append(trade_data)

if trade_list:
    all_trades = pd.concat(trade_list).sort_values("date")
    all_trades.reset_index(drop=True, inplace=True)

    all_trades["long_equity"] = (1.0 + all_trades["long_return"]).cumprod()
    all_trades["short_equity"] = (1.0 + all_trades["short_return"]).cumprod()
    all_trades["portfolio_equity"] = (1.0 + all_trades["portfolio_return"]).cumprod()

    plt.figure(figsize=(12, 7), dpi=200)
    plt.xticks(rotation=45)
    plt.suptitle("Gross Cumulative Performance")
    plt.title("Monthly Rebalancing")
    timeline = pd.to_datetime(all_trades["date"])
    plt.plot(timeline, (all_trades["long_equity"] - 1.0) * 100.0)
    plt.plot(timeline, (all_trades["short_equity"] - 1.0) * 100.0)
    plt.plot(timeline, (all_trades["portfolio_equity"] - 1.0) * 100.0)
    plt.legend(["Top Decile", "Bottom Decile", "Long-Short"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative % Returns")
    output_path = Path("momentum_equity_curve.png")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()

    monthly_returns = all_trades["portfolio_return"]
    cumulative = all_trades["portfolio_equity"]
    years = len(monthly_returns) / 12.0 if len(monthly_returns) else 0
    cagr = cumulative.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = monthly_returns.std(ddof=1) * np.sqrt(12) if len(monthly_returns) > 1 else np.nan
    sharpe_ratio = (monthly_returns.mean() * np.sqrt(12)) / volatility if volatility and not np.isnan(volatility) and volatility != 0 else np.nan
    roll_max = cumulative.cummax()
    drawdowns = cumulative / roll_max - 1
    max_drawdown = drawdowns.min() if not drawdowns.empty else np.nan

    print("\nBacktest Summary Metrics")
    print("========================")
    print(f"CAGR: {cagr:.2%}" if not np.isnan(cagr) else "CAGR: n/a")
    print(f"Annualized Volatility: {volatility:.2%}" if not np.isnan(volatility) else "Annualized Volatility: n/a")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "Sharpe Ratio: n/a")
    print(f"Max Drawdown: {max_drawdown:.2%}" if not np.isnan(max_drawdown) else "Max Drawdown: n/a")
    print(f"Equity curve saved to: {output_path.resolve()}")
else:
    print("No trades were generated. Check data availability or configuration settings.")
