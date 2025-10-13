# -*- coding: utf-8 -*-
"""
Created in 2025

Monthly-rebalanced cross-sectional momentum backtest with optional
volatility/EWMA/entropy filters. Classic mode reproduces QG results.

Notes:
  • Timeframe, PIT universe, SPY merge, rounding, and forward return windows
    all match the QG reference backtest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sqlalchemy
import warnings
from typing import Dict, Optional
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

from relative_market_features import (
    add_entropy_info_features, ewm_moments, latest_per_ticker
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

polygon_params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": polygon_api_key,
}
POLYGON_TIMEOUT = 30

bench_response = connection_session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}",
    params=polygon_params,
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

global_price_start = (months[0] - pd.Timedelta(days=365 + 60)).strftime("%Y-%m-%d")
global_price_end = months[-1].strftime("%Y-%m-%d")

price_cache: Dict[str, pd.DataFrame] = {}


def _load_polygon_history(ticker: str) -> pd.DataFrame:
    if ticker in price_cache:
        return price_cache[ticker]

    response = connection_session.get(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{global_price_start}/{global_price_end}",
        params=polygon_params,
        timeout=POLYGON_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        empty = pd.DataFrame(columns=["c", "t"]).set_index(pd.Index([], name="t"))
        price_cache[ticker] = empty
        return empty

    data = pd.json_normalize(results).set_index("t")
    idx = pd.to_datetime(data.index, unit="ms", utc=True).tz_convert("America/New_York")
    data.index = idx
    local_dates = idx.tz_localize(None).normalize()
    data["date"] = local_dates
    price_cache[ticker] = data
    return data


def _score_month(index: int) -> pd.DataFrame:
    month = months[index]
    next_month = months[index + 1]
    point_in_time_universe = point_in_time_snapshots.get(month)
    if point_in_time_universe is None or point_in_time_universe.empty:
        return pd.DataFrame([])

    tickers = point_in_time_universe["ticker"].to_numpy()

    start_date = month - pd.Timedelta(days=365 + 60)
    last_month_date = month - pd.Timedelta(days=30)

    monthly_rows = []

    # (Optionally) collect inputs for entropy in one go (panel)
    panel_parts = []

    month_str = month.strftime("%Y-%m-%d")
    next_month_str = next_month.strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            history = _load_polygon_history(ticker)
            if history.empty:
                continue

            data = history[(history["date"] >= start_date) & (history["date"] <= next_month)].copy()
            if data.empty:
                continue

            twelve_minus_one = data[data["date"] < last_month_date].copy().tail(252)
            if len(twelve_minus_one) < 252:
                continue

            # Merge with benchmark (QG exact)
            bmerge = pd.merge(
                left=benchmark_panel,
                right=twelve_minus_one[["c", "date"]],
                on="date"
            )
            bmerge["benchmark_pct_change"] = np.round(bmerge["c_x"].pct_change() * 100, 2).fillna(0)
            bmerge["ticker_pct_change"] = np.round(bmerge["c_y"].pct_change() * 100, 2).fillna(0)

            covmat = np.cov(bmerge["ticker_pct_change"], bmerge["benchmark_pct_change"])
            cov_tb = covmat[0, 1]
            var_b = np.var(bmerge["benchmark_pct_change"])
            beta = cov_tb / var_b if var_b != 0 else 0.0

            tkr_ret_period = np.round(((twelve_minus_one["c"].iloc[-1] - twelve_minus_one["c"].iloc[0]) /
                                       twelve_minus_one["c"].iloc[0]) * 100, 2)
            std_daily = bmerge["ticker_pct_change"].std() * np.sqrt(252)
            sharpe = (tkr_ret_period / std_daily) if std_daily != 0 else 0.0

            # Optional EWMA for month
            if SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
                mdf = data[data["date"] < last_month_date][["c", "date"]].copy()
                mdf["ticker"] = ticker
                mdf = mdf.set_index(["date", "ticker"]).sort_index()
                mdf["returns"] = mdf.groupby(level="ticker")["c"].pct_change()
                ewm_stats = ewm_moments(mdf, ret_col="returns", halflife=EWM_HALFLIFE, use_lagged_data=True)
                ewm_last = ewm_stats.groupby(level="ticker").tail(1)
                ewm_mean = float(ewm_last["ewm_mean"].iloc[0]) * 100.0
                ewm_vol = float(ewm_last["ewm_vol_ann"].iloc[0])
            else:
                ewm_mean = None
                ewm_vol = None

            if SCORING_MODE == "classic":
                mom_score = beta * sharpe
            elif SCORING_MODE == "vol_scaled":
                mom_score = beta * (tkr_ret_period / (std_daily if std_daily != 0 else 1e-12))
            elif SCORING_MODE == "ewma":
                if ewm_vol is None or ewm_vol == 0:
                    mom_score = 0.0
                else:
                    mom_score = beta * (ewm_mean / ewm_vol)
            elif SCORING_MODE == "ewma_vol_scaled":
                if ewm_vol is None or ewm_vol == 0:
                    mom_score = 0.0
                else:
                    mom_score = beta * (ewm_mean / ewm_vol)
            else:
                mom_score = beta * sharpe

            next_period = data[(data["date"] >= month) & (data["date"] <= next_month)].copy().sort_index()
            next_ret = np.round(((next_period["c"].iloc[-1] - next_period["c"].iloc[0]) /
                                 next_period["c"].iloc[0]) * 100, 2) if next_period.shape[0] >= 2 else np.nan

            monthly_rows.append(pd.DataFrame([{
                "entry_date": month_str,
                "ticker": ticker,
                "beta": beta,
                "sharpe": sharpe,
                "12-1_return": tkr_ret_period,
                "mom_score": mom_score,
                "forward_returns": next_ret,
                "exit_date": next_month_str
            }]))

            # Prepare panel for entropy filtering
            if USE_ENTROPY_FILTER:
                tmp = data[data["date"] < last_month_date][["c", "date"]].copy()
                tmp["ticker"] = ticker
                panel_parts.append(tmp)

        except Exception as error:
            print(error)
            continue

    if not monthly_rows:
        return pd.DataFrame([])

    full_period_ticker_data = pd.concat(monthly_rows)

    if USE_ENTROPY_FILTER and panel_parts:
        panel = pd.concat(panel_parts).set_index(["date", "ticker"]).sort_index()
        panel["returns"] = panel.groupby(level="ticker")["c"].pct_change()

        feats = add_entropy_info_features(
            panel[["returns"]].copy(), price_col="closeadj", use_lagged_data=True,
            pe_window=ENTROPY_CFG["pe_window"], pe_m=ENTROPY_CFG["pe_m"], pe_tau=ENTROPY_CFG["pe_tau"],
            se_window=ENTROPY_CFG["se_window"], cs_bins=ENTROPY_CFG["cs_bins"]
        )
        last_feats = latest_per_ticker(feats, month, [
            f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}",
            f"spectral_entropy_w{ENTROPY_CFG['se_window']}",
            f"peer_surprisal_cs_bins{ENTROPY_CFG['cs_bins']}",
            "peer_surprisal_z",
            f"peer_jsd_bins{ENTROPY_CFG['cs_bins']}",
            "peer_jsd_z",
        ])
        pe_col = f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}"
        se_col = f"spectral_entropy_w{ENTROPY_CFG['se_window']}"

        ok = (
            (last_feats[pe_col].between(0.15, 0.98)) &
            (last_feats[se_col].between(0.15, 0.98))
        )
        keep = set(last_feats[ok].index.astype(str))
        full_period_ticker_data = full_period_ticker_data[full_period_ticker_data["ticker"].isin(keep)]

    # Top/bottom 10 for this month
    top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(N_LONG)
    bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(N_SHORT)

    return full_period_ticker_data, top_decile, bot_decile

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
        avg_time = np.mean(times)
        eta = (datetime.now() + timedelta(seconds=int(avg_time * iterations_remaining)))
        print(f"{iteration}% complete after {month_label}, ETA ~ {eta}")

    except Exception as macro_error:
        print(macro_error)
        continue

full_dataset = pd.concat(full_data_list) if full_data_list else pd.DataFrame([])
top_decile_dataset = pd.concat(top_decile_list) if top_decile_list else pd.DataFrame([])
bot_decile_dataset = pd.concat(bot_decile_list) if bot_decile_list else pd.DataFrame([])

# =============================================================================
# Backtest (unchanged math)
# =============================================================================
covered_dates = full_dataset["entry_date"].drop_duplicates().values if not full_dataset.empty else []

trade_list = []
for covered_date in covered_dates:
    long_uni = top_decile_dataset[top_decile_dataset["entry_date"] == covered_date].copy()
    short_uni = bot_decile_dataset[bot_decile_dataset["entry_date"] == covered_date].copy()
    trade_data = pd.DataFrame([{
        "date": covered_date,
        "long": long_uni["forward_returns"].mean(),
        "short": short_uni["forward_returns"].mean() * -1
    }])
    trade_list.append(trade_data)

if trade_list:
    all_trades = pd.concat(trade_list).sort_values("date")
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
    plt.show()
    plt.close()
