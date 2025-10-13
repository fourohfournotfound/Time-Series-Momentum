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
import mysql.connector
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
# Prefer loading the point-in-time universe from MySQL so the public script
# retains parity with the in-house environment.  When the database is not
# reachable (e.g. during documentation builds or on hosted notebooks) we fall
# back to the checked-in CSV extract so that the rest of the backtest can still
# run end-to-end.
universe = None
try:
    engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')
    universe = pd.read_sql(
        "historical_liquid_tickers_polygon",
        con=engine
    ).drop_duplicates(subset=["date", "ticker"])
except Exception as db_error:
    print(
        "Warning: unable to connect to MySQL (" \
        f"{db_error}). Falling back to local CSV extract."
    )
    universe = pd.read_csv("historical_liquid_tickers.csv")
    for redundant_col in {"Unnamed: 0", "index"}:
        if redundant_col in universe.columns:
            universe = universe.drop(columns=[redundant_col])

if universe is None:
    raise RuntimeError("Failed to load point-in-time universe from both MySQL and CSV sources.")

if "date" in universe.columns:
    universe["date"] = pd.to_datetime(universe["date"]).dt.strftime("%Y-%m-%d")

universe = universe.drop_duplicates(subset=["date", "ticker"])

connection_session = requests.Session()
connection_session.headers.update({"User-Agent": "mtum-backtest/1.0"})

bench = connection_session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}"
    f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
).json()
benchmark_data = pd.json_normalize(bench["results"]).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# =============================================================================
# Backtesting - Feature Generation + Forward Return Calc
# =============================================================================
full_data_list = []
top_decile_list = []
bot_decile_list = []
times = []

months = universe["date"].drop_duplicates().values

def _score_month(month: str) -> pd.DataFrame:
    point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
    point_in_time_date = point_in_time_dates[-1]
    point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")
    tickers = point_in_time_universe["ticker"].drop_duplicates().values

    start_date = (pd.to_datetime(month) - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
    end_date = month
    next_month_date = np.sort(months[months > month])[0]
    last_month_date = (pd.to_datetime(month) - timedelta(days=30)).strftime("%Y-%m-%d")

    monthly_rows = []

    # (Optionally) collect inputs for entropy in one go (panel)
    panel_parts = []

    for ticker in tickers:
        try:
            ud = connection_session.get(
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{next_month_date}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
            ).json()
            if "results" not in ud:
                continue

            data = pd.json_normalize(ud["results"]).set_index("t")
            data.index = pd.to_datetime(data.index, unit="ms", utc=True).tz_convert("America/New_York")
            data["date"] = data.index.strftime("%Y-%m-%d")

            twelve_minus_one = data[data["date"] < last_month_date].copy().tail(252)
            if len(twelve_minus_one) < 252:
                continue

            # Merge with benchmark (QG exact)
            bmerge = pd.merge(
                left=benchmark_data[["c", "date"]],
                right=twelve_minus_one[["c", "date"]],
                on="date"
            )
            bmerge["benchmark_pct_change"] = np.round(bmerge["c_x"].pct_change() * 100, 2).fillna(0)
            bmerge["ticker_pct_change"]    = np.round(bmerge["c_y"].pct_change() * 100, 2).fillna(0)

            covmat = np.cov(bmerge["ticker_pct_change"], bmerge["benchmark_pct_change"])
            cov_tb = covmat[0, 1]
            var_b  = np.var(bmerge["benchmark_pct_change"])
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
                ewm_vol  = float(ewm_last["ewm_vol_ann"].iloc[0])
            else:
                ewm_mean = None
                ewm_vol  = None

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

            next_period = data[(data["date"] >= month) & (data["date"] <= next_month_date)].copy().sort_index()
            next_ret = np.round(((next_period["c"].iloc[-1] - next_period["c"].iloc[0]) /
                                 next_period["c"].iloc[0]) * 100, 2) if next_period.shape[0] >= 2 else np.nan

            monthly_rows.append(pd.DataFrame([{
                "entry_date": month,
                "ticker": ticker,
                "beta": beta,
                "sharpe": sharpe,
                "12-1_return": tkr_ret_period,
                "mom_score": mom_score,
                "forward_returns": next_ret,
                "exit_date": next_month_date
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
        last_feats = latest_per_ticker(feats, pd.Timestamp(month), [
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
for month in months[:-1]:
    try:
        start_time = datetime.now()
        out = _score_month(month)
        if isinstance(out, tuple):
            full_period_ticker_data, top_decile, bot_decile = out
            full_data_list.append(full_period_ticker_data)
            top_decile_list.append(top_decile)
            bot_decile_list.append(bot_decile)

        seconds_to_complete = (datetime.now() - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(months == month)[0][0] / len(months)) * 100, 2)
        iterations_remaining = len(months) - np.where(months == month)[0][0]
        avg_time = np.mean(times)
        eta = (datetime.now() + timedelta(seconds=int(avg_time * iterations_remaining)))
        print(f"{iteration}% complete, ETA ~ {eta}")

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
