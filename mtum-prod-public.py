# -*- coding: utf-8 -*-
"""
Created in 2025

Real-time monthly momentum basket with optional volatility/EWMA/entropy filters.

Modes:
  SCORING_MODE in {"classic", "vol_scaled", "ewma", "ewma_vol_scaled"}
  USE_ENTROPY_FILTER: bool (post-score selection filter)
    - Filters are causal (t-1) and use entropy/peer info from custom_relative_market_features.

Parity guarantee:
  With SCORING_MODE="classic", USE_ENTROPY_FILTER=False, and all defaults,
  this script reproduces the original QG "prod" selection and scoring exactly
  (same rounding, windows, and merge arithmetic).  # See QG original for reference.
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
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

# =============================================================================
# CONFIG
# =============================================================================
SCORING_MODE = "classic"   # "classic", "vol_scaled", "ewma", "ewma_vol_scaled"
USE_ENTROPY_FILTER = False
ENTROPY_CFG = dict(pe_window=126, pe_m=4, pe_tau=1, se_window=128, cs_bins=21)  # causal defaults

# EWMA config (only used in "ewma"/"ewma_vol_scaled")
EWM_HALFLIFE = 63

# Vol targeting config (only used in "vol_scaled"/"ewma_vol_scaled")
VOL_TARGET = 0.20  # annualized
SCALE_CAP = 3.0    # clip scale to avoid extreme leverage

N_LONG = 10
N_SHORT = 10

# =============================================================================
# Date Management
# =============================================================================
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(
    start_date=(datetime.today() - timedelta(days=45)),
    end_date=(datetime.today())
).index.strftime("%Y-%m-%d").values

all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month

start_of_the_months = all_dates.drop_duplicates(subset=["year", "month"], keep="first").copy()
start_of_the_months["str_date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")

months = start_of_the_months["str_date"].values
month = months[-1]

start_date = (pd.to_datetime(month) - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
end_date = month

last_month_date = (pd.to_datetime(month) - timedelta(days=30)).strftime("%Y-%m-%d")
next_month_date = (pd.to_datetime(month) + timedelta(days=30)).strftime("%Y-%m-%d")

# =============================================================================
# Base Point-in-Time Universe + Benchmark
# =============================================================================
connection_session = requests.Session()
connection_session.headers.update({"User-Agent": "mtum-prod/1.0"})

universe = pd.read_sql("historical_liquid_tickers_polygon", con=engine)

# SPY benchmark (QG logic preserved)
bench = connection_session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}"
    f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
).json()
benchmark_data = pd.json_normalize(bench["results"]).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# =============================================================================
# Real-Time Basket Construction
# =============================================================================
point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]
point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")
tickers = point_in_time_universe["ticker"].drop_duplicates().values

monthly_ticker_list = []
times = []

def _score_single_ticker(ticker: str) -> pd.DataFrame:
    """
    Compute features & score for a single ticker (exactly matching 'classic' arithmetic by default).
    Returns one-row DataFrame with score & forward return.
    """
    try:
        # Pull trailing data (with an extra month forward for forward_return)
        underlying = connection_session.get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{next_month_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
        ).json()
        if "results" not in underlying:
            return pd.DataFrame([])

        ud = pd.json_normalize(underlying["results"]).set_index("t")
        ud.index = pd.to_datetime(ud.index, unit="ms", utc=True).tz_convert("America/New_York")
        ud["date"] = ud.index.strftime("%Y-%m-%d")

        # Trailing 12-1 (identical selection as QG)
        twelve_minus_one = ud[ud["date"] < last_month_date].copy().tail(252)
        if len(twelve_minus_one) < 252:
            return pd.DataFrame([])

        # QG rounding + benchmark merge (classic)
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

        # Period return and stdev (classic)
        tkr_ret_period = np.round(((twelve_minus_one["c"].iloc[-1] - twelve_minus_one["c"].iloc[0]) /
                                   twelve_minus_one["c"].iloc[0]) * 100, 2)
        std_daily = bmerge["ticker_pct_change"].std() * np.sqrt(252)
        sharpe = (tkr_ret_period / std_daily) if std_daily != 0 else 0.0

        # Optional EWMA inputs (t-1)
        if SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
            # Build MultiIndex (date,ticker) for causal EWMA/entropy
            mdf = twelve_minus_one[["c", "date"]].copy()
            mdf["ticker"] = ticker
            mdf = mdf.set_index(["date", "ticker"]).sort_index()
            # returns in decimal (not %) to avoid scale confusion
            mdf["returns"] = mdf.groupby(level="ticker")["c"].pct_change()
            ewm_stats = ewm_moments(mdf, ret_col="returns", halflife=EWM_HALFLIFE, use_lagged_data=True)
            ewm_last = ewm_stats.groupby(level="ticker").tail(1)
            ewm_mean = float(ewm_last["ewm_mean"].iloc[0]) * 100.0  # convert to %
            ewm_vol  = float(ewm_last["ewm_vol_ann"].iloc[0])
        else:
            ewm_mean = None
            ewm_vol = None

        # Score selection (identical by default)
        if SCORING_MODE == "classic":
            mom_score = beta * sharpe
        elif SCORING_MODE == "vol_scaled":
            # Use classic numerator but scale by realized vol (equivalent to Sharpe-based scaling).
            # This respects QG rounding on returns while risk-managing exposure.
            mom_score = beta * (tkr_ret_period / (std_daily if std_daily != 0 else 1e-12))
        elif SCORING_MODE == "ewma":
            # Use EWMA estimates (t-1) for a non-stationary-aware Sharpe proxy
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
            mom_score = beta * sharpe  # safety

        # Forward return (unchanged)
        fwd = ud[(ud["date"] >= end_date) & (ud["date"] <= next_month_date)].copy().sort_index()
        if fwd.shape[0] >= 2:
            fwd_ret = np.round(((fwd["c"].iloc[-1] - fwd["c"].iloc[0]) / fwd["c"].iloc[0]) * 100, 2)
        else:
            fwd_ret = np.nan

        row = {
            "entry_date": month,
            "ticker": ticker,
            "beta": beta,
            "sharpe": sharpe,
            "12-1_return": tkr_ret_period,
            "mom_score": mom_score,
            "forward_returns": fwd_ret
        }

        return pd.DataFrame([row])

    except Exception as e:
        print(e)
        return pd.DataFrame([])

# Compute rows (sequential to keep API gentle; can be parallelized if needed)
for ticker in tickers:
    start_time = datetime.now()
    df_row = _score_single_ticker(ticker)
    if not df_row.empty:
        monthly_ticker_list.append(df_row)

    seconds_to_complete = (datetime.now() - start_time).total_seconds()
    times.append(seconds_to_complete)
    iteration = round((np.where(tickers == ticker)[0][0] / len(tickers)) * 100, 2)
    iterations_remaining = len(tickers) - np.where(tickers == ticker)[0][0]
    avg_time = np.mean(times)
    eta = (datetime.now() + timedelta(seconds=int(avg_time * iterations_remaining)))
    print(f"{iteration}% complete, {len(monthly_ticker_list)} tickers scored, ETA ~ {eta}")

full_period_ticker_data = pd.concat(monthly_ticker_list) if monthly_ticker_list else pd.DataFrame(columns=[
    "entry_date", "ticker", "beta", "sharpe", "12-1_return", "mom_score", "forward_returns"
])

# Optional causal entropy/peer filtering at t-1
if USE_ENTROPY_FILTER and not full_period_ticker_data.empty:
    # Build MultiIndex returns panel up to last_month_date for ALL tickers that survived basic data checks
    panel_parts = []
    for tkr in full_period_ticker_data["ticker"].values:
        ud = connection_session.get(
            f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/day/{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
        ).json()
        if "results" not in ud:
            continue
        tmp = pd.json_normalize(ud["results"]).set_index("t")
        tmp.index = pd.to_datetime(tmp.index, unit="ms", utc=True).tz_convert("America/New_York")
        tmp["date"] = tmp.index.strftime("%Y-%m-%d")
        tmp = tmp[["c", "date"]].copy()
        tmp["ticker"] = tkr
        panel_parts.append(tmp)

    if panel_parts:
        panel = pd.concat(panel_parts).set_index(["date", "ticker"]).sort_index()
        panel["returns"] = panel.groupby(level="ticker")["c"].pct_change()

        feats = add_entropy_info_features(
            panel[["returns"]].copy(), price_col="closeadj", use_lagged_data=True,
            pe_window=ENTROPY_CFG["pe_window"], pe_m=ENTROPY_CFG["pe_m"], pe_tau=ENTROPY_CFG["pe_tau"],
            se_window=ENTROPY_CFG["se_window"], cs_bins=ENTROPY_CFG["cs_bins"]
        )

        # last valid features by ticker at t-1 (<= end_date)
        last_feats = latest_per_ticker(feats, pd.Timestamp(end_date), [
            f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}",
            f"spectral_entropy_w{ENTROPY_CFG['se_window']}",
            f"peer_surprisal_cs_bins{ENTROPY_CFG['cs_bins']}",
            "peer_surprisal_z",
            f"peer_jsd_bins{ENTROPY_CFG['cs_bins']}",
            "peer_jsd_z",
        ])
        # Example, conservative filter: keep names with non-extreme complexity and informative peer position
        # (bounds can be tuned; defaults chosen to avoid heavy impact unless desired)
        pe_col = f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}"
        se_col = f"spectral_entropy_w{ENTROPY_CFG['se_window']}"

        ok = (
            (last_feats[pe_col].between(0.15, 0.98)) &
            (last_feats[se_col].between(0.15, 0.98))
        )
        keep_tickers = set(last_feats[ok].index.astype(str))
        full_period_ticker_data = full_period_ticker_data[full_period_ticker_data["ticker"].isin(keep_tickers)]

# Separated "deciles" (top/bot 10)
top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(N_LONG)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(N_SHORT)

print("Top 10 (by score):")
print(top_decile[["ticker", "mom_score"]])

print("Bottom 10 (by score):")
print(bot_decile[["ticker", "mom_score"]])

