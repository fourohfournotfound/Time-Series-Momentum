# -*- coding: utf-8 -*-
"""Production momentum feature generation (point-in-time safe).

The script loads the latest tradable universe produced by
``point-in-time-options.py`` and computes momentum features that are available
at the close of ``as_of_date``.  No forward returns or any data beyond this
timestamp are accessed, ensuring production scoring mirrors the information set
used during backtesting.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import mysql.connector
import numpy as np
import pandas as pd
import requests
import sqlalchemy
from pandas_market_calendars import get_calendar


POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
DB_ENGINE = sqlalchemy.create_engine(
    "mysql+mysqlconnector://user:pass@localhost:3306/my_database"
)
CSV_FALLBACK_PATH = os.getenv("PIT_UNIVERSE_CSV", "historical_liquid_tickers.csv")


# =============================================================================
# Utilities
# =============================================================================

nyse = get_calendar("NYSE")


def trading_days(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.tz_convert("America/New_York")


def _get(url: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _load_universe() -> pd.DataFrame:
    """Load the PIT universe, preferring MySQL but falling back to CSV."""

    try:
        df = pd.read_sql("SELECT * FROM historical_liquid_tickers_polygon", con=DB_ENGINE)
        print("Loaded PIT universe from MySQL database.")
    except Exception as exc:  # pragma: no cover - DB guardrail
        print(f"[WARN] Unable to load universe from MySQL ({exc}). Trying CSV fallback...")
        if not os.path.exists(CSV_FALLBACK_PATH):
            raise RuntimeError(
                "No PIT universe available: database unreachable and CSV fallback missing."
            ) from exc
        df = pd.read_csv(CSV_FALLBACK_PATH)
        print(f"Loaded PIT universe from CSV fallback at {CSV_FALLBACK_PATH}.")

    required_cols = {"as_of_date", "rebalance_date", "ticker"}
    if not required_cols.issubset(df.columns):
        raise AssertionError(
            f"Universe missing required columns: {required_cols - set(df.columns)}"
        )

    return df.drop_duplicates(subset=["as_of_date", "rebalance_date", "ticker"], keep="last")


def _latest_rebalance(universe: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    universe["rebalance_date"] = pd.to_datetime(universe["rebalance_date"])
    universe["as_of_date"] = pd.to_datetime(universe["as_of_date"])

    today_local = datetime.now(timezone.utc).astimezone()
    td = trading_days(today_local - timedelta(days=30), today_local)
    max_allowed = td.max()

    valid_rebals = np.sort(universe[universe["rebalance_date"] <= max_allowed]["rebalance_date"].unique())
    if len(valid_rebals) == 0:
        raise RuntimeError("No valid rebalance dates found in the universe table.")

    rebalance_date = pd.to_datetime(valid_rebals[-1])
    as_of_date = universe.loc[universe["rebalance_date"] == rebalance_date, "as_of_date"].iloc[0]
    return rebalance_date, as_of_date


def _spy_history(as_of_date: pd.Timestamp) -> pd.DataFrame:
    resp = _get(
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{as_of_date.strftime('%Y-%m-%d')}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    df = pd.json_normalize(resp.get("results", [])).set_index("t")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df


def _compute_features_for_ticker(
    ticker: str,
    as_of_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    start_date = (as_of_date - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    resp = _get(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{as_of_str}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    df = pd.json_normalize(resp.get("results", [])).set_index("t")
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")

    skip_cutoff = (as_of_date - timedelta(days=30)).strftime("%Y-%m-%d")
    window = df[df["date"] < skip_cutoff].tail(252).copy()
    if len(window) < 252:
        return pd.DataFrame()

    merged = pd.merge(
        spy_df[spy_df["date"] <= skip_cutoff][["c", "date"]],
        window[["c", "date"]],
        on="date",
        how="inner",
        suffixes=("_m", "_s"),
    )
    if merged.shape[0] < 200:
        return pd.DataFrame()

    merged["ret_m"] = merged["c_m"].pct_change().fillna(0.0)
    merged["ret_s"] = merged["c_s"].pct_change().fillna(0.0)

    cov = np.cov(merged["ret_s"], merged["ret_m"])
    beta = 0.0 if cov[1, 1] == 0 else cov[0, 1] / cov[1, 1]

    mom_12_1 = (window["c"].iloc[-1] / window["c"].iloc[0]) - 1.0

    X = np.vstack([np.ones(len(merged["ret_m"])), merged["ret_m"].values]).T
    y = merged["ret_s"].values
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ coef
        resid_mom = (np.prod(1.0 + resid) - 1.0)
    except Exception:
        resid_mom = np.nan

    sigma_ann = merged["ret_s"].std() * np.sqrt(252)
    mom_vol_scaled = mom_12_1 / (sigma_ann + 1e-8)

    return pd.DataFrame(
        [
            {
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "beta_capm": float(beta),
                "mom_12_1": float(mom_12_1),
                "resid_mom": float(resid_mom) if np.isfinite(resid_mom) else np.nan,
                "sigma_ann": float(sigma_ann),
                "mom_vol_scaled": float(mom_vol_scaled),
            }
        ]
    )


# =============================================================================
# Main execution
# =============================================================================

universe = _load_universe()
rebalance_date, as_of_date = _latest_rebalance(universe)

mask = pd.to_datetime(universe["rebalance_date"]) == rebalance_date
tickers = universe.loc[mask, "ticker"].drop_duplicates().to_numpy()
if len(tickers) == 0:
    raise RuntimeError(f"Universe for {rebalance_date:%Y-%m-%d} is empty.")

spy_history = _spy_history(as_of_date)

feature_rows: list[pd.DataFrame] = []
for i, ticker in enumerate(tickers, start=1):
    try:
        features = _compute_features_for_ticker(
            ticker,
            as_of_date,
            rebalance_date,
            spy_history,
        )
    except Exception as exc:  # pragma: no cover - network guardrail
        print(f"[WARN] Feature generation failed for {ticker}: {exc}")
        features = pd.DataFrame()
    if not features.empty:
        feature_rows.append(features)
    if i % 100 == 0:
        print(f"Processed {i} / {len(tickers)} tickers")

if not feature_rows:
    raise RuntimeError("No features computed for the latest rebalance date.")

features_df = pd.concat(feature_rows, ignore_index=True)
features_df.set_index(["entry_date", "ticker"], inplace=True)

if not features_df.index.is_unique:
    raise AssertionError("Duplicate (entry_date, ticker) pairs found in features.")

print("Feature dataframe shape:", features_df.shape)
print(features_df.head())

ranked = features_df.copy()
ranked["score"] = ranked["mom_vol_scaled"].where(
    np.isfinite(ranked["mom_vol_scaled"]),
    np.nan,
)
ranked["score"].fillna(ranked["resid_mom"], inplace=True)
ranked["score"].fillna(ranked["mom_12_1"], inplace=True)
ranked.sort_values(by="score", ascending=False, inplace=True)

top_decile = ranked.groupby(level=0).head(10).reset_index()["ticker"].tolist()
bottom_decile = ranked.groupby(level=0).tail(10).reset_index()["ticker"].tolist()

print("Top decile tickers:", top_decile)
print("Bottom decile tickers:", bottom_decile)

