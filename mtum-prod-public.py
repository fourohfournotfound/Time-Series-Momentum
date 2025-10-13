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
from functools import lru_cache
from pathlib import Path

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
PICKLE_FALLBACK_PATH = os.getenv("PIT_UNIVERSE_PICKLE", "historical_liquid_tickers.pkl")
PRICE_CACHE_DIR = Path(os.getenv("MTUM_PRICE_CACHE_DIR", "data_cache"))
PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX_AGE_DAYS = int(os.getenv("MTUM_CACHE_MAX_AGE_DAYS", "3"))


# =============================================================================
# Utilities
# =============================================================================

nyse = get_calendar("NYSE")

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100, pool_maxsize=500, max_retries=3
)
session.mount("https://", adapter)
session.headers.update({"Accept": "application/json"})


def _calendar_bound(ts: datetime) -> pd.Timestamp:
    """Convert arbitrary timestamp to NYSE calendar input (tz-naive New York)."""

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp
    return stamp.tz_convert("America/New_York").tz_localize(None)


@lru_cache(maxsize=512)
def trading_days(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    schedule = nyse.schedule(
        start_date=_calendar_bound(start_date),
        end_date=_calendar_bound(end_date),
    )
    index = schedule.index
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")
    return index


def _get(url: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _cache_path(ticker: str) -> Path:
    return PRICE_CACHE_DIR / f"{ticker.upper()}_daily.pkl"


def _cache_is_fresh(path: Path) -> bool:
    if CACHE_MAX_AGE_DAYS < 0:
        return True
    max_age = timedelta(days=CACHE_MAX_AGE_DAYS)
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - modified <= max_age


def _load_price_history(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    ticker = ticker.upper()
    as_of_raw = pd.Timestamp(as_of_date)
    if as_of_raw.tzinfo is None:
        as_of_local = as_of_raw.tz_localize("America/New_York")
    else:
        as_of_local = as_of_raw.tz_convert("America/New_York")
    as_of_date = as_of_local.tz_localize(None)
    cache_path = _cache_path(ticker)
    base_df: pd.DataFrame | None = None

    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if not cached.empty:
                cached.index = pd.to_datetime(cached.index)
                base_df = cached.sort_index()
        except Exception as exc:
            print(f"[WARN] Failed to read cache for {ticker}: {exc}")

    if base_df is not None:
        last_cached = pd.Timestamp(base_df.index.max()) if not base_df.empty else None
        if last_cached is not None and last_cached >= as_of_date and _cache_is_fresh(cache_path):
            return base_df
        fetch_start = (
            (last_cached + timedelta(days=1)) if last_cached is not None else as_of_date - timedelta(days=365 + 90)
        )
    else:
        fetch_start = as_of_date - timedelta(days=365 + 90)
        base_df = pd.DataFrame()

    fetch_start_str = pd.Timestamp(fetch_start).strftime("%Y-%m-%d")
    fetch_end_str = as_of_date.strftime("%Y-%m-%d")
    if pd.Timestamp(fetch_start_str) <= pd.Timestamp(fetch_end_str):
        resp = _get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{fetch_start_str}/{fetch_end_str}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        payload = pd.json_normalize(resp.get("results", []))
        if payload.empty:
            new_df = pd.DataFrame()
        else:
            payload = payload.drop_duplicates(subset=["t"], keep="last")
            timestamps = pd.to_datetime(payload["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
            payload["date"] = timestamps.dt.tz_localize(None).normalize()
            new_df = (
                payload.dropna(subset=["date"])
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")[["o", "h", "l", "c", "v", "vw"]]
            )
            new_df.index = pd.to_datetime(new_df.index)
    else:
        new_df = pd.DataFrame()

    combined = pd.concat([base_df, new_df], axis=0)
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    try:
        combined.to_pickle(cache_path)
    except Exception as exc:  # pragma: no cover - file guardrail
        print(f"[WARN] Failed to update price cache for {ticker}: {exc}")
    return combined.loc[combined.index <= as_of_date].copy()


def _ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise PIT universe columns for backwards compatibility."""

    df = df.copy()
    drop_candidates = [c for c in df.columns if c.startswith("Unnamed")] + ["index"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

    if "rebalance_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "rebalance_date"})

    if "ticker" not in df.columns:
        raise AssertionError("Universe missing required column: 'ticker'")

    def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("America/New_York")
        return ts.tz_convert("America/New_York")

    @lru_cache(maxsize=1024)
    def _previous_trading_day(ts_key: str) -> pd.Timestamp:
        ts = pd.Timestamp(ts_key)
        window = trading_days(ts - timedelta(days=7), ts)
        past = window[window < ts]
        if len(past) == 0:
            raise RuntimeError(f"No prior trading day found for {ts}")
        return past.max()

    if "rebalance_date" not in df.columns:
        raise AssertionError("Universe missing required column: 'rebalance_date'")

    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])

    if "as_of_date" not in df.columns:
        df["as_of_date"] = df["rebalance_date"].apply(
            lambda x: _previous_trading_day(str(_to_ny(x))).tz_localize(None)
        )
    else:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    required_cols = {"as_of_date", "rebalance_date", "ticker"}
    if not required_cols.issubset(df.columns):
        raise AssertionError(
            f"Universe missing required columns: {required_cols - set(df.columns)}"
        )

    return df


def _load_universe() -> pd.DataFrame:
    """Load the PIT universe, preferring MySQL but falling back to CSV."""

    loaded_from_csv = False
    try:
        df = pd.read_sql("SELECT * FROM historical_liquid_tickers_polygon", con=DB_ENGINE)
        print("Loaded PIT universe from MySQL database.")
    except Exception as exc:  # pragma: no cover - DB guardrail
        print(f"[WARN] Unable to load universe from MySQL ({exc}). Trying local cache...")
        df = None
        if os.path.exists(PICKLE_FALLBACK_PATH):
            try:
                df = pd.read_pickle(PICKLE_FALLBACK_PATH)
                print(f"Loaded PIT universe from {PICKLE_FALLBACK_PATH}.")
            except Exception as pickle_exc:
                print(f"[WARN] Failed to load pickle fallback ({pickle_exc}).")
        if df is None:
            if not os.path.exists(CSV_FALLBACK_PATH):
                raise RuntimeError(
                    "No PIT universe available: database unreachable and local fallbacks missing."
                ) from exc
            df = pd.read_csv(CSV_FALLBACK_PATH)
            print(f"Loaded PIT universe from CSV fallback at {CSV_FALLBACK_PATH}.")
            loaded_from_csv = True

    df = _ensure_universe_schema(df)

    if loaded_from_csv:
        try:
            df.to_csv(CSV_FALLBACK_PATH, index=False)
            df.to_pickle(PICKLE_FALLBACK_PATH)
            print("Normalised PIT universe cached locally with latest schema.")
        except Exception as cache_exc:  # pragma: no cover - file guardrail
            print(f"[WARN] Failed to refresh PIT universe cache: {cache_exc}")

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


def _compute_features_for_ticker(
    ticker: str,
    as_of_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    history = _load_price_history(ticker, as_of_date)
    if history.empty:
        return pd.DataFrame()

    history = history.loc[history.index <= as_of_date].copy()
    history["date"] = history.index.strftime("%Y-%m-%d")

    skip_cutoff = (as_of_date - timedelta(days=30)).strftime("%Y-%m-%d")
    window = history[history["date"] < skip_cutoff].tail(252).copy()
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

spy_history = _load_price_history("SPY", as_of_date)
if spy_history.empty:
    raise RuntimeError("Unable to load SPY history for benchmarking.")
spy_history = spy_history.loc[spy_history.index <= as_of_date].copy()
spy_history["date"] = spy_history.index.strftime("%Y-%m-%d")

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

