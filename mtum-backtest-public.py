# -*- coding: utf-8 -*-
"""Monthly momentum backtest with point-in-time safeguards.

This backtest mirrors the production feature pipeline by relying solely on the
universe information known at ``as_of_date`` (the trading day before the first
trading day of the month).  Portfolios are entered at the **next trading day
open** following the rebalance date and exited at the **next trading day open**
after the subsequent rebalance, ensuring signal availability precedes trade
execution.

The script also fixes the previous runtime bug encountered when the MySQL
database is unavailable by falling back to the on-disk CSV universe cache.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
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
# Calendar helpers
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


def next_trading_day(ts: pd.Timestamp) -> pd.Timestamp:
    window = trading_days(ts - timedelta(days=3), ts + timedelta(days=7))
    future = window[window > ts]
    if len(future) == 0:
        raise RuntimeError(f"No trading day found after {ts}")
    return future.min()


def _get(url: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# =============================================================================
# Universe loading
# =============================================================================

def _ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise PIT universe columns for backwards compatibility."""

    df = df.copy()

    # Drop artefact columns from older CSV exports (e.g. pandas index column).
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

    def _previous_trading_day(ts: pd.Timestamp) -> pd.Timestamp:
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
            lambda x: _previous_trading_day(_to_ny(x)).tz_localize(None)
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
    loaded_from_csv = False
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
        loaded_from_csv = True

    df = _ensure_universe_schema(df)

    if loaded_from_csv:
        try:
            df.to_csv(CSV_FALLBACK_PATH, index=False)
            print("Normalised PIT universe cached back to CSV with latest schema.")
        except Exception as cache_exc:  # pragma: no cover - file guardrail
            print(f"[WARN] Failed to refresh PIT universe CSV: {cache_exc}")

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df.drop_duplicates(subset=["as_of_date", "rebalance_date", "ticker"], keep="last")


universe = _load_universe()

start_backtest = pd.to_datetime("2019-02-01")
end_backtest = datetime.now(timezone.utc).astimezone() - timedelta(days=1)

months = (
    universe[
        (universe["rebalance_date"] >= start_backtest)
        & (universe["rebalance_date"] <= end_backtest)
    ]["rebalance_date"].drop_duplicates().sort_values().to_numpy()
)

if months.size < 2:
    raise RuntimeError("Not enough monthly rebalances available for backtest.")


# =============================================================================
# Data helpers
# =============================================================================

HISTORY_START = os.getenv("MTUM_HISTORY_START", "2016-01-01")
_history_end_env = os.getenv("MTUM_HISTORY_END")
if _history_end_env:
    HISTORY_END = _history_end_env
else:
    _last_month = pd.Timestamp(months[-1])
    _history_end_dt = next_trading_day(_last_month)
    if _history_end_dt.tzinfo is not None:
        _history_end_dt = _history_end_dt.tz_convert("America/New_York").tz_localize(None)
    else:
        _history_end_dt = pd.Timestamp(_history_end_dt)
    HISTORY_END = (_history_end_dt + timedelta(days=7)).strftime("%Y-%m-%d")

PRICE_CACHE: dict[str, pd.DataFrame] = {}
FORWARD_RETURN_CACHE: dict[tuple[str, str, str], float] = {}
FEATURE_CACHE: dict[tuple[str, str], dict] = {}


def _price_history(ticker: str) -> pd.DataFrame:
    key = ticker.upper()
    cached = PRICE_CACHE.get(key)
    if cached is not None:
        return cached

    resp = _get(
        f"https://api.polygon.io/v2/aggs/ticker/{key}/range/1/day/{HISTORY_START}/{HISTORY_END}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    raw = pd.json_normalize(resp.get("results", []))
    if raw.empty:
        df = pd.DataFrame(columns=["o", "c", "h", "l", "v", "vw"])
    else:
        raw = raw.drop_duplicates(subset=["t"], keep="last")
        timestamps = pd.to_datetime(raw["t"], unit="ms", utc=True).tz_convert(
            "America/New_York"
        )
        raw["date"] = timestamps.tz_localize(None).normalize()
        df = (
            raw.dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")[["o", "c", "h", "l", "v", "vw"]]
        )
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

    PRICE_CACHE[key] = df
    return df


SPY = _price_history("SPY")
if SPY.empty:
    raise RuntimeError("Unable to load SPY history for benchmarking.")


def compute_features(ticker: str, as_of_date: pd.Timestamp) -> dict:
    cache_key = (ticker, pd.Timestamp(as_of_date).strftime("%Y-%m-%d"))
    if cache_key in FEATURE_CACHE:
        return FEATURE_CACHE[cache_key]

    history = _price_history(ticker)
    if history.empty:
        FEATURE_CACHE[cache_key] = {}
        return {}

    as_of_date = pd.Timestamp(as_of_date)
    history = history.loc[history.index <= as_of_date]
    if history.empty:
        FEATURE_CACHE[cache_key] = {}
        return {}

    skip_cutoff = as_of_date - timedelta(days=30)
    window = history.loc[history.index < skip_cutoff].tail(252).copy()
    if window.shape[0] < 252:
        FEATURE_CACHE[cache_key] = {}
        return {}

    spy_window = SPY.loc[SPY.index <= skip_cutoff][["c"]]
    spy_window = spy_window.loc[spy_window.index >= window.index.min()].copy()
    if spy_window.empty:
        FEATURE_CACHE[cache_key] = {}
        return {}

    merged = spy_window.rename(columns={"c": "c_m"}).join(
        window[["c"]].rename(columns={"c": "c_s"}), how="inner"
    )
    if merged.shape[0] < 200:
        FEATURE_CACHE[cache_key] = {}
        return {}

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
    vol_scaled = mom_12_1 / (sigma_ann + 1e-8)

    result = {
        "beta_capm": float(beta),
        "mom_12_1": float(mom_12_1),
        "resid_mom": float(resid_mom) if np.isfinite(resid_mom) else np.nan,
        "sigma_ann": float(sigma_ann),
        "mom_vol_scaled": float(vol_scaled),
    }
    FEATURE_CACHE[cache_key] = result
    return result


def forward_open_to_open(ticker: str, entry_rebalance: pd.Timestamp, exit_rebalance: pd.Timestamp) -> float:
    entry_rebalance = pd.Timestamp(entry_rebalance)
    exit_rebalance = pd.Timestamp(exit_rebalance)

    cache_key = (
        ticker,
        entry_rebalance.strftime("%Y-%m-%d"),
        exit_rebalance.strftime("%Y-%m-%d"),
    )
    if cache_key in FORWARD_RETURN_CACHE:
        return FORWARD_RETURN_CACHE[cache_key]

    history = _price_history(ticker)
    if history.empty:
        FORWARD_RETURN_CACHE[cache_key] = np.nan
        return np.nan

    def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("America/New_York")
        return ts.tz_convert("America/New_York")

    def _normalize(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        return ts.normalize()

    entry_trade_day = _normalize(next_trading_day(_to_ny(entry_rebalance)))
    exit_trade_day = _normalize(next_trading_day(_to_ny(exit_rebalance)))
    entry_rebalance_day = _normalize(_to_ny(entry_rebalance))
    exit_rebalance_day = _normalize(_to_ny(exit_rebalance))

    try:
        price_entry = float(history.at[entry_trade_day, "o"])
        price_exit = float(history.at[exit_trade_day, "o"])
    except KeyError:
        try:
            price_entry = float(history.at[entry_rebalance_day, "c"])
            price_exit = float(history.at[exit_rebalance_day, "c"])
        except KeyError:
            FORWARD_RETURN_CACHE[cache_key] = np.nan
            return np.nan

    forward_return = float((price_exit / price_entry) - 1.0) * 100.0
    FORWARD_RETURN_CACHE[cache_key] = forward_return
    return forward_return


# =============================================================================
# Backtest loop
# =============================================================================

feature_panels: list[pd.DataFrame] = []
monthly_returns: list[dict] = []

for month_value in months[:-1]:
    month_ts = pd.Timestamp(month_value)
    next_month_idx = np.where(months > month_value)[0]
    if len(next_month_idx) == 0:
        break
    next_month_ts = pd.Timestamp(months[next_month_idx[0]])

    month_universe = universe[universe["rebalance_date"] == month_ts]
    if month_universe.empty:
        continue
    as_of_date = pd.Timestamp(month_universe["as_of_date"].iloc[0])

    feature_rows = []
    for ticker in month_universe["ticker"].drop_duplicates():
        feats = compute_features(ticker, as_of_date)
        if not feats:
            continue
        feature_rows.append(
            {
                "entry_date": month_ts.strftime("%Y-%m-%d"),
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                **feats,
            }
        )

    if not feature_rows:
        continue

    feature_df = pd.DataFrame(feature_rows).set_index(["entry_date", "ticker"])
    feature_df["rank_score"] = feature_df["mom_vol_scaled"].where(
        np.isfinite(feature_df["mom_vol_scaled"]),
        np.nan,
    )
    feature_df["rank_score"].fillna(feature_df["resid_mom"], inplace=True)
    feature_df["rank_score"].fillna(feature_df["mom_12_1"], inplace=True)
    feature_df.sort_values("rank_score", ascending=False, inplace=True)

    top_decile = feature_df.groupby(level=0).head(10).reset_index()
    bottom_decile = feature_df.groupby(level=0).tail(10).reset_index()

    spy_forward = forward_open_to_open("SPY", month_ts, next_month_ts)
    forward_rows = []

    for side, data in (("long", top_decile), ("short", bottom_decile)):
        for _, row in data.iterrows():
            fwd = forward_open_to_open(row["ticker"], month_ts, next_month_ts)
            if np.isnan(fwd):
                continue
            beta = feature_df.loc[(row["entry_date"], row["ticker"]), "beta_capm"]
            risk_adj = fwd - beta * spy_forward
            forward_rows.append(
                {
                    "date": row["entry_date"],
                    "ticker": row["ticker"],
                    "side": side,
                    "forward_return_pct": fwd,
                    "risk_adj_forward_pct": risk_adj,
                }
            )

    if not forward_rows:
        continue

    forward_df = pd.DataFrame(forward_rows)
    long_mean = forward_df[forward_df["side"] == "long"]["risk_adj_forward_pct"].mean()
    short_mean = -forward_df[forward_df["side"] == "short"]["risk_adj_forward_pct"].mean()

    monthly_returns.append(
        {
            "date": month_ts.strftime("%Y-%m-%d"),
            "long": long_mean,
            "short": short_mean,
        }
    )

    feature_panels.append(feature_df.reset_index())


if not monthly_returns:
    raise RuntimeError("Backtest produced no tradable months.")

features_panel = pd.concat(feature_panels, ignore_index=True).set_index(["entry_date", "ticker"]).sort_index()
monthly_returns_df = pd.DataFrame(monthly_returns).sort_values("date")

monthly_returns_df["long_pnl"] = monthly_returns_df["long"].cumsum()
monthly_returns_df["short_pnl"] = monthly_returns_df["short"].cumsum()
monthly_returns_df["long_short_pnl"] = monthly_returns_df["long_pnl"] + monthly_returns_df["short_pnl"]

print("Backtest months:", monthly_returns_df.shape[0])
print(monthly_returns_df.tail())


# =============================================================================
# Plot performance
# =============================================================================

plt.figure(figsize=(10, 6), dpi=200)
plt.suptitle("Gross Cumulative Performance (Risk-Adjusted, Open→Open)")
plt.plot(pd.to_datetime(monthly_returns_df["date"]), monthly_returns_df["long_pnl"], label="Top Decile")
plt.plot(pd.to_datetime(monthly_returns_df["date"]), monthly_returns_df["short_pnl"], label="Bottom Decile")
plt.plot(
    pd.to_datetime(monthly_returns_df["date"]),
    monthly_returns_df["long_short_pnl"],
    label="Long-Short",
)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Cumulative % Returns")
plt.legend()
plt.tight_layout()
plt.show()
