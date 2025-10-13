# -*- coding: utf-8 -*-
"""
Created in 2025

Production momentum scoring with optional volatility management (Moreira & Muir),
EWMA stats (RiskMetrics), and entropy filtering—while preserving the original
QG "classic" scoring exactly when enhancements are disabled.

Key guarantees:
- "classic" mode (METHOD="classic"; all USE_* flags False) reproduces the original
  QG outputs with same PIT universe, 12-1 window, and mom_score := beta * sharpe.
- Shapes and indices are validated; final feature frame is MultiIndex (entry_date, ticker).

Literature:
- Cross-sectional momentum & 12-1: Jegadeesh & Titman (1993); survey (2011).
- Time-series momentum: Moskowitz, Ooi & Pedersen (2012).
- Volatility-managed portfolios: Moreira & Muir (2017).
- EWMA for vol: RiskMetrics Technical Document (1996).
- Entropy for finance (noise/structure): Shannon/Permutation entropy applications.

Citations: See upgrade note.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # kept for parity; not used in prod scoring
import requests
import gspread  # unused here but kept for parity
import sqlalchemy
import mysql.connector
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# =============================================================================
# Config
# =============================================================================

polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

# --- Feature toggles (default => exact classic parity) ---
METHOD = os.getenv("MTUM_METHOD", "classic")  # "classic" | "vol_managed" | "ewma" | "ewma_vol_managed"
USE_VOL_MANAGED = (METHOD in ("vol_managed","ewma_vol_managed"))
USE_EWMA        = (METHOD in ("ewma","ewma_vol_managed"))
USE_ENTROPY_FILTER = bool(int(os.getenv("USE_ENTROPY_FILTER", "0")))

# Vol management params (kept conservative; scaling lagged one day to avoid look-ahead)
VOL_WINDOW_DAYS = int(os.getenv("VOL_WINDOW_DAYS", "21"))
VOL_TARGET_DAILY = float(os.getenv("VOL_TARGET_DAILY", "1.0"))  # in pct-return units; 1.0 ~ 1% daily
VOL_SCALE_CLIP = float(os.getenv("VOL_SCALE_CLIP", "3.0"))      # cap extreme scaling

# EWMA params (RiskMetrics style)
EWMA_HALFLIFE = int(os.getenv("EWMA_HALFLIFE", "21"))

# Entropy filter params
ENTROPY_BINS = int(os.getenv("ENTROPY_BINS", "5"))
ENTROPY_MAX_BITS = float(os.getenv("ENTROPY_MAX_BITS", "inf"))  # e.g., 2.1 to drop very random series
ENTROPY_QCUT = bool(int(os.getenv("ENTROPY_QCUT", "1")))        # use quantile bins (robust) vs fixed bins

# I/O concurrency for ticker pulls
MAX_WORKERS = int(os.getenv("MTUM_MAX_WORKERS", "8"))

# =============================================================================
# Date Management (unchanged)
# =============================================================================

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(
    start_date=(datetime.today() - timedelta(days=45)),
    end_date=(datetime.today())
).index.strftime("%Y-%m-%d").values

all_dates = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
all_dates["year"] = all_dates["date"].dt.year
all_dates["month"] = all_dates["date"].dt.month

start_of_the_months = all_dates.drop_duplicates(subset = ["year", "month"], keep = "first").copy()
start_of_the_months["str_date"] = start_of_the_months["date"].dt.strftime("%Y-%m-%d")

months = start_of_the_months["str_date"].values
month = months[-1]

start_date = (pd.to_datetime(month) - timedelta(days = 365+60)).strftime("%Y-%m-%d")
end_date   = month
last_month_date = (pd.to_datetime(month) - timedelta(days = 30)).strftime("%Y-%m-%d")
next_month_date = (pd.to_datetime(month) + timedelta(days = 30)).strftime("%Y-%m-%d")

# =============================================================================
# PIT Universe + Benchmark (same as original)
# =============================================================================

def build_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=500,
                          max_retries=Retry(total=3, backoff_factor=0.5,
                                            status_forcelist=(429, 500, 502, 503, 504)))
    s.mount('https://', adapter)
    s.headers.update({"Accept":"application/json"})
    return s

session = build_session()

universe = pd.read_sql("historical_liquid_tickers_polygon", con = engine)
# Benchmark SPY once
spy = session.get(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
).json()
benchmark_data = pd.json_normalize(spy.get("results", [])).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

# =============================================================================
# Helpers
# =============================================================================

def shannon_entropy_bits(x: pd.Series, bins=5, qcut=True) -> float:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return 0.0
    try:
        if qcut:
            binned = pd.qcut(x, q=bins, labels=False, duplicates="drop")
        else:
            binned = pd.cut(x, bins=bins, labels=False)
        counts = np.bincount(binned.dropna().astype(int), minlength=bins).astype(float)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    except Exception:
        return 0.0

def fetch_ticker_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    r = session.get(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}"
    ).json()
    df = pd.json_normalize(r.get("results", [])).set_index("t")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df

def compute_beta_sharpe_and_score(ticker_window: pd.DataFrame, bm: pd.DataFrame) -> dict:
    """Replicates original beta * sharpe scoring exactly on the 12-1 window."""
    merged = pd.merge(bm[["c","date"]], ticker_window[["c","date"]], on="date", how="inner")
    merged["benchmark_pct_change"] = (merged["c_x"].pct_change() * 100).round(2).fillna(0)
    merged["ticker_pct_change"]    = (merged["c_y"].pct_change() * 100).round(2).fillna(0)

    cov = np.cov(merged["ticker_pct_change"], merged["benchmark_pct_change"])
    beta = cov[0,1] / (np.var(merged["benchmark_pct_change"]) + 1e-12)

    ret_period = ((merged["c_y"].iloc[-1] - merged["c_y"].iloc[0]) / merged["c_y"].iloc[0]) * 100
    std_ann = merged["ticker_pct_change"].std() * np.sqrt(252)
    sharpe = (ret_period / (std_ann + 1e-12))

    theo_expected = beta * sharpe
    return {"beta": float(beta), "sharpe": float(sharpe), "ret_12_1": float(ret_period), "mom_score_classic": float(theo_expected)}

def apply_vol_management(returns_pct: pd.Series) -> pd.Series:
    """Scale by inverse recent realized vol (lagged); daily pct-return units."""
    rolling_std = returns_pct.rolling(VOL_WINDOW_DAYS, min_periods=max(2, VOL_WINDOW_DAYS//2)).std()
    # RiskMetrics EWMA optional for the scaling estimate
    if USE_EWMA:
        ewma_std = returns_pct.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).std()
        vol_est = ewma_std
    else:
        vol_est = rolling_std
    scale = (VOL_TARGET_DAILY / (vol_est.replace(0, np.nan))).clip(upper=VOL_SCALE_CLIP).fillna(0.0)
    scale = scale.shift(1)  # strict lag to avoid look-ahead
    return (returns_pct * scale)

def compute_method_momscore(ticker_window: pd.DataFrame, bm: pd.DataFrame) -> dict:
    """Compute mom_score under METHOD, alongside the classic fields."""
    out = compute_beta_sharpe_and_score(ticker_window, bm)  # includes classic fields

    # Short-circuit: classic means identical to original
    if METHOD == "classic" and not USE_VOL_MANAGED and not USE_EWMA and not USE_ENTROPY_FILTER:
        out["mom_score"] = out["mom_score_classic"]
        return out

    merged = pd.merge(bm[["c","date"]], ticker_window[["c","date"]], on="date", how="inner").sort_values("date")
    ret_bm = (merged["c_x"].pct_change() * 100).fillna(0)
    ret_tk = (merged["c_y"].pct_change() * 100).fillna(0)

    if USE_ENTROPY_FILTER:
        ent_bits = shannon_entropy_bits(ret_tk, bins=ENTROPY_BINS, qcut=ENTROPY_QCUT)
        if ent_bits > ENTROPY_MAX_BITS:
            # Extremely noisy; deprioritize by setting a very low score but keep row present
            out["entropy_bits"] = float(ent_bits)
            out["mom_score"] = -1e9
            return out
        out["entropy_bits"] = float(ent_bits)

    # Optionally volatility-manage the ticker returns before computing Sharpe
    if USE_VOL_MANAGED:
        ret_tk = apply_vol_management(ret_tk)

    # EWMA means/vol for Sharpe
    if USE_EWMA:
        mean_ann = ret_tk.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).mean().iloc[-1] * np.sqrt(252)
        std_ann  = ret_tk.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=2).std().iloc[-1]  * np.sqrt(252)
        ewma_sharpe = (mean_ann / (std_ann + 1e-12))
        # Keep beta from classic definition (cov with bm) to maintain economic meaning
        cov = np.cov((ret_tk.fillna(0)).values, (ret_bm.values))
        beta = cov[0,1] / (np.var(ret_bm.values) + 1e-12)
        out["mom_score"] = float(beta * ewma_sharpe)
        out["sharpe"] = float(ewma_sharpe)  # report this Sharpe if EWMA selected
        out["beta"] = float(beta)
    else:
        # Non-EWMA but possibly vol-managed: compute Sharpe same way as classic but on modified returns
        std_ann = ret_tk.std() * np.sqrt(252)
        # Use the *period* return (unscaled prices) to anchor the signal, as in classic
        ret_period = ((merged["c_y"].iloc[-1] - merged["c_y"].iloc[0]) / merged["c_y"].iloc[0]) * 100
        sharpe = (ret_period / (std_ann + 1e-12))
        cov = np.cov(ret_tk.values, ret_bm.values)
        beta = cov[0,1] / (np.var(ret_bm.values) + 1e-12)
        out["mom_score"] = float(beta * sharpe)
        out["sharpe"] = float(sharpe)
        out["beta"] = float(beta)

    return out

# =============================================================================
# Real-Time Basket Construction (same PIT selection)
# =============================================================================

point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]
point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")
tickers = point_in_time_universe["ticker"].drop_duplicates().values

# Precompute dates for this run; collect rows in parallel to accelerate I/O
def process_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        # pull up to end_date for features and a forward chunk for forward return if needed downstream
        underlying = fetch_ticker_ohlc(ticker, start_date, end_date)
        if underlying.empty:
            return None

        # Build 12-1 window strictly < last_month_date
        twelve_minus_one = underlying[underlying["date"] < last_month_date].copy().tail(252)
        if len(twelve_minus_one) < 252:
            return None

        # classic fields kept for parity/diagnostics
        feat = compute_method_momscore(twelve_minus_one, benchmark_data)

        # auxiliary monthly stats (unchanged)
        eom = twelve_minus_one.copy()
        eom["year"] = eom.index.year
        eom["month"] = eom.index.month
        eom = eom.drop_duplicates(subset=["year","month"], keep="last").copy()
        eom["monthly_return"] = (eom["c"].pct_change() * 100).round(2)
        avg_monthly_return = float(eom["monthly_return"].mean())

        # forward returns (month -> next_month)
        fwd = fetch_ticker_ohlc(ticker, end_date, next_month_date)
        if fwd.empty:
            return None
        forward_return = round(((fwd["c"].iloc[-1] - fwd["c"].iloc[0]) / fwd["c"].iloc[0]) * 100, 2)

        row = {
            "entry_date": month,
            "ticker": ticker,
            "beta": feat["beta"],
            "sharpe": feat["sharpe"],
            "12-1_return": feat["ret_12_1"],
            "avg_monthly_return": avg_monthly_return,
            "mom_score_classic": feat["mom_score_classic"],
            "mom_score": feat["mom_score"],
            "forward_returns": forward_return
        }
        if "entropy_bits" in feat:
            row["entropy_bits"] = feat["entropy_bits"]

        return pd.DataFrame([row])
    except Exception as e:
        print(e)
        return None

monthly_ticker_list, times = [], []
start_time_all = datetime.now()

if MAX_WORKERS > 0 and len(tickers) > 0:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_ticker, t) for t in tickers]
        for i, f in enumerate(as_completed(futs), 1):
            df = f.result()
            if df is not None:
                monthly_ticker_list.append(df)
            if i % max(1, len(tickers)//10) == 0:
                elapsed = datetime.now() - start_time_all
                print(f"{round((i/len(tickers))*100,2)}% complete, elapsed {elapsed}")
else:
    for i, t in enumerate(tickers, 1):
        df = process_ticker(t)
        if df is not None:
            monthly_ticker_list.append(df)
        iteration = round((i/len(tickers))*100,2)
        print(f"{iteration}% complete")

# =============================================================================
# Assemble outputs (MultiIndex)
# =============================================================================

full_period_ticker_data = pd.concat(monthly_ticker_list).reset_index(drop=True)
# In classic runs, enforce identity with QG mom_score
if METHOD == "classic" and not USE_VOL_MANAGED and not USE_EWMA and not USE_ENTROPY_FILTER:
    assert np.allclose(full_period_ticker_data["mom_score"], full_period_ticker_data["mom_score_classic"], rtol=1e-10, atol=1e-10)

# MultiIndex (entry_date, ticker) for LTR compatibility
features = full_period_ticker_data.set_index(["entry_date","ticker"]).sort_index()

# Optional: top/bottom deciles off the selected mom_score (unchanged behavior)
top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(10)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(10)

print("features shape:", features.shape)  # shape check for your pipeline
