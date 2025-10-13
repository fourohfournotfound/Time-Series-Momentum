# -*- coding: utf-8 -*-
"""
Real-time monthly momentum basket with optional volatility/EWMA/entropy filters.
“Classic” reproduces QG math exactly (same rounding/merges/windows).

Speed-ups:
  • ThreadPool across tickers with a single requests.Session + retries.
  • Minimal allocations; vectorized calculations.
"""

import os, time
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlalchemy
import mysql.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

from relative_market_features import add_entropy_info_features, ewm_moments, latest_per_ticker

# -------------------- CONFIG --------------------
polygon_api_key = os.getenv("POLYGON_API_KEY", "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3")
engine = sqlalchemy.create_engine('mysql+mysqlconnector://user:pass@localhost:3306/my_database')

SCORING_MODE = os.getenv("SCORING_MODE", "classic")  # "classic","vol_scaled","ewma","ewma_vol_scaled"
USE_ENTROPY_FILTER = bool(int(os.getenv("USE_ENTROPY_FILTER", "0")))
ENTROPY_CFG = dict(pe_window=126, pe_m=4, pe_tau=1, se_window=128, cs_bins=21)  # causal defaults
EWM_HALFLIFE = int(os.getenv("EWM_HALFLIFE", "63"))
N_LONG = int(os.getenv("N_LONG", "10"))
N_SHORT = int(os.getenv("N_SHORT", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS_PROD", "8"))
TIMEOUT = (5, 30)
USER_AGENT = "mtum-prod/2.0"
# ------------------------------------------------

# Dates (latest month)
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date=(datetime.today() - timedelta(days=45)), end_date=(datetime.today())).index
months = trading_dates.to_series().dt.tz_localize(None).dt.normalize().drop_duplicates(keep="first").dt.strftime("%Y-%m-%d").values
month = months[-1]
start_date = (pd.to_datetime(month) - timedelta(days=365 + 60)).strftime("%Y-%m-%d")
end_date = month
last_month_date = (pd.to_datetime(month) - timedelta(days=30)).strftime("%Y-%m-%d")
next_month_date = (pd.to_datetime(month) + timedelta(days=30)).strftime("%Y-%m-%d")

# Universe + SPY
universe = pd.read_sql("historical_liquid_tickers_polygon", con=engine)
def _build_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=MAX_WORKERS * 2))
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s
SESSION = _build_session()

def _get_json(url: str, params: dict | None = None) -> dict:
    params = params or {}
    try:
        r = SESSION.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                time.sleep(float(ra)); r = SESSION.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"HTTP error: {e} | URL={url} | params={params}")
        return {}

spy = _get_json(
    f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{trading_dates[-1].strftime('%Y-%m-%d')}",
    {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key}
)
benchmark_data = pd.json_normalize(spy.get("results", []))
if benchmark_data.empty:
    raise RuntimeError("Failed to fetch SPY benchmark.")
benchmark_data = benchmark_data.set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["date"] = benchmark_data.index.strftime("%Y-%m-%d")

point_in_time_dates = np.sort(universe[universe["date"] <= month]["date"].drop_duplicates().values)
point_in_time_date = point_in_time_dates[-1]
tickers = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep="last")["ticker"].values

def _score_ticker(ticker: str) -> pd.DataFrame:
    try:
        ud = _get_json(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{next_month_date}",
            {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key}
        )
        rows = ud.get("results", [])
        if not rows:
            return pd.DataFrame([])
        df = pd.json_normalize(rows).set_index("t")
        df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
        df["date"] = df.index.strftime("%Y-%m-%d")

        tmo = df[df["date"] < last_month_date].copy().tail(252)
        if tmo.shape[0] < 252:
            return pd.DataFrame([])

        bmerge = pd.merge(benchmark_data[["c", "date"]], tmo[["c", "date"]], on="date")
        bmerge["benchmark_pct_change"] = np.round(bmerge["c_x"].pct_change() * 100, 2).fillna(0)
        bmerge["ticker_pct_change"]    = np.round(bmerge["c_y"].pct_change() * 100, 2).fillna(0)

        cov = np.cov(bmerge["ticker_pct_change"], bmerge["benchmark_pct_change"])
        var_b = np.var(bmerge["benchmark_pct_change"])
        beta = (cov[0, 1] / var_b) if var_b != 0 else 0.0

        tkr_ret_period = np.round(((tmo["c"].iloc[-1] - tmo["c"].iloc[0]) / tmo["c"].iloc[0]) * 100, 2)
        std_daily = bmerge["ticker_pct_change"].std() * np.sqrt(252)
        sharpe = (tkr_ret_period / std_daily) if std_daily != 0 else 0.0

        if SCORING_MODE == "classic":
            mom_score = beta * sharpe
        else:
            # Optional EWMA modes (causal)
            if SCORING_MODE in {"ewma", "ewma_vol_scaled"}:
                mi = tmo[["c", "date"]].copy(); mi["ticker"] = ticker
                mi = mi.set_index(["date", "ticker"]).sort_index()
                mi["returns"] = mi.groupby(level="ticker")["c"].pct_change()
                em = ewm_moments(mi, ret_col="returns", halflife=EWM_HALFLIFE, use_lagged_data=True).groupby(level="ticker").tail(1)
                ewm_mean = float(em["ewm_mean"].iloc[0]) * 100.0
                ewm_vol  = float(em["ewm_vol_ann"].iloc[0])
                mom_score = 0.0 if ewm_vol == 0 else beta * (ewm_mean / ewm_vol)
            else:
                # vol_scaled falls back to Sharpe proxy with realized vol
                mom_score = beta * (tkr_ret_period / (std_daily if std_daily != 0 else 1e-12))

        fwd = df[(df["date"] >= end_date) & (df["date"] <= next_month_date)].copy().sort_index()
        fwd_ret = np.round(((fwd["c"].iloc[-1] - fwd["c"].iloc[0]) / fwd["c"].iloc[0]) * 100, 2) if fwd.shape[0] >= 2 else np.nan

        return pd.DataFrame([{
            "entry_date": month, "ticker": ticker, "beta": beta, "sharpe": sharpe,
            "12-1_return": tkr_ret_period, "mom_score": mom_score, "forward_returns": fwd_ret
        }])
    except Exception as e:
        print(f"score err {ticker}: {e}")
        return pd.DataFrame([])

rows = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(_score_ticker, t) for t in tickers]
    for f in as_completed(futs):
        df = f.result()
        if not df.empty:
            rows.append(df)

full_period_ticker_data = pd.concat(rows) if rows else pd.DataFrame(columns=[
    "entry_date","ticker","beta","sharpe","12-1_return","mom_score","forward_returns"
])

# Optional entropy-based filter (causal)
if USE_ENTROPY_FILTER and not full_period_ticker_data.empty:
    parts = []
    for tkr in full_period_ticker_data["ticker"].values:
        ud = _get_json(
            f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/day/{start_date}/{end_date}",
            {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_api_key}
        )
        rs = ud.get("results", [])
        if not rs: 
            continue
        tmp = pd.json_normalize(rs).set_index("t")
        tmp.index = pd.to_datetime(tmp.index, unit="ms", utc=True).tz_convert("America/New_York")
        tmp["date"] = tmp.index.strftime("%Y-%m-%d")
        tmp = tmp[["c", "date"]].copy(); tmp["ticker"] = tkr
        parts.append(tmp)
    if parts:
        panel = pd.concat(parts).set_index(["date", "ticker"]).sort_index()
        panel["returns"] = panel.groupby(level="ticker")["c"].pct_change()
        feats = add_entropy_info_features(panel[["returns"]], price_col="closeadj", use_lagged_data=True,
                                          pe_window=ENTROPY_CFG["pe_window"], pe_m=ENTROPY_CFG["pe_m"], pe_tau=ENTROPY_CFG["pe_tau"],
                                          se_window=ENTROPY_CFG["se_window"], cs_bins=ENTROPY_CFG["cs_bins"])
        last_feats = latest_per_ticker(
            feats, pd.Timestamp(end_date), [
                f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}",
                f"spectral_entropy_w{ENTROPY_CFG['se_window']}",
                f"peer_surprisal_cs_bins{ENTROPY_CFG['cs_bins']}",
                "peer_surprisal_z", f"peer_jsd_bins{ENTROPY_CFG['cs_bins']}", "peer_jsd_z"
            ]
        )
        pe_col = f"perm_entropy_w{ENTROPY_CFG['pe_window']}_m{ENTROPY_CFG['pe_m']}_tau{ENTROPY_CFG['pe_tau']}"
        se_col = f"spectral_entropy_w{ENTROPY_CFG['se_window']}"
        ok = (last_feats[pe_col].between(0.15, 0.98)) & (last_feats[se_col].between(0.15, 0.98))
        keep = set(last_feats[ok].index.astype(str))
        full_period_ticker_data = full_period_ticker_data[full_period_ticker_data["ticker"].isin(keep)]

top_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=False).head(N_LONG)
bot_decile = full_period_ticker_data.sort_values(by="mom_score", ascending=True).head(N_SHORT)

print("Top 10:")
print(top_decile[["ticker", "mom_score"]])
print("Bottom 10:")
print(bot_decile[["ticker", "mom_score"]])

