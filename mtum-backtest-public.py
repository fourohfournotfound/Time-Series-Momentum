# -*- coding: utf-8 -*-
"""Momentum backtest with monthly rebalancing.

This version focuses on the public dataset shipped with the repository.  The
script supports the same command line flags as the internal research version so
it can be swapped into existing notebooks and CI jobs.  The defaults reproduce
the ``classic`` QG momentum configuration when a Polygon.io API key and the
historical liquid tickers table are available.

Key characteristics
-------------------
* Point-in-time universe sourced from a database or the bundled CSV backup.
* Vectorised feature pipeline (12-1 momentum, beta, Sharpe, EWMA moments).
* Optional entropy filter that matches the production basket implementation.
* Volatility targeting with multiple estimators (standard, EWMA, trimmed "t").
* Graceful HTTP caching and incremental downloads for Polygon data.
* Summary plot + compounded performance statistics at the end of each run.

The implementation deliberately avoids non-standard dependencies so it can run
in the execution environment used for the public course notebooks.
"""

from __future__ import annotations

import argparse
import math
import os
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sqlalchemy

from data_providers import PolygonAggsClient
from relative_market_features import (
    add_entropy_info_features,
    ewm_moments,
    latest_per_ticker,
    volatility_target_scale,
)


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntropyConfig:
    """Parameters shared with the internal research scripts."""

    pe_window: int = 126
    pe_m: int = 4
    pe_tau: int = 1
    se_window: int = 128
    cs_bins: int = 21


@dataclass(frozen=True)
class BacktestConfig:
    """User supplied options that drive a backtest run."""

    provider: str
    polygon_api_key: Optional[str]
    database_url: Optional[str]
    universe_table: str
    universe_csv: Path
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp]
    cache_dir: Path
    momentum_type: str
    vol_estimator: str
    vol_target: float
    scale_cap: float
    n_long: int
    n_short: int
    use_entropy: bool
    entropy: EntropyConfig
    ewm_halflife: int
    min_history: int
    progress: bool
    output_dir: Path
    benchmark: str
    threads: int
    prefetch: bool


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Monthly rebalanced time-series momentum backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            python mtum-backtest-public.py \\
                --provider polygon \\
                --database-url sqlite:///universe.db \\
                --universe-table historical_liquid_tickers_polygon \\
                --start-date 2010-01-01 \\
                --momentum-type classic \\
                --vol-estimator t \\
                --prefetch
            """
        ),
    )

    parser.add_argument("--provider", choices=["polygon"], default="polygon")
    parser.add_argument("--polygon-api-key", dest="polygon_api_key", default=os.environ.get("POLYGON_API_KEY"))
    parser.add_argument("--database-url", dest="database_url", default=None)
    parser.add_argument("--universe-table", dest="universe_table", default="historical_liquid_tickers_polygon")
    parser.add_argument(
        "--universe-csv",
        dest="universe_csv",
        default="historical_liquid_tickers.csv",
        help="CSV fallback for the point-in-time universe",
    )
    parser.add_argument("--start-date", dest="start_date", type=pd.Timestamp, default=pd.Timestamp("2019-01-01"))
    parser.add_argument("--end-date", dest="end_date", type=lambda s: pd.Timestamp(s) if s else None, default=None)
    parser.add_argument("--cache-dir", dest="cache_dir", default="cache", type=Path)
    parser.add_argument(
        "--momentum-type",
        dest="momentum_type",
        choices=["classic", "vol_scaled", "ewma", "ewma_vol_scaled"],
        default="classic",
    )
    parser.add_argument(
        "--vol-estimator",
        dest="vol_estimator",
        choices=["standard", "ewma", "t"],
        default="standard",
        help="Volatility estimator used for position scaling.",
    )
    parser.add_argument("--vol-target", dest="vol_target", type=float, default=0.20)
    parser.add_argument("--scale-cap", dest="scale_cap", type=float, default=3.0)
    parser.add_argument("--n-long", dest="n_long", type=int, default=10)
    parser.add_argument("--n-short", dest="n_short", type=int, default=10)
    parser.add_argument("--use-entropy", dest="use_entropy", action="store_true")
    parser.add_argument("--entropy-pe-window", dest="pe_window", type=int, default=126)
    parser.add_argument("--entropy-pe-m", dest="pe_m", type=int, default=4)
    parser.add_argument("--entropy-pe-tau", dest="pe_tau", type=int, default=1)
    parser.add_argument("--entropy-se-window", dest="se_window", type=int, default=128)
    parser.add_argument("--entropy-cs-bins", dest="cs_bins", type=int, default=21)
    parser.add_argument("--ewm-halflife", dest="ewm_halflife", type=int, default=63)
    parser.add_argument("--min-history", dest="min_history", type=int, default=252)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--output-dir", dest="output_dir", default="output", type=Path)
    parser.add_argument("--benchmark", dest="benchmark", default="SPY")
    parser.add_argument("--threads", dest="threads", type=int, default=8)
    parser.add_argument("--prefetch", dest="prefetch", action="store_true")

    args = parser.parse_args(argv)

    entropy_cfg = EntropyConfig(
        pe_window=args.pe_window,
        pe_m=args.pe_m,
        pe_tau=args.pe_tau,
        se_window=args.se_window,
        cs_bins=args.cs_bins,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return BacktestConfig(
        provider=args.provider,
        polygon_api_key=args.polygon_api_key,
        database_url=args.database_url,
        universe_table=args.universe_table,
        universe_csv=Path(args.universe_csv),
        start_date=pd.Timestamp(args.start_date).normalize(),
        end_date=args.end_date.normalize() if args.end_date else None,
        cache_dir=Path(args.cache_dir),
        momentum_type=args.momentum_type,
        vol_estimator=args.vol_estimator,
        vol_target=args.vol_target,
        scale_cap=args.scale_cap,
        n_long=args.n_long,
        n_short=args.n_short,
        use_entropy=args.use_entropy,
        entropy=entropy_cfg,
        ewm_halflife=args.ewm_halflife,
        min_history=args.min_history,
        progress=args.progress,
        output_dir=output_dir,
        benchmark=args.benchmark.upper(),
        threads=max(1, args.threads),
        prefetch=args.prefetch,
    )


# ---------------------------------------------------------------------------
# Universe loading and preprocessing
# ---------------------------------------------------------------------------


def _load_universe(cfg: BacktestConfig) -> pd.DataFrame:
    """Fetch the point-in-time universe from SQL or the CSV fallback."""

    universe: Optional[pd.DataFrame] = None

    if cfg.database_url:
        try:
            engine = sqlalchemy.create_engine(cfg.database_url)
            with engine.connect() as connection:
                universe = pd.read_sql(cfg.universe_table, con=connection)
        except (ModuleNotFoundError, sqlalchemy.exc.SQLAlchemyError) as exc:
            warnings.warn(
                f"Unable to load universe from {cfg.database_url!r} ({exc}). Falling back to CSV.",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            if "engine" in locals():
                engine.dispose()

    if universe is None:
        if not cfg.universe_csv.exists():
            raise FileNotFoundError(
                f"Universe CSV {cfg.universe_csv} not found and no database connection succeeded."
            )
        universe = pd.read_csv(cfg.universe_csv)

    if "date" not in universe.columns or "ticker" not in universe.columns:
        raise ValueError("Universe must contain 'date' and 'ticker' columns")

    universe = universe.drop_duplicates(subset=["date", "ticker"])
    universe["date"] = pd.to_datetime(universe["date"], utc=False).dt.tz_localize(None)
    universe = universe.sort_values(["date", "ticker"])

    mask = universe["date"] >= cfg.start_date
    if cfg.end_date is not None:
        mask &= universe["date"] <= cfg.end_date
    universe = universe.loc[mask].copy()

    if universe.empty:
        raise RuntimeError("Universe is empty after applying date filters")

    return universe


# ---------------------------------------------------------------------------
# Market data retrieval
# ---------------------------------------------------------------------------


def _init_polygon(cfg: BacktestConfig) -> PolygonAggsClient:
    if cfg.polygon_api_key is None:
        raise ValueError("A Polygon.io API key is required. Use --polygon-api-key or set POLYGON_API_KEY.")
    session = requests.Session()
    session.headers.update({"User-Agent": "mtum-backtest/1.0"})
    return PolygonAggsClient(cfg.polygon_api_key, session=session)


def _cache_key(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return cache_dir / f"{safe}_{start}_{end}.parquet"


def _load_history(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    provider: PolygonAggsClient,
) -> pd.DataFrame:
    cache_path = _cache_key(cache_dir, ticker, start, end)
    if cache_path.exists():
        try:
            frame = pd.read_parquet(cache_path)
            frame["date"] = pd.to_datetime(frame["date"], utc=False)
            return frame
        except Exception:
            cache_path.unlink(missing_ok=True)

    frame = provider.get_daily_bars(ticker, start, end)
    if frame.empty:
        cache_path.touch(exist_ok=True)
        return frame

    frame.to_parquet(cache_path, index=False)
    return frame


def _fetch_histories(
    tickers: Sequence[str],
    price_start: str,
    price_end: str,
    cfg: BacktestConfig,
    provider: PolygonAggsClient,
) -> Dict[str, pd.DataFrame]:
    cache_dir = cfg.cache_dir / cfg.provider
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}

    def _task(symbol: str) -> Tuple[str, pd.DataFrame]:
        history = _load_history(symbol, price_start, price_end, cache_dir, provider)
        if "c" in history.columns:
            history = history.rename(columns={"c": "close"})
        return symbol, history

    if cfg.prefetch:
        with ThreadPoolExecutor(max_workers=cfg.threads) as executor:
            futures = {executor.submit(_task, symbol): symbol for symbol in tickers}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, hist = future.result()
                except Exception as exc:
                    warnings.warn(f"Failed to download history for {symbol}: {exc}")
                    hist = pd.DataFrame(columns=["date", "close", "ticker"])
                    sym = symbol
                results[sym] = hist
    else:
        for symbol in tickers:
            try:
                sym, hist = _task(symbol)
            except Exception as exc:
                warnings.warn(f"Failed to download history for {symbol}: {exc}")
                hist = pd.DataFrame(columns=["date", "close", "ticker"])
                sym = symbol
            results[sym] = hist

    return results


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def _build_price_panel(histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker, frame in histories.items():
        if frame.empty:
            continue
        if "close" not in frame.columns and "c" in frame.columns:
            frame = frame.rename(columns={"c": "close"})
        sub = frame[["date", "close"]].copy()
        sub["ticker"] = ticker
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=False)
    panel = panel.drop_duplicates(subset=["date", "ticker"])
    panel.sort_values(["date", "ticker"], inplace=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()
    panel["return_dec"] = panel.groupby(level="ticker")["close"].pct_change()
    panel["return_pct"] = panel["return_dec"] * 100.0
    return panel


def _student_t_vol(series: pd.Series, min_periods: int) -> pd.Series:
    """Robust volatility estimator inspired by a Student's t fit.

    The estimator trims the outer 5% of observations to dampen the influence of
    very heavy tails (mirroring the effect of fitting a low degree-of-freedom
    Student distribution) and rescales the standard deviation to an annualised
    figure.  The behaviour is deterministic and does not require SciPy.
    """

    def _calc(arr: np.ndarray) -> float:
        mask = np.isfinite(arr)
        arr = arr[mask]
        if arr.size < min_periods:
            return np.nan
        arr.sort()
        trim = max(0, int(math.floor(arr.size * 0.05)))
        if trim > 0 and arr.size > 2 * trim:
            arr = arr[trim:-trim]
        if arr.size < 2:
            return np.nan
        return float(np.std(arr, ddof=1))

    trimmed = series.groupby(level="ticker", group_keys=False).apply(
        lambda s: s.rolling(window=min_periods, min_periods=min_periods).apply(_calc, raw=True)
    )
    return trimmed * np.sqrt(252.0)


def _attach_market_features(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    panel = panel.join(benchmark[["return_pct", "return_dec"]].rename(
        columns={"return_pct": "bench_return_pct", "return_dec": "bench_return_dec"}
    ), on="date")

    bench_var = benchmark["return_pct"].rolling(window=cfg.min_history, min_periods=cfg.min_history).var()
    panel = panel.join(bench_var.rename("bench_var"), on="date")

    def _cov_with_bench(group: pd.DataFrame) -> pd.Series:
        return group["return_pct"].rolling(window=cfg.min_history, min_periods=cfg.min_history).cov(
            group["bench_return_pct"]
        )

    panel["cov_with_bench"] = panel.groupby(level="ticker", group_keys=False).apply(_cov_with_bench)
    panel["beta"] = panel["cov_with_bench"] / panel["bench_var"].replace(0.0, np.nan)
    panel["beta"] = panel["beta"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    panel["close_shift"] = panel.groupby(level="ticker")["close"].shift(cfg.min_history)
    panel["12-1_return"] = ((panel["close"] - panel["close_shift"]) / panel["close_shift"]) * 100.0

    realized_std = (
        panel.groupby(level="ticker")["return_dec"].rolling(window=cfg.min_history, min_periods=cfg.min_history).std().droplevel(0)
        * np.sqrt(252.0)
    )
    panel["std_daily"] = realized_std
    panel["sharpe"] = np.where(
        panel["std_daily"].replace(0.0, np.nan).notna(),
        panel["12-1_return"] / panel["std_daily"],
        0.0,
    )

    # EWMA stats are required for ewma-based momentum and as an optional vol estimator.
    ewm_inputs = panel[["return_dec"]].rename(columns={"return_dec": "returns"})
    ewm_features = ewm_moments(ewm_inputs, ret_col="returns", halflife=cfg.ewm_halflife, use_lagged_data=True)
    panel = panel.join(ewm_features)
    panel["ewm_mean_pct"] = panel["ewm_mean"] * 100.0

    if cfg.vol_estimator == "t":
        trimmed_vol = _student_t_vol(panel["return_dec"], min_periods=cfg.min_history)
        panel["t_vol_ann"] = trimmed_vol.reindex(panel.index)
    else:
        panel["t_vol_ann"] = np.nan

    return panel


def _momentum_score(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    if cfg.momentum_type == "classic":
        return panel["beta"] * panel["sharpe"]
    if cfg.momentum_type == "vol_scaled":
        return panel["beta"] * (panel["12-1_return"] / panel["std_daily"].replace(0.0, np.nan))
    if cfg.momentum_type in {"ewma", "ewma_vol_scaled"}:
        denom = panel["ewm_vol_ann"].replace(0.0, np.nan)
        return panel["beta"] * (panel["ewm_mean_pct"] / denom)
    return panel["beta"] * panel["sharpe"]


def _vol_scale(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    if cfg.momentum_type in {"vol_scaled", "ewma_vol_scaled"}:
        if cfg.vol_estimator == "standard":
            vol_source = panel["std_daily"].rename("vol")
        elif cfg.vol_estimator == "ewma":
            vol_source = panel["ewm_vol_ann"].rename("vol")
        elif cfg.vol_estimator == "t":
            vol_source = panel["t_vol_ann"].rename("vol")
        else:
            raise ValueError(f"Unknown vol estimator {cfg.vol_estimator!r}")

        vol_frame = vol_source.to_frame()
        vol_frame["vol"] = vol_frame["vol"].replace(0.0, np.nan)
        scale = volatility_target_scale(vol_frame, vol_col="vol", target_vol=cfg.vol_target, cap=cfg.scale_cap)
        return scale
    return pd.Series(1.0, index=panel.index)


# ---------------------------------------------------------------------------
# Monthly scoring
# ---------------------------------------------------------------------------


def _score_month(
    month_idx: int,
    months: Sequence[pd.Timestamp],
    universe_by_date: Dict[pd.Timestamp, pd.DataFrame],
    price_frame: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entry_date = months[month_idx]
    exit_date = months[month_idx + 1]
    universe = universe_by_date.get(entry_date)
    if universe is None or universe.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    tickers = universe["ticker"].to_numpy()

    mask = (price_frame["date"] <= entry_date) & price_frame["ticker"].isin(tickers)
    features = price_frame.loc[mask]
    if features.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    latest = (
        features.sort_values(["ticker", "date"])
        .drop_duplicates(subset="ticker", keep="last")
        .copy()
    )

    fwd_mask = (price_frame["date"] >= entry_date) & (price_frame["date"] <= exit_date) & (price_frame["ticker"].isin(tickers))
    fwd_window = price_frame.loc[fwd_mask, ["date", "ticker", "close"]]
    if fwd_window.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    fwd_window = fwd_window.sort_values(["ticker", "date"])
    valid = fwd_window.groupby("ticker").filter(lambda g: g.shape[0] >= 2)
    if valid.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    entry_prices = valid.groupby("ticker").head(1).set_index("ticker")["close"]
    exit_prices = valid.groupby("ticker").tail(1).set_index("ticker")["close"]
    fwd_returns = ((exit_prices - entry_prices) / entry_prices) * 100.0

    latest = latest.merge(fwd_returns.rename("forward_returns"), on="ticker", how="inner")
    if latest.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    latest["entry_date"] = entry_date.strftime("%Y-%m-%d")
    latest["exit_date"] = exit_date.strftime("%Y-%m-%d")

    selection = latest[
        [
            "entry_date",
            "exit_date",
            "ticker",
            "beta",
            "std_daily",
            "ewm_vol_ann",
            "t_vol_ann",
            "12-1_return",
            "ewm_mean_pct",
            "mom_score",
            "forward_returns",
        ]
    ]

    if cfg.use_entropy:
        history_slice = price_frame.loc[
            (price_frame["date"] <= entry_date) & (price_frame["ticker"].isin(tickers)),
            ["date", "ticker", "close"],
        ].dropna()
        if not history_slice.empty:
            history_slice = history_slice.set_index(["date", "ticker"]).sort_index()
            history_slice["returns"] = history_slice.groupby(level="ticker")["close"].pct_change()
            feats = add_entropy_info_features(
                history_slice[["returns"]],
                price_col="closeadj",
                use_lagged_data=True,
                pe_window=cfg.entropy.pe_window,
                pe_m=cfg.entropy.pe_m,
                pe_tau=cfg.entropy.pe_tau,
                se_window=cfg.entropy.se_window,
                cs_bins=cfg.entropy.cs_bins,
            )
            last_feats = latest_per_ticker(
                feats,
                entry_date,
                [
                    f"perm_entropy_w{cfg.entropy.pe_window}_m{cfg.entropy.pe_m}_tau{cfg.entropy.pe_tau}",
                    f"spectral_entropy_w{cfg.entropy.se_window}",
                ],
            )
            pe_col = f"perm_entropy_w{cfg.entropy.pe_window}_m{cfg.entropy.pe_m}_tau{cfg.entropy.pe_tau}"
            se_col = f"spectral_entropy_w{cfg.entropy.se_window}"
            valid_mask = (
                last_feats[pe_col].between(0.15, 0.98)
                & last_feats[se_col].between(0.15, 0.98)
            )
            keep = set(last_feats[valid_mask].index.astype(str))
            selection = selection[selection["ticker"].isin(keep)]

    if selection.empty:
        return (pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]))

    top_decile = selection.sort_values("mom_score", ascending=False).head(cfg.n_long)
    bot_decile = selection.sort_values("mom_score", ascending=True).head(cfg.n_short)

    return selection, top_decile, bot_decile


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _compile_trades(top_decile: pd.DataFrame, bot_decile: pd.DataFrame) -> pd.DataFrame:
    trade_dates = sorted(set(top_decile["entry_date"]) & set(bot_decile["entry_date"]))
    trades: List[pd.DataFrame] = []
    for date in trade_dates:
        long_side = top_decile[top_decile["entry_date"] == date]
        short_side = bot_decile[bot_decile["entry_date"] == date]
        if long_side.empty or short_side.empty:
            continue
        frame = pd.DataFrame(
            {
                "date": [date],
                "long_return": [long_side["forward_returns"].mean() / 100.0],
                "short_return": [(-short_side["forward_returns"].mean()) / 100.0],
            }
        )
        frame["portfolio_return"] = frame["long_return"] + frame["short_return"]
        trades.append(frame)

    if not trades:
        return pd.DataFrame([])

    out = pd.concat(trades, ignore_index=True).sort_values("date")
    out["date"] = pd.to_datetime(out["date"])
    out["long_equity"] = (1.0 + out["long_return"]).cumprod()
    out["short_equity"] = (1.0 + out["short_return"]).cumprod()
    out["portfolio_equity"] = (1.0 + out["portfolio_return"]).cumprod()
    return out


def _print_metrics(trades: pd.DataFrame, output_path: Path) -> None:
    monthly_returns = trades["portfolio_return"]
    cumulative = trades["portfolio_equity"]

    years = len(monthly_returns) / 12.0 if len(monthly_returns) else 0.0
    cagr = cumulative.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = monthly_returns.std(ddof=1) * np.sqrt(12.0) if len(monthly_returns) > 1 else np.nan
    sharpe = (monthly_returns.mean() * np.sqrt(12.0)) / volatility if volatility not in (0, np.nan) else np.nan
    rolling_max = cumulative.cummax()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min() if not drawdowns.empty else np.nan

    print("\nBacktest Summary Metrics")
    print("========================")
    print(f"CAGR: {cagr:.2%}" if not np.isnan(cagr) else "CAGR: n/a")
    print(
        f"Annualized Volatility: {volatility:.2%}" if not np.isnan(volatility) else "Annualized Volatility: n/a"
    )
    print(f"Sharpe Ratio: {sharpe:.2f}" if not np.isnan(sharpe) else "Sharpe Ratio: n/a")
    print(f"Max Drawdown: {max_drawdown:.2%}" if not np.isnan(max_drawdown) else "Max Drawdown: n/a")
    print(f"Equity curve saved to: {output_path.resolve()}")


def _plot_equity(trades: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "momentum_equity_curve.png"
    timeline = trades["date"]

    plt.figure(figsize=(12, 7), dpi=200)
    plt.xticks(rotation=45)
    plt.suptitle("Gross Cumulative Performance")
    plt.title("Monthly Rebalancing")
    plt.plot(timeline, (trades["long_equity"] - 1.0) * 100.0, label="Top Decile")
    plt.plot(timeline, (trades["short_equity"] - 1.0) * 100.0, label="Bottom Decile")
    plt.plot(timeline, (trades["portfolio_equity"] - 1.0) * 100.0, label="Long-Short")
    plt.xlabel("Date")
    plt.ylabel("Cumulative % Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)

    universe = _load_universe(cfg)
    months = sorted(universe["date"].drop_duplicates())
    if cfg.end_date is None:
        final_date = months[-1]
    else:
        final_date = min(cfg.end_date, months[-1])

    months = [m for m in months if cfg.start_date <= m <= final_date]
    if len(months) < 2:
        raise RuntimeError("Universe must contain at least two rebalancing months")

    universe_by_date = {date: frame for date, frame in universe.groupby("date")}

    price_buffer = timedelta(days=cfg.min_history + 60)
    price_start = (months[0] - price_buffer).strftime("%Y-%m-%d")
    price_end = (final_date + timedelta(days=35)).strftime("%Y-%m-%d")

    provider = _init_polygon(cfg)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    tickers = universe["ticker"].dropna().unique().tolist()
    histories = _fetch_histories(tickers, price_start, price_end, cfg, provider)
    if not histories:
        raise RuntimeError("No price histories were downloaded")

    price_panel = _build_price_panel(histories)
    if price_panel.empty:
        raise RuntimeError("Price panel is empty – check the Polygon API key and universe")

    # Build benchmark panel (always SPY by default).
    bench_resp = provider.session.get(
        f"https://api.polygon.io/v2/aggs/ticker/{cfg.benchmark}/range/1/day/2010-01-01/{price_end}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": cfg.polygon_api_key},
        timeout=provider.config.timeout,
    )
    bench_resp.raise_for_status()
    bench_json = bench_resp.json()
    bench_frame = pd.DataFrame(bench_json.get("results", []))
    if bench_frame.empty:
        raise RuntimeError(f"Benchmark response for {cfg.benchmark} was empty")
    bench_frame["timestamp"] = pd.to_datetime(bench_frame["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    bench_frame["date"] = bench_frame["timestamp"].dt.tz_localize(None)
    bench_frame.set_index("date", inplace=True)
    bench_frame.sort_index(inplace=True)
    bench_frame["return_pct"] = bench_frame["c"].pct_change() * 100.0
    bench_frame["return_dec"] = bench_frame["return_pct"] / 100.0

    price_panel = _attach_market_features(price_panel, bench_frame, cfg)
    price_panel["mom_score"] = _momentum_score(price_panel, cfg)
    price_panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    price_panel.dropna(subset=["mom_score", "12-1_return"], inplace=True)

    vol_scale = _vol_scale(price_panel, cfg)
    price_panel["vol_scale"] = vol_scale.reindex(price_panel.index).fillna(1.0)
    price_panel["mom_score"] *= price_panel["vol_scale"]

    price_frame = price_panel.reset_index()

    full_records: List[pd.DataFrame] = []
    top_deciles: List[pd.DataFrame] = []
    bot_deciles: List[pd.DataFrame] = []
    timings: List[float] = []

    total_iterations = len(months) - 1
    for idx in range(total_iterations):
        start_time = datetime.now()
        selection, top_decile, bot_decile = _score_month(idx, months, universe_by_date, price_frame, cfg)
        if not selection.empty:
            full_records.append(selection)
        if not top_decile.empty:
            top_deciles.append(top_decile)
        if not bot_decile.empty:
            bot_deciles.append(bot_decile)

        timings.append((datetime.now() - start_time).total_seconds())
        if cfg.progress:
            completed = idx + 1
            pct = round(100 * completed / total_iterations, 2)
            mean_time = float(np.mean(timings)) if timings else 0.0
            remaining = mean_time * (total_iterations - completed)
            print(f"{pct:.2f}% complete after {months[idx].date()}, ETA in ~ {timedelta(seconds=int(remaining))}")

    if not top_deciles or not bot_deciles:
        raise RuntimeError("No trades generated – check configuration or data availability")

    full_dataset = pd.concat(full_records, ignore_index=True) if full_records else pd.DataFrame([])
    top_dataset = pd.concat(top_deciles, ignore_index=True)
    bot_dataset = pd.concat(bot_deciles, ignore_index=True)

    trades = _compile_trades(top_dataset, bot_dataset)
    if trades.empty:
        raise RuntimeError("Trades DataFrame is empty – ensure both long and short baskets are populated")

    equity_path = _plot_equity(trades, cfg.output_dir)
    _print_metrics(trades, equity_path)


if __name__ == "__main__":
    main()

