# -*- coding: utf-8 -*-
"""Utility functions for constructing momentum and risk-aware features.

The helpers in this module follow the research-driven guidelines outlined in the
user specification.  They operate on ``pandas`` objects indexed by a
``MultiIndex`` with levels ``("ticker", "date")`` to ensure that we can build
signals across many instruments efficiently while avoiding look-ahead bias.

Each function performs light validation (index layout, column presence) so that
the calling scripts fail fast when a pre-condition is violated.  The
implementation is intentionally dependency-light – only ``numpy`` and
``pandas`` – so that the utilities can be reused both in backtests and live
production scripts.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _assert_multiindex(df: pd.DataFrame | pd.Series, *, expected_levels: Tuple[str, str] = ("ticker", "date")) -> None:
    """Ensure the frame/series has the expected two-level MultiIndex."""

    index = df.index if isinstance(df, pd.Series) else df.index
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("Input must use a two-level MultiIndex")
    level_names = list(index.names)
    if level_names != list(expected_levels):
        raise ValueError(f"Expected MultiIndex levels {expected_levels}, found {tuple(level_names)}")


def _normalize_halflives(half_lives: Iterable[int]) -> Tuple[int, ...]:
    values = tuple(int(x) for x in half_lives if int(x) > 0)
    if not values:
        raise ValueError("At least one positive half-life is required")
    return values


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


def make_log_returns_from_prices(prices: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Compute log returns per ticker.

    Parameters
    ----------
    prices:
        DataFrame indexed by (ticker, date).
    price_col:
        Column containing close prices.
    """

    _assert_multiindex(prices)
    if price_col not in prices.columns:
        raise KeyError(f"Column '{price_col}' not present in prices DataFrame")
    px = prices[[price_col]].sort_index()
    grouped = px.groupby(level="ticker")[price_col]
    log_ret = grouped.apply(lambda s: np.log(s) - np.log(s.shift(1)))
    log_ret.name = "ret"
    log_ret = log_ret.reindex(px.index)
    return log_ret


def trailing_return(prices: pd.DataFrame, window: int = 252, price_col: str = "close") -> pd.Series:
    """Compute trailing (window-1) period simple returns per ticker.

    Returns the simple percentage change over the specified rolling window.
    The output is aligned with the input index, with NaNs for warm-up periods.
    """

    _assert_multiindex(prices)
    if price_col not in prices.columns:
        raise KeyError(f"Column '{price_col}' not present in prices DataFrame")
    px = prices[[price_col]].sort_index()
    grouped = px.groupby(level="ticker")[price_col]
    trailing = grouped.transform(lambda s: s.pct_change(periods=window - 1))
    trailing.name = f"return_{window}d"
    return trailing


def ewm_momentum_returns(ret: pd.Series, half_lives: Iterable[int] = (63, 126, 252), *, min_frac: float = 0.67) -> pd.DataFrame:
    """Exponentially-weighted moving-average of returns per ticker."""

    _assert_multiindex(ret)
    hl_values = _normalize_halflives(half_lives)
    out = pd.DataFrame(index=ret.index)
    for hl in hl_values:
        minp = max(3, int(round(min_frac * hl)))
        col = f"mom_ewm_hl{hl}"
        out[col] = ret.groupby(level="ticker").transform(
            lambda s: s.ewm(halflife=hl, adjust=False, min_periods=minp).mean()
        )
    out = out.groupby(level="ticker").shift(1)
    return out


def ema_crossover_price(
    prices: pd.DataFrame,
    *,
    fast_hl: int = 32,
    slow_hl: int = 128,
    price_col: str = "close",
    min_frac: float = 0.67,
) -> pd.Series:
    """Normalized EMA crossover (fast - slow) / price."""

    _assert_multiindex(prices)
    if price_col not in prices.columns:
        raise KeyError(f"Column '{price_col}' not present in prices DataFrame")
    px = prices[[price_col]].sort_index()
    grouped = px.groupby(level="ticker")[price_col]
    min_fast = max(3, int(round(min_frac * fast_hl)))
    min_slow = max(3, int(round(min_frac * slow_hl)))
    ema_fast = grouped.transform(lambda s: s.ewm(halflife=fast_hl, adjust=False, min_periods=min_fast).mean())
    ema_slow = grouped.transform(lambda s: s.ewm(halflife=slow_hl, adjust=False, min_periods=min_slow).mean())
    crossover = (ema_fast - ema_slow) / px[price_col].replace(0, np.nan)
    crossover.name = "ema_xover_norm"
    crossover = crossover.groupby(level="ticker").shift(1)
    return crossover


def ewm_volatility(ret: pd.Series, *, hl: int = 20, annualize: int = 252, min_frac: float = 0.67) -> pd.Series:
    """Exponentially-weighted volatility (annualized)."""

    _assert_multiindex(ret)
    minp = max(3, int(round(min_frac * hl)))
    vol = ret.groupby(level="ticker").transform(
        lambda s: s.ewm(halflife=hl, adjust=False, min_periods=minp).std(bias=False)
    )
    vol = vol.groupby(level="ticker").shift(1)
    vol = vol * np.sqrt(annualize)
    vol.name = "vol_ewm_ann"
    return vol


def rolling_volatility(ret: pd.Series, *, window: int = 63, annualize: int = 252) -> pd.Series:
    """Simple rolling volatility (annualized)."""

    _assert_multiindex(ret)
    vol = ret.groupby(level="ticker").transform(
        lambda s: s.rolling(window=window, min_periods=window).std()
    )
    vol = vol.groupby(level="ticker").shift(1)
    vol = vol * np.sqrt(annualize)
    vol.name = "vol_rolling_ann"
    return vol


def yang_zhang_vol(prices: pd.DataFrame, *, window: int = 20, annualize: int = 252) -> pd.Series:
    """Yang-Zhang volatility estimator (annualized)."""

    required = {"open", "high", "low", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"Yang-Zhang volatility requires columns: {sorted(missing)}")
    _assert_multiindex(prices)

    def _compute(df: pd.DataFrame) -> pd.Series:
        o = np.log(df["open"])
        h = np.log(df["high"])
        l = np.log(df["low"])
        c = np.log(df["close"])

        co = c - o
        oo = o - o.shift(1)
        rs = (h - c) * (h - o) + (l - c) * (l - o)

        co2_roll = co.pow(2).rolling(window=window).mean()
        oo2_roll = oo.pow(2).rolling(window=window).mean()
        rs_roll = rs.rolling(window=window).mean()

        k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
        var = oo2_roll + k * co2_roll + (1 - k) * rs_roll
        return np.sqrt(var)

    vol = prices.groupby(level="ticker", group_keys=False).apply(_compute)
    vol = vol.groupby(level="ticker").shift(1)
    vol = vol * np.sqrt(annualize)
    vol.name = "vol_yz_ann"
    return vol


def sharpe_like_trend(mom_ewm: pd.DataFrame, vol_ann: pd.Series, *, annualize: int = 252) -> pd.DataFrame:
    """Convert momentum signals to Sharpe-like units using contemporaneous vol."""

    _assert_multiindex(mom_ewm)
    _assert_multiindex(vol_ann)
    vol_daily = vol_ann / np.sqrt(annualize)
    out = pd.DataFrame(index=mom_ewm.index)
    for column in mom_ewm.columns:
        out[f"{column}_sr"] = (mom_ewm[column] / vol_daily) * np.sqrt(annualize)
    return out


def forward_risk_adjusted_label(
    ret: pd.Series,
    vol_ann_at_t: pd.Series,
    *,
    horizon: int = 21,
    annualize: int = 252,
) -> pd.Series:
    """Future risk-adjusted cumulative return label."""

    _assert_multiindex(ret)
    _assert_multiindex(vol_ann_at_t)
    shifted = ret.groupby(level="ticker").shift(-1)
    fwd_sum = (
        shifted.groupby(level="ticker")
        .rolling(window=horizon)
        .sum()
    )
    fwd_sum.index = fwd_sum.index.droplevel(-1)
    vol_daily = vol_ann_at_t / np.sqrt(annualize)
    denom = vol_daily * np.sqrt(horizon)
    label = fwd_sum / denom
    label.name = f"y_fwd_{horizon}d_sr"
    return label


def vol_managed_scaler(
    factor_ret: pd.Series,
    *,
    hl: int = 63,
    target_vol_ann: float = 0.1,
    annualize: int = 252,
) -> pd.Series:
    """Compute Moreira-Muir style volatility managed leverage series."""

    if factor_ret.index.nlevels != 1:
        raise ValueError("factor_ret must be indexed by a single DatetimeIndex")
    vol_daily = factor_ret.ewm(halflife=hl, adjust=False, min_periods=max(3, int(round(0.67 * hl)))).std()
    target_daily = target_vol_ann / np.sqrt(annualize)
    scaler = (target_daily / vol_daily).shift(1)
    scaler.name = "vol_target_scaler"
    return scaler


def parse_halflife_string(value: str) -> Tuple[int, ...]:
    """Parse a comma separated list of integers into a tuple of half-lives."""

    if not value:
        raise ValueError("Half-life list cannot be empty")
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("Half-life list cannot be empty")
    return _normalize_halflives(int(part) for part in parts)


__all__ = [
    "ema_crossover_price",
    "ewm_momentum_returns",
    "ewm_volatility",
    "forward_risk_adjusted_label",
    "make_log_returns_from_prices",
    "parse_halflife_string",
    "rolling_volatility",
    "sharpe_like_trend",
    "trailing_return",
    "vol_managed_scaler",
    "yang_zhang_vol",
]

