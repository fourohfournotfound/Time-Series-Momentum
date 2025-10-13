#!/usr/bin/env python3
"""
Relative market + entropy/information-theory utilities (causal, shape-preserving).

All functions:
  • preserve MultiIndex (date, ticker) and row count for additive features,
  • avoid look-ahead by shifting to t-1 where required,
  • robust to missing data / small cross-sections.

Exports:
  - add_entropy_info_features(...): permutation/spectral entropy (PE/SE) + peer surprisal & JSD.
  - add_relative_to_market(...): simple causal market-return & shrunk beta.
  - ewm_moments(...): per-ticker causal EWMA mean/vol (annualized).
  - volatility_target_scale(...): per-ticker causal vol-target scaling factor.
  - latest_per_ticker(...): last valid (<= cutoff) features per ticker.

Notes:
  • PE/SE normalized to [0,1]. Peer metrics computed from r[t-1].
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f
        return _wrap

# ---------------------------
# Validation / helpers
# ---------------------------

def validate_relative_features(df: pd.DataFrame,
                               original_shape: Tuple[int, int],
                               verbose: bool = False) -> None:
    assert isinstance(df.index, pd.MultiIndex) and list(df.index.names)[:2] == ["date", "ticker"], \
        "Index must remain MultiIndex ['date','ticker']"
    assert df.shape[0] == int(original_shape[0]), \
        f"Row count changed ({original_shape[0]} -> {df.shape[0]})"
    if verbose:  # place for future logging
        pass

def _ensure_returns(df: pd.DataFrame, price_col: str = "closeadj") -> pd.Series:
    if "returns" in df.columns:
        return df["returns"]
    base = price_col
    if f"{price_col}_raw" in df.columns:
        base = f"{price_col}_raw"
    elif "closeadj_raw" in df.columns:
        base = "closeadj_raw"
    r = df.groupby(level="ticker")[base].pct_change()
    return r

def _robust_minmax_series(s: pd.Series, window: int = 252, minp_ratio: float = 0.5) -> pd.Series:
    minp = max(10, int(window * minp_ratio))
    lo = s.rolling(window, min_periods=minp).min().shift(1)
    hi = s.rolling(window, min_periods=minp).max().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mm = (s - lo) / (hi - lo)
    return mm.clip(0.0, 1.0)

def _choose_market_ec_col(cols: pd.Index) -> Optional[str]:
    ec_cols = [c for c in cols if str(c).startswith("market_EC_")]
    return ec_cols[0] if ec_cols else None

# ---------------------------
# Simple relative-to-market example (causal)
# ---------------------------

def add_relative_to_market(
    df: pd.DataFrame,
    price_col: str = "closeadj",
    beta_window: int = 60,
    use_lagged_data: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    assert isinstance(df.index, pd.MultiIndex) and df.index.names == ["date", "ticker"]
    out = df.copy()
    base_col = price_col if price_col in out.columns else ("closeadj" if "closeadj" in out.columns else None)
    if base_col is None:
        return out

    rets = out.groupby("ticker")[base_col].pct_change()
    mkt = rets.groupby(level="date").mean()
    if use_lagged_data:
        mkt = mkt.shift(1)
    mkt = mkt.rename("market_return")
    out = out.join(mkt, on="date")

    stk_ret = rets.rename("stk_ret")
    tmp = (
        out.reset_index()
        .join(stk_ret.reset_index(drop=False).rename(columns={base_col: "xx"}), on=["date", "ticker"], rsuffix="_r")
    )
    tmp["stk_ret"] = tmp["xx"]
    tmp.sort_values(["ticker", "date"], inplace=True)

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["stk_ret"].rolling(beta_window, min_periods=max(10, beta_window // 2)).cov(g["market_return"])
        var = g["market_return"].rolling(beta_window, min_periods=max(10, beta_window // 2)).var()
        raw = cov / (var + 1e-8)
        beta = 0.67 * raw + 0.33 * 1.0
        return beta.shift(1)

    tmp["beta_60d"] = tmp.groupby("ticker", group_keys=False).apply(_beta)
    out["beta_60d"] = tmp.set_index(["date", "ticker"])["beta_60d"]
    return out

# ---------------------------
# Entropy / Information-theory features (causal)
# ---------------------------

@njit(cache=True)
def _factorials_upto(n: int) -> np.ndarray:
    f = np.ones(n + 1, dtype=np.int64)
    for i in range(2, n + 1):
        f[i] = f[i - 1] * i
    return f

@njit(cache=True)
def _lehmer_code_from_perm(perm: np.ndarray) -> int:
    m = perm.size
    fact = _factorials_upto(m)
    code = 0
    for i in range(m):
        c = 0
        pi = perm[i]
        for j in range(i + 1, m):
            if perm[j] < pi:
                c += 1
        code += c * fact[m - 1 - i]
    return code

@njit(cache=True)
def _ordinal_code(values: np.ndarray) -> int:
    order = np.argsort(values)
    return _lehmer_code_from_perm(order)

@njit(cache=True)
def _sliding_permutation_entropy(
    x: np.ndarray,
    window: int,
    m: int = 4,
    tau: int = 1,
    min_patterns: int = 20,
    normalize: bool = True,
) -> np.ndarray:
    n = x.size
    H = np.empty(n, dtype=np.float64)
    H.fill(np.nan)
    if n == 0 or window <= 0 or m < 3 or tau < 1:
        return H
    span = (m - 1) * tau
    max_start = n - span
    if max_start <= 0:
        return H

    codes = np.full(max_start, -1, dtype=np.int32)
    tmp = np.empty(m, dtype=np.float64)
    for s in range(max_start):
        ok = True
        for k in range(m):
            v = x[s + k * tau]
            if not np.isfinite(v):
                ok = False
                break
            tmp[k] = v
        if ok:
            codes[s] = _ordinal_code(tmp)

    mfac = _factorials_upto(m)[m]
    counts = np.zeros(mfac, dtype=np.int64)
    total = 0
    s_lo_prev = -10 ** 9
    s_hi_prev = -10 ** 9

    for t in range(n):
        s_lo = t - window + 1
        s_hi = t - span
        if s_hi < 0:
            H[t] = np.nan
            continue
        if s_lo < 0:
            s_lo = 0
        if s_hi > max_start - 1:
            s_hi = max_start - 1
        if s_lo > s_hi:
            H[t] = np.nan
            continue

        if s_lo_prev == -10 ** 9:
            total = 0
            counts[:] = 0
            for s in range(s_lo, s_hi + 1):
                c = codes[s]
                if c >= 0:
                    counts[c] += 1
                    total += 1
        else:
            for s in range(s_lo_prev, min(s_lo, s_hi_prev + 1)):
                if s < 0:
                    continue
                if s > s_hi_prev:
                    break
                c = codes[s]
                if c >= 0:
                    counts[c] -= 1
                    total -= 1
            for s in range(max(s_hi_prev + 1, s_lo), s_hi + 1):
                c = codes[s]
                if c >= 0:
                    counts[c] += 1
                    total += 1

        s_lo_prev, s_hi_prev = s_lo, s_hi

        if total < min_patterns or total <= 0:
            H[t] = np.nan
            continue

        ent = 0.0
        invT = 1.0 / float(total)
        for j in range(mfac):
            c = counts[j]
            if c > 0:
                p = c * invT
                ent -= p * np.log(p)
        if normalize:
            ent /= np.log(mfac)
        H[t] = ent

    return H

def _rolling_spectral_entropy(
    x: np.ndarray,
    window: int,
    update_freq: int = 5,
    nfft: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    n = x.size
    H = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window <= 4:
        return H
    if nfft is None:
        nfft = 1 << (window - 1).bit_length()

    t = window - 1
    prev = np.nan
    while t < n:
        seg = x[t - window + 1 : t + 1]
        if np.isfinite(seg).sum() >= max(8, window // 2):
            seg = seg - np.nanmean(seg)
            sp = np.fft.rfft(seg, n=nfft)
            psd = (sp.real ** 2 + sp.imag ** 2)
            psd = np.where(np.isfinite(psd), psd, 0.0)
            psd_sum = psd.sum()
            if psd_sum > 0:
                p = psd / psd_sum
                ent = -(p * np.log(p + 1e-300)).sum()
                if normalize:
                    ent /= np.log(p.size)
                prev = ent
            else:
                prev = np.nan
        else:
            prev = np.nan
        H[t] = prev
        t_next = min(n, t + update_freq)
        H[t + 1 : t_next] = prev
        t = t_next
    return H

def _cross_sectional_surprisal_from_bins(
    r_lag: pd.Series,
    n_bins: int = 21,
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    if r_lag.empty:
        return (
            pd.Series(np.nan, index=r_lag.index),
            pd.Series(np.nan, index=r_lag.index),
            np.array([]),
        )
    assert isinstance(r_lag.index, pd.MultiIndex) and "date" in r_lag.index.names and "ticker" in r_lag.index.names
    surpr_parts, p_parts = [], []
    for d, s in r_lag.groupby(level="date"):
        x = s.to_numpy(dtype=float)
        x_f = x[np.isfinite(x)]
        if x_f.size < max(8, n_bins // 2):
            surpr_parts.append(pd.Series(np.nan, index=s.index))
            p_parts.append(pd.Series(np.nan, index=s.index))
            continue
        lo = float(np.quantile(x_f, q_low)); hi = float(np.quantile(x_f, q_high))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
            lo, hi = float(np.min(x_f)), float(np.max(x_f))
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
                surpr_parts.append(pd.Series(np.nan, index=s.index))
                p_parts.append(pd.Series(np.nan, index=s.index))
                continue
        edges_d = np.linspace(lo, hi, n_bins + 1)
        b = np.clip(np.digitize(x, edges_d) - 1, 0, n_bins - 1)
        counts = np.bincount(b, minlength=n_bins).astype(float)
        total = counts.sum()
        if total <= 0:
            surpr_parts.append(pd.Series(np.nan, index=s.index))
            p_parts.append(pd.Series(np.nan, index=s.index))
            continue
        p = counts / total
        p_i = p[b]
        surpr = -np.log(np.clip(p_i, 1e-12, 1.0)) / np.log(float(n_bins))
        surpr_parts.append(pd.Series(surpr, index=s.index, dtype=float))
        p_parts.append(pd.Series(p_i, index=s.index, dtype=float))
    surprisal = pd.concat(surpr_parts).sort_index().rename("peer_surprisal")
    p_series = pd.concat(p_parts).sort_index().rename("peer_prob")
    return surprisal, p_series, np.array([])

def _peer_jsd_pointmass(
    r_lag: pd.Series,
    n_bins: int = 21,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.Series:
    if r_lag.empty:
        return pd.Series(np.nan, index=r_lag.index, name=f"peer_jsd_bins{n_bins}")

    def _H(p: np.ndarray) -> float:
        p = np.clip(p, 1e-300, 1.0)
        return float(-(p * np.log(p)).sum())

    out_parts = []
    for d, s in r_lag.groupby(level="date"):
        x = s.to_numpy(dtype=float)
        x_f = x[np.isfinite(x)]
        if x_f.size < max(8, n_bins // 2):
            out_parts.append(pd.Series(np.nan, index=s.index))
            continue
        lo = float(np.quantile(x_f, q_low)); hi = float(np.quantile(x_f, q_high))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
            lo, hi = float(np.min(x_f)), float(np.max(x_f))
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
                out_parts.append(pd.Series(np.nan, index=s.index)); continue
        edges = np.linspace(lo, hi, n_bins + 1)
        b = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(b, minlength=n_bins).astype(float)
        total = counts.sum()
        if total <= 0:
            out_parts.append(pd.Series(np.nan, index=s.index)); continue
        q = counts / total
        Hq = _H(q)
        jsd_by_bin = np.empty(n_bins, dtype=float)
        for k in range(n_bins):
            M = q.copy() * 0.5
            M[k] += 0.5
            HM = _H(M)
            jsd_by_bin[k] = HM - 0.5 * Hq  # H(δ_k)=0
        jsd_i = jsd_by_bin[b]
        out_parts.append(pd.Series(jsd_i, index=s.index, dtype=float))
    return pd.concat(out_parts).sort_index().rename(f"peer_jsd_bins{n_bins}")

def _cs_robustz(df: pd.DataFrame, col: str, clip: float = 8.0) -> pd.Series:
    s = df[col].astype(float)
    med = s.groupby(level="date").transform("median")
    q25 = s.groupby(level="date").transform(lambda x: x.quantile(0.25))
    q75 = s.groupby(level="date").transform(lambda x: x.quantile(0.75))
    iqr = (q75 - q25).replace(0.0, np.nan)
    iqr = iqr.fillna(iqr.median()).replace(0, np.nan)
    z = (s - med) / (iqr / 1.349 + 1e-8)
    return z.clip(-clip, clip)

def _pca_factor_mix_entropy(
    panel: pd.DataFrame,
    window: int = 252,
    topk: int = 5,
    update_freq: int = 5,
    min_valid_ratio: float = 0.6,
) -> pd.DataFrame:
    all_dates = panel.index.to_numpy()
    out_frames = []
    start = window
    if start >= len(all_dates):
        return pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for idx in range(start, len(all_dates), update_freq):
        t = all_dates[idx]
        hist = panel.iloc[idx - window : idx]
        minp = max(20, int(min_valid_ratio * window))
        good = hist.count() >= minp
        cols = list(good[good].index)
        if len(cols) < 3:
            H = pd.Series(np.nan, index=panel.columns)
        else:
            X = hist[cols].dropna(axis=0, how="any")
            if X.shape[0] < minp or X.shape[1] < 3:
                H = pd.Series(np.nan, index=panel.columns)
            else:
                Xc = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
                C = np.corrcoef(Xc.values, rowvar=False)
                evals, evecs = np.linalg.eigh(C)
                idxs = np.argsort(evals)[::-1][: min(topk, len(evals))]
                lam = np.maximum(evals[idxs], 1e-12)
                V = evecs[:, idxs]
                W = V ** 2 * lam[None, :]
                Wsum = W.sum(axis=1, keepdims=True)
                P = np.where(Wsum > 0, W / Wsum, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Hi = -(P * np.log(P + 1e-300)).sum(axis=1) / np.log(P.shape[1])
                H = pd.Series(Hi, index=cols, dtype=float)
                H = H.reindex(panel.columns)
        out_frames.append(pd.DataFrame([H.values], index=[t], columns=panel.columns))
    Hdf = pd.concat(out_frames).sort_index()
    Hdf = Hdf.reindex(panel.index).ffill()
    return Hdf

def add_entropy_info_features(
    df: pd.DataFrame,
    price_col: str = "closeadj",
    use_lagged_data: bool = True,
    pe_window: int = 126,
    pe_m: int = 4,
    pe_tau: int = 1,
    pe_min_patterns: int = 20,
    se_window: int = 128,
    se_update_freq: int = 5,
    se_nfft: Optional[int] = None,
    cs_bins: int = 21,
    pca_window: Optional[int] = None,
    pca_topk: int = 5,
    pca_update_freq: int = 5,
    pca_min_valid_ratio: float = 0.6,
    verbose: bool = False,
) -> pd.DataFrame:
    assert isinstance(df.index, pd.MultiIndex) and df.index.names == ["date", "ticker"], \
        "Index must be MultiIndex ['date','ticker'] (sorted)."
    out = df.copy()

    r = _ensure_returns(out, price_col=price_col)
    r_for = r.groupby(level="ticker").shift(1) if use_lagged_data else r

    # Permutation entropy per ticker
    pe_list = []
    for _, g in r_for.groupby(level="ticker"):
        x = g.to_numpy(dtype=float)
        H = _sliding_permutation_entropy(
            x, window=pe_window, m=pe_m, tau=pe_tau, min_patterns=pe_min_patterns, normalize=True
        )
        pe_list.append(pd.Series(H, index=g.index))
    pe = pd.concat(pe_list).sort_index()
    pe_name = f"perm_entropy_w{pe_window}_m{pe_m}_tau{pe_tau}"
    out[pe_name] = pe

    # Spectral entropy per ticker
    se_list = []
    for _, g in r_for.groupby(level="ticker"):
        x = g.to_numpy(dtype=float)
        Hs = _rolling_spectral_entropy(
            x, window=se_window, update_freq=se_update_freq, nfft=se_nfft, normalize=True
        )
        se_list.append(pd.Series(Hs, index=g.index))
    se = pd.concat(se_list).sort_index()
    se_name = f"spectral_entropy_w{se_window}"
    out[se_name] = se

    # Cross-sectional peer surprisal/JSD using r[t-1]
    surprisal, _p_series, _edges = _cross_sectional_surprisal_from_bins(r_for, n_bins=cs_bins)
    out[f"peer_surprisal_cs_bins{cs_bins}"] = surprisal
    jsd = _peer_jsd_pointmass(r_for, n_bins=cs_bins)
    out[f"peer_jsd_bins{cs_bins}"] = jsd

    # Robust per-date z-scores
    out["peer_surprisal_z"] = _cs_robustz(out, f"peer_surprisal_cs_bins{cs_bins}")
    out["peer_jsd_z"] = _cs_robustz(out, f"peer_jsd_bins{cs_bins}")

    # Optional factor-mix entropy via PCA (≤ t-1)
    if pca_window is not None and pca_window > 20:
        panel = r_for.unstack("ticker")
        Hdf = _pca_factor_mix_entropy(
            panel, window=pca_window, topk=pca_topk, update_freq=pca_update_freq, min_valid_ratio=pca_min_valid_ratio
        )
        fm = Hdf.stack().reindex(out.index)
        out[f"factor_mix_entropy_k{pca_topk}_w{pca_window}"] = fm.astype(float)

    assert out.shape[0] == df.shape[0]
    return out

# ---------------------------
# NEW: Causal EWMA & Vol-target helpers
# ---------------------------

def ewm_moments(
    df: pd.DataFrame,
    ret_col: str = "returns",
    halflife: int = 63,
    use_lagged_data: bool = True,
) -> pd.DataFrame:
    """Per-ticker EWMA mean & vol (annualized) of returns at t-1 if use_lagged_data."""
    assert isinstance(df.index, pd.MultiIndex) and df.index.names == ["date", "ticker"]
    s = df[ret_col].astype(float)

    def _per_ticker(x: pd.Series) -> pd.DataFrame:
        m = x.ewm(halflife=halflife, adjust=False, min_periods=max(10, halflife//2)).mean()
        v = x.ewm(halflife=halflife, adjust=False, min_periods=max(10, halflife//2)).var()
        if use_lagged_data:
            m = m.shift(1); v = v.shift(1)
        out = pd.DataFrame({"ewm_mean": m, "ewm_vol_ann": np.sqrt(v) * np.sqrt(252.0)})
        return out

    out = s.groupby(level="ticker", group_keys=False).apply(_per_ticker)
    return out

def volatility_target_scale(
    df: pd.DataFrame,
    vol_col: str = "ewm_vol_ann",
    target_vol: float = 0.20,
    cap: float = 3.0,
) -> pd.Series:
    """Scale factor per row to hit target annualized vol; clipped to [0, cap]."""
    v = df[vol_col].astype(float).replace(0.0, np.nan)
    scale = target_vol / (v + 1e-8)
    return scale.clip(lower=0.0, upper=cap).fillna(0.0)

def latest_per_ticker(df: pd.DataFrame, cutoff_date: pd.Timestamp, cols: List[str]) -> pd.DataFrame:
    """Return last available row <= cutoff_date for each ticker, with columns 'cols'."""
    assert isinstance(df.index, pd.MultiIndex) and df.index.names == ["date", "ticker"]
    sub = df.loc[pd.IndexSlice[:cutoff_date, :], cols].copy()
    last_idx = sub.groupby(level="ticker").tail(1)
    last_idx = last_idx.reset_index(level="date", drop=False)
    last_idx = last_idx.reset_index().set_index("ticker")
    return last_idx

__all__ = [
    "validate_relative_features",
    "add_relative_to_market",
    "add_entropy_info_features",
    "ewm_moments",
    "volatility_target_scale",
    "latest_per_ticker",
    "_cross_sectional_surprisal_from_bins",
    "_peer_jsd_pointmass",
]
