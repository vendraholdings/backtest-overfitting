"""Multiple-testing adjustments and Sharpe haircuts (Harvey & Liu, 2015).

The two-sided p-value of a Sharpe ratio is p = 2 (1 - Phi(|t|)) with t = SR_period * sqrt(T). Adjusted p-values
(Bonferroni, Holm, Benjamini-Hochberg-Yekutieli) map to an adjusted t-statistic and hence an adjusted
Sharpe ratio; the haircut is 1 - SR_adjusted / SR.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def pvalues(sr_period: np.ndarray, t_obs: int) -> np.ndarray:
    """Two-sided normal p-values of per-period Sharpe ratios estimated from ``t_obs`` observations."""
    return 2 * norm.sf(np.abs(np.asarray(sr_period, dtype=np.float64)) * math.sqrt(t_obs))


def adjust(p: np.ndarray, method: str, m: int | None = None) -> np.ndarray:
    """Adjusted p-values. ``m`` overrides the number of tests (e.g. an effective N). When ``m`` < len(p) the Holm
    multiplier (m - k + 1) is floored at 1."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    m = n if m is None else m
    order = np.argsort(p)
    ps = p[order]
    k = np.arange(1, n + 1)
    if method == "bonferroni":
        adj_sorted = ps * m
    elif method == "holm":
        adj_sorted = np.maximum.accumulate(ps * np.maximum(m - k + 1, 1))
    elif method == "bhy":
        c = np.sum(1.0 / np.arange(1, m + 1))
        adj_sorted = np.minimum.accumulate((ps * m * c / k)[::-1])[::-1]
    elif method == "bh":
        adj_sorted = np.minimum.accumulate((ps * m / k)[::-1])[::-1]
    else:
        raise ValueError(method)
    adj = np.empty(n)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def haircut_sharpe(sr_annual: float, t_obs: int, p_adj: float, periods_per_year: float) -> tuple[float, float]:
    """Adjusted annualised Sharpe ratio and haircut for an adjusted two-sided p-value."""
    p_adj = min(max(p_adj, 1e-300), 1.0)
    t_adj = float(norm.isf(p_adj / 2))
    sr_adj = t_adj / math.sqrt(t_obs) * math.sqrt(periods_per_year)
    sr_adj = min(sr_adj, abs(sr_annual))
    return sr_adj, 1.0 - sr_adj / abs(sr_annual)


def haircut_table(sr_period: np.ndarray, t_obs: int, periods_per_year: float, m_eff: int | None = None) -> dict:
    """Best-trial haircut under each method for the nominal number of tests and, optionally, an effective number."""
    sr_period = np.asarray(sr_period)
    best = int(np.argmax(sr_period))
    p = pvalues(sr_period, t_obs)
    sr_ann = float(sr_period[best] * math.sqrt(periods_per_year))
    out = {"best_index": best, "best_sharpe_annual": sr_ann, "p_raw": float(p[best]), "n_tests": len(p)}
    for label, m in (("nominal", None), ("effective", m_eff)):
        if label == "effective" and m_eff is None:
            continue
        for method in ("bonferroni", "holm", "bhy"):
            adj = adjust(p, method, m)
            sr_adj, hc = haircut_sharpe(sr_ann, t_obs, float(adj[best]), periods_per_year)
            out[f"{label}_{method}"] = {"p_adj": float(adj[best]), "sharpe_adj": sr_adj, "haircut": hc,
                                        "n_significant_5pct": int((adj < 0.05).sum())}
    return out
