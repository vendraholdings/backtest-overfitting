"""Sharpe-ratio statistics: PSR, expected maximum Sharpe, DSR, minimum track record length.

All ``sr`` arguments are per-period (non-annualised) Sharpe ratios unless a name says otherwise.
Formulas follow Bailey & Lopez de Prado (2012, 2014).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 252 / 21  # monthly rebalance on a 21-trading-day horizon


def sharpe(r: np.ndarray, axis: int = 0, periods_per_year: float = PERIODS_PER_YEAR) -> np.ndarray:
    """Annualised Sharpe ratio (sample std, ddof=1) of per-period returns along ``axis``."""
    r = np.asarray(r, dtype=np.float64)
    return r.mean(axis) / r.std(axis, ddof=1) * math.sqrt(periods_per_year)


def moments(r: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, skewness and (non-excess) kurtosis of a return series."""
    r = np.asarray(r, dtype=np.float64)
    return float(r.mean() / r.std(ddof=1)), float(skew(r)), float(kurtosis(r, fisher=False))


def sr_std_error(sr: float, n: int, skew_: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of the per-period Sharpe ratio estimate (Mertens 2002 / Opdyke 2007)."""
    return math.sqrt((1.0 - skew_ * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1))


def psr(sr: float, sr_benchmark: float, n: int, skew_: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe ratio: P(true SR > sr_benchmark) given the estimate ``sr`` from ``n`` observations."""
    return float(norm.cdf((sr - sr_benchmark) / sr_std_error(sr, n, skew_, kurt)))


def expected_max_sr(n_trials: float, var_sr: float, mean_sr: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates with cross-trial variance ``var_sr`` (per period)."""
    if n_trials <= 1:
        return mean_sr
    n = float(n_trials)
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e))
    return mean_sr + math.sqrt(var_sr) * z


def dsr(r: np.ndarray, trial_srs: np.ndarray, n_trials: float | None = None) -> tuple[float, float]:
    """Deflated Sharpe ratio of return series ``r`` (the selected trial).

    ``trial_srs`` are the per-period Sharpe ratios of all trials; their variance sets the search noise.
    ``n_trials`` defaults to ``len(trial_srs)``; pass an effective N to account for correlated trials.
    Returns (DSR probability, deflation threshold SR0 per period).
    """
    trial_srs = np.asarray(trial_srs, dtype=np.float64)
    n_trials = len(trial_srs) if n_trials is None else n_trials
    sr0 = expected_max_sr(n_trials, float(np.var(trial_srs, ddof=1)))
    sr, sk, ku = moments(r)
    return psr(sr, sr0, len(r), sk, ku), sr0


def min_track_record_length(sr: float, sr_benchmark: float = 0.0, skew_: float = 0.0, kurt: float = 3.0,
                            alpha: float = 0.05) -> float:
    """Minimum number of observations for PSR(sr_benchmark) to reach 1 - alpha. ``inf`` if sr <= sr_benchmark."""
    if sr <= sr_benchmark:
        return math.inf
    return 1.0 + (1.0 - skew_ * sr + (kurt - 1.0) / 4.0 * sr**2) * (norm.ppf(1 - alpha) / (sr - sr_benchmark)) ** 2
