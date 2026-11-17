"""White's Reality Check and Hansen's test for superior predictive ability (SPA).

The benchmark is zero excess return, so the performance differential of trial k is its own return series
d[t, k]. Resampling uses the stationary bootstrap (Politis & Romano, 1994). Resampled means are obtained as
W @ d / T, where W holds the per-resample counts of each original observation, so all trials are handled in
one matrix product per chunk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(t: int, mean_block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, t) integer indices of stationary-bootstrap resamples with geometric block lengths (mean ``mean_block``)."""
    p = 1.0 / mean_block
    pos = np.arange(t)
    restart = rng.random((n_boot, t)) < p
    restart[:, 0] = True
    starts = rng.integers(0, t, (n_boot, t))
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + (pos - last)) % t


def stationary_bootstrap_counts(t: int, mean_block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, t) matrix: how often each original observation appears in each stationary-bootstrap resample."""
    idx = stationary_bootstrap_indices(t, mean_block, n_boot, rng)
    flat = (np.arange(n_boot)[:, None] * t + idx).ravel()
    return np.bincount(flat, minlength=n_boot * t).reshape(n_boot, t).astype(np.float32)


@dataclass
class SPAResult:
    p_rc: float            # White's Reality Check (max of mean differentials)
    p_spa_upper: float     # Hansen SPA, upper (equal to the studentised RC)
    p_spa_consistent: float
    p_spa_lower: float
    stat_rc: float
    stat_spa: float
    best_index: int
    n_boot: int
    mean_block: float


def spa_tests(d: np.ndarray, n_boot: int = 100_000, mean_block: float = 2.0, seed: int = 0, chunk: int = 2000) -> SPAResult:
    d = np.asarray(d, dtype=np.float32)
    t, k = d.shape
    dbar = d.mean(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, k), dtype=np.float32)
    for a in range(0, n_boot, chunk):
        b = min(chunk, n_boot - a)
        w = stationary_bootstrap_counts(t, mean_block, b, rng)
        boot[a:a + b] = (w @ d) / t
    omega = boot.std(0, ddof=1).astype(np.float64) * math.sqrt(t)  # std of sqrt(T) * bootstrapped mean
    omega = np.maximum(omega, 1e-12)
    centred = boot - dbar.astype(np.float32)  # bootstrapped mean minus sample mean
    stat_rc = math.sqrt(t) * float(dbar.max())
    stat_spa = max(0.0, float((math.sqrt(t) * dbar / omega).max()))
    v_rc = math.sqrt(t) * centred.max(1)
    p_rc = float(np.mean(v_rc >= stat_rc))
    loglog = math.sqrt(2 * math.log(math.log(t)))
    mu = {
        "upper": np.zeros(k),
        "consistent": np.where(math.sqrt(t) * dbar / omega <= -loglog, dbar, 0.0),
        "lower": np.minimum(dbar, 0.0),
    }
    p = {}
    for name, m in mu.items():
        z = (math.sqrt(t) * (centred + m.astype(np.float32)) / omega.astype(np.float32)).max(1)
        p[name] = float(np.mean(np.maximum(z, 0.0) >= stat_spa))
    return SPAResult(p_rc, p["upper"], p["consistent"], p["lower"], stat_rc, stat_spa, int(dbar.argmax()), n_boot, mean_block)
