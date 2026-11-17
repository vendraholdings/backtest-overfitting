"""Synthetic trial sets with a known answer, for validating the diagnostics."""
from __future__ import annotations

import numpy as np

from .sharpe import PERIODS_PER_YEAR


def _scale(sr_annual: float) -> float:
    """Per-period mean for unit per-period volatility at the given annual Sharpe ratio."""
    return sr_annual / np.sqrt(PERIODS_PER_YEAR)


def noise(t: int, n: int, rng: np.random.Generator, vol: float = 0.04) -> np.ndarray:
    """N independent zero-mean return series (true Sharpe 0)."""
    return rng.normal(0.0, vol, (t, n))


def noise_plus_signal(t: int, n: int, sr_annual: float, rng: np.random.Generator, vol: float = 0.04) -> np.ndarray:
    """Noise trials plus one trial (column 0) with true annual Sharpe ``sr_annual``."""
    x = noise(t, n, rng, vol)
    x[:, 0] += _scale(sr_annual) * vol
    return x


def correlated_families(t: int, n_families: int, per_family: int, rho: float, rng: np.random.Generator,
                        vol: float = 0.04, signal_family_sr: float | None = None) -> np.ndarray:
    """Families of trials sharing a common factor (within-family correlation ``rho``); optionally one family
    carries a common true annual Sharpe ratio."""
    cols = []
    for f in range(n_families):
        common = rng.normal(0.0, 1.0, (t, 1))
        idio = rng.normal(0.0, 1.0, (t, per_family))
        x = vol * (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
        if f == 0 and signal_family_sr is not None:
            x += _scale(signal_family_sr) * vol
        cols.append(x)
    return np.hstack(cols)
