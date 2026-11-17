"""Monte Carlo validation of the diagnostics on synthetic trial sets with a known answer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import synthetic as syn
from .cscv import cscv
from .sharpe import dsr
from .spa import spa_tests


def _one(x: np.ndarray, planted: int | None, spa_boot: int, seed: int) -> dict:
    res = cscv(x, s=16)
    sr = x.mean(0) / x.std(0, ddof=1)
    best = int(sr.argmax())
    d, _ = dsr(x[:, best], sr)
    spa = spa_tests(x, n_boot=spa_boot, seed=seed)
    return {"pbo": res.pbo, "dsr": d, "p_spa": spa.p_spa_consistent, "p_rc": spa.p_rc,
            "best_is_planted": float(planted is not None and best == planted)}


def run_validation(reps: int = 40, t: int = 249, n: int = 300, signals=(0.0, 0.5, 1.0, 1.5, 2.0),
                   family_shape=(15, 20), rho: float = 0.8, spa_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Rows: scenario x planted annual Sharpe x replication."""
    rows = []
    for si, sr in enumerate(signals):
        for rep in range(reps):
            rng = np.random.default_rng([seed, si, rep])
            x = syn.noise(t, n, rng) if sr == 0 else syn.noise_plus_signal(t, n, sr, rng)
            rows.append({"scenario": "independent", "planted_sharpe": sr, "rep": rep,
                         **_one(x, 0 if sr else None, spa_boot, rep)})
            nf, pf = family_shape
            rng = np.random.default_rng([seed, 100 + si, rep])
            x = syn.correlated_families(t, nf, pf, rho, rng, signal_family_sr=sr if sr else None)
            rows.append({"scenario": "families", "planted_sharpe": sr, "rep": rep,
                         **_one(x, None, spa_boot, rep)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scenario", "planted_sharpe"])
    return g.agg(pbo=("pbo", "mean"), pbo_sd=("pbo", "std"), dsr=("dsr", "mean"), dsr_gt_95=("dsr", lambda v: (v > 0.95).mean()),
                 spa_reject_5pct=("p_spa", lambda v: (v < 0.05).mean()), rc_reject_5pct=("p_rc", lambda v: (v < 0.05).mean()),
                 best_is_planted=("best_is_planted", "mean")).reset_index()
