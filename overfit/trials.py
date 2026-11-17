"""Trial-set structure: correlation, effective number of independent trials, and search-size curves."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .cscv import cscv
from .sharpe import PERIODS_PER_YEAR


def correlation(returns: np.ndarray) -> np.ndarray:
    return np.corrcoef(np.asarray(returns, dtype=np.float64), rowvar=False)


def eigen_effective_n(corr: np.ndarray) -> dict:
    """Eigenvalue-based counts: participation ratio, components for 90 % of variance, and components above the
    Marchenko-Pastur upper edge (requires the T/N ratio, supplied through ``corr.shape`` and ``t_obs`` by caller)."""
    lam = np.clip(np.linalg.eigvalsh(corr)[::-1], 0, None)
    lam = lam / lam.sum()
    pr = float(1.0 / np.sum(lam**2))
    n90 = int(np.searchsorted(np.cumsum(lam), 0.90) + 1)
    return {"participation_ratio": pr, "n_components_90pct": n90, "top_eigenvalue_share": float(lam[0])}


def marchenko_pastur_count(corr: np.ndarray, t_obs: int) -> int:
    """Number of eigenvalues of the correlation matrix above the Marchenko-Pastur upper edge (1 + sqrt(N/T))^2."""
    n = corr.shape[0]
    lam = np.linalg.eigvalsh(corr)
    return int((lam > (1 + np.sqrt(n / t_obs)) ** 2).sum())


def cluster_effective_n(corr: np.ndarray, thresholds=(0.5, 0.7, 0.9)) -> dict:
    """Number of clusters from average-linkage clustering on d = sqrt((1 - rho) / 2), cut at several correlations.

    A cut at correlation rho groups trials whose average pairwise correlation exceeds rho.
    Also returns the leaf order for heatmaps.
    """
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    out = {"order": leaves_list(z).tolist()}
    for rho in thresholds:
        labels = fcluster(z, t=np.sqrt((1 - rho) / 2), criterion="distance")
        out[f"clusters_rho_{rho}"] = int(labels.max())
    return out


def null_sharpe_matrix(returns: np.ndarray, reps: int, mean_block: float, rng: np.random.Generator,
                       annualise: bool = True) -> np.ndarray:
    """(reps, N) Sharpe ratios of the trials under a zero-skill null that keeps the cross-trial correlation:
    every trial is demeaned, then time is resampled jointly with the stationary bootstrap."""
    from .spa import stationary_bootstrap_counts
    r = np.asarray(returns, dtype=np.float64)
    t = r.shape[0]
    x = (r - r.mean(0)).astype(np.float32)
    w = stationary_bootstrap_counts(t, mean_block, reps, rng)
    m1 = (w @ x) / t
    m2 = (w @ x**2) / t
    sr = m1 / np.sqrt(np.maximum(m2 - m1**2, 1e-18))
    return sr * np.sqrt(PERIODS_PER_YEAR) if annualise else sr


def calibrated_effective_n(returns: np.ndarray, reps: int = 500, mean_block: float = 2.0, seed: int = 0) -> dict:
    """Number of independent trials N* for which the independent-trials formula for the expected maximum Sharpe
    (Bailey & Lopez de Prado) reproduces the simulated zero-skill maximum, given the simulated cross-trial variance.

    N* is capped at the nominal N. Returns N*, the simulated expected maximum and the null cross-trial variance
    (all per period).
    """
    from scipy.optimize import brentq

    from .sharpe import expected_max_sr
    null = null_sharpe_matrix(returns, reps, mean_block, np.random.default_rng(seed), annualise=False)
    var_null = float(null.var(axis=1, ddof=1).mean())
    target = float(null.max(axis=1).mean())
    n_max = returns.shape[1]
    capped = False
    f = lambda n: expected_max_sr(n, var_null) - target
    if f(1.0001) >= 0:
        n_star = 1.0
    elif f(n_max) <= 0:
        n_star, capped = float(n_max), True
    else:
        n_star = float(brentq(f, 1.0001, n_max))
    return {"n_eff": n_star, "capped_at_nominal": capped, "null_expected_max_sr": target, "null_var_sr": var_null,
            "null_expected_max_sharpe_annual": target * np.sqrt(PERIODS_PER_YEAR)}


def search_size_curve(returns: np.ndarray, sizes, reps: int = 50, seed: int = 0, s: int = 16, mean_block: float = 2.0,
                      null_reps: int = 200) -> list[dict]:
    """For each subsample size n: expected best full-sample Sharpe and CSCV PBO over random subsets of n trials,
    plus a zero-skill benchmark: the expected best Sharpe of n trials after removing every trial's mean and
    resampling time jointly (stationary bootstrap), which keeps the cross-trial correlation.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=np.float64)
    t, n_all = r.shape
    ann = np.sqrt(PERIODS_PER_YEAR)
    sr_full = r.mean(0) / r.std(0, ddof=1) * ann
    null_sr = null_sharpe_matrix(r, null_reps, mean_block, rng)  # (null_reps, N)
    rows = []
    for n in sizes:
        best, pbo, null_best = [], [], []
        for _ in range(reps):
            sub = rng.choice(n_all, size=n, replace=False)
            best.append(sr_full[sub].max())
            if n >= 2:
                pbo.append(cscv(r[:, sub], s=s).pbo)
            null_best.append(null_sr[rng.integers(0, null_reps), sub].max())
        rows.append({"n_trials": int(n),
                     "best_sharpe_mean": float(np.mean(best)), "best_sharpe_lo": float(np.percentile(best, 5)),
                     "best_sharpe_hi": float(np.percentile(best, 95)),
                     "pbo_mean": float(np.mean(pbo)) if pbo else float("nan"),
                     "pbo_lo": float(np.percentile(pbo, 5)) if pbo else float("nan"),
                     "pbo_hi": float(np.percentile(pbo, 95)) if pbo else float("nan"),
                     "null_best_sharpe_mean": float(np.mean(null_best)),
                     "null_best_sharpe_lo": float(np.percentile(null_best, 5)),
                     "null_best_sharpe_hi": float(np.percentile(null_best, 95))})
    return rows
