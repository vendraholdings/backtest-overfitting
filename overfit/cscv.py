"""Combinatorially symmetric cross-validation and the probability of backtest overfitting.

Bailey, Borwein, Lopez de Prado & Zhu (2017). The matrix of returns (T periods x N trials) is cut into S
equal blocks (leading remainder periods are dropped). Every choice of S/2 blocks is a training set and
its complement the test set. The Sharpe ratio of every trial on every training and test set is computed
at once from block sums and sums of squares with one matrix product.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def split_matrix(s: int) -> np.ndarray:
    """0/1 matrix of shape (C(s, s/2), s); row = blocks assigned to the training set."""
    combos = list(combinations(range(s), s // 2))
    m = np.zeros((len(combos), s))
    for i, c in enumerate(combos):
        m[i, list(c)] = 1.0
    return m


def block_stats(returns: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-block sum and sum of squares, shape (s, N), and the block length. Drops leading remainder rows."""
    t, _ = returns.shape
    blen = t // s
    x = np.asarray(returns[t - blen * s:], dtype=np.float64).reshape(s, blen, -1)
    return x.sum(1), (x**2).sum(1), blen


def _sharpe_from_sums(sm: np.ndarray, sq: np.ndarray, n: int) -> np.ndarray:
    mean = sm / n
    var = (sq - n * mean**2) / (n - 1)
    return mean / np.sqrt(var)


@dataclass
class CSCVResult:
    is_sr: np.ndarray        # (n_splits,) per-period in-sample Sharpe of the selected trial
    oos_sr: np.ndarray       # (n_splits,) out-of-sample Sharpe of the selected trial
    selected: np.ndarray     # (n_splits,) index of the selected trial
    omega: np.ndarray        # (n_splits,) relative OOS rank of the selected trial, rank / (N + 1)
    logit: np.ndarray        # (n_splits,) ln(omega / (1 - omega))
    oos_mean_sr: np.ndarray  # (n_splits,) mean OOS Sharpe over all trials
    n_trials: int
    n_blocks: int
    joint: np.ndarray | None = None  # (bins, bins) normalised joint density of IS rank x OOS rank over all trials
    trial_oos_mean: np.ndarray | None = None   # (N,) mean OOS Sharpe of each trial over all splits
    trial_is_mean: np.ndarray | None = None    # (N,) mean IS Sharpe of each trial over all splits

    @property
    def pbo(self) -> float:
        return float(np.mean(self.logit <= 0))

    def summary(self) -> dict:
        slope, intercept = np.polyfit(self.is_sr, self.oos_sr, 1)
        return {
            "pbo": self.pbo,
            "n_splits": int(len(self.logit)),
            "n_trials": self.n_trials,
            "n_blocks": self.n_blocks,
            "degradation_slope": float(slope),
            "degradation_intercept": float(intercept),
            "prob_oos_loss": float(np.mean(self.oos_sr < 0)),
            "mean_is_sr": float(self.is_sr.mean()),
            "mean_oos_sr": float(self.oos_sr.mean()),
            "mean_oos_sr_all_trials": float(self.oos_mean_sr.mean()),
            "sd1_dominance": stochastic_dominance(self.oos_sr, self.oos_mean_sr, order=1),
            "sd2_dominance": stochastic_dominance(self.oos_sr, self.oos_mean_sr, order=2),
        }


def stochastic_dominance(selected: np.ndarray, reference: np.ndarray, order: int = 1, grid: int = 400) -> float:
    """Share of grid points at which the selected-trial OOS distribution dominates the reference.

    order 1: CDF(selected) <= CDF(reference). order 2: integrated CDF(selected) <= integrated CDF(reference).
    A value of 1.0 means dominance everywhere; values near 0 mean the reference dominates.
    """
    lo, hi = min(selected.min(), reference.min()), max(selected.max(), reference.max())
    x = np.linspace(lo, hi, grid)
    f = (np.sort(selected)[:, None] <= x).mean(0)
    g = (np.sort(reference)[:, None] <= x).mean(0)
    if order == 2:
        f, g = np.cumsum(f), np.cumsum(g)
    return float(np.mean(f <= g + 1e-12))


def cscv(returns: np.ndarray, s: int = 16, joint_bins: int | None = None, chunk: int = 2048) -> CSCVResult:
    """Run CSCV on a (T, N) return matrix. ``joint_bins`` additionally returns the IS x OOS rank joint density."""
    if s % 2:
        raise ValueError("s must be even")
    returns = np.asarray(returns)
    n_trials = returns.shape[1]
    bs, bq, blen = block_stats(returns, s)
    m = split_matrix(s)
    n_half = blen * s // 2
    out = {k: [] for k in ("is_sr", "oos_sr", "selected", "omega", "oos_mean_sr")}
    joint = np.zeros((joint_bins, joint_bins)) if joint_bins else None
    oos_acc, is_acc = np.zeros(n_trials), np.zeros(n_trials)
    for a in range(0, len(m), chunk):
        mi = m[a:a + chunk]
        is_sr = _sharpe_from_sums(mi @ bs, mi @ bq, n_half)
        oos_sr = _sharpe_from_sums((1 - mi) @ bs, (1 - mi) @ bq, n_half)
        sel = is_sr.argmax(1)
        rows = np.arange(len(mi))
        sel_oos = oos_sr[rows, sel]
        # rank 1 = lowest OOS Sharpe; ties count half
        below = (oos_sr < sel_oos[:, None]).sum(1)
        equal = (oos_sr == sel_oos[:, None]).sum(1)
        rank = below + (equal + 1) / 2
        out["is_sr"].append(is_sr[rows, sel])
        out["oos_sr"].append(sel_oos)
        out["selected"].append(sel)
        out["omega"].append(rank / (n_trials + 1))
        out["oos_mean_sr"].append(oos_sr.mean(1))
        oos_acc += oos_sr.sum(0)
        is_acc += is_sr.sum(0)
        if joint is not None:
            ri = is_sr.argsort(1).argsort(1) / (n_trials - 1)
            ro = oos_sr.argsort(1).argsort(1) / (n_trials - 1)
            bi = np.minimum((ri * joint_bins).astype(int), joint_bins - 1)
            bo = np.minimum((ro * joint_bins).astype(int), joint_bins - 1)
            np.add.at(joint, (bi.ravel(), bo.ravel()), 1)
    cat = {k: np.concatenate(v) for k, v in out.items()}
    omega = cat["omega"]
    return CSCVResult(logit=np.log(omega / (1 - omega)), n_trials=n_trials, n_blocks=s,
                      joint=None if joint is None else joint / joint.sum(),
                      trial_oos_mean=oos_acc / len(m), trial_is_mean=is_acc / len(m), **cat)


def cscv_slow(returns: np.ndarray, s: int = 16) -> dict:
    """Reference implementation with explicit loops (used by tests on small cases)."""
    t, n = returns.shape
    blen = t // s
    x = np.asarray(returns[t - blen * s:], dtype=np.float64)
    blocks = [x[i * blen:(i + 1) * blen] for i in range(s)]
    omegas, is_s, oos_s = [], [], []
    for train in combinations(range(s), s // 2):
        test = [b for b in range(s) if b not in train]
        tr = np.vstack([blocks[b] for b in train])
        te = np.vstack([blocks[b] for b in test])
        sr_tr = tr.mean(0) / tr.std(0, ddof=1)
        sr_te = te.mean(0) / te.std(0, ddof=1)
        k = int(np.argmax(sr_tr))
        rank = (sr_te < sr_te[k]).sum() + ((sr_te == sr_te[k]).sum() + 1) / 2
        omegas.append(rank / (n + 1))
        is_s.append(sr_tr[k])
        oos_s.append(sr_te[k])
    omega = np.array(omegas)
    return {"omega": omega, "logit": np.log(omega / (1 - omega)), "is_sr": np.array(is_s), "oos_sr": np.array(oos_s),
            "pbo": float(np.mean(np.log(omega / (1 - omega)) <= 0))}
