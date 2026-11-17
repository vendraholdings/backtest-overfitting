"""Compute every result from data/trials.parquet. Writes results/summary.json and results/*.parquet.

    uv run python scripts/run_all.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from overfit.cscv import cscv
from overfit.haircut import haircut_table
from overfit.sharpe import PERIODS_PER_YEAR, dsr, expected_max_sr, min_track_record_length, moments, psr, sharpe
from overfit.spa import spa_tests
from overfit.trials import (calibrated_effective_n, cluster_effective_n, correlation, eigen_effective_n, marchenko_pastur_count,
                            search_size_curve)
from overfit.validate import run_validation, summarise

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced bootstrap / replication counts (smoke test)")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "trials.parquet")
    cfg = pd.read_parquet(ROOT / "data" / "configs.parquet")
    names = [c for c in df.columns if c != "date"]
    R = df[names].to_numpy(np.float64)
    T, N = R.shape
    fam = cfg.set_index("trial").loc[names, "model"].to_numpy()
    out: dict = {"n_trials": N, "n_periods": T, "first_date": str(df["date"].min().date()),
                 "last_date": str(df["date"].max().date()), "periods_per_year": PERIODS_PER_YEAR,
                 "n_lgb": int((fam == "lgb").sum()), "n_mlx": int((fam == "mlx").sum())}

    # ---- the selected (best full-sample net Sharpe) trial -------------------------------------------------
    sr_ann = sharpe(R)
    sr_per = R.mean(0) / R.std(0, ddof=1)
    best = int(sr_ann.argmax())
    sr_b, sk_b, ku_b = moments(R[:, best])
    out["best"] = {"trial": names[best], "model": str(fam[best]), "sharpe_annual": float(sr_ann[best]),
                   "skew": sk_b, "kurtosis": ku_b, "psr_vs_zero": psr(sr_b, 0.0, T, sk_b, ku_b),
                   "min_track_record_periods": min_track_record_length(sr_b, 0.0, sk_b, ku_b),
                   "mean_net_return_per_period": float(R[:, best].mean())}
    out["sharpe_distribution"] = {"mean": float(sr_ann.mean()), "median": float(np.median(sr_ann)), "std": float(sr_ann.std(ddof=1)),
                                  "max": float(sr_ann.max()), "min": float(sr_ann.min()),
                                  "share_positive": float((sr_ann > 0).mean())}
    lag1 = [(r * np.roll(r, 1))[1:].sum() / (r**2).sum() for r in (R - R.mean(0)).T]
    out["lag1_autocorrelation"] = {"mean": float(np.mean(lag1)), "p5": float(np.percentile(lag1, 5)), "p95": float(np.percentile(lag1, 95))}

    # ---- effective number of independent trials -----------------------------------------------------------
    corr = correlation(R)
    iu = np.triu_indices(N, 1)
    eig, clu = eigen_effective_n(corr), cluster_effective_n(corr)
    n_mp = marchenko_pastur_count(corr, T)
    out["correlation"] = {"mean_pairwise": float(corr[iu].mean()), "median_pairwise": float(np.median(corr[iu])),
                          **eig, **{k: v for k, v in clu.items() if k != "order"}, "marchenko_pastur_components": n_mp}
    order = np.array(clu["order"])
    cc = corr[np.ix_(order, order)]
    edges = np.linspace(0, N, 151).astype(int)[:-1]  # 150 x 150 block-averaged correlation matrix for plotting
    pooled = np.add.reduceat(np.add.reduceat(cc, edges, axis=0), edges, axis=1)
    sizes_ = np.diff(np.append(edges, N))
    pd.DataFrame((pooled / np.outer(sizes_, sizes_)).astype(np.float32)).to_parquet(RES / "corr_pooled.parquet")
    cluster_rank = np.empty(N, dtype=int)
    cluster_rank[order] = np.arange(N)
    cal = calibrated_effective_n(R, reps=100 if a.quick else 1000, mean_block=2.0, seed=5)
    out["effective_n_calibrated"] = cal
    n_eff = {"nominal": N, "calibrated": cal["n_eff"], "cluster_rho_0.9": clu["clusters_rho_0.9"],
             "cluster_rho_0.7": clu["clusters_rho_0.7"]}

    # ---- DSR with nominal and effective N -----------------------------------------------------------------
    var_sr = float(np.var(sr_per, ddof=1))
    out["dsr"] = {"var_sr_per_period": var_sr}
    for label, n in n_eff.items():
        d, sr0 = dsr(R[:, best], sr_per, n)
        out["dsr"][label] = {"n": float(n), "dsr": d, "threshold_sharpe_annual": sr0 * np.sqrt(PERIODS_PER_YEAR),
                             "expected_max_sharpe_annual": expected_max_sr(n, var_sr) * np.sqrt(PERIODS_PER_YEAR)}

    from scipy.stats import norm as _norm
    sr0_mc = cal["null_expected_max_sr"]
    sd_b = np.sqrt((1 - sk_b * sr_b + (ku_b - 1) / 4 * sr_b**2) / (T - 1))
    out["dsr"]["simulated_null_threshold"] = {"dsr": float(_norm.cdf((sr_b - sr0_mc) / sd_b)),
                                              "threshold_sharpe_annual": cal["null_expected_max_sharpe_annual"]}

    # ---- CSCV ---------------------------------------------------------------------------------------------
    res = cscv(R, s=16, joint_bins=40)
    out["cscv"] = res.summary()
    pd.DataFrame({"is_sr": res.is_sr, "oos_sr": res.oos_sr, "selected": res.selected, "omega": res.omega, "logit": res.logit,
                  "oos_mean_sr": res.oos_mean_sr}).astype("float32").to_parquet(RES / "cscv_splits.parquet")
    pd.DataFrame(res.joint).to_parquet(RES / "cscv_joint.parquet")
    land = cfg.set_index("trial").loc[names].reset_index()
    land["sharpe_full"] = sr_ann
    land["oos_mean_sr_annual"] = res.trial_oos_mean * np.sqrt(PERIODS_PER_YEAR)
    land["is_mean_sr_annual"] = res.trial_is_mean * np.sqrt(PERIODS_PER_YEAR)
    land["cluster_rank"] = cluster_rank
    land["n_selected"] = np.bincount(res.selected, minlength=N)
    land.to_parquet(RES / "trial_landscape.parquet")
    sel_counts = pd.Series(np.bincount(res.selected, minlength=N), index=names).sort_values(ascending=False)
    out["cscv"]["distinct_selected_trials"] = int((sel_counts > 0).sum())
    out["cscv"]["top_selected_share"] = float(sel_counts.iloc[0] / sel_counts.sum())
    out["cscv"]["selected_family_share_lgb"] = float(np.mean(fam[res.selected] == "lgb"))
    sens = {}
    for s in (8, 12, 16, 20):
        r = cscv(R, s=s)
        sens[str(s)] = {"pbo": r.pbo, "n_splits": len(r.logit), "slope": r.summary()["degradation_slope"]}
    out["cscv_block_sensitivity"] = sens
    out["cscv_by_family"] = {f: cscv(R[:, fam == f], s=16).summary() | {"n": int((fam == f).sum())} for f in ("lgb", "mlx")}

    # ---- haircuts -----------------------------------------------------------------------------------------
    out["haircut"] = haircut_table(sr_per, T, PERIODS_PER_YEAR, m_eff=max(1, int(round(cal["n_eff"]))))

    # ---- Reality Check / SPA ------------------------------------------------------------------------------
    n_boot = 5000 if a.quick else 100_000
    spa = {}
    for L in (1, 2, 3, 6, 12):
        r = spa_tests(R, n_boot=n_boot, mean_block=L, seed=1)
        spa[str(L)] = r.__dict__ | {"best_trial": names[r.best_index]}
    out["spa"] = {"n_boot": n_boot, "headline_mean_block": 2, "by_block": spa}
    out["spa_by_family"] = {f: spa_tests(R[:, fam == f], n_boot=n_boot, mean_block=2, seed=2).__dict__ for f in ("lgb", "mlx")}

    # ---- size of the SPA test under the null, by bootstrap block length ----------------------------------
    from overfit import synthetic as syn
    reps_size = 20 if a.quick else 150
    size = {}
    for L in (1, 2, 3, 6):
        rej = [spa_tests(syn.noise(249, 300, np.random.default_rng(1000 + i)), n_boot=1500, mean_block=L, seed=i).p_spa_consistent < 0.05
               for i in range(reps_size)]
        size[str(L)] = float(np.mean(rej))
    out["spa_size_by_block"] = {"reps": reps_size, "n_trials": 300, "n_boot": 1500, "rejection_rate_5pct": size}

    # ---- search-size curve --------------------------------------------------------------------------------
    sizes = [10, 20, 50, 100, 200, 500, 1000, N] if not a.quick else [10, 100, N]
    curve = search_size_curve(R, sizes, reps=5 if a.quick else 30, seed=3, null_reps=100 if a.quick else 300)
    pd.DataFrame(curve).to_parquet(RES / "search_size_curve.parquet")
    out["search_size_curve"] = curve

    # ---- synthetic validation -----------------------------------------------------------------------------
    val = run_validation(reps=10 if a.quick else 100, spa_boot=1000 if a.quick else 2000)
    val.to_parquet(RES / "synthetic_runs.parquet")
    sm = summarise(val)
    sm.to_parquet(RES / "synthetic_summary.parquet")
    out["synthetic"] = sm.to_dict("records")

    out["runtime_seconds"] = round(time.time() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("best", "dsr", "cscv", "haircut")}, indent=1, default=float)[:4000])
    print("runtime", out["runtime_seconds"])


if __name__ == "__main__":
    main()
