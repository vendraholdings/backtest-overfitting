"""Single source of every number quoted in the README, the paper and the site, plus the figure captions."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = (252 / 21) ** 0.5


def _p(x: float) -> str:
    if x < 1e-3:
        return "< 0.001"
    return f"{x:.3f}"


def facts(summary: dict | None = None) -> dict:
    s = summary or json.loads((ROOT / "results" / "summary.json").read_text())
    b, d, c, h = s["best"], s["dsr"], s["cscv"], s["haircut"]
    sd, co, sp = s["sharpe_distribution"], s["correlation"], s["spa"]["by_block"]["2"]
    curve = {r["n_trials"]: r for r in s["search_size_curve"]}
    syn = {(r["scenario"], r["planted_sharpe"]): r for r in s["synthetic"]}
    sens = s["cscv_block_sensitivity"]
    f = {
        "n_trials": f"{s['n_trials']:,}", "n_lgb": f"{s['n_lgb']:,}", "n_mlx": f"{s['n_mlx']:,}", "n_periods": str(s["n_periods"]),
        "first_date": s["first_date"], "last_date": s["last_date"],
        "mean_sr": f"{sd['mean']:.2f}", "sd_sr": f"{sd['std']:.2f}", "min_sr": f"{sd['min']:.2f}", "median_sr": f"{sd['median']:.2f}",
        "best_sr": f"{b['sharpe_annual']:.2f}", "best_skew": f"{b['skew']:.2f}", "best_kurt": f"{b['kurtosis']:.1f}",
        "best_mintrl": f"{b['min_track_record_periods']:.0f}",
        "corr_mean": f"{co['mean_pairwise']:.2f}", "top_eig": f"{100 * co['top_eigenvalue_share']:.0f}",
        "clusters_09": str(co["clusters_rho_0.9"]), "clusters_07": str(co["clusters_rho_0.7"]), "mp_components": str(co["marchenko_pastur_components"]),
        "part_ratio": f"{co['participation_ratio']:.1f}",
        "n_eff": f"{s['effective_n_calibrated']['n_eff']:,.0f}",
        "null_max": f"{d['simulated_null_threshold']['threshold_sharpe_annual']:.2f}",
        "dsr": f"{d['nominal']['dsr']:.3f}", "dsr_thr": f"{d['nominal']['threshold_sharpe_annual']:.2f}",
        "dsr_null": f"{d['simulated_null_threshold']['dsr']:.4f}",
        "dsr_c09": f"{d['cluster_rho_0.9']['dsr']:.4f}",
        "hc_bonf": f"{h['nominal_bonferroni']['sharpe_adj']:.2f}", "hc_bonf_pct": f"{100 * h['nominal_bonferroni']['haircut']:.0f}",
        "hc_bhy": f"{h['nominal_bhy']['sharpe_adj']:.2f}", "hc_bhy_pct": f"{100 * h['nominal_bhy']['haircut']:.0f}",
        "p_raw": f"{h['p_raw']:.1e}", "p_bonf": f"{h['nominal_bonferroni']['p_adj']:.4f}",
        "n_sig_bonf": str(h["nominal_bonferroni"]["n_significant_5pct"]), "n_sig_holm": str(h["nominal_holm"]["n_significant_5pct"]),
        "n_sig_bhy": f"{h['nominal_bhy']['n_significant_5pct']:,}",
        "p_rc": _p(sp["p_rc"]), "p_spa": _p(sp["p_spa_consistent"]), "n_boot": f"{s['spa']['n_boot']:,}",
        "spa_block_note": "all mean block lengths from 1 to 12 give p < 0.001" if all(v["p_spa_consistent"] < 1e-3 for v in s["spa"]["by_block"].values()) else "see results/summary.json",
        "pbo": f"{c['pbo']:.2f}", "pbo3": f"{c['pbo']:.3f}", "n_splits": f"{c['n_splits']:,}",
        "pbo_lgb": f"{s['cscv_by_family']['lgb']['pbo']:.2f}", "pbo_mlx": f"{s['cscv_by_family']['mlx']['pbo']:.2f}",
        "slope": f"{c['degradation_slope']:.2f}", "is_sel": f"{c['mean_is_sr'] * ANN:.2f}", "oos_sel": f"{c['mean_oos_sr'] * ANN:.2f}",
        "oos_avg": f"{c['mean_oos_sr_all_trials'] * ANN:.2f}", "p_oos_loss": f"{c['prob_oos_loss']:.3f}",
        "n_selected": str(c["distinct_selected_trials"]), "top_sel_share": f"{100 * c['top_selected_share']:.1f}",
        "pbo_s8": f"{sens['8']['pbo']:.2f}", "pbo_s12": f"{sens['12']['pbo']:.2f}", "pbo_s20": f"{sens['20']['pbo']:.2f}",
        "curve_best_10": f"{curve[10]['best_sharpe_mean']:.2f}", "curve_best_all": f"{curve[s['n_trials']]['best_sharpe_mean']:.2f}",
        "curve_null_10": f"{curve[10]['null_best_sharpe_mean']:.2f}", "curve_null_all": f"{curve[s['n_trials']]['null_best_sharpe_mean']:.2f}",
        "curve_pbo_min": f"{min(r['pbo_mean'] for r in curve.values()):.2f}", "curve_pbo_max": f"{max(r['pbo_mean'] for r in curve.values()):.2f}",
        "spa_size_reps": str(s["spa_size_by_block"]["reps"]),
        **{f"spa_size_{L}": f"{100 * v:.0f} %" for L, v in s["spa_size_by_block"]["rejection_rate_5pct"].items()},
        "lag1": f"{s['lag1_autocorrelation']['mean']:.2f}",
        "syn_noise_pbo": f"{syn[('independent', 0.0)]['pbo']:.2f}", "syn_noise_dsr95": f"{syn[('independent', 0.0)]['dsr_gt_95']:.2f}",
        "syn_pbo_10": f"{syn[('independent', 1.0)]['pbo']:.2f}", "syn_pbo_15": f"{syn[('independent', 1.5)]['pbo']:.3f}",
        "syn_pbo_20": f"{syn[('independent', 2.0)]['pbo']:.3f}", "syn_dsr95_10": f"{syn[('independent', 1.0)]['dsr_gt_95']:.2f}",
        "syn_dsr95_15": f"{syn[('independent', 1.5)]['dsr_gt_95']:.2f}", "syn_dsr95_20": f"{syn[('independent', 2.0)]['dsr_gt_95']:.2f}",
        "syn_spa_size": f"{syn[('independent', 0.0)]['spa_reject_5pct']:.2f}", "syn_rc_size": f"{syn[('independent', 0.0)]['rc_reject_5pct']:.2f}",
        "syn_fam_pbo_10": f"{syn[('families', 1.0)]['pbo']:.2f}", "syn_fam_noise_pbo": f"{syn[('families', 0.0)]['pbo']:.2f}",
    }
    return f


def fmt(text: str, f: dict | None = None, left: str = "<<", right: str = ">>") -> str:
    """Replace <<key>> placeholders; unknown keys raise."""
    f = f or facts()
    pat = re.compile(re.escape(left) + r"\s*([A-Za-z0-9_]+)\s*" + re.escape(right))
    def sub(m):
        if m.group(1) not in f:
            raise KeyError(m.group(1))
        return f[m.group(1)]
    return pat.sub(sub, text)


CAPTIONS = {
    1: ("In-sample versus out-of-sample Sharpe ratio of the selected trial",
        "Annualised Sharpe ratio of the trial with the highest in-sample Sharpe ratio against its Sharpe ratio on the held-out blocks, "
        "for all <<n_splits>> CSCV splits (S = 16). The fitted slope is <<slope>>: splits with a higher in-sample value have a lower "
        "out-of-sample value. Part of this slope is mechanical, because the two halves of a split partition one fixed sample."),
    2: ("Distribution of the logit of the out-of-sample rank",
        "Logit of the relative out-of-sample rank of the selected trial across <<n_splits>> splits. Splits with λ ≤ 0 (shaded) are those "
        "in which the selected trial ranks at or below the median out of sample; their share is the PBO, <<pbo3>>."),
    3: ("Joint density of in-sample and out-of-sample rank",
        "Density of a trial's in-sample rank percentile and out-of-sample rank percentile over all <<n_trials>> trials and all splits, relative to "
        "independence (log base 2, floored at 2^-6). The mass concentrated at low ranks on both axes shows that the weakest trials are "
        "consistently weak, while the upper part of the ranking is close to independent between the two halves."),
    4: ("Hyper-parameter landscape",
        "Mean out-of-sample Sharpe ratio of each trial, plotted against the two hyper-parameters most rank-correlated with it within each "
        "model family. Small random offsets separate overlapping grid points."),
    5: ("Effect of the size of the search",
        "Left: best full-sample Sharpe ratio among n randomly drawn trials (blue) and the expected best Sharpe ratio of n trials with zero "
        "skill and the same correlation structure (red); bands are 5th to 95th percentiles. Right: PBO of the same subsets. The best observed Sharpe ratio "
        "rises from <<curve_best_10>> to <<curve_best_all>> as n grows from 10 to <<n_trials>>, while PBO stays between <<curve_pbo_min>> and <<curve_pbo_max>>."),
    6: ("Sharpe ratio after each adjustment",
        "Annualised Sharpe ratio of the selected trial after multiple-testing haircuts and after subtracting the deflation thresholds, "
        "with the in-sample and out-of-sample CSCV averages for the selected trial and for the average trial."),
    7: ("Synthetic validation",
        "Mean PBO, share of runs with a deflated Sharpe ratio above 0.95, and SPA rejection rate at the 5 % level, as a function of the planted "
        "annual Sharpe ratio. Blue: 300 independent trials. Red: 15 families of 20 trials with within-family correlation 0.8. T = 249 in all cases."),
    8: ("Correlation structure of the trials",
        "Correlation of net returns between trials, ordered by hierarchical clustering and averaged in 150 × 150 blocks. "
        "The mean pairwise correlation is <<corr_mean>> and the first principal component carries <<top_eig>> % of the variance."),
}
