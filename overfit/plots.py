"""Figures 1-8 (matplotlib, static) and the interactive 3D figures (Plotly). All read from results/ and data/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import theme  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANN = np.sqrt(252 / 21)
CMAP = LinearSegmentedColormap.from_list("seq", [(p, c) for p, c in theme.SEQUENTIAL])


def load() -> dict:
    res = ROOT / "results"
    return {"summary": json.loads((res / "summary.json").read_text()),
            "splits": pd.read_parquet(res / "cscv_splits.parquet"),
            "joint": pd.read_parquet(res / "cscv_joint.parquet").to_numpy(),
            "land": pd.read_parquet(res / "trial_landscape.parquet"),
            "curve": pd.read_parquet(res / "search_size_curve.parquet"),
            "syn": pd.read_parquet(res / "synthetic_summary.parquet"),
            "corr": pd.read_parquet(res / "corr_pooled.parquet").to_numpy()}


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig1(d, out):
    s = d["splits"]
    x, y = s.is_sr * ANN, s.oos_sr * ANN
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    hb = ax.hexbin(x, y, gridsize=45, cmap=CMAP, mincnt=1, linewidths=0.2)
    lim = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    ax.plot(lim, lim, color=theme.MUTED, ls="--", lw=1, label="OOS = IS")
    b, a = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color=theme.MLX, label=f"Regression slope {b:.2f}")
    ax.set(xlabel="In-sample Sharpe of the selected trial (annualised)", ylabel="Out-of-sample Sharpe (annualised)")
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Number of splits", pad=0.02)
    _save(fig, out, "fig1_is_vs_oos")


def fig2(d, out):
    lam = d["splits"].logit.to_numpy()
    pbo = d["summary"]["cscv"]["pbo"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bins = np.linspace(lam.min(), lam.max(), 36)
    ax.hist(lam[lam > 0], bins=bins, color=theme.SHADE, edgecolor="white", lw=0.4, label="λ > 0")
    ax.hist(lam[lam <= 0], bins=bins, color=theme.MLX, edgecolor="white", lw=0.4, label=f"λ ≤ 0 (PBO = {pbo:.3f})")
    ax.axvline(0, color=theme.INK, lw=1)
    ax.set(xlabel="Logit λ of the out-of-sample rank of the selected trial", ylabel="Number of splits")
    ax.legend()
    _save(fig, out, "fig2_logit_hist")


def joint_grid(d):
    j = np.log2(np.maximum(d["joint"] * d["joint"].size, 2.0**-6))  # log2 of density relative to independence
    g = (np.arange(j.shape[0]) + 0.5) / j.shape[0]
    return g, j


def fig3(d, out, elev=28, azim=-55, name="fig3_rank_surface", size=(6.0, 4.6)):
    g, j = joint_grid(d)
    X, Y = np.meshgrid(g, g, indexing="ij")
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, j, cmap=CMAP, linewidth=0.2, edgecolor="#ffffff", antialiased=True, rstride=1, cstride=1)
    ax.set(xlabel="In-sample rank", ylabel="Out-of-sample rank", zlabel="log2 density ratio")
    ax.zaxis.labelpad = -2
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True)
    if name:
        _save(fig, out, name)
    return fig


def _hyper_axes(land: pd.DataFrame, family: str):
    """Two numeric hyper-parameters most associated with OOS Sharpe within a family."""
    sub = land[land.model == family]
    cols = [c for c in sub.columns if c.startswith("cfg_") and c != "cfg_seed" and pd.api.types.is_numeric_dtype(sub[c]) and sub[c].nunique() > 1]
    score = {c: abs(sub[c].rank().corr(sub["oos_mean_sr_annual"].rank())) for c in cols}
    return sorted(score, key=score.get, reverse=True)[:2], sub


def fig4(d, out):
    land = d["land"]
    fig = plt.figure(figsize=(9.6, 4.4))
    for i, fam in enumerate(("lgb", "mlx")):
        (cx, cy), sub = _hyper_axes(land, fam)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        rng = np.random.default_rng(0)
        def f(c):
            v = sub[c].astype(float)
            return (np.log10(v), " (log10)") if (v > 0).all() and v.max() / v.min() >= 10 else (v, "")
        (xv, xl), (yv, yl) = f(cx), f(cy)
        jx = rng.normal(0, 0.012 * (xv.max() - xv.min()), len(sub))
        jy = rng.normal(0, 0.012 * (yv.max() - yv.min()), len(sub))
        ax.scatter(xv + jx, yv + jy, sub.oos_mean_sr_annual, s=5, color=theme.FAMILY[fam], alpha=0.5, depthshade=True)
        ax.set(xlabel=cx[4:] + xl, ylabel=cy[4:] + yl, zlabel="Mean OOS Sharpe")
        ax.set_title(theme.FAMILY_LABEL[fam], loc="left")
        ax.view_init(elev=24, azim=-60)
    _save(fig, out, "fig4_landscape")


def fig5(d, out):
    c = d["curve"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a1.plot(c.n_trials, c.best_sharpe_mean, color=theme.LGB, marker="o", ms=3.5, label="Observed best of n trials")
    a1.fill_between(c.n_trials, c.best_sharpe_lo, c.best_sharpe_hi, color=theme.LGB, alpha=0.15, lw=0)
    a1.plot(c.n_trials, c.null_best_sharpe_mean, color=theme.MLX, marker="o", ms=3.5, label="Zero-skill expectation (same correlation)")
    a1.fill_between(c.n_trials, c.null_best_sharpe_lo, c.null_best_sharpe_hi, color=theme.MLX, alpha=0.15, lw=0)
    a1.set(xscale="log", xlabel="Number of trials searched, n", ylabel="Best full-sample Sharpe (annualised)", ylim=(-0.3, 1.15))
    a1.legend(loc="center right")
    a2.plot(c.n_trials, c.pbo_mean, color=theme.INK, marker="o", ms=3.5)
    a2.fill_between(c.n_trials, c.pbo_lo, c.pbo_hi, color=theme.INK, alpha=0.12, lw=0)
    a2.axhline(0.5, color=theme.MUTED, ls="--", lw=1)
    a2.set(xscale="log", xlabel="Number of trials searched, n", ylabel="PBO (CSCV, S = 16)", ylim=(0, 1))
    _save(fig, out, "fig5_search_size")


def fig6(d, out):
    s = d["summary"]
    hc, dsr = s["haircut"], s["dsr"]
    rows = [("Selected on full sample", s["best"]["sharpe_annual"], theme.INK),
            ("After Bonferroni / Holm haircut (1,979 tests)", hc["nominal_bonferroni"]["sharpe_adj"], theme.LGB),
            ("After BHY haircut (1,979 tests)", hc["nominal_bhy"]["sharpe_adj"], theme.LGB),
            ("Excess over DSR threshold (expected maximum)", s["best"]["sharpe_annual"] - dsr["nominal"]["threshold_sharpe_annual"], theme.LGB),
            ("Excess over simulated zero-skill maximum", s["best"]["sharpe_annual"] - dsr["simulated_null_threshold"]["threshold_sharpe_annual"], theme.LGB),
            ("Average trial, full sample", s["sharpe_distribution"]["mean"], theme.MUTED),
            ("CSCV: selected trial, in sample", s["cscv"]["mean_is_sr"] * ANN, theme.MLX),
            ("CSCV: selected trial, out of sample", s["cscv"]["mean_oos_sr"] * ANN, theme.MLX),
            ("CSCV: average trial, out of sample", s["cscv"]["mean_oos_sr_all_trials"] * ANN, theme.MUTED)]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.plot([0, v], [yi, yi], color=theme.GRID, lw=1.2, zorder=1)
        ax.scatter([v], [yi], color=col, s=42, zorder=2)
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color=theme.INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set(xlabel="Annualised Sharpe ratio", xlim=(0, 1.3))
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig6_ladder")


def fig7(d, out):
    syn = d["syn"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3), sharex=True)
    spec = [("pbo", "PBO (mean)"), ("dsr_gt_95", "Share of runs with DSR > 0.95"), ("spa_reject_5pct", "SPA rejection rate (5 % level)")]
    for ax, (col, lab) in zip(axs, spec):
        for scen, colr, nm in (("independent", theme.LGB, "300 independent trials"), ("families", theme.MLX, "15 families of 20 (ρ = 0.8)")):
            s = syn[syn.scenario == scen]
            ax.plot(s.planted_sharpe, s[col], marker="o", ms=4, color=colr, label=nm)
        ax.set(xlabel="Planted annual Sharpe ratio", ylabel=lab, ylim=(-0.02, 1.02))
    axs[0].axhline(0.5, color=theme.MUTED, ls="--", lw=1)
    axs[2].axhline(0.05, color=theme.MUTED, ls="--", lw=1)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig7_synthetic")


def fig8(d, out):
    s = d["summary"]["correlation"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(d["corr"], cmap=CMAP, vmin=0.4, vmax=1.0, interpolation="nearest")
    ax.set(xlabel="Trials (hierarchical cluster order)", ylabel="Trials (hierarchical cluster order)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Correlation of net returns (block average)", pad=0.02)
    ax.set_title(f"Mean pairwise correlation {s['mean_pairwise']:.2f}")
    _save(fig, out, "fig8_correlation")


def all_figures(out: Path | None = None) -> None:
    theme.apply()
    out = out or ROOT / "paper" / "figures"
    d = load()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f(d, out)
