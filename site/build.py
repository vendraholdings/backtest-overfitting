"""Build the static site into site/dist: one page, Plotly (bundled, no CDN) for the 3D figures.

    uv run python site/build.py   (or: make site)
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overfit import theme  # noqa: E402
from overfit.facts import CAPTIONS, facts, fmt  # noqa: E402
from overfit.plots import _hyper_axes, joint_grid, load  # noqa: E402

DIST = ROOT / "site" / "dist"
REPO = "https://github.com/vendraholdings/backtest-overfitting"


def fig3_plotly(d) -> str:
    g, j = joint_grid(d)
    fig = go.Figure(go.Surface(x=g, y=g, z=j.T, colorscale=theme.SEQUENTIAL, showscale=True,
                               colorbar=dict(title="log2 ratio", thickness=12, len=0.6),
                               hovertemplate="IS rank %{x:.2f}<br>OOS rank %{y:.2f}<br>log2 density ratio %{z:.2f}<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(height=560, scene=dict(
        xaxis_title="In-sample rank", yaxis_title="Out-of-sample rank", zaxis_title="log2 density ratio",
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))), margin=dict(l=0, r=0, t=10, b=0)))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="fig3-plot", config=dict(displaylogo=False))


def fig4_plotly(d, fam: str) -> str:
    land = d["land"]
    (cx, cy), sub = _hyper_axes(land, fam)
    rng = np.random.default_rng(0)
    def f(c):
        v = sub[c].astype(float)
        return (np.log10(v), c[4:] + " (log10)") if (v > 0).all() and v.max() / v.min() >= 10 else (v.to_numpy(), c[4:])
    (xv, xl), (yv, yl) = f(cx), f(cy)
    jx = rng.normal(0, 0.012 * np.ptp(xv), len(sub))
    jy = rng.normal(0, 0.012 * np.ptp(yv), len(sub))
    text = [f"{r.trial}<br>full-sample Sharpe {r.sharpe_full:.2f}<br>mean OOS Sharpe {r.oos_mean_sr_annual:.2f}" for r in sub.itertuples()]
    fig = go.Figure(go.Scatter3d(x=xv + jx, y=yv + jy, z=sub.oos_mean_sr_annual, mode="markers", text=text, hoverinfo="text",
                                 marker=dict(size=2.6, color=theme.FAMILY[fam], opacity=0.6)))
    fig.update_layout(**theme.plotly_layout(height=440, scene=dict(xaxis_title=xl, yaxis_title=yl, zaxis_title="Mean OOS Sharpe"),
                                            margin=dict(l=0, r=0, t=10, b=0)))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"fig4-{fam}", config=dict(displaylogo=False))


def figure_block(n: int, body: str, f: dict) -> str:
    title, cap = CAPTIONS[n]
    return (f'<figure id="fig{n}">{body}<figcaption><strong>Figure {n}.</strong> {html.escape(title)}. '
            f'{html.escape(fmt(cap, f))}</figcaption></figure>')


def img(name: str, alt: str) -> str:
    return f'<img src="figures/{name}.png" alt="{html.escape(alt)}" loading="lazy">'


def build() -> None:
    f, d = facts(), load()
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "figures").mkdir(parents=True)
    for p in (ROOT / "paper" / "figures").glob("*.png"):
        shutil.copy(p, DIST / "figures" / p.name)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", DIST / "plotly.min.js")
    shutil.copy(ROOT / "results" / "summary.json", DIST / "summary.json")
    for extra in ("paper/paper.pdf",):
        if (ROOT / extra).exists():
            shutil.copy(ROOT / extra, DIST / Path(extra).name)
    paper = '<a href="paper.pdf">Paper (PDF)</a>' if (DIST / "paper.pdf").exists() else ""
    cards = [(f["pbo"], "Probability of backtest overfitting (CSCV)"), (f["best_sr"], "Sharpe ratio of the selected configuration"),
             (f["hc_bonf"], "Sharpe ratio after Bonferroni / Holm haircut"), (f["oos_sel"] + " vs " + f["oos_avg"], "Out-of-sample Sharpe: selected vs average trial")]
    cards_html = "".join(f'<div class="card"><div class="num">{html.escape(a)}</div><div class="lab">{html.escape(b)}</div></div>' for a, b in cards)
    body = {
        3: figure_block(3, fig3_plotly(d), f),
        1: figure_block(1, img("fig1_is_vs_oos", "Hexbin of in-sample against out-of-sample Sharpe"), f),
        2: figure_block(2, img("fig2_logit_hist", "Histogram of the logit of the out-of-sample rank"), f),
        4: figure_block(4, '<div class="two">' + fig4_plotly(d, "lgb") + fig4_plotly(d, "mlx") + "</div>", f),
        5: figure_block(5, img("fig5_search_size", "Best Sharpe and PBO against search size"), f),
        6: figure_block(6, img("fig6_ladder", "Sharpe ratio after each adjustment"), f),
        7: figure_block(7, img("fig7_synthetic", "Synthetic validation"), f),
        8: figure_block(8, img("fig8_correlation", "Clustered correlation matrix"), f),
    }
    page = TEMPLATE
    repl = {"CARDS": cards_html, "PAPER_LINK": paper, "REPO": REPO, **{f"FIG{k}": v for k, v in body.items()}}
    for k, v in repl.items():
        page = page.replace("{{" + k + "}}", v)
    page = fmt(page, f)
    (DIST / "index.html").write_text(page)
    print("site written to", DIST)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="color-scheme" content="light"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest overfitting</title>
<meta name="description" content="Overfitting diagnostics applied to a search over <<n_trials>> equity-ranking models.">
<style>
:root{--ink:#1f2933;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;--soft:#f4f7fa;--blue:#1f6fb2;--accent:#c8553d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "DejaVu Sans",Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:0 16px 64px}
header{padding:40px 0 8px}h1{font-size:1.9rem;line-height:1.25;margin:0 0 8px}h2{font-size:1.25rem;margin:40px 0 8px;border-top:1px solid var(--line);padding-top:24px}
p.lead{color:var(--muted);margin:0 0 16px;max-width:760px}nav a{margin-right:16px;color:var(--blue);text-decoration:none}nav a:hover{text-decoration:underline}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:24px 0}
.card{background:var(--soft);border-radius:6px;padding:14px 16px}.num{font-size:1.6rem;font-weight:600}.lab{color:var(--muted);font-size:.85rem}
figure{margin:20px 0}figure img{max-width:100%;height:auto}figcaption{color:var(--muted);font-size:.88rem;max-width:820px;margin-top:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}@media(max-width:720px){.two{grid-template-columns:1fr}}
details{border:1px solid var(--line);border-radius:6px;padding:8px 14px;margin:10px 0}summary{cursor:pointer;font-weight:600}
code{background:var(--soft);padding:1px 5px;border-radius:3px}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
footer{color:var(--muted);font-size:.85rem;margin-top:48px}
</style><script src="plotly.min.js"></script></head><body><main>
<header><h1>How much of a backtest is luck? Selection bias in a <<n_trials>>-model equity search</h1>
<p class="lead">A search over <<n_trials>> machine-learning configurations (<<n_lgb>> gradient-boosted trees, <<n_mlx>> neural networks) produced a best
net Sharpe ratio of <<best_sr>>. Standard overfitting diagnostics are applied to the full set of trials, and checked on synthetic data with a known answer.</p>
<nav><a href="#results">Results</a><a href="#figures">Figures</a><a href="#methods">Methods</a>{{PAPER_LINK}}<a href="{{REPO}}">Source code</a></nav></header>
<div class="cards">{{CARDS}}</div>
<h2 id="results">Summary of results</h2>
<p>The trials share one underlying signal. Every configuration has a positive net Sharpe ratio (mean <<mean_sr>>, standard deviation <<sd_sr>>), the Reality Check and SPA tests reject
the null of no skill (p <<p_spa>>), and the best configuration remains at <<hc_bonf>> after a Bonferroni or Holm haircut for <<n_trials>> tests.
The choice among configurations is far less informative. Across <<n_splits>> combinatorially symmetric cross-validation splits the configuration with the best in-sample Sharpe ratio
(<<is_sel>> on average) has an out-of-sample Sharpe ratio of <<oos_sel>>, against <<oos_avg>> for an average configuration; PBO is <<pbo>>.</p>
<h2 id="figures">Figures</h2>
{{FIG3}}{{FIG1}}{{FIG2}}{{FIG4}}{{FIG5}}{{FIG6}}{{FIG7}}{{FIG8}}
<h2 id="methods">Methods</h2>
<details><summary>Data and search</summary><p>The trials come from a walk-forward search of cross-sectional return forecasts for large-cap US equities: <<n_periods>> monthly rebalances
from <<first_date>> to <<last_date>>, long the top and short the bottom decile of predicted returns, net of 10 basis points per trade. The universe uses current index members, which
flatters absolute returns; the analysis concerns selection among the trials, not tradable performance.</p></details>
<details><summary>PBO by combinatorially symmetric cross-validation</summary><p>The 249 periods are cut into S = 16 blocks. Each of the C(16, 8) = <<n_splits>> choices of eight blocks is a training set and the complement the test set.
The trial with the highest training Sharpe ratio is selected and its relative rank among all trials on the test set gives the logit λ. PBO is the share of splits with λ ≤ 0 (Bailey, Borwein, López de Prado and Zhu, 2017).</p></details>
<details><summary>Deflated Sharpe ratio and haircuts</summary><p>The deflated Sharpe ratio compares the best Sharpe ratio with the expected maximum of <<n_trials>> trials under no skill (threshold <<dsr_thr>>), adjusting for skewness and kurtosis
(Bailey and López de Prado, 2014). Haircuts follow Harvey and Liu (2015) with Bonferroni, Holm and Benjamini-Hochberg-Yekutieli adjustments.</p></details>
<details><summary>Reality Check and SPA</summary><p>White's Reality Check and Hansen's test for superior predictive ability use the stationary bootstrap (mean block length 2, <<n_boot>> resamples) against a zero-return benchmark.</p></details>
<details><summary>Limitations</summary><p>Lag-1 autocorrelation of trial returns averages <<lag1>>, so annualising by the square root of 12 is conservative. The CSCV slope is partly mechanical. The universe is survivorship-biased.
Full discussion in the paper.</p></details>
<footer>Source code and data: <a href="{{REPO}}">{{REPO}}</a>. Released under the MIT licence.</footer>
</main>
</body></html>
"""

if __name__ == "__main__":
    build()
