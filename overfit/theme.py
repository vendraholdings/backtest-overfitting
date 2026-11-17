"""Shared figure theme: one palette and font for the site, the paper and the README."""
from __future__ import annotations

import matplotlib as mpl

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BG = "#ffffff"
LGB = "#1f6fb2"      # LightGBM family
MLX = "#c8553d"      # MLP family
ACCENT = "#2f7d6d"
SHADE = "#d9e2ec"
FAMILY = {"lgb": LGB, "mlx": MLX}
FAMILY_LABEL = {"lgb": "LightGBM", "mlx": "Neural network (MLP)"}
FONT = ["DejaVu Sans"]
SEQUENTIAL = [[0.0, "#f4f7fa"], [0.5, "#7fa8cf"], [1.0, "#1b3a5c"]]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 200,
        "lines.linewidth": 1.6,
    })


def plotly_layout(**kw) -> dict:
    base = dict(font=dict(family="DejaVu Sans, Helvetica, Arial, sans-serif", size=12, color=INK),
                paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=10, r=10, t=40, b=10))
    base.update(kw)
    return base
