"""Render figures 1-8: light into paper/figures (for the paper), dark into docs/figures (README and site).

    uv run python scripts/make_figures.py            # both
    uv run python scripts/make_figures.py dark       # one mode only
"""
import sys

from overfit.plots import all_figures

modes = sys.argv[1:] or ["light", "dark"]
for m in modes:
    all_figures(mode=m)
    print(f"figures ({m}) written")
