"""Cross-check overfit.sharpe.dsr against the research repository's deflated_sharpe().

Run inside the research repository's environment with this repo on the path:
    cd <research repo> && PYTHONPATH=<this repo> uv run python <this repo>/scripts/crosscheck_quant.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from quant.research.xsec import deflated_sharpe

from overfit.sharpe import PERIODS_PER_YEAR, dsr

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "trials.parquet")
R = df.drop(columns="date").to_numpy(np.float64)
sr = R.mean(0) / R.std(0, ddof=1)
worst = 0.0
for k in np.argsort(sr)[::-1][:25]:
    p_ref, thr_ref = deflated_sharpe(R[:, k], sr)
    p, sr0 = dsr(R[:, k], sr)
    worst = max(worst, abs(p - p_ref), abs(sr0 * np.sqrt(PERIODS_PER_YEAR) - thr_ref))
print(f"max |difference| over the 25 best trials: {worst:.2e}")
assert worst < 1e-6
