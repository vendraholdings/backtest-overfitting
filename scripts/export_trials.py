"""Rebuild per-period net returns for every trial of the search and write data/trials.parquet.

Runs against the research repository that produced the trials:
    cd <research repo> && uv run python <this repo>/scripts/export_trials.py

Output
    data/trials.parquet  : date + one float32 column per trial (net long-short return per period)
    data/configs.parquet : one row per trial (id, model family, hyper-parameters, reported metrics)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from quant.research.xsec import backtest, H

RESULTS = Path.home() / "quant" / "data" / "results"
OUT = Path(__file__).resolve().parents[1] / "data"
COST_BPS = 10.0
TOL = 1e-6


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cols, rows, dates, max_err = {}, [], None, 0.0
    for model in ("lgb", "mlx"):
        for line in (RESULTS / f"{model}.jsonl").read_text().splitlines():
            m = json.loads(line)
            tid = m["id"]
            bt = backtest(pl.read_parquet(RESULTS / model / f"{tid}.parquet"), cost_bps=COST_BPS)
            d = bt["date"].to_list()
            if dates is None:
                dates = d
            if d != dates:
                raise SystemExit(f"date index mismatch for {tid}")
            r = bt["net"].to_numpy()
            sr = float(r.mean() / r.std(ddof=1) * math.sqrt(252 / H))
            err = abs(sr - m["sharpe_net"])
            max_err = max(max_err, err)
            if err > TOL:
                raise SystemExit(f"Sharpe mismatch for {tid}: {sr} vs {m['sharpe_net']}")
            cols[f"{model}_{tid}"] = r.astype(np.float32)
            row = {"trial": f"{model}_{tid}", "model": model}
            row |= {f"cfg_{k}": v for k, v in m["cfg"].items() if k != "id"}
            row |= {k: m[k] for k in ("mean_ic", "ic_ir", "sharpe_gross", "sharpe_net", "cagr_net", "maxdd_net", "periods")}
            rows.append(row)
    df = pd.DataFrame(cols)
    df.insert(0, "date", pd.to_datetime(dates))
    df.to_parquet(OUT / "trials.parquet", index=False)
    pd.DataFrame(rows).to_parquet(OUT / "configs.parquet", index=False)
    print(f"trials={len(cols)} periods={len(dates)} max |Sharpe error|={max_err:.2e}")


if __name__ == "__main__":
    main()
