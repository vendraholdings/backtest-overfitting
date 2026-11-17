"""Render paper/paper.typ from the template and the results, then compile paper/paper.pdf.

    uv run python scripts/build_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overfit.facts import CAPTIONS, facts, fmt  # noqa: E402


def main() -> None:
    f = facts()
    for n, (title, cap) in CAPTIONS.items():
        f[f"cap{n}"] = f"*{title}.* {fmt(cap, f)}"
    src = fmt((ROOT / "paper" / "paper.typ.tmpl").read_text(), f)
    (ROOT / "paper" / "paper.typ").write_text(src)
    typst.compile(str(ROOT / "paper" / "paper.typ"), output=str(ROOT / "paper" / "paper.pdf"))
    print("paper/paper.pdf written")


if __name__ == "__main__":
    main()
