"""Consistency of the generated documents (run after `make results paper readme`)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README, PAPER, SITE = ROOT / "README.md", ROOT / "paper" / "paper.typ", ROOT / "site" / "dist" / "index.html"


def _numbers(text: str) -> set[str]:
    text = text.split("## References")[0]                   # bibliographic volumes and pages
    text = re.sub(r"```.*?```", "", text, flags=re.S)       # code blocks
    text = re.sub(r"\(https?://[^)]*\)|<https?://[^>]*>|https?://\S+", "", text)  # urls
    text = re.sub(r"\([^)]*\.(?:gif|pdf|png)\)", "", text)  # file links
    text = re.sub(r"\b(?:19|20)\d\d\b", "", text)           # years in references
    return set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)) - {"1", "2", "3", "4", "5", "8", "16", "10", "70", "20", "12"}


@pytest.mark.skipif(not (README.exists() and PAPER.exists()), reason="documents not generated")
def test_readme_numbers_appear_in_paper():
    paper = PAPER.read_text()
    missing = sorted(n for n in _numbers(README.read_text()) if n not in paper)
    assert not missing, missing


@pytest.mark.skipif(not (PAPER.exists() and SITE.exists()), reason="site not built")
def test_site_has_no_unresolved_placeholders():
    html = SITE.read_text()
    assert "<<" not in html and "{{" not in html
    assert html.count("<figure") == 8 and "fig3-plot" in html


@pytest.mark.skipif(not PAPER.exists(), reason="paper not generated")
def test_paper_has_no_unresolved_placeholders():
    assert "<<" not in PAPER.read_text()
