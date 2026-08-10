"""The rewrite's hand-typed front-matter page numbers must match a real render.

The Table of Contents is a Word field and repaginates on open, but the List of
Figures and List of Tables are plain paragraphs, and the cached TOC entries are
what every non-Word reader sees. Nothing corrected or checked those numbers
before this test: any edit that changed body length drifted them silently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "report_tools" / "check_pagination.py"
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_pagination", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pagination"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Render the report once and share it across the tests in this module."""
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} is not present in this checkout")
    checker = load_checker()
    pytest.importorskip("fitz", reason="PyMuPDF is required to read the render")
    try:
        checker.find_soffice()
    except checker.PaginationError as exc:
        pytest.skip(str(exc))
    outdir = tmp_path_factory.mktemp("render")
    return checker, checker.render_pdf(REPORT, outdir)


def test_front_matter_page_numbers_match_the_render(rendered) -> None:
    checker, pdf = rendered
    report = checker.check(REPORT, pdf)
    assert report["mismatches"] == [], "\n".join(
        f"[{row['list']}] {row['title']}: says {row['claimed']}, is {row['actual']}"
        for row in report["mismatches"]
    )


def test_every_front_matter_entry_is_actually_found(rendered) -> None:
    """A renamed heading must fail loudly rather than silently pass."""
    checker, pdf = rendered
    report = checker.check(REPORT, pdf)
    missing = [row for row in report["entries"] if row["actual"] is None]
    assert not missing, f"front-matter entries with no matching heading: {missing}"


def test_checker_fires_on_a_wrong_page_number(rendered, monkeypatch) -> None:
    """Must-fire proof: a doctored claim has to be reported as a mismatch."""
    checker, pdf = rendered
    real = checker.claimed_entries()
    doctored = {kind: list(pairs) for kind, pairs in real.items()}
    title, page = doctored["tables"][0]
    doctored["tables"][0] = (title, str(int(page) + 7))

    monkeypatch.setattr(checker, "claimed_entries", lambda *a, **k: doctored)
    report = checker.check(REPORT, pdf)

    assert any(
        row["status"] == "mismatch" and row["title"] == title
        for row in report["mismatches"]
    ), "the pagination check did not fire on a deliberately wrong page number"


def test_checker_distinguishes_a_caption_from_a_cross_reference() -> None:
    """The probe must anchor at line start, or in-text mentions win the race."""
    checker = load_checker()
    texts = [
        "Benchmarking\n17\nThe controls are summarised in Table A.1 below.",
        "Benchmarking\n18\nTable A.1. Primary run controls",
    ]
    labels = checker.printed_labels(texts)
    assert checker.locate(texts, labels, "Table A.1.") == 18
