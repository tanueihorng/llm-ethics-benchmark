"""The rewrite's hand-typed front-matter page numbers must match a real render.

The Table of Contents is a Word field and repaginates on open, but the List of
Figures and List of Tables are plain paragraphs, and the cached TOC entries are
what every non-Word reader sees. Microsoft Word is the submission authority;
the checker records the few verified Word/LibreOffice pagination differences
explicitly instead of silently accepting either renderer.
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


def test_word_authority_is_explicit_and_must_fire(rendered, monkeypatch) -> None:
    """Known renderer differences remain narrow and still reject a wrong claim."""
    checker, pdf = rendered
    expected = {
        ("contents", "4.7 Research-Question Synthesis"): 26,
        ("contents", "6.2 Relationship to Prior Work"): 35,
        ("contents", "6.4 Scorer Validity as a Research Result"): 38,
        ("contents", "6.5 Threats to Validity"): 39,
        ("figures", "Figure 4.2. Precision sweep across fp16, INT8, and NF4"): 26,
    }
    assert checker.WORD_AUTHORITATIVE_PAGES == expected

    real = checker.claimed_entries()
    doctored = {kind: list(pairs) for kind, pairs in real.items()}
    title = "4.7 Research-Question Synthesis"
    doctored["contents"] = [
        (entry, "99" if entry == title else page)
        for entry, page in doctored["contents"]
    ]
    monkeypatch.setattr(checker, "claimed_entries", lambda *a, **k: doctored)
    report = checker.check(REPORT, pdf)
    row = next(entry for entry in report["entries"] if entry["title"] == title)
    assert row["authority"] == "microsoft_word"
    assert row["actual"] == 26
    assert row["status"] == "mismatch"


def test_checker_distinguishes_a_caption_from_a_cross_reference() -> None:
    """The probe must anchor at line start, or in-text mentions win the race."""
    checker = load_checker()
    texts = [
        "Benchmarking\n17\nThe controls are summarised in Table A.1 below.",
        "Benchmarking\n18\nTable A.1. Primary run controls",
    ]
    labels = checker.printed_labels(texts)
    assert checker.locate(texts, labels, "Table A.1.") == 18


def test_line_initial_cross_reference_does_not_win_over_the_caption() -> None:
    """T57: line-start position alone is a property of the current line breaks.

    The Chapter 3-4 flow pass reflowed a sentence ending "in Table A.1." onto a
    line start, and the checker reported Table A.1 on page 18 instead of 46 -- a
    false mismatch with no pagination error behind it. The workaround at the time
    was to reword the sentence, which left the gate green for a reason any future
    reflow could undo.

    The fix requires the caption's own wording, not just the label. Both halves
    are asserted here: without the caption words the checker still picks the
    wrong page (which is exactly the defect), and with them it picks the right
    one. If someone ever weakens the matcher back, the first assertion is what
    documents what they are giving up.
    """
    checker = load_checker()
    texts = [
        # p18: a body sentence that happens to begin a line with the label.
        "Benchmarking\n18\nrun controls are summarised in\n"
        "Table A.1. The remaining decoding settings follow the same policy.",
        # p46: the real caption.
        "Benchmarking\n46\nTable A.1. Primary run controls",
    ]
    labels = checker.printed_labels(texts)
    words = checker.caption_words("tables", "Table A.1. Primary run controls")

    assert checker.locate(texts, labels, "Table A.1.") == 18, (
        "label-only matching should still be fooled; if this changed, the "
        "regression test below is no longer testing anything"
    )
    assert checker.locate(texts, labels, "Table A.1.", words) == 46


def test_caption_overlap_tolerates_an_abbreviated_front_matter_title() -> None:
    """The list is an abbreviation of the caption, so exact prefix would misfire.

    "Table 2.1. Positioning of representative prior work" in the List of Tables
    against "Table 2.1. Scoped positioning of representative prior work..." on
    the page. Overlap accepts that; an exact match would report not_found on a
    correct document, which is a worse failure than the one being fixed.
    """
    checker = load_checker()
    texts = [
        "Benchmarking\n10\nTable 2.1. Scoped positioning of representative prior "
        "work. Entries are compiled from each cited paper's own reporting.",
    ]
    labels = checker.printed_labels(texts)
    words = checker.caption_words("tables", "Table 2.1. Positioning of representative prior work")
    assert checker.locate(texts, labels, "Table 2.1.", words) == 10
