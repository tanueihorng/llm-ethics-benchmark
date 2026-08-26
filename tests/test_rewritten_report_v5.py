"""Locks for the v5 language-cleanup rewrite line, read off the RENDERED document.

The v5 pass (2026-08-25) replaces compressed statistical shorthand ("survives
correction", "survivors", "contrast", "corrected family", "multiplicity",
"unresolved", "construct", scorer-dependent, point estimate, inferential
threshold, heterogeneity, treatment effect, interpretation-label jargon) with
conventional academic English while preserving every number, citation,
benchmark name, research question, table value, section heading, and conclusion.

Why this file reads the DOCX rather than the builder: the ``epistemic_hedge``
scan covers the v5 builder SOURCE, but that source must quote every banned
string as the left-hand side of a correction pair, so a source-level scan cannot
distinguish "this shorthand ships" from "this shorthand is what was deleted"
(the same argument as ``tests/test_rewritten_report_v4.py``). The rendered
document has no such ambiguity.

Every forbidden pattern is paired with a pre-fix example sentence that
``test_forbidden_patterns_bite`` feeds back through the matcher, so a pattern
that has been weakened into uselessness fails loudly instead of passing
vacuously (D54 / D56). Clear a real hit by REWORDING the claim, never by
softening a pattern.

DOCX are untracked (D60), so a missing file is a skip, not a failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("docx")
from docx import Document  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

V4_DOCUMENTS = [
    "docs/rewritten_fyp_report_v4.docx",
    "docs/rewritten_fyp_interim_v4.docx",
]
V5_DOCUMENTS = [
    "docs/rewritten_fyp_report_v5.docx",
    "docs/rewritten_fyp_interim_v5.docx",
]

# The table cells whose PROSE was reworded (no value changed): Table 6.2's
# "Present study" row and the Table 5.2 generation-truncation row. Every other
# table cell in v5 must equal v4 byte for byte.
KNOWN_TABLE_CELL_CHANGES = {
    "ASR −0.040 to +0.040 (−4.0 to +4.0 points), none statistically significant "
    "after correction; MMLU to −0.090 (−9.0 points)",
    "60.3% truncation at 128; the effect disappears at 512",
}


def load(relative: str) -> Document:
    path = REPO_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not built (DOCX are untracked, D60)")
    return Document(path)


def body_and_cells(document: Document):
    """(paragraph texts, flattened table-cell texts)."""
    paragraphs = [p.text for p in document.paragraphs]
    cells = []
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                cells.extend(p.text for p in cell.paragraphs)
    return paragraphs, cells


def all_text(document: Document) -> str:
    paragraphs, cells = body_and_cells(document)
    return "\n".join(paragraphs + cells)


# ---------------------------------------------------------------------------
# banned shorthand vocabulary
# ---------------------------------------------------------------------------

# Each entry is (pattern, the pre-fix v4 wording it was written against).
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bsurviv(?:e|es|ed|ing|or|ors)\b",
        "Only the Llama decrease is individually significant, and no ASR result survives BH-FDR.",
    ),
    (
        r"\bcorrected family\b",
        "The corrected family contains three survivors.",
    ),
    (
        r"\bstatistical family\b",
        "The primary statistical family encompasses 20 NF4 comparisons.",
    ),
    (
        r"\binferential threshold\b",
        "Only one outcome crosses the inferential threshold.",
    ),
    (
        r"\bprecision contrast\b",
        "This study therefore measures all four outcomes under a single controlled precision contrast.",
    ),
    (
        r"\bone shared path\b",
        "harmful compliance does not follow one shared path across fp16, INT8, and NF4",
    ),
    (
        r"\btreatment effects?\b",
        "Paired deltas remain comparable as treatment effects.",
    ),
    (
        r"\btreatment-dependent\b",
        "so proxy error can itself become treatment-dependent",
    ),
    (
        r"\bscorer-dependent\b",
        "Phi-4-mini's over-refusal decrease is scorer-dependent.",
    ),
    (
        r"\bheterogeneity\b",
        "Overall, the evidence supports heterogeneity, not one shared response to bit-width.",
    ),
    (
        r"\bunresolved\b",
        "the remaining estimates are mixed or unresolved",
    ),
    (
        r"\bmethodological gap\b",
        "This creates a practical methodological gap for developers.",
    ),
]


@pytest.mark.parametrize("relative", V5_DOCUMENTS)
def test_no_banned_shorthand(relative: str) -> None:
    text = all_text(load(relative)).lower()
    remaining = [pattern for pattern, _ in FORBIDDEN_PATTERNS if re.search(pattern, text)]
    assert not remaining, f"{relative}: banned shorthand remains: {remaining}"


def test_forbidden_patterns_bite() -> None:
    """Each pattern must match its own pre-fix example (no vacuous locks)."""
    for pattern, example in FORBIDDEN_PATTERNS:
        assert re.search(pattern, example.lower()), f"pattern no longer bites: {pattern!r}"


@pytest.mark.parametrize("relative", V5_DOCUMENTS)
def test_rewritten_sentences_present(relative: str) -> None:
    """Spot locks: representative replacement sentences actually ship."""
    text = " ".join(all_text(load(relative)).split())
    expected = [
        # survive-correction -> significance after correction
        "None of the five NF4 harmful-compliance comparisons remains statistically "
        "significant after correction for multiple comparisons.",
        # survivor -> remained statistically significant
        "Exactly three primary comparisons remained statistically significant after "
        "Benjamini–Hochberg false-discovery-rate correction",
        # individually significant -> before/after correction distinction
        "This difference is statistically significant before multiple-comparison "
        "correction, but it is no longer statistically significant after the 20 primary "
        "comparisons are adjusted using the Benjamini–Hochberg procedure.",
        # construct validation spelled out
        "requires validation that the scoring method is measuring the intended behaviour",
        # power analysis / inconclusive, not equivalence
        "their non-significant results should be treated as inconclusive rather than as "
        "evidence that fp16 and NF4 are equivalent",
        # heterogeneity -> differs across models
        "Overall, the effect of lower precision differs across models and benchmarks "
        "rather than following one consistent pattern.",
        # interpretation label explained where used
        "the report labels this pattern broad degradation",
        # scorer-dependence stated plainly
        "That result depends on the scoring method, because the independent semantic judge "
        "does not reproduce it.",
        # construct validity defined at first use
        "Construct validity asks whether each benchmark and scorer measures the behaviour "
        "it is intended to measure.",
        # mixed or unresolved replaced
        "the remaining estimated differences vary in direction or do not provide clear "
        "statistical evidence of a change",
    ]
    missing = [sentence for sentence in expected if sentence not in text]
    assert not missing, f"{relative}: rewritten sentences missing: {missing}"


# ---------------------------------------------------------------------------
# structural preservation against v4
# ---------------------------------------------------------------------------


def _numeric_inventory(text: str) -> list[str]:
    # Drop tabbed lines first: cached TOC/list page numbers are layout metadata
    # verified by the pagination checker, not result values.
    text = "\n".join(line for line in text.splitlines() if "\t" not in line)
    text = re.sub(r"\[(?:\d+)(?:\]-\[\d+)?\]", "", text)
    return re.findall(r"(?<![A-Za-z])[-+−]?(?:\d{2,}(?:\.\d+)?|\d+\.\d+)%?", text)


def _citation_sequence(document: Document) -> list[int]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_ieee_citation_order_for_v5",
        REPO_ROOT / "tests" / "test_ieee_citation_order.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CITATION = module.CITATION
    expand_ranges = module.expand_ranges
    iter_block_paragraphs = module.iter_block_paragraphs

    sequence = []
    for paragraph in iter_block_paragraphs(document):
        for match in CITATION.finditer(expand_ranges(paragraph.text)):
            sequence.append(int(match.group(1)))
    return sequence


def _research_questions(text: str) -> list[str]:
    questions = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^RQ[1-5]\.", stripped):
            question = re.sub(r"\s+", " ", stripped)
            if question not in questions:
                questions.append(question)
    return questions


@pytest.mark.parametrize("v4_relative", V4_DOCUMENTS)
def test_numbers_citations_questions_headings_preserved(v4_relative: str) -> None:
    v5_relative = v4_relative.replace("_v4.docx", "_v5.docx")
    v4_doc = load(v4_relative)
    v5_doc = load(v5_relative)

    v4_text = all_text(v4_doc)
    v5_text = all_text(v5_doc)

    # 1. numerical multiset: nothing lost, nothing invented
    v4_numbers, v5_numbers = _numeric_inventory(v4_text), _numeric_inventory(v5_text)
    missing = sorted(set(v4_numbers) - set(v5_numbers))
    dropped_counts = {
        token: (v4_numbers.count(token), v5_numbers.count(token))
        for token in set(v4_numbers)
        if v5_numbers.count(token) < v4_numbers.count(token)
    }
    invented = sorted(set(v5_numbers) - set(v4_numbers))
    assert not dropped_counts, f"{v5_relative}: numeric tokens lost: {dropped_counts}"
    assert not invented, f"{v5_relative}: numeric tokens invented: {invented}"
    assert not missing, f"{v5_relative}: numeric tokens missing: {missing}"

    # 2. citation marker sequence identical
    assert _citation_sequence(v4_doc) == _citation_sequence(v5_doc), (
        f"{v5_relative}: citation sequence changed"
    )

    # 3. research questions verbatim
    assert _research_questions(v4_text) == _research_questions(v5_text), (
        f"{v5_relative}: research-question wording changed"
    )

    # 4. every heading preserved (section numbering intact)
    v4_headings = [p.text for p in v4_doc.paragraphs if p.style.name.startswith("Heading")]
    v5_headings = [p.text for p in v5_doc.paragraphs if p.style.name.startswith("Heading")]
    assert v4_headings == v5_headings, f"{v5_relative}: heading structure changed"

    # 5. paragraph count unchanged
    assert len(v4_doc.paragraphs) == len(v5_doc.paragraphs), (
        f"{v5_relative}: paragraph count changed"
    )


@pytest.mark.parametrize("v4_relative", V4_DOCUMENTS)
def test_table_cells_unchanged_except_declared_prose_cell(v4_relative: str) -> None:
    v5_relative = v4_relative.replace("_v4.docx", "_v5.docx")
    _, v4_cells = body_and_cells(load(v4_relative))
    _, v5_cells = body_and_cells(load(v5_relative))

    assert len(v4_cells) == len(v5_cells), f"{v5_relative}: table cell count changed"
    allowed = {" ".join(cell.split()) for cell in KNOWN_TABLE_CELL_CHANGES}
    differences = [
        (old[:60], new[:60])
        for old, new in zip(v4_cells, v5_cells)
        if old != new and " ".join(new.split()) not in allowed
    ]
    assert not differences, f"{v5_relative}: unexpected table-cell edits: {differences}"
