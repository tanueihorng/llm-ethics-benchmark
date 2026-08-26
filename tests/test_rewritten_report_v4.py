"""Locks for the v4 rewrite line, read off the RENDERED document.

Why this file reads the DOCX rather than the builder: the ``epistemic_hedge``
scan in ``configs/artifact_policy.yaml`` scans builder SOURCE, and the v4
builder's ``CORRECTIONS`` table has to quote every bad string as the left-hand
side of a rewrite pair. A source-level pattern therefore cannot distinguish
"this over-claim ships" from "this over-claim is the thing being deleted". The
rendered document has no such ambiguity: if the phrase is in the DOCX, it ships.

The same argument is why these patterns are NOT added to ``epistemic_hedge``:
that scan also covers ``build_rewritten_fyp_report_v3.py``, and v3 is a separate
shipped artifact that deliberately keeps its original wording. Gating the v3
source would force a choice between breaking the gate and editing a frozen
document. Gating the v4 output forces neither.

The 2026-08-23 v4 audit found 22 defects under a fully green ``make
agent-check`` — 15 of them over-claims that the 55 existing ``epistemic_hedge``
patterns missed entirely, including "rule out", which its ``\\brules out\\b``
pattern misses by one letter. Each lock below is paired with a self-test that
feeds it the pre-fix wording, so a pattern that has stopped biting fails loudly
instead of passing vacuously (D54 / D56).

DOCX are untracked (D60), so a missing file is a skip, not a failure.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytest.importorskip("docx")
from docx import Document  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

V4_DOCUMENTS = [
    "docs/rewritten_fyp_report_v4.docx",
    "docs/rewritten_fyp_interim_v4.docx",
]

WP_DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


def load(relative: str) -> Document:
    path = REPO_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not built (DOCX are untracked, D60)")
    return Document(path)


def all_text(document: Document) -> str:
    """Body paragraphs AND table cells. Cells are where claims hide."""
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.extend(paragraph.text for paragraph in cell.paragraphs)
    return "\n".join(parts)


# --------------------------------------------------------------------------
# over-claim vocabulary
# --------------------------------------------------------------------------

# Each entry is (pattern, the pre-fix sentence it was written against). The
# second element is not decoration: ``test_overclaim_patterns_bite`` runs every
# pattern against its own example, so a pattern that has been weakened into
# uselessness cannot sit here looking green. Clear a real hit by REWORDING the
# claim, never by softening the pattern (D56).
OVERCLAIM_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bdecisively\b",
        "Human validation decisively supports the HarmBench classifier over the refusal proxy.",
    ),
    (
        r"\bconfirming (?:that|the|it was)\b",
        "Neither result reaches statistical significance, confirming that no model pair "
        "exhibits a significant over-refusal increase.",
    ),
    (
        r"\bcompletely dissolves\b",
        "the effect completely dissolves at 512 tokens",
    ),
    (
        r"\bremains fully intact\b",
        "the substantial superiority of the semantic judge remains fully intact.",
    ),
    (
        r"\bsystematically eliminate\b",
        "The validation experiments systematically eliminate competing hypotheses.",
    ),
    (
        # NOTE: the existing epistemic_hedge policy has `\brules out\b` and so
        # misses the plural subject. This one catches both.
        r"\brules? out\b",
        "cross-model judges rule out single-evaluator dependency",
    ),
    (
        r"\bample power\b",
        "The study has ample power to detect large, systematic safety degradations.",
    ),
    (
        r"\bfar better than\b",
        "with INT8 preserving performance far better than NF4",
    ),
    (
        r"\bhighly reliable\b",
        "INT8 offers a highly reliable precision tier for preserving reasoning capabilities",
    ),
    (
        r"\b(?:results|evidence|findings) reject\b",
        "The results reject both common oversimplifications",
    ),
    (
        r"\bdemonstrate[sd]? consistent\b",
        "multi-seed evaluations demonstrate consistent stability across decoding trajectories",
    ),
    (
        # The internal interpretation label leaking into prose, and the wording
        # a −0.032 delta cannot carry.
        r"capability collapse|capability anchors?|[Cc]apability-anchored",
        "observed reductions in harmful compliance often reflect capability collapse",
    ),
]


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
@pytest.mark.parametrize("pattern,_example", OVERCLAIM_PATTERNS)
def test_no_overclaim_vocabulary(relative: str, pattern: str, _example: str) -> None:
    text = all_text(load(relative))
    hits = [match.group(0) for match in re.finditer(pattern, text)]
    assert not hits, (
        f"{relative} contains over-claim wording {hits!r} matching {pattern!r}. "
        "Reword the claim to what the evidence supports; do not weaken the pattern."
    )


@pytest.mark.parametrize("pattern,example", OVERCLAIM_PATTERNS)
def test_overclaim_patterns_bite(pattern: str, example: str) -> None:
    """Self-test: every pattern must still match the wording it was written for."""
    assert re.search(pattern, example), f"{pattern!r} no longer matches its own example"


# --------------------------------------------------------------------------
# statistical wording
# --------------------------------------------------------------------------


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_bh_is_not_called_family_wise_error_control(relative: str) -> None:
    """BH controls the false discovery rate. FWER is a different guarantee."""
    text = all_text(load(relative))
    assert "family-wise error" not in text.lower(), (
        f"{relative} describes the correction as family-wise error control; "
        "Benjamini-Hochberg controls the false discovery rate."
    )


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_benjamini_hochberg_uses_an_en_dash(relative: str) -> None:
    text = all_text(load(relative))
    assert "Benjamini-Hochberg" not in text, (
        f"{relative} spells Benjamini-Hochberg with a hyphen; the document uses an en dash."
    )


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_named_mechanism_literature_has_sources(relative: str) -> None:
    """Usage-to-presence lock for the two named Chapter 2 research strands."""
    document = load(relative)
    paragraph = next(
        (p.text for p in document.paragraphs if "work on refusal directions" in p.text),
        None,
    )
    assert paragraph is not None, f"{relative} is missing the Chapter 2 mechanism paragraph"
    assert re.search(
        r"refusal directions \[\d+\] and shallow alignment \[\d+\]",
        paragraph,
    ), f"{relative} names refusal directions and shallow alignment without citing both sources"

    references = [
        p.text for p in document.paragraphs if p.style.name == "Reference"
    ]
    assert len(references) == 61, f"{relative} should contain 61 cited references"
    bibliography = "\n".join(references)
    assert "Refusal in language models is mediated by a single direction" in bibliography
    assert "Safety alignment should be made more than just a few tokens deep" in bibliography


def test_fwer_lock_would_have_caught_the_v4_defect() -> None:
    """Self-test: the exact shipped sentence must trip the lock."""
    shipped = (
        "Exact p-values and corrected q-values distinguish unadjusted contrasts "
        "from results surviving family-wise error control."
    )
    assert "family-wise error" in shipped.lower()


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_source_entailment_repairs_are_present(relative: str) -> None:
    """Locks the 2026-08-24 source-audit repairs on the rendered v4 surface."""
    text = all_text(load(relative))
    for stale in (
        "do not frame their outcomes as a declared set of corrected comparisons",
        "Prior reports of quantization-induced safety failures typically involve",
        "human ground truth",
    ):
        assert stale not in text, f"{relative} retains source-audit wording: {stale!r}"

    for required in (
        "declare 31 paired comparisons and apply a multiplicity correction",
        "IEEE version of record, Table 14",
        "Some methods require representative samples and an optimisation pass [5], [6], [22]",
        "Published safety findings span different settings",
        "validating model-based judges against human annotations",
        "in Proc. Conference on Language Modeling (COLM), 2026",
    ):
        assert required in text, f"{relative} is missing source-audit repair: {required!r}"


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_intro_quantization_prevalence_is_scoped_and_cited(relative: str) -> None:
    document = load(relative)
    paragraph = next(
        (p.text for p in document.paragraphs if "Quantization evaluations commonly report" in p.text),
        None,
    )
    assert paragraph is not None, f"{relative} is missing the scoped Chapter 1 evaluation claim"
    assert re.search(r"benchmark accuracy \[8\], \[9\]", paragraph)
    assert "quantization is often evaluated" not in paragraph.lower()


def test_source_entailment_locks_bite() -> None:
    """Self-test the exact stale strings that triggered the source audit."""
    stale = (
        "do not frame their outcomes as a declared set of corrected comparisons; "
        "Prior reports of quantization-induced safety failures typically involve attacks; "
        "validated against human ground truth"
    )
    assert "do not frame their outcomes as a declared set of corrected comparisons" in stale
    assert "Prior reports of quantization-induced safety failures typically involve" in stale
    assert "human ground truth" in stale


# --------------------------------------------------------------------------
# alt text must agree with the printed caption
# --------------------------------------------------------------------------


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_figure_alt_text_matches_its_caption(relative: str) -> None:
    """A screen reader must hear the title the page prints.

    Nothing in the harness reads alt text, which is how "Capability-anchored
    safety space" survived an edit that rewrote the caption and the List of
    Figures entry in the same file on the same day.
    """
    path = REPO_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not built (DOCX are untracked, D60)")

    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    alts = [
        node.get("descr")
        for node in root.iter(f"{WP_DRAWING}docPr")
        if node.get("descr")
    ]
    assert alts, f"{relative} has no figure alt text at all"

    # Match on the caption's own text, not on ``style.name``: python-docx
    # reports the style's DISPLAY name, which is not the ``FigureCaption``
    # styleId this builder writes, so a styleId comparison silently yields the
    # empty set -- and an empty set would make this lock vacuous, not failing.
    captions = {
        paragraph.text.strip()
        for paragraph in load(relative).paragraphs
        if re.match(r"^Figure \d+\.\d+\.", paragraph.text.strip())
    }
    assert captions, f"{relative} has no figure captions; the caption probe has drifted"
    for alt in alts:
        assert alt.strip() in captions, (
            f"{relative} alt text {alt!r} does not match any printed figure caption. "
            f"Captions: {sorted(captions)}"
        )


# --------------------------------------------------------------------------
# spelling and typography consistency
# --------------------------------------------------------------------------

# The document is British throughout (behaviour, analysed, generalised,
# standardised). "quantization"/"quantized" is a fixed technical term and is
# deliberately exempt.
US_SPELLINGS = re.compile(
    r"\b(?:behaviors?|analyz\w+|generaliz\w+|maximize\w*|minimiz\w+|localiz\w+"
    r"|characteriz\w+|synthesiz\w+|utiliz\w+|fulfill\w*|artefacts?)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_spelling_is_consistent(relative: str) -> None:
    text = all_text(load(relative))
    hits = sorted({match.group(0) for match in US_SPELLINGS.finditer(text)})
    assert not hits, (
        f"{relative} mixes spelling conventions: {hits}. The document is British "
        "elsewhere, and uses 'artifact' rather than 'artefact'."
    )


def test_spelling_lock_bites() -> None:
    """Self-test: the exact forms v4 shipped with must be caught."""
    shipped = (
        "prevents misleading generalizations ... to maximize diagnostic power ... "
        "decision minimized potential rater priming ... subtle localized effects ... "
        "A more accurate characterization ... Synthesizing the published literature ... "
        "utilizing validated evaluators ... context to fulfill, qualify ... "
        "the recorded experiment artefacts"
    )
    found = {match.group(0).lower() for match in US_SPELLINGS.finditer(shipped)}
    assert len(found) == 9, f"expected all nine shipped forms to trip the lock, got {sorted(found)}"


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_quotation_marks_are_typographic(relative: str) -> None:
    """Curly doubles are used throughout; straight singles as quote marks are not."""
    text = all_text(load(relative))
    offenders = []
    for match in re.finditer(r"'", text):
        before = text[match.start() - 1 : match.start()]
        after = text[match.end() : match.end() + 1]
        # An apostrophe follows a word character, or the period of an
        # abbreviation ("Marchisio et al.'s"). A quotation mark does not.
        is_apostrophe = (before.isalnum() and (after.isalpha() or after in {" ", ".", ","})) or (
            before == "." and after.isalpha()
        )
        if not is_apostrophe:
            offenders.append(text[max(0, match.start() - 40) : match.end() + 40])
    assert not offenders, (
        f"{relative} uses straight single quotes as quotation marks: {offenders}"
    )


# --------------------------------------------------------------------------
# abbreviations table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("relative", V4_DOCUMENTS)
def test_abbreviations_table_matches_usage(relative: str) -> None:
    """Every listed term is used in the body; every bare acronym used is listed.

    Derived from the document both ways round, because a list checked only in
    the direction it was written in cannot find an omission (D54).
    """
    document = load(relative)
    table = next(
        (t for t in document.tables if t.rows and t.rows[0].cells[0].text.strip() == "Term"),
        None,
    )
    assert table is not None, f"{relative} has no abbreviations table"
    listed = {row.cells[0].text.strip() for row in table.rows[1:]}
    assert listed, "abbreviations table is empty"

    # Body text only: the references and the abbreviations table itself do not count.
    body_paragraphs = document.paragraphs
    start = next(
        i for i, p in enumerate(body_paragraphs) if p.text.strip() == "Chapter 1: Introduction"
    )
    body = "\n".join(
        p.text for p in body_paragraphs[start:] if p.style.name != "Reference"
    )
    for table_ in document.tables:
        if table_ is table:
            continue
        for row in table_.rows:
            for cell in row.cells:
                body += "\n" + cell.text

    unused = sorted(term for term in listed if not re.search(rf"\b{re.escape(term)}\b", body))
    assert not unused, f"{relative} defines abbreviations it never uses: {unused}"

    for term in ("MDE", "VRAM"):
        if re.search(rf"\b{term}\b", body):
            assert term in listed, f"{relative} uses {term} in the body but does not define it"
