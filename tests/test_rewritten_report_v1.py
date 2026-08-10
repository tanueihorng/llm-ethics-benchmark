"""Regression checks for the standalone corpus-informed report."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def report_parts() -> tuple[ElementTree.Element, ElementTree.Element]:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} is not present in this checkout")
    with zipfile.ZipFile(REPORT) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        settings = ElementTree.fromstring(archive.read("word/settings.xml"))
    return document, settings


def paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))


def table_row_texts(document: ElementTree.Element) -> list[str]:
    rows = []
    for row in document.findall(".//w:tr", NS):
        cells = []
        for cell in row.findall("w:tc", NS):
            cell_text = "".join(
                paragraph_text(paragraph)
                for paragraph in cell.findall(".//w:p", NS)
            )
            cells.append(cell_text)
        rows.append(" | ".join(cells))
    return rows


def test_negative_numbers_use_one_minus_sign() -> None:
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    assert not re.search(r"(?<![A-Za-z0-9])-\d", visible_text), (
        "negative numeric values must use U+2212 rather than an ASCII hyphen"
    )
    assert "−0.090" in visible_text


def test_toc_is_a_refreshable_field_with_visible_cached_entries() -> None:
    document, settings = report_parts()
    instructions = " ".join(
        node.text or "" for node in document.findall(".//w:instrText", NS)
    )
    assert 'TOC \\o "1-2" \\h \\z \\u' in instructions
    update = settings.find("w:updateFields", NS)
    assert update is not None and update.get(f"{{{W_NS}}}val") == "true"
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    assert "Chapter 1: Introduction" in visible_text
    assert "References" in visible_text


def test_references_are_unnumbered_back_matter() -> None:
    document, _ = report_parts()
    heading_texts = []
    for paragraph in document.findall(".//w:p", NS):
        style = paragraph.find("w:pPr/w:pStyle", NS)
        if style is not None and style.get(f"{{{W_NS}}}val") == "Heading1":
            heading_texts.append(paragraph_text(paragraph))
    assert "References" in heading_texts
    assert "Chapter 8: References" not in heading_texts


def test_prior_work_table_carries_scope_and_no_novelty_disclaimer() -> None:
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    assert "Entries are compiled from each cited paper's abstract and stated scope." in visible_text
    assert "Not reported” is not evidence of absence." in visible_text
    assert "The table is not a novelty or priority claim." in visible_text
    assert "Blank or partial cells reveal the gap" not in visible_text


def test_entailment_audit_corrections_are_present() -> None:
    """Pin all eight source-fitness repairs confirmed on 2026-07-28."""
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    table_rows = table_row_texts(document)

    assert (
        "report perplexity, accuracy, latency, or memory as primary outcomes [5]-[7]"
        in visible_text
    )
    assert "primary outcomes [5]-[8]" not in visible_text

    assert (
        "SORRY-Bench grounds refusal evaluation in a human-in-the-loop taxonomy "
        "of unsafe topics"
        in visible_text
    )
    assert (
        "Do-Not-Answer's six action categories [45] and XSTest's "
        "full-compliance, partial-refusal and full-refusal labelling [18]"
        in visible_text
    )
    assert "SORRY-Bench's human-grounded refusal taxonomy" not in visible_text

    assert "XSTest / OR-Bench [18], [19]" in visible_text
    assert "Contrast set only" in visible_text
    assert "Over-refusal focus; unsafe items are a calibration contrast" in visible_text
    assert "Benign refusal only" not in visible_text

    assert (
        "HarmBench [24] | Not reported | Yes | Not reported | Not reported | "
        "Classifier | ASR conditional on the selected threat model"
        in table_rows
    )
    assert (
        "Kharinaev et al. [10] | Yes | Yes | Not reported | Partial | "
        "Human / mixed | Different model/method mix"
        in table_rows
    )
    assert (
        "HarmLevelBench [35] | Yes | Graded | Not reported | Not reported | "
        "Human / mixed | Attack-dependent construct"
        in table_rows
    )

    assert "Threat model must be stated" not in visible_text


def test_t47_substantive_gap_repairs_are_present() -> None:
    """Pin the three T47 substantive-gap repairs applied on 2026-07-29.

    Gap 2 (the treatment library was never versioned, and the report referred
    three times to a "recorded software stack" that the document did not
    contain), gap 8 ("reference budget" used in the Abstract before its
    definition in Chapter 5) and gap 9 (the Llama ARC FDR survivor carried no
    deployment-unit translation while the other two survivors did).
    """
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    table_rows = table_row_texts(document)

    # Gap 8: the Abstract defines the budget where it first uses it.
    assert "HarmBench's reference generation budget of 512 new tokens" in visible_text
    assert "At the 512-token reference budget" not in visible_text

    # Gap 9: all three FDR survivors carry a deployment-unit translation.
    assert "roughly nine additional questions per hundred" in visible_text
    assert "roughly three additional ARC questions per hundred" in visible_text
    assert "twelve fewer regex-flagged benign prompts" in visible_text

    # Gap 2: the treatment library is named, overridden defaults are listed,
    # and the declared floors are labelled as floors rather than as captured
    # runtime versions.
    assert any(
        row.startswith("Quantization | bitsandbytes NF4 with double quantization")
        for row in table_rows
    )
    assert any(
        row.startswith("Library defaults overridden | Eager attention")
        for row in table_rows
    )
    assert any(
        "bitsandbytes 0.43.0 or later" in row
        and "not captured in the immutable outputs" in row
        for row in table_rows
    )

    # Gap 2: no surviving reference to a software record the report lacks.
    assert "recorded software stack" not in visible_text


def test_t47_gap6_prior_work_figures_are_present() -> None:
    """Pin every published figure imported into Table 6.2 on 2026-08-02.

    T47 gap 6: prior work was discussed in §6.2 without a single published
    number beside this study's. Each value below was transcribed from the
    source's own results table and then verified by two independent agents
    under separate entailment and comparability lenses, never batched (D57).
    Provenance and the per-figure verdicts are in
    docs/audits/2026-08-02_prior_work_numbers/PROVENANCE.md.
    """
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    table_rows = table_row_texts(document)
    row_text = "\n".join(table_rows)

    # Decoding Compressed Trust [37]: MMLU at three model sizes.
    assert "MMLU −1.5 points at 13B (53.6 to 52.1); −1.6 at 7B and −0.9 at 70B" in row_text
    # The source says only "16-bit dense"; it never states the float format.
    assert "vs 16-bit dense" in row_text
    assert "GPTQ 4-bit, 128-sample calibration" in row_text

    # Kharinaev et al. [10]: the paper's own "Difference from FP 16" columns,
    # with the inverted polarity spelled out.
    assert "0.571, 0.536 and 4.500 points on XSafety" in row_text
    assert "3.561 and 8.622 on OpenSafetyMini" in row_text
    assert "while QUIK rises 0.187" in row_text
    assert "Positive means less safe" in row_text

    # HarmLevelBench [35]: direct-query human ASR, and the scorer spread.
    assert "33.9 to 44.6, a gain of 10.7 points" in row_text
    assert "+16.1 by string match and +1.7 by GPT judge" in row_text
    assert "10.7 points is six queries" in row_text

    # Q-resafe [36]: the paper's own headline column, the true range, and the
    # un-attacked base rate that makes the reference level interpretable.
    assert "29.8 to 37.1 points at τ=0.95" in row_text
    assert "4.5 to 17.3 points across six" in row_text
    assert "scores 0.3 un-attacked" in row_text

    # Q-Misalign [39]: the control column, not the attacker-prepared headline.
    assert "0.07 to 0.07, 0.05 to 0.05, 0.00 to 0.00" in row_text
    assert "prepared" in row_text and "to misbehave once quantized" in row_text

    # The scale conversion must be stated, or the table reads backwards.
    assert "a delta of −0.090 is a drop of 9.0 points" in visible_text
    assert "−0.040 to +0.040 (−4.0 to +4.0 points)" in row_text

    # Evidence grade (D56): the non-archival source is labelled as one.
    assert "non-archival workshop paper" in visible_text


def test_t47_gap6_refuted_figures_stay_out() -> None:
    """Must-fire guard: the values verification rejected must not reappear.

    Each string below is a real number from one of the five papers that a
    rebuild or a future edit could plausibly reintroduce, and each would
    misattribute an attack, a fine-tuning step or a prepared backdoor to
    ordinary quantization.
    """
    document, _ = report_parts()
    visible_text = "\n".join(
        paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
    )
    row_text = "\n".join(table_row_texts(document))
    haystack = visible_text + "\n" + row_text

    # Q-Misalign's headline belongs to an attacker-prepared checkpoint. It may
    # be named only as the quarantined counter-example, never as a bare delta.
    assert "0.00 to 0.97" not in haystack

    # Q-resafe's 42.4 is the decoding-attack maximum over sampling settings,
    # and 0.3 -> 42.4 pairs an un-attacked baseline with an attacked arm.
    assert "42.4" not in haystack

    # Q-resafe's Risk-I rows quantize with a fine-tuning step, so their ASRs
    # confound fine-tuning with precision reduction.
    assert "16.9" not in haystack

    # The drafted Q-resafe range floor was wrong: +5.7 is a Gemma 8-bit gap,
    # not the Llama minimum, which is +4.5.
    assert "5.7 to 17.3" not in haystack

    # "no method dominates" was drafted as a quotation; that string appears in
    # neither arXiv version of Kharinaev et al.
    assert "no method dominates" not in haystack
