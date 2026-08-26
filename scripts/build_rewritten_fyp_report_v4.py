#!/usr/bin/env python3
"""Build the v4 rewrite with the user's HW0288 funnel introduction.

The v4 line is deliberately non-destructive: it builds the verified v3 prose
line first, then replaces only the Chapter 1 opening and the two short problem
statement paragraphs that would otherwise repeat that opening.  The v3 report
and interim artefacts remain unchanged. Like v3, the bibliography ultimately
comes from ``scripts/build_rewritten_fyp_report_v1.py``.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_rewritten_fyp_report_v3.py"
OUT = ROOT / "docs" / "rewritten_fyp_report_v4.docx"


# The assignment introduction, adapted only where necessary to use the v3
# report's existing bibliography (rather than introducing out-of-order markers
# for refs 32/33/41).  The funnel is preserved: deployment context ->
# interpretive gap -> study response and boundaries.
INTRODUCTION = [
    (
        "This project studies five matched model pairs drawn from four compact open-weight model families: "
        "Qwen3, the Llama 3 family, Phi-4-mini, and Mistral-7B [1]–[3], [51]. Instruction-tuned models at "
        "this scale are increasingly practical to run locally, because they fit on consumer GPUs and because "
        "low-bit compression has been developed explicitly for on-device and mobile inference [5]–[7]. "
        "Local inference can keep prompts and generated responses on the user's device when no remote "
        "service is involved, but it does not by itself guarantee safe or reliable behaviour. Quantization "
        "reduces memory demand by representing model weights at lower precision, and in this study NF4 and "
        "LLM.int8 are the two lower-precision conditions. These transformations can affect more than "
        "storage: a model may preserve some task accuracy while changing how it answers harmful or benign "
        "requests. Safety and capability therefore need to be measured together rather than inferred from "
        "model size or memory savings alone."
    ),
    (
        "Quantization evaluations commonly report perplexity or benchmark accuracy [8], [9], but those measures "
        "alone do not reveal whether an apparent reduction in harmful compliance is a genuine alignment change or "
        "a loss of capability. Safety evaluation has a second complication: a model may comply less with "
        "harmful prompts while also refusing benign prompts more often, or it may appear safer because its "
        "responses have become less capable. Prior work also shows that behavioural and trustworthiness effects "
        "can depend on both the model and the quantization method [8]–[10]. This project therefore treats harmful compliance and "
        "benign over-refusal as distinct outcomes and evaluates them alongside general capability. HarmBench "
        "and XSTest provide complementary ways to measure these behaviours [18], [24]. The interpretive "
        "problem is not only to detect a change, but to determine what the change means. A controlled "
        "comparison must hold the model revision, prompts, decoding settings, and sample pairing constant. "
        "Without such controls, aggregate safety scores may be mistaken for improved alignment when they "
        "instead reflect degraded task competence or scorer behaviour. This creates a practical methodological "
        "gap for developers who need evidence that is reproducible and interpretable before deployment."
    ),
    (
        "To address this gap, this study evaluates matched fp16–NF4 versions of five revision-pinned model "
        "pairs, with a separate INT8 comparison at the same baseline revisions. The primary comparison "
        "applies NF4 at load time to the same weights as the fp16 baseline; INT8 supplies an additional "
        "precision point rather than being treated as the same quantization method. Outputs are evaluated "
        "on four tasks: HarmBench harmful compliance, XSTest benign over-refusal, a six-subject MMLU subset, "
        "and ARC-Challenge capability. The analysis uses paired bootstrap confidence intervals, exact "
        "McNemar tests, Benjamini–Hochberg false-discovery-rate control, and power-oriented interpretation, "
        "and scorer validity is examined with the official HarmBench classifier, an independent API judge, "
        "Llama Guard, and human-labelled samples. Section 1.4 states the boundaries of that design. The aim "
        "is to determine whether quantization-related safety changes remain when capability and scorer "
        "validity are considered together, providing a transparent basis for interpreting the deployment "
        "risks of compact open-weight models."
    ),
]

PROBLEM_STATEMENT = (
    "The problem addressed by this study is interpretive rather than merely operational: a change in harmful "
    "compliance cannot be labelled an alignment shift until it is read alongside benign over-refusal, "
    "capability benchmarks, and evidence that the scorer measures what it is taken to measure. Each of "
    "those three can move a safety number on its own, and none of them is visible in an attack-success "
    "rate reported by itself."
)

OBJECTIVE_STATEMENT = (
    "This study therefore measures all four outcomes on the same prompts under a single controlled precision "
    "contrast, and treats the scorer as an object of study rather than as a fixed instrument. Chapter 4 "
    "reports the outcome measurements and Chapter 5 reports the scorer evidence."
)

# --- v4 text corrections, applied after the v3 build --------------------------
#
# These are exact-substring rewrites over the built document's body paragraphs
# and table cells. They live HERE rather than in the v1/v3 builders on purpose:
# v1 and v3 are separate shipped artifacts and must not change because v4 was
# corrected. That is the same non-destructive contract as the Chapter 1 swap.
#
# Every entry must match exactly once. A correction that stops matching (because
# upstream prose moved) fails the build instead of silently doing nothing --
# a no-op rewrite is the failure mode that lets a "fixed" defect ship.
#
# Sources: the 2026-08-23 full v4 audit. Grouped by what the change is for.
CORRECTIONS: list[tuple[str, str]] = [
    # -- statistical accuracy ---------------------------------------------
    # BH controls the false discovery rate, not the family-wise error rate.
    (
        "Exact p-values and corrected q-values distinguish unadjusted contrasts from "
        "results surviving family-wise error control.",
        "Exact p-values and corrected q-values distinguish unadjusted contrasts from "
        "results surviving false-discovery-rate control.",
    ),
    # The tables carry one combined fp16->NF4 column, not two.
    (
        "Baseline and NF4 columns display observed rates",
        "The fp16→NF4 columns display observed rates",
    ),
    ("survived Benjamini-Hochberg false-discovery-rate", "survived Benjamini–Hochberg false-discovery-rate"),
    # -- claims that outran the evidence ----------------------------------
    # A non-significant result cannot confirm an absence (the report says so
    # itself in 5.7), and Phi's two mappings say nothing about the other four
    # pairs. Both faults were in one sentence.
    (
        "Neither result reaches statistical significance, confirming that no model pair "
        "exhibits a significant over-refusal increase under semantic evaluation.",
        "Neither result reaches statistical significance. Read together with the other four "
        "pairs, which are also non-significant under the judge, this leaves no over-refusal "
        "increase detectable under semantic evaluation at the sample sizes used; it does not "
        "establish that no such increase exists (Section 5.7).",
    ),
    (
        "The results reject both common oversimplifications: that quantization inherently "
        "improves safety alignment, and that it systematically causes safety degradation. In "
        "reality, observed reductions in harmful compliance often reflect capability collapse, "
        "while apparent increases in risk are largely statistical or measurement artifacts.",
        "The results support neither common oversimplification: neither that quantization "
        "inherently improves safety alignment, nor that it systematically degrades it. Where "
        "harmful compliance fell in this study it fell alongside a measured capability loss, "
        "and the increases that appeared were small enough to be consistent with sampling "
        "variation or scorer error.",
    ),
    (
        "the effect completely dissolves at 512 tokens (classifier delta +0.005 with p = 1.000; "
        "API-judge delta +0.010 with p = 0.688), confirming it was a truncation artifact.",
        "the effect disappears at 512 tokens (classifier delta +0.005 with p = 1.000; API-judge "
        "delta +0.010 with p = 0.688), which is what a truncation artifact would predict.",
    ),
    (
        "Human validation decisively supports the HarmBench classifier over the refusal proxy "
        "and the semantic judge over the keyword regex. Furthermore, the 128-token Qwen3-1.7B "
        "truncation anomaly is resolved at 512 tokens, confirming the robustness of the primary "
        "findings.",
        "Against a single annotator on each 200-item sample, human validation favours the "
        "HarmBench classifier over the refusal proxy, and the semantic judge over the keyword "
        "regex, by a wide margin in both cases. The 128-token Qwen3-1.7B anomaly does not "
        "reappear at 512 tokens. No one of these checks is decisive on its own, but none of "
        "them overturns the primary findings.",
    ),
    (
        "The study has ample power to detect large, systematic safety degradations, though "
        "subtle localized effects remain below the resolution threshold.",
        "The study is therefore powered for large, systematic safety degradations; smaller "
        "localised effects remain below its resolution threshold.",
    ),
    (
        "The reported null finding is substantiated across multiple independent dimensions: zero "
        "HarmBench contrasts survive multiplicity correction, three distinct semantic judges find "
        "no significant ASR increases, the 128-token Qwen3-1.7B artifact dissolves at the "
        "512-token reference budget, and multi-seed evaluations demonstrate consistent stability "
        "across decoding trajectories.",
        "The reported null finding is supported along four partly independent dimensions: no "
        "HarmBench contrast survives multiplicity correction, the primary classifier and two "
        "further judges find no significant ASR increases, the 128-token Qwen3-1.7B artifact "
        "does not survive the move to the 512-token reference budget, and no decoding seed in "
        "the multi-seed arm shows a consistent rise in harmful compliance.",
    ),
    (
        "The validation experiments systematically eliminate competing hypotheses: human "
        "annotations expose the over-flagging bias of refusal proxies, cross-model judges rule "
        "out single-evaluator dependency, the 512-token protocol eliminates truncation artifacts, "
        "and multi-seed testing bounds sampling noise.",
        "Each validation experiment addresses one competing explanation: human annotations expose "
        "the over-flagging bias of refusal proxies, cross-model judges test whether the result "
        "depends on a single evaluator, the 512-token protocol removes the truncation that "
        "affected the 128-token runs, and multi-seed testing bounds sampling noise.",
    ),
    (
        "the substantial superiority of the semantic judge over the keyword regex remains fully intact.",
        "the kappa gap between the semantic judge and the keyword regex is unchanged on that subset.",
    ),
    (
        "RQ4 reveals that capability degradation varies substantially by model architecture and "
        "precision, with INT8 preserving performance far better than NF4.",
        "RQ4 reveals that capability degradation varies substantially by model architecture and "
        "precision; INT8 stays closer to fp16 than NF4 on the largest losses, though not on "
        "every contrast.",
    ),
    (
        "INT8 offers a highly reliable precision tier for preserving reasoning capabilities",
        "INT8 was the more conservative precision tier for reasoning capability across these five pairs",
    ),
    (
        "The primary contribution of this work resides in the rigorous empirical evidence and "
        "methodological validation of safety-capability trade-offs",
        "The primary contribution of this work is the empirical evidence and the accompanying "
        "scorer validation for safety-capability trade-offs",
    ),
    # The abstract asserted a memory/compute saving the study explicitly did
    # not measure (6.3 says so outright).
    (
        "Post-training quantization significantly reduces the computational and memory "
        "requirements of instruction-tuned language models",
        "Post-training quantization lowers the precision of a model's weights to reduce "
        "its memory footprint",
    ),
    # -- the INT8 shape claim, against precision_sweep.json's own labels ---
    # Only qwen_2b and llama_3_2_3b are non-monotonic there; qwen_4b is graded.
    (
        "Figure 4.2 shows that harmful compliance is generally non-monotonic across fp16, INT8, "
        "and NF4.",
        "Figure 4.2 shows that harmful compliance does not follow one shared path across fp16, "
        "INT8, and NF4: two pairs move non-monotonically, and the rest disagree with each other "
        "in direction.",
    ),
    (
        "These non-monotonic paths argue against one alignment effect caused by lower precision.",
        "Qwen3-4B is the only pair whose harmful compliance rises at every step, and that rise is "
        "not significant; the other four paths differ from it in shape or in direction. Together "
        "they argue against one alignment effect caused by lower precision.",
    ),
    # -- grammar ----------------------------------------------------------
    ("it does not determine the study main conclusion", "it does not determine the study's main conclusion"),
    (
        "is stronger evidence practice than retroactively redefining the protocol.",
        "is better research practice than retroactively redefining the protocol.",
    ),
    # NOTE: corrections run BEFORE citation renumbering, so they must not
    # contain bracket markers -- the numbers here would still be v3's. This
    # fragment is repaired in two bracket-free halves for that reason.
    (
        "a simplification: Do-Not-Answer's six action categories",
        "a simplification: Do-Not-Answer uses six action categories",
    ),
    (
        " and XSTest's full-compliance, partial-refusal and full-refusal labelling",
        ", and XSTest labels responses as full compliance, partial refusal, or full refusal",
    ),
    (
        "analysis programs, result derived scoring files, and automated verification",
        "analysis programs, the derived scoring files, and automated verification",
    ),
    # -- one exclusion list, stated once, complete -------------------------
    (
        "monetary cost, or general adversarial robustness; its safety conclusions",
        "monetary cost, general adversarial robustness, or multimodal behaviour; its safety conclusions",
    ),
    # -- spelling: the document is British everywhere else -----------------
    ("the recorded experiment artefacts", "the recorded experiment artifacts"),
    ("prevents misleading generalizations", "prevents misleading generalisations"),
    ("A more accurate characterization of low-bit", "A more accurate characterisation of low-bit"),
    ("Synthesizing the published literature", "Synthesising the published literature"),
    ("oversamples scorer disagreements to maximize", "oversamples scorer disagreements to maximise"),
    ("this pre-annotation decision minimized potential", "this pre-annotation decision minimised potential"),
    ("sufficient context to fulfill, qualify", "sufficient context to fulfil, qualify"),
    ("utilizing validated evaluators", "using validated evaluators"),
    # -- typography: curly doubles everywhere else, straight singles here ---
    ("generalise from 'quantization' as", "generalise from “quantization” as"),
    ("motivated by 'empty jailbreaks':", "motivated by “empty jailbreaks”:"),
    ("broader 'unsafe response' construct", "broader “unsafe response” construct"),
    # -- source-entailment wording ----------------------------------------
    (
        "validating model-based judges against human ground truth",
        "validating model-based judges against human annotations",
    ),
    (
        "substantially higher alignment with human ground truth than the keyword refusal proxy",
        "substantially higher agreement with the human annotator than the keyword refusal proxy",
    ),
    (
        "model-based semantic evaluators agree far more closely with human ground truth",
        "model-based semantic evaluators agree much more closely with the human annotations",
    ),
    (
        "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” "
        "arXiv:2503.05061, 2025 (preprint).",
        "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” "
        "in Proc. Conference on Language Modeling (COLM), 2026. arXiv:2503.05061.",
    ),
]

# These source-fitness repairs use the FINAL v4 reference numbers and therefore
# run only after first-use renumbering. Each replacement remains fail-closed.
POST_RENUMBER_CORRECTIONS: list[tuple[str, str]] = [
    (
        "The three quantization evaluations reviewed here emphasise average benchmark retention "
        "and do not frame their outcomes as a declared set of corrected comparisons [8]–[10].",
        "Kharinaev et al. declare 31 paired comparisons and apply a multiplicity correction [10], "
        "whereas the other two evaluations emphasise aggregate retention across many tasks [8], [9].",
    ),
    (
        "Some methods require representative samples and an optimisation pass, while BitsAndBytes "
        "load-time quantization can be applied without a task-specific calibration set.",
        "Some methods require representative samples and an optimisation pass [5], [6], [22], while "
        "BitsAndBytes load-time quantization can be applied without a task-specific calibration set "
        "[24], [25].",
    ),
    (
        "Prior reports of quantization-induced safety failures typically involve active adversarial "
        "jailbreaks, destructive calibration regimes, or backdoored weights, rather than standard "
        "post-training NF4 loading of unmodified instruction-tuned checkpoints.",
        "Published safety findings span different settings: Kharinaev et al. and HarmLevelBench "
        "evaluate ordinary quantization [10], [29], Q-resafe uses an active decoding attack [30], "
        "and Q-Misalign studies a deliberately prepared checkpoint [33]. These designs answer "
        "different questions from standard load-time NF4 on unmodified checkpoints.",
    ),
    (
        "Kharinaev et al. report a safety ratio and calculate the difference as FP16 minus quantized, "
        "so a positive difference means the quantized model was judged less safe, the opposite "
        "orientation to the attack success rates in the neighbouring rows [10].",
        "In the IEEE version of record, Table 14 reports safety ratios and defines the difference as "
        "FP16 minus quantized, so a positive difference means that the quantized model was judged "
        "less safe, the opposite orientation to the attack success rates in the neighbouring rows [10].",
    ),
    (
        "Llama-3.1-8B-Instruct; AWQ, QUIK and QUIP# at int4, vs bfloat16",
        "Llama-3.1-8B-Instruct; AWQ, QUIK and QUIP# at int4, vs bfloat16; "
        "IEEE version of record, Table 14",
    ),
]

# Figure 1.1's alt text was never updated when its caption and List of Figures
# entry were rewritten, so a screen reader announced a different title from the
# printed one -- and it still carried the retired "capability-anchored" phrase.
# v1's own caption DOES say "Capability-anchored", so this is fixed here, for
# the v4 line only, rather than in the shared v1 builder.
ALT_TEXT_FIXES = {
    "Figure 1.1. Capability-anchored safety space using MMLU and HarmBench deltas.":
        "Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance.",
}

# PTQ was defined and never used; MDE and VRAM are used bare in the body.
ABBREVIATION_DROPS = ("PTQ",)
ABBREVIATION_ADDS = (
    ("MDE", "Minimum detectable effect"),
    ("VRAM", "Video memory on the GPU"),
)

# The longer funnel opening moves the latter Chapter 1 entries one numbered
# page later, while the Chapter 2 break and all later chapters remain stable.
#
# The four entries at the bottom are NOT caused by the v4 rewrite -- they were
# already wrong in v3, inherited from lists that were paginated against v1's
# shorter body. Values below are what a LibreOffice render of v4 actually
# shows. Word is the submission renderer and can differ by a page; the three
# contents entries are a TOC field that Word regenerates on open, so only the
# List of Tables entry (a plain paragraph nothing corrects) is load-bearing.
V4_PAGE_NUMBERS = {
    "1.2 Problem Statement": 2,
    "4.6 INT8 Precision Point": 27,
    "6.5 Threats to Validity": 40,
    "Table 4.2. NF4 capability results": 23,
    "1.3 Research Questions": 3,
    "1.5 Contributions": 4,
    "1.6 Report Structure": 4,
    "Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance": 3,
}



# --- IEEE citation renumbering -------------------------------------------------
#
# Rewriting Chapter 1 changed the ORDER in which references are first cited. The
# funnel introduction cites [5]-[7], [8]-[10], [18] and [24] before the markers
# that v3 introduced earlier, so the inherited numbering no longer ascends: the
# first-use sequence runs 1, 2, 3, 5, 6, 7, 8, 9, 10, 18, 24, 20, ... IEEE
# requires reference numbers to follow strict first-use order, so the numbering
# is recomputed here rather than inherited from v3.
#
# The final v4 line retains the references that remain cited after the Chapter 1
# replacement and drops any inherited entries that no longer have a body cite.
# The current build retains 61 references and drops old v3 entries [4], [14], [17].
# Old [15] (Arditi) and [16] (Qi) remain necessary because Chapter 2 explicitly
# discusses refusal directions and shallow alignment.
# The set is computed below; these values document the present output rather than
# assuming that every inherited reference survives a future prose change.
#
# This runs on the rendered document, after the Chapter 1 replacement, because
# that is the only point where the final citation order is known. Every citing
# paragraph in this document carries exactly one run (verified 2026-08-23), so
# run-level substitution preserves formatting.

CITATION = re.compile(r"\[(\d+)\]")
CITATION_RANGE = re.compile(r"\[(\d+)\]\s*[\u2013\u2014-]\s*\[(\d+)\]")
REFERENCE_ENTRY = re.compile(r"^\[(\d+)\]\s+\S")


def _iter_block_paragraphs(doc):
    """Every paragraph in true body order, including those inside tables."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        yield paragraph


def _split_document(doc):
    """One traversal, body order: (reference entries, body paragraphs).

    Classifying in a single pass is deliberate. Matching paragraphs ACROSS two
    traversals is unsafe: python-docx builds a fresh wrapper each time, and
    ``id(paragraph._element)`` can be recycled once a temporary proxy is
    collected, so an id-keyed skip set silently excludes the wrong paragraphs.
    That bug renumbered five references into orphans before it was caught.
    """
    entries, body, seen_heading = [], [], False
    for paragraph in _iter_block_paragraphs(doc):
        text = paragraph.text.strip()
        if text == "References":
            seen_heading = True
            body.append(paragraph)
            continue
        match = REFERENCE_ENTRY.match(text)
        if seen_heading and match:
            entries.append((int(match.group(1)), paragraph))
        else:
            body.append(paragraph)
    return entries, body


def _expand_ranges(text: str) -> str:
    def expand(match):
        low, high = int(match.group(1)), int(match.group(2))
        if high < low:
            return match.group(0)
        return "".join(f"[{n}]" for n in range(low, high + 1))

    return CITATION_RANGE.sub(expand, text)


def _format_markers(numbers) -> str:
    """[4][5][6] -> '[4]-[6]' (en dash) for three or more consecutive, else '[4], [7]'."""
    numbers = sorted(numbers)
    if len(numbers) >= 3 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"[{numbers[0]}]\u2013[{numbers[-1]}]"
    return ", ".join(f"[{n}]" for n in numbers)


def _renumber_citations(doc) -> dict:
    reference_paragraphs, body = _split_document(doc)
    listed = {number for number, _ in reference_paragraphs}
    if not listed:
        raise AssertionError("No bibliography entries found; cannot renumber citations.")

    order = []
    for paragraph in body:
        for match in CITATION.finditer(_expand_ranges(paragraph.text)):
            number = int(match.group(1))
            if number in listed and number not in order:
                order.append(number)
    mapping = {old: new for new, old in enumerate(order, start=1)}
    dropped = sorted(listed - set(order))

    def rewrite(text: str) -> str:
        text = _expand_ranges(text)

        # replace maximal groups of adjacent markers so ranges can re-form
        def group(match):
            numbers = [int(n) for n in CITATION.findall(match.group(0))]
            if any(n not in mapping for n in numbers):
                raise AssertionError(f"citation to a dropped/unknown reference: {match.group(0)}")
            return _format_markers(mapping[n] for n in numbers)

        # Keep punctuation and whitespace after the final marker outside the
        # match. The former pattern consumed a trailing `, `, producing joins
        # such as `[16]and` and `[45]which` throughout the rendered report.
        citation_group = r"\[\d+\](?:(?:\s*,\s*|\s*[–—-]\s*|\s*)\[\d+\])*"
        return re.sub(citation_group, group, text)

    for paragraph in body:
        if "[" not in paragraph.text:
            continue
        for run in paragraph.runs:
            if CITATION.search(run.text):
                run.text = rewrite(run.text)

    # rewrite the bibliography: keep printed order of the retained entries by new number
    retained = sorted(
        ((mapping[number], paragraph) for number, paragraph in reference_paragraphs if number in mapping),
        key=lambda pair: pair[0],
    )
    texts = {}
    for new_number, paragraph in retained:
        texts[new_number] = re.sub(r"^\[\d+\]", f"[{new_number}]", paragraph.text, count=1)
    slots = [paragraph for _, paragraph in reference_paragraphs]
    for slot, new_number in zip(slots, sorted(texts)):
        if slot.runs:
            slot.runs[0].text = texts[new_number]
            for extra in slot.runs[1:]:
                extra.text = ""
    for slot in slots[len(texts):]:
        slot._element.getparent().remove(slot._element)

    return {"kept": len(texts), "dropped": dropped, "mapping": mapping}


def _assert_ieee_order(doc) -> None:
    reference_paragraphs, body = _split_document(doc)
    listed = [number for number, _ in reference_paragraphs]
    if listed != list(range(1, len(listed) + 1)):
        raise AssertionError(f"bibliography is not numbered 1..N after renumbering: {listed[:12]}")
    order = []
    for paragraph in body:
        for match in CITATION.finditer(_expand_ranges(paragraph.text)):
            number = int(match.group(1))
            if number in set(listed) and number not in order:
                order.append(number)
    uncited = sorted(set(listed) - set(order))
    if uncited:
        raise AssertionError(f"references listed but never cited after renumbering: {uncited}")
    if order != listed:
        first = next((i for i, (a, b) in enumerate(zip(order, listed), 1) if a != b), None)
        raise AssertionError(
            f"citations are not in strict first-use order; diverges at position {first}: {order[:12]}"
        )


def _load_v3_builder():
    spec = importlib.util.spec_from_file_location("build_rewritten_fyp_report_v3_for_v4", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace(paragraph, text: str, base) -> None:
    base._replace_paragraph_text(paragraph, text)


def _find_paragraph(doc: Document, exact: str):
    matches = [paragraph for paragraph in doc.paragraphs if paragraph.text == exact]
    if len(matches) != 1:
        raise AssertionError(f"Expected one paragraph {exact!r}, found {len(matches)}")
    return matches[0]


def _all_text_paragraphs(doc: Document):
    """Body paragraphs plus every paragraph inside a table cell."""
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _apply_corrections(doc: Document, base) -> int:
    """Apply CORRECTIONS, requiring each to match exactly once."""
    paragraphs = list(_all_text_paragraphs(doc))
    for old, new in CORRECTIONS:
        hits = [paragraph for paragraph in paragraphs if old in paragraph.text]
        if len(hits) != 1:
            raise AssertionError(
                f"correction matched {len(hits)} paragraphs, expected exactly 1: {old[:90]!r}"
            )
        base._replace_paragraph_text(hits[0], hits[0].text.replace(old, new))
    return len(CORRECTIONS)


def _apply_post_renumber_corrections(doc: Document, base) -> int:
    """Apply source-fitness repairs whose citations use final v4 numbering."""
    paragraphs = list(_all_text_paragraphs(doc))
    for old, new in POST_RENUMBER_CORRECTIONS:
        hits = [paragraph for paragraph in paragraphs if old in paragraph.text]
        if len(hits) != 1:
            raise AssertionError(
                f"post-renumber correction matched {len(hits)} paragraphs, expected exactly 1: {old[:90]!r}"
            )
        base._replace_paragraph_text(hits[0], hits[0].text.replace(old, new))
    return len(POST_RENUMBER_CORRECTIONS)


def _restore_chapter_two_mechanism_citations(doc: Document, base) -> None:
    """Attach the two sources named in Chapter 2 before IEEE renumbering.

    The identifiers here are deliberately the inherited v3 numbers.  The v4
    renumberer runs afterwards and assigns their final first-use numbers.  An
    exact one-paragraph match makes this fail closed if the prose moves.
    """
    old = "Together with work on refusal directions and shallow alignment, this provides"
    new = (
        "Together with work on refusal directions [15] and shallow alignment [16], "
        "this provides"
    )
    matches = [paragraph for paragraph in _all_text_paragraphs(doc) if old in paragraph.text]
    if len(matches) != 1:
        raise AssertionError(
            f"mechanism citation restoration matched {len(matches)} paragraphs, expected exactly 1"
        )
    base._replace_paragraph_text(matches[0], matches[0].text.replace(old, new))


def _fix_alt_text(doc: Document) -> int:
    """Rewrite drawing alt text so it agrees with the printed caption."""
    fixed = 0
    for shape in doc.element.body.iter(
        "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"
    ):
        current = shape.get("descr")
        if current in ALT_TEXT_FIXES:
            shape.set("descr", ALT_TEXT_FIXES[current])
            fixed += 1
    if fixed != len(ALT_TEXT_FIXES):
        raise AssertionError(f"alt-text fix matched {fixed} drawings, expected {len(ALT_TEXT_FIXES)}")
    return fixed


def _fix_abbreviations(doc: Document, base) -> None:
    """Drop unused entries from the abbreviations table and add the missing ones."""
    table = next(
        (t for t in doc.tables if t.rows and t.rows[0].cells[0].text.strip() == "Term"),
        None,
    )
    if table is None:
        raise AssertionError("abbreviations table not found (no header cell 'Term')")

    for term in ABBREVIATION_DROPS:
        row = next((r for r in table.rows if r.cells[0].text.strip() == term), None)
        if row is None:
            raise AssertionError(f"abbreviation {term!r} not present to drop")
        row._element.getparent().remove(row._element)

    template = table.rows[1]
    for term, meaning in ABBREVIATION_ADDS:
        if any(r.cells[0].text.strip() == term for r in table.rows):
            raise AssertionError(f"abbreviation {term!r} already listed")
        import copy

        new_row = copy.deepcopy(template._element)
        template._element.getparent().append(new_row)
        row = table.rows[-1]
        base._replace_paragraph_text(row.cells[0].paragraphs[0], term)
        base._replace_paragraph_text(row.cells[1].paragraphs[0], meaning)

    # keep the table alphabetical, as it already was
    header = table.rows[0]._element
    body_rows = [r._element for r in table.rows[1:]]
    parent = header.getparent()
    for element in body_rows:
        parent.remove(element)
    for element in sorted(body_rows, key=lambda e: "".join(e.itertext()).strip().lower()):
        parent.append(element)


def _find_next_body_paragraphs(doc: Document, heading_text: str, count: int):
    paragraphs = doc.paragraphs
    heading_indices = [i for i, paragraph in enumerate(paragraphs) if paragraph.text == heading_text]
    if len(heading_indices) != 1:
        raise AssertionError(f"Expected one heading {heading_text!r}, found {len(heading_indices)}")
    start = heading_indices[0] + 1
    found = []
    for paragraph in paragraphs[start:]:
        if paragraph.style.name.startswith("Heading"):
            break
        if paragraph.text.strip():
            found.append(paragraph)
            if len(found) == count:
                return found
    raise AssertionError(f"Could not find {count} body paragraphs after {heading_text!r}")


def build_report() -> Path:
    base = _load_v3_builder()
    with tempfile.TemporaryDirectory(prefix="fyp_report_v4_") as tmp:
        tmp_docx = Path(tmp) / "v3_source.docx"
        original_out = base.OUT
        try:
            base.OUT = tmp_docx
            base.build_report()
        finally:
            base.OUT = original_out

        doc = Document(tmp_docx)
        intro_paragraphs = _find_next_body_paragraphs(doc, "1.1 Background and Motivation", 4)
        for paragraph, replacement in zip(intro_paragraphs[:3], INTRODUCTION):
            _replace(paragraph, replacement, base)
        # The fourth paragraph was a one-sentence transition that restated 1.2
        # and dangled after a paragraph already closing the section. Remove it
        # rather than rewrite it: 1.2 makes the point directly.
        orphan = intro_paragraphs[3]
        orphan._element.getparent().remove(orphan._element)

        problem_paragraphs = _find_next_body_paragraphs(doc, "1.2 Problem Statement", 3)
        _replace(problem_paragraphs[0], PROBLEM_STATEMENT, base)
        _replace(problem_paragraphs[1], OBJECTIVE_STATEMENT, base)

        applied = _apply_corrections(doc, base)
        _restore_chapter_two_mechanism_citations(doc, base)
        _fix_alt_text(doc)
        _fix_abbreviations(doc, base)

        for paragraph in doc.paragraphs:
            if "\t" not in paragraph.text:
                continue
            title, page = paragraph.text.rsplit("\t", 1)
            if title in V4_PAGE_NUMBERS and page.isdigit():
                _replace(paragraph, f"{title}\t{V4_PAGE_NUMBERS[title]}", base)

        chapter_one = []
        in_chapter_one = False
        for paragraph in doc.paragraphs:
            if paragraph.text == "Chapter 1: Introduction":
                in_chapter_one = True
            elif in_chapter_one and paragraph.text == "Chapter 2: Literature Review":
                break
            if in_chapter_one:
                chapter_one.append(paragraph.text)
        text = "\n".join(chapter_one)
        for forbidden in ("Gemma 2", "toxicity, bias", "Deepseek R1", "popular GPTs model"):
            if forbidden.lower() in text.lower():
                raise AssertionError(f"Legacy assignment-introduction wording remains: {forbidden!r}")
        for expected in INTRODUCTION + [PROBLEM_STATEMENT, OBJECTIVE_STATEMENT]:
            if expected not in text:
                raise AssertionError(f"v4 introduction text is missing: {expected[:80]!r}")

        summary = _renumber_citations(doc)
        _assert_ieee_order(doc)
        post_applied = _apply_post_renumber_corrections(doc, base)
        _assert_ieee_order(doc)
        print(
            f"    citations renumbered: {summary['kept']} references, "
            f"dropped as uncited: {summary['dropped'] or 'none'}"
        )
        print(f"    text corrections applied: {applied}")
        print(f"    post-renumber source corrections applied: {post_applied}")

        doc.core_properties.subject = "Plain-academic-English FYP report rewrite v4"
        doc.core_properties.comments = (
            "Non-destructive v4 rewrite derived from the v3 evidence and layout source. "
            "Chapter 1 uses the author's HW0288 funnel introduction, adapted to the report bibliography."
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build_report())
