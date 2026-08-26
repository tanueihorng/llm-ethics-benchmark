"""Scope boundaries and construct definitions must survive as STANDALONE sentences.

A prose edit can preserve every number, every citation and every citation's
first-use order while still damaging the report, by demoting a limitation into a
subordinate clause:

    "The study evaluates direct requests. No jailbreak augmentation was applied."
    -> "Although no jailbreak augmentation was applied, the study ..."

Both forms contain the same tokens, so `verify_report_claims` and the numeric /
citation diffs used during prose passes cannot see the difference. What changed
is the sentence's force, which is exactly the D56 claim-fitness defect class
arriving through a different door.

This module pins the sentences whose RIGIDITY is load-bearing: threat-model
boundaries, construct definitions, the bounded-null statement, evidence-grade
labels, and the non-novelty disclaimers. Each must appear in the rendered report
as its own complete sentence -- not merged, not subordinated, not reworded.

Clearing a failure here means restoring the sentence, never relaxing the list.
Adding a genuinely new scope sentence to the report should add a line below.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"
# The interim build derives its body from the report, but it is a separate submitted
# artefact and is checked in its own right rather than by assumption (T53, T56).
INTERIM = REPO_ROOT / "docs" / "rewritten_fyp_interim_v1.docx"
GUARDED = {"report": REPORT, "interim": INTERIM}

# Sentences that must remain standalone. Grouped by why they are load-bearing.
LOCKED_SENTENCES = [
    # Threat model (D43): safety claims are scoped to direct requests.
    "Preservation under direct requests is not evidence that the same model will "
    "remain safe under a targeted attack.",
    # Construct definitions: what an outcome counts as.
    "A response is a HarmBench success only when the official behaviour-specific "
    "classifier judges that it exhibits or clearly attempts the target behaviour under "
    "the benchmark's criteria.",
    # The bounded null: what the headline does NOT establish.
    "It does not establish that NF4 preserves all forms of safety, that effects are "
    "absent for every small model, or that an effect smaller than the study's "
    "detection threshold is absent.",
    # Mechanism is a hypothesis, not a causal finding.
    "The behavioural data alone cannot identify the internal mechanism, so the "
    "explanation remains a hypothesis rather than a causal finding.",
    "It does not establish that any refusal decision changed in the present models.",
    # Evidence grade (D56): a preprint must say so.
    "Reference [35] is a non-archival workshop paper distributed as a preprint.",
    # Unmeasured axis: the cost side of the trade-off was not measured.
    "The study does not contain direct VRAM, latency, or throughput measurements, so "
    "it cannot quantify the engineering benefit side of that choice.",
    # The study's central interpretive rule.
    "A lower attack-success rate is not evidence of improved alignment if the same "
    "model also loses the ability to answer ordinary questions.",
    # Truncation analysis does not over-claim.
    "The mismatch category remains visible, so the analysis does not claim that every "
    "budget difference is truncation.",
    # Prior-work table (T52/D59): basis, absence-of-evidence, and non-novelty.
    "Entries are compiled from each cited paper's abstract and stated scope.",
    "“Not reported” is not evidence of absence.",
    "The table is not a novelty or priority claim.",
    # Literature synthesis must not collapse into a single directional claim.
    "The literature therefore does not support a single statement that quantization "
    "either preserves or harms alignment.",
    # --- added 2026-08-10 before the Chapter 2 flow pass (T58) ---
    # Threat-model boundary, stated where the review hands over to the design.
    "The present experiment instead uses direct harmful requests, so its claim is "
    "narrower: it estimates a precision effect under the DirectRequest condition.",
    # The intervention is named, and results do not generalise to "quantization".
    "A result obtained with one method is evidence about that intervention, not about "
    "all low-bit arithmetic.",
    # Safety-layer mechanism is a hypothesis for these models, not a finding.
    "It remains a hypothesis for the present models, because the experiment measures "
    "behaviour rather than identifying a causal internal circuit.",
    # Worst-case attack evidence must not be read as prevalence.
    "They do not imply that every normally trained checkpoint will fail under ordinary "
    "NF4 loading.",
    # The study's own claim ceiling.
    "It does not attempt to establish a universal law of quantization.",
    # Table 2.1's absence-of-evidence disclaimer, in the body as well as the caption.
    "“Not reported” means that the consulted source did not establish the property; it "
    "is not evidence that the authors did not consider it.",
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[“])")


def body_sentences(path: Path) -> list[str]:
    """Standalone sentences of the report body, Chapter 1 through References."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    xml = re.sub(r"<w:tbl>.*?</w:tbl>", "", xml, flags=re.S)  # tables are not prose
    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        style = re.search(r'w:val="(Heading\d|Title|Caption|TOC\d?)"', para)
        if text.strip():
            paragraphs.append((style.group(1) if style else "Body", text.strip()))
    start = next(i for i, (_, t) in enumerate(paragraphs) if t.startswith("Chapter 1"))
    end = next(
        i for i, (s, t) in enumerate(paragraphs) if s == "Heading1" and t.strip() == "References"
    )
    body = " ".join(t for s, t in paragraphs[start:end] if s == "Body")
    return [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip()]


@pytest.fixture(scope="module")
def sentences_by_document() -> dict[str, list[str]]:
    present = {name: path for name, path in GUARDED.items() if path.exists()}
    if not present:
        pytest.skip("no rewrite-family document is present in this checkout")
    return {name: body_sentences(path) for name, path in present.items()}


@pytest.mark.parametrize("document", sorted(GUARDED))
@pytest.mark.parametrize("locked", LOCKED_SENTENCES, ids=lambda s: s[:45])
def test_scope_sentence_survives_as_its_own_sentence(
    locked: str, document: str, sentences_by_document: dict[str, list[str]]
) -> None:
    if document not in sentences_by_document:
        pytest.skip(f"{GUARDED[document].name} is not present in this checkout")
    sentences = sentences_by_document[document]
    if locked in sentences:
        return
    softened = [s for s in sentences if locked.rstrip(".") in s]
    if softened:
        pytest.fail(
            f"[{document}] scope sentence is no longer standalone -- it was merged or "
            "subordinated, which preserves every token while changing the claim's force:\n"
            f"  locked: {locked}\n"
            f"  found:  {softened[0]}"
        )
    pytest.fail(f"[{document}] scope sentence is missing from the document entirely:\n  {locked}")


def test_locked_list_has_no_duplicates_and_is_sentence_shaped() -> None:
    assert len(set(LOCKED_SENTENCES)) == len(LOCKED_SENTENCES), "duplicate locked sentence"
    for locked in LOCKED_SENTENCES:
        assert locked.endswith("."), f"not a full sentence: {locked}"
        assert len(SENTENCE_SPLIT.split(locked)) == 1, f"more than one sentence: {locked}"
