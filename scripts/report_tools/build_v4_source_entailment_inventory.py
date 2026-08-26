#!/usr/bin/env python3
"""Build the reproducible source-entailment inventory for rewritten interim v4.

This script deliberately reads the rendered DOCX rather than trusting the builder's
reference array.  It inventories citation-bearing prose sentences and table rows,
expands IEEE citation ranges, and records the v4-reference-to-local-PDF mapping.
It does not assign semantic verdicts; those require reading the cited source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCX = ROOT / "docs" / "rewritten_fyp_interim_v4.docx"
OUT_DIR = ROOT / "docs" / "audits" / "2026-08-24_v4_source_entailment"

CITATION_RE = re.compile(
    r"\[(\d+)\](?:\s*[\-–—]\s*\[(\d+)\])?|\[(\d+(?:\s*,\s*\d+)+)\]"
)
REFERENCE_RE = re.compile(r"^\[(\d+)\]\s+(.*)$")


# The Sources/ filenames retain the pre-v4 numbering.  This mapping is by work
# identity, not by coincidental numeric equality.
LOCAL_SOURCES: dict[int, str | None] = {
    1: "Sources/[01] Yang et al. - Qwen3 technical report.pdf",
    2: "Sources/[02] Grattafiori et al. - The Llama 3 herd of models.pdf",
    3: "Sources/[03] Abouelenin et al. - Phi-4-Mini technical report - Compact yet powerful multimodal language models via Mixture-of-LoRAs.pdf",
    4: "Sources/[51] Jiang et al. - Mistral 7B.pdf",
    5: "Sources/[05] Lin et al. - AWQ - Activation-aware weight quantization for on-device LLM compression and acceleration.pdf",
    6: "Sources/[06] Frantar et al. - GPTQ - Accurate post-training quantization for generative pre-trained transformers.pdf",
    7: "Sources/[07] Dettmers et al. - The case for 4-bit precision - k-bit inference scaling laws.pdf",
    8: "Sources/[08] Jin et al. - A comprehensive evaluation of quantization strategies for large language models.pdf",
    9: "Sources/[09] Li et al. - Evaluating quantized large language models.pdf",
    10: "Sources/[10.1] Kharinaev et al. - Investigating the impact of quantization methods on the safety and reliability of large language models (IEEE Access version of record).pdf",
    11: "Sources/[18] Röttger et al. - XSTest - A test suite for identifying exaggerated safety behaviours in large language models.pdf",
    12: "Sources/[24.1] Mazeika et al. - HarmBench - A standardized evaluation framework for automated red teaming and robust refusal (PMLR v235 version of record).pdf",
    13: "Sources/[20] Huang et al. - Position - TrustLLM - Trustworthiness in large language models.pdf",
    14: "Sources/[21] Wang et al. - DecodingTrust - A comprehensive assessment of trustworthiness in GPT models.pdf",
    15: "Sources/[22] Zhang et al. - SafetyBench - Evaluating the safety of large language models.pdf",
    16: "Sources/[23] Ouyang et al. - Training language models to follow instructions with human feedback.pdf",
    17: "Sources/[25] Zou et al. - Universal and transferable adversarial attacks on aligned language models.pdf",
    18: "Sources/[26] Perez et al. - Red teaming language models with language models.pdf",
    19: "Sources/[27] Hendrycks et al. - Measuring massive multitask language understanding.pdf",
    20: "Sources/[28] Clark et al. - Think you have solved question answering Try ARC, the AI2 reasoning challenge.pdf",
    21: "Sources/[29] Badshah et al. - Quantifying the capabilities of LLMs across scale and precision.pdf",
    22: "Sources/[30] Xiao et al. - SmoothQuant - Accurate and efficient post-training quantization for large language models.pdf",
    23: "Sources/[31] Frantar et al. - SparseGPT - Massive language models can be accurately pruned in one-shot.pdf",
    24: "Sources/[32] Dettmers et al. - QLoRA - Efficient finetuning of quantized LLMs.pdf",
    25: "Sources/[33] Dettmers et al. - LLM.int8() - 8-bit matrix multiplication for transformers at scale.pdf",
    26: "Sources/[34] Guo et al. - On calibration of modern neural networks.pdf",
    27: "Sources/[11] Proskurina et al. - When quantization affects confidence of large language models.pdf",
    28: "Sources/[13] Marchisio et al. - How does quantization affect multilingual LLMs.pdf",
    29: "Sources/[35] Belkhiter et al. - HarmLevelBench - Evaluating harm-level compliance and the impact of quantization on model alignment.pdf",
    30: "Sources/[36] Chen et al. - Assessing safety risks and quantization-aware safety patching for quantized large language models.pdf",
    31: "Sources/[37] Hong et al. - Decoding compressed trust - Scrutinizing the trustworthiness of efficient LLMs under compression.pdf",
    32: "Sources/[38] Shao et al. - OmniQuant - Omnidirectionally calibrated quantization for large language models.pdf",
    33: "Sources/[39] Dong et al. - Durable quantization conditioned misalignment attack on large language models.pdf",
    34: "Sources/[40] Li et al. - Safety layers in aligned large language models - The key to LLM security.pdf",
    35: "Sources/[15] Arditi et al. - Refusal in language models is mediated by a single direction.pdf",
    36: "Sources/[16] Qi et al. - Safety alignment should be made more than just a few tokens deep.pdf",
    37: "Sources/[12] Egashira et al. - Exploiting LLM quantization.pdf",
    38: "Sources/[41] Souly et al. - A StrongREJECT for empty jailbreaks.pdf",
    39: "Sources/[42] Xie et al. - SORRY-Bench - Systematically evaluating large language model safety refusal.pdf",
    40: "Sources/[43] Chao et al. - JailbreakBench - An open robustness benchmark for jailbreaking large language models.pdf",
    41: "Sources/[44] Han et al. - WildGuard - Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs.pdf",
    42: "Sources/[19] Cui et al. - OR-Bench - An over-refusal benchmark for large language models.pdf",
    43: "Sources/[45] Wang et al. - Do-Not-Answer - Evaluating safeguards in LLMs.pdf",
    44: "Sources/[46] Kim et al. - Prometheus 2 - An open source language model specialized in evaluating other language models.pdf",
    45: "Sources/[47] Zheng et al. - Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.pdf",
    46: "Sources/[48] Wei et al. - Jailbroken - How does LLM safety training fail.pdf",
    47: None,
    48: None,
    49: None,
    50: None,
    51: None,
    52: "Sources/[55] Card et al. - With little power comes great responsibility.pdf",
    53: "Sources/[56] Artstein et al. - Survey article - Inter-coder agreement for computational linguistics.pdf",
    54: None,
    55: "Sources/[58] Inan et al. - Llama Guard - LLM-based input-output safeguard for human-AI conversations.pdf",
    56: None,
    57: "Sources/[60] Sainz et al. - NLP evaluation in trouble - On the need to measure LLM data contamination for each benchmark.pdf",
    58: "Sources/[61] Gu et al. - A survey on LLM-as-a-judge.pdf",
    59: "Sources/[62] Krumdick et al. - No free labels - Limitations of LLM-as-a-judge without human grounding.pdf",
    60: None,
    61: None,
}

# This similarly named download is not Efron and Tibshirani's book.  Retaining
# the mismatch explicitly prevents a future audit from counting it as coverage.
MISMATCHED_CANDIDATES = {
    47: "/Users/tanueihorng/Downloads/Introduction_to_Bootstrap.pdf",
}


@dataclass
class Unit:
    unit_id: str
    kind: str
    chapter: str
    section: str
    block_index: int
    row_index: int | None
    text: str
    references: list[int]


def iter_blocks(parent: DocumentObject):
    for child in parent.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, parent)


def clean(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def expand_citations(text: str) -> list[int]:
    refs: set[int] = set()
    for match in CITATION_RE.finditer(text):
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            refs.update(range(min(start, end), max(start, end) + 1))
        else:
            refs.update(int(value.strip()) for value in match.group(3).split(","))
    return sorted(refs)


def split_sentences(text: str) -> list[str]:
    """Conservative splitter sufficient for the report's academic prose."""
    protected = text
    replacements = {
        "et al.": "et al<prd>",
        "e.g.": "e<prd>g<prd>",
        "i.e.": "i<prd>e<prd>",
        "Fig.": "Fig<prd>",
        "Eq.": "Eq<prd>",
        "Sec.": "Sec<prd>",
        "Dr.": "Dr<prd>",
        "Prof.": "Prof<prd>",
        "vs.": "vs<prd>",
    }
    for source, target in replacements.items():
        protected = protected.replace(source, target)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<prd>", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“‘\[(])", protected)
    return [clean(part.replace("<prd>", ".")) for part in parts if clean(part)]


def main() -> None:
    document = Document(DOCX)
    units: list[Unit] = []
    bibliography: dict[int, str] = {}
    chapter = "Front matter"
    section = "Front matter"
    in_references = False
    block_index = 0

    for kind, block in iter_blocks(document):
        if kind == "paragraph":
            text = clean(block.text)
            style = block.style.name if block.style else ""
            if style == "Heading 1" and text:
                in_references = text == "References"
                chapter = text
                section = text
            elif style in {"Heading 2", "Heading 3"} and text:
                section = text

            if in_references:
                match = REFERENCE_RE.match(text)
                if match:
                    bibliography[int(match.group(1))] = match.group(2)
            else:
                for sentence in split_sentences(text):
                    refs = expand_citations(sentence)
                    if refs:
                        units.append(
                            Unit(
                                unit_id=f"P{len(units) + 1:03d}",
                                kind="prose_sentence",
                                chapter=chapter,
                                section=section,
                                block_index=block_index,
                                row_index=None,
                                text=sentence,
                                references=refs,
                            )
                        )
        else:
            if not in_references:
                for row_index, row in enumerate(block.rows):
                    text = " | ".join(clean(cell.text) for cell in row.cells)
                    refs = expand_citations(text)
                    if refs:
                        units.append(
                            Unit(
                                unit_id=f"T{len(units) + 1:03d}",
                                kind="table_row",
                                chapter=chapter,
                                section=section,
                                block_index=block_index,
                                row_index=row_index,
                                text=text,
                                references=refs,
                            )
                        )
        block_index += 1

    if sorted(bibliography) != list(range(1, len(bibliography) + 1)):
        raise SystemExit(f"Expected a contiguous reference list, got {sorted(bibliography)}")
    if set(bibliography) != set(LOCAL_SOURCES):
        raise SystemExit(
            "Local-source map is stale: "
            f"bibliography={sorted(bibliography)}, mapping={sorted(LOCAL_SOURCES)}"
        )
    for unit in units:
        unknown = set(unit.references) - set(bibliography)
        if unknown:
            raise SystemExit(f"{unit.unit_id} cites unknown references: {sorted(unknown)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "inventory.json").write_text(
        json.dumps([asdict(unit) for unit in units], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (OUT_DIR / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_id",
                "kind",
                "chapter",
                "section",
                "block_index",
                "row_index",
                "text",
                "references",
            ],
        )
        writer.writeheader()
        for unit in units:
            row = asdict(unit)
            row["references"] = ",".join(map(str, unit.references))
            writer.writerow(row)

    with (OUT_DIR / "source_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reference",
                "bibliography_entry",
                "local_pdf",
                "local_status",
                "rejected_candidate",
                "rejection_reason",
            ],
        )
        writer.writeheader()
        for ref_number in sorted(bibliography):
            source = LOCAL_SOURCES[ref_number]
            source_path = None if source is None else Path(source)
            if source_path is not None and not source_path.is_absolute():
                source_path = ROOT / source_path
            exists = bool(source_path and source_path.exists())
            writer.writerow(
                {
                    "reference": ref_number,
                    "bibliography_entry": bibliography[ref_number],
                    "local_pdf": str(source_path) if source_path else "",
                    "local_status": "present" if exists else "not_found",
                    "rejected_candidate": MISMATCHED_CANDIDATES.get(ref_number, ""),
                    "rejection_reason": (
                        "Different work: Stephanie Fook Chong (2011), not Efron and Tibshirani (1993)."
                        if ref_number in MISMATCHED_CANDIDATES
                        else ""
                    ),
                }
            )

    expanded_occurrences = sum(len(unit.references) for unit in units)
    prose = sum(unit.kind == "prose_sentence" for unit in units)
    table = sum(unit.kind == "table_row" for unit in units)
    summary = {
                "references": len(bibliography),
                "citation_bearing_units": len(units),
                "prose_sentences": prose,
                "table_rows": table,
                "expanded_reference_occurrences": expanded_occurrences,
                "local_pdfs_present": sum(
                    1
                    for source in LOCAL_SOURCES.values()
                    if source
                    and (Path(source) if Path(source).is_absolute() else ROOT / source).exists()
                ),
                "audited_docx_sha256": hashlib.sha256(DOCX.read_bytes()).hexdigest(),
            }
    (OUT_DIR / "audit_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
