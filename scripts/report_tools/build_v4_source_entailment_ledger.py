#!/usr/bin/env python3
"""Attach human source-entailment verdicts to the v4 citation inventory.

The inventory builder supplies the independent denominator from the rendered
DOCX.  This file records the human reading outcome and page-level trail.  A
default of ``supported`` is used only after the unit was reviewed; exceptions
are explicit below so the material findings remain machine-filterable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "docs" / "audits" / "2026-08-24_v4_source_entailment"

# Physical PDF pages, not journal pagination.  "Record" means an authoritative
# DOI/publisher metadata page was checked because the full text is not held.
EVIDENCE: dict[int, str] = {
    1: "local PDF pp. 1–2",
    2: "local PDF pp. 1–2",
    3: "local PDF p. 1",
    4: "local PDF pp. 1–2",
    5: "local PDF pp. 1–4, 10",
    6: "local PDF p. 1",
    7: "local PDF pp. 1–2",
    8: "local PDF pp. 1–3",
    9: "local PDF pp. 2, 4",
    10: "IEEE version of record pp. 1, 3, 6, 9–12 and Table 14 on physical p. 19",
    11: "local PDF pp. 2, 4–8",
    12: "PMLR version of record pp. 1, 3–5, 32",
    13: "local PDF pp. 1, 3–5",
    14: "local PDF pp. 1, 31",
    15: "local PDF pp. 2, 5",
    16: "local PDF pp. 1–2",
    17: "local PDF pp. 1–2",
    18: "local PDF pp. 1–3",
    19: "local PDF p. 1",
    20: "local PDF p. 1",
    21: "local PDF pp. 1, 4, 6",
    22: "local PDF pp. 1, 4",
    23: "local PDF p. 1",
    24: "local PDF pp. 1, 4, 27",
    25: "local PDF pp. 1–2",
    26: "local PDF p. 1",
    27: "local PDF pp. 1, 3–4",
    28: "local PDF pp. 1–4",
    29: "local PDF pp. 1–7",
    30: "local PDF pp. 1–8, 14–16",
    31: "local PDF pp. 4, 6, 15–16, 22",
    32: "local PDF pp. 1–3",
    33: "local PDF pp. 1–3, 7–10",
    34: "local PDF pp. 1–2, 27",
    35: "local PDF pp. 1–2",
    36: "ICLR paper pp. 1–4",
    37: "local PDF pp. 1–2",
    38: "local PDF pp. 1, 19–25",
    39: "local PDF pp. 1–6, 15",
    40: "local PDF pp. 1, 5–9",
    41: "local PDF pp. 1–3",
    42: "local PDF pp. 1, 5, 9",
    43: "local PDF pp. 1–9",
    44: "local PDF pp. 1–5, 9",
    45: "local PDF pp. 1–5",
    46: "local PDF pp. 1–2",
    47: "book identity verified; correct full text not held locally",
    48: "DOI 10.1007/BF02295996 metadata/abstract; full text not held locally",
    49: "DOI 10.1111/j.2517-6161.1995.tb02031.x metadata/abstract; full text not held locally",
    50: "DOI 10.1177/001316446002000104 metadata; full text not held locally",
    51: "DOI 10.2307/2531961 metadata; full text not held locally",
    52: "local PDF pp. 1–5",
    53: "local PDF pp. 1–2, 16, 19",
    54: "DOI 10.2307/2529310 metadata; full text not held locally",
    55: "local PDF pp. 1–2",
    56: "DOI 10.1007/BF01068419 publisher abstract; full text not held locally",
    57: "local PDF pp. 1–2",
    58: "local PDF pp. 16–28",
    59: "local PDF pp. 1–2, 9",
    60: "DOI 10.2307/3001968 metadata; full text not held locally",
    61: "publisher record DOI 10.4324/9780203771587; full text not held locally",
}


# unit_id: (verdict, priority, rationale)
OVERRIDES = {
    "P002": (
        "supported_with_qualification",
        "P3",
        "The sources establish low-bit/on-device practicality, but do not benchmark the exact five report checkpoints on a single consumer-GPU definition.",
    ),
    "P013": (
        "supported_with_qualification",
        "P3",
        "The source verifies SafetyBench's multiple-choice safety-knowledge format; the knowledge-versus-generative-refusal distinction is the report's stated methodological inference.",
    ),
    "P024": (
        "supported_with_qualification",
        "P3",
        "The reviewed evaluations support broad average retention, but results are heterogeneous by model, task, method and scale.",
    ),
    "P032": (
        "supported_as_hypothesis",
        "P3",
        "The cited papers establish a refusal direction and shallow alignment; the quantization mechanism is explicitly labelled a hypothesis in the following sentence.",
    ),
    "P035": (
        "supported_as_synthesis",
        "P3",
        "The paper supports broad four-bit preservation; 'most similar' is a transparent comparison made by this report, not a finding stated by the source.",
    ),
    "P062": (
        "supported_with_qualification",
        "P3",
        "The sources document evaluator design and human comparison; 'measurement systems evolve' is a broad synthesis rather than a directly tested proposition.",
    ),
    "P063": ("method_identity_verified", "none", "The citations name the paired bootstrap and exact McNemar procedures used by the report."),
    "P065": ("method_identity_verified", "none", "The citation identifies bootstrap resampling; prompt-level pairing is also a property of the report's implementation."),
    "P066": ("method_identity_verified", "none", "McNemar's paper is the correct source for a matched binary comparison based on discordant pairs."),
    "P067": ("method_identity_verified", "none", "Benjamini and Hochberg is the correct source for false-discovery-rate control."),
    "P068": ("method_identity_verified", "none", "Cohen is the correct source for chance-corrected nominal agreement."),
    "P069": ("method_identity_verified", "none", "Connor and Card are fit sources for paired-proportion power/sample-size reasoning and power-aware NLP evaluation."),
    "P070": ("method_identity_verified", "none", "Artstein and Poesio support prevalence/coefficient caution; Landis and Koch supply the conventional verbal bands."),
    "P072": ("method_identity_verified", "none", "Schuirmann is the correct source for the two one-sided tests equivalence procedure."),
    "P076": ("method_identity_verified", "none", "Wilcoxon and Cohen are the named sources for the stated non-parametric comparison and power interpretation."),
    "P090": (
        "supported_with_qualification",
        "P3",
        "The reviewed studies support average four-bit retention, but the sentence correctly limits that generalisation by pointing to the report's compact-model counterexample.",
    ),
    "P093": (
        "supported_with_qualification",
        "P3",
        "Reference 28 establishes language-dependent quantization effects; the prediction about Qwen and non-Latin scripts is explicitly an external-validity inference.",
    ),
}


def main() -> None:
    units = json.loads((AUDIT_DIR / "inventory.json").read_text(encoding="utf-8"))
    manifest = json.loads((AUDIT_DIR / "audit_manifest.json").read_text(encoding="utf-8"))
    if manifest["citation_bearing_units"] != len(units):
        raise SystemExit("Inventory/manifest count mismatch")
    missing_evidence = sorted({ref for unit in units for ref in unit["references"]} - EVIDENCE.keys())
    if missing_evidence:
        raise SystemExit(f"No evidence trail for references: {missing_evidence}")
    unknown_overrides = sorted(set(OVERRIDES) - {unit["unit_id"] for unit in units})
    if unknown_overrides:
        raise SystemExit(f"Stale verdict overrides: {unknown_overrides}")

    fields = [
        "unit_id", "kind", "chapter", "section", "block_index", "row_index",
        "text", "references", "verdict", "priority", "evidence", "rationale",
    ]
    rows = []
    for unit in units:
        verdict, priority, rationale = OVERRIDES.get(
            unit["unit_id"],
            ("supported", "none", "The cited source text supports the attributed description at the recorded locator."),
        )
        refs = unit["references"]
        rows.append(
            {
                **unit,
                "references": ",".join(map(str, refs)),
                "verdict": verdict,
                "priority": priority,
                "evidence": "; ".join(f"[{ref}] {EVIDENCE[ref]}" for ref in refs),
                "rationale": rationale,
            }
        )

    with (AUDIT_DIR / "ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    (AUDIT_DIR / "ledger_summary.json").write_text(
        json.dumps(
            {
                "audited_docx_sha256": manifest["audited_docx_sha256"],
                "units": len(rows),
                "verdict_counts": dict(sorted(counts.items())),
                "priority_findings": {
                    priority: sum(row["priority"] == priority for row in rows)
                    for priority in ("P1", "P2", "P3")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((AUDIT_DIR / "ledger_summary.json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
