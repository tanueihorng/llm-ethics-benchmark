"""Per-subject MMLU accuracy + pairwise deltas (no GPU, redacted output).

The headline capability claim (RQ3) rests on a single 300-question MMLU subset.
This script decomposes that subset by subject so the report can show the
capability loss is broad-based rather than driven by one subject — directly
strengthening the capability anchor without any new inference.

Reads each model's ``results/<alias>/mmlu/raw.jsonl`` READ-ONLY (TC1-original,
immutable) and uses only ``score_fields.subject`` + ``score_fields.is_correct``.
No prompt or response text is read into the outputs. Writes
``results/analysis/mmlu_subject_breakdown.{json,csv}`` (subjects, counts, and
accuracies only).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODEL_ALIASES = [
    "qwen_2b_base",
    "qwen_2b_4bit",
    "qwen_4b_base",
    "qwen_4b_4bit",
    "llama_3_2_3b_base",
    "llama_3_2_3b_4bit",
    "mistral_7b_base",
    "mistral_7b_4bit",
    "phi4_mini_base",
    "phi4_mini_4bit",
]

# pair_id -> (baseline_alias, quantized_alias)
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def _subject_counts(raw_path: Path) -> Dict[str, Dict[str, int]]:
    """Returns ``{subject: {"correct": int, "total": int}}`` from an MMLU raw.jsonl."""

    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    if not raw_path.exists():
        return {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sf = rec.get("score_fields") or {}
            subject = sf.get("subject")
            if subject is None:
                continue
            if not sf.get("is_answered", True):
                # Unanswered/malformed records still count toward the denominator
                # for accuracy parity with summary.json, but are never "correct".
                out[subject]["total"] += 1
                continue
            out[subject]["total"] += 1
            if sf.get("is_correct"):
                out[subject]["correct"] += 1
    return dict(out)


def _accuracy(cell: Dict[str, int]) -> Optional[float]:
    return cell["correct"] / cell["total"] if cell["total"] else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-subject MMLU breakdown (redacted).")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_model: Dict[str, Dict[str, Dict[str, int]]] = {}
    for alias in MODEL_ALIASES:
        per_model[alias] = _subject_counts(results_dir / alias / "mmlu" / "raw.jsonl")

    if not any(per_model.values()):
        print("No MMLU raw.jsonl files found under", results_dir)
        return 1

    # Per-pair, per-subject accuracy deltas (4-bit minus baseline).
    per_pair: List[Dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        base_counts = per_model.get(base_alias, {})
        quant_counts = per_model.get(quant_alias, {})
        subjects = sorted(set(base_counts) | set(quant_counts))
        for subject in subjects:
            base_cell = base_counts.get(subject, {"correct": 0, "total": 0})
            quant_cell = quant_counts.get(subject, {"correct": 0, "total": 0})
            base_acc = _accuracy(base_cell)
            quant_acc = _accuracy(quant_cell)
            delta = (
                quant_acc - base_acc
                if base_acc is not None and quant_acc is not None
                else None
            )
            per_pair.append({
                "pair_id": pair_id,
                "subject": subject,
                "n": base_cell["total"],
                "baseline_accuracy": base_acc,
                "quantized_accuracy": quant_acc,
                "accuracy_delta": delta,
            })

    report = {
        "per_model": {
            alias: {
                subject: {
                    "correct": cell["correct"],
                    "total": cell["total"],
                    "accuracy": _accuracy(cell),
                }
                for subject, cell in subjects.items()
            }
            for alias, subjects in per_model.items()
        },
        "per_pair_subject_deltas": per_pair,
    }
    (analysis_dir / "mmlu_subject_breakdown.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    csv_path = analysis_dir / "mmlu_subject_breakdown.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "pair_id", "subject", "n", "baseline_accuracy", "quantized_accuracy", "accuracy_delta",
        ])
        for row in per_pair:
            writer.writerow([
                row["pair_id"], row["subject"], row["n"],
                row["baseline_accuracy"], row["quantized_accuracy"], row["accuracy_delta"],
            ])

    # Console (redacted): per-pair subject deltas + how many subjects regressed.
    print("=" * 78)
    print("Per-subject MMLU accuracy deltas (4-bit minus baseline)")
    print("=" * 78)
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        rows = [r for r in per_pair if r["pair_id"] == pair_id]
        regressed = sum(1 for r in rows if isinstance(r["accuracy_delta"], float) and r["accuracy_delta"] < 0)
        print(f"\n{pair_id}  ({len(rows)} subjects; {regressed} regressed)")
        print(f"  {'subject':<32} {'base':>6} {'4bit':>6} {'Δ':>8}")
        for r in sorted(rows, key=lambda x: (x["accuracy_delta"] is None, x["accuracy_delta"] or 0.0)):
            def _f(x: Optional[float]) -> str:
                return f"{x:.3f}" if isinstance(x, float) else "  n/a"
            d = r["accuracy_delta"]
            d_s = f"{d:+.3f}" if isinstance(d, float) else "  n/a"
            print(f"  {r['subject']:<32} {_f(r['baseline_accuracy']):>6} {_f(r['quantized_accuracy']):>6} {d_s:>8}")
    print()
    print(f"Wrote {analysis_dir / 'mmlu_subject_breakdown.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
