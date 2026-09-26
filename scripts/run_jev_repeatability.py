#!/usr/bin/env python3
"""Repeat Jev judgments near the decision threshold, retaining redacted data only."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.judges.validation import JevJudgeBackend, JudgeInput  # noqa: E402
from scripts.jev_full_analysis import ALIASES  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selected(results_dir: Path, band: float) -> list[tuple[str, str, float, bool]]:
    selected = []
    for alias in ALIASES:
        path = results_dir / alias / "harmbench" / "scores.judge.jev.jsonl"
        for row in _read_jsonl(path):
            probability = row.get("judge_probability")
            if probability is None or abs(float(probability) - 0.5) > band:
                continue
            selected.append((
                alias,
                str(row["prompt_id"]),
                float(probability),
                bool(row["judge_harmful"]),
            ))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_512")
    parser.add_argument("--output-dir", default="results_512/analysis")
    parser.add_argument("--model-id", default="jev-1.13.0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--acknowledge-external-egress", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_external_egress:
        parser.error(
            "--acknowledge-external-egress is required because selected saved behavior "
            "and generation text will be sent transiently to TypeSafe"
        )
    if not 0 <= args.band <= 0.5:
        parser.error("--band must be between 0 and 0.5")

    results_dir = Path(args.results_dir)
    chosen = _selected(results_dir, args.band)
    raw_maps = {
        alias: {
            str(row["prompt_id"]): row
            for row in _read_jsonl(results_dir / alias / "harmbench" / "raw.jsonl")
        }
        for alias in ALIASES
    }
    inputs = []
    for alias, prompt_id, _probability, _label in chosen:
        raw = raw_maps[alias][prompt_id]
        inputs.append(JudgeInput(
            prompt_id=f"{alias}:{prompt_id}",
            behavior=str(raw.get("prompt_text", "")),
            generation=str(raw.get("response", "")),
        ))

    backend = JevJudgeBackend(model_id=args.model_id, threshold=args.threshold)
    try:
        repeated = backend.classify_batch(inputs)
    finally:
        backend.close()

    rows = []
    for original, result in zip(chosen, repeated):
        alias, prompt_id, original_probability, original_label = original
        rows.append({
            "model_alias": alias,
            "prompt_id": prompt_id,
            "original_probability": original_probability,
            "repeat_probability": result.probability,
            "original_harmful": original_label,
            "repeat_harmful": result.harmful,
            "status": result.status,
            "judge_model_id_requested": args.model_id,
            "decision_threshold": args.threshold,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "jev_repeatability.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    answered = [row for row in rows if row["status"] == "ok"]
    differences = [
        abs(float(row["repeat_probability"]) - float(row["original_probability"]))
        for row in answered
    ]
    summary = {
        "purpose": "jev_threshold_band_repeatability",
        "selection": f"abs(original_probability - 0.5) <= {args.band}",
        "judge_model_id_requested": args.model_id,
        "judge_model_ids_observed": sorted(backend.observed_model_ids),
        "decision_threshold": args.threshold,
        "selected_count": len(rows),
        "answered_count": len(answered),
        "error_count": len(rows) - len(answered),
        "binary_agreement": (
            sum(row["original_harmful"] == row["repeat_harmful"] for row in answered)
            / len(answered) if answered else None
        ),
        "threshold_flip_count": sum(
            row["original_harmful"] != row["repeat_harmful"] for row in answered
        ),
        "mean_absolute_probability_change": statistics.fmean(differences) if differences else None,
        "median_absolute_probability_change": statistics.median(differences) if differences else None,
        "max_absolute_probability_change": max(differences) if differences else None,
        "contains_raw_text": False,
    }
    (output_dir / "jev_repeatability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
