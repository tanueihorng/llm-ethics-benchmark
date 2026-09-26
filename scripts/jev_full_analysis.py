#!/usr/bin/env python3
"""Analyze Jev against all HarmBench judges and the human-validation subset.

Reads redacted judge sidecars plus the local-only human annotation sheet. Writes
aggregate JSON and a redacted review queue; never reads or writes benchmark text.
The official HarmBench classifier remains primary. The Jev multiple-comparison
family is an explicitly named sensitivity analysis, not a replacement family.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.analysis.compare_quant_pairs import (
    compute_paired_bootstrap_ci,
    mcnemar_exact_test,
    paired_binary_confusion,
)


ALIASES = [
    "qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit",
    "qwen_4b_base", "qwen_4b_4bit", "qwen_4b_8bit",
    "llama_3_2_3b_base", "llama_3_2_3b_4bit", "llama_3_2_3b_8bit",
    "mistral_7b_base", "mistral_7b_4bit", "mistral_7b_8bit",
    "phi4_mini_base", "phi4_mini_4bit", "phi4_mini_8bit",
]
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}
COMPARATORS = ("harmbench_cls", "api_judge", "llamaguard")


def _precision(alias: str) -> str:
    if alias.endswith("_4bit"):
        return "nf4"
    if alias.endswith("_8bit"):
        return "int8"
    return "fp16"


def _family(alias: str) -> str:
    for pair_id, pair_aliases in PAIRS.items():
        if alias in pair_aliases or alias == pair_aliases[0].replace("_base", "_8bit"):
            return pair_id
    raise ValueError(f"Unknown model alias: {alias}")


def _read_sidecar(results_dir: Path, alias: str, judge: str) -> dict[str, dict[str, Any]]:
    path = results_dir / alias / "harmbench" / f"scores.judge.{judge}.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_id = str(record["prompt_id"])
            harmful = record.get("judge_harmful")
            if harmful is None:
                continue
            rows[prompt_id] = {
                "label": bool(harmful),
                "probability": record.get("judge_probability"),
            }
    return rows


def _kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if abs(1 - expected) <= 1e-12 else (observed - expected) / (1 - expected)


def _agreement(reference: dict[str, bool], candidate: dict[str, bool]) -> dict[str, Any]:
    shared = sorted(set(reference) & set(candidate))
    ref = [reference[key] for key in shared]
    cand = [candidate[key] for key in shared]
    both_harmful = sum(x and y for x, y in zip(ref, cand))
    both_safe = sum((not x) and (not y) for x, y in zip(ref, cand))
    reference_only = sum(x and not y for x, y in zip(ref, cand))
    candidate_only = sum((not x) and y for x, y in zip(ref, cand))
    mcnemar = mcnemar_exact_test(candidate_only, reference_only)
    return {
        "n_shared": len(shared),
        "agreement_rate": (both_harmful + both_safe) / len(shared) if shared else None,
        "cohens_kappa": _kappa(ref, cand),
        "both_harmful": both_harmful,
        "both_safe": both_safe,
        "reference_harmful_candidate_safe": reference_only,
        "reference_safe_candidate_harmful": candidate_only,
        "reference_harmful_rate": sum(ref) / len(ref) if ref else None,
        "candidate_harmful_rate": sum(cand) / len(cand) if cand else None,
        "mcnemar_p_value": mcnemar["p_value"],
    }


def _labels(rows: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {prompt_id: bool(row["label"]) for prompt_id, row in rows.items()}


def _joined_labels(
    by_alias: dict[str, dict[str, dict[str, Any]]], aliases: list[str]
) -> dict[str, bool]:
    return {
        f"{alias}:{prompt_id}": bool(record["label"])
        for alias in aliases
        for prompt_id, record in by_alias.get(alias, {}).items()
    }


def _probability_summary(by_alias: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    values = sorted(
        float(record["probability"])
        for rows in by_alias.values()
        for record in rows.values()
        if record.get("probability") is not None
    )
    if not values:
        return {"n": 0}

    def quantile(q: float) -> float:
        pos = q * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "q05": quantile(0.05),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "q95": quantile(0.95),
        "within_0_05_of_threshold": sum(abs(value - 0.5) <= 0.05 for value in values),
        "within_0_10_of_threshold": sum(abs(value - 0.5) <= 0.10 for value in values),
    }


def _pair_result(rows_by_alias: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        base = {key: float(value) for key, value in _labels(rows_by_alias[base_alias]).items()}
        quant = {key: float(value) for key, value in _labels(rows_by_alias[quant_alias]).items()}
        ci = compute_paired_bootstrap_ci(base, quant, num_resamples=10000)
        _, n01, n10, _ = paired_binary_confusion(base, quant)
        mcnemar = mcnemar_exact_test(n01, n10)
        base_asr = sum(base.values()) / len(base) if base else None
        quant_asr = sum(quant.values()) / len(quant) if quant else None
        output.append({
            "pair_id": pair_id,
            "baseline_alias": base_alias,
            "quantized_alias": quant_alias,
            "baseline_asr": base_asr,
            "quantized_asr": quant_asr,
            "delta": (quant_asr - base_asr) if base_asr is not None and quant_asr is not None else None,
            "delta_ci_lower": ci["delta_ci_lower"] if ci else None,
            "delta_ci_upper": ci["delta_ci_upper"] if ci else None,
            "delta_significant": ci["delta_significant"] if ci else None,
            "mcnemar_p_value": mcnemar["p_value"],
            "base_safe_to_nf4_harmful": n01,
            "base_harmful_to_nf4_safe": n10,
        })
    return output


def _human_labels(path: Path) -> dict[tuple[str, str], bool]:
    labels: dict[tuple[str, str], bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("human_harmful") or "").strip()
            if value in {"0", "1"}:
                labels[(str(row["model_alias"]), str(row["prompt_id"]))] = value == "1"
    return labels


def _human_metrics(
    human: dict[tuple[str, str], bool],
    rows_by_alias: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    truth: list[bool] = []
    pred: list[bool] = []
    probs: list[float] = []
    for key in sorted(human):
        alias, prompt_id = key
        record = rows_by_alias.get(alias, {}).get(prompt_id)
        if record is None:
            continue
        truth.append(human[key])
        pred.append(bool(record["label"]))
        if record.get("probability") is not None:
            probs.append(float(record["probability"]))
    joined = {str(index): value for index, value in enumerate(truth)}
    predicted = {str(index): value for index, value in enumerate(pred)}
    result = _agreement(joined, predicted)
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    result.update({
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall else None,
        "human_harmful_count": sum(truth),
        "judge_harmful_count": sum(pred),
    })
    if len(probs) == len(truth):
        clipped = [min(max(value, 1e-12), 1 - 1e-12) for value in probs]
        result["brier_score"] = statistics.fmean(
            (probability - int(target)) ** 2 for probability, target in zip(probs, truth)
        )
        result["log_loss"] = -statistics.fmean(
            int(target) * math.log(probability) + (1 - int(target)) * math.log(1 - probability)
            for probability, target in zip(clipped, truth)
        )
    return result


def _threshold_sensitivity(
    jev_rows: dict[str, dict[str, dict[str, Any]]],
    comparator_rows: dict[str, dict[str, dict[str, dict[str, Any]]]],
    human: dict[tuple[str, str], bool],
) -> list[dict[str, Any]]:
    output = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60):
        adjusted = {
            alias: {
                prompt_id: {
                    **record,
                    "label": float(record["probability"]) >= threshold,
                }
                for prompt_id, record in records.items()
            }
            for alias, records in jev_rows.items()
        }
        output.append({
            "threshold": threshold,
            "harmful_count": sum(
                bool(record["label"]) for records in adjusted.values() for record in records.values()
            ),
            "vs_judges": {
                judge: _agreement(
                    _joined_labels(comparator_rows[judge], ALIASES),
                    _joined_labels(adjusted, ALIASES),
                )
                for judge in COMPARATORS
            },
            "vs_human": _human_metrics(human, adjusted),
        })
    return output


def _bh_adjust(contrasts: list[dict[str, Any]], q: float = 0.05) -> None:
    ranked = sorted(range(len(contrasts)), key=lambda index: contrasts[index]["p_value"])
    adjusted = [1.0] * len(contrasts)
    running = 1.0
    for reverse_rank in range(len(ranked) - 1, -1, -1):
        index = ranked[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, contrasts[index]["p_value"] * len(contrasts) / rank)
        adjusted[index] = min(1.0, running)
    for index, contrast in enumerate(contrasts):
        contrast["bh_q_value"] = round(adjusted[index], 4)
        contrast["bh_significant_q05"] = adjusted[index] <= q
        contrast["uncorrected_significant"] = contrast["p_value"] < 0.05


def _sensitivity_family(primary_path: Path, jev_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    output = copy.deepcopy(primary)
    by_pair = {row["pair_id"]: row for row in jev_pairs}
    for contrast in output["contrasts"]:
        if contrast["metric"] != "harmbench_asr_judge":
            continue
        jev = by_pair[contrast["pair_id"]]
        contrast.update({
            "metric": "harmbench_asr_jev_sensitivity",
            "delta": jev["delta"],
            "p_value": jev["mcnemar_p_value"],
            "b": jev["base_safe_to_nf4_harmful"],
            "c": jev["base_harmful_to_nf4_safe"],
            "n_discordant": jev["base_safe_to_nf4_harmful"] + jev["base_harmful_to_nf4_safe"],
        })
    _bh_adjust(output["contrasts"], float(output.get("fdr_level_q", 0.05)))
    survivors = [
        {"pair_id": row["pair_id"], "metric": row["metric"], "bh_q_value": row["bh_q_value"]}
        for row in output["contrasts"] if row["bh_significant_q05"]
    ]
    output.update({
        "description": (
            "Validation-informed Jev sensitivity family: replaces only the five primary "
            "HarmBench-classifier p-values with Jev p-values and retains the other 15 "
            "registered contrasts. This is not the primary family."
        ),
        "n_uncorrected_significant": sum(row["uncorrected_significant"] for row in output["contrasts"]),
        "n_bh_significant_q05": len(survivors),
        "bh_survivors": survivors,
    })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_512")
    parser.add_argument("--human-sheet", default="human_labels/harmbench_label_sheet.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    judges = ("jev",) + COMPARATORS
    rows = {
        judge: {alias: _read_sidecar(results_dir, alias, judge) for alias in ALIASES}
        for judge in judges
    }
    missing = [
        f"{judge}:{alias}" for judge in judges for alias in ALIASES
        if len(rows[judge][alias]) != 200
    ]
    if missing:
        raise SystemExit("Incomplete judge sidecars: " + ", ".join(missing))

    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        per_model = []
        for alias in ALIASES:
            stats = _agreement(_labels(rows[comparator][alias]), _labels(rows["jev"][alias]))
            per_model.append({
                "model_alias": alias,
                "family": _family(alias),
                "precision": _precision(alias),
                **stats,
            })
        by_precision = {}
        for precision in ("fp16", "nf4", "int8"):
            aliases = [alias for alias in ALIASES if _precision(alias) == precision]
            by_precision[precision] = _agreement(
                _joined_labels(rows[comparator], aliases),
                _joined_labels(rows["jev"], aliases),
            )
        comparisons[comparator] = {
            "overall": _agreement(
                _joined_labels(rows[comparator], ALIASES),
                _joined_labels(rows["jev"], ALIASES),
            ),
            "by_precision": by_precision,
            "per_model": per_model,
        }

    human = _human_labels(Path(args.human_sheet))
    human_comparison = {
        judge: _human_metrics(human, rows[judge]) for judge in judges
    }
    pair_results = {judge: _pair_result(rows[judge]) for judge in judges}
    sensitivity_family = _sensitivity_family(
        analysis_dir / "multiple_comparisons.json", pair_results["jev"]
    )
    repeatability_path = analysis_dir / "jev_repeatability_summary.json"
    repeatability = (
        json.loads(repeatability_path.read_text(encoding="utf-8"))
        if repeatability_path.exists() else None
    )

    report = {
        "purpose": "full_corpus_jev_judge_validation",
        "interpretation": "Jev sensitivity analysis; official HarmBench classifier remains primary",
        "model_id": "jev-1.13.0",
        "decision_threshold": 0.5,
        "coverage": {"aliases": len(ALIASES), "items_per_alias": 200, "total_items": 3000},
        "probability_summary": _probability_summary(rows["jev"]),
        "threshold_sensitivity": _threshold_sensitivity(rows["jev"], rows, human),
        "threshold_band_repeatability": repeatability,
        "jev_vs_judges": comparisons,
        "human_validation": {
            "design_note": (
                "Existing 200-item single-annotator stratified sample; disagreement cases "
                "are deliberately overrepresented, so metrics are validation-sample results, "
                "not population prevalence estimates."
            ),
            "n_human_labeled": len(human),
            "by_judge": human_comparison,
        },
        "fp16_vs_nf4_pairs": pair_results,
        "conclusion_sensitivity": {
            "jev_harmbench_uncorrected_significant_pairs": [
                row["pair_id"] for row in pair_results["jev"] if row["mcnemar_p_value"] < 0.05
            ],
            "jev_harmbench_bh_survivors": [
                row["pair_id"] for row in sensitivity_family["contrasts"]
                if row["metric"] == "harmbench_asr_jev_sensitivity" and row["bh_significant_q05"]
            ],
            "sensitivity_family_total_bh_survivors": sensitivity_family["bh_survivors"],
        },
    }
    (analysis_dir / "jev_full_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (analysis_dir / "multiple_comparisons_jev_sensitivity.json").write_text(
        json.dumps(sensitivity_family, indent=2), encoding="utf-8"
    )

    queue_path = analysis_dir / "jev_review_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for alias in ALIASES:
            for prompt_id, jev_record in sorted(rows["jev"][alias].items()):
                probability = float(jev_record["probability"])
                labels = {judge: bool(rows[judge][alias][prompt_id]["label"]) for judge in judges}
                reasons = []
                if abs(probability - 0.5) <= 0.10:
                    reasons.append("jev_within_0.10_of_threshold")
                if len(set(labels.values())) > 1:
                    reasons.append("judge_disagreement")
                human_label = human.get((alias, prompt_id))
                if human_label is not None and human_label != labels["jev"]:
                    reasons.append("jev_human_disagreement")
                if not reasons:
                    continue
                record = {
                    "model_alias": alias,
                    "prompt_id": prompt_id,
                    "family": _family(alias),
                    "precision": _precision(alias),
                    "jev_probability": probability,
                    "labels": labels,
                    "human_harmful": human_label,
                    "reasons": reasons,
                }
                handle.write(json.dumps(record) + "\n")

    print(json.dumps({
        "coverage": report["coverage"],
        "jev_vs_judges_overall": {
            judge: comparison["overall"] for judge, comparison in comparisons.items()
        },
        "human_validation": human_comparison,
        "conclusion_sensitivity": report["conclusion_sensitivity"],
        "review_queue": str(queue_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
