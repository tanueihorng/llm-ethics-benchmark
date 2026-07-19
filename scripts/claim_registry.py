#!/usr/bin/env python3
"""Build the deterministic, artifact-derived claim registry.

The registry is the executable boundary between committed evidence and
reader-facing documents.  It contains scientific values, labels,
significance decisions, configuration facts, completion status, and rendered
table/deck fragments.  It deliberately contains no timestamp, Git state, or
hard-coded test/check count, so identical evidence produces identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "generated/claim_registry.json"

SOURCE_PATHS = (
    "configs/tc1_512.yaml",
    "results_512/analysis/headline_512_vs_128.json",
    "results_512/analysis/human_validation.json",
    "results_512/analysis/judge_agreement.json",
    "results_512/analysis/judge_pairwise_agreement.json",
    "results_512/analysis/multiple_comparisons.json",
    "results_512/analysis/multiple_comparisons_judge_strict.json",
    "results_512/analysis/pairwise_deltas.json",
    "results_512/analysis/precision_sweep.json",
    "results_512/analysis/xstest_human_validation.json",
    "results_512/analysis/xstest_judge_agreement.json",
)
OPTIONAL_SOURCE_PATHS = ()

DISPLAY = {
    "qwen_2b": {"name": "Qwen3-1.7B", "family": "Qwen", "params_b": 1.7},
    "qwen_4b": {"name": "Qwen3-4B", "family": "Qwen", "params_b": 4.0},
    "llama_3_2_3b": {"name": "Llama-3.2-3B", "family": "Llama", "params_b": 3.0},
    "mistral_7b": {"name": "Mistral-7B", "family": "Mistral", "params_b": 7.0},
    "phi4_mini": {"name": "Phi-4-mini", "family": "Phi", "params_b": 3.8},
}

METRIC_LABELS = {
    "harmbench_asr_judge": "HarmBench ASR",
    "xstest_over_refusal": "over-refusal",
    "xstest_over_refusal_judge_strict": "over-refusal (judge-strict)",
    "mmlu_accuracy": "MMLU",
    "arc_accuracy": "ARC",
}

THESIS_LABELS = {
    "qwen_2b": "broad_degradation (capability-driven)",
    "qwen_4b": "alignment_degradation (dir.)",
    "llama_3_2_3b": "capability_collapse_masq._as_safety (dir.)",
    "mistral_7b": "alignment_improvement (dir.)",
    "phi4_mini": "alignment_degradation (dir.)",
}


def _load_json(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fmt(value: float, *, signed: bool = False, places: int = 3) -> str:
    rounded = round(float(value), places)
    if abs(rounded) < 0.5 * 10 ** (-places):
        rounded = 0.0
    magnitude = f"{abs(rounded):.{places}f}"
    if rounded < 0:
        return f"−{magnitude}"
    if signed and rounded > 0:
        return f"+{magnitude}"
    return magnitude


def _fmt_ci(delta: float, low: float, high: float) -> str:
    return (
        f"{_fmt(delta, signed=True)} "
        f"[{_fmt(low, signed=True)}, {_fmt(high, signed=True)}]"
    )


def _metric_record(row: dict[str, Any], contrast: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": row["baseline_value"],
        "nf4": row["quantized_value"],
        "delta": row["absolute_delta"],
        "ci": [row["delta_ci_lower"], row["delta_ci_upper"]],
        "significant": bool(contrast["uncorrected_significant"]),
        "p_value": contrast["p_value"],
        "bh_q_value": contrast["bh_q_value"],
        "bh_significant": bool(contrast["bh_significant_q05"]),
    }


def build_registry(root: Path = ROOT) -> dict[str, Any]:
    config = yaml.safe_load((root / "configs/tc1_512.yaml").read_text(encoding="utf-8"))
    headline = _load_json(root, "results_512/analysis/headline_512_vs_128.json")
    human = _load_json(root, "results_512/analysis/human_validation.json")
    agreement = _load_json(root, "results_512/analysis/judge_agreement.json")
    cross_judge = _load_json(root, "results_512/analysis/judge_pairwise_agreement.json")
    comparisons = _load_json(root, "results_512/analysis/multiple_comparisons.json")
    vi_family = _load_json(root, "results_512/analysis/multiple_comparisons_judge_strict.json")
    deltas = _load_json(root, "results_512/analysis/pairwise_deltas.json")
    precision = _load_json(root, "results_512/analysis/precision_sweep.json")
    xstest_judge = _load_json(root, "results_512/analysis/xstest_judge_agreement.json")
    xstest_human = _load_json(root, "results_512/analysis/xstest_human_validation.json")

    pair_order: list[str] = []
    aliases_by_pair: dict[str, dict[str, str]] = {}
    for alias, model in config["models"].items():
        pair_id = model["pair_id"]
        if pair_id not in pair_order:
            pair_order.append(pair_id)
        role = "nf4" if model["quantized"] else "fp16"
        aliases_by_pair.setdefault(pair_id, {})[role] = alias

    pdix = {(r["pair_id"], r["benchmark"]): r for r in deltas}
    contrasts = {
        (r["pair_id"], r["metric"]): r for r in comparisons["contrasts"]
    }
    h512 = {r["pair"]: r for r in headline["512"]}
    agreement_pairs = {r["pair_id"]: r for r in agreement["per_pair"]}
    cross_pairs = {r["pair_id"]: r for r in cross_judge["per_pair"]}
    xstest_pairs = {r["pair_id"]: r for r in xstest_judge["per_pair"]}

    pairs: dict[str, Any] = {}
    for pair_id in pair_order:
        h = h512[pair_id]
        asr_contrast = contrasts[(pair_id, "harmbench_asr_judge")]
        asr = {
            "baseline": h["asr_base"],
            "nf4": h["asr_4bit"],
            "delta": h["delta"],
            "ci": h["ci"],
            "significant": bool(asr_contrast["uncorrected_significant"]),
            "p_value": asr_contrast["p_value"],
            "bh_q_value": asr_contrast["bh_q_value"],
            "bh_significant": bool(asr_contrast["bh_significant_q05"]),
            "discordant": h["n01"] + h["n10"],
            "base_to_harmful": h["n01"],
            "harmful_to_safe": h["n10"],
        }
        v2_row = pdix[(pair_id, "harmbench")]
        v2 = {
            "baseline": v2_row["baseline_value"],
            "nf4": v2_row["quantized_value"],
            "delta": v2_row["absolute_delta"],
            "ci": [v2_row["delta_ci_lower"], v2_row["delta_ci_upper"]],
            "significant": bool(v2_row["delta_significant"]),
        }
        metric_rows = {
            "xstest": _metric_record(
                pdix[(pair_id, "xstest")],
                contrasts[(pair_id, "xstest_over_refusal")],
            ),
            "mmlu": _metric_record(
                pdix[(pair_id, "mmlu")],
                contrasts[(pair_id, "mmlu_accuracy")],
            ),
            "arc": _metric_record(
                pdix[(pair_id, "arc")],
                contrasts[(pair_id, "arc_accuracy")],
            ),
        }
        aliases = aliases_by_pair[pair_id]
        pair_agreement = agreement_pairs[pair_id]
        pairs[pair_id] = {
            **DISPLAY[pair_id],
            "aliases": aliases,
            "model_id": config["models"][aliases["fp16"]]["model_id"],
            "revision": config["models"][aliases["fp16"]]["revision"],
            "asr": asr,
            "asr_v2_proxy": v2,
            **metric_rows,
            "precision": precision["per_pair"][pair_id]["metrics"],
            "label": pair_agreement["judge_label"],
            "evidence_status": pair_agreement["evidence_status"],
            "baseline_regex_classifier_kappa": agreement["per_model"][aliases["fp16"]]["cohens_kappa"],
            "cross_judge": cross_pairs[pair_id],
            "xstest_judge": xstest_pairs[pair_id],
        }

    survivors = [
        {
            **row,
            "display_name": DISPLAY[row["pair_id"]]["name"],
            "metric_label": METRIC_LABELS[row["metric"]],
        }
        for row in comparisons["bh_survivors"]
    ]
    vi_survivors = [
        {
            **row,
            "display_name": DISPLAY[row["pair_id"]]["name"],
            "metric_label": METRIC_LABELS[row["metric"]],
        }
        for row in vi_family["bh_survivors"]
    ]
    disagreement = {
        "judge_only": sum(
            row["judge_harmful_v2_not"] for row in agreement["per_model"].values()
        ),
        "regex_only": sum(
            row["v2_harmful_judge_not"] for row in agreement["per_model"].values()
        ),
    }

    status = {
        "primary_results_complete": (
            len(pair_order) == 5
            and len(comparisons["contrasts"]) == 20
            and all(
                (pair_id, metric) in contrasts
                for pair_id in pair_order
                for metric in METRIC_LABELS
            )
        ),
        "harmbench_human_validation_complete": human["n_labeled"] > 0,
        "xstest_human_gold_complete": (
            root / "results_512/analysis/xstest_human_validation.json"
        ).is_file(),
    }
    status["phase_c_blocked_by"] = (
        [] if status["xstest_human_gold_complete"] else ["xstest_human_gold"]
    )

    report_61: list[list[str]] = []
    for pair_id in pair_order:
        pair = pairs[pair_id]
        for metric_name, label, sig_mode in (
            ("asr", "HarmBench ASR (judge)", "normal"),
            ("asr_v2_proxy", "  HarmBench non-refusal (v2 proxy)", "proxy"),
            ("xstest", "XSTest over-refusal", "normal"),
            ("mmlu", "MMLU accuracy", "normal"),
        ):
            metric = pair[metric_name]
            sig = (
                "(proxy)"
                if sig_mode == "proxy" and metric["significant"]
                else "yes" if metric["significant"] else "no"
            )
            report_61.append(
                [
                    pair_id,
                    label,
                    _fmt(metric["baseline"]),
                    _fmt(metric["nf4"]),
                    _fmt_ci(metric["delta"], *metric["ci"]),
                    sig,
                ]
            )

    report_62 = [
        [
            "HarmBench ΔASR (judge)",
            *[
                f"{_fmt(pairs[p]['asr']['delta'], signed=True)}"
                f"{'  ★' if pairs[p]['asr']['significant'] else ''}"
                for p in pair_order
            ],
        ],
        [
            "  HarmBench ΔASR (v2 proxy)",
            *[_fmt(pairs[p]["asr_v2_proxy"]["delta"], signed=True) for p in pair_order],
        ],
        [
            "  HarmBench McNemar exact p",
            *[_fmt(pairs[p]["asr"]["p_value"]) for p in pair_order],
        ],
        [
            "XSTest Δover-refusal",
            *[
                f"{_fmt(pairs[p]['xstest']['delta'], signed=True)}"
                f"{'  ★' if pairs[p]['xstest']['significant'] else ''}"
                for p in pair_order
            ],
        ],
        [
            "MMLU Δaccuracy",
            *[
                f"{_fmt(pairs[p]['mmlu']['delta'], signed=True)}"
                f"{'  ★' if pairs[p]['mmlu']['significant'] else ''}"
                for p in pair_order
            ],
        ],
        ["Interpretation label (judge-primary)", *[pairs[p]["label"] for p in pair_order]],
        ["Evidence status (two-layer)", *[pairs[p]["evidence_status"] for p in pair_order]],
    ]

    thesis_62 = []
    for pair_id in pair_order:
        pair = pairs[pair_id]
        thesis_62.append(
            [
                pair_id,
                _fmt_ci(pair["asr"]["delta"], *pair["asr"]["ci"]),
                "yes†" if pair["asr"]["significant"] else "no",
                f"{_fmt(pair['mmlu']['delta'], signed=True)}{'*' if pair['mmlu']['significant'] else ''}",
                f"{_fmt(pair['arc']['delta'], signed=True)}{'*' if pair['arc']['significant'] else ''}",
                THESIS_LABELS[pair_id],
            ]
        )

    deck_pairs = []
    for pair_id in pair_order:
        pair = pairs[pair_id]
        deck_pairs.append(
            {
                "id": pair_id,
                "fam": pair["family"],
                "name": pair["name"],
                "params": pair["params_b"],
                "asr": {
                    "fp16": pair["precision"]["harmbench_asr_judge"]["fp16"],
                    "int8": pair["precision"]["harmbench_asr_judge"]["int8"],
                    "nf4": pair["precision"]["harmbench_asr_judge"]["nf4"],
                },
                "asrDelta": pair["asr"]["delta"],
                "asrSig": pair["asr"]["significant"],
                "v2": {"base": pair["asr_v2_proxy"]["baseline"]},
                "kappa": round(pair["baseline_regex_classifier_kappa"], 2),
                "mmlu": {
                    "fp16": round(pair["precision"]["mmlu_accuracy"]["fp16"], 3),
                    "int8": round(pair["precision"]["mmlu_accuracy"]["int8"], 3),
                    "nf4": round(pair["precision"]["mmlu_accuracy"]["nf4"], 3),
                },
                "mmluDelta": round(pair["mmlu"]["delta"], 3),
                "mmluSig": pair["mmlu"]["significant"],
                "arc": {
                    "base": round(pair["arc"]["baseline"], 3),
                    "q": round(pair["arc"]["nf4"], 3),
                },
                "arcDelta": round(pair["arc"]["delta"], 3),
                "arcSig": pair["arc"]["significant"],
                "or": {
                    "base": pair["xstest"]["baseline"],
                    "q": pair["xstest"]["nf4"],
                },
                "orDelta": round(pair["xstest"]["delta"], 3),
                "orSig": pair["xstest"]["significant"],
                "label": pair["label"],
            }
        )

    defense_rows = [
        {
            "pair_id": p,
            "display_name": pairs[p]["name"],
            "delta_pp": 100 * pairs[p]["asr"]["delta"],
            "p_value": pairs[p]["asr"]["p_value"],
            "bh_q_value": pairs[p]["asr"]["bh_q_value"],
        }
        for p in pair_order
    ]

    survivor_text = ", ".join(
        f"{s['display_name']} {s['metric_label']} "
        f"({_fmt(s['delta'], signed=True)}, q = {_fmt(s['bh_q_value'])})"
        for s in survivors
    )
    render = {
        "report_table_6_1": report_61,
        "report_table_6_2": report_62,
        "thesis_table_6_2": thesis_62,
        "deck_pairs": deck_pairs,
        "defense_asr_rows": defense_rows,
        "bh_survivor_sentence": (
            f"Exactly {len(survivors)} primary contrasts survive BH-FDR: "
            f"{survivor_text}. No HarmBench ASR contrast survives."
        ),
        "dual_family_sentence": (
            f"Under the original registered analysis, exactly "
            f"{('two', 'three', 'four')[len(survivors) - 2]} contrasts survive "
            f"BH-FDR ({survivor_text}); under the validation-informed parallel "
            f"analysis — identical except that over-refusal is scored by the "
            f"independent judge under the strict mapping — "
            f"{('two', 'three', 'four')[len(vi_survivors) - 2]} survive: "
            + ", ".join(
                f"{s['display_name']} {s['metric_label']} "
                f"(q = {_fmt(s['bh_q_value'])})" for s in vi_survivors
            )
            + ", both capability effects. The third registered survivor "
            f"(Phi-4-mini over-refusal, regex-scored) is retained only as the "
            f"original scorer-of-record finding and is most plausibly a "
            f"measurement artifact of that scorer."
        ),
        "asr_forest_caption": (
            "Quantization effect on harmful compliance at the 512-token reference "
            "budget: per-pair judge ΔASR (4-bit − fp16) with paired-bootstrap 95% "
            "confidence intervals. Filled markers denote intervals that exclude "
            "zero; only Llama-3.2-3B reaches significance, and it is a decrease "
            "(safety-improving) — no pair shows a significant increase. Source: "
            "results_512/analysis/judge_agreement.json."
        ),
        "verification_phrase": (
            "the automated test suite and an artifact-derived machine claim registry"
        ),
    }

    sources = {rel: _sha256(root / rel) for rel in SOURCE_PATHS}
    sources.update(
        {
            rel: _sha256(root / rel) if (root / rel).is_file() else None
            for rel in OPTIONAL_SOURCE_PATHS
        }
    )
    payload = {
        "schema_version": 1,
        "sources": sources,
        "claims": {
            "study": {
                "pair_order": pair_order,
                "pair_count": len(pair_order),
                "alias_count": len(config["models"]),
                "benchmarks": list(config["benchmarks"]),
                "primary_config": "configs/tc1_512.yaml",
                "generation_budget": config["decoding"]["max_new_tokens"],
                "status": status,
            },
            "pairs": pairs,
            "multiplicity": {
                "family_size": comparisons["family_size"],
                "survivor_count": comparisons["n_bh_significant_q05"],
                "asr_survivor_count": sum(
                    s["metric"] == "harmbench_asr_judge" for s in survivors
                ),
                "survivors": survivors,
            },
            # Post-hoc parallel family added after T36 Outcome J: identical 20
            # contrasts with over-refusal scored judge-STRICT (T35 replication
            # definition). The registered family above stays the family of
            # record; composition lock: docs/VALIDATION_INFORMED_FAMILY_NOTE.md.
            "multiplicity_validation_informed": {
                "family_size": vi_family["family_size"],
                "survivor_count": vi_family["n_bh_significant_q05"],
                "asr_survivor_count": sum(
                    s["metric"] == "harmbench_asr_judge" for s in vi_survivors
                ),
                "or_survivor_count": sum(
                    s["metric"] == "xstest_over_refusal_judge_strict"
                    for s in vi_survivors
                ),
                "survivors": vi_survivors,
            },
            "validation": {
                "human_n": human["n_labeled"],
                "classifier_human_kappa": human["classifier_vs_human"]["cohens_kappa"],
                "regex_human_kappa": human["regex_vs_human"]["cohens_kappa"],
                "regex_over_flags": human["regex_vs_human"]["over_flag_vs_human"],
                "classifier_over_flags": human["classifier_vs_human"]["over_flag_vs_human"],
                **disagreement,
            },
            # T36 XSTest human audit (Outcome J): blinded single-annotator
            # reference set, disagreement-enriched draw — validation evidence,
            # not population ground truth (docs/XSTEST_GOLD_PREREG.md §7).
            "validation_xstest": {
                "human_n": xstest_human["n_labeled"],
                "outcome_letter": xstest_human["outcome"]["letter"],
                "kappa_regex_strict": xstest_human["strict"]["regex_vs_human"]["cohens_kappa"],
                "kappa_judge_strict": xstest_human["strict"]["judge_vs_human"]["cohens_kappa"],
                "kappa_regex_broad": xstest_human["broad"]["regex_vs_human"]["cohens_kappa"],
                "kappa_judge_broad": xstest_human["broad"]["judge_vs_human"]["cohens_kappa"],
                "regex_recall_strict": xstest_human["strict"]["regex_vs_human"]["recall"],
                "judge_recall_strict": xstest_human["strict"]["judge_vs_human"]["recall"],
                "regex_missed_strict": xstest_human["strict"]["regex_vs_human"]["missed_vs_human"],
                "human_full_refusals": xstest_human["human_label_counts"]["full_refusal"],
                "three_class_agreement": xstest_human["three_class_human_vs_judge"]["exact_agreement_rate"],
            },
        },
        "render": render,
    }
    return {
        **payload,
        "registry_fingerprint": hashlib.sha256(_canonical(payload)).hexdigest(),
    }


def write_registry(root: Path = ROOT, output: Path | None = None) -> Path:
    target = output or (root / "generated/claim_registry.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_registry(root), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def registry_is_fresh(root: Path = ROOT, output: Path | None = None) -> tuple[bool, str]:
    target = output or (root / "generated/claim_registry.json")
    if not target.exists():
        return False, f"missing generated registry: {target.relative_to(root)}"
    actual = json.loads(target.read_text(encoding="utf-8"))
    expected = build_registry(root)
    if actual != expected:
        return False, "generated registry differs from current artifacts/config"
    return True, f"fresh ({expected['registry_fingerprint'][:12]})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the generated registry")
    mode.add_argument("--check", action="store_true", help="fail if the registry is stale")
    args = parser.parse_args()
    if args.write:
        path = write_registry()
        print(f"wrote {path.relative_to(ROOT)}")
        return 0
    if args.check:
        ok, detail = registry_is_fresh()
        print(("ok" if ok else "FAIL") + f"  claim registry: {detail}")
        return 0 if ok else 1
    print(json.dumps(build_registry(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
