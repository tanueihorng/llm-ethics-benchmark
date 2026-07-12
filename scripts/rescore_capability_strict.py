#!/usr/bin/env python3
"""Strict-parser capability sensitivity for MMLU + ARC (T38 / D46).

The primary MMLU/ARC scorer parses the option letter with a four-tier cascade
(``ethical_benchmark.benchmarks.utils.parse_choice_index``): the two intentional,
format-anchored rules (leading option letter; explicit "answer is X"), then two
LENIENT fallbacks (first in-range capital anywhere; bare numeric). Report §6.5 /
Ch8 note that the lenient fallback fires far more often on the degraded 4-bit
outputs (for qwen_2b_4bit 48.7% of MMLU items vs 3.3% at fp16), and that the
fallback scores less accurately, so part of the headline MMLU gap reflects an
answer-format shift rather than only lost knowledge -- with the number stated in
prose but persisted nowhere.

This script builds the first COMMITTED artifact behind that claim. It re-parses
the saved responses with the STRICT parser (tiers 1-2 only;
``parse_choice_index_strict``) and reports, as a redacted derived sidecar, the
strict accuracy, the tier-usage split (strict / lenient-fallback / unanswered),
and the strict-vs-primary capability delta per pair. It is a SENSITIVITY layer:
the committed primary MMLU/ARC accuracies are unchanged; raw.jsonl / summary.json
are never modified (immutability contract, mirroring rescore_harmbench.py).

Pre-locked decision (docs/agent_tasks/T36-T39-hardening-batch.md, WS-C): the
strict parser keeps tiers 1-2 and scores an otherwise-unparsable response as
UNANSWERED -> incorrect (the primary scorer's own parse-failure convention). The
reportable question is fixed before the run: does the qwen_2b ΔMMLU stay negative
and significant under the strict parser, and what is the strict magnitude?

Usage:
    python scripts/rescore_capability_strict.py [--dry-run] [--results-dir results_512]
                                                [--models ...] [--benchmarks mmlu arc]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.analysis.compare_quant_pairs import (  # noqa: E402
    compute_paired_bootstrap_ci,
    mcnemar_exact_test,
    paired_binary_confusion,
)
from ethical_benchmark.benchmarks.utils import (  # noqa: E402
    parse_choice_index_strict_with_tier,
)

SCORER_VERSION = "parser_strict_tiers12_2026-07-12"

MODEL_ALIASES = [
    "qwen_2b_base", "qwen_2b_4bit",
    "qwen_4b_base", "qwen_4b_4bit",
    "llama_3_2_3b_base", "llama_3_2_3b_4bit",
    "mistral_7b_base", "mistral_7b_4bit",
    "phi4_mini_base", "phi4_mini_4bit",
]

PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}

_OPTION_LINE = re.compile(r"(?m)^\(?\s*([A-Z])[.)] ")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def infer_num_choices(prompt_text: str, gold_index: int) -> int:
    """Infers the option count from the prompt's Options block.

    Both MMLU and ARC build ``Options:\\nA. ...\\nB. ...\\nChoose the best answer``,
    so the number of choices is the count of option lines. Falls back to a bound
    derived from the gold index if the block is unexpectedly missing, and is
    clamped to the sane [2, 8] range.
    """
    block = prompt_text
    m = re.search(r"Options:\n(.*?)\nChoose the best answer", prompt_text, re.S)
    if m:
        block = m.group(1)
    letters = _OPTION_LINE.findall(block)
    n = len(letters)
    if not (2 <= n <= 8):
        n = max(gold_index + 1, 4)
    return n


def _strict_rescore_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Re-parses one MMLU/ARC record with the strict parser.

    Returns strict fields plus a four-way ``parse_tier``:
    'leading_letter' (tier 1) and 'answer_declaration' (tier 2) are the strict-
    answered items; 'lenient_fallback' is where only the primary's lenient tiers
    3-4 answered; 'unanswered' is where neither parser answered. The report's
    "leading-letter rule (75%)" is the accuracy within 'leading_letter'; its
    "fallback (46%)" is the accuracy within 'lenient_fallback'.
    """
    response = record.get("response", "") or ""
    sf = record.get("score_fields") or {}
    gold = sf.get("gold_index")
    primary_answered = bool(sf.get("is_answered", False))
    primary_correct = bool(sf.get("is_correct", False))
    primary_pred = sf.get("predicted_index")

    num_choices = infer_num_choices(record.get("prompt_text", "") or "",
                                    gold if isinstance(gold, int) else 0)
    strict_pred, strict_tier = parse_choice_index_strict_with_tier(response, num_choices)
    strict_answered = strict_pred is not None
    strict_correct = (strict_pred == gold) if strict_answered else False

    if strict_answered:
        tier = strict_tier            # "leading_letter" | "answer_declaration"
    elif primary_answered:
        tier = "lenient_fallback"
    else:
        tier = "unanswered"

    # Invariant: the strict parser is a restriction of the full cascade, so when
    # it answers, the full parser answered the same index via tiers 1-2. Recorded
    # (not asserted) so a mismatch surfaces in the aggregate rather than crashing.
    strict_vs_primary_mismatch = bool(
        strict_answered and primary_pred is not None and strict_pred != primary_pred
    )

    return {
        "predicted_index_strict": strict_pred,
        "is_answered_strict": strict_answered,
        "is_correct_strict": strict_correct,
        "primary_answered": primary_answered,
        "primary_correct": primary_correct,
        "parse_tier": tier,
        "num_choices": num_choices,
        "strict_vs_primary_mismatch": strict_vs_primary_mismatch,
    }


def _score_sidecar_row(record: Dict[str, Any], diag: Dict[str, Any]) -> Dict[str, Any]:
    """Redacted per-prompt strict-score row (no prompt/response text)."""
    row = {
        "prompt_id": record.get("prompt_id"),
        "score_fields": {
            "predicted_index_strict": diag["predicted_index_strict"],
            "gold_index": (record.get("score_fields") or {}).get("gold_index"),
            "is_answered_strict": diag["is_answered_strict"],
            "is_correct_strict": diag["is_correct_strict"],
            "parse_tier": diag["parse_tier"],
            "subject": (record.get("score_fields") or {}).get("subject"),
        },
        "scorer_version": SCORER_VERSION,
    }
    for key in ("model_alias", "pair_id", "quantized", "benchmark", "seed"):
        if key in record:
            row[key] = record[key]
    return row


def _rescore_benchmark(model_alias: str, benchmark: str, results_root: Path,
                       dry_run: bool) -> Optional[Dict[str, Any]]:
    """Builds the strict sidecars for one (model, benchmark); returns per-alias stats.

    Also returns the per-prompt strict correctness map (for paired deltas).
    """
    model_dir = results_root / model_alias / benchmark
    raw_path = model_dir / "raw.jsonl"
    if not raw_path.exists():
        return None

    records = _read_jsonl(raw_path)
    sidecar_rows: List[Dict[str, Any]] = []
    strict_correct_by_id: Dict[str, float] = {}
    primary_correct_by_id: Dict[str, float] = {}

    n_total = len(records)
    tier_counts = {"leading_letter": 0, "answer_declaration": 0,
                   "lenient_fallback": 0, "unanswered": 0}
    strict_correct = primary_correct = 0
    lead_correct = fallback_correct = 0
    mismatches = 0

    for rec in records:
        diag = _strict_rescore_record(rec)
        sidecar_rows.append(_score_sidecar_row(rec, diag))
        pid = str(rec.get("prompt_id"))
        strict_correct_by_id[pid] = 1.0 if diag["is_correct_strict"] else 0.0
        primary_correct_by_id[pid] = 1.0 if diag["primary_correct"] else 0.0
        tier_counts[diag["parse_tier"]] += 1
        strict_correct += int(diag["is_correct_strict"])
        primary_correct += int(diag["primary_correct"])
        if diag["parse_tier"] == "leading_letter":
            lead_correct += int(diag["primary_correct"])
        elif diag["parse_tier"] == "lenient_fallback":
            fallback_correct += int(diag["primary_correct"])
        mismatches += int(diag["strict_vs_primary_mismatch"])

    n_strict = tier_counts["leading_letter"] + tier_counts["answer_declaration"]
    n_lead = tier_counts["leading_letter"]
    n_fb = tier_counts["lenient_fallback"]
    summary = {
        "model_alias": model_alias,
        "benchmark": benchmark,
        "scorer_version": SCORER_VERSION,
        "derived_from": "raw.jsonl",
        "num_samples": n_total,
        "strict_accuracy": (strict_correct / n_total) if n_total else None,
        "primary_accuracy": (primary_correct / n_total) if n_total else None,
        "strict_answered_rate": (n_strict / n_total) if n_total else None,
        "tier_usage": {
            "leading_letter": tier_counts["leading_letter"],
            "answer_declaration": tier_counts["answer_declaration"],
            "lenient_fallback": tier_counts["lenient_fallback"],
            "unanswered": tier_counts["unanswered"],
            "fallback_frac": (n_fb / n_total) if n_total else None,
        },
        # Accuracy of the primary scorer within a parse tier — the numbers behind
        # the report's "fallback accuracy 46% vs leading-letter 75%" claim. The
        # leading-letter figure is tier 1 ONLY (the report's phrase); the strict
        # accuracy above additionally counts the lower-accuracy tier-2 items.
        "leading_letter_accuracy": (lead_correct / n_lead) if n_lead else None,
        "fallback_accuracy": (fallback_correct / n_fb) if n_fb else None,
        "strict_vs_primary_mismatch_count": mismatches,
    }

    if not dry_run:
        with (model_dir / "scores.parser_strict.jsonl").open("w", encoding="utf-8") as fh:
            for row in sidecar_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (model_dir / "summary.parser_strict.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "strict_correct_by_id": strict_correct_by_id,
        "primary_correct_by_id": primary_correct_by_id,
    }


def _pair_delta(base: Dict[str, float], quant: Dict[str, float]) -> Dict[str, Any]:
    """ΔACC (4-bit - baseline) with paired bootstrap CI (2000, seed 42) + McNemar."""
    ci = compute_paired_bootstrap_ci(base, quant, num_resamples=2000, seed=42)
    _, n01, n10, _ = paired_binary_confusion(base, quant)
    mcnemar = mcnemar_exact_test(n01, n10)
    delta = ci["delta_mean"] if ci else None
    return {
        "delta": delta,
        "ci_lower": ci["delta_ci_lower"] if ci else None,
        "ci_upper": ci["delta_ci_upper"] if ci else None,
        "significant": ci["delta_significant"] if ci else None,
        "mcnemar_discordant": mcnemar["discordant"],
        "mcnemar_p_value": mcnemar["p_value"],
        "direction": ("down" if (delta is not None and delta < 0)
                      else "up" if (delta is not None and delta > 0) else "flat"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict-parser MMLU/ARC capability sensitivity.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write sidecars/aggregate.")
    parser.add_argument("--results-dir", default="results_512",
                        help="Results root (default: results_512, the primary study).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Aliases to rescore (default: the ten NF4 aliases).")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "arc"],
                        help="Capability benchmarks to rescore (default: mmlu arc).")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix for the analysis-dir aggregate filename.")
    args = parser.parse_args()

    models = args.models or MODEL_ALIASES
    results_root = Path(args.results_dir).resolve()
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # benchmark -> alias -> {summary, strict_correct_by_id, primary_correct_by_id}
    per: Dict[str, Dict[str, Dict[str, Any]]] = {b: {} for b in args.benchmarks}
    for alias in models:
        for bench in args.benchmarks:
            res = _rescore_benchmark(alias, bench, results_root, args.dry_run)
            if res is not None:
                per[bench][alias] = res

    per_alias_rows: List[Dict[str, Any]] = []
    per_pair: List[Dict[str, Any]] = []
    for bench in args.benchmarks:
        for alias in models:
            if alias in per[bench]:
                per_alias_rows.append(per[bench][alias]["summary"])
        for pair_id, (b_alias, q_alias) in PAIRS.items():
            if b_alias in per[bench] and q_alias in per[bench]:
                bs, qs = per[bench][b_alias], per[bench][q_alias]
                per_pair.append({
                    "pair_id": pair_id,
                    "benchmark": bench,
                    "strict_delta": _pair_delta(bs["strict_correct_by_id"], qs["strict_correct_by_id"]),
                    "primary_delta": _pair_delta(bs["primary_correct_by_id"], qs["primary_correct_by_id"]),
                })

    report = {
        "scorer_version": SCORER_VERSION,
        "description": ("Strict-parser (tiers 1-2 only) capability sensitivity for MMLU + ARC. "
                        "The lenient tiers 3-4 are disabled; an unparsable response is scored "
                        "unanswered -> incorrect. Sensitivity layer only: the primary committed "
                        "accuracies are unchanged (T38 / D46)."),
        "results_dir": str(results_root),
        "per_alias": per_alias_rows,
        "per_pair": per_pair,
    }
    agg_json = analysis_dir / f"parser_strict_sensitivity{args.out_suffix}.json"
    csv_path = analysis_dir / f"parser_strict_sensitivity{args.out_suffix}.csv"
    if not args.dry_run:
        agg_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["model_alias", "benchmark", "primary_accuracy", "strict_accuracy",
                             "fallback_frac", "leading_letter_accuracy", "fallback_accuracy",
                             "unanswered"])
            for s in per_alias_rows:
                writer.writerow([s["model_alias"], s["benchmark"], s["primary_accuracy"],
                                 s["strict_accuracy"], s["tier_usage"]["fallback_frac"],
                                 s["leading_letter_accuracy"], s["fallback_accuracy"],
                                 s["tier_usage"]["unanswered"]])

    # Console (redacted).
    print("=" * 92)
    print(f"Strict-parser capability sensitivity — {SCORER_VERSION}")
    print("=" * 92)
    print(f"{'model_alias':<20} {'bench':<6} {'primary':>8} {'strict':>8} "
          f"{'fallback%':>10} {'lead acc':>9} {'fb acc':>8}")
    print("-" * 92)

    def _f(x: Optional[float], pct: bool = False) -> str:
        if not isinstance(x, (int, float)):
            return "   n/a"
        return f"{x*100:.1f}%" if pct else f"{x:.3f}"

    for s in per_alias_rows:
        print(f"{s['model_alias']:<20} {s['benchmark']:<6} {_f(s['primary_accuracy']):>8} "
              f"{_f(s['strict_accuracy']):>8} {_f(s['tier_usage']['fallback_frac'], True):>10} "
              f"{_f(s['leading_letter_accuracy']):>9} {_f(s['fallback_accuracy']):>8}")
    print()
    print("Per-pair ΔACC (4-bit − baseline): strict vs primary, with 95% CI + McNemar")
    print(f"{'pair_id':<14} {'bench':<6} {'strict Δ':>9} {'strict CI':>20} {'sig':>5} "
          f"{'primary Δ':>10}")
    print("-" * 78)
    for p in per_pair:
        sd, pd = p["strict_delta"], p["primary_delta"]
        lo, hi = sd.get("ci_lower"), sd.get("ci_upper")
        ci_s = (f"[{lo:+.3f}, {hi:+.3f}]" if isinstance(lo, float) and isinstance(hi, float) else "n/a")
        print(f"{p['pair_id']:<14} {p['benchmark']:<6} {_f(sd.get('delta')):>9} {ci_s:>20} "
              f"{str(sd.get('significant')):>5} {_f(pd.get('delta')):>10}")
    print()
    if args.dry_run:
        print("DRY RUN — no files written.")
    else:
        print(f"Wrote {agg_json}")
        print(f"Wrote {csv_path}")
        print("Per-prompt strict sidecars: results_512/<model>/{mmlu,arc}/scores.parser_strict.jsonl")
        print("Immutable originals: raw.jsonl and summary.json are not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
