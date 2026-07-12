"""Aggregate the T18 multi-seed sensitivity arm (no GPU, redacted output).

Reads ``results_sensitivity/seed<N>/<alias>/harmbench/`` for each seed and model:
the v2 HarmBench ASR from ``summary.json`` and, when present, the judge ASR from
the ``scores.judge.harmbench_cls.jsonl`` sidecar. For each pair it forms the
per-seed ΔASR (4-bit − baseline) under each scorer, then summarises the spread
across seeds (mean, sd, min, max, sign-consistency) and compares it with the
greedy main-study delta from ``results/analysis/judge_agreement.json``.

The question this answers: is the quantization ΔASR stable across decodes, or did
the single greedy decode land on a lucky/unlucky value? Outputs only aggregate
numbers (means, sds, counts) to ``results/analysis/sensitivity_multiseed.{json,csv}``;
no prompt or response text is read into the outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ethical_benchmark.analysis.compare_quant_pairs import mcnemar_exact_test

# NF4 main-study pairs covered by the multi-seed arm. T39 (D46) extends the arm
# from the original three pairs to all five (adds mistral_7b + phi4_mini @512).
# MUST stay in sync with generate_sensitivity_jobs.MODELS_BY_PAIR — asserted by
# tests/test_sensitivity.py::test_pair_lists_in_sync (a pair missing from either
# list is silently dropped from the arm).
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def summarise_deltas(per_seed_delta: List[float]) -> Dict[str, Any]:
    """Summarise a list of per-seed ΔASR values into a stability report.

    Args:
        per_seed_delta: ΔASR (4-bit − baseline) for each seed, one float per seed.

    Returns:
        Dict with n_seeds, mean, sd (sample stdev; 0.0 for a single seed), min,
        max, and ``sign_consistent`` (True iff every seed has the same sign, i.e.
        the effect direction never flips across seeds). Empty input -> n_seeds 0.

    Side Effects:
        None.
    """

    n = len(per_seed_delta)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "sd": None, "min": None, "max": None,
                "sign_consistent": None}
    mean = statistics.fmean(per_seed_delta)
    sd = statistics.stdev(per_seed_delta) if n > 1 else 0.0
    signs = {(1 if d > 0 else -1 if d < 0 else 0) for d in per_seed_delta}
    nonzero = {s for s in signs if s != 0}
    sign_consistent = len(nonzero) <= 1
    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "min": min(per_seed_delta),
        "max": max(per_seed_delta),
        "sign_consistent": sign_consistent,
    }


def _v2_asr(model_dir: Path) -> Optional[float]:
    # Prefer the v2 rescore sidecar over the runtime summary.json: for the
    # original 2026-05-27 models summary.json carries the *v1* regex, whereas the
    # v2 (current) regex lives in summary.v2.json. Reading summary.json here would
    # silently pick up the stale v1 value on a re-run (mirrors the fix in
    # precision_sweep_analysis.py). Fall back to summary.json only when no v2
    # sidecar exists (e.g. a model rescored at v2 at runtime).
    summary = model_dir / "summary.v2.json"
    if not summary.exists():
        summary = model_dir / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    return (data.get("metrics") or {}).get("attack_success_rate")


def _judge_records(model_dir: Path, judge_name: str) -> Optional[Dict[str, bool]]:
    """Per-prompt judge verdicts keyed by prompt_id (redacted sidecar only)."""
    path = model_dir / f"scores.judge.{judge_name}.jsonl"
    if not path.exists():
        return None
    out: Dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            jh = rec.get("judge_harmful")
            pid = rec.get("prompt_id")
            if isinstance(jh, bool) and pid is not None:
                out[str(pid)] = jh
    return out or None


def _judge_asr(model_dir: Path, judge_name: str) -> Optional[float]:
    recs = _judge_records(model_dir, judge_name)
    if not recs:
        return None
    return sum(recs.values()) / len(recs)


def per_seed_mcnemar(base: Dict[str, bool], quant: Dict[str, bool]) -> Dict[str, Any]:
    """Exact paired McNemar for one seed's judge verdicts (matched prompt_ids).

    b = baseline-harmful / 4-bit-safe, c = baseline-safe / 4-bit-harmful, so a
    positive delta (c > b) is a harm increase. Reuses the canonical
    ``mcnemar_exact_test`` so per-seed p-values are computed identically to the
    primary contrasts in ``multiple_comparisons.json``.
    """
    ids = sorted(set(base) & set(quant))
    b = sum(1 for i in ids if base[i] and not quant[i])
    c = sum(1 for i in ids if quant[i] and not base[i])
    mc = mcnemar_exact_test(b, c)
    return {
        "n_matched": len(ids),
        "b": b,
        "c": c,
        "p_value": mc["p_value"],
        "significant_p05": mc["p_value"] < 0.05,
    }


def _greedy_judge_deltas(analysis_dir: Path) -> Dict[str, float]:
    path = analysis_dir / "judge_agreement.json"
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("per_pair", []):
        if row.get("judge_harm_delta") is not None:
            out[row["pair_id"]] = row["judge_harm_delta"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed sensitivity aggregation.")
    parser.add_argument("--sensitivity-root", default="results_sensitivity")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--judge-name", default="harmbench_cls")
    args = parser.parse_args()

    sens_root = Path(args.sensitivity_root)
    analysis_dir = Path(args.results_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(p for p in sens_root.glob("seed*") if p.is_dir()) if sens_root.exists() else []
    if not seed_dirs:
        print(f"No sensitivity results under {sens_root}/seed*. Run the T18 sbatch on TC1 first.")
        return 1

    greedy = _greedy_judge_deltas(analysis_dir)
    per_pair: List[Dict[str, Any]] = []

    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        v2_deltas: List[float] = []
        judge_deltas: List[float] = []
        judge_per_seed: List[Dict[str, Any]] = []
        seeds_used: List[str] = []
        for seed_dir in seed_dirs:
            base_dir = seed_dir / base_alias / "harmbench"
            quant_dir = seed_dir / quant_alias / "harmbench"
            seed_used = False
            b_v2, q_v2 = _v2_asr(base_dir), _v2_asr(quant_dir)
            if b_v2 is not None and q_v2 is not None:
                v2_deltas.append(q_v2 - b_v2)
                seed_used = True
            b_recs = _judge_records(base_dir, args.judge_name)
            q_recs = _judge_records(quant_dir, args.judge_name)
            if b_recs and q_recs:
                b_j = sum(b_recs.values()) / len(b_recs)
                q_j = sum(q_recs.values()) / len(q_recs)
                judge_deltas.append(q_j - b_j)
                # Per-seed exact McNemar so the report's per-seed delta list and
                # "k/n seeds individually significant" claims are asserted
                # against this committed artifact (audit 2026-07-08: these were
                # previously stated in the report but persisted nowhere).
                mc = per_seed_mcnemar(b_recs, q_recs)
                mc["seed"] = seed_dir.name
                mc["delta"] = q_j - b_j
                judge_per_seed.append(mc)
                seed_used = True
            # T24 fix: a seed counts as used if EITHER scorer produced a delta.
            # Previously seeds_used was appended only in the v2 branch, so a seed
            # with a judge sidecar but no/partial summary.json under-counted.
            if seed_used:
                seeds_used.append(seed_dir.name)

        v2_summary = summarise_deltas(v2_deltas)
        judge_summary = summarise_deltas(judge_deltas)
        if judge_per_seed:
            judge_summary["per_seed"] = judge_per_seed
            judge_summary["n_seeds_significant_p05"] = sum(
                1 for s in judge_per_seed if s["significant_p05"]
            )
        greedy_judge = greedy.get(pair_id)
        # Does the greedy headline sit inside the observed multi-seed range?
        # A small absolute tolerance guards against float artifacts: llama's
        # greedy delta is -0.04000000000000001 vs a seed minimum of
        # -0.039999999999999994 - identical to 3 dp, yet a strict <= called
        # it out-of-range (audit P3).
        greedy_in_range = None
        if greedy_judge is not None and judge_summary["n_seeds"] > 0:
            tol = 1e-9
            greedy_in_range = (
                (judge_summary["min"] - tol) <= greedy_judge <= (judge_summary["max"] + tol)
            )

        per_pair.append({
            "pair_id": pair_id,
            "seeds_used": seeds_used,
            "v2_delta": v2_summary,
            "judge_delta": judge_summary,
            "greedy_judge_delta": greedy_judge,
            "greedy_in_multiseed_range": greedy_in_range,
        })

    report = {"sensitivity_root": str(sens_root), "per_pair": per_pair}
    (analysis_dir / "sensitivity_multiseed.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    csv_path = analysis_dir / "sensitivity_multiseed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "pair_id", "scorer", "n_seeds", "mean_delta", "sd_delta",
            "min_delta", "max_delta", "sign_consistent", "greedy_judge_delta",
            "greedy_in_range",
        ])
        for p in per_pair:
            for scorer in ("v2_delta", "judge_delta"):
                s = p[scorer]
                writer.writerow([
                    p["pair_id"], scorer.replace("_delta", ""), s["n_seeds"],
                    s["mean"], s["sd"], s["min"], s["max"], s["sign_consistent"],
                    p["greedy_judge_delta"], p["greedy_in_multiseed_range"],
                ])

    # Console (redacted).
    print("=" * 92)
    print("T18 multi-seed sensitivity — HarmBench ΔASR stability across decodes")
    print("=" * 92)
    print(f"{'pair_id':<14} {'scorer':<7} {'seeds':>5} {'mean Δ':>9} {'sd':>7} "
          f"{'[min, max]':>18} {'sign?':>6} {'greedy':>8}")
    print("-" * 92)

    def _f(x: Optional[float]) -> str:
        return f"{x:+.3f}" if isinstance(x, (int, float)) else "  n/a"

    for p in per_pair:
        for scorer in ("v2_delta", "judge_delta"):
            s = p[scorer]
            rng = (f"[{s['min']:+.3f}, {s['max']:+.3f}]"
                   if s["n_seeds"] else "n/a")
            print(f"{p['pair_id']:<14} {scorer.replace('_delta',''):<7} {s['n_seeds']:>5} "
                  f"{_f(s['mean']):>9} {(_f(s['sd']) if s['sd'] is not None else ' n/a'):>7} "
                  f"{rng:>18} {str(s['sign_consistent']):>6} {_f(p['greedy_judge_delta']):>8}")
    print()
    print(f"Wrote {analysis_dir / 'sensitivity_multiseed.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
