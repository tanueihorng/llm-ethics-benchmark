"""Compare the HarmBench judge against the v2 regex scorer (no GPU required).

Reads the redacted judge sidecars produced by ``run_judge_validation`` and the
v2 score sidecars, and reports:

* per-(model) agreement %, Cohen's kappa, and the 2x2 confusion of
  v2-harmful vs judge-harmful on HarmBench;
* per-(model) HarmBench ASR under v2 vs under the judge;
* per-(pair) HarmBench ΔASR under v2 vs under the judge, and whether the
  pair's interpretation label would change if the judge ASR replaced the
  v2 ASR (holding the v2 over-refusal and MMLU deltas fixed, since the
  HarmBench classifier only judges HarmBench).

All inputs are IDs + booleans; no prompt or response text is read or written.
Outputs ``results/analysis/judge_agreement.json`` and ``...csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.analysis.compare_quant_pairs import (  # noqa: E402
    classify_pair_change,
    compute_paired_bootstrap_ci,
    label_evidence_status,
    mcnemar_exact_test,
    paired_binary_confusion,
)

MODEL_ALIASES = [
    "qwen_2b_base",
    "qwen_2b_4bit",
    "qwen_4b_base",
    "qwen_4b_4bit",
    "llama_3_2_3b_base",
    "llama_3_2_3b_4bit",
]

# pair_id -> (baseline_alias, quantized_alias)
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
}


def _load_jsonl_flag(path: Path, flag_key: str) -> Dict[str, Optional[bool]]:
    """Returns {prompt_id: bool/None} for one boolean field in a JSONL sidecar."""

    out: Dict[str, Optional[bool]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = str(rec.get("prompt_id"))
            val = rec.get(flag_key)
            if isinstance(val, bool):
                out[pid] = val
            elif flag_key == "score_fields":  # v2: nested
                out[pid] = bool((val or {}).get("harmful_compliance", False))
            else:
                out[pid] = None
    return out


def _load_judge_outcomes(model_dir: Path, judge_name: str) -> Dict[str, float]:
    """Reads judge_harmful per prompt as 0/1 from scores.judge.<name>.jsonl."""

    path = model_dir / f"scores.judge.{judge_name}.jsonl"
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            jh = rec.get("judge_harmful")
            if isinstance(jh, bool):
                out[str(rec.get("prompt_id"))] = 1.0 if jh else 0.0
    return out


def _load_v2_harmful(model_dir: Path) -> Dict[str, bool]:
    """Reads v2 harmful_compliance per prompt from scores.v2.jsonl."""

    path = model_dir / "scores.v2.jsonl"
    out: Dict[str, bool] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = str(rec.get("prompt_id"))
            sf = rec.get("score_fields") or {}
            out[pid] = bool(sf.get("harmful_compliance", False))
    return out


def _cohens_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    """Cohen's kappa for two boolean labelings of the same items."""

    n = len(a)
    if n == 0:
        return None
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    denom = 1.0 - pe
    if denom <= 1e-12:
        # Chance agreement is (near-)total: both raters are effectively constant
        # and identical (e.g. both label every prompt not-harmful when ASR≈0).
        # Kappa is 0/0 here — genuinely undefined. Report None rather than a
        # spurious 1.0, which would otherwise read as "perfect agreement" for a
        # case that carries no agreement signal (the exact trap a near-zero-ASR
        # baseline or a second judge would hit).
        return None
    return (po - pe) / denom


def _confusion(v2: Dict[str, bool], judge: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    """2x2 confusion of v2-harmful vs judge-harmful over shared prompt ids."""

    shared = sorted(set(v2) & {pid for pid, val in judge.items() if val is not None})
    a = [v2[pid] for pid in shared]
    b = [bool(judge[pid]) for pid in shared]
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    v2_only = sum(1 for x, y in zip(a, b) if x and not y)
    judge_only = sum(1 for x, y in zip(a, b) if not x and y)
    n = len(shared)
    agreement = (both + neither) / n if n else None
    return {
        "n_shared": n,
        "both_harmful": both,
        "neither_harmful": neither,
        "v2_harmful_judge_not": v2_only,
        "judge_harmful_v2_not": judge_only,
        "agreement_rate": agreement,
        "cohens_kappa": _cohens_kappa(a, b),
        "v2_asr": (sum(a) / n) if n else None,
        "judge_asr": (sum(b) / n) if n else None,
    }


def _read_summary_metric(path: Path, key: str) -> Optional[float]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("metrics") or {}).get(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge vs v2 agreement report.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--judge-name", default="harmbench_cls")
    parser.add_argument("--benchmark", default="harmbench")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Per-model confusion / agreement.
    per_model: Dict[str, Dict[str, Any]] = {}
    judge_asr_by_model: Dict[str, Optional[float]] = {}
    any_judge = False
    for alias in MODEL_ALIASES:
        model_dir = results_dir / alias / args.benchmark
        v2 = _load_v2_harmful(model_dir)
        judge_path = model_dir / f"scores.judge.{args.judge_name}.jsonl"
        judge = _load_jsonl_flag(judge_path, "judge_harmful")
        if judge:
            any_judge = True
        conf = _confusion(v2, judge)
        per_model[alias] = conf
        judge_asr_by_model[alias] = conf["judge_asr"]

    if not any_judge:
        print("No judge sidecars found (scores.judge.{}.jsonl).".format(args.judge_name))
        print("Run scripts/run_judge_validation.py on TC1 first, then re-run this report.")
        return 1

    # Per-pair: compare ΔASR under v2 vs judge and check label stability.
    # v2 OR/MMLU deltas come from the already-computed pairwise_deltas.json.
    pairwise_path = analysis_dir / "pairwise_deltas.json"
    v2_deltas: Dict[Tuple[str, str], float] = {}
    v2_sig: Dict[Tuple[str, str], Optional[bool]] = {}
    if pairwise_path.exists():
        for row in json.loads(pairwise_path.read_text(encoding="utf-8")):
            v2_deltas[(row["pair_id"], row["benchmark"])] = row["absolute_delta"]
            v2_sig[(row["pair_id"], row["benchmark"])] = row.get("delta_significant")

    per_pair: List[Dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        base_judge = judge_asr_by_model.get(base_alias)
        quant_judge = judge_asr_by_model.get(quant_alias)
        judge_delta = (
            quant_judge - base_judge
            if base_judge is not None and quant_judge is not None
            else None
        )
        v2_harm_delta = v2_deltas.get((pair_id, "harmbench"))
        or_delta = v2_deltas.get((pair_id, "xstest"))
        mmlu_delta = v2_deltas.get((pair_id, "mmlu"))

        # Paired bootstrap CI on the judge HarmBench delta (same 2000-resample,
        # seed-42 procedure used for the v2 deltas), so the judge-primary
        # HarmBench finding carries the same statistical apparatus as v2.
        base_outcomes = _load_judge_outcomes(results_dir / base_alias / args.benchmark, args.judge_name)
        quant_outcomes = _load_judge_outcomes(results_dir / quant_alias / args.benchmark, args.judge_name)
        ci = compute_paired_bootstrap_ci(base_outcomes, quant_outcomes)
        judge_ci_lo = ci["delta_ci_lower"] if ci else None
        judge_ci_hi = ci["delta_ci_upper"] if ci else None
        judge_sig = ci["delta_significant"] if ci else None

        # McNemar's exact test: the textbook-correct test for a paired-binary
        # ΔASR. Reports exactly how many prompts flipped each way so the reader
        # sees what the headline effect rests on (effect-size transparency).
        # n01 = base safe / 4-bit harmful (became harmful under quantization);
        # n10 = base harmful / 4-bit safe (became safe).
        _, n01, n10, _ = paired_binary_confusion(base_outcomes, quant_outcomes)
        mcnemar = mcnemar_exact_test(n01, n10)

        mmlu_sig = v2_sig.get((pair_id, "mmlu"))
        or_sig = v2_sig.get((pair_id, "xstest"))

        v2_label = classify_pair_change(v2_harm_delta, or_delta, mmlu_delta)
        judge_label = classify_pair_change(judge_delta, or_delta, mmlu_delta)
        # Two-layer scheme: the label is the direction/threshold reading; the
        # evidence status reports whether the delta(s) it keys on are significant.
        evidence_status = label_evidence_status(
            label=judge_label,
            harm_significant=judge_sig,
            capability_significant=mmlu_sig,
            over_refusal_significant=or_sig,
        )
        per_pair.append({
            "pair_id": pair_id,
            "v2_harm_delta": v2_harm_delta,
            "judge_harm_delta": judge_delta,
            "judge_harm_delta_ci_lower": judge_ci_lo,
            "judge_harm_delta_ci_upper": judge_ci_hi,
            "judge_harm_delta_significant": judge_sig,
            "mcnemar_discordant": mcnemar["discordant"],
            "mcnemar_b_base_to_harmful": mcnemar["b"],
            "mcnemar_c_harmful_to_safe": mcnemar["c"],
            "mcnemar_p_value": mcnemar["p_value"],
            "over_refusal_delta": or_delta,
            "over_refusal_delta_significant": or_sig,
            "mmlu_delta": mmlu_delta,
            "mmlu_delta_significant": mmlu_sig,
            "v2_label": v2_label,
            "judge_label": judge_label,
            "evidence_status": evidence_status,
            "label_changed": v2_label != judge_label,
        })

    report = {
        "judge_name": args.judge_name,
        "benchmark": args.benchmark,
        "per_model": per_model,
        "per_pair": per_pair,
    }
    (analysis_dir / "judge_agreement.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Redacted CSV (per model).
    csv_path = analysis_dir / "judge_agreement.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "model_alias", "n_shared", "agreement_rate", "cohens_kappa",
            "v2_asr", "judge_asr", "both_harmful", "neither_harmful",
            "v2_harmful_judge_not", "judge_harmful_v2_not",
        ])
        for alias in MODEL_ALIASES:
            c = per_model[alias]
            writer.writerow([
                alias, c["n_shared"], c["agreement_rate"], c["cohens_kappa"],
                c["v2_asr"], c["judge_asr"], c["both_harmful"], c["neither_harmful"],
                c["v2_harmful_judge_not"], c["judge_harmful_v2_not"],
            ])

    # Console (redacted).
    print("=" * 84)
    print(f"Judge vs v2 agreement — judge={args.judge_name} benchmark={args.benchmark}")
    print("=" * 84)
    print(f"{'model_alias':<22} {'n':>4} {'agree':>7} {'kappa':>7} {'v2 ASR':>8} {'judge ASR':>10}")
    print("-" * 84)
    for alias in MODEL_ALIASES:
        c = per_model[alias]
        def _f(x: Optional[float]) -> str:
            return f"{x:.3f}" if isinstance(x, float) else "   n/a"
        print(f"{alias:<22} {c['n_shared']:>4} {_f(c['agreement_rate']):>7} "
              f"{_f(c['cohens_kappa']):>7} {_f(c['v2_asr']):>8} {_f(c['judge_asr']):>10}")
    print()
    print("Per-pair judge HarmBench deltas with paired-bootstrap 95% CIs, and label vs v2:")
    print(f"{'pair_id':<14} {'judge Δ':>8} {'95% CI':>20} {'sig?':>5}  {'judge label':<32} {'evidence':<12} {'(v2 label)'}")
    print("-" * 116)
    for p in per_pair:
        def _d(x: Optional[float]) -> str:
            return f"{x:+.3f}" if isinstance(x, float) else "  n/a"
        lo, hi = p.get("judge_harm_delta_ci_lower"), p.get("judge_harm_delta_ci_upper")
        ci_s = f"[{lo:+.3f}, {hi:+.3f}]" if isinstance(lo, float) and isinstance(hi, float) else "n/a"
        sig = p.get("judge_harm_delta_significant")
        print(f"{p['pair_id']:<14} {_d(p['judge_harm_delta']):>8} {ci_s:>20} {str(sig):>5}  "
              f"{p['judge_label']:<32} {p['evidence_status']:<12} ({p['v2_label']})")
    print()
    print("McNemar exact test on the paired HarmBench delta (what the effect rests on):")
    print(f"{'pair_id':<14} {'+harmful':>9} {'-harmful':>9} {'discordant':>11} {'exact p':>9}")
    print("-" * 60)
    for p in per_pair:
        pv = p.get("mcnemar_p_value")
        pv_s = f"{pv:.4f}" if isinstance(pv, float) else "n/a"
        print(f"{p['pair_id']:<14} {p.get('mcnemar_b_base_to_harmful', 0):>9} "
              f"{p.get('mcnemar_c_harmful_to_safe', 0):>9} {p.get('mcnemar_discordant', 0):>11} {pv_s:>9}")
    print()
    print(f"Wrote {analysis_dir / 'judge_agreement.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
