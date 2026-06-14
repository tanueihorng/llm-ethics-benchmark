"""Judge-vs-judge agreement on HarmBench (T22, no GPU, redacted output).

Cross-checks the primary HarmBench scorer against a second, independent judge to
test whether the ASR finding is robust to the choice of judge (Priority-1
weakness W3; report §7.4). By default it compares ``harmbench_cls`` (the primary
classifier, D16) with ``api_judge`` (the second judge, run via
``run_judge_validation.py --backend api_judge``).

Two views, both from the redacted per-prompt judge sidecars
(``scores.judge.<name>.jsonl``: ``prompt_id`` + ``judge_harmful`` only — no
prompt or response text):

  * Per model: agreement rate + Cohen's kappa between the two judges over the
    shared prompt set, and each judge's ASR.
  * Per pair: the second judge's ΔASR (4-bit − baseline) with McNemar's exact
    test and paired-bootstrap significance, shown beside the primary judge's,
    plus whether the two judges agree on the sign and on significance of ΔASR.

Writes ``results/analysis/judge_pairwise_agreement.{json,csv}``. Reuses the same
McNemar / bootstrap apparatus as the primary analysis so the second judge's ΔASR
carries identical statistics.
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
    compute_paired_bootstrap_ci,
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
    "mistral_7b_base",
    "mistral_7b_4bit",
    "phi4_mini_base",
    "phi4_mini_4bit",
]

PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def load_judge_outcomes(model_dir: Path, judge_name: str) -> Dict[str, float]:
    """Returns ``{prompt_id: 0.0/1.0}`` from a redacted judge sidecar.

    Records whose ``judge_harmful`` is null (parse error / API error) are
    skipped, so the shared-prompt intersection only contains items both judges
    actually labelled.
    """

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
            pid = rec.get("prompt_id")
            harmful = rec.get("judge_harmful")
            if pid is None or harmful is None:
                continue
            out[str(pid)] = 1.0 if harmful else 0.0
    return out


def agreement_and_kappa(
    a: Dict[str, float], b: Dict[str, float]
) -> Tuple[int, Optional[float], Optional[float]]:
    """Returns ``(n_shared, agreement_rate, cohens_kappa)`` for two labelings.

    Kappa is ``None`` when undefined (no shared items, or expected agreement is
    1.0 — the degenerate all-same-label case), matching the convention in
    ``compare_quant_pairs`` / ``judge_agreement``.
    """

    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n == 0:
        return 0, None, None
    xa = [a[k] for k in shared]
    xb = [b[k] for k in shared]
    po = sum(1 for i in range(n) if xa[i] == xb[i]) / n
    pa1 = sum(xa) / n
    pb1 = sum(xb) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = None if (1 - pe) == 0 else (po - pe) / (1 - pe)
    return n, po, kappa


def _asr(d: Dict[str, float]) -> Optional[float]:
    return sum(d.values()) / len(d) if d else None


def _sign(x: Optional[float]) -> Optional[int]:
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _pair_delta(base: Dict[str, float], quant: Dict[str, float]) -> Dict[str, Any]:
    """ΔASR (quant − base) with McNemar + bootstrap significance for one judge."""

    base_asr = _asr(base)
    quant_asr = _asr(quant)
    delta = (
        quant_asr - base_asr
        if base_asr is not None and quant_asr is not None
        else None
    )
    _, n01, n10, _ = paired_binary_confusion(base, quant)
    mcnemar = mcnemar_exact_test(n01, n10)
    ci = compute_paired_bootstrap_ci(base, quant)
    return {
        "baseline_asr": base_asr,
        "quantized_asr": quant_asr,
        "delta": delta,
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_b_base_to_harmful": mcnemar["b"],
        "mcnemar_c_harmful_to_safe": mcnemar["c"],
        "delta_ci_lower": ci["delta_ci_lower"] if ci else None,
        "delta_ci_upper": ci["delta_ci_upper"] if ci else None,
        "delta_significant": ci["delta_significant"] if ci else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge-vs-judge agreement on HarmBench (redacted)."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--judge-a", default="harmbench_cls",
                        help="Primary judge sidecar name (default: harmbench_cls).")
    parser.add_argument("--judge-b", default="api_judge",
                        help="Second judge sidecar name (default: api_judge).")
    parser.add_argument("--benchmark", default="harmbench")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    a_name, b_name = args.judge_a, args.judge_b

    # Per-model agreement between the two judges.
    per_model: List[Dict[str, Any]] = []
    found_b = False
    for alias in MODEL_ALIASES:
        mdir = results_dir / alias / args.benchmark
        a = load_judge_outcomes(mdir, a_name)
        b = load_judge_outcomes(mdir, b_name)
        if b:
            found_b = True
        n, agree, kappa = agreement_and_kappa(a, b)
        per_model.append({
            "model_alias": alias,
            "n_shared": n,
            "agreement_rate": agree,
            "cohens_kappa": kappa,
            f"{a_name}_asr": _asr(a),
            f"{b_name}_asr": _asr(b),
        })

    if not found_b:
        print(f"No '{b_name}' judge sidecars found (scores.judge.{b_name}.jsonl).")
        print(f"Run: python scripts/run_judge_validation.py --backend {b_name} ...")
        return 1

    # Per-pair ΔASR under each judge + cross-judge concordance.
    per_pair: List[Dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        base_dir = results_dir / base_alias / args.benchmark
        quant_dir = results_dir / quant_alias / args.benchmark
        a_delta = _pair_delta(load_judge_outcomes(base_dir, a_name),
                              load_judge_outcomes(quant_dir, a_name))
        b_delta = _pair_delta(load_judge_outcomes(base_dir, b_name),
                              load_judge_outcomes(quant_dir, b_name))
        sign_a, sign_b = _sign(a_delta["delta"]), _sign(b_delta["delta"])
        per_pair.append({
            "pair_id": pair_id,
            f"{a_name}_delta": a_delta["delta"],
            f"{a_name}_significant": a_delta["delta_significant"],
            f"{a_name}_mcnemar_p": a_delta["mcnemar_p_value"],
            f"{b_name}_delta": b_delta["delta"],
            f"{b_name}_significant": b_delta["delta_significant"],
            f"{b_name}_mcnemar_p": b_delta["mcnemar_p_value"],
            "sign_agree": (sign_a is not None and sign_a == sign_b),
            "significance_agree": (a_delta["delta_significant"] == b_delta["delta_significant"]),
        })

    report = {
        "judge_a": a_name,
        "judge_b": b_name,
        "benchmark": args.benchmark,
        "per_model": per_model,
        "per_pair": per_pair,
    }
    (analysis_dir / "judge_pairwise_agreement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    csv_path = analysis_dir / "judge_pairwise_agreement.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "pair_id",
            f"{a_name}_delta", f"{a_name}_significant",
            f"{b_name}_delta", f"{b_name}_significant",
            "sign_agree", "significance_agree",
        ])
        for r in per_pair:
            writer.writerow([
                r["pair_id"],
                r[f"{a_name}_delta"], r[f"{a_name}_significant"],
                r[f"{b_name}_delta"], r[f"{b_name}_significant"],
                r["sign_agree"], r["significance_agree"],
            ])

    # Console (redacted).
    def _f(x: Optional[float]) -> str:
        return f"{x:.3f}" if isinstance(x, float) else "  n/a"

    print("=" * 78)
    print(f"Judge-vs-judge agreement on {args.benchmark}: {a_name} vs {b_name}")
    print("=" * 78)
    print(f"{'model':<22} {'n':>4} {'agree':>7} {'kappa':>7} {a_name[:10]:>11} {b_name[:10]:>11}")
    for m in per_model:
        print(f"{m['model_alias']:<22} {m['n_shared']:>4} {_f(m['agreement_rate']):>7} "
              f"{_f(m['cohens_kappa']):>7} {_f(m[f'{a_name}_asr']):>11} {_f(m[f'{b_name}_asr']):>11}")
    print("\nPer-pair ΔASR (4-bit − baseline) under each judge:")
    print(f"  {'pair':<14} {a_name[:12]:>13} {'sig':>5} {b_name[:12]:>13} {'sig':>5} {'sign✓':>6} {'sig✓':>5}")
    for r in per_pair:
        da = f"{r[f'{a_name}_delta']:+.3f}" if isinstance(r[f'{a_name}_delta'], float) else " n/a"
        db = f"{r[f'{b_name}_delta']:+.3f}" if isinstance(r[f'{b_name}_delta'], float) else " n/a"
        print(f"  {r['pair_id']:<14} {da:>13} {str(r[f'{a_name}_significant']):>5} "
              f"{db:>13} {str(r[f'{b_name}_significant']):>5} "
              f"{str(r['sign_agree']):>6} {str(r['significance_agree']):>5}")
    print()
    print(f"Wrote {analysis_dir / 'judge_pairwise_agreement.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
