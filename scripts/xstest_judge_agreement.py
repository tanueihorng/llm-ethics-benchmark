"""Compare the XSTest refusal judge against the v2 regex scorer (no GPU required).

Reads the redacted XSTest judge sidecars produced by ``run_xstest_judge`` and
the v2 XSTest score sidecars, and reports (T35 / D45):

* per-(model) agreement %, Cohen's kappa, and the 2x2 confusion of the judge's
  over-refusal call vs the v2 regex ``is_refusal`` — under BOTH the strict
  (full refusal only) and broad (partial or full refusal) mappings;
* per-(model) XSTest over-refusal rate under the regex vs the judge (strict/broad);
* per-(pair) over-refusal ΔOR (NF4 - fp16) under the judge strict/broad mappings,
  with paired-bootstrap 95% CIs (10,000 resamples, seed 42) + McNemar exact test.

The deterministic v2 regex remains the study's PRIMARY over-refusal scorer; this
is a sensitivity check (the BH-FDR family is untouched). All inputs are IDs +
booleans; no prompt or response text is read or written. Outputs
``<results_dir>/analysis/xstest_judge_agreement.json`` and ``...csv``.
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
    "qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit",
    "qwen_4b_base", "qwen_4b_4bit", "qwen_4b_8bit",
    "llama_3_2_3b_base", "llama_3_2_3b_4bit", "llama_3_2_3b_8bit",
    "mistral_7b_base", "mistral_7b_4bit", "mistral_7b_8bit",
    "phi4_mini_base", "phi4_mini_4bit", "phi4_mini_8bit",
]

# pair_id -> (baseline_alias, quantized_alias). NF4 main-study pairs only; the
# INT8 (_8bit) aliases are reported per-model, never paired (D45 pre-reg).
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def _cohens_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    """Cohen's kappa for two boolean labelings of the same items.

    Returns None on the degenerate all-constant case (pe == 1), matching
    judge_agreement._cohens_kappa — a spurious 1.0 there would read as "perfect
    agreement" for a slice carrying no agreement signal.
    """

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
        return None
    return (po - pe) / denom


def _load_v2_refusal(model_dir: Path) -> Dict[str, bool]:
    """Reads v2 is_refusal per prompt from xstest scores.v2.jsonl."""

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
            out[pid] = bool(sf.get("is_refusal", False))
    return out


def _load_judge_refusal(model_dir: Path, judge_name: str, field: str) -> Dict[str, bool]:
    """Reads a judge refusal boolean per prompt from scores.judge.<name>.jsonl.

    ``field`` is ``judge_refusal_strict`` or ``judge_refusal_broad``. Records
    whose field is null (parse error / skipped) are omitted.
    """

    path = model_dir / f"scores.judge.{judge_name}.jsonl"
    out: Dict[str, bool] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            val = rec.get(field)
            if isinstance(val, bool):
                out[str(rec.get("prompt_id"))] = val
    return out


def _confusion(regex: Dict[str, bool], judge: Dict[str, bool]) -> Dict[str, Any]:
    """2x2 confusion of regex is_refusal vs a judge refusal call over shared ids."""

    shared = sorted(set(regex) & set(judge))
    a = [regex[pid] for pid in shared]
    b = [judge[pid] for pid in shared]
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    regex_only = sum(1 for x, y in zip(a, b) if x and not y)
    judge_only = sum(1 for x, y in zip(a, b) if not x and y)
    n = len(shared)
    return {
        "n_shared": n,
        "both_refusal": both,
        "neither_refusal": neither,
        "regex_refusal_judge_not": regex_only,
        "judge_refusal_regex_not": judge_only,
        "agreement_rate": (both + neither) / n if n else None,
        "cohens_kappa": _cohens_kappa(a, b),
        "regex_or": (sum(a) / n) if n else None,
        "judge_or": (sum(b) / n) if n else None,
    }


def _bool_to_outcomes(d: Dict[str, bool]) -> Dict[str, float]:
    return {pid: (1.0 if v else 0.0) for pid, v in d.items()}


def _pair_delta(base: Dict[str, bool], quant: Dict[str, bool]) -> Dict[str, Any]:
    """Over-refusal ΔOR (NF4 - fp16) with bootstrap CI + McNemar on one mapping."""

    base_o = _bool_to_outcomes(base)
    quant_o = _bool_to_outcomes(quant)
    ci = compute_paired_bootstrap_ci(base_o, quant_o, num_resamples=10000)
    _, n01, n10, _ = paired_binary_confusion(base_o, quant_o)
    mcnemar = mcnemar_exact_test(n01, n10)
    delta = ci["delta_mean"] if ci else None
    return {
        "delta": delta,
        "ci_lower": ci["delta_ci_lower"] if ci else None,
        "ci_upper": ci["delta_ci_upper"] if ci else None,
        "significant": ci["delta_significant"] if ci else None,
        "mcnemar_discordant": mcnemar["discordant"],
        "mcnemar_b_to_refusal": mcnemar["b"],
        "mcnemar_c_to_compliance": mcnemar["c"],
        "mcnemar_p_value": mcnemar["p_value"],
        "direction": ("down" if (delta is not None and delta < 0)
                      else "up" if (delta is not None and delta > 0) else "flat"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="XSTest refusal judge vs v2 regex agreement (T35).")
    parser.add_argument("--results-dir", default="results_512")
    parser.add_argument("--judge-name", default="xstest_api")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Aliases to report (default: the fifteen XSTest aliases). The INT8 "
                             "(_8bit) members appear per-model only; the per-pair section pairs "
                             "base vs 4bit.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix for xstest_judge_agreement{suffix}.{json,csv}.")
    args = parser.parse_args()

    models = args.models or MODEL_ALIASES
    pairs = {pid: (b, q) for pid, (b, q) in PAIRS.items() if b in models and q in models}
    results_dir = Path(args.results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Per-model confusion / agreement under both mappings.
    per_model: Dict[str, Dict[str, Any]] = {}
    any_judge = False
    for alias in models:
        model_dir = results_dir / alias / "xstest"
        regex = _load_v2_refusal(model_dir)
        strict = _load_judge_refusal(model_dir, args.judge_name, "judge_refusal_strict")
        broad = _load_judge_refusal(model_dir, args.judge_name, "judge_refusal_broad")
        if strict or broad:
            any_judge = True
        per_model[alias] = {
            "strict_vs_regex": _confusion(regex, strict),
            "broad_vs_regex": _confusion(regex, broad),
        }

    if not any_judge:
        print("No XSTest judge sidecars found (scores.judge.{}.jsonl).".format(args.judge_name))
        print("Run scripts/run_xstest_judge.py first, then re-run this report.")
        return 1

    # Per-pair ΔOR under the judge strict/broad mappings.
    per_pair: List[Dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in pairs.items():
        base_dir = results_dir / base_alias / "xstest"
        quant_dir = results_dir / quant_alias / "xstest"
        base_strict = _load_judge_refusal(base_dir, args.judge_name, "judge_refusal_strict")
        quant_strict = _load_judge_refusal(quant_dir, args.judge_name, "judge_refusal_strict")
        base_broad = _load_judge_refusal(base_dir, args.judge_name, "judge_refusal_broad")
        quant_broad = _load_judge_refusal(quant_dir, args.judge_name, "judge_refusal_broad")
        per_pair.append({
            "pair_id": pair_id,
            "judge_strict": _pair_delta(base_strict, quant_strict),
            "judge_broad": _pair_delta(base_broad, quant_broad),
        })

    report = {
        "judge_name": args.judge_name,
        "benchmark": "xstest",
        "description": ("XSTest over-refusal: independent API refusal judge (3-class) vs the v2 "
                        "regex. Sensitivity check only; the regex remains the primary scorer and "
                        "the BH-FDR family is untouched (T35 / D45). STRICT = full_refusal only; "
                        "BROAD = partial_refusal or full_refusal."),
        "per_model": per_model,
        "per_pair": per_pair,
    }
    (analysis_dir / f"xstest_judge_agreement{args.out_suffix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Redacted CSV (per model, both mappings).
    csv_path = analysis_dir / f"xstest_judge_agreement{args.out_suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "model_alias", "mapping", "n_shared", "agreement_rate", "cohens_kappa",
            "regex_or", "judge_or", "both_refusal", "neither_refusal",
            "regex_refusal_judge_not", "judge_refusal_regex_not",
        ])
        for alias in models:
            for mapping, key in (("strict", "strict_vs_regex"), ("broad", "broad_vs_regex")):
                c = per_model[alias][key]
                writer.writerow([
                    alias, mapping, c["n_shared"], c["agreement_rate"], c["cohens_kappa"],
                    c["regex_or"], c["judge_or"], c["both_refusal"], c["neither_refusal"],
                    c["regex_refusal_judge_not"], c["judge_refusal_regex_not"],
                ])

    # Console (redacted).
    def _f(x: Optional[float]) -> str:
        return f"{x:.3f}" if isinstance(x, float) else "   n/a"

    print("=" * 96)
    print(f"XSTest refusal judge vs v2 regex — judge={args.judge_name}")
    print("=" * 96)
    print(f"{'model_alias':<22} {'map':<7} {'n':>4} {'agree':>7} {'kappa':>7} {'regex OR':>9} {'judge OR':>9}")
    print("-" * 96)
    for alias in models:
        for mapping, key in (("strict", "strict_vs_regex"), ("broad", "broad_vs_regex")):
            c = per_model[alias][key]
            print(f"{alias:<22} {mapping:<7} {c['n_shared']:>4} {_f(c['agreement_rate']):>7} "
                  f"{_f(c['cohens_kappa']):>7} {_f(c['regex_or']):>9} {_f(c['judge_or']):>9}")
    print()
    print("Per-pair over-refusal ΔOR (NF4 - fp16) under the judge, with 95% CIs + McNemar:")
    print(f"{'pair_id':<14} {'map':<7} {'ΔOR':>8} {'95% CI':>20} {'sig?':>5} {'dir':>5} {'exact p':>9}")
    print("-" * 84)
    for p in per_pair:
        for mapping, key in (("strict", "judge_strict"), ("broad", "judge_broad")):
            d = p[key]
            lo, hi = d.get("ci_lower"), d.get("ci_upper")
            ci_s = f"[{lo:+.3f}, {hi:+.3f}]" if isinstance(lo, float) and isinstance(hi, float) else "n/a"
            dv = d.get("delta")
            dv_s = f"{dv:+.3f}" if isinstance(dv, float) else "  n/a"
            pv = d.get("mcnemar_p_value")
            pv_s = f"{pv:.4f}" if isinstance(pv, float) else "n/a"
            print(f"{p['pair_id']:<14} {mapping:<7} {dv_s:>8} {ci_s:>20} "
                  f"{str(d.get('significant')):>5} {d.get('direction',''):>5} {pv_s:>9}")
    print()
    print(f"Wrote {analysis_dir / ('xstest_judge_agreement' + args.out_suffix + '.json')}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
