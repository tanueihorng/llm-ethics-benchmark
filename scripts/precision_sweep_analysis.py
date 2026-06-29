"""Precision-sweep analysis (T29) — fp16 -> INT8 -> NF4 across the five pairs.

Mac-side, no GPU. For each pair it reads the safety and capability summaries at
three precisions — fp16 (``<pair>_base``), INT8 (``<pair>_8bit``) and NF4
(``<pair>_4bit``) — and reports the trend: the metric at each precision, the
deltas vs fp16, and whether INT8 sits monotonically between fp16 and NF4 (the
"intermediate precision" expectation) or the degradation is concentrated at one
step (a "cliff"). HarmBench ASR uses the PRIMARY judge scorer (D16); the v2
proxy is carried for transparency. INT8 entries that have not been run yet read
as ``None`` and are reported as pending rather than crashing.

It also emits, per pair, the **fp16-vs-INT8 HarmBench ASR McNemar exact test**
under BOTH judges (the official HarmBench classifier and the GPT-4o api_judge):
the discordant counts ``b`` / ``c``, ``n_discordant``, and the exact two-sided
binomial ``p_value``, computed from the committed redacted judge sidecars
(``scores.judge.<backend>.jsonl``). This mirrors the
``{b, c, n_discordant, p_value}`` contrast schema in
``results/analysis/multiple_comparisons.json`` (which covers the NF4-vs-fp16
family) so the §6.15 INT8 finding — the only judge-robust INT8 safety move,
Llama-3B ΔASR=+0.040 — can be cited from a committed artifact rather than a
hand-derived footnote. Reuses the SAME McNemar primitives as the main study
(``ethical_benchmark.analysis.compare_quant_pairs``); no number is recomputed by
a parallel path.

Writes ``results/analysis/precision_sweep.{json,csv}`` (aggregate only).
Run after the INT8 matrix + judge jobs (configs/tc1_int8.yaml) land and are
SCP'd back; the fp16/NF4 summaries already exist from the main run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.analysis.compare_quant_pairs import (  # noqa: E402
    mcnemar_exact_test,
    paired_binary_confusion,
)

# pair_id -> {precision: alias}. fp16 = baseline, int8 = LLM.int8, nf4 = 4-bit.
PAIRS: Dict[str, Dict[str, str]] = {
    "qwen_2b": {"fp16": "qwen_2b_base", "int8": "qwen_2b_8bit", "nf4": "qwen_2b_4bit"},
    "qwen_4b": {"fp16": "qwen_4b_base", "int8": "qwen_4b_8bit", "nf4": "qwen_4b_4bit"},
    "llama_3_2_3b": {"fp16": "llama_3_2_3b_base", "int8": "llama_3_2_3b_8bit", "nf4": "llama_3_2_3b_4bit"},
    "mistral_7b": {"fp16": "mistral_7b_base", "int8": "mistral_7b_8bit", "nf4": "mistral_7b_4bit"},
    "phi4_mini": {"fp16": "phi4_mini_base", "int8": "phi4_mini_8bit", "nf4": "phi4_mini_4bit"},
}
PRECISIONS = ["fp16", "int8", "nf4"]

# HarmBench judges whose redacted per-prompt sidecars carry ``judge_harmful``
# booleans: the PRIMARY official classifier (D16) and the GPT-4o cross-check
# (D26). Both score HarmBench only. Listed primary-first.
JUDGE_BACKENDS = ["harmbench_cls", "api_judge"]
HARMBENCH_DIR = "harmbench"

# metric key -> (benchmark subdir, summary filename, metrics field, higher_is_worse)
# NOTE on the regex-derived columns (HarmBench v2 proxy AND XSTest over-refusal):
# the original ten models' summary.json carries the *v1* refusal scorer (the
# v1->v2 rescore wrote a separate summary.v2.json; D11-D13), whereas the INT8 run
# already used the v2 patterns at runtime. Reading these columns from summary.json
# would therefore compare v1 (old) against v2 (INT8) in the same column. We read
# summary.v2.json instead (present for every alias after rescore_harmbench), with
# a graceful fallback to summary.json, so both regex-derived columns are
# apples-to-apples across all three precisions. The PRIMARY judge column is
# unaffected (it has always read the judge sidecar).
METRICS = {
    "harmbench_asr_judge": ("harmbench", "summary.judge.harmbench_cls.json", "attack_success_rate", True),
    "harmbench_asr_v2": ("harmbench", "summary.v2.json", "attack_success_rate", True),
    "xstest_over_refusal": ("xstest", "summary.v2.json", "over_refusal_rate", True),
    "mmlu_accuracy": ("mmlu", "summary.json", "accuracy", False),
    "arc_accuracy": ("arc", "summary.json", "accuracy", False),
}


def _read_metric(results_dir: Path, alias: str, bench: str, fname: str, field: str) -> Optional[float]:
    path = results_dir / alias / bench / fname
    if not path.exists() and fname.endswith(".v2.json"):
        # Fallback for any alias whose v2 sidecar was not built: the runtime
        # summary. (For the INT8 aliases these are identical anyway.)
        path = results_dir / alias / bench / fname.replace(".v2.json", ".json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    val = data.get("metrics", {}).get(field)
    return float(val) if isinstance(val, (int, float)) else None


def _trend(values: Dict[str, Optional[float]], higher_is_worse: bool) -> Dict[str, object]:
    """Deltas vs fp16 + a monotonicity/cliff classification across fp16->int8->nf4."""
    f, i, n = values["fp16"], values["int8"], values["nf4"]
    out: Dict[str, object] = {
        "fp16": f, "int8": i, "nf4": n,
        "delta_int8_vs_fp16": (None if (i is None or f is None) else round(i - f, 4)),
        "delta_nf4_vs_fp16": (None if (n is None or f is None) else round(n - f, 4)),
        "delta_nf4_vs_int8": (None if (n is None or i is None) else round(n - i, 4)),
        "shape": "int8_pending" if i is None else None,
    }
    if i is None or f is None or n is None:
        return out
    # "worse" direction: higher for ASR/over-refusal, lower for accuracy.
    seq = [f, i, n] if higher_is_worse else [-f, -i, -n]
    mono = seq[0] <= seq[1] <= seq[2] or seq[0] >= seq[1] >= seq[2]
    # cliff = one step carries the bulk of the fp16->nf4 change.
    total = abs(n - f)
    step1, step2 = abs(i - f), abs(n - i)
    if not mono:
        shape = "non_monotonic"
    elif total < 1e-9:
        shape = "flat"
    elif step1 >= 0.8 * total:
        shape = "cliff_at_int8"   # most degradation already by 8-bit
    elif step2 >= 0.8 * total:
        shape = "cliff_at_nf4"    # 8-bit ~ fp16, the drop is the 4-bit step
    else:
        shape = "graded"          # int8 sits intermediate
    out["shape"] = shape
    return out


def _load_judge_outcomes(results_dir: Path, alias: str, backend: str) -> Dict[str, float]:
    """Reads ``judge_harmful`` per prompt as 0.0/1.0 from a redacted judge sidecar.

    Mirrors ``scripts/judge_agreement._load_judge_outcomes`` (same convention:
    only records whose ``judge_harmful`` is a genuine bool are kept, so
    ``error``/``skipped`` rows are excluded from the paired set). Returns an empty
    dict if the sidecar is absent (e.g. INT8 not yet judged) so the caller can
    report the contrast as unavailable rather than crashing.
    """

    path = results_dir / alias / HARMBENCH_DIR / f"scores.judge.{backend}.jsonl"
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


def _int8_mcnemar(results_dir: Path, aliases: Dict[str, str]) -> Dict[str, Optional[Dict[str, object]]]:
    """fp16-vs-INT8 HarmBench ASR McNemar exact test, per judge.

    For each judge backend, pairs the fp16 (``<pair>_base``) and INT8
    (``<pair>_8bit``) per-prompt ``judge_harmful`` outcomes over their shared
    prompt ids and reports the discordant counts and exact two-sided p-value.
    ``b`` = fp16-safe / INT8-harmful (harmful compliance APPEARED under INT8);
    ``c`` = fp16-harmful / INT8-safe (disappeared). The aggregate
    ``delta`` (= (b - c) / n) equals INT8 ASR − fp16 ASR by construction. The
    {b, c, n_discordant, p_value} keys mirror the contrast schema in
    ``results/analysis/multiple_comparisons.json``; ``n``/``delta``/
    ``uncorrected_significant``/``direction`` are carried for parity. A backend
    with no usable paired data (sidecar missing on either member) is ``None``.
    """

    fp16_alias, int8_alias = aliases["fp16"], aliases["int8"]
    out: Dict[str, Optional[Dict[str, object]]] = {}
    for backend in JUDGE_BACKENDS:
        fp16 = _load_judge_outcomes(results_dir, fp16_alias, backend)
        int8 = _load_judge_outcomes(results_dir, int8_alias, backend)
        shared = set(fp16) & set(int8)
        if not shared:
            out[backend] = None
            continue
        # baseline = fp16, quantized = INT8 -> n01 = appeared (b), n10 = disappeared (c).
        _, n01, n10, _ = paired_binary_confusion(fp16, int8)
        test = mcnemar_exact_test(n01, n10)  # {discordant, b, c, p_value}
        n = len(shared)
        delta = round((test["b"] - test["c"]) / n, 4)
        out[backend] = {
            "n": n,
            "b": test["b"],
            "c": test["c"],
            "n_discordant": test["discordant"],
            "delta": delta,
            "p_value": test["p_value"],
            "uncorrected_significant": test["p_value"] < 0.05,
            "direction": "harmful-ward" if delta >= 0 else "safe-ward",
        }
    return out


def analyse(results_dir: Path) -> Dict[str, object]:
    per_pair: Dict[str, object] = {}
    for pair_id, aliases in PAIRS.items():
        metrics: Dict[str, object] = {}
        for mkey, (bench, fname, field, hiw) in METRICS.items():
            vals = {prec: _read_metric(results_dir, aliases[prec], bench, fname, field)
                    for prec in PRECISIONS}
            metrics[mkey] = _trend(vals, hiw)
        per_pair[pair_id] = {"aliases": aliases, "metrics": metrics,
                             "int8_present": all(
                                 metrics[m]["int8"] is not None for m in
                                 ("harmbench_asr_judge", "mmlu_accuracy")),
                             "harmbench_asr_int8_mcnemar": _int8_mcnemar(results_dir, aliases)}
    return {"scorer_version": "precision_sweep_v1", "precisions": PRECISIONS, "per_pair": per_pair}


def _write_csv(out: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair_id", "metric", "fp16", "int8", "nf4",
                    "d_int8_vs_fp16", "d_nf4_vs_fp16", "shape"])
        for pair_id, rec in out["per_pair"].items():
            for mkey, t in rec["metrics"].items():
                w.writerow([pair_id, mkey, t["fp16"], t["int8"], t["nf4"],
                            t["delta_int8_vs_fp16"], t["delta_nf4_vs_fp16"], t["shape"]])


def main() -> int:
    parser = argparse.ArgumentParser(description="fp16 -> INT8 -> NF4 precision-sweep analysis (T29).")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    out = analyse(results_dir)
    adir = results_dir / "analysis"
    adir.mkdir(exist_ok=True)
    (adir / "precision_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_csv(out, adir / "precision_sweep.csv")

    pending = [p for p, r in out["per_pair"].items() if not r["int8_present"]]
    print("=" * 78)
    print("PRECISION SWEEP  fp16 -> INT8 -> NF4   (HarmBench ASR = judge primary)")
    print("=" * 78)
    if pending:
        print(f"INT8 not yet run for: {', '.join(pending)} (showing fp16/NF4 only for those).")
    print(f"{'pair':<14}{'metric':<22}{'fp16':>8}{'int8':>8}{'nf4':>8}{'shape':>16}")
    for pair_id, rec in out["per_pair"].items():
        for mkey in ("harmbench_asr_judge", "mmlu_accuracy", "arc_accuracy", "xstest_over_refusal"):
            t = rec["metrics"][mkey]
            def s(v): return "  pend" if v is None else f"{v:.3f}"
            print(f"{pair_id:<14}{mkey:<22}{s(t['fp16']):>8}{s(t['int8']):>8}{s(t['nf4']):>8}{str(t['shape']):>16}")
    print("-" * 78)
    print("fp16 -> INT8 HarmBench ASR McNemar exact test (b=appeared, c=disappeared):")
    print(f"{'pair':<14}{'judge':<16}{'b':>4}{'c':>4}{'n_disc':>8}{'p_value':>10}{'sig?':>6}")
    for pair_id, rec in out["per_pair"].items():
        for backend in JUDGE_BACKENDS:
            m = rec["harmbench_asr_int8_mcnemar"].get(backend)
            if m is None:
                print(f"{pair_id:<14}{backend:<16}{'n/a':>4}{'':>4}{'':>8}{'pend':>10}{'':>6}")
            else:
                print(f"{pair_id:<14}{backend:<16}{m['b']:>4}{m['c']:>4}{m['n_discordant']:>8}"
                      f"{m['p_value']:>10.4f}{str(m['uncorrected_significant']):>6}")
    print("-" * 78)
    print(f"Wrote {adir/'precision_sweep.json'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
