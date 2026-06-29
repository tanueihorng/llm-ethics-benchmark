#!/usr/bin/env python3
"""Family-wise multiple-comparison correction (Benjamini-Hochberg FDR) and a
minimum-detectable-effect / power analysis for the main fp16-vs-NF4 study.

Motivation (publication-grade reporting). The report previously *acknowledged*
in prose (§6.5) that no multiplicity correction was applied and that the single
significant NF4 ΔASR (Qwen-1.7B, p = 0.027) "would not survive a strict
family-wise correction", but it never *reported* the corrected view, and it
carried no power / minimum-detectable-effect (MDE) statement. A statistics
reviewer at a peer-reviewed venue requires both. This script produces them as a
committed, redaction-safe artifact (counts and aggregate statistics only; no
prompt or response text, no per-prompt IDs).

What it computes, over the family of primary NF4-vs-fp16 contrasts
(5 pairs x {HarmBench ASR (judge), MMLU, ARC, XSTest over-refusal} = 20 tests):
  * the exact McNemar two-sided p-value for each paired-binary contrast,
    reusing ``mcnemar_exact_test`` from the analysis module so the numbers are
    identical to the rest of the pipeline;
  * Benjamini-Hochberg adjusted q-values and which contrasts survive at
    q < 0.05 (FDR control), reported alongside the uncorrected flags;
  * for the HarmBench ASR axis, the per-pair minimum detectable effect (MDE)
    for a two-sided alpha = 0.05, power = 0.80 McNemar test at the observed
    discordant rate, and the post-hoc power for the observed delta -- so that
    "only one significant effect" is quantified as partly a power floor, not
    merely asserted.

Inputs (all already committed except raw.jsonl, which is the immutable TC1
original read locally for MMLU/ARC per-prompt correctness):
  * results/analysis/judge_agreement.json   (HarmBench judge b/c/p/delta/CI)
  * results/<alias>/{mmlu,arc}/raw.jsonl     (per-prompt is_correct)
  * results/<alias>/xstest/scores.v2.jsonl   (per-prompt is_refusal)

Output: results/analysis/multiple_comparisons.json (+ .csv).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ethical_benchmark.analysis.compare_quant_pairs import (
    mcnemar_exact_test,
    paired_binary_confusion,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ANALYSIS = RESULTS / "analysis"

# pair_id -> (fp16 alias, nf4 alias). Shared model_id; only the quant flag differs.
PAIRS: Dict[str, Tuple[str, str]] = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}

# Standard normal quantiles used in the MDE / power formulas.
Z_ALPHA_2 = 1.959963984540054   # two-sided alpha = 0.05
Z_BETA = 0.8416212335729143     # power = 0.80


def _phi(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _paired_outcomes(
    base_alias: str, quant_alias: str, benchmark: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Returns (base, quant) maps prompt_id -> 0/1 for a paired-binary outcome.

    MMLU/ARC read ``score_fields.is_correct`` from the immutable raw.jsonl;
    XSTest reads ``score_fields.is_refusal`` from the committed v2 sidecar.
    Both members of a pair share the same prompt_id set by construction.
    """
    if benchmark in ("mmlu", "arc"):
        field, src = "is_correct", "raw.jsonl"
    elif benchmark == "xstest":
        field, src = "is_refusal", "scores.v2.jsonl"
    else:
        raise ValueError(f"unsupported per-prompt benchmark: {benchmark}")

    def load(alias: str) -> Dict[str, int]:
        path = RESULTS / alias / benchmark / src
        out: Dict[str, int] = {}
        for r in _read_jsonl(path):
            sf = r.get("score_fields", {})
            val = sf.get(field)
            if val is None:
                continue
            out[str(r["prompt_id"])] = 1 if val else 0
        return out

    return load(base_alias), load(quant_alias)


def _confusion_from_maps(
    base: Dict[str, int], quant: Dict[str, int]
) -> Tuple[int, int, int]:
    """Returns (b, c, n) where b = base0->quant1, c = base1->quant0 over the
    shared prompt set. 'positive' here is the metric's event (correct / refusal /
    harmful); direction is recorded but McNemar's p is direction-agnostic."""
    shared = sorted(set(base) & set(quant))
    b = sum(1 for pid in shared if base[pid] == 0 and quant[pid] == 1)
    c = sum(1 for pid in shared if base[pid] == 1 and quant[pid] == 0)
    return b, c, len(shared)


def mde_mcnemar(psi: float, n: int) -> Optional[float]:
    """Minimum detectable marginal-proportion difference for a two-sided
    alpha = 0.05, power = 0.80 McNemar test, given discordant proportion psi
    and n paired observations (Connor 1987 / Lachin normal approximation):

        delta_MDE ~= (z_{1-alpha/2} + z_{1-beta}) * sqrt(psi / n).
    """
    if n <= 0 or psi <= 0:
        return None
    return (Z_ALPHA_2 + Z_BETA) * math.sqrt(psi / n)


def power_mcnemar(b: int, c: int, n: int) -> Optional[float]:
    """Post-hoc power to detect the observed marginal difference under McNemar's
    normal approximation, using the observed discordant split."""
    if n <= 0:
        return None
    psi = (b + c) / n
    delta = abs(b - c) / n
    if psi <= 0 or delta <= 0:
        return 0.0
    denom = psi - delta * delta
    if denom <= 0:
        return 1.0
    z = (delta * math.sqrt(n) - Z_ALPHA_2 * math.sqrt(psi)) / math.sqrt(denom)
    return _phi(z)


def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Returns (adjusted_q_values, reject_flags) for BH-FDR at level q.

    Adjusted values are the standard step-up monotone BH q-values; reject_flags
    apply the BH threshold (largest k with p_(k) <= k/m * q)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    # step-up: iterate from largest p to smallest, enforce monotonicity
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * m / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    # threshold-based rejection
    reject = [False] * m
    crit = -1
    for rank in range(1, m + 1):
        i = order[rank - 1]
        if pvals[i] <= rank / m * q:
            crit = rank
    if crit > 0:
        for rank in range(1, crit + 1):
            reject[order[rank - 1]] = True
    return adj, reject


def main() -> None:
    ja = json.load((ANALYSIS / "judge_agreement.json").open())
    judge_per_pair = {row["pair_id"]: row for row in ja["per_pair"]}

    contrasts: List[dict] = []

    for pid, (base, quant) in PAIRS.items():
        # HarmBench ASR (judge) -- reuse the committed McNemar from judge_agreement.
        jp = judge_per_pair[pid]
        b, c = jp["mcnemar_b_base_to_harmful"], jp["mcnemar_c_harmful_to_safe"]
        contrasts.append({
            "pair_id": pid, "metric": "harmbench_asr_judge",
            "n": 200, "b": b, "c": c, "n_discordant": b + c,
            "delta": round(jp["judge_harm_delta"], 4),
            "p_value": jp["mcnemar_p_value"],
            "uncorrected_significant": bool(jp["mcnemar_p_value"] < 0.05),
            "direction": "harmful-ward" if b >= c else "safe-ward",
        })

        # MMLU, ARC, XSTest -- compute McNemar from per-prompt paired outcomes.
        for benchmark, metric in (
            ("mmlu", "mmlu_accuracy"),
            ("arc", "arc_accuracy"),
            ("xstest", "xstest_over_refusal"),
        ):
            bm, qm = _paired_outcomes(base, quant, benchmark)
            bb, cc, n = _confusion_from_maps(bm, qm)
            mc = mcnemar_exact_test(bb, cc)
            base_rate = sum(bm.values()) / max(len(bm), 1)
            quant_rate = sum(qm.values()) / max(len(qm), 1)
            contrasts.append({
                "pair_id": pid, "metric": metric,
                "n": n, "b": bb, "c": cc, "n_discordant": bb + cc,
                "delta": round(quant_rate - base_rate, 4),
                "p_value": mc["p_value"],
                "uncorrected_significant": bool(mc["p_value"] < 0.05),
                "direction": "up" if bb >= cc else "down",
            })

    # Benjamini-Hochberg over the full 20-test family.
    pvals = [c["p_value"] for c in contrasts]
    qvals, rejects = benjamini_hochberg(pvals, q=0.05)
    for c, qv, rj in zip(contrasts, qvals, rejects):
        c["bh_q_value"] = round(qv, 4)
        c["bh_significant_q05"] = bool(rj)

    # MDE / power for the HarmBench ASR axis (the safety headline lives here).
    asr_power = []
    for c in contrasts:
        if c["metric"] != "harmbench_asr_judge":
            continue
        psi = c["n_discordant"] / c["n"]
        asr_power.append({
            "pair_id": c["pair_id"],
            "discordant_rate": round(psi, 4),
            "mde_delta_asr": round(mde_mcnemar(psi, c["n"]) or 0.0, 4),
            "post_hoc_power": round(power_mcnemar(c["b"], c["c"], c["n"]) or 0.0, 4),
        })
    # A representative MDE at the median discordant rate across pairs.
    psis = sorted(c["n_discordant"] / c["n"] for c in contrasts if c["metric"] == "harmbench_asr_judge")
    median_psi = psis[len(psis) // 2]
    representative_mde = round(mde_mcnemar(median_psi, 200) or 0.0, 4)

    n_survive = sum(1 for c in contrasts if c["bh_significant_q05"])
    n_uncorrected = sum(1 for c in contrasts if c["uncorrected_significant"])

    out = {
        "description": (
            "Benjamini-Hochberg FDR correction over the family of 20 primary "
            "NF4-vs-fp16 contrasts (5 pairs x {HarmBench ASR judge, MMLU, ARC, "
            "XSTest over-refusal}), plus McNemar MDE/power for the HarmBench ASR "
            "axis. Counts/aggregates only; redaction-safe."
        ),
        "family_size": len(contrasts),
        "alpha": 0.05,
        "fdr_level_q": 0.05,
        "n_uncorrected_significant": n_uncorrected,
        "n_bh_significant_q05": n_survive,
        "bh_survivors": [
            {"pair_id": c["pair_id"], "metric": c["metric"], "p_value": round(c["p_value"], 4),
             "bh_q_value": c["bh_q_value"], "delta": c["delta"]}
            for c in contrasts if c["bh_significant_q05"]
        ],
        "power_analysis": {
            "test": "McNemar exact (two-sided), normal-approx MDE/power",
            "target_alpha": 0.05,
            "target_power": 0.80,
            "representative_mde_delta_asr_at_median_discordant_rate": representative_mde,
            "median_discordant_rate": round(median_psi, 4),
            "note": (
                "At n=200 and the observed HarmBench discordant rates, the "
                "minimum detectable ASR delta for 80%% power is ~%.2f; the only "
                "NF4 ASR effect at/above this floor is Qwen-1.7B (+0.055). "
                "'Only one significant effect' is therefore partly a power floor, "
                "not solely a substantive null." % representative_mde
            ),
            "per_pair_harmbench_asr": asr_power,
        },
        "contrasts": contrasts,
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "multiple_comparisons.json").write_text(json.dumps(out, indent=2))

    # Flat CSV for spreadsheet inspection.
    with (ANALYSIS / "multiple_comparisons.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "metric", "n", "b", "c", "n_discordant", "delta",
                    "p_value", "uncorrected_significant", "bh_q_value", "bh_significant_q05", "direction"])
        for c in contrasts:
            w.writerow([c["pair_id"], c["metric"], c["n"], c["b"], c["c"], c["n_discordant"],
                        c["delta"], round(c["p_value"], 5), c["uncorrected_significant"],
                        c["bh_q_value"], c["bh_significant_q05"], c["direction"]])

    print(f"Family size: {len(contrasts)} contrasts")
    print(f"Uncorrected significant (p<0.05): {n_uncorrected}")
    print(f"BH-significant (q<0.05): {n_survive}")
    print("BH survivors:")
    for s in out["bh_survivors"]:
        print(f"  {s['pair_id']:14s} {s['metric']:22s} p={s['p_value']:.4f} q={s['bh_q_value']:.4f} delta={s['delta']:+.4f}")
    print(f"Representative MDE (ASR, 80% power, median discordant rate): {representative_mde}")
    print("Per-pair HarmBench ASR power:")
    for ap in asr_power:
        print(f"  {ap['pair_id']:14s} discordant={ap['discordant_rate']:.3f} "
              f"MDE={ap['mde_delta_asr']:.3f} post-hoc-power={ap['post_hoc_power']:.3f}")


if __name__ == "__main__":
    main()
