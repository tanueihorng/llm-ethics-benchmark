#!/usr/bin/env python3
"""Capability-axis power and equivalence analysis (T48 / audit findings C14, C15).

Two questions the study previously asserted rather than tested:

  C15  "the study has enough power to resolve the capability losses"
       The published power analysis computed a minimum detectable effect for
       HarmBench ASR only. This module runs the IDENTICAL Connor/Lachin
       normal-approximation procedure (imported from multiple_comparisons, not
       reimplemented) on the MMLU and ARC NF4-vs-fp16 contrasts, so the
       capability power claim rests on a number instead of an inference from
       "it came out significant" (which is the post-hoc power fallacy Card
       et al. 2020, footnote 2, explicitly warn against).

  C14  "INT8 is capability-lossless"
       Ten non-significant contrasts are not evidence of equivalence. This
       module runs TOST (two one-sided tests) on the paired INT8-vs-fp16
       capability contrasts, which is the test that can actually support an
       equivalence claim.

Margin declaration (must be read as post-hoc): the primary equivalence margin
DELTA_PRIMARY = 0.05 was fixed before the TOST was computed, on the grounds
that a five-percentage-point capability change is the smallest difference that
would plausibly alter a deployment decision between two precisions of the same
checkpoint. It is NOT derived from the observed data. The additional
"tightest common margin" reported alongside it IS derived from the data and is
descriptive only; it must never be presented as the pre-declared margin.

Reads immutable raw.jsonl records only; writes derived artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multiple_comparisons as mc  # noqa: E402  (path set above)
from multiple_comparisons import (  # noqa: E402
    PAIRS,
    _confusion_from_maps,
    _paired_outcomes,
    _phi,
    mde_mcnemar,
)

REPO = Path(__file__).resolve().parent.parent

# multiple_comparisons binds RESULTS at module level to the 128-token `results/`
# tree and only rebinds it inside its own main(). Importing its loaders WITHOUT
# repointing RESULTS silently reads the wrong tree — that happened on the first
# run of this script and produced a plausible-looking but wrong ΔMMLU (−0.0867
# instead of −0.090). _assert_tree_matches_summaries below is the guard: every
# baseline rate recomputed from the paired records must equal the accuracy in
# that alias's summary.json, or the run aborts.

CAPABILITY_BENCHMARKS = ("mmlu", "arc")
DELTA_PRIMARY = 0.05          # pre-declared equivalence margin (see module docstring)
ALPHA_TOST = 0.05
Z_90 = 1.6448536269514722     # two-sided 90% CI == the TOST-consistent interval


def _int8_alias(base_alias: str) -> str:
    """qwen_2b_base -> qwen_2b_8bit."""
    if not base_alias.endswith("_base"):
        raise ValueError(f"unexpected baseline alias: {base_alias}")
    return base_alias[: -len("_base")] + "_8bit"


def paired_delta(b: int, c: int, n: int) -> Tuple[float, Optional[float], float]:
    """Returns (delta, se, discordant_rate) for a paired-binary contrast.

    delta = (b - c)/n is quant_rate - base_rate, i.e. the signed marginal
    difference. The variance is the standard paired-proportion form
    Var = [(b + c) - (b - c)^2 / n] / n^2.
    """
    if n <= 0:
        return 0.0, None, 0.0
    delta = (b - c) / n
    psi = (b + c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n ** 2)
    se = math.sqrt(var) if var > 0 else None
    return delta, se, psi


def tost(delta: float, se: Optional[float], margin: float) -> Dict[str, object]:
    """Two one-sided tests for equivalence within +/- margin.

    p_tost = max(p_lower, p_upper); equivalence is declared when p_tost < alpha,
    which is identical to the 90% CI lying strictly inside (-margin, +margin).
    """
    if se is None or se == 0.0:
        # No discordant pairs at all: the paired difference is exactly zero with
        # no sampling variability under the normal approximation. Equivalence
        # holds trivially for any positive margin, but flag it as degenerate.
        inside = abs(delta) < margin
        return {
            "margin": margin,
            "p_tost": 0.0 if inside else 1.0,
            "equivalent": bool(inside),
            "degenerate_zero_variance": True,
        }
    p_lower = 1.0 - _phi((delta + margin) / se)   # H0: d <= -margin
    p_upper = _phi((delta - margin) / se)         # H0: d >= +margin
    p_tost = max(p_lower, p_upper)
    return {
        "margin": margin,
        "p_tost": round(p_tost, 6),
        "equivalent": bool(p_tost < ALPHA_TOST),
        "degenerate_zero_variance": False,
    }


def tightest_margin(delta: float, se: Optional[float]) -> Optional[float]:
    """Smallest margin at which this contrast would be declared equivalent.

    This is |delta| + z_(1-alpha) * se, i.e. the outer edge of the 90% CI.
    DATA-DERIVED: descriptive only, never the pre-declared margin.
    """
    if se is None:
        return abs(delta)
    return abs(delta) + Z_90 * se


def _assert_tree_matches_summaries(
    alias: str, benchmark: str, outcome_map: Dict[str, int]
) -> None:
    """Abort if the paired records disagree with that alias's own summary.json.

    This is the guard against reading the wrong results tree (see module header).
    A mismatch means the loader resolved to a different directory than intended,
    which is silent and produces numbers that look entirely reasonable.
    """
    summary_path = mc.RESULTS / alias / benchmark / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"ABORT: missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    expected = summary.get("metrics", {}).get("accuracy")
    if expected is None:
        raise SystemExit(f"ABORT: no accuracy metric in {summary_path}")
    if not outcome_map:
        raise SystemExit(f"ABORT: empty paired records for {alias}/{benchmark}")
    observed = sum(outcome_map.values()) / len(outcome_map)
    if abs(observed - expected) > 1e-9:
        raise SystemExit(
            f"ABORT: {alias}/{benchmark} paired records give accuracy "
            f"{observed:.6f} but summary.json says {expected:.6f}. "
            f"The loader is reading {mc.RESULTS} — wrong results tree?"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT8 capability equivalence (TOST) + capability-axis MDE."
    )
    parser.add_argument(
        "--results-dir",
        default="results_512",
        help="Results tree to analyse (default: results_512, the primary study).",
    )
    args = parser.parse_args()
    mc.RESULTS = REPO / args.results_dir
    out_dir = REPO / args.results_dir / "analysis"
    if not mc.RESULTS.is_dir():
        raise SystemExit(f"ABORT: no such results tree: {mc.RESULTS}")
    print(f"results tree: {mc.RESULTS}")

    equivalence_rows: List[dict] = []
    mde_rows: List[dict] = []

    for pair_id, (base_alias, nf4_alias) in PAIRS.items():
        int8_alias = _int8_alias(base_alias)
        for benchmark in CAPABILITY_BENCHMARKS:
            # ---- C14: INT8 vs fp16 equivalence -------------------------------
            base_map, int8_map = _paired_outcomes(base_alias, int8_alias, benchmark)
            _assert_tree_matches_summaries(base_alias, benchmark, base_map)
            _assert_tree_matches_summaries(int8_alias, benchmark, int8_map)
            b, c, n = _confusion_from_maps(base_map, int8_map)
            delta, se, psi = paired_delta(b, c, n)
            result = tost(delta, se, DELTA_PRIMARY)
            lo = delta - Z_90 * se if se else delta
            hi = delta + Z_90 * se if se else delta
            equivalence_rows.append(
                {
                    "pair_id": pair_id,
                    "benchmark": benchmark,
                    "contrast": "int8_vs_fp16",
                    "n": n,
                    "b_gained": b,
                    "c_lost": c,
                    "discordant_rate": round(psi, 4),
                    "delta": round(delta, 4),
                    "se": round(se, 5) if se else None,
                    "ci90_low": round(lo, 4),
                    "ci90_high": round(hi, 4),
                    "tightest_equivalence_margin": (
                        round(tightest_margin(delta, se), 4)
                    ),
                    **result,
                }
            )

            # ---- C15: capability MDE on the NF4 contrast ---------------------
            base_map2, nf4_map = _paired_outcomes(base_alias, nf4_alias, benchmark)
            _assert_tree_matches_summaries(nf4_alias, benchmark, nf4_map)
            b2, c2, n2 = _confusion_from_maps(base_map2, nf4_map)
            delta2, _se2, psi2 = paired_delta(b2, c2, n2)
            mde = mde_mcnemar(psi2, n2)
            mde_rows.append(
                {
                    "pair_id": pair_id,
                    "benchmark": benchmark,
                    "contrast": "nf4_vs_fp16",
                    "n": n2,
                    "discordant_rate": round(psi2, 4),
                    "observed_delta": round(delta2, 4),
                    "mde_80pct_power": round(mde, 4) if mde else None,
                    "observed_above_mde": (
                        bool(mde is not None and abs(delta2) >= mde)
                    ),
                }
            )

    n_equiv = sum(1 for r in equivalence_rows if r["equivalent"])
    worst = max(equivalence_rows, key=lambda r: r["tightest_equivalence_margin"])
    common_margin = worst["tightest_equivalence_margin"]
    resolved = [r for r in mde_rows if r["observed_above_mde"]]

    payload = {
        "description": (
            "Capability-axis equivalence (INT8 vs fp16) and minimum detectable "
            "effect (NF4 vs fp16), computed from the immutable paired per-item "
            "records in results_512. Added post hoc in response to audit "
            "findings C14 and C15; see docs/PROJECT_LOG.md."
        ),
        "scorer": "exact-match (is_correct) from raw.jsonl; no judge involved",
        "equivalence": {
            "test": "TOST (two one-sided tests), normal approximation for paired proportions",
            "alpha": ALPHA_TOST,
            "margin_primary": DELTA_PRIMARY,
            "margin_declaration": (
                "Fixed before computation on deployment-relevance grounds, not "
                "derived from the observed data. The analysis as a whole is "
                "post hoc: it was added after the INT8 results were known."
            ),
            "n_contrasts": len(equivalence_rows),
            "n_equivalent_at_primary_margin": n_equiv,
            "tightest_common_margin_DATA_DERIVED": round(common_margin, 4),
            "tightest_common_margin_note": (
                "Smallest margin at which ALL contrasts would be declared "
                "equivalent (the widest 90%-CI edge, on "
                f"{worst['pair_id']}/{worst['benchmark']}). Descriptive only: "
                "it is chosen with knowledge of the data and must not be "
                "presented as a pre-declared margin."
            ),
            "contrasts": equivalence_rows,
        },
        "capability_power": {
            "test": "McNemar normal-approximation MDE (Connor 1987 / Lachin), "
                    "identical procedure to the HarmBench MDE in multiple_comparisons.py",
            "target_alpha": 0.05,
            "target_power": 0.8,
            "n_contrasts": len(mde_rows),
            "n_observed_at_or_above_mde": len(resolved),
            "resolved_contrasts": [
                f"{r['pair_id']}/{r['benchmark']} ({r['observed_delta']:+.4f} "
                f"vs MDE {r['mde_80pct_power']:.4f})" for r in resolved
            ],
            "contrasts": mde_rows,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_equivalence.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    with (out_dir / "capability_equivalence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(equivalence_rows[0].keys()))
        w.writeheader()
        w.writerows(equivalence_rows)
    with (out_dir / "capability_mde.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mde_rows[0].keys()))
        w.writeheader()
        w.writerows(mde_rows)

    print(f"INT8 equivalence at margin +/-{DELTA_PRIMARY}: "
          f"{n_equiv}/{len(equivalence_rows)} contrasts equivalent")
    print(f"  tightest margin covering all contrasts (data-derived): "
          f"{common_margin:.4f}")
    for r in equivalence_rows:
        print(f"   {r['pair_id']:<14} {r['benchmark']:<5} n={r['n']:<5} "
              f"d={r['delta']:+.4f}  90% CI [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}]  "
              f"p_TOST={r['p_tost']:.4f}  {'EQUIV' if r['equivalent'] else 'not equiv'}")
    print()
    print(f"Capability MDE (NF4 contrasts): {len(resolved)}/{len(mde_rows)} "
          f"observed effects at or above their 80%-power floor")
    for r in mde_rows:
        flag = "RESOLVED" if r["observed_above_mde"] else "below floor"
        print(f"   {r['pair_id']:<14} {r['benchmark']:<5} n={r['n']:<5} "
              f"psi={r['discordant_rate']:.3f}  d={r['observed_delta']:+.4f}  "
              f"MDE={r['mde_80pct_power']:.4f}  {flag}")


if __name__ == "__main__":
    main()
