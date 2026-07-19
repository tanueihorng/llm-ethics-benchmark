#!/usr/bin/env python3
"""Validation-informed parallel BH-FDR family (post-hoc, composition-locked).

Recomputes the study's 20-contrast multiplicity family with the XSTest
over-refusal column scored by the gpt-4o judge under the STRICT mapping (the
T35-preregistered replication definition), after the T36 human audit found the
judge substantially better aligned with the single annotator than the v2 regex
(docs/XSTEST_GOLD_PREREG.md §7, Outcome J). Composition, expectations, and
reporting rules were locked BEFORE this script first ran:
docs/VALIDATION_INFORMED_FAMILY_NOTE.md.

Inputs (both committed, never modified):
  results_512/analysis/multiple_comparisons.json     — the registered family;
      its 15 non-over-refusal p-values are reused verbatim.
  results_512/analysis/xstest_judge_agreement.json   — T35; per-pair
      judge-strict exact-McNemar p-values for the 5 over-refusal contrasts.

Output (new sibling artifact; the registered family is the family of record):
  results_512/analysis/multiple_comparisons_judge_strict.json

Deterministic; no RNG, no network, no GPU. Run:
  python scripts/validation_informed_family.py [--results-dir results_512]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multiple_comparisons import benjamini_hochberg

ROOT = Path(__file__).resolve().parents[1]


def build(results_dir: str = "results_512") -> dict:
    analysis = ROOT / results_dir / "analysis"
    registered = json.loads((analysis / "multiple_comparisons.json").read_text())
    judge = json.loads((analysis / "xstest_judge_agreement.json").read_text())

    strict_by_pair = {p["pair_id"]: p["judge_strict"] for p in judge["per_pair"]}

    contrasts = []
    for c in registered["contrasts"]:
        if c["metric"] == "xstest_over_refusal":
            js = strict_by_pair[c["pair_id"]]
            contrasts.append({
                "pair_id": c["pair_id"],
                "metric": "xstest_over_refusal_judge_strict",
                "n": 250,
                "n_discordant": js["mcnemar_discordant"],
                "b": js["mcnemar_b_to_refusal"],
                "c": js["mcnemar_c_to_compliance"],
                "delta": js["delta"],
                "ci_lower": js["ci_lower"],
                "ci_upper": js["ci_upper"],
                "p_value": js["mcnemar_p_value"],
                "uncorrected_significant": bool(js["significant"]),
                "direction": js["direction"],
                "replaces_registered_p": c["p_value"],
            })
        else:
            contrasts.append({k: c[k] for k in (
                "pair_id", "metric", "n", "b", "c", "n_discordant", "delta",
                "p_value", "uncorrected_significant", "direction")})

    pvals = [c["p_value"] for c in contrasts]
    qvals, reject = benjamini_hochberg(pvals, q=0.05)
    for c, qv, rj in zip(contrasts, qvals, reject):
        c["bh_q_value"] = round(qv, 4)
        c["bh_significant_q05"] = bool(rj)

    survivors = [
        {"pair_id": c["pair_id"], "metric": c["metric"], "delta": c["delta"],
         "p_value": c["p_value"], "bh_q_value": c["bh_q_value"]}
        for c in contrasts if c["bh_significant_q05"]
    ]
    return {
        "description": (
            "Validation-informed PARALLEL BH-FDR family (post-hoc, added after "
            "T36 Outcome J): the registered 20-contrast family with the XSTest "
            "over-refusal column scored by the gpt-4o judge, STRICT mapping "
            "(T35-preregistered replication definition). The registered "
            "regex-scored family (multiple_comparisons.json) remains the "
            "family of record; this artifact never modifies it. Composition "
            "locked before computation: docs/VALIDATION_INFORMED_FAMILY_NOTE.md."
        ),
        "family_size": len(contrasts),
        "alpha": registered["alpha"],
        "fdr_level_q": registered["fdr_level_q"],
        "or_scorer": "xstest_api judge, strict mapping (judge_xstest_refusal_api_v1_2026-07-12)",
        "n_uncorrected_significant": sum(c["uncorrected_significant"] for c in contrasts),
        "n_bh_significant_q05": len(survivors),
        "bh_survivors": survivors,
        "registered_family_survivor_count": registered["n_bh_significant_q05"],
        "contrasts": contrasts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results_512")
    args = ap.parse_args()
    out = build(args.results_dir)
    path = ROOT / args.results_dir / "analysis" / "multiple_comparisons_judge_strict.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"Wrote {path}")
    print(f"BH survivors ({out['n_bh_significant_q05']}):")
    for s in out["bh_survivors"]:
        print(f"  {s['pair_id']:<14} {s['metric']:<18} delta {s['delta']:+.3f}  q {s['bh_q_value']}")
    print(f"(registered family of record: {out['registered_family_survivor_count']} survivors — unchanged)")


if __name__ == "__main__":
    main()
