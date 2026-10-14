#!/usr/bin/env python3
"""Why did NF4 lower prefill-arm harmful compliance? Post hoc, descriptive (T87).

Joins the numbers-only continuation sidecars written on TC1
(scripts/prefill_continuation_stats.py) with the HarmBench classifier labels and
asks, per FP16/NF4 pair:
  * Are NF4 continuations shorter (paired Wilcoxon signed-rank on word counts)?
  * Do they hit the 512-token cap less often, or contain explicit refusal
    wording more often (exact McNemar)?
  * Among prompts that were harmful at FP16 but not at NF4, did the NF4 reply
    contain refusal wording ("recovered"), stop short (< 50 words, no refusal
    wording), or carry on at length without refusal wording?

Not prespecified (PREFILL_PREREG §11 follow-up). The 50-word cut-off is a
descriptive convenience, not a validated threshold. Refusal wording is the
study's keyword list, which sees explicit refusals only.

Output: results_prefill_512/analysis/prefill_continuation.json. No text is read.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.analysis.compare_quant_pairs import mcnemar_exact_test  # noqa: E402

PAIRS = ["qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"]
SHORT_WORDS = 50


def _load(path: Path, key: str | None = None) -> Dict[str, Any]:
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["prompt_id"]: (r[key] if key else r) for r in rows}


def _med(values: List[float]) -> float | None:
    return float(st.median(values)) if values else None


def _paired_binary(a: Dict[str, bool], b: Dict[str, bool], ids: List[str]) -> Dict[str, Any]:
    only_b = sum(b[i] and not a[i] for i in ids)
    only_a = sum(a[i] and not b[i] for i in ids)
    return {
        "fp16_rate": sum(a[i] for i in ids) / len(ids),
        "nf4_rate": sum(b[i] for i in ids) / len(ids),
        "only_nf4": only_b,
        "only_fp16": only_a,
        "p_mcnemar": mcnemar_exact_test(only_b, only_a)["p_value"],
    }


def _categorise(stats: Dict[str, Any], ids: List[str]) -> Dict[str, int]:
    out = {"refusal_wording": 0, "short_no_refusal_wording": 0, "long_no_refusal_wording": 0}
    for i in ids:
        s = stats[i]
        if s["refusal_kw"]:
            out["refusal_wording"] += 1
        elif s["cont_words"] < SHORT_WORDS:
            out["short_no_refusal_wording"] += 1
        else:
            out["long_no_refusal_wording"] += 1
    return out


def analyse_pair(root: Path, pair: str) -> Dict[str, Any]:
    base, quant = f"{pair}_base", f"{pair}_4bit"
    stats = {m: _load(root / m / "harmbench" / "scores.continuation_stats.jsonl") for m in (base, quant)}
    harm = {m: _load(root / m / "harmbench" / "scores.judge.harmbench_cls.jsonl", "judge_harmful") for m in (base, quant)}
    ids = sorted(set(stats[base]) & set(stats[quant]) & set(harm[base]) & set(harm[quant]))

    words_b = [stats[base][i]["cont_words"] for i in ids]
    words_q = [stats[quant][i]["cont_words"] for i in ids]
    diffs = [q - b for b, q in zip(words_b, words_q)]
    w = wilcoxon(words_q, words_b) if any(diffs) else None

    fp_only = [i for i in ids if harm[base][i] and not harm[quant][i]]
    nf_only = [i for i in ids if harm[quant][i] and not harm[base][i]]
    both = [i for i in ids if harm[base][i] and harm[quant][i]]

    def lengths(group: List[str]) -> Dict[str, Any]:
        b = [stats[base][i]["cont_words"] for i in group]
        q = [stats[quant][i]["cont_words"] for i in group]
        return {"n": len(group), "fp16_median_words": _med(b), "nf4_median_words": _med(q),
                "median_paired_diff_words": _med([y - x for x, y in zip(b, q)])}

    return {
        "pair_id": pair,
        "n": len(ids),
        "length": {
            "fp16_median_words": _med(words_b),
            "nf4_median_words": _med(words_q),
            "median_paired_diff_words": _med(diffs),
            "wilcoxon_p": float(w.pvalue) if w else 1.0,
        },
        "near_cap": _paired_binary({i: stats[base][i]["near_cap"] for i in ids},
                                   {i: stats[quant][i]["near_cap"] for i in ids}, ids),
        "refusal_wording": _paired_binary({i: stats[base][i]["refusal_kw"] for i in ids},
                                          {i: stats[quant][i]["refusal_kw"] for i in ids}, ids),
        "harmful_fp16_only": {**lengths(fp_only),
                              "nf4_reply_type": _categorise(stats[quant], fp_only),
                              "fp16_reply_type": _categorise(stats[base], fp_only)},
        "harmful_nf4_only": {**lengths(nf_only),
                             "nf4_reply_type": _categorise(stats[quant], nf_only),
                             "fp16_reply_type": _categorise(stats[base], nf_only)},
        "harmful_both": lengths(both),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results_prefill_512")
    args = parser.parse_args()
    root = ROOT / args.results_dir
    rows = [analyse_pair(root, p) for p in PAIRS]
    out = {
        "status": "post hoc, descriptive; not prespecified (PREFILL_PREREG §11 follow-up)",
        "short_words_cutoff": SHORT_WORDS,
        "refusal_wording": "study keyword list (ethical_benchmark.benchmarks.utils), explicit refusals only",
        "pairs": rows,
    }
    path = root / "analysis" / "prefill_continuation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    for r in rows:
        L, C, R, F, N = r["length"], r["near_cap"], r["refusal_wording"], r["harmful_fp16_only"], r["harmful_nf4_only"]
        print(f"\n{r['pair_id']}  (n={r['n']})")
        print(f"  words median FP16 {L['fp16_median_words']:.0f} NF4 {L['nf4_median_words']:.0f}; "
              f"paired diff {L['median_paired_diff_words']:+.0f}; Wilcoxon p {L['wilcoxon_p']:.3g}")
        print(f"  near 512 cap FP16 {C['fp16_rate']:.3f} NF4 {C['nf4_rate']:.3f} (p {C['p_mcnemar']:.3g}); "
              f"refusal wording FP16 {R['fp16_rate']:.3f} NF4 {R['nf4_rate']:.3f} (p {R['p_mcnemar']:.3g})")
        print(f"  harmful FP16 only: n={F['n']} NF4 reply {F['nf4_reply_type']}; words FP16 {F['fp16_median_words']} NF4 {F['nf4_median_words']}")
        print(f"  harmful NF4 only:  n={N['n']} FP16 reply {N['fp16_reply_type']}; words FP16 {N['fp16_median_words']} NF4 {N['nf4_median_words']}")


if __name__ == "__main__":
    main()
