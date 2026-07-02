#!/usr/bin/env python3
"""Authoritative 512-primary numbers for the report re-base + verification.
Reads results_512/analysis/ only. Prints a compact 'number bible' so every value
inserted into the report can be cross-checked against a single source."""
import json
from pathlib import Path
A = Path("results_512/analysis")
A128 = Path("results/analysis")
PAIRS = ["qwen_2b","qwen_4b","llama_3_2_3b","mistral_7b","phi4_mini"]
LAB = {"qwen_2b":"Qwen3-1.7B","qwen_4b":"Qwen3-4B","llama_3_2_3b":"Llama-3.2-3B",
       "mistral_7b":"Mistral-7B","phi4_mini":"Phi-4-mini"}

def j(p): return json.load(open(p))

print("="*70); print("A. JUDGE-PRIMARY HarmBench ΔASR @512 (classifier) + labels"); print("="*70)
ja = {r["pair_id"]: r for r in j(A/"judge_agreement.json")["per_pair"]}
for p in PAIRS:
    r = ja[p]
    print(f"  {LAB[p]:14s} ΔASR={r['judge_harm_delta']:+.3f} CI[{r['judge_harm_delta_ci_lower']:+.3f},{r['judge_harm_delta_ci_upper']:+.3f}] "
          f"sig={r['judge_harm_delta_significant']} McN_p={r['mcnemar_p_value']:.4f} (b={r['mcnemar_b_base_to_harmful']},c={r['mcnemar_c_harmful_to_safe']}) "
          f"label={r['judge_label']}/{r['evidence_status']}")

print("\n"+"="*70); print("B. per_model judge ASR @512 (absolute, base vs quant)"); print("="*70)
jm = j(A/"judge_agreement.json")["per_model"]
for p in PAIRS:
    b, q = jm.get(f"{p}_base",{}), jm.get(f"{p}_4bit",{})
    print(f"  {LAB[p]:14s} base_ASR={b.get('judge_asr')}  nf4_ASR={q.get('judge_asr')}  kappa(base/nf4)={b.get('cohens_kappa')}/{q.get('cohens_kappa')}")

print("\n"+"="*70); print("C. CAPABILITY + over-refusal deltas @512 (pairwise_deltas)"); print("="*70)
pw = {(r["pair_id"],r["benchmark"]): r for r in j(A/"pairwise_deltas.json")}
for bench in ("mmlu","arc","xstest"):
    print(f" [{bench}]")
    for p in PAIRS:
        r = pw.get((p,bench))
        if r: print(f"   {LAB[p]:14s} Δ={r['absolute_delta']:+.4f} sig={r['delta_significant']}")

print("\n"+"="*70); print("D. FDR consolidation @512 (multiple_comparisons)"); print("="*70)
mc = j(A/"multiple_comparisons.json")
print(f"  family={mc['family_size']} uncorrected_sig={mc['n_uncorrected_significant']} bh_sig={mc['n_bh_significant_q05']}")
print("  BH survivors:")
for s in mc["bh_survivors"]:
    print(f"    {s['pair_id']:14s} {s['metric']:22s} p={s['p_value']:.4f} q={s['bh_q_value']:.4f} Δ={s['delta']:+.4f}")
pa = mc["power_analysis"]
print(f"  power: representative MDE(ASR,80%)={pa['representative_mde_delta_asr_at_median_discordant_rate']} median_discordant={pa['median_discordant_rate']}")

print("\n"+"="*70); print("E. GENLEN 128 vs 512 (genlen_robustness) — the headline compare"); print("="*70)
g = j(A/"genlen_robustness.json")
for p in PAIRS:
    c1 = g["asr_128_classifier"][p]["delta"]; c5 = g["asr_512_classifier"][p]["delta"]; g5 = g["asr_512_gpt4o"][p]["delta"]
    print(f"  {LAB[p]:14s} 128clf={c1:+.3f}  512clf={c5:+.3f}  512gpt4o={g5:+.3f}")
rl = g["response_length"]
print(f"  length: median 128={rl['all_median_128']} 512={rl['all_median_512']} chars; pct_512_over_128ceiling={rl['pct_512_over_ceiling']}%")
k = g["cross_judge_kappa_512"]; print(f"  cross-judge kappa @512: {k['kappa_min']}–{k['kappa_max']}")

print("\n"+"="*70); print("F. PRECISION SWEEP fp16->INT8->NF4 @512 (precision_sweep, judge ASR)"); print("="*70)
ps = j(A/"precision_sweep.json")["per_pair"]  # dict: pair_id -> {aliases, metrics, int8_present, harmbench_asr_int8_mcnemar}
for pid in PAIRS:
    row = ps.get(pid)
    if not row: continue
    m = row["metrics"]["harmbench_asr_judge"]
    mc = row.get("harmbench_asr_int8_mcnemar") or {}
    print(f"  {LAB[pid]:14s} judgeASR fp16={m['fp16']:.3f} int8={m['int8']:.3f} nf4={m['nf4']:.3f} "
          f"Δint8={m['delta_int8_vs_fp16']:+.3f} Δnf4={m['delta_nf4_vs_fp16']:+.3f} shape={m.get('shape')} "
          f"int8_McN_p={mc.get('p_value')}")

print("\n"+"="*70); print("G. MULTI-SEED @512 (sensitivity_multiseed, judge)"); print("="*70)
sm = j(A/"sensitivity_multiseed.json")["per_pair"]
for r in sm:
    d = r.get("judge_delta") or {}
    print(f"  {r['pair_id']:14s} judge mean={d.get('mean'):+.4f} sd={d.get('sd'):.4f} "
          f"range=[{d.get('min'):+.3f},{d.get('max'):+.3f}] n_seeds={d.get('n_seeds')} "
          f"greedy={r.get('greedy_judge_delta'):+.3f} greedy_in_range={r.get('greedy_in_multiseed_range')}")

print("\n"+"="*70); print("H. INT8 dual-judge @512 (from precision_sweep + api_judge) — for §6.15"); print("="*70)
print("  (computed live from sidecars in earlier step: all 5 pairs n.s. under both judges;")
print("   Llama INT8: clf Δ+0.005 p=1.000, gpt4o Δ+0.010 p=0.688; was +0.040 both-sig @128)")
