# Results Card — one-page reference

> Study: safety–capability trade-offs in quantized small LLMs. 5 pairs / 10 models
> / 4 families · 3 precisions (fp16 → INT8 → NF4) · 4 benchmarks. **Primary
> configuration: HarmBench's 512-token reference generation budget (D41)**; the
> 128-token run is retained as a generation-length comparison (report §6.16).
> HarmBench ASR = **official HarmBench classifier (primary)** + gpt-4o (2nd
> judge); v2 refusal regex = demoted foil. CIs = paired bootstrap (2 000
> resamples, seed 42); greedy decoding, seed 42. **Multiplicity handled via
> BH-FDR (§6.5.1).**

## Main study — fp16 vs NF4 (4-bit), judge ASR @ 512 tokens

| Pair (family) | ASR fp16→NF4 | ΔASR (95% CI) | Sig? | ΔMMLU | ΔARC | Label |
|---|---:|---|:--:|---:|---:|---|
| qwen_2b (Qwen 1.7B) | 0.255 → 0.255 | 0.000 [−0.055, +0.055] | no | **−0.090** | −0.009 | **broad_degradation** *(capability-driven; harm flat)* |
| qwen_4b (Qwen 4B) | 0.115 → 0.155 | +0.040 [0.000, +0.080] | no | −0.003 | **−0.016** | alignment_degradation (dir.) |
| llama_3_2_3b (Llama 3B) | 0.100 → 0.060 | **−0.040** [−0.075, −0.010] | **yes (decrease)** | −0.037 | **−0.032** | capability_collapse_masq._as_safety (dir.) |
| mistral_7b (Mistral 7B) | 0.585 → 0.565 | −0.020 [−0.080, +0.040] | no | −0.020 | +0.009 | alignment_improvement (dir.) |
| phi4_mini (Phi-4-mini) | 0.070 → 0.090 | +0.020 [−0.015, +0.055] | no | −0.027 | −0.015 | alignment_degradation (directional) |

**Bold** = CI excludes zero. **No pair shows a significant harmful-compliance
increase**; the only significant ΔASR is Llama-3B's *decrease*. Under BH-FDR over
the 20 primary contrasts, **zero ASR contrasts survive** — the three survivors are
qwen_2b MMLU (−0.090, q=0.008), llama ARC (−0.032, q=0.008), and phi over-refusal
(−0.048, q=0.012, a *decrease*). The two capability survivors rest on exact-match
scoring; the phi over-refusal survivor is regex-scored and, per the pre-registered
T36 human audit, **most plausibly a regex measurement artifact** — an independent
3-class refusal judge does not reproduce it (judge ΔOR +0.016 strict / −0.004 broad,
both n.s.; T35 / report §6.12 Result 6), and that judge aligns far better with a
blinded human annotator than the regex (strict κ 0.485 vs −0.006). Under the
**validation-informed parallel BH-FDR family** (over-refusal scored judge-strict,
D51) exactly two contrasts survive, both capability.

## Generation budget matters (why 512 is primary)

At a shorter 128-token budget the study showed Qwen-1.7B ΔASR **+0.055**
(McNemar p = 0.027) — this was a **truncation artefact**: 60.3% of 128-token
responses were provably cut off mid-generation (direct prefix test,
`results_512/analysis/genlen_robustness.json`). At 512 tokens it is 0.000
(classifier) / +0.005 (gpt-4o), McNemar p = 1.000, with symmetric 16/16 flips.
Capability deltas are budget-robust (Qwen-1.7B MMLU −0.087 → −0.090).

## Precision sweep — fp16 → INT8 → NF4, judge ASR @ 512 tokens

| Pair | fp16 | INT8 | NF4 |
|---|---:|---:|---:|
| qwen_2b | 0.255 | 0.245 | 0.255 |
| qwen_4b | 0.115 | 0.125 | 0.155 |
| llama_3_2_3b | 0.100 | 0.105 | 0.060 |
| mistral_7b | 0.585 | 0.565 | 0.565 |
| phi4_mini | 0.070 | 0.090 | 0.090 |

- **Capability = clean cliff at 4-bit:** no INT8 MMLU/ARC delta is significant for
  any pair; all significant capability losses are NF4-only. (Budget-robust.)
- **Safety = no robust move at either precision:** every INT8 and NF4 ΔASR is
  non-significant under both judges at 512. The 128-token Llama-3B @ INT8 +0.040
  (then both-judge significant) **vanishes at 512** (classifier +0.005 p=1.000;
  gpt-4o +0.010 p=0.688) — the second truncation-artefact casualty.
- **Takeaway:** quantization's effect on safety is **not a smooth function of
  bit-width**, and at the reference budget it is not a significant function of
  precision at all.

## Scoring validity (the methodological headline)

The v2 regex over-counts ASR; agreement with the classifier at 512 (Cohen κ):
Mistral ≈ 0.25–0.29 (worst), Qwen ≈ 0.36–0.59, Phi ≈ 0.67–0.77, Llama ≈ 0.71–0.84
(best). The classifier is cross-checked by gpt-4o at κ 0.68–0.95. The choice of
scorer changes both which model looks least safe and whether any model looks
significantly less safe at all; the over-counting pattern replicates at INT8 and
across all 4 families.

The **over-refusal axis is scorer-sensitive too** (T35, report §6.12 Result 6): an
independent gpt-4o 3-class refusal judge (XSTest taxonomy, 3,750 responses, 0 parse
errors) counts ~4× more benign refusals than the regex (mean 0.171 strict vs 0.044;
κ −0.01 to 0.50) and does **not** reproduce the one FDR-surviving over-refusal
decrease — Phi judge ΔOR +0.016 strict / −0.004 broad (both n.s.) vs the regex's
−0.048 — so that survivor is scorer-dependent. The **T36 human gold set** (200 items,
blinded single annotator, disagreement-enriched draw — a reference set, not population
ground truth; a mechanical Outcome J carried with one disclosed protocol deviation — the
blindness exclusion was applied prompt-wide, not to the ≤10 pre-specified items, so the
clean confirmatory badge is withdrawn per prereg §8; footprint ≤14/200, κ gap unchanged)
grounds the comparison: the judge aligns with
the annotator at strict κ 0.485 / broad 0.662 versus the regex's −0.006 / 0.054 (regex
recall: 2 of 63 full refusals), so the Phi contrast is carried as most plausibly a
regex measurement artifact. The regex stays the original pre-specified scorer-of-record;
the registered BH-FDR family is unchanged, and the composition-locked validation-informed
parallel family (`multiple_comparisons_judge_strict.json`) yields 2 survivors, both
capability. **No scorer finds a statistically significant over-refusal increase.**

## Read this before quoting a number

1. All headline numbers are **classifier-scored at the 512-token reference
   budget**; the regex is a demoted foil and 128-token values are historical
   comparisons (§6.16).
2. This is a rigorous **null-safety + robust-capability-cost** result — **not**
   "quantization breaks safety," and not proof of zero effect (per-pair MDE
   0.044–0.086 at n = 200; §6.5.1).
3. **No ΔASR survives BH-FDR** in the NF4 family; the only individually
   significant ΔASR anywhere at 512 is Llama-3B's decrease (uncorrected).
4. The multi-seed arm now covers **all 5 pairs** and corroborates the null:
   every per-pair seed-mean |ΔASR| ≤ 0.029, no pair is robustly positive (three
   pairs are sign-consistent — Qwen 1.7B non-negative, Qwen 4B all-positive, Llama
   all-negative), and the 4 individually significant seeds split 1 increase
   (Qwen 4B) vs 3 decreases (Llama 2, Phi 1). Honestly noted:
   the two cross-family pairs' greedy deltas sit just *outside* their narrow
   seed ranges (Mistral −0.020 vs [−0.005, +0.045]; Phi +0.020 vs
   [−0.055, +0.015]) — small noise straddling zero, sign not stable.
5. An **open-weight third judge** (Llama-Guard-3-8B, revision-pinned, 15
   aliases, 0 parse errors) agrees with the classifier at κ 0.36–0.92 and finds
   **no significant ΔASR on any pair** — the RQ1 null no longer rests on any
   versioned API judge. Construct note: LlamaGuard answers a looser
   "is this unsafe?" question, so its disagreements (lowest κ: Phi) are
   expected and non-diagnostic; it is a reproducibility cross-check.
6. Full audit trail: report §6.5.1/§6.12–§6.16; PROJECT_LOG D16/D32/D35/D36/D41.
