# T44 Phase 1b — Unlocked-claims coverage map

**Method:** 5 extraction agents + 1 lock-inventory agent (Opus @high) in parallel; 4 diff judges (Opus @xhigh); the 5th diff (thesis) ran INLINE by the Fable orchestrator after the workflow agent died on the user's session limit — recorded as a model-policy deviation. Rule applied: uncertain coverage = UNLOCKED; a lock covering a different instance/surface of the same number does NOT cover this one.

**Inputs:** 86 locks (verify_report_claims.py + claim_registry) × 328 extracted claims (45/79/52 report chunks, 79 thesis, 73 interim).

**Totals:** 18 high · 85+~18 medium · 73+~14 low unlocked claims.

## The four structural patterns (what Phase 2+ should act on)

1. **§6.14 mechanism probe: ZERO lock coverage** — Wilcoxon verdicts, AUC 0.76/0.64, 50/42 flip counts, per-pair margin shifts all unchecked (FS-2).
2. **§6.15/§6.16 half-locked** — INT8 ASR cells are locked but the INT8 MMLU/ARC 'capability-lossless' verdict is not; Table 6.5's non-Qwen 128-column cells unlocked (FS-3).
3. **The interim surface is systematically under-locked** — 9 high-severity claims (BH survivors incl. q-values, RQ block, precision table cells, Table 6.1 κs, validation-informed 2-survivor verdict) duplicate report/thesis numbers, but the gates read those builders, not the interim (FS-4).
4. **Duplicate-instance blindness** — presence gates are whole-file string searches, so a drifted duplicate (e.g. an abstract restating a §6 number) passes while any pinned instance survives (FS-6). Most mediums are this class.

## High-severity unlocked claims (Phase 2 work list)

- **[interim]** build_fyp_interim.js:133 (abstract) — At 512 tokens no pair shows a significant increase in harmful compliance; not one HarmBench contrast survives Benjamini-Hochberg correction.
  - why: The RQ1/BH null verdict is not gated by any build_fyp_interim.js lock; the 'Not one HarmBench ASR contrast survives' gate reads report/thesis, not the interim file.
- **[interim]** build_fyp_interim.js:133 (abstract) — Effects surviving multiplicity are capability losses and one benign over-refusal change; INT8 essentially free; 128-token INT8-specific safety increase vanishes at 512.
  - why: This synthesis verdict (incl. the Phi over-refusal survivor and INT8 vanishing) has no interim-targeted lock; only 'classifier delta +0.005' is gated here.
- **[interim]** build_fyp_interim.js:283-286 (ch6.1 Table 6.1) — Judge-vs-regex kappa by family: Qwen 0.36-0.59 (regex 0.595 vs judge 0.255 on 1.7B base); Mistral 0.25-0.29 (regex 0.890 vs judge 0.565 on 4-bit); Llama 0.71-0.84; Phi 0.67-0.77.
  - why: The scorer-validation family kappas and 0.595/0.255/0.890/0.565 values are gated only in report/thesis ja locks; no interim lock reads this table.
- **[interim]** build_fyp_interim.js:297 (ch6.2) — Only exact-test-significant over-refusal delta is Phi-4-mini's decrease dOR = -0.048, CI [-0.076, -0.020], regex-scored; independent judge dOR +0.016 strict / -0.004 broad.
  - why: The -0.048 BH survivor with CI and the strict/broad judge reversal are gated only in report/thesis; interim lock #1 does not include -0.048 or the judge deltas.
- **[interim]** build_fyp_interim.js:307-311 (ch6.6 Table 6.3) — Precision sweep judge ASR fp16/INT8/NF4: qwen_2b 0.255/0.245/0.255; qwen_4b 0.115/0.125/0.155; llama 0.100/0.105/0.060; mistral 0.585/0.565/0.565; phi 0.070/0.090/0.090.
  - why: These 15 result cells are hardcoded (not registry-rendered) and the matching precision_sweep lock targets build_fyp_thesis_v4.js, not the interim surface.
- **[interim]** build_fyp_interim.js:317 (ch6.7) — BH-FDR (q<0.05) over twenty contrasts: exactly three survive, none harmful-compliance: Qwen MMLU (-0.090, q=0.008), Llama ARC (-0.032, q=0.008), Phi over-refusal (-0.048, q=0.012); not one HarmBench ASR contrast survives.
  - why: The three-survivor count, q-values, -0.048, and 'not one HarmBench survives' verdict are gated only in report/thesis mc locks; interim gates only -0.090*/-0.032*.
- **[interim]** build_fyp_interim.js:317 (ch6.7) — Validation-informed parallel correction leaves two contrasts surviving, both capability: Qwen MMLU (q=0.008), Llama ARC (q=0.008).
  - why: multiple_comparisons_judge_strict n_bh==2 is gated only in report/thesis locks; no interim lock reads this validation-informed verdict.
- **[interim]** build_fyp_interim.js:322 (ch7) — RQ answers RQ1-RQ5 (no significant harmful-compliance increase; no significant over-refusal increase; capability loss survives correction; smallest model most sensitive; power-bounded safety null with capability cliff).
  - why: The five headline RQ verdicts have no build_fyp_interim.js-targeted lock.
- **[interim]** build_fyp_interim.js:358 (ch10) — None of five pairs shows a significant increase in harmful compliance at NF4; not one ASR contrast survives correction; only individually significant move is a decrease; surviving effects are capability losses and one benign over-refusal change.
  - why: The conclusion verdict is not gated by any interim-targeted lock; matching mc gates read report/thesis files.
- **[report]** build_fyp_report_v5.js:954-960 (§6.16 Table 6.5) — 128-vs-512 ΔASR table: @128-classifier column Qwen3-4B +0.025, Llama 0.000, Mistral -0.040, Phi 0.000; and @512-gpt-4o column Qwen3-4B +0.040, Mistral -0.005, Phi +0.020.
  - why: Only the Qwen3-1.7B row (+0.055@128 #17, 0.000@512 #5, +0.005 gpt-4o #18) and Llama gpt-4o -0.035 (#18) are locked; the rest of the @128 classifier column and the qwen_4b/mistral/phi gpt-4o cells have no lock, yet this is the central token-budget robustness table.
- **[report]** build_fyp_report_v5.js:942 (§6.15) — Significance verdict: no INT8 MMLU/ARC delta is significant for any pair (INT8 is capability-lossless); every significant capability loss appears only at four-bit.
  - why: precision_sweep locks (#41/#42) cover INT8 HarmBench only and #43 covers XSTest only; no lock evaluates INT8 MMLU/ARC significance, so this headline §6.15 verdict is unchecked.
- **[report]** build_fyp_report_v5.js:936 (§6.14) — Paired margin shift Δm significant every pair (Wilcoxon p<0.001, n=200); comply-ward Qwen1.7B +1.2, Qwen4B +4.0; refuse-ward Llama -1.1, Phi -0.3; Cohen dz -0.74 to +1.8; Qwen4B baseline margin median 13.0 vs 1.7B 3.8; flips 9 vs 21.
  - why: No lock references §6.14 mechanistic analysis at all; these are significance verdicts and result numbers with zero coverage.
- **[report]** build_fyp_report_v5.js:935 (§6.14) — Central mechanistic result: pooled margin-flip AUC 0.76, z-scored 0.64, near-chance Qwen1.7B 0.61 / Mistral 0.54; of 92 flips 50 harmful-ward / 42 safe-ward; four valid-proxy families lean ~2:1 harmful-ward (30 vs 14).
  - why: Entire §6.14 is outside the lock inventory; these AUC/flip figures carry the mechanistic conclusion and are unchecked.
- **[thesis]** build_fyp_thesis_v4.js:278 (§6.1) — Scorer-relocation verdicts: at 128 regex placed significant increases on Qwen3-4B and Mistral, classifier on Qwen3-1.7B alone; at 512 the classifier removes every significant increase.
  - why: Significance-relocation verdict unpinned on the thesis surface (report-side locks don't read this builder; only the structural 128-scoping guard applies).
- **[thesis]** build_fyp_thesis_v4.js:303 (§6.6) — INT8 capability verdict: every INT8 MMLU/ARC point within ~1.3pp of fp16, below NF4 losses of 3–9 points.
  - why: Same gap as report §6.15: no lock evaluates INT8 MMLU/ARC anywhere; thesis instance also unpinned.
- **[thesis]** build_fyp_thesis_v4.js:321 (ch7) — RQ1–RQ5 answer block (no significant harmful-compliance increase; no over-refusal rise under any scorer; capability-only survivors; power-bounded null).
  - why: The verdict block restates locked facts but no lock reads these sentences; drift here would pass.
- **[thesis]** build_fyp_thesis_v4.js:325 (§7.2) — LlamaGuard block: κ 0.36–0.92 per alias (Llama 0.89/0.92 high, Phi 0.36/0.38 low), every per-pair ΔASR n.s., largest Qwen3-1.7B −0.050 p=0.076.
  - why: llamaguard lock targets the report builder only; thesis instance unpinned.
- **[thesis]** build_fyp_thesis_v4.js:326 (§7.2) — XSTest human gold: strict κ 0.485 vs −0.006, broad 0.662 vs 0.054; regex 2/63 full refusals; deviation footprint ≤14/200, gap +0.51.
  - why: xstest_human_validation locks target the report builder; thesis instances unpinned.

## Medium (compact; full text in phase1_coverage_data.json)

- [report] build_fyp_report_v5.js:330 (Abstract): Refusal regex's harmful set very nearly contains the classifier's — disagreement is one-directional 
- [report] build_fyp_report_v5.js:331 (Abstract): Power analysis bounds minimum detectable ΔASR ≈ 0.04–0.09 per pair at n=200.
- [report] build_fyp_report_v5.js:331 (Abstract): Five pairs across four families (Qwen, Llama, Mistral, Phi; 1.7–7.2 B) at three precisions (fp16, IN
- [report] build_fyp_report_v5.js:404 (Ch1 Scope): All inference uses deterministic greedy decoding with temperature 0.0 and top-p 1.0.
- [report] build_fyp_report_v5.js:515-521 (Table 3.3): Decoding controls: temp 0.0, top_p 1.0, max_new_tokens 512, repetition_penalty 1.0, max_input_tokens
- [report] build_fyp_report_v5.js:544-549 (Table 3.4): Interpretation-label thresholds (alignment_degradation ΔASR≥+0.02 & ΔMMLU>−0.03; alignment_improveme
- [report] build_fyp_report_v5.js:503 (Ch3.4): MMLU six subjects (business_ethics, clinical_knowledge, college_biology, high_school_world_history, 
- [report] build_fyp_report_v5.js:506 (Ch3.5): Primary HarmBench ASR scorer is the official classifier cais/HarmBench-Llama-2-13b-cls, full precisi
- [report] build_fyp_report_v5.js:428 (Ch2): TrustLLM evaluates sixteen LLMs over thirty datasets across six trust dimensions; SafetyBench tests 
- [report] build_fyp_report_v5.js:443 (Ch2): Kharinaev et al. evaluate 66 quantized variants and find quantization can degrade safety alignment.
- [report] build_fyp_report_v5.js:443 (Ch2): Hong et al. report 4-bit broadly preserves trustworthiness across five compression methods and eight
- [interim] build_fyp_interim.js:133 (abstract): Judge-versus-regex agreement is family-dependent, Cohen's kappa 0.25 to 0.84.
- [interim] build_fyp_interim.js:133 (abstract): Qwen3-1.7B 128-token ASR delta +0.055 dissolves to 0.000 under classifier at 512, McNemar p=1.000 (s
- [interim] build_fyp_interim.js:177 (ch1.2): HarmBench authors show token-budget swings of up to 30 percent and standardize the budget to 512 tok
- [interim] build_fyp_interim.js:222-225 (ch3.3 Table 3.2): Benchmark sample budgets: HarmBench n=200; XSTest n=250; MMLU n=300; ARC-Challenge n=1172.
- [interim] build_fyp_interim.js:228 (ch3.3): ASR equals HarmBench's DirectRequest baseline, the weakest threat model; no GCG, PAIR, or AutoDAN at
- [interim] build_fyp_interim.js:243 (ch3.6): Greedy temp 0.0 seed 42; paired-bootstrap 95% CIs, 2,000 resamples; McNemar exact; BH-FDR over twent
- [interim] build_fyp_interim.js:262-272 (ch5 Table 5.1): Config summary incl. Llama-Guard-3-8B third judge revision-pinned, bootstrap 2000 + McNemar + BH-FDR
- [interim] build_fyp_interim.js:275 (ch5): Zero classifier parse errors over 3,000 judged generations at the 512-token budget.
- [interim] build_fyp_interim.js:281 (ch6.1): At 128 tokens, regex placed significant increases on Qwen3-4B and Mistral, classifier on Qwen3-1.7B 
- [interim] build_fyp_interim.js:281 (ch6.1): Regex over-flagged 101 responses, classifier 18, human 37 (of 200).
- [interim] build_fyp_interim.js:289 (ch6.1 Figure 1): Eight of ten points lie below the diagonal; two marginal exceptions Llama base and Phi 4-bit.
- [interim] build_fyp_interim.js:291 (ch6.2): Only ASR delta with CI excluding zero is Llama -0.040 [-0.075, -0.010], McNemar p=0.021; does not su
- [interim] build_fyp_interim.js:297 (ch6.2): Qwen3-1.7B over-refusal decrease -0.024, CI [-0.048, 0.000], McNemar p=0.109, non-significant.
- [interim] build_fyp_interim.js:297 (ch6.2): Qwen3-1.7B ASR delta exactly zero at 512; thirty-two discordant prompts split sixteen against sixtee
- [interim] build_fyp_interim.js:299 (ch6.3): 60.3 percent of 2,000 paired responses cut off, 30.5 percent stop naturally, 9.2 percent diverge (ex
- [interim] build_fyp_interim.js:299 (ch6.3): Truncation is family-heterogeneous: 93.5-98.0 percent for Mistral, 3.0-4.0 percent for Llama.
- [interim] build_fyp_interim.js:299 (ch6.3): Qwen3-1.7B ASR +0.055 (McNemar p=0.027) at 128 to exactly 0.000 (p=1.000) under classifier at 512, s
- [interim] build_fyp_interim.js:299 (ch6.3): Five-seed decoding arm: per-seed dASR mean +0.013, range [0.000, +0.035], no seed significant, all f
- [interim] build_fyp_interim.js:299 (ch6.3): Five-pair extended multi-seed: Qwen3-4B mean +0.029 (1/5 sig); Llama -0.024 (2/5 decreases); Mistral
- [interim] build_fyp_interim.js:299 (ch6.3): Greedy inside seed range for three original pairs; outside with sign flip for Mistral (greedy -0.020
- [interim] build_fyp_interim.js:301 (ch6.4): Capability point estimates non-positive under NF4 except Mistral +0.9 percentage points on ARC (n.s.
- [interim] build_fyp_interim.js:301 (ch6.4): Significant capability loss also Qwen3-4B ARC -0.016; Qwen3-1.7B ARC -0.009 not significant.
- [interim] build_fyp_interim.js:301 (ch6.4): Llama pair labelled capability collapse masquerading as safety rather than alignment improvement.
- [interim] build_fyp_interim.js:303 (ch6.5): Qwen3-1.7B pair's thirty-two discordant prompts split exactly sixteen to sixteen (symmetric flips).
- [interim] build_fyp_interim.js:305 (ch6.6): INT8 MMLU/ARC within ~1.3 percentage points of fp16, below NF4 losses of three to nine points; signi
- [interim] build_fyp_interim.js:305 (ch6.6): 128-era Llama INT8-specific increase +0.040 does not replicate at 512: classifier +0.005 (p=1.000), 
- [interim] build_fyp_interim.js:316 (ch6.7): CIs about +/-0.05; MDE for 80% power about 0.06 for a representative pair; per-pair 0.04-0.09 at n=2
- [interim] build_fyp_interim.js:326 (ch7.2): Cross-judge agreement Cohen's kappa 0.68-0.95 across all ten models at the reference budget.
- [interim] build_fyp_interim.js:326 (ch7.2): gpt-4o three-class judge re-scored all 3,750 benign responses, zero parse failures; Phi dOR reverses
- [interim] build_fyp_interim.js:326 (ch7.2): 200-item blinded human audit (mechanical Outcome J, one disclosed deviation): judge strict kappa 0.4
- [interim] build_fyp_interim.js:334 (ch8): 200 HarmBench prompts give ~+/-0.05 CIs; MDE at 80% power ~0.06 representative (0.04-0.09 across pai
- [interim] build_fyp_interim.js:336 (ch8): Judge grounding: classifier kappa 0.59 vs regex 0.11; over-refusal 200-item blinded audit judge stri
- [interim] build_fyp_interim.js:338 (ch8): 128-era signals: Qwen3-1.7B NF4 +0.055 via truncation; Llama-3B INT8 +0.040 via cross-run divergence
- [interim] build_fyp_interim.js:346 (ch9.1): Llama-Guard-3-8B across fifteen aliases: Cohen's kappa 0.36-0.92 against classifier over ten base/NF
- [interim] build_fyp_interim.js:364-388 (References): 25 numbered IEEE references; key citations LLM.int8() arXiv:2208.07339, QLoRA arXiv:2305.14314, Harm
- [interim] build_fyp_interim.js:393 (Appendix A): Checksum manifest pins 300 immutable raw files across both results trees and the multi-seed arm.
- [report] build_fyp_report_v5.js:934 (§6.14): First-token refusal-margin AUC 0.86-0.90 for Qwen1.7B/Qwen4B/Llama/Phi vs 0.69 for Mistral; well-dis
- [report] build_fyp_report_v5.js:937 (§6.14): Entropy confound control: Qwen1.7B neutral-token entropy rises ~10x more than harmful (+0.21 vs +0.0
- [report] build_fyp_report_v5.js:964 (§6.16): 128-token capability deltas Qwen1.7B ΔMMLU -0.087, Llama ΔARC -0.028, Qwen4B ΔARC -0.021 (paired wit
- [report] build_fyp_report_v5.js:813,833,861,927 / §6.11.5 (interpretation labels): Pair labels: Qwen1.7B broad_degradation (confirmed), Qwen4B alignment_degradation (directional), Lla
- [report] build_fyp_report_v5.js:784,805,833,907 (recurring): Qwen3-4B HarmBench McNemar p=0.096 (the closest-to-significant NF4 ASR increase; recurs §6.2, §6.5, 
- [report] build_fyp_report_v5.js:782,833,848,879 (recurring): Qwen3-4B ΔMMLU -0.003 (-0.3 pp) with CI [-0.040, +0.033], n.s.
- [report] build_fyp_report_v5.js:802,805,852,879 (recurring): Qwen3-1.7B lenient-parser ΔARC -0.009 (n.s.), used to argue the ARC ordering reverses vs the strict 
- [report] build_fyp_report_v5.js:826-827 (§6.6.2): Qwen1.7B per-category HarmBench ΔASR: misinfo +0.18, harassment +0.11, illegal +0.03, generic +0.05,
- [report] build_fyp_report_v5.js:890-901 (§6.12 Table 6.3): Judge-vs-v2 agreement percentage column (65.0/68.5/85.0/85.0/95.0/98.0/68.0/66.5/95.5/96.5) plus the
- [report] build_fyp_report_v5.js:909 (§6.12 Result 5): Human validation derived rates: classifier flags confirmed 61% vs regex 23%; the 200-item set is '12
- [report] build_fyp_report_v5.js:771 (§6.1.1): v1->v2 regex ΔASR shifts: Qwen1.7B -0.120->-0.025, Qwen4B -0.045->+0.065, Llama +0.030->0.000; MMLU 
- [report] build_fyp_report_v5.js:805 (§6.5): Parser fallback accuracies: MMLU fallback 46% vs leading-letter 75%, ARC fallback 66%; fp16 MMLU fal
- [report] build_fyp_report_v5.js:846 (RQ2) / :929 (§6.13): Phi-4-mini over-refusal baseline 0.128->0.080 with McNemar p=0.002; Mistral over-refusal 0.000 (0.00
- [report] build_fyp_report_v5.js:759 (§6.1) / :923 (Table 6.4 caption): Bootstrap methodology: paired bootstrap 2,000 resamples (seed 42) for main CIs and 10,000 resamples 
- [report] build_fyp_report_v5.js:823 (§6.6.1): Per-pair minimum detectable ΔASR 0.079 for Qwen1.7B at n=200; five pairs span MDE 0.044-0.086.
- [report] build_fyp_report_v5.js:834-835 (§6.8): Qwen3-4B MMLU per-subject deltas: business ethics -4.0 pp, college biology -3.0 pp, macroeconomics +
- [report] build_fyp_report_v5.js:770 (§6.1.1): v2 scorer provenance: version tag v2_expanded_refusal_patterns_2026-05-28; v1 used 14 refusal patter
- [report] build_fyp_report_v5.js:672-687 (§5.1 Table 5.1): TC1 hardware/policy: partition UGGPU-TC1, V100 32GB x3/node, 20 CPU cores/user, 64GB mem/user, wallt
- [report] build_fyp_report_v5.js:694-696 (§5.2 Table 5.2): Software stack: Python 3.11.15, torch 2.11.0+cu130, transformers 5.5.0.
- [report] build_fyp_report_v5.js:1074 (References [16]): Decoding Compressed Trust ref metadata: ICML PMLR vol. 235, 2024, arXiv:2403.15447
- [report] build_fyp_report_v5.js:1078 (References [20]): McNemar 1947 ref: Psychometrika vol. 12, no. 2, pp. 153-157, 1947
- [report] build_fyp_report_v5.js:1081 (References [23]): Benjamini & Hochberg BH-FDR ref: JRSS Series B vol. 57, no. 1, pp. 289-300, 1995
- [report] build_fyp_report_v5.js:1082 (References [24]): Cohen kappa ref: Educational and Psychological Measurement vol. 20, no. 1, pp. 37-46, 1960
- [report] build_fyp_report_v5.js:1083 (References [25]): Landis & Koch ref: Biometrics vol. 33, no. 1, pp. 159-174, 1977
- [report] build_fyp_report_v5.js:1101-1108 (App A YAML qwen_2b_base): model_id Qwen/Qwen3-1.7B, revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, size_b 1.7, benchmarks 
- [report] build_fyp_report_v5.js:1122-1130 (App A YAML qwen_4b_base): model_id Qwen/Qwen3-4B, revision 1cfa9a7208912126459214e8b04321603b3df60c, size_b 4.0
- [report] build_fyp_report_v5.js:1143-1152 (App A YAML llama_3_2_3b_base): model_id meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95, size_b
- [report] build_fyp_report_v5.js:1168-1177 (App A YAML mistral_7b_base): model_id mistralai/Mistral-7B-Instruct-v0.3, revision c170c708c41dac9275d15a8fff4eca08d52bab71, size
- [report] build_fyp_report_v5.js:1190-1200 (App A YAML phi4_mini_base): model_id microsoft/Phi-4-mini-instruct, revision cfbefacb99257ffa30c83adab238a50856ac3083, size_b 3.
- [report] build_fyp_report_v5.js:1214-1220 (App A YAML decoding): top_p 1.0, repetition_penalty 1.0, max_input_tokens 1024, use_chat_template true
- [report] build_fyp_report_v5.js:1238-1250 (App A YAML mmlu): mmlu: dataset cais/mmlu, split test, max_samples 300, batch_size 4, 6 subjects (business_ethics, cli
- [report] build_fyp_report_v5.js:1433 (App G 2026-07-19 13:30): ~0.06 MDE scoped as representative/median-pair; per-pair MDEs span 0.044-0.086
- [report] build_fyp_report_v5.js:1435 (App G 2026-07-14 12:00) [strict-parser row]: 4-bit ARC primary delta -0.009; MMLU lenient fallback 48.7% and fp16 fallbacks 2.5%/3.3%
- [report] build_fyp_report_v5.js:1436 (App G 2026-07-12 18:00): XSTest judge is gpt-4o snapshot gpt-4o-2024-08-06
- [report] build_fyp_report_v5.js:1437 (App G 2026-07-11 16:00): HarmBench ASR per-pair minimum detectable effects about 0.04-0.09 at 200 items
- [report] build_fyp_report_v5.js:1441 (App G 2026-06-15 18:00) [T26 results row]: T26 128-era: Mistral judge dASR -0.040 (v2 proxy +0.055; second-judge kappa 0.60-0.63), Phi dASR 0.0
- [report] build_fyp_report_v5.js:1442 (App G 2026-06-14 15:45): T18 multi-seed Qwen 1.7B dASR mean +0.024 across 5 seeds; +0.055 rising in 5/6 harm categories; T22 
- [report] build_fyp_report_v5.js:1443 (App G 2026-06-06): D16 judge validation: job 61047, n=200x6, 0 parse errors; regex over-counts Qwen kappa~0.19-0.37, Ll

(+~18 thesis mediums, see data JSON note) · Lows: 73 workflow + ~14 thesis (cover/config/history trivia; in data JSON).

## Blind spots of this map
- Extraction recall is agent-limited: a claim the extractors missed is invisible here (mitigation: Phase 8's basis-free pass).
- Humanized builders and LaTeX mirrors were NOT inventoried (they mirror the masters; Phase 4 owns cross-surface consistency).
- Deck/README/RESULTS_CARD claims not inventoried (partially locked via outer-surface checks; Phase 4).
