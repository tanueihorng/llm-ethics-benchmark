# Methods vs. Standards — external-conformance and design-validity audit (2026-07-10)

Scope: the executed protocol was reconstructed **from machine artifacts only** (configs, sbatch, raw.jsonl metadata, sidecars, code), then compared against (a) the report/thesis text and (b) primary external sources fetched during this audit. Epistemic tags: READ = observed in a repo artifact; EXTERNAL = verified against a fetched primary source; INFERRED = derived.

## 1. Protocol as executed (reconstructed, artifact-only) — READ

| Element | Executed value | Source |
|---|---|---|
| Models | 5 pairs, shared model_id per pair, fp16 vs on-the-fly NF4 (+5 INT8 aliases) | configs/tc1_512.yaml; raw.jsonl metadata (60 runs scanned) |
| Quantization | BitsAndBytesConfig(load_in_4bit, nf4, double_quant=True, compute dtype=resolved fp16); INT8 = load_in_8bit | loader.py:40-44; fail-loud guard loader.py:95-117 |
| Decoding | max_new_tokens 512, T=0.0 (do_sample=False), top_p 1.0, max_input 1024, chat template, enable_thinking=False, batch 4, seed 42 | generation_config embedded in all 60 raw files |
| Benchmarks | HarmBench standard n=200 (walledai mirror), XSTest 250 benign (of 450), MMLU 6 subjects n=300, ARC-Challenge test n=1172 | summary.json benchmark_config, all 15 aliases |
| Primary ASR scorer | cais/HarmBench-Llama-2-13b-cls, fp16, official model-card template, greedy | judges/validation.py:83-105 + sidecars |
| Second judge | gpt-4o (unpinned alias), condensed 1-sentence rubric | api_judge sidecars (200 × 15 aliases) |
| Statistics | Paired bootstrap (2000, seed 42) CIs; McNemar exact (significance criterion); BH-FDR q=0.05 over 20 contrasts; MDE/power | compare_quant_pairs.py; multiple_comparisons.py |

Report Ch3/Ch5/Appendix A were diffed against this reconstruction: **core protocol claims are faithful** (decoding, seeds, n's, BnB config, judge model/template, sample distributions, Appendix A == tc1_512.yaml including all 10 revision pins).

## 2. Conformance to external standards — EXTERNAL (all fetched this audit)

| Standard | Study's choice | Conformant? | Evidence |
|---|---|---|---|
| HarmBench generation budget | 512 tokens primary | ✅ Yes; quote "we standardize this parameter to N=512 to allow the metric to converge" verified verbatim, correct context (target generation length) | arXiv:2402.04249v2 |
| HarmBench behaviors | 200 standard behaviors via walledai mirror | ✅ Mirror exact-match 200/200 vs official CSV. ⚠️ P3: pools official val(41)+test(159); official ASR reporting uses test-only. Undisclosed. | official harmbench_behaviors_text_all.csv |
| HarmBench classifier | Official cais 13B cls, fp16, **model-card template** | ⚠️ P2: template matches the HF model-card LLAMA2_CLS_PROMPT byte-for-byte, but the GitHub **harness** variant differs (bracketed headers + 3 extra rules: caveats-count, clearly-tries-counts, English). Fork undisclosed. Matched-pair ΔASR immune (both members judged identically); absolute ASRs not harness-comparable. | HF model card vs raw eval_utils.py |
| HarmBench judge input | Full saved response (≤512 target tokens) | ⚠️ P3: official pipeline re-clips to 512 *classifier-tokenizer* tokens before judging. Practically negligible at this budget. | evaluate_completions.py --num_tokens |
| HarmBench threat model | DirectRequest (no attacks) | ✅ Conformant with the official DirectRequest baseline, and disclosed at 5+ locations incl. abstract | official baseline code |
| XSTest artifact | Exact 450/450 canonical prompts, 250 safe used | ✅ artifact-faithful | github.com/paul-rottger/xstest |
| XSTest scoring | v2 refusal regex | ⚠️ P2: paper's primary protocol is human annotation (with GPT-4 assist); string-matching appears there only as an imperfect baseline. Study discloses regex limitation but the deviation from the benchmark's sanctioned scoring is not framed as such. No task-specific human validation of the over-refusal scorer (one of 3 BH survivors rests on it). | Röttger et al., NAACL 2024 |
| MMLU protocol | Zero-shot, chat-templated, 6 subjects, n=300 | ⚠️ Official convention is 5-shot over 57 subjects; study **disclosly** anchors zero-shot (report line ~816) — absolute accuracies not leaderboard-comparable; deltas internally valid. Disclosure present in report; absent in thesis/interim (P3). | arXiv:2009.03300 |
| ARC-Challenge | allenai/ai2_arc test split n=1172, zero-shot MCQ | ✅ | plugin arc.py:38-39 + artifact n |
| NF4 | QLoRA NF4 + double-quant (the QLoRA object) | ✅ | loader config vs arXiv:2305.14314 |
| LLM.int8 | load_in_8bit; description in report conformant | ✅ | arXiv:2208.07339 |
| κ interpretation | "moderate"/"negligible" labels | ⚠️ P3: verbal conventions uncited (no Landis & Koch); two labels generous vs convention |

## 3. Design-validity adjudication (can the RQ be answered?) — verdict: NO FATAL THREAT

Threats adjudicated MATERIAL-BUT-DISCLOSED (with unusually thorough hedging, verified at source):
- **Power/MDE**: safety axis underpowered for small effects (MDE 0.044–0.086; qwen_2b post-hoc power ≈0); the report words every null as a detection floor, not absence. ✅ disclosed
- **Single-seed greedy primary** + multiseed on only 3/5 pairs — disclosed; artifact matches every published multiseed claim.
- **DirectRequest-only** — disclosed at 5 locations including the abstract.
- **bnb-NF4-only generalizability; 6-subject MMLU** — scoped explicitly; capability double-anchored via ARC.
- **Judge validity chain** — κ 0.59 vs single author-annotator supports "closer to human than regex", NOT "validated oracle"; report wording mostly stays on the right side; see P1 findings on the gold set (truncated annotator view; unweighted stratified κ).

Localized overreach beyond disclosure (P2):
- **Causal language**: classify_pair_change is co-occurrence thresholding, but Ch10/Table 3.4 assert causal verdicts ("is a capability side-effect, not improved alignment").
- **Self-continuation artifact**: "cancels in every matched-pair delta" asserted, never measured.
- **Run-level variance tension**: "no within-condition stochastic variance" vs the study's own 9.2% cross-run greedy divergence; plus one pair member (qwen_2b_base @512 harmbench) is a smoke+resume two-job composite contradicting "every delta computed within a single run" (bounded: inside CI, p=1.000 null safe).
- **Contamination** never discussed (largely neutralized by matched pairs, still worth one sentence).

## 4. Research gap and contribution — verdict: GAP STANDS, four supporting claims overreach

- The hedged conjunction claim ("No prior study, to the author's knowledge, combines matched-pair loading + judge-validated scorer + over-refusal axis + capability anchor + multi-precision on compact models") was **adversarially hunted and not falsified** (targeted searches: quantization+XSTest, matched-pair fp16-vs-4bit safety, compact-model safety, truncation/ASR artefacts). No unhedged "first/only/novel" claim exists in any document (grep-verified).
- **P2 — closest prior work uncited**: Hong et al., *Decoding Compressed Trust* (ICML 2024; arXiv:2403.15447) jointly evaluates trustworthiness + capability of quantized LLMs and already argues risk "cannot be uncovered by looking at benign performance alone". Cited in none of the three documents (grep 0 hits).
- **P2 — thesis gap clause** "almost none trace the effect across more than one precision or method" contradicted by the works the same chapter cites (Kharinaev: 66 variants/6 methods; HarmLevelBench: AWQ+GPTQ; Q-resafe: 4 methods).
- **P2 — §1.5 contribution (b)** claims provenance confounds "affect most published comparisons" — unsupported; the cited studies quantize identical weights themselves.
- **P2 — scorer-validity contribution** ("consequences beyond this study") does not engage StrongREJECT (Souly et al. 2024), which already established scorer choice drives safety-eval conclusions.
- Positioning vs Egashira (adversarial quantization) verified accurate and modest. Citation-description accuracy verified against abstracts for all six quantization-behaviour references (one nuance: Kharinaev DOES validate its judge against 4,200 human labels — report gap component (ii) implies judge validation is absent from prior quantization-safety work; overbroad).

## 5. Bottom line (methods)

The experimental design **can answer the stated research question as worded**, the protocol is faithfully executed and substantially conformant with benchmark standards, and the disclosure discipline is far above typical FYP level. The defects are: two scorer-conformance deviations needing a disclosure sentence each (classifier template variant; XSTest regex vs sanctioned scoring), a handful of overreaching supporting claims in the gap/contribution framing, causal phrasing beyond the labeling rule's semantics, and the human-gold-set execution flaws (2000-char annotator view; unweighted stratified κ) that cap what "human-grounded" may claim.
