## Full FYP report (docx-js line): v2 → v3

*From:* `docs/archive/FYP_Report_2026-06-26_v2.docx`  
*To:* `docs/archive/FYP_Report_2026-06-26_v3.docx`  

**Paragraph/body blocks:** +0 added · −0 removed · ~133 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~4 · front matter blocks 35→35

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~3 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Compact instruction-tuned language models are almost always deployed quantized, yet their safety is…
  - ❌ The design loads the baseline and quantized members of each pair from identical weights, applying q…
  - ❌ Under the classifier, four-bit NF4 never significantly reduces harmful compliance and significantly…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix C: Pydantic Configuration Schema Summary | 0 | 0 | 18 | 0 |
| Appendix D: Test Inventory | 0 | 0 | 2 | 0 |
| Appendix E: Repository Layout | 0 | 0 | 2 | 0 |
| Appendix G: Document Revision History | 0 | 0 | 3 | 0 |
| Appendix H: Scorer Incident and Future-Session Handoff | 0 | 0 | 3 | 0 |
| Chapter 10: Conclusion | 0 | 0 | 3 | 0 |
| Chapter 1: Introduction | 0 | 0 | 5 | 0 |
| Chapter 2: Literature Review | 0 | 0 | 4 | 0 |
| Chapter 3: Methodology | 0 | 0 | 8 | 0 |
| Chapter 4: System Design and Implementation | 0 | 0 | 3 | 0 |
| Chapter 5: Experimental Setup | 0 | 0 | 2 | 0 |
| Chapter 6: Results and Analysis | 0 | 0 | 65 | 0 |
| Chapter 7: Discussion and Threats to Validity | 0 | 0 | 5 | 0 |
| Chapter 8: Limitations | 0 | 0 | 2 | 0 |
| Chapter 9: Future Work | 0 | 0 | 5 | 0 |

### Edited paragraphs (133; change window shown)

<details><summary>show</summary>

- (front matter): “… release. Whether quantization shifts safety alignment [−— ][+(][− —][+)] is unresolved …”
- (front matter): “… refusal-counting regex over-counts harmful compliance [−— ][+(][− —][+)] and replacing …”
- (front matter): “…ochberg correction over the family of primary contrasts[− —][+,][−,][+rather tha][−ot] s…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “… model weights, any operation that alters those weights[− —][+,][− —][+,] has the potent…”
- Chapter 1: Introduction § 1.2 Problem Statement: “…del, measured on benign prompts, may also refuse those [−— ][+(][− —][+)] and on capabil…”
- Chapter 1: Introduction § 1.5 Contributions: “… refusal-counting regex over-counts harmful compliance [−— ][+(][− —][+)] and replacing …”
- Chapter 1: Introduction § 1.5 Contributions: “…production Python and 1,800 lines of automated tests; 3[−06][+29][−thre][+fiv]e files) w…”
- Chapter 1: Introduction § 1.5 Contributions: “…n safety change, and the effect is not bit-width-graded[+.][−—][+These results are] supp…”
- Chapter 2: Literature Review § 2.1 Large-Scale Ethical Benchmarking of LLMs: “…, they typically report each safety axis independently [−— ][+(][− —][+)] without explic…”
- Chapter 2: Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “…g benign prompts that incidentally resemble unsafe ones[− —][+,] a failure mode known as…”
- Chapter 2: Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “… curated unsafe prompts and measure attack success rate[− —][+,][− —][+,] the fraction o…”
- Chapter 2: Literature Review § 2.4 Quantization and Behavioural Effects: “… et al. [16] show that widely used BitsAndBytes schemes[− —][+,][− —][+,][−Importantly, …”
- Chapter 3: Methodology § 3.1 Experimental Design: “…BitsAndBytes NF4 quantization on the fly at load time. [−Crucially, b][+B]oth pair membe…”
- Chapter 3: Methodology § 3.3 Quantization Approach: “…e dtype to follow the resolved torch_dtype of the model[− —][+,] that is, float16 on CUD…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “…y dimension is measured by two complementary benchmarks[− —][+:][− —][+,] so that the al…”
- Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier: “…l HarmBench classifier (cais/HarmBench-Llama-2-13b-cls)[− —][+,][−— ][+(][− —][+)][+.][−…”
- Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier: “The trade-off of deterministic parsing [−— ][+(][− —][+)] is exactly why the offic…”
- Chapter 3: Methodology § 3.7 Metrics and Pairwise Analysis: “…test is McNemar's exact test on the discordant prompts [−— ][+(][− —][+)] rather than an…”
- Chapter 3: Methodology § 3.8 Interpretation Framework: “…ance worsens beyond threshold with capability preserved[− —][+:][− —][+:][− —][+:][−,][+…”
- Chapter 3: Methodology § 3.8 Interpretation Framework: “…t point estimate from being read as a confirmed finding[+.][−— f][+F][+,] the Qwen 4B pa…”
- Chapter 4: System Design and Implementation § 4.3 Model Loading and Quantization Path: “…ing if the flag is false despite a quantization request[+.][−— t][+T]his protects agains…”
- Chapter 4: System Design and Implementation § 4.5 Benchmark Plugins: “…nch and XSTest see exactly the same notion of "refusal"[− —][+,] a non-trivial property …”
- Chapter 4: System Design and Implementation § 4.10 Testing: “The repository ships with a verification suite of 3[−06][+29][−thre][+fiv][−06][+29] test…”
- Chapter 5: Experimental Setup § 5.2 Software Environment: “…nts.txt) \|\| pytest \| (installed via requirements.txt; 3[−06][+29] tests passing)”
- Chapter 5: Experimental Setup § 5.6 Run Plan: “…th group_by=model, the framework emits six sbatch files[− —][+,][− —][+;] see §6.4.1.) T…”
- Chapter 6: Results and Analysis: “…extension subsequently added two further matched pairs [−— ][+(][− —][+)] run on the sam…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “… zero under the official classifier, and it is positive[− —][+:][− —][+,][+.][−— t][+T][…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…figures (produced by the v2 expanded refusal classifier[− —][+,][− —][+,] including "I c…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…2: Qwen 1.7B ΔASR moved from −0.120 (v1) to −0.025 (v2)[+,][−—][+where][− —][+,] so the …”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…nal because its ΔASR CI touches zero (McNemar p = 0.18)[− —][+;] it is not a statistical…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…he 1.7B model experiences large capability degradation [−— ][+(][+,][−(][−— ][+,][−—][+w…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…non-refusal proxy the baseline ASR gap looks far larger[− —][+,][− —][+,] but that refle…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…mary (judge) scorer. The 1.7B pair worsens on both axes[− —][+:][− —][+,][− —][+,][− —][…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “… is preserved, but HarmBench ASR rises beyond threshold[− —][+;] the model becomes more …”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “… is preserved, but HarmBench ASR falls beyond threshold[− —][+;] quantization reduces ha…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “Broad degradation: fallback[+.][−— t][+T]ypically signals over-ref…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “…study illustrate why the capability anchor is necessary[− —][+,][− —][+:][− —][+:][− —][…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “…t, a structurally different second capability benchmark[− —][+,][− —][+,][− —][+,][− —][…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “…ial-proportion delta is approximately ±0.05 (Table 6.1)[+,][−—][+so][− —][+:][− —][+;][+…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “… to the family of twenty primary NF4-vs-fp16 contrasts [−— ][+(][+,][−(][− —][+,][−The h…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “…ection floor as much as a substantive absence of effect[− —][+,] a limitation that a lar…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…he broad_degradation label (evidence status: confirmed)[− —][+;][−honest ][− —][+,] comb…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “… (4-bit) are genuine instances of the harmful behaviour[− —][+,] and that the *true* har…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…t. The capability anchor is what makes this unambiguous[− —][+:] the significant ΔMMLU r…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “… prompt-sampling variance only, not generation variance[− —][+,] the one Priority-1 weak…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…e maximum seed) and is not sign-consistent across seeds[− —][+,][− —][+,][− —][+,] the h…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…ssifier). The aggregate +0.055 increase is broad-based [−— ][+(][− —][+)] rather than an…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…akdown.py from the saved generations and judge sidecars[− —][+,][− —][+,] reported to ch…”
- Chapter 6: Results and Analysis § 6.7 Preliminary Scale Observations: Baseline Comparison: “…s baseline divergence motivates the matched-pair design[− —][+:] quantization effects mu…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “…abel by point estimate with evidence status directional[− —][+:][− —][+,][− —][+:] a sug…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “…y, so any safety move here is not a capability artefact[− —][+;][− —][+,] a stark contra…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “… degradation, while the larger model is far more robust[+,][−—][+with][−with][+and] capa…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…mily extension adds one significant over-refusal delta [−— ][+(][− —][+)][−— ][+(][− —][…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “… shows ΔASR = +0.055 (CI [+0.010, +0.100], significant)[− —][+:][− —][+,][− —][+,][− —][…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…icant over-refusal delta is Phi's, and it is a decrease[− —][+:][− —][+;] if anything, t…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…percentage points (CI [−0.040, +0.033], not significant[− —][+,] within measurement nois…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “Yes[− —][+,][− —][+,][+,][−—][+since] a second benchmark rough…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…safety (judge), all three ΔASR point estimates are ≥ 0 [−— ][+(][− —][+)][−— ][+(][− —][…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…the proxy) to Qwen 1.7B (under the official classifier)[− —][+,][−,][+rather tha][−ot][−…”
- Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b): “…ifier. HarmBench ASR at full precision is 0.040 (judge)[− —][+:][−Notably, t][+T][− —][+…”
- Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b): “…business ethics (+4.0 pp). The MMLU decline is moderate[− —][+,] larger than Qwen 4B (−0…”
- Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b): “…le safety direction. This is diagnostically informative[− —][+:][− —][+,] strong evidenc…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…HarmBench ASR is 0.040 for Llama vs 0.135 for Qwen 1.7B[− —][+,][− —][+,][−:][+.][−t][+T…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…ly significant: Qwen 1.7B (+0.055, CI [+0.010, +0.100])[− —][+,][− —][+:] the proxy's ap…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “… the five ΔOR deltas are not statistically significant [−— ][+(][− —][+),][+,][−—][+and]…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…ioning is consistent with a size-sensitivity hypothesis[− —][+:] Llama 3B, at 3 billion …”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…y artefacts?" The Qwen 1.7B pair degrades on both axes [−— ][+(][− —][+),][− —][+,][−— ]…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…al HarmBench classifier, cais/HarmBench-Llama-2-13b-cls[− —][+,] the fine-tuned judge mo…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “… ran in full precision (float16) on a 32 GB Tesla V100 [−— ][+(][− —][+)] because the ju…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…sus the regex 'non-refusal' proxy, one marker per model[+.][−— e][+E][− —][+:] low for Q…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Result 1[− —][+:][− —][+,][− —][+,] is adopted as the primar…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Result 2[− —][+:][− —][+:] it corrected a refusal-c…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Result 3[− —][+:][− —][+:] the disagreement is one-…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Result 4[− —][+:][− —][+,][− —][+,][− —][+,][−The honest reading a][+A][− is that] the Qw…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…establishes that the classifier and the regex disagree [−— ][+(][− —][+)] rather than th…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…ASR scorer): mistral_7b (Mistral-7B-Instruct-v0.3, 7.2B[− —][+,] the largest model in th…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…refusal rate (0.835) rising to 0.890 under quantization[− —][+,][− —][+:][−— ][+(][− —][…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…armful compliance is low and identical across the pair [−— ][+(][− —][+),][− —][+:][− —]…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…). The proxy-reliability split now spans four families [−— ][+(][− —][+),][−— ][+(][− —]…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…lysis measured each model's first-token refusal margin [−— ][+(][− —][+)] on all 200 Har…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “… opens with a refusal-shaped preamble and then complies[− —][+:] its refuse-versus-compl…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “Central result[− —][+:][− honest][−— ][+(][− —][+),][− —][+,][−d][+strong][−ci][−ive][+t]…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…cific direction, not that it uniformly erodes refusal. [−Tellingly, ][− —][+:] it has re…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “Capability versus alignment[− —][+:][− —][+,][− —][+:][− —][+:][− —][+,][− —][+,] has sin…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…sentially flat through INT8 and falls only at four-bit [−— ][+(][− —][+),] whereas the s…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “… is the bitsandbytes LLM.int8() mixed-precision method [−— ][+(][− —][+)] applied on the…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…it. The capability answer is unambiguous. No INT8 delta[− —][+,][− —][+,][− —][+,] footn…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…is the more revealing result, and it is also not graded[− —][+:][− —][+,][− —][+,][−Deci…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…≈8–9 prompts) on a single pair, and it is non-monotonic[− —][+,][−— ][+(][− —][+)] rathe…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…o Qwen pairs, 0.59 for Phi-4-mini and 0.76 for Llama-3B[− —][+,][−aken together, t][+.][…”
- Chapter 7: Discussion and Threats to Validity § 7.1 Internal Validity: “…ses the regex parser, and the primary HarmBench scorer [−— ][+(][− —][+)] is run with gr…”
- Chapter 7: Discussion and Threats to Validity § 7.4 Scorer Choice and Remaining Validity Threat: “…her than genuine harmful compliance and over-counts ASR[− —][+,][− —][+,][−— ][+(][− —][…”
- Chapter 7: Discussion and Threats to Validity § 7.5 Cross-Family Comparison Caveat: “…lausibly reflect any combination of these factors, not [−just][+only] family identity. T…”
- Chapter 7: Discussion and Threats to Validity § 7.6 Deployment Implications: “…he scale range studied. The most safety-relevant result[− —][+,][− —][+,][−— ][+(][− —][…”
- Chapter 7: Discussion and Threats to Validity § 7.7 Positioning Against Prior Work: “…Chen et al., ICML 2025); that the effect is non-uniform[− —][+,][− —][+,][+,][−—][+where…”
- Chapter 8: Limitations: “…shows the MMLU magnitudes are partly benchmark-specific[− —][+.][− —][+,] so capability …”
- Chapter 8: Limitations: “…degraded answer-format adherence under the token budget[− —][+,] a real capability effec…”
- Chapter 9: Future Work: “…owing the MMLU magnitudes are partly benchmark-specific[− —][+,] notably the Qwen 1.7B d…”
- Chapter 9: Future Work: “…ening more rigorously; and (3) an INT8 precision point [−— ][+(][− —][+)] which traced t…”
- Chapter 9: Future Work: “…yment regime. Both pairs use the identical methodology [−— ][+(][− —][+),][− —][+.][− —]…”
- Chapter 9: Future Work: “… method, an INT8 precision point (bitsandbytes LLM.int8[− —][+,][− —][+:] Qwen-1.7B's in…”
- Chapter 9: Future Work: “…nd reproduced the finding's direction (§6.12, Result 4)[− —][+,] substantially resolving…”
- Chapter 10: Conclusion: “…le per-model matrix jobs, and a verification suite of 3[−06][+29][− —][+,][− —][+,] whic…”
- Chapter 10: Conclusion: “… MMLU falls significantly (−0.087, CI [−0.137, −0.037])[− —][+:][− —][+,][− —][+;][− —][…”
- Chapter 10: Conclusion: “T[−aken together, t][− —][+,][− —][+,] is the durable contribut…”
- Appendix C: Pydantic Configuration Schema Summary: “study_name : str [−—][+:] human-readable study ide…”
- Appendix C: Pydantic Configuration Schema Summary: “models : dict[str, ModelConfig] [−—][+:] keyed by model alias.”
- Appendix C: Pydantic Configuration Schema Summary: “decoding : DecodingConfig [−—][+:] generation parameters.”
- Appendix C: Pydantic Configuration Schema Summary: “benchmarks : dict[str, BenchmarkConfig] [−—][+:] keyed by benchmark name.”
- Appendix C: Pydantic Configuration Schema Summary: “slurm : SlurmConfig [−—][+:] cluster directives and b…”
- Appendix C: Pydantic Configuration Schema Summary: “family : str [−—][+:] e.g. "qwen", "llama".”
- Appendix C: Pydantic Configuration Schema Summary: “size_b : float [−—][+:] parameter count in billi…”
- Appendix C: Pydantic Configuration Schema Summary: “quantized : bool [−—][+:] true triggers NF4 loadin…”
- Appendix C: Pydantic Configuration Schema Summary: “pair_id : str [−—][+:] links baseline and 4-bit…”
- Appendix C: Pydantic Configuration Schema Summary: “model_id : str [−—][+:] Hugging Face repo id.”
- Appendix C: Pydantic Configuration Schema Summary: “trust_remote_code : bool [−—][+:] disabled by default.”
- Appendix C: Pydantic Configuration Schema Summary: “dtype : str [−—][+:] one of "auto", "float16"…”
- Appendix C: Pydantic Configuration Schema Summary: “benchmarks : list[str] [−—][+:] must reference top-level…”
- Appendix C: Pydantic Configuration Schema Summary: “revision : str \| None [−—][+:] optional Hugging Face co…”
- Appendix C: Pydantic Configuration Schema Summary: “dataset_name : str [−—][+:] Hugging Face dataset id.”
- …and 13 more (see .full.json)

</details>

### Renamed headings (23)
- “Chapter 1 — Introduction” → “Chapter 1: Introduction”
- “Chapter 2 — Literature Review” → “Chapter 2: Literature Review”
- “Chapter 3 — Methodology” → “Chapter 3: Methodology”
- “Chapter 4 — System Design and Implementation” → “Chapter 4: System Design and Implementation”
- “Chapter 5 — Experimental Setup” → “Chapter 5: Experimental Setup”
- “Chapter 6 — Results and Analysis” → “Chapter 6: Results and Analysis”
- “RQ1 — Does 4-bit NF4 quantization increase harmful compliance?” → “RQ1: Does 4-bit NF4 quantization increase harmful compliance?”
- “RQ2 — Does 4-bit NF4 quantization increase over-refusal on benign pro…” → “RQ2: Does 4-bit NF4 quantization increase over-refusal on benign prom…”
- “RQ3 — Does 4-bit NF4 quantization degrade general capability?” → “RQ3: Does 4-bit NF4 quantization degrade general capability?”
- “RQ4 — Are smaller models more sensitive to quantization within the sa…” → “RQ4: Are smaller models more sensitive to quantization within the sam…”
- “RQ5 — Are effects consistent across model families?” → “RQ5: Are effects consistent across model families?”
- “Chapter 7 — Discussion and Threats to Validity” → “Chapter 7: Discussion and Threats to Validity”
- “Chapter 8 — Limitations” → “Chapter 8: Limitations”
- “Chapter 9 — Future Work” → “Chapter 9: Future Work”
- “Chapter 10 — Conclusion” → “Chapter 10: Conclusion”
- “Appendix A — Final TC1 Configuration” → “Appendix A: Final TC1 Configuration”
- “Appendix B — Example Generated SLURM Script” → “Appendix B: Example Generated SLURM Script”
- “Appendix C — Pydantic Configuration Schema Summary” → “Appendix C: Pydantic Configuration Schema Summary”
- “Appendix D — Test Inventory” → “Appendix D: Test Inventory”
- “Appendix E — Repository Layout” → “Appendix E: Repository Layout”
- “Appendix F — Glossary” → “Appendix F: Glossary”
- “Appendix G — Document Revision History” → “Appendix G: Document Revision History”
- “Appendix H — Scorer Incident and Future-Session Handoff” → “Appendix H: Scorer Incident and Future-Session Handoff”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
