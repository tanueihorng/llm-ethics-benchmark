## Rewritten interim report (submission line): Interim progress (SELECTED T53 artifact) → Interim v5 (retitle-only of rewrite v5)

*From:* `docs/rewritten_fyp_interim_progress.docx`  
*To:* `docs/rewritten_fyp_interim_v5.docx`  
*Decisions:* D72, D73, D75  

**Paragraph/body blocks:** +199 added · −26 removed · ~86 edited · ⟳11 citation-renumber-only · tables: +8 −3 ~1 · front matter blocks 13→18

*Front matter (cover/abstract/TOC/lists):* +52 −2 ~23 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Interim Report
  - ❌ TAN UEI HORNG (UTAN001)
  - ❌ Matriculation No.: U2320729F
  - ❌ Submitted in Partial Fulfilment of the Requirements for the Degree of Bachelor of Engineering (Comp…
  - ❌ Post-training quantization lowers the precision of a model's weights to reduce its memory footprint…
  - ➕ Academic Year 2025/2026
  - ➕ Declaration of Originality
  - ➕ I declare that this report is my own work and that all sources used have been acknowledged. The res…
  - ➕ Generative-AI tools assisted software implementation, test authoring, document review, and language…
  - ➕ Acknowledgements
  - ➕ I thank my supervisor, Dr. Zhang Jiehuang, for his guidance throughout this project. I also thank t…
  - ➕ List of Figures
  - ➕ Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance 3
  - ➕ Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow 15
  - ➕ Figure 4.1. Per-pair HarmBench classifier ΔASR with paired confidence intervals 24
  - ➕ Figure 4.2. Precision sweep across fp16, INT8, and NF4 28
  - ➕ Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy 31

### Chapters / parts ADDED
- ➕ Chapter 5: Scorer Validation and Robustness
- ➕ Chapter 6: Discussion
- ➕ Appendix A: Reproducibility Summary
- ➕ Appendix B: Minimum Detectable Capability Effects
- ➕ Appendix C: Human-Validation Boundaries

### Sections removed
- ❌ Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — 5.1 Progress Achieved

### Sections added
- ➕ Chapter 1: Introduction § 1.6 Report Structure — 1.6 Report Structure
- ➕ Chapter 2: Literature Review § 2.7 Implications for Experimental Design — 2.7 Implications for Experimental Design
- ➕ Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — 3.7 Reproducibility Controls
- ➕ Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — 3.8 Analysis Workflow and Q…
- ➕ Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — 5.1 HarmBench Classifier …
- ➕ Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — 5.2 Independent HarmBench Judges
- ➕ Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — 5.3 Human Validation…
- ➕ Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — 5.4 XSTest Scorer Se…
- ➕ Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — 5.5 Generation-Length Robustness
- ➕ Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — 5.6 Multi-Seed Decoding Sensitivity
- ➕ Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — 5.7 Power and the Meaning of the …
- ➕ Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — 5.8 Robustness Synthesis
- ➕ Chapter 6: Discussion § 6.1 Answer to the Central Question — 6.1 Answer to the Central Question
- ➕ Chapter 6: Discussion § 6.3 Capability Effects and Deployment — 6.3 Capability Effects and Deployment
- ➕ Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — 6.4 Validity of the Scoring Methods
- ➕ Chapter 6: Discussion § 6.5 Threats to Validity — 6.5 Threats to Validity
- ➕ Chapter 6: Discussion § 6.5 Threats to Validity — 6.5.1 Internal Validity
- ➕ Chapter 6: Discussion § 6.5 Threats to Validity — 6.5.2 Construct Validity
- ➕ Chapter 6: Discussion § 6.5 Threats to Validity — 6.5.3 External Validity
- ➕ Chapter 6: Discussion § 6.6 Practical Implications — 6.6 Practical Implications
- ➕ Chapter 7: Conclusion and Future Work § 7.1 Conclusions — 7.1 Conclusions
- ➕ Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — 7.2 Prioritised Future Work

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Reproducibility Summary | 3 | 0 | 0 | 0 |
| Appendix B: Minimum Detectable Capability Effects | 2 | 0 | 0 | 0 |
| Appendix C: Human-Validation Boundaries | 3 | 0 | 0 | 0 |
| Chapter 1: Introduction | 9 | 7 | 9 | 0 |
| Chapter 2: Literature Review | 4 | 0 | 20 | 3 |
| Chapter 3: Methodology and Experimental Setup | 11 | 1 | 10 | 1 |
| Chapter 4: Results | 8 | 8 | 24 | 0 |
| Chapter 5: Progress and Remaining Work | 0 | 8 | 0 | 0 |
| Chapter 5: Scorer Validation and Robustness | 43 | 0 | 0 | 0 |
| Chapter 6: Discussion | 47 | 0 | 0 | 0 |
| Chapter 7: Conclusion and Future Work | 13 | 0 | 0 | 0 |
| References | 4 | 0 | 0 | 7 |

### Removed paragraphs (24 body; 2 front matter)

<details><summary>show</summary>

- `074eaa6f` (146w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact open-weight language models are becoming increasingly feasible for local deployme…”
- `88e2bcfd` (46w) Chapter 1: Introduction § 1.2 Problem Statement — “This study therefore measures four outcomes under a single controlled precision contrast:…”
- `4d5099c9` (77w) Chapter 1: Introduction § 1.3 Research Questions — “RQ1 \| Does NF4 quantization measurably increase harmful compliance relative to matched f…”
- `18fd7f51` (3w) Chapter 1: Introduction § 1.5 Contributions — “Controlled matched-pair design”
- `3634d854` (3w) Chapter 1: Introduction § 1.5 Contributions — “Joint behavioural evaluation”
- `b387f6cb` (2w) Chapter 1: Introduction § 1.5 Contributions — “Scorer-validation study”
- `e76322e8` (3w) Chapter 1: Introduction § 1.5 Contributions — “Paired statistical analysis”
- `9652cc28` (65w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Interpretation labels combine direction and capability context. Alignment degradation mea…”
- `d35b475e` (46w) Chapter 4: Results § 4.1 Results at a Glance — “Applying multiple-comparison correction clarifies that the primary statistically defensib…”
- `3ae8151b` (58w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “Llama-3.2-3B decreases from 0.100 to 0.060. Its individually significant ΔASR is −0.040, …”
- `e396b821` (51w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “No pair shows a statistically significant increase in over-refusal, and four pairs are no…”
- `b8d052c7` (64w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “The primary family retains the regex result because it was the pre-specified outcome. By …”
- `cea2c399` (45w) Chapter 4: Results § 4.4 Capability (RQ3) — “Mistral loses 0.020 on MMLU but gains 0.0094 on ARC, while Phi loses 0.0267 on MMLU and 0…”
- `237c1278` (88w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels are descriptive summaries of the combined changes. Qwen3-1.7B i…”
- `76b7a267` (46w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ1 asks whether NF4 increases harmful compliance. The point estimates are 0.000, +0.040,…”
- `7feb3ee5` (43w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ3 asks whether capability falls. Qwen3-1.7B MMLU and Llama ARC survive correction. Qwen…”
- `f2db43ad` (24w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “At the interim report stage, the literature review, methodology, experimental setup, and …”
- `ec4cea62` (94w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “The primary evidence is sufficiently developed for the interim report to state its centra…”
- `24dd71ed` (7w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “Table 5.1. Completed work and repository evidence”
- `48ad7931` (32w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “Workstream \| Status \|\| Literature review \| Complete \|\| Methodology and setup \| Com…”
- `573d9689` (50w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “The remaining work is the detailed final-report writing for two chapters whose underlying…”
- `ece581f9` (61w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “The Discussion chapter will then connect the primary results to the literature, explain t…”
- `f29efd0a` (5w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “Table 5.2. Remaining final-report writing”
- `d6e2d70f` (59w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “Planned chapter \| Evidence already complete \| Writing still required \|\| Scorer Valida…”

</details>

### Added paragraphs (147)

<details><summary>show</summary>

- `c361956e` (148w) Chapter 1: Introduction § 1.1 Background and Motivation — “This project studies five matched model pairs drawn from four compact open-weight model f…”
- `f009efa9` (45w) Chapter 1: Introduction § 1.2 Problem Statement — “This study therefore measures all four outcomes on the same prompts under a single contro…”
- `8fee655f` (13w) Chapter 1: Introduction § 1.3 Research Questions — “RQ1. Does NF4 quantization measurably increase harmful compliance relative to matched fp1…”
- `6db2e100` (9w) Chapter 1: Introduction § 1.3 Research Questions — “RQ2. Does NF4 quantization increase over-refusal on benign prompts?”
- `bd8cc667` (10w) Chapter 1: Introduction § 1.3 Research Questions — “RQ3. Does NF4 quantization reduce capability on MMLU or ARC-Challenge?”
- `26273b59` (17w) Chapter 1: Introduction § 1.3 Research Questions — “RQ4. How do the effects vary across model families, model scales, and the additional INT8…”
- `65c0ff2b` (19w) Chapter 1: Introduction § 1.3 Research Questions — “RQ5. Does the substantive conclusion remain credible under alternative scorers, compariso…”
- `38329771` (20w) Chapter 1: Introduction § 1.5 Contributions — “Reproducible benchmarking pipeline. The implementation preserves immutable raw model outp…”
- `a69a0820` (80w) Chapter 1: Introduction § 1.6 Report Structure — “Chapter 2 reviews quantization, safety and refusal evaluation, using capability results t…”
- `af20b12a` (69w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “The reviewed literature suggests that a convincing quantization-safety experiment must co…”
- `7cef32ab` (70w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “Statistical design is equally important. Twenty model–outcome comparisons create many opp…”
- `d2834c81` (79w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `c9becd11` (52w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “Finally, research integrity affects interpretation. Revised scoring should not overwrite …”
- `8de7c3c9` (82w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Interpretation labels combine the direction of the safety change with the capability cont…”
- `d9910245` (47w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “The command-line interface exposes single-run, matrix, analysis, and agent-status paths. …”
- `c2965a8e` (42w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Testing covers configuration validation, model reuse, result resumption, refusal parsing,…”
- `c9fd291c` (80w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “Every model revision in the primary configuration is pinned, and the report records the s…”
- `54a44ec6` (102w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “The project uses automated tests and claim verification to detect stale numbers and incon…”
- `6fd56950` (67w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “Reproduction has two levels. Analysis replay starts from the stored redacted scores and r…”
- `e92f2f4d` (45w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “The report avoids a stronger claim of exact historical rerun because not every earliest p…”
- `d8f6407f` (68w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “The analysis begins by joining baseline and quantized records on prompt identifier, and c…”
- `a29a8c2e` (49w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Multiple-comparison correction is performed from the complete 20-row analysis record rath…”
- `d786c363` (62w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Figures are produced by scripts from the analysis JSON rather than drawn manually. The fo…”
- `22c55306` (56w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Quality control has both machine and human components. Tests detect schema failures, stal…”
- `ba0cf0c3` (45w) Chapter 4: Results § 4.1 Results at a Glance — “After multiple-comparison correction, the effects that remain statistically significant c…”
- `a08e89d4` (75w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “Llama-3.2-3B decreases from 0.100 to 0.060. The estimated change in harmful compliance is…”
- `1054430d` (61w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “No pair shows a statistically significant increase in over-refusal. Four pairs show no st…”
- `d2ebd9b4` (66w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “The primary analysis retains the regex score because it was pre-specified. The validation…”
- `7947d39e` (48w) Chapter 4: Results § 4.4 Capability (RQ3) — “Mistral loses 0.020 on MMLU but gains 0.0094 on ARC, while Phi loses 0.0267 on MMLU and 0…”
- `cf64abba` (117w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels summarise the observed combination of safety and capability cha…”
- `cd3b1c2e` (56w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ1 asks whether NF4 increases harmful compliance. The estimated changes are 0.000, +0.04…”
- `194d13ba` (60w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ3 asks whether capability falls. Qwen3-1.7B MMLU and Llama ARC remain statistically sig…”
- `c37cc049` (43w) Chapter 5: Scorer Validation and Robustness — “The primary findings depend on measurement choices. This chapter tests the central assump…”
- `81ae1f3e` (50w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “Figure 5.1 compares the HarmBench classifier with the refusal-regex proxy. The proxy syst…”
- `0ee0173d` (72w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The divergence is largest in model families whose response style does not align with the …”
- `55de89e7` (56w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The disagreement is not a constant offset. If the proxy added the same bias to both preci…”
- `e11acff5` (43w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The correction was made through derived scoring files, and original generations and summa…”
- `77c87660` (12w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy and their agreement.”
- `3581d5e0` (51w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “The API judge agrees strongly with the HarmBench classifier across the ten fp16/NF4 alias…”
- `832c7288` (72w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Llama Guard provides an open-weight cross-check. Its agreement with the behaviour-specifi…”
- `355a4d35` (78w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “The API judge and classifier agree most strongly on the Llama pair, where kappa exceeds 0…”
- `242ef201` (60w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Llama Guard's per-pair directions agree with the classifier for four pairs. Qwen3-1.7B is…”
- `436aa5d4` (8w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Table 5.1. Per-pair HarmBench deltas under three judges”
- `a89d65bd` (67w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Pair \| HarmBench classifier Δ \| API judge Δ \| Llama Guard Δ \| Llama Guard p<.05 \|\| …”
- `7ddffd57` (54w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The 200-item human sample supports the choice of scorer. The HarmBench classifier reaches…”
- `054b528c` (52w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The human-labelled sample deliberately oversamples the items on which the two scorers dis…”
- `f26a1c09` (48w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The classifier's recall is 0.757 and its precision is 0.609 on the human-labelled sample.…”
- `09ad75c9` (29w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “Kappa is affected by prevalence and marginal distributions. The report therefore presents…”
- `74889509` (84w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “An independent three-class semantic refusal judge does not replicate the Phi-4-mini over-…”
- `352e23b9` (84w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “Human annotation clarifies this discrepancy. On a blinded validation sample, the semantic…”
- `77dbcd4a` (79w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The XSTest human validation protocol incorporated one disclosed procedural deviation: the…”
- `6e873070` (24w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The human set contains 103 full compliances, 34 partial refusals, and 63 full refusals. T…”
- `d05b77ab` (47w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The broad refusal mapping includes partial refusals alongside complete rejections. While …”
- `e83d7fca` (46w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “In a validation-informed sensitivity analysis where the keyword regex is replaced by the …”
- `725df4be` (58w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The 128-token runs truncated 60.3% of the 2,000 HarmBench responses under the direct pref…”
- `6aaddb0a` (40w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The longer generation limit also changes baseline compliance rates by providing models su…”
- `5f760f63` (49w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Truncation is highly model-dependent. At 128 tokens it affects most Mistral and Phi respo…”
- `c103316e` (48w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “A direct prefix test distinguishes responses that are exact prefixes of the longer output…”
- `6ea914d3` (10w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Table 5.2. HarmBench classifier deltas at 128 and 512 tokens”
- `471e7675` (75w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Pair \| ΔASR @128 \| ΔASR @512 \| McNemar p @128 \| McNemar p @512 \|\| Qwen3-1.7B (128/5…”
- `51fb1eee` (62w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “The five-seed stochastic-decoding analysis tests whether the greedy result is specific to…”
- `61133bba` (42w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Across multiple decoding seeds, the mean changes for Mistral and Phi remain closely clust…”
- `26558871` (23w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Varying the random decoding seed captures output variance introduced by non-deterministic…”
- `b287f2e0` (43w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “One Qwen3-4B seed and one Phi seed are individually significant, in opposite directions, …”
- `fe89291c` (9w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Table 5.3. Five-seed HarmBench sensitivity at the 512-token budget”
- `41fe7d0c` (69w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Pair \| Greedy Δ \| Seed mean Δ \| Seed range \| Seeds p<.05 \|\| Qwen3-1.7B \| +0.000 \|…”
- `67a1847a` (45w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2 illustrates the relationship between the Qwen3-1.7B greedy estimate and its fi…”
- `15d7cad5` (9w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity.”
- `d4727d49` (57w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “A non-significant statistical test does not establish that the true effect is precisely z…”
- `53ce624d` (59w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Four partly independent checks support the null result: no HarmBench comparison remains s…”
- `c17c4eb3` (48w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Capability evaluations differ in statistical sensitivity because MMLU and ARC have differ…”
- `dc4e8c3d` (29w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Within the detection limits of the experimental design, the study finds no statistically …”
- `73d51686` (44w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Each validation experiment addresses one competing explanation: human annotations expose …”
- `03d77ebb` (48w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “No single check is decisive. Together, however, they make a large hidden increase in harm…”
- `dca7fad2` (31w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Analysing why scorers disagree is more informative than averaging their scores. When eval…”
- `c92b5a3e` (39w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The empirical findings show why harmful compliance, capability, and scorer behaviour need…”
- `7327ba1c` (36w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “These conclusions apply strictly to the evaluated model architectures, NF4/INT8 precision…”
- `35f26245` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results support neither common oversimplification: neither that quantization inherent…”
- `46af4b45` (37w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more accurate description of low-bit quantization in this study is instability in indiv…”
- `e00ae762` (8w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “Table 6.1. Possible explanations for the HarmBench result”
- `23eec909` (99w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “Candidate explanation \| Discriminating evidence \| Assessment \|\| True practical preser…”
- `eb90b087` (52w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The result is closest to Decoding Compressed Trust, which reports that trustworthiness is…”
- `4725b6bc` (80w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Safety-degradation studies remain relevant. Kharinaev et al. report that quantization can…”
- `3b7bb1b6` (82w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2 compares published results with those of this study rather than leaving the com…”
- `2dc1dbb9` (78w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2. Comparison with published quantization results. Values are transcribed from ea…”
- `364d32bf` (477w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Study \| Quantized model \| Reported change, in the source's own metric \| Scorer \| Chie…”
- `51646ed1` (117w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Two features of the table deserve emphasis because they are easy to misread. In the IEEE …”
- `3db5c862` (54w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesising the published literature reinforces the contextual nature of safety findings…”
- `1db3b675` (65w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The scorer findings align with StrongREJECT's warning about empty jailbreaks [38] and SOR…”
- `4d54570d` (55w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The capability findings also qualify earlier capability-focused quantization work. The th…”
- `99068cea` (47w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Marchisio et al.'s multilingual result is methodologically relevant even though the prese…”
- `360bcef1` (48w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The direct-request scope explains why adversarial studies should not be used as either co…”
- `3c6db3a3` (55w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The capability result is practically important because a compressed model is chosen for d…”
- `018d7322` (59w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “INT8 provides a useful intermediate point. Its capability comparisons are close to fp16 w…”
- `dabccde2` (52w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The two benchmarks of general capability also caution against a single-number deployment …”
- `a9b97883` (70w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The deployment recommendation is limited because latency, memory use, and energy consumpt…”
- `c854dcb1` (52w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “Deployment evaluation should begin with the requirements of the intended application. If …”
- `93404a4c` (49w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “Safety should be evaluated using the final deployment configuration. A pre-quantized chec…”
- `ea00eff4` (42w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Validating scoring rubrics is central to rigorous empirical evaluation. The HarmBench key…”
- `e1191da2` (63w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “At the same time, semantic judges introduce their own limitations. The HarmBench API judg…”
- `3f206f37` (62w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “This has a general implication for LLM evaluation. A benchmark score should be understood…”
- `1fc5b7b2` (59w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The two scorer failures point in opposite directions. On HarmBench, a refusal proxy over-…”
- `6b14428e` (49w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Both the preregistered regex analysis and the validated semantic judge evaluation are rep…”
- `3d6b9a17` (52w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The protocol deviation demonstrates the same principle. The blindness exclusion was more …”
- `5e8986c0` (65w) Chapter 6: Discussion § 6.5 Threats to Validity — “The matched revisions, shared prompts, fixed decoding controls, and paired tests strength…”
- `5ab62fdc` (47w) Chapter 6: Discussion § 6.5 Threats to Validity — “The XSTest human-labelling exercise deviated from its pre-registered blindness-exclusion …”
- `6561a09b` (58w) Chapter 6: Discussion § 6.5 Threats to Validity — “The sensitivity analyses were conducted after the primary runs. Their role is exploratory…”
- `b297d8f2` (41w) Chapter 6: Discussion § 6.5 Threats to Validity — “Several analysis choices are rule-based, including parser behaviour and interpretation la…”
- `d8405d61` (65w) Chapter 6: Discussion § 6.5 Threats to Validity — “Construct validity asks whether each benchmark and scorer measures the behaviour it is in…”
- `070cc14b` (37w) Chapter 6: Discussion § 6.5 Threats to Validity — “Human validation uses one annotator on each 200-item sample. The comparisons are therefor…”
- `9314c02b` (60w) Chapter 6: Discussion § 6.5 Threats to Validity — “The HarmBench classifier is trained for the benchmark's behaviour criterion, which aligns…”
- `5260fcba` (35w) Chapter 6: Discussion § 6.5 Threats to Validity — “Over-refusal is especially policy-sensitive. A partial refusal can be counted as acceptab…”
- `385d9a16` (46w) Chapter 6: Discussion § 6.5 Threats to Validity — “The five pairs cover four families and a compact-model range, but they are not a random s…”
- `0a739661` (43w) Chapter 6: Discussion § 6.5 Threats to Validity — “The study's strongest generalisable contribution is methodological: match the weights, ev…”
- `6527c7a0` (36w) Chapter 6: Discussion § 6.5 Threats to Validity — “English-language benchmarks dominate the study. Qwen and other multilingual families may …”
- `29777691` (38w) Chapter 6: Discussion § 6.5 Threats to Validity — “The V100 environment is relevant because numerical kernels and supported attention paths …”
- `dfbe62fd` (15w) Chapter 6: Discussion § 6.6 Practical Implications — “Do not inherit a safety claim from an fp16 checkpoint without testing the deployed precis…”
- `b4a80789` (15w) Chapter 6: Discussion § 6.6 Practical Implications — “Use a behaviour-specific semantic scorer for harmful compliance; retain transparent proxi…”
- `9fa90ab4` (16w) Chapter 6: Discussion § 6.6 Practical Implications — “Pair safety with benign-refusal and capability benchmarks so that inability is not report…”
- `92fd7382` (19w) Chapter 6: Discussion § 6.6 Practical Implications — “Prefer INT8 when its smaller compression benefit is sufficient and task accuracy is criti…”
- `787d09ce` (15w) Chapter 6: Discussion § 6.6 Practical Implications — “Treat direct-request results as direct-request results. A separate adversarial evaluation…”
- `31a7a9d7` (33w) Chapter 6: Discussion § 6.6 Practical Implications — “The practical takeaway for system designers is procedural: quantized language models shou…”
- `65a5546d` (36w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This research conducted a controlled empirical evaluation of five matched compact languag…”
- `7c327904` (96w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “In summary, RQ1 receives a negative answer: no NF4 change in harmful compliance remains s…”
- `8c4bac67` (32w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The central methodological insight is that safety evaluations must not rely solely on key…”
- `b79a981a` (28w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The primary contribution of this work is the empirical evidence and the accompanying scor…”
- `f61af5a0` (36w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “From an engineering standpoint, INT8 was the more conservative precision tier for reasoni…”
- `49cae9b3` (26w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Add an adversarial HarmBench evaluation using at least one standardised attack. This is t…”
- `8af85f5c` (15w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Collect a second independent annotator for both human-validation samples and report inter…”
- `ba18a510` (17w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Compare NF4 and LLM.int8 with GPTQ, AWQ, and a GGUF deployment configuration on the same …”
- `f76d536a` (23w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Measure peak memory, model footprint, load time, throughput, and energy so that the capab…”
- `3b2e1e4b` (20w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Extend capability evaluation to open-ended reasoning, coding, and workload-specific tasks…”
- `b9bb6376` (18w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Increase the number of decoding seeds and apply the seed analysis to XSTest and capabilit…”
- `b959d18f` (24w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Future investigations should maintain rigorous evaluation standards by preserving immutab…”
- `b1320ed6` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The most critical future extensions include extending safety evaluations to adversarial j…”
- `f879f1b3` (12w) References — “[58] J. Gu et al., “A survey on LLM-as-a-judge,” arXiv:2411.15594, 2024 (preprint).”
- `184b440f` (23w) References — “[59] M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human gro…”
- `8546d089` (17w) References — “[60] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. …”
- `2010eeed` (18w) References — “[61] J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale,…”
- `ef85f809` (5w) Appendix A: Reproducibility Summary — “Table A.1. Primary run controls”
- `e103bd5b` (225w) Appendix A: Reproducibility Summary — “Control \| Value \|\| Primary configuration \| configs/tc1_512.yaml \|\| Model pairs \| 5…”
- `c0a4b867` (31w) Appendix A: Reproducibility Summary — “The repository contains the pinned model revisions, full configuration schema, generated …”
- `2b478133` (11w) Appendix B: Minimum Detectable Capability Effects — “Table B.1. Observed capability effects and 80% power minimum detectable effects”
- `5dd3900a` (134w) Appendix B: Minimum Detectable Capability Effects — “Pair \| Benchmark \| n \| Observed Δ \| MDE \| \|Δ\| > MDE \|\| Qwen3-1.7B \| MMLU \| 300…”
- `3124bafb` (44w) Appendix C: Human-Validation Boundaries — “Both human-validation sets contain 200 labelled responses and one annotator. The HarmBenc…”
- `63dcf5aa` (29w) Appendix C: Human-Validation Boundaries — “No prompt or response text is reproduced in this report. The stored validation files cont…”
- `41701ee2` (65w) Appendix C: Human-Validation Boundaries — “The XSTest blindness exclusion was applied prompt-wide instead of to the limited alias–pr…”

</details>

### Edited paragraphs (86; change window shown)

<details><summary>show</summary>

- (front matter): “[−FYP ]Interim Report”
- (front matter): “TAN UEI HORNG[+ (UTAN001)]”
- (front matter): “…ment of the Requirements for the Degree of Bachelor of [−Comput][+Engineer][−in ][+(][+)]”
- (front matter): “…ultiple-comparison correction. Three results remained s[+tatistically s][−reveal][+show]…”
- (front matter): “2.4 Refusal, Harmfulness, and Threat Models [−8][+9]”
- (front matter): “Chapter 3: Methodology and Experimental Setup 1[−3][+4]”
- (front matter): “3.1 Matched-Pair Experimental Design 1[−3][+4]”
- (front matter): “3.2 Models and Precision Conditions 1[−4][+5]”
- (front matter): “3.3 Benchmarks and Outcomes 1[−5][+6]”
- (front matter): “3.4 Scoring and Validation 1[−6][+7]”
- (front matter): “3.5 Statistical Analysis 1[−6][+7]”
- (front matter): “3.6 Implementation and Execution 1[−8][+9]”
- (front matter): “Chapter 4: Results [−19][+22]”
- (front matter): “4.1 Results at a Glance [−19][+22]”
- (front matter): “4.2 Harmful Compliance (RQ1) 2[−0][+3]”
- (front matter): “4.3 Over-Refusal on Benign Prompts (RQ2) 2[−1][+4]”
- (front matter): “4.4 Capability (RQ3) 2[−2][+5]”
- (front matter): “4.5 Model Scale and Family Differences (RQ4) 2[−3][+6]”
- (front matter): “4.6 INT8 Precision Point 2[−4][+7]”
- (front matter): “4.7 Research-Question Synthesis 2[−5][+8]”
- (front matter): “Chapter [−5][+7][−Pr][+C][−gre][+nclu][−s][+ion][−R][+Futur][−maining][−26][+44]”
- (front matter): “[−5][+6][−m][+l][+t][+o][+sh][−ng][+p to Prior][−26][+37]”
- (front matter): “References [−28][+46]”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…ment change or a loss of capability. Safety evaluation [−i][+ha][−furth][+a s][−r][+cond…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…4 at load time to the same weights as the fp16 baseline[−, while][+;][−power-oriented ][…”
- Chapter 1: Introduction § 1.2 Problem Statement: “The[+ central][+one of ][+at][−ve][+on][+asu][+ment a][−y operati][+e: ][−l. A] change in…”
- Chapter 1: Introduction § 1.2 Problem Statement: “…n the direction of decreased harmful compliance, while [+the estimated MMLU change is a …”
- Chapter 1: Introduction § 1.4 Scope: “…enge. These multiple-choice instruments provide useful [−a][+refere][−h][+e p][−r][+int]…”
- Chapter 1: Introduction § 1.5 Contributions: “[+Controlled matched-pair design. ]The study applies NF4 or …”
- Chapter 1: Introduction § 1.5 Contributions: “[+Joint behavioural evaluation. ]The study evaluates harmf…”
- Chapter 1: Introduction § 1.5 Contributions: “[+Scorer-validation study. ][−and XST][+r][+ul][− ][−coring][−benchmark-specific][+two] h…”
- Chapter 1: Introduction § 1.5 Contributions: “[+Paired statistical analysis. ][+correction for ][−icity][+e][+mpariso][−trol][+s][−d po…”
- Chapter 2: Literature Review: “…iment must control: the compression method, the safety [−c][+behavi][+ur bei][+g mea][−t…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…s [20]. Neither represents all useful behaviour. Their [−va][+ro][−u][−di][+expl][−g][+a…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…tched quantization study benefits from fewer outcomes w[+hose scoring cr][−h][+eria are …”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…es, equivocates, or supplies partial information. That [+extr][−dd][+ uncerta][+n][+y ][…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…that the effect is task-specific. The report therefore [−avoids colla][+kee][−ing][+sepa…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “… pruning, can remove many weights in one shot [23]. The[−refore, the][+re therefo][− int…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…on on aggregate tasks, especially as model size grows [[−3][+7][−, [8], ][+–][9]. Howeve…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…ety and capability changed smoothly as precision fell, [+beh][− graded fp16–INT8–NF4 c][…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…evel of harm in a response [29]. These are related but [−non-][+d][−d][+ffer][+ ways of …”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…needed precisely because worst-case vulnerability and t[+he t][−treatment ][+ of ordinar…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…prior study most similar to this experiment because it [−observe][+find][−bro][+th][−d][…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…es everything. XSTest provides a smaller human-curated [−diagno][+te][−ic][+ targeted at…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…al [41]. It is useful as an independent moderator, but [−i][−s][+he][−con][+it applie][−…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…In a DirectRequest test, the prompt is fixed and the co[+mpariso][−trast] focuses on the…”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…e making model-specific errors that distort a paired co[+mpariso][−trast].”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…valuable for reproducibility, a semantic judge tests co[−nstruct co][+ of the behaviour …”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…[41]. Their openness permits rerunning the scorer, yet [−construct ][+ between what the …”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “… and discloses a pre-registration deviation. These two [+v][−u][+li][+at][−t][+on exerci…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…ps with a five-pair matched design, NF4 as the primary [−i][+qua][−erve][+izatio][+ cond…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…eparated from capability, scorer error, truncation, or [+the nu][−ul][+ber of comparison…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…sion selected by the TC1 CUDA environment, whereas the [−tre][+qu][−tme][+ized][+able A.…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “The primary [+quan][−re][+iz][−me][+io][+ condi][+ion] is NF4 four-bit quantiza…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…cale are observed grouping variables rather than random[+ly ass][−s][+gn][+condi][−reatm…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…ce prompts, and ARC-Challenge provides a larger second [+c][+pability be][−o][+ma][+k] w…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…esholds, which is why a similar absolute delta can be s[+tatistically s][−a][+yet i][−d …”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “…riginal analysis while testing whether its conclusion s[−urv][+t][−v][+ll holds wh][−s][…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…primary NF4 set of corrected comparisons contains 20 co[+mpariso][−trast][−n][+rrec][+s …”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…tion used in the scorer sensitivity analysis [55], and [+the behaviour ][+ label][−co][+…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “… not remove measurement error. Exactly three primary co[+mparisons remained statisticall…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…ions and summaries are treated as immutable run files. [−R][+Outputs of r][− output][+ p…”
- Chapter 4: Results: “This chapter reports the primary matched-pair results [−and summa][+befo][−iz][−s][− comp…”
- Chapter 4: Results § 4.1 Results at a Glance: “The primary [+analy][−tat][−tical][−family en][−mpass][+v][+r]s 20 NF4 comparisons acro…”
- Chapter 4: Results § 4.1 Results at a Glance: “… sizes and statistical uncertainty. Exact p-values and [−correc][+adjus][−cont][−a][+e][…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “No[+ne of the five][+mparisons remains statistically significa][+ afte][−ast survives mul…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “…eline ASR is 0.585, more than eight times Phi's 0.070. [−P][+The p][+iff][−lta][+rence][…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “Qwen3-4B [−i][+show][−point ][− for][+d][−d][+ in][+, at 0.040][−is ][+pa][−h][+t][−ef][+…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “… from safe to harmful, and that imbalance produces the [+decrease that ][−nd][+s stat][−…”
- Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2): “…ts in the set of 250 prompts, which is large enough to [+remain ][−u][+tatistically sign…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…r Qwen3-1.7B, MMLU accuracy falls from 0.643 to 0.553, [−representing ][−lo][+decrea][−s…”
- Chapter 4: Results § 4.4 Capability (RQ3): “Llama-3.2-3B [+s][−a][+ow][+de][−ont][+e][−t][+e][+remain][−u][+ statistically significan…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…while ARC falls from 0.876 to 0.860. The ARC result is [+stat][−nd][+st][−vidu][+c][+ be…”
- Chapter 4: Results § 4.4 Capability (RQ3): “… similar in magnitude but estimated more precisely, so [+the ][+re][+lt ][−v][+ema][−v][…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…0% power thresholds, whereas several smaller observed d[+iff][−lta][+rence][−Th][+App][+…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…ive pairs in the joint space these results define. The [−MMLU point ][+d MMLU change is …”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “…whereas the 4B model changes by −0.0033. However, that [+differen][−ontrast][+e][−suppor…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “…s different practical meaning at those baselines. This [−he][+varia][−e][+ion ac][−g][+s…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “[−C][+Diffe][+ent m][−ss-][+del ][−y][+ies produce][+s in different][+,][+which c][−lso][…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “The INT8 comparison asks whether the observed pattern [−i][+change][−ed][+ually][−follow]…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…upports practical similarity for the INT8 capability co[+mpariso][−trast]s within the st…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “For deployment, the[− evidenc][+s][−fav][+results supp][−u][−s][+t] a staged decision. IN…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “… in shape or in direction. Together they argue against [−o][+a si][+gl][−caused by][+of]…”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “…benign over-refusal. The pre-specified regex finds no s[+tatistically s][+ statistically…”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “…y while NF4 contains the largest losses. Overall, the e[−v][+ffect of lower prec][+sion …”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “RQ5 [−inve][+a][−tigate][+k][−remain r][+h][−bust][+ld][−fav][+supp][−u][+t]s the HarmBen…”

</details>

### Citation-renumber-only paragraphs: 11 (text identical once [n] markers stripped; details in .full.json)

### Renamed headings (2)
- “Chapter 5: Progress and Remaining Work” → “Chapter 7: Conclusion and Future Work”
- “5.2 Remaining Work” → “6.2 Relationship to Prior Work”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
