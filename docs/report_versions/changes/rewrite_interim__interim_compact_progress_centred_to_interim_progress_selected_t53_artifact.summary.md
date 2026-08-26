## Rewritten interim report (submission line): Interim compact (progress-centred) → Interim progress (SELECTED T53 artifact)

*From:* `docs/rewritten_fyp_interim_compact.docx`  
*To:* `docs/rewritten_fyp_interim_progress.docx`  
*Decisions:* D71, D72, D73  

**Paragraph/body blocks:** +66 added · −92 removed · ~52 edited · ⟳54 citation-renumber-only · tables: +4 −6 ~0 · front matter blocks 16→13

*Front matter (cover/abstract/TOC/lists):* +5 −17 ~25 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ FYP Interim Report
  - ❌ TAN UEI HORNG
  - ❌ Submitted in Partial Fulfilment of the Requirements for the Degree of Bachelor of Computing in Comp…
  - ❌ Academic Year 2025/2026
  - ❌ Declaration of Originality
  - ❌ I declare that this report is my own work and that all sources used have been acknowledged. The res…
  - ❌ Generative-AI tools assisted software implementation, test authoring, document review, and language…
  - ➕ Matriculation No.: U2320729F

### Chapters / parts REMOVED
- ❌ Chapter 5: Scorer Validation and Robustness
- ❌ Chapter 6: Discussion

### Sections removed
- ❌ Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — 5.1 HarmBench Classifier …
- ❌ Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — 5.2 Independent HarmBench Judges
- ❌ Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — 5.3 Human Validation…
- ❌ Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — 5.4 XSTest Scorer Se…
- ❌ Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — 5.5 Generation-Length Robustness
- ❌ Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null — 5.6 Power and the Meaning of the …
- ❌ Chapter 5: Scorer Validation and Robustness § 5.7 Robustness Synthesis — 5.7 Robustness Synthesis
- ❌ Chapter 6: Discussion § 6.1 Answer to the Central Question — 6.1 Answer to the Central Question
- ❌ Chapter 6: Discussion § 6.2 Threats to Validity — 6.2 Threats to Validity
- ❌ Chapter 6: Discussion § 6.2 Threats to Validity — 6.2.1 Internal Validity
- ❌ Chapter 6: Discussion § 6.2 Threats to Validity — 6.2.2 Construct Validity
- ❌ Chapter 6: Discussion § 6.2 Threats to Validity — 6.2.3 External Validity
- ❌ Chapter 6: Discussion § 6.3 Practical Implications — 6.3 Practical Implications
- ❌ Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — 7.2 Problems Encountered and Resolu…
- ❌ Chapter 7: Progress and Remaining Work § 7.4 Submission Timeline — 7.4 Submission Timeline

### Sections added
- ➕ Chapter 1: Introduction § 1.5 Contributions — 1.5 Contributions
- ➕ Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — 2.1 Trustworthiness and Safety Evaluation
- ➕ Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — 2.4 Refusal, Harmfulness, and Threat Models
- ➕ Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — 2.5 Automatic Evaluators and Human Valid…
- ➕ Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — 3.6 Implementation and Execution

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Chapter 1: Introduction | 9 | 5 | 6 | 0 |
| Chapter 2: Literature Review | 27 | 3 | 3 | 9 |
| Chapter 3: Methodology and Experimental Setup | 5 | 0 | 3 | 6 |
| Chapter 4: Results | 1 | 1 | 13 | 0 |
| Chapter 5: Progress and Remaining Work | 6 | 0 | 2 | 0 |
| Chapter 5: Scorer Validation and Robustness | 0 | 34 | 0 | 0 |
| Chapter 6: Discussion | 0 | 24 | 0 | 0 |
| Chapter 7: Progress and Remaining Work | 0 | 8 | 0 | 0 |
| References | 13 | 0 | 0 | 39 |

### Removed paragraphs (75 body; 17 front matter)

<details><summary>show</summary>

- `8fee655f` (13w) Chapter 1: Introduction § 1.3 Research Questions — “RQ1. Does NF4 quantization measurably increase harmful compliance relative to matched fp1…”
- `6db2e100` (9w) Chapter 1: Introduction § 1.3 Research Questions — “RQ2. Does NF4 quantization increase over-refusal on benign prompts?”
- `bd8cc667` (10w) Chapter 1: Introduction § 1.3 Research Questions — “RQ3. Does NF4 quantization reduce capability on MMLU or ARC-Challenge?”
- `26273b59` (17w) Chapter 1: Introduction § 1.3 Research Questions — “RQ4. How do the effects vary across model families, model scales, and the additional INT8…”
- `65c0ff2b` (19w) Chapter 1: Introduction § 1.3 Research Questions — “RQ5. Does the substantive conclusion remain credible under alternative scorers, compariso…”
- `2187b631` (77w) Chapter 2: Literature Review § 2.2 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
- `80ed90a9` (74w) Chapter 2: Literature Review § 2.2 Quantization and Safety Behaviour — “Mechanistic studies examine how refusal is represented inside the model. Safety Layers ar…”
- `24a5cb08` (61w) Chapter 2: Literature Review § 2.3 Synthesis and Research Gap — “The gap is not merely the absence of the same benchmark combination in prior work. The ma…”
- `007d76ee` (41w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “Cross-family response styles also affect measurement. Mistral provides the clearest examp…”
- `2f705058` (40w) Chapter 5: Scorer Validation and Robustness — “The primary findings depend on measurement choices. This chapter directly tests whether t…”
- `0002377c` (56w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The HarmBench classifier and the refusal-regex proxy diverge because the proxy systematic…”
- `0ee0173d` (72w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The divergence is largest in model families whose response style does not align with the …”
- `33b2675e` (52w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The disagreement is not a constant offset. If the proxy added the same bias to both preci…”
- `3b95e61a` (43w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The correction was made through derived scoring files, and original generations and summa…”
- `a7df0eb4` (69w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “The API judge agrees strongly with the HarmBench classifier across the ten fp16/NF4 alias…”
- `dba2e15a` (72w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Llama Guard provides an open-weight cross-check. Its agreement with the behaviour-specifi…”
- `355a4d35` (78w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “The API judge and classifier agree most strongly on the Llama pair, where kappa exceeds 0…”
- `8b77fe0b` (60w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Llama Guard's per-pair directions agree with the classifier for four pairs. Qwen3-1.7B is…”
- `436aa5d4` (8w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Table 5.1. Per-pair HarmBench deltas under three judges”
- `a89d65bd` (67w) Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges — “Pair \| HarmBench classifier Δ \| API judge Δ \| Llama Guard Δ \| Llama Guard p<.05 \|\| …”
- `b2c15277` (53w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The 200-item human sample supports the choice of scorer. The HarmBench classifier reaches…”
- `8f07fe92` (42w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The human audit sample purposefully oversamples scorer disagreements to maximise diagnost…”
- `4b0d73ee` (49w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The classifier's recall is 0.757 and its precision is 0.609 on the audit sample. The rege…”
- `09ad75c9` (29w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “Kappa is affected by prevalence and marginal distributions. The report therefore presents…”
- `3c1dd4e2` (84w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “An independent three-class semantic refusal judge does not replicate the Phi-4-mini over-…”
- `c96aea40` (92w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “Human annotation clarifies this discrepancy. On a blinded validation sample, the semantic…”
- `7c0a6d9d` (79w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The XSTest human validation protocol incorporated one disclosed procedural deviation. The…”
- `6e873070` (24w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The human set contains 103 full compliances, 34 partial refusals, and 63 full refusals. T…”
- `a0e201a9` (48w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The broad refusal mapping includes partial refusals alongside complete rejections. While …”
- `a5f8c447` (45w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “In a validation-informed sensitivity analysis where the keyword regex is replaced by the …”
- `74267aba` (56w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The 128-token runs truncated 60.3% of the 2,000 HarmBench responses under the direct pref…”
- `6aaddb0a` (40w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The longer generation limit also changes baseline compliance rates by providing models su…”
- `5f760f63` (49w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Truncation is highly model-dependent. At 128 tokens it affects most Mistral and Phi respo…”
- `c103316e` (48w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “A direct prefix test distinguishes responses that are exact prefixes of the longer output…”
- `6ea914d3` (10w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Table 5.2. HarmBench classifier deltas at 128 and 512 tokens”
- `471e7675` (75w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “Pair \| ΔASR @128 \| ΔASR @512 \| McNemar p @128 \| McNemar p @512 \|\| Qwen3-1.7B (128/5…”
- `d5057f9b` (55w) Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null — “A non-significant statistical test does not establish that the true effect is precisely z…”
- `15278f0e` (69w) Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null — “Four partly independent findings support the reported null. No HarmBench contrast survive…”
- `ca86e935` (42w) Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null — “Capability evaluations possess varying statistical sensitivity due to differing sample si…”
- `91bf1523` (26w) Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null — “Within the detection limits of the experimental design, no increase in harmful compliance…”
- `2753755e` (44w) Chapter 5: Scorer Validation and Robustness § 5.7 Robustness Synthesis — “Each validation experiment addresses one competing explanation. Human annotations expose …”
- `03d77ebb` (48w) Chapter 5: Scorer Validation and Robustness § 5.7 Robustness Synthesis — “No single check is decisive. Together, however, they make a large hidden increase in harm…”
- `041ea2ef` (28w) Chapter 5: Scorer Validation and Robustness § 5.7 Robustness Synthesis — “Systematic analysis of scorer divergence provides deeper methodological insights than unc…”
- `6f480479` (39w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The empirical findings underscore the necessity of jointly evaluating harmful compliance,…”
- `7327ba1c` (36w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “These conclusions apply strictly to the evaluated model architectures, NF4/INT8 precision…”
- `3af673a8` (51w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results support neither the claim that quantization inherently improves safety alignm…”
- `fe93659b` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The post hoc refusal-margin analysis is consistent with model-specific prompt-level bound…”
- `e00ae762` (8w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “Table 6.1. Possible explanations for the HarmBench result”
- `612ec9a2` (102w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “Candidate explanation \| Discriminating evidence \| Assessment \|\| True practical preser…”
- `fdce09ce` (61w) Chapter 6: Discussion § 6.2 Threats to Validity — “The matched revisions, shared prompts, fixed decoding controls, and paired tests strength…”
- `0f958bd9` (47w) Chapter 6: Discussion § 6.2 Threats to Validity — “The XSTest human audit deviated from its pre-registered blindness-exclusion rule. The dev…”
- `cf727453` (51w) Chapter 6: Discussion § 6.2 Threats to Validity — “The sensitivity analyses were conducted after the primary runs. Their role is diagnostic,…”
- `46c6e700` (41w) Chapter 6: Discussion § 6.2 Threats to Validity — “Several analysis choices are rule-based, including parser behaviour and interpretation la…”
- `42bde384` (48w) Chapter 6: Discussion § 6.2 Threats to Validity — “HarmBench DirectRequest measures harmful compliance under direct unsafe instructions. It …”
- `070cc14b` (37w) Chapter 6: Discussion § 6.2 Threats to Validity — “Human validation uses one annotator on each 200-item sample. The comparisons are therefor…”
- `2a484d4f` (50w) Chapter 6: Discussion § 6.2 Threats to Validity — “The HarmBench classifier is trained for the benchmark's behaviour criterion, which suppor…”
- `5260fcba` (35w) Chapter 6: Discussion § 6.2 Threats to Validity — “Over-refusal is especially policy-sensitive. A partial refusal can be counted as acceptab…”
- `385d9a16` (46w) Chapter 6: Discussion § 6.2 Threats to Validity — “The five pairs cover four families and a compact-model range, but they are not a random s…”
- `ecc137e8` (42w) Chapter 6: Discussion § 6.2 Threats to Validity — “The study's strongest generalisable contribution lies in its method of matching the weigh…”
- `7b74f692` (36w) Chapter 6: Discussion § 6.2 Threats to Validity — “English-language benchmarks dominate the study. Qwen and other multilingual families may …”
- `b06b4140` (38w) Chapter 6: Discussion § 6.2 Threats to Validity — “The V100 environment is relevant because numerical kernels and supported attention paths …”
- `dfbe62fd` (15w) Chapter 6: Discussion § 6.3 Practical Implications — “Do not inherit a safety claim from an fp16 checkpoint without testing the deployed precis…”
- `b5c33471` (15w) Chapter 6: Discussion § 6.3 Practical Implications — “Use a behaviour-specific semantic scorer for harmful compliance; retain transparent proxi…”
- `9fa90ab4` (16w) Chapter 6: Discussion § 6.3 Practical Implications — “Pair safety with benign-refusal and capability benchmarks so that inability is not report…”
- `92fd7382` (19w) Chapter 6: Discussion § 6.3 Practical Implications — “Prefer INT8 when its smaller compression benefit is sufficient and task accuracy is criti…”
- `787d09ce` (15w) Chapter 6: Discussion § 6.3 Practical Implications — “Treat direct-request results as direct-request results. A separate adversarial evaluation…”
- `19835a1f` (33w) Chapter 6: Discussion § 6.3 Practical Implications — “For system designers, the practical takeaway is that quantized language models should be …”
- `9be97917` (71w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “At the interim-submission point, the planned experimental matrix is complete. The committ…”
- `2a556776` (75w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “The principal result is stable across the committed analysis surfaces. No harmful-complia…”
- `001b9d71` (52w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “Workstream \| Status \| Primary repository evidence \|\| Experiment matrix \| Complete \|…”
- `bfa9e04a` (133w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Three issues materially changed the analysis. First, the original 128-token runs frequent…”
- `61802f4e` (7w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Table 7.2. Problems encountered and evidence-led responses”
- `2a91460c` (63w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Problem \| Response \| Evidence \|\| Generation truncation at 128 tokens \| Re-ran the ma…”
- `34d24dab` (138w) Chapter 7: Progress and Remaining Work § 7.3 Remaining Work — “The scientific submission is not blocked by an unfinished benchmark run. The immediate wo…”
- `16126b10` (96w) Chapter 7: Progress and Remaining Work § 7.4 Submission Timeline — “Date or period \| Milestone \| Planned action \|\| 24 Aug 2026 \| Interim report \| Submi…”

</details>

### Added paragraphs (61)

<details><summary>show</summary>

- `4d5099c9` (77w) Chapter 1: Introduction § 1.3 Research Questions — “RQ1 \| Does NF4 quantization measurably increase harmful compliance relative to matched f…”
- `18fd7f51` (3w) Chapter 1: Introduction § 1.5 Contributions — “Controlled matched-pair design”
- `6c67f43b` (24w) Chapter 1: Introduction § 1.5 Contributions — “The study applies NF4 or LLM.int8 at load time to the same revision-pinned model weights,…”
- `3634d854` (3w) Chapter 1: Introduction § 1.5 Contributions — “Joint behavioural evaluation”
- `5f6c4574` (17w) Chapter 1: Introduction § 1.5 Contributions — “The study evaluates harmful compliance, benign over-refusal, MMLU accuracy, and ARC-Chall…”
- `b387f6cb` (2w) Chapter 1: Introduction § 1.5 Contributions — “Scorer-validation study”
- `d0d0d988` (20w) Chapter 1: Introduction § 1.5 Contributions — “HarmBench and XSTest scoring are cross-checked using the HarmBench classifier, an indepen…”
- `e76322e8` (3w) Chapter 1: Introduction § 1.5 Contributions — “Paired statistical analysis”
- `3207abdf` (14w) Chapter 1: Introduction § 1.5 Contributions — “The study combines bootstrap confidence intervals, exact McNemar tests, multiplicity cont…”
- `a6c634a6` (85w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “TrustLLM evaluates trustworthiness across safety, truthfulness, fairness, robustness, pri…”
- `60b5b8df` (66w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “Instruction following and safety are not independent. The InstructGPT study shows how hum…”
- `c1555c35` (64w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “A separate body of work examines adversarial attacks. Universal suffix attacks can transf…”
- `2423adef` (59w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “Benchmarks of general capability are needed because apparent safety can be caused by redu…”
- `b042804c` (87w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “The three broad trustworthiness suites also illustrate a recurring trade-off in benchmark…”
- `2dd0d3eb` (75w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “SafetyBench's multiple-choice format is valuable for safety knowledge, but knowing the co…”
- `bf7fad41` (74w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “MMLU and ARC measure different forms of multiple-choice capability. MMLU spans many acade…”
- `ce7efbde` (77w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
- `c49cf259` (74w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “Mechanistic studies examine how refusal is represented inside the model. Safety Layers ar…”
- `f10eb40d` (58w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “A refusal detector and a harmfulness classifier answer different questions. StrongREJECT …”
- `0eaa7086` (64w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “SORRY-Bench evaluates safety refusal across 44 fine-grained unsafe topics, 440 balanced i…”
- `e1a1bb51` (70w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “JailbreakBench standardises attack resources, chat templates, system prompts, scoring, an…”
- `a1588cee` (77w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “Direct-request and jailbreak evaluations fail for different reasons. In a DirectRequest t…”
- `9ed53e9a` (90w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “XSTest and OR-Bench complement each other. XSTest is small enough for detailed human insp…”
- `36d74fd3` (83w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “StrongREJECT introduces response quality into jailbreak evaluation [38]. This serves a si…”
- `a224e28d` (100w) Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — “SORRY-Bench grounds refusal evaluation in a human-in-the-loop taxonomy of unsafe topics, …”
- `b4137501` (49w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Do-Not-Answer evaluates 939 risky prompts and 5,634 responses and finds that a trained 60…”
- `1c9acb00` (69w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Prometheus 2 provides an open evaluator for direct assessment and pairwise ranking under …”
- `ac866b09` (47w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Agreement should be reported by model or pair when possible. A pooled agreement score can…”
- `6794451b` (65w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “MT-Bench reports more than 80% agreement between strong LLM judges and human preferences …”
- `55af2592` (55w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Open-weight evaluators reduce dependence on an external service. Prometheus 2 is designed…”
- `cd3e4ee9` (57w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Human annotation is not automatically definitive. Rubrics can be ambiguous, annotators ca…”
- `6deb6472` (56w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “The present validation design follows that principle. The HarmBench sample compares a beh…”
- `0af3e7b6` (56w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “Table 2.1 compares the studies relevant to this experiment rather than a claim that this …”
- `ee41e6c1` (35w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “Table 2.1. Scoped positioning of representative prior work. Entries are compiled from eac…”
- `eaa70134` (226w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “Study \| Precision intervention \| Harmful compliance \| Over-refusal \| Capability \| Sc…”
- `4bbff14f` (62w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The gap is not merely the absence, in Table 2.1, of the same benchmark combination. The m…”
- `50b9aa2d` (11w) Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design — “Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow.”
- `00bbaad0` (73w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “The implementation is a benchmark framework rather than the main research contribution. A…”
- `8f438a00` (63w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Runs were executed through SLURM on an NTU TC1 node with a 32 GB Tesla V100 GPU, and comp…”
- `f11dfb14` (50w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Original raw generations and summaries are treated as immutable run files. Revised scorer…”
- `31fd290a` (63w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Cluster policy also affects reproducibility. User compute runs only through batch jobs, c…”
- `7fc59732` (40w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “Cross-family response styles also affect measurement. Mistral is the clearest case: its r…”
- `f2db43ad` (24w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “At the interim report stage, the literature review, methodology, experimental setup, and …”
- `ec4cea62` (94w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “The primary evidence is sufficiently developed for the interim report to state its centra…”
- `48ad7931` (32w) Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved — “Workstream \| Status \|\| Literature review \| Complete \|\| Methodology and setup \| Com…”
- `573d9689` (50w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “The remaining work is the detailed final-report writing for two chapters whose underlying…”
- `ece581f9` (61w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “The Discussion chapter will then connect the primary results to the literature, explain t…”
- `d6e2d70f` (59w) Chapter 5: Progress and Remaining Work § 5.2 Remaining Work — “Planned chapter \| Evidence already complete \| Writing still required \|\| Scorer Valida…”
- `6d6d57a9` (27w) References — “[13] Y. Huang, L. Sun, et al., “Position: TrustLLM: Trustworthiness in large language mod…”
- `879236ef` (23w) References — “[14] B. Wang et al., “DecodingTrust: A comprehensive assessment of trustworthiness in GPT…”
- `5b5dfc8b` (22w) References — “[15] Z. Zhang et al., “SafetyBench: Evaluating the safety of large language models,” in P…”
- `6f19e83c` (25w) References — “[16] L. Ouyang et al., “Training language models to follow instructions with human feedba…”
- `8efef347` (27w) References — “[17] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal an…”
- `45ac3ef5` (43w) References — “[18] E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, …”
- `4a868421` (31w) References — “[19] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “…”
- `2304129a` (31w) References — “[20] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord…”
- `a8d986e2` (20w) References — “[38] A. Souly et al., “A StrongREJECT for empty jailbreaks,” in Proc. NeurIPS Datasets an…”
- `67ae1a68` (22w) References — “[39] T. Xie et al., “SORRY-Bench: Systematically evaluating large language model safety r…”
- `2638ed4d` (26w) References — “[41] S. Han et al., “WildGuard: Open one-stop moderation tools for safety risks, jailbrea…”
- `ef73ebd3` (28w) References — “[42] J. Cui, W.-L. Chiang, I. Stoica, and C.-J. Hsieh, “OR-Bench: An over-refusal benchma…”
- `67ad6db9` (21w) References — “[45] L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Proc. …”

</details>

### Edited paragraphs (52; change window shown)

<details><summary>show</summary>

- (front matter): “[+FYP ]Interim Report”
- (front matter): “TAN UEI HORNG[− (UTAN001)]”
- (front matter): “…ment of the Requirements for the Degree of Bachelor of [−Engineer][+Comput][−(][+in ][−)]”
- (front matter): “2.[−1][+2][−5][+6]”
- (front matter): “2.[−2][+3][−6][+7]”
- (front matter): “2.[−3][+6][−7][+11]”
- (front matter): “Chapter 3: Methodology and Experimental Setup [−9][+13]”
- (front matter): “3.1 Matched-Pair Experimental Design [−9][+13]”
- (front matter): “3.2 Models and Precision Conditions 1[−0][+4]”
- (front matter): “3.3 Benchmarks and Outcomes 1[−0][+5]”
- (front matter): “3.4 Scoring and Validation 1[−1][+6]”
- (front matter): “3.5 Statistical Analysis 1[−2][+6]”
- (front matter): “Chapter 4: Results 1[−4][+9]”
- (front matter): “4.1 Results at a Glance 1[−4][+9]”
- (front matter): “4.2 Harmful Compliance (RQ1) [−15][+20]”
- (front matter): “4.3 Over-Refusal on Benign Prompts (RQ2) [+2][−6]”
- (front matter): “4.4 Capability (RQ3) [−17][+22]”
- (front matter): “4.5 Model Scale and Family Differences (RQ4) [−18][+23]”
- (front matter): “4.6 INT8 Precision Point [−19][+24]”
- (front matter): “4.7 Research-Question Synthesis 2[−0][+5]”
- (front matter): “[+2.][−.4][−XSTes][+Au][+omatic][−Sc][+Evaluat][−er Sen][−itivity][−23][+10]”
- (front matter): “Chapter [−7][+5][−9][+6]”
- (front matter): “[−7][+5][−9][+6]”
- (front matter): “[−7][+5][−3][+2][−30][+26]”
- (front matter): “References [−31][+28]”
- Chapter 1: Introduction § 1.1 Background and Motivation: “[+Compact open-weight language models are becoming increasingly feasible for local deploy…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…l prompts while also refusing benign prompts more often[−,] or appears safer because…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…4 at load time to the same weights as the fp16 baseline[−;][+, while][−Section 1.4 state…”
- Chapter 1: Introduction § 1.2 Problem Statement: “…s an object of study rather than as a fixed instrument.[− Chapter 4 reports the outcome …”
- Chapter 1: Introduction § 1.2 Problem Statement: “…eports the results after multiple-comparison correction[−,][−Ch][+summ][−pte][− 5 exam][…”
- Chapter 1: Introduction § 1.4 Scope: “…nes. The safety threat model is HarmBench DirectRequest[−, under which][+:] models recei…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…on on aggregate tasks, especially as model size grows [[−7][+3][−–][+, [8], ][−0][+8][− …”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…iment because it observes broad four-bit preservation [[−2][+1][−.][+:][−B][+b][+both ][…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…monstrate that targeted worst-case failures can occur [[−25][+33][−29][+37][−3][+4][+4][…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…the loading precision. The quantization settings and de[+clared de][−r][+sp][−ord][+ifi]…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…erature 0.0, and a 512-token maximum generation budget.[+ Figure 3.1 shows the study des…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “… NF4 set of corrected comparisons contains 20 contrasts[−, corresponding to][+:][−eva][+…”
- Chapter 4: Results: “This chapter reports the primary matched-pair results [−b][+and summariz][−fore][+s][+ co…”
- Chapter 4: Results § 4.1 Results at a Glance: “…viving over-refusal effect is a decrease for Phi-4-mini[−,][+.][−w][+T][−i][+e s][−h Cha…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “… for both precisions. The aggregate null does not mean [−that ][−.][+:][−S][+s]ixteen pr…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “…ces the individually significant decrease. Section 4.4 [+ex][−rov][+la][−de][+n][−necess…”
- Chapter 4: Results § 4.4 Capability (RQ3): “Llama-3.2-3B has an ARC contrast of −0.032 (unrounded[− value][+:] −0.0316), which surviv…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…s, whereas several smaller observed deltas do not. The [+power ][+t][−por][+a][+smaller …”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “… −0.0033. However, that contrast is much smaller on ARC[−, where][+:] the 1.7B decrease …”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “… are based only on the direction of the point estimates[−.][+:] Qwen3-4B and Phi show po…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…s not follow one shared path across fp16, INT8, and NF4[− because][+:][+,][−o][+res][−he…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…d ±0.05 margin. However, this result is method-specific[− because][+:] LLM.int8 and NF4 …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “Qwen3-1.7B illustrates the large capability decrease[−.][+:] MMLU is slightly higher …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…me pattern. Qwen3-4B rises from 0.115 to 0.125 to 0.155[−.][+,][−T][+but neit][+r][−INT8…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “… with p = 1.000; API-judge delta +0.010 with p = 0.688)[−.][+,][−T][+w][−s budget-sensit…”
- Chapter 5: Progress and Remaining Work § 5.1 Progress Achieved: “Table [−7][+5].1. Completed work and re…”
- Chapter 5: Progress and Remaining Work § 5.2 Remaining Work: “Table [−7][+5][−3][+2][−m][+f][+na][−estones ][+-][−c][+p][−ded][+t][+writ][− PROJECT_LOG…”

</details>

### Citation-renumber-only paragraphs: 54 (text identical once [n] markers stripped; details in .full.json)

### Renamed headings (6)
- “2.1 Quantization Methods and Capability Effects” → “2.2 Quantization Methods and Capability Effects”
- “2.2 Quantization and Safety Behaviour” → “2.3 Quantization and Safety Behaviour”
- “2.3 Synthesis and Research Gap” → “2.6 Synthesis and Research Gap”
- “Chapter 7: Progress and Remaining Work” → “Chapter 5: Progress and Remaining Work”
- “7.1 Progress Achieved” → “5.1 Progress Achieved”
- “7.3 Remaining Work” → “5.2 Remaining Work”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
