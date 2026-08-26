## Rewritten interim report (submission line): Interim v4 (funnel intro, comprehensive fallback) → Interim compact (progress-centred)

*From:* `docs/rewritten_fyp_interim_v4.docx`  
*To:* `docs/rewritten_fyp_interim_compact.docx`  
*Decisions:* D68, D70, D71, D72  

**Paragraph/body blocks:** +19 added · −166 removed · ~74 edited · ⟳47 citation-renumber-only · tables: +3 −6 ~1 · front matter blocks 18→16

*Front matter (cover/abstract/TOC/lists):* +3 −41 ~32 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Acknowledgements
  - ❌ I thank my supervisor, Dr. Zhang Jiehuang, for his guidance throughout this project. I also thank t…
  - ❌ List of Figures
  - ❌ Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance 3
  - ❌ Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow 15
  - ❌ Figure 4.1. Per-pair HarmBench classifier ΔASR with paired confidence intervals 24
  - ❌ Figure 4.2. Precision sweep across fp16, INT8, and NF4 28
  - ❌ Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy 30
  - ❌ Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity 34
  - ❌ List of Tables
  - ❌ Table 2.1. Positioning of representative prior work 11
  - ❌ Table 3.1. Model pairs and precision conditions 15

### Chapters / parts REMOVED
- ❌ Appendix A: Reproducibility Summary
- ❌ Appendix B: Minimum Detectable Capability Effects
- ❌ Appendix C: Human-Validation Boundaries

### Sections removed
- ❌ Chapter 1: Introduction § 1.5 Contributions — 1.5 Contributions
- ❌ Chapter 1: Introduction § 1.6 Report Structure — 1.6 Report Structure
- ❌ Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — 2.1 Trustworthiness and Safety Evaluation
- ❌ Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models — 2.4 Refusal, Harmfulness, and Threat Models
- ❌ Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — 2.5 Automatic Evaluators and Human Valid…
- ❌ Chapter 2: Literature Review § 2.7 Implications for Experimental Design — 2.7 Implications for Experimental Design
- ❌ Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — 3.6 Implementation and Execution
- ❌ Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — 3.7 Reproducibility Controls
- ❌ Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — 3.8 Analysis Workflow and Q…
- ❌ Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — 5.6 Multi-Seed Decoding Sensitivity
- ❌ Chapter 6: Discussion § 6.2 Relationship to Prior Work — 6.2 Relationship to Prior Work
- ❌ Chapter 6: Discussion § 6.3 Capability Effects and Deployment — 6.3 Capability Effects and Deployment
- ❌ Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — 6.4 Validity of the Scoring Methods
- ❌ Chapter 7: Conclusion and Future Work § 7.1 Conclusions — 7.1 Conclusions
- ❌ Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — 7.2 Prioritised Future Work

### Sections added
- ➕ Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — 7.1 Progress Achieved
- ➕ Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — 7.2 Problems Encountered and Resolu…
- ➕ Chapter 7: Progress and Remaining Work § 7.3 Remaining Work — 7.3 Remaining Work
- ➕ Chapter 7: Progress and Remaining Work § 7.4 Submission Timeline — 7.4 Submission Timeline

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Reproducibility Summary | 0 | 3 | 0 | 0 |
| Appendix B: Minimum Detectable Capability Effects | 0 | 2 | 0 | 0 |
| Appendix C: Human-Validation Boundaries | 0 | 3 | 0 | 0 |
| Chapter 1: Introduction | 0 | 6 | 5 | 0 |
| Chapter 2: Literature Review | 2 | 30 | 3 | 10 |
| Chapter 3: Methodology and Experimental Setup | 1 | 16 | 3 | 4 |
| Chapter 4: Results | 0 | 0 | 13 | 0 |
| Chapter 5: Scorer Validation and Robustness | 1 | 10 | 15 | 0 |
| Chapter 6: Discussion | 2 | 25 | 3 | 1 |
| Chapter 7: Conclusion and Future Work | 0 | 13 | 0 | 0 |
| Chapter 7: Progress and Remaining Work | 10 | 0 | 0 | 0 |
| References | 0 | 17 | 0 | 32 |

### Removed paragraphs (125 body; 41 front matter)

<details><summary>show</summary>

- `7f51116b` (27w) Chapter 1: Introduction § 1.5 Contributions — “Controlled matched-pair design. The study applies NF4 or LLM.int8 at load time to the sam…”
- `c0f2f1c2` (20w) Chapter 1: Introduction § 1.5 Contributions — “Joint behavioural evaluation. The study evaluates harmful compliance, benign over-refusal…”
- `3491f7c6` (17w) Chapter 1: Introduction § 1.5 Contributions — “Paired statistical analysis. The study combines bootstrap confidence intervals, exact McN…”
- `2fe7a6d7` (20w) Chapter 1: Introduction § 1.5 Contributions — “Scorer-validation study. HarmBench results are cross-checked using the HarmBench classifi…”
- `6b01873f` (20w) Chapter 1: Introduction § 1.5 Contributions — “Reproducible benchmarking pipeline. The implementation preserves immutable raw model outp…”
- `a69a0820` (80w) Chapter 1: Introduction § 1.6 Report Structure — “Chapter 2 reviews quantization, safety and refusal evaluation, using capability results t…”
- `a6c634a6` (85w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “TrustLLM evaluates trustworthiness across safety, truthfulness, fairness, robustness, pri…”
- `60b5b8df` (66w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “Instruction following and safety are not independent. The InstructGPT study shows how hum…”
- `c1555c35` (64w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “A separate body of work examines adversarial attacks. Universal suffix attacks can transf…”
- `2423adef` (59w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “Benchmarks of general capability are needed because apparent safety can be caused by redu…”
- `b042804c` (87w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “The three broad trustworthiness suites also illustrate a recurring trade-off in benchmark…”
- `2dd0d3eb` (75w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “SafetyBench's multiple-choice format is valuable for safety knowledge, but knowing the co…”
- `bf7fad41` (74w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “MMLU and ARC measure different forms of multiple-choice capability. MMLU spans many acade…”
- `ce7efbde` (77w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
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
- `98cb3301` (226w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “Study \| Precision intervention \| Harmful compliance \| Over-refusal \| Capability \| Sc…”
- `4bbff14f` (62w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The gap is not merely the absence, in Table 2.1, of the same benchmark combination. The m…”
- `e9e161c3` (66w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “The reviewed literature suggests that a convincing quantization-safety experiment must co…”
- `785e94b6` (69w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “Statistical design is equally important. Twenty model–outcome contrasts create many oppor…”
- `f0c9fa40` (76w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `c9becd11` (52w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “Finally, research integrity affects interpretation. Revised scoring should not overwrite …”
- `50b9aa2d` (11w) Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design — “Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow.”
- `f91f5485` (60w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Cohen's kappa measures chance-corrected agreement between scorers [50]. Sample-size and m…”
- `00bbaad0` (73w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “The implementation is a benchmark framework rather than the main research contribution. A…”
- `8f438a00` (63w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Runs were executed through SLURM on an NTU TC1 node with a 32 GB Tesla V100 GPU, and comp…”
- `f11dfb14` (50w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Original raw generations and summaries are treated as immutable run files. Revised scorer…”
- `d9910245` (47w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “The command-line interface exposes single-run, matrix, analysis, and agent-status paths. …”
- `31fd290a` (63w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Cluster policy also affects reproducibility. User compute runs only through batch jobs, c…”
- `c2965a8e` (42w) Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution — “Testing covers configuration validation, model reuse, result resumption, refusal parsing,…”
- `c9fd291c` (80w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “Every model revision in the primary configuration is pinned, and the report records the s…”
- `3eb7a16a` (96w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “The project uses automated tests and claim verification to detect stale numbers and incon…”
- `6fd56950` (67w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “Reproduction has two levels. Analysis replay starts from the stored redacted scores and r…”
- `e92f2f4d` (45w) Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls — “The report avoids a stronger claim of exact historical rerun because not every earliest p…”
- `d8f6407f` (68w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “The analysis begins by joining baseline and quantized records on prompt identifier, and c…”
- `80117cad` (49w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Multiplicity correction is performed from the complete 20-row analysis record rather than…”
- `d786c363` (62w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Figures are produced by scripts from the analysis JSON rather than drawn manually. The fo…”
- `22c55306` (56w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Quality control has both machine and human components. Tests detect schema failures, stal…”
- `c66d893b` (50w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “Figure 5.1 compares the HarmBench classifier with the refusal-regex proxy. The proxy syst…”
- `77c87660` (12w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy and their agreement.”
- `51fb1eee` (62w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “The five-seed stochastic-decoding analysis tests whether the greedy result is specific to…”
- `61133bba` (42w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Across multiple decoding seeds, the mean changes for Mistral and Phi remain closely clust…”
- `26558871` (23w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Varying the random decoding seed captures output variance introduced by non-deterministic…”
- `b29248f9` (43w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “One Qwen3-4B seed and one Phi seed are individually significant, in opposite directions, …”
- `fe89291c` (9w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Table 5.3. Five-seed HarmBench sensitivity at the 512-token budget”
- `41fe7d0c` (69w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Pair \| Greedy Δ \| Seed mean Δ \| Seed range \| Seeds p<.05 \|\| Qwen3-1.7B \| +0.000 \|…”
- `e9296fda` (45w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2 illustrates the relationship between the Qwen3-1.7B greedy point estimate and …”
- `15d7cad5` (9w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity.”
- `35f26245` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results support neither common oversimplification: neither that quantization inherent…”
- `b214d2ad` (31w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more accurate characterisation of low-bit quantization is prompt-level decision boundar…”
- `560085e1` (49w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The result is closest to Decoding Compressed Trust, which reports broad preservation of t…”
- `4725b6bc` (80w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Safety-degradation studies remain relevant. Kharinaev et al. report that quantization can…”
- `3b7bb1b6` (82w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2 compares published results with those of this study rather than leaving the com…”
- `2dc1dbb9` (78w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2. Comparison with published quantization results. Values are transcribed from ea…”
- `7c2a4703` (475w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Study \| Quantized model \| Reported change, in the source's own metric \| Scorer \| Chie…”
- `51646ed1` (117w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Two features of the table deserve emphasis because they are easy to misread. In the IEEE …”
- `3db5c862` (54w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesising the published literature reinforces the contextual nature of safety findings…”
- `0d0dc949` (63w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The scorer findings align with StrongREJECT's warning about empty jailbreaks [38] and SOR…”
- `4d54570d` (55w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The capability findings also qualify earlier capability-focused quantization work. The th…”
- `b3f40fb5` (47w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Marchisio et al.'s multilingual result is methodologically relevant even though the prese…”
- `360bcef1` (48w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “The direct-request scope explains why adversarial studies should not be used as either co…”
- `3c6db3a3` (55w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The capability result is practically important because a compressed model is chosen for d…”
- `4e4076d6` (59w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “INT8 provides a useful intermediate point. Its capability contrasts are close to fp16 wit…”
- `dabccde2` (52w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The two benchmarks of general capability also caution against a single-number deployment …”
- `a9b97883` (70w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “The deployment recommendation is limited because latency, memory use, and energy consumpt…”
- `c854dcb1` (52w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “Deployment evaluation should begin with the requirements of the intended application. If …”
- `93404a4c` (49w) Chapter 6: Discussion § 6.3 Capability Effects and Deployment — “Safety should be evaluated using the final deployment configuration. A pre-quantized chec…”
- `ea00eff4` (42w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Validating scoring rubrics is central to rigorous empirical evaluation. The HarmBench key…”
- `38b87d9f` (62w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “At the same time, semantic judges introduce their own limitations. The HarmBench API judg…”
- `281cb713` (61w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “This has a general implication for LLM evaluation. A benchmark score should be understood…”
- `623fc063` (57w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The two scorer failures point in opposite directions. On HarmBench, a refusal proxy over-…”
- `f2b8a59d` (45w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Both the preregistered regex analysis and the validated semantic judge evaluation are rep…”
- `3d6b9a17` (52w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The protocol deviation demonstrates the same principle. The blindness exclusion was more …”
- `65a5546d` (36w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This research conducted a controlled empirical evaluation of five matched compact languag…”
- `983160fc` (95w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “In summary, RQ1 is answered by a robust null: no NF4 change in harmful compliance remains…”
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
- `f879f1b3` (12w) References — “[58] J. Gu et al., “A survey on LLM-as-a-judge,” arXiv:2411.15594, 2024 (preprint).”
- `184b440f` (23w) References — “[59] M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human gro…”
- `8546d089` (17w) References — “[60] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. …”
- `2010eeed` (18w) References — “[61] J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale,…”
- `ef85f809` (5w) Appendix A: Reproducibility Summary — “Table A.1. Primary run controls”
- `e103bd5b` (225w) Appendix A: Reproducibility Summary — “Control \| Value \|\| Primary configuration \| configs/tc1_512.yaml \|\| Model pairs \| 5…”
- `c0a4b867` (31w) Appendix A: Reproducibility Summary — “The repository contains the pinned model revisions, full configuration schema, generated …”
- `2b478133` (11w) Appendix B: Minimum Detectable Capability Effects — “Table B.1. Observed capability effects and 80% power minimum detectable effects”
- `5dd3900a` (134w) Appendix B: Minimum Detectable Capability Effects — “Pair \| Benchmark \| n \| Observed Δ \| MDE \| \|Δ\| > MDE \|\| Qwen3-1.7B \| MMLU \| 300…”
- `de0a2e06` (40w) Appendix C: Human-Validation Boundaries — “Both human-validation sets contain 200 labelled responses and one annotator. The HarmBenc…”
- `63dcf5aa` (29w) Appendix C: Human-Validation Boundaries — “No prompt or response text is reproduced in this report. The stored validation files cont…”
- `007569ff` (65w) Appendix C: Human-Validation Boundaries — “The XSTest blindness exclusion was applied prompt-wide instead of to the limited alias–pr…”

</details>

### Added paragraphs (16)

<details><summary>show</summary>

- `2187b631` (77w) Chapter 2: Literature Review § 2.2 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
- `24a5cb08` (61w) Chapter 2: Literature Review § 2.3 Synthesis and Research Gap — “The gap is not merely the absence of the same benchmark combination in prior work. The ma…”
- `d6064dd7` (60w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Cohen's kappa measures chance-corrected agreement between scorers [37]. Sample-size and m…”
- `0002377c` (56w) Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy — “The HarmBench classifier and the refusal-regex proxy diverge because the proxy systematic…”
- `3af673a8` (51w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results support neither the claim that quantization inherently improves safety alignm…”
- `fe93659b` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The post hoc refusal-margin analysis is consistent with model-specific prompt-level bound…”
- `9be97917` (71w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “At the interim-submission point, the planned experimental matrix is complete. The committ…”
- `2a556776` (75w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “The principal result is stable across the committed analysis surfaces. No harmful-complia…”
- `8c28d73b` (7w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “Table 7.1. Completed work and repository evidence”
- `001b9d71` (52w) Chapter 7: Progress and Remaining Work § 7.1 Progress Achieved — “Workstream \| Status \| Primary repository evidence \|\| Experiment matrix \| Complete \|…”
- `bfa9e04a` (133w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Three issues materially changed the analysis. First, the original 128-token runs frequent…”
- `61802f4e` (7w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Table 7.2. Problems encountered and evidence-led responses”
- `2a91460c` (63w) Chapter 7: Progress and Remaining Work § 7.2 Problems Encountered and Resolutions — “Problem \| Response \| Evidence \|\| Generation truncation at 128 tokens \| Re-ran the ma…”
- `34d24dab` (138w) Chapter 7: Progress and Remaining Work § 7.3 Remaining Work — “The scientific submission is not blocked by an unfinished benchmark run. The immediate wo…”
- `3223da3e` (7w) Chapter 7: Progress and Remaining Work § 7.4 Submission Timeline — “Table 7.3. Remaining milestones recorded in PROJECT_LOG.md”
- `16126b10` (96w) Chapter 7: Progress and Remaining Work § 7.4 Submission Timeline — “Date or period \| Milestone \| Planned action \|\| 24 Aug 2026 \| Interim report \| Submi…”

</details>

### Edited paragraphs (74; change window shown)

<details><summary>show</summary>

- (front matter): “2.[−2][+1][−6][+5]”
- (front matter): “2.[−3][+2][−7][+6]”
- (front matter): “2.[−6][+3][−11][+7]”
- (front matter): “Chapter 3: Methodology and Experimental Setup [−14][+9]”
- (front matter): “3.1 Matched-Pair Experimental Design [−14][+9]”
- (front matter): “3.2 Models and Precision Conditions 1[−5][+0]”
- (front matter): “3.3 Benchmarks and Outcomes 1[−6][+0]”
- (front matter): “3.4 Scoring and Validation 1[−7][+1]”
- (front matter): “3.5 Statistical Analysis 1[−7][+2]”
- (front matter): “Chapter 4: Results [−22][+14]”
- (front matter): “4.1 Results at a Glance [−22][+14]”
- (front matter): “4.2 Harmful Compliance (RQ1) [−23][+15]”
- (front matter): “4.3 Over-Refusal on Benign Prompts (RQ2) [−24][+16]”
- (front matter): “4.4 Capability (RQ3) [−25][+17]”
- (front matter): “4.5 Model Scale and Family Differences (RQ4) [−26][+18]”
- (front matter): “4.6 INT8 Precision Point [−27][+19]”
- (front matter): “4.7 Research-Question Synthesis 2[−8][+0]”
- (front matter): “Chapter 5: Scorer Validation and Robustness 2[−9][+1]”
- (front matter): “5.1 HarmBench Classifier versus Refusal Proxy 2[−9][+1]”
- (front matter): “5.2 Independent HarmBench Judges [−30][+21]”
- (front matter): “5.3 Human Validation of Harmful-Compliance Scorers [−31][+22]”
- (front matter): “5.4 XSTest Scorer Sensitivity and Human Validation [+2][−1]”
- (front matter): “5.5 Generation-Length Robustness [−3][+4]”
- (front matter): “5.[−7][+6][−3][+2]4”
- (front matter): “5.[−8][+7][−3][+2]5”
- (front matter): “Chapter 6: Discussion [−3][+2]6”
- (front matter): “6.1 Answer to the Central Question [−3][+2]6”
- (front matter): “[−6][+7][−2][+3][−l][+m][−t][−o][−sh][−p to Prior][+ng][−6][+0]”
- (front matter): “6.[−5][+2][−40][+26]”
- (front matter): “6.[−6][+3][−4][+8]”
- (front matter): “Chapter 7: [−C][+Pr][−nclu][+gre][−ion][+s][−Futur][+R][+maining][−43][+29]”
- (front matter): “References [−45][+31]”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…nd LLM.int8 are the two lower-precision conditions. The[−s][+ir ][− ][+ffec][−ransformat…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…ment change or a loss of capability. Safety evaluation [−ha][+i][−a s][+furth][−cond][+r…”
- Chapter 1: Introduction § 1.2 Problem Statement: “…is study is interpretive rather than merely operational[−:][+.][−a][+A] change in harmfu…”
- Chapter 1: Introduction § 1.2 Problem Statement: “This study therefore measures [−all ][−s on the same prompt][+: harmful compliance, benig…”
- Chapter 1: Introduction § 1.4 Scope: “…nes. The safety threat model is HarmBench DirectRequest[−:][+, under which] models recei…”
- Chapter 2: Literature Review § 2.1 Quantization Methods and Capability Effects: “… substantially larger than the automatic-metric loss [2[−8][+0][−:][+ because][−is][+wou…”
- Chapter 2: Literature Review § 2.2 Quantization and Safety Behaviour: “…iment because it observes broad four-bit preservation [[+2][−1][−:][+.][−b][+B][−both ][…”
- Chapter 2: Literature Review § 2.3 Synthesis and Research Gap: “…monstrate that targeted worst-case failures can occur [[−33][+25][−37][+29][−4][+3][−4][…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…ment fixed while changing only the loading precision. T[−able A.1 records t][−the declar…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…erature 0.0, and a 512-token maximum generation budget.[− Figure 3.1 shows the study des…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “… NF4 set of corrected comparisons contains 20 contrasts[−:][+, corresponding to][−m][+ev…”
- Chapter 4: Results: “… the validation and sensitivity analyses. Each section [−follow][+present][− same order:…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “… for both precisions. The aggregate null does not mean [+that ][−:][+.][−s][+S]ixteen pr…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “…ces the individually significant decrease. Section 4.4 [+provid][−xplain][+necessary con…”
- Chapter 4: Results § 4.4 Capability (RQ3): “Llama-3.2-3B has an ARC contrast of −0.032 (unrounded[−:][+ value] −0.0316), which surviv…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…holds, whereas several smaller observed deltas do not. [−App][+Th][+ a][−d][+alys][−x B]…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “… −0.0033. However, that contrast is much smaller on ARC[−:][+, where] the 1.7B decrease …”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “… are based only on the direction of the point estimates[−:][+.] Qwen3-4B and Phi show po…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “…amily response styles also affect measurement. Mistral [+prov][+de][+example be][+u][−:]…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…s not follow one shared path across fp16, INT8, and NF4[−:][+ because][−,][+othe][−e][−t…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…d ±0.05 margin. However, this result is method-specific[−:][+ because] LLM.int8 and NF4 …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “Qwen3-1.7B illustrates the large capability decrease[−:][+.] MMLU is slightly higher …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…me pattern. Qwen3-4B rises from 0.115 to 0.125 to 0.155[−,][+.][−but neit][+T][−r][−incr…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “… with p = 1.000; API-judge delta +0.010 with p = 0.688)[−,][+.][−w][+T][+s budget-sensit…”
- Chapter 5: Scorer Validation and Robustness: “…y findings depend on measurement choices. This chapter [−tests the central assumptions ]…”
- Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy: “…itten. This preserves a record of both scoring versions[−:][+.][−t][+T]he classifier res…”
- Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges: “…l condition kappa from 0.68 to 0.95. It reproduces the [−direction and ][−ach mod][+v][−…”
- Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges: “… classifier for four pairs. Qwen3-1.7B is the exception[−:][+.][−t][+T]he classifier is …”
- Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers: “…esent baseline population rates. The key takeaway is th[−e rel][−ive fidelity of the eva…”
- Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers: “….235. This error profile matches the conceptual concern[−:][+ that] absence of a refusal…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “… not establish that no such increase exists (Section 5.[−7][+6]).”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “Human[− ground-truth][−T][+Against the single annotator, t][+re][−pture][+ll][− only][−tr…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “…rotocol incorporated one disclosed procedural deviation[−:][+.][−t][+T]he blindness excl…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “…ile this increases the semantic judge's agreement with [+t][−uma][+e si][+gle][−s][+t][−…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “…e Benjamini–Hochberg correction across the entire study[−:][+. These are] the Qwen MMLU …”
- Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null: “[−The reported null finding is supported along f][+F][+fin][−m][+ngs support th][−nsi][+ …”
- Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null: “…ty due to differing sample sizes between MMLU and ARC. [−As detailed in Appendix B, o][+…”
- Chapter 5: Scorer Validation and Robustness § 5.6 Power and the Meaning of the Null: “[−The empirical evidence demonstrates the absence of a statistically significant increase…”
- Chapter 5: Scorer Validation and Robustness § 5.7 Robustness Synthesis: “…lidation experiment addresses one competing explanation[−:][+.][−h][+H]uman annotations …”
- Chapter 6: Discussion § 6.1 Answer to the Central Question: “…ns the earlier Qwen result \|\| Decode-specific result \| [−F][+Qwen3-4B posit][−e-seed …”
- Chapter 6: Discussion § 6.2 Threats to Validity: “The study's strongest generalisable contribution [+l][+es in it][+ ][−logical:][+f][+ing]…”
- Chapter 6: Discussion § 6.3 Practical Implications: “[−T][+For system designers, t][−for system designers ][−procedur][+th][−l:][+t] quantized…”

</details>

### Citation-renumber-only paragraphs: 47 (text identical once [n] markers stripped; details in .full.json)

### Renamed headings (11)
- “2.2 Quantization Methods and Capability Effects” → “2.1 Quantization Methods and Capability Effects”
- “2.3 Quantization and Safety Behaviour” → “2.2 Quantization and Safety Behaviour”
- “2.6 Synthesis and Research Gap” → “2.3 Synthesis and Research Gap”
- “5.7 Power and the Meaning of the Null” → “5.6 Power and the Meaning of the Null”
- “5.8 Robustness Synthesis” → “5.7 Robustness Synthesis”
- “6.5 Threats to Validity” → “6.2 Threats to Validity”
- “6.5.1 Internal Validity” → “6.2.1 Internal Validity”
- “6.5.2 Construct Validity” → “6.2.2 Construct Validity”
- “6.5.3 External Validity” → “6.2.3 External Validity”
- “6.6 Practical Implications” → “6.3 Practical Implications”
- “Chapter 7: Conclusion and Future Work” → “Chapter 7: Progress and Remaining Work”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
