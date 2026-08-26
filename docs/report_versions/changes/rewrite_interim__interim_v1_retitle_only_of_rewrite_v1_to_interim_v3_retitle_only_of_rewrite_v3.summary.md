## Rewritten interim report (submission line): Interim v1 (retitle-only of rewrite v1) → Interim v3 (retitle-only of rewrite v3)

*From:* `docs/rewritten_fyp_interim_v1.docx`  
*To:* `docs/rewritten_fyp_interim_v3.docx`  
*Decisions:* D59, D65  

**Paragraph/body blocks:** +64 added · −64 removed · ~191 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~4 · front matter blocks 18→18

*Front matter (cover/abstract/TOC/lists):* +2 −2 ~60 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Post-training quantization makes compact instruction-tuned language models practical on resource-co…
  - ❌ Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance 2
  - ❌ Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow 15
  - ❌ Figure 4.1. Per-pair HarmBench classifier ΔASR with paired confidence intervals 24
  - ❌ Figure 4.2. Precision sweep across fp16, INT8, and NF4 28
  - ❌ Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy 30
  - ❌ Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity 34
  - ❌ Table 3.1. Model pairs and precision conditions 15
  - ❌ Table 3.2. Benchmark roles, sample sizes, and primary outcomes 16
  - ❌ Table 4.1. NF4 safety and over-refusal results 22
  - ❌ Table 4.2. NF4 capability results 22
  - ❌ Table 5.1. Per-pair HarmBench deltas under three judges 31
  - ➕ Post-training quantization significantly reduces the computational and memory requirements of instr…

### Sections removed
- ❌ Chapter 6: Discussion § 6.4 Scorer Validity as a Research Result — 6.4 Scorer Validity as a Research Result

### Sections added
- ➕ Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — 6.4 Validity of the Scoring Methods

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Reproducibility Summary | 0 | 0 | 2 | 0 |
| Appendix C: Human-Validation Boundaries | 0 | 0 | 2 | 0 |
| Chapter 1: Introduction | 9 | 9 | 9 | 0 |
| Chapter 2: Literature Review | 7 | 7 | 25 | 0 |
| Chapter 3: Methodology and Experimental Setup | 2 | 2 | 35 | 0 |
| Chapter 4: Results | 11 | 11 | 18 | 0 |
| Chapter 5: Scorer Validation and Robustness | 16 | 16 | 13 | 0 |
| Chapter 6: Discussion | 10 | 10 | 24 | 0 |
| Chapter 7: Conclusion and Future Work | 7 | 7 | 3 | 0 |

### Removed paragraphs (62 body; 2 front matter)

<details><summary>show</summary>

- `670aa556` (94w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact instruction-tuned language models have moved capable generation into a deployment…”
- `edda1567` (83w) Chapter 1: Introduction § 1.1 Background and Motivation — “Post-training quantization addresses that constraint by representing weights, and sometim…”
- `8962c00b` (87w) Chapter 1: Introduction § 1.1 Background and Motivation — “The quantization method papers reviewed here report perplexity, accuracy, latency, or mem…”
- `3582c3db` (89w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour gives a concrete reason for concern. Helpful and harmless assistant tra…”
- `357c7e93` (88w) Chapter 1: Introduction § 1.2 Problem Statement — “Safety evaluation is difficult because refusal is not synonymous with safety. A model tha…”
- `5a09b094` (70w) Chapter 1: Introduction § 1.2 Problem Statement — “This project asks whether quantization changes the balance among harmful compliance, beni…”
- `57196406` (91w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 previews the answer. Each pair is placed by its change in MMLU accuracy and it…”
- `f5e73cbe` (58w) Chapter 1: Introduction § 1.4 Scope — “The capability instruments are a six-subject MMLU subset and ARC-Challenge. They provide …”
- `c6d1e676` (13w) Chapter 1: Introduction § 1.5 Contributions — “A reproducible benchmark implementation with redacted derived sidecars and immutable orig…”
- `64aa999a` (84w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “The three broad trustworthiness suites also illustrate a recurring trade-off in benchmark…”
- `733526ac` (66w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “Mechanistic studies of safety layers offer another perspective. Safety Layers argues that…”
- `719f423d` (43w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Grounding — “Agreement should be reported by model or pair when possible. A pooled coefficient can hid…”
- `295ff979` (58w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “Five gaps follow. First, the quantized and full-precision artefacts should come from the …”
- `b68d779b` (62w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The gap is not merely the absence, in Table 2.1, of the same benchmark combination. The m…”
- `152ce9ba` (66w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature also sets the report's claim boundary. Quantization-conditioned attacks sh…”
- `b4e6fd2b` (64w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A null result also needs positive design. It should state what effect sizes the sample ca…”
- `353bd76c` (55w) Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design — “The independent variable is therefore the precision-loading path, and the dependent varia…”
- `d9ebadf2` (46w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Multiplicity correction is performed from the complete 20-row registry rather than from a…”
- `c4b696ad` (34w) Chapter 4: Results § 4.1 Results at a Glance — “The primary multiplicity family contains 20 NF4 contrasts. Exactly three contrasts surviv…”
- `17f6a409` (37w) Chapter 4: Results § 4.1 Results at a Glance — “The correction therefore moves the report away from a pair-by-pair search for dramatic st…”
- `65c23caf` (49w) Chapter 4: Results § 4.1 Results at a Glance — “Tables 4.1 and 4.2 separate safety from capability so that each comparison remains readab…”
- `5478afb8` (39w) Chapter 4: Results § 4.1 Results at a Glance — “The sign convention is consistent: positive ASR and over-refusal deltas are adverse, whil…”
- `9a4fe2be` (32w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “The cross-family pattern is heterogeneous in direction but homogeneous in inferential sta…”
- `faeb150f` (42w) Chapter 4: Results § 4.4 Capability (RQ3) — “Capability contains the clearest costs. Qwen3-1.7B MMLU falls from 0.643 to 0.553, a decr…”
- `28d51c73` (62w) Chapter 4: Results § 4.4 Capability (RQ3) — “The two anchors prevent overgeneralisation from one benchmark. Qwen3-1.7B has a large MML…”
- `6fa85fe7` (37w) Chapter 4: Results § 4.4 Capability (RQ3) — “Figure 1.1 places all five pairs in the joint space these results define. The capability …”
- `3a270562` (67w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels are descriptive summaries of joint movement. Qwen3-1.7B is broa…”
- `78d3ed5c` (55w) Chapter 4: Results § 4.6 INT8 Precision Point — “The 512-token INT8 rerun also resolves a historical warning. A Llama INT8 increase had ap…”
- `e16624bb` (71w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ5 asks whether the conclusion survives measurement and inference changes. The API judge…”
- `eb95d0ab` (55w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Grounding of Harmful-Compliance Scorers — “The sample is stratified toward scorer disagreements, so its class prevalence is not an e…”
- `f1e51750` (44w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The independent three-class refusal judge does not reproduce the Phi-4-mini regex decreas…”
- `09317d2d` (65w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “Human validation explains the discrepancy. Under the strict definition, the judge reaches…”
- `3e514fa4` (78w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The XSTest human audit contains a disclosed protocol deviation. The blindness exclusion w…”
- `9ac67dc5` (44w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The broad mapping increases the refusal class by counting partial refusals. The judge's k…”
- `cac56756` (39w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The validation-informed multiplicity family replaces only the over-refusal instrument wit…”
- `7da2c72d` (61w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The comparison also changes absolute baseline rates because longer responses give models …”
- `1ab65fde` (49w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Mistral and Phi remain close to zero on average. Their greedy deltas lie just outside nar…”
- `3989dee2` (33w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “A seed is not an independent model replicate. The weights and prompt set are reused, and …”
- `61831944` (48w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2 shows the Qwen3-1.7B greedy headline against each individual seed. The greedy …”
- `32bc7480` (51w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Failure to reject is not proof that the true effect is zero. With 200 paired HarmBench it…”
- `bed896dd` (65w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The null is nevertheless more informative than a single non-significant p-value. It is su…”
- `bea7b931` (44w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Capability nulls have different resolution because MMLU and ARC sample sizes differ. Appe…”
- `e02dbf4f` (35w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The defensible claim is the absence of a corrected increase large enough to be detected i…”
- `b4216fb2` (70w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “The validation results can be read as a sequence of eliminated explanations. The refusal …”
- `f21db879` (48w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “This pattern is more valuable than agreement for its own sake. Where instruments disagree…”
- `0c6358aa` (64w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The study began with a distinction: an apparent safety change may reflect alignment, capa…”
- `4358f8d2` (58w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “This conclusion is deliberately bounded. It does not establish that NF4 preserves all for…”
- `d4b06a9b` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The evidence also rejects two tempting simplifications. Quantization is not reliably safe…”
- `3f7cb1dc` (57w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more useful description is model-specific boundary instability under a capability cost.…”
- `535fdae3` (78w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2 places the published figures beside this study's rather than leaving the compar…”
- `69391981` (84w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Read together, the table supports the narrower reading this report has taken throughout. …”
- `d37f1220` (42w) Chapter 6: Discussion § 6.4 Scorer Validity as a Research Result — “Scorer validation is not an auxiliary engineering check, because it changes the scientifi…”
- `afa0ec4f` (60w) Chapter 6: Discussion § 6.4 Scorer Validity as a Research Result — “For the XSTest judge sensitivity study, its dated pre-registration and subsequent validat…”
- `f04dc365` (34w) Chapter 6: Discussion § 6.5 Threats to Validity — “Human validation uses one annotator on each 200-item sample. The comparisons are therefor…”
- `22e7a8fb` (41w) Chapter 6: Discussion § 6.6 Practical Implications — “The practical message is procedural rather than promotional. NF4 is not declared unsafe, …”
- `f64ea199` (47w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This project evaluated five matched compact-language-model pairs across HarmBench, XSTest…”
- `63f48b81` (89w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “RQ1 is answered by a bounded null: no NF4 harmful-compliance contrast survives the 20-tes…”
- `3e5cddf1` (37w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The central lesson is methodological. Quantization effects cannot be interpreted from a s…”
- `ec932de7` (52w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This report also changes how the contribution is presented. The benchmarking framework, c…”
- `e9f72da5` (56w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “Within its boundaries, the study provides a clear deployment conclusion. INT8 preserves c…”
- `51f2e6be` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “These extensions should preserve the discipline used here: immutable original artefacts, …”
- `6e48b120` (45w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The first two extensions are the most urgent. An adversarial arm would enlarge the safety…”

</details>

### Added paragraphs (62)

<details><summary>show</summary>

- `378b0d76` (89w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact instruction-tuned language models such as Qwen3 [1], the Llama 3 family [2], Phi-…”
- `a5c48389` (85w) Chapter 1: Introduction § 1.1 Background and Motivation — “Post-training quantization is one way to address this constraint: it represents model wei…”
- `f45df69c` (105w) Chapter 1: Introduction § 1.1 Background and Motivation — “Most literature on quantization focuses primarily on perplexity, task accuracy, latency, …”
- `5e1667f2` (86w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour provides a concrete reason to evaluate quantized models directly. Instr…”
- `5106f738` (92w) Chapter 1: Introduction § 1.2 Problem Statement — “Evaluating model safety is challenging because refusal behaviour is not synonymous with t…”
- `c196ec07` (84w) Chapter 1: Introduction § 1.2 Problem Statement — “The objective of this study is to determine how post-training quantization changes harmfu…”
- `48c2ccdf` (95w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 places each model pair's change in MMLU accuracy alongside its change in harmf…”
- `96dcab85` (59w) Chapter 1: Introduction § 1.4 Scope — “General capability is evaluated using a representative six-subject MMLU subset and ARC-Ch…”
- `6b01873f` (20w) Chapter 1: Introduction § 1.5 Contributions — “Reproducible benchmarking pipeline. The implementation preserves immutable raw model outp…”
- `b042804c` (87w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “The three broad trustworthiness suites also illustrate a recurring trade-off in benchmark…”
- `e6fcef1d` (72w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “Mechanistic studies examine how refusal is represented inside the model. Safety Layers ar…”
- `ac866b09` (47w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Agreement should be reported by model or pair when possible. A pooled agreement score can…”
- `3f95ffc3` (61w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature review identifies five gaps. First, the quantized and full-precision model…”
- `4bbff14f` (62w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The gap is not merely the absence, in Table 2.1, of the same benchmark combination. The m…”
- `4e51ca35` (84w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature also defines the empirical boundaries of this research. Prior studies on a…”
- `f0c9fa40` (76w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `742c828b` (55w) Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design — “The independent variable is therefore the precision configuration, and the dependent vari…”
- `80117cad` (49w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Multiplicity correction is performed from the complete 20-row analysis record rather than…”
- `34d0b49f` (56w) Chapter 4: Results § 4.1 Results at a Glance — “The primary statistical family encompasses 20 NF4 comparisons across the five model pairs…”
- `f05446c2` (44w) Chapter 4: Results § 4.1 Results at a Glance — “Applying multiple-comparison correction clarifies that the primary statistically defensib…”
- `71e63bbd` (45w) Chapter 4: Results § 4.1 Results at a Glance — “Tables 4.1 and 4.2 present the safety and capability results. Baseline and NF4 columns di…”
- `8b79307d` (41w) Chapter 4: Results § 4.1 Results at a Glance — “Throughout this report, positive deltas in ASR or over-refusal indicate undesirable incre…”
- `76f7b133` (39w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “The direction of the change differs across model families, but none of the harmful-compli…”
- `d62e1908` (55w) Chapter 4: Results § 4.4 Capability (RQ3) — “The most pronounced and statistically robust effects of NF4 quantization appear on capabi…”
- `9c349d60` (49w) Chapter 4: Results § 4.4 Capability (RQ3) — “Employing two distinct capability benchmarks prevents misleading generalizations from a s…”
- `ba7bd1da` (35w) Chapter 4: Results § 4.4 Capability (RQ3) — “Figure 1.1 places all five pairs in the joint space these results define. The MMLU point …”
- `237c1278` (88w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels are descriptive summaries of the combined changes. Qwen3-1.7B i…”
- `0794e09e` (48w) Chapter 4: Results § 4.6 INT8 Precision Point — “The 512-token INT8 evaluation resolves an earlier experimental anomaly. While Llama INT8 …”
- `1a726d8f` (66w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ5 investigates whether the primary conclusions remain robust across alternative evaluat…”
- `9a122934` (47w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The human audit sample purposefully oversamples scorer disagreements to maximize diagnost…”
- `9b2bc1f5` (58w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “An independent three-class semantic refusal judge does not replicate the Phi-4-mini over-…”
- `f924b68a` (85w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “Human ground-truth annotation clarifies this discrepancy. On a blinded validation sample,…”
- `6c435dd4` (77w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The XSTest human validation protocol incorporated one disclosed procedural deviation: the…”
- `d05b77ab` (47w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The broad refusal mapping includes partial refusals alongside complete rejections. While …”
- `2c35e383` (43w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “In a validation-informed sensitivity analysis where the keyword regex is replaced by the …”
- `cc437b0d` (40w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The longer generation limit also changes baseline compliance rates by providing models su…”
- `61133bba` (42w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Across multiple decoding seeds, the mean changes for Mistral and Phi remain closely clust…”
- `26558871` (23w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Varying the random decoding seed captures output variance introduced by non-deterministic…”
- `e9296fda` (45w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2 illustrates the relationship between the Qwen3-1.7B greedy point estimate and …”
- `3437eabd` (57w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “A non-significant statistical test does not establish that the true effect is precisely z…”
- `e592e2de` (44w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The reported null finding is substantiated across multiple independent dimensions: zero H…”
- `17d9d721` (47w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Capability evaluations possess varying statistical sensitivity due to differing sample si…”
- `737334db` (32w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The empirical evidence demonstrates the absence of a statistically significant increase i…”
- `f941f104` (34w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “The validation experiments systematically eliminate competing hypotheses: human annotatio…”
- `041ea2ef` (28w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Systematic analysis of scorer divergence provides deeper methodological insights than unc…”
- `6f480479` (39w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The empirical findings underscore the necessity of jointly evaluating harmful compliance,…”
- `7327ba1c` (36w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “These conclusions apply strictly to the evaluated model architectures, NF4/INT8 precision…”
- `059f115f` (41w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results reject both common oversimplifications: that quantization inherently improves…”
- `933688e1` (31w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more accurate characterization of low-bit quantization is prompt-level decision boundar…”
- `3b7bb1b6` (82w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2 compares published results with those of this study rather than leaving the com…”
- `3fbc552d` (38w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesizing the published literature reinforces the contextual nature of safety findings…”
- `2960a8c4` (42w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Validating scoring rubrics is central to rigorous empirical evaluation. The HarmBench key…”
- `f2b8a59d` (45w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Both the preregistered regex analysis and the validated semantic judge evaluation are rep…”
- `070cc14b` (37w) Chapter 6: Discussion § 6.5 Threats to Validity — “Human validation uses one annotator on each 200-item sample. The comparisons are therefor…”
- `31a7a9d7` (33w) Chapter 6: Discussion § 6.6 Practical Implications — “The practical takeaway for system designers is procedural: quantized language models shou…”
- `65a5546d` (36w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This research conducted a controlled empirical evaluation of five matched compact languag…”
- `c3e72e87` (87w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “In summary, RQ1 is answered by a robust null: no NF4 change in harmful compliance remains…”
- `8c4bac67` (32w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The central methodological insight is that safety evaluations must not rely solely on key…”
- `4c481d63` (28w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The primary contribution of this work resides in the rigorous empirical evidence and meth…”
- `ab396b5b` (33w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “From an engineering standpoint, INT8 offers a highly reliable precision tier for preservi…”
- `3b336751` (24w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Future investigations should maintain rigorous evaluation standards by preserving immutab…”
- `b1320ed6` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The most critical future extensions include extending safety evaluations to adversarial j…”

</details>

### Edited paragraphs (191; change window shown)

<details><summary>show</summary>

- (front matter): “1.3 Research Questions [−3][+2]”
- (front matter): “1.5 Contributions [−4][+3]”
- (front matter): “1.6 Report Structure [−4][+3]”
- (front matter): “2.5 Automatic Evaluators and Human [−Gr][+Validati][−u][−ding] 10”
- (front matter): “Chapter 3: Methodology and Experimental Setup 1[−3][+4]”
- (front matter): “3.1 Matched-Pair Experimental Design 1[−3][+4]”
- (front matter): “3.2 Models and Precision Conditions 1[−4][+5]”
- (front matter): “3.3 Benchmarks and Outcomes 1[−5][+6]”
- (front matter): “3.4 Scoring and Validation 1[−6][+7]”
- (front matter): “3.5 Statistical Analysis 1[−6][+7]”
- (front matter): “3.6 Implementation and Execution 1[−8][+9]”
- (front matter): “3.7 Reproducibility Controls [−19][+20]”
- (front matter): “3.8 Analysis Workflow and Quality Control [−19][+20]”
- (front matter): “Chapter 4: Results 2[−1][+2]”
- (front matter): “4.1 Results at a Glance 2[−1][+2]”
- (front matter): “4.2 Harmful Compliance (RQ1) 2[−2][+3]”
- (front matter): “4.3 Over-Refusal on Benign Prompts (RQ2) 2[−3][+4]”
- (front matter): “4.4 Capability (RQ3) 2[−4][+5]”
- (front matter): “4.5 Model Scale and Family Differences (RQ4) 2[−5][+6]”
- (front matter): “4.6 INT8 Precision Point 2[−5][+6]”
- (front matter): “4.7 Research-Question Synthesis 2[−6][+8]”
- (front matter): “Chapter 5: Scorer Validation and Robustness 2[−8][+9]”
- (front matter): “5.1 HarmBench Classifier versus Refusal Proxy 2[−8][+9]”
- (front matter): “5.2 Independent HarmBench Judges [−29][+30]”
- (front matter): “5.3 Human [−Gr][+Validati][−u][−ding][−0][+1]”
- (front matter): “5.4 XSTest Scorer Sensitivity and Human Validation 3[−0][+1]”
- (front matter): “5.5 Generation-Length Robustness 3[−1][+2]”
- (front matter): “5.6 Multi-Seed Decoding Sensitivity 3[−2][+3]”
- (front matter): “5.7 Power and the Meaning of the Null 3[−3][+4]”
- (front matter): “5.8 Robustness Synthesis 3[−4][+5]”
- (front matter): “Chapter 6: Discussion 3[−5][+6]”
- (front matter): “6.1 Answer to the Central Question 3[−5][+6]”
- (front matter): “6.2 Relationship to Prior Work 3[−5][+6]”
- (front matter): “6.3 Capability [−a][+Effect][−the][+and][−Cost ][−8][+9]”
- (front matter): “6.5 Threats to Validity [−39][+41]”
- (front matter): “6.6 Practical Implications 4[−1][+2]”
- (front matter): “Chapter 7: Conclusion and Future Work 4[−2][+3]”
- (front matter): “7.1 Conclusions 4[−2][+3]”
- (front matter): “7.2 Prioritised Future Work 4[−2][+3]”
- (front matter): “References 4[−4][+5]”
- (front matter): “Appendix A: Reproducibility Summary 4[−7][+8]”
- (front matter): “Appendix B: Minimum Detectable Capability Effects 4[−8][+9]”
- (front matter): “Appendix C: Human-Validation Boundaries [−49][+50]”
- (front matter): “Figure 1.1. C[−apability-][+h][−chor][+g][−d][+s][−safety space us][−g][+ accuracy][−d][+…”
- (front matter): “… precision arms, benchmark battery, and analysis flow 1[−4][+5]”
- (front matter): “…ench classifier ΔASR with paired confidence intervals 2[−3][+4]”
- (front matter): “…igure 4.2. Precision sweep across fp16, INT8, and NF4 2[−6][+8]”
- (front matter): “…armBench classifier ASR versus the refusal-regex proxy [−29][+30]”
- (front matter): “….7B greedy result and five-seed HarmBench sensitivity 3[−3][+4]”
- (front matter): “Table 3.1. Model pairs and precision conditions 1[−4][+5]”
- (front matter): “…. Benchmark roles, sample sizes, and primary outcomes 1[−5][+6]”
- (front matter): “Table 4.1. NF4 safety and over-refusal results 2[−1][+2]”
- (front matter): “Table 4.2. NF4 capability results 2[−1][+2]”
- (front matter): “…able 5.1. Per-pair HarmBench deltas under three judges [−29][+31]”
- (front matter): “….2. HarmBench classifier deltas at 128 and 512 tokens 3[−2][+3]”
- (front matter): “Table 5.3. Five-seed HarmBench sensitivity 3[−2][+3]”
- (front matter): “Table 6.1. [−C][+P][−mp][+ssibl][−ting][−n][+res][−l][+t][−5][+6]”
- (front matter): “Table 6.2. [−P][+Comparison with p][+r][−ffec][+sul][−placed beside this study's ][−6][+7]”
- (front matter): “Table A.1. Primary run controls 4[−7][+8]”
- (front matter): “….1. Capability effects and minimum detectable effects 4[−8][+9]”
- Chapter 1: Introduction § 1.2 Problem Statement: “Figure 1.1. C[−apability-][+h][−chor][+g][−d][+s][−safety space us][−g][+ accuracy][−d][+…”
- Chapter 1: Introduction § 1.3 Research Questions: “…ilies, model scales, and the additional INT8 precision [−p][+c][+nd][+tio][−t]?”
- Chapter 1: Introduction § 1.3 Research Questions: “… conclusion remain credible under alternative scorers, [+comparison with ][−gr][+ann][−u…”
- Chapter 1: Introduction § 1.4 Scope: “…ma-3.2-3B, Mistral-7B, and Phi-4-mini. A separate INT8 [+comp][−m][+ison][−r][+study do]…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A c][+C][+.][+The s][−hat][+udy][+model ][+, keeping the checkpoint constant while chan…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A j][+J][+ behavioural][+.][−of][+The study evaluates][+ accuracy][+curacy ac][+ matche…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A p][+P][+.][−w][+The study comb][−th][+nes] bootstrap confidence int…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A s][+S][+. HarmBench results are cross-checked] using the HarmBench clas…”
- Chapter 1: Introduction § 1.6 Report Structure: “…2 reviews quantization, safety and refusal evaluation, [+using ][−anch][+results t][−r][…”
- Chapter 2: Literature Review: “…mpression method, the safety construct, the capability [−a][+be][−o][+ma][+k][+i][−r][+n…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “A separate [−l][+body of work exam][−terature studi][+n][−a][+how][−e][+a][+s][−u][+respo…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “[−C][+Benchmarks of general c][− anchors][−p][+ca][−rcha][−in][+reduced cap][−a][+occu][−…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…easurement uncertainty is one reason the present study [−tr][+valu][+e][−a][+i][+y expl]…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “MMLU and ARC [+me][−l][−o][+ure][+ent][−in][+forms][−use][+o][+ m][+tiple-choice][−w][+c]…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…eatures through a higher-precision path [33]. The INT8 [+comp][−m][+ison][−p][+c][+nd][+…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…acy can remain stable while other numerical properties [−mov][+chang][+responses on ][−b…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “GPTQ, AWQ, SmoothQuant, NF4, and LLM.int8 pr[−ot][−ct][+serve][+parts of the model's ][−s…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “The [−sc][−le][+mount][− the] calibration data also di…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “Capability[−-foc][+ st][−s][+di][−d evaluation][−a][+context][−use][−ul pri][+ the][−a][+…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “The INT8 [+comp][−m][+ison][−a][+the de][−g][+ree of comp][−ve][+o][−ess][+ the quantizat…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “[−The s][+S][−-specific][+compare ][− i][+e][+t][−h][+yp][+s of model pa][− comparison un…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “They also [+use ][+ent][+scor][+g][−outco][− c][+th][−n][+d][−truction]. Some studies use…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “[−The a][+A][−establi][− p][−ssibility rather][+w][+t failures ca][−p][+occu][+, but not …”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “Decoding Compressed Trust is the [−most compatible ][+study most simila][+ to this ][−sul…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “JailbreakBench standardises attack [−a][−t][−fa][+sour][−t][+e]s, chat templates, system…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “Direct[− ][+-][−s][−s][+ev][−so alloc][+u][−e causal resp][+i][−ib][+ fa][−ity][+ for][−l…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “XSTest and OR-Bench [−offer another ][+ e][−ry][+ch][−pai][+othe][−can][+does ][−laim][+o…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…ces response quality into jailbreak evaluation [41]. Th[−at move ][−conc][+s][+rves a si…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…finer-grained harmfulness scoring as future work [42]. [−That binary refusal is itself a…”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “… tests construct coverage, and a human-labelled sample [−gr][+is used t][+ eval][−nds][+…”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…rotocol changed. It then uses the human set to compare [−in][−t][+co][−um][−nt][+r][−n ]…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “Table 2.1 [+compares the studies relevant to th][−a sco][+ex][−d pos][+r][+men][−ioning a…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “… primary intervention, INT8 as an additional precision [−p][+c][+nd][+tio][−t][+ benchma…”
- Chapter 2: Literature Review § 2.7 Implications for Experimental Design: “…etention and do not frame their outcomes as a declared [+set of corrected co][−ult][+par…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…ame model identifier and pinned revision. The baseline [−is l][+m][−a][−d][+l][−at][+use…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…t quantization with double quantization and a compute d[+ata ][−th][+m][+ched][−f][+t][−…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…s a shorter pilot budget from silently determining the [−he][+m][−dl][−e][+ conclusion].”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…nalysis chapter because pairing is part of the design, [+specified before the fi][−ot ][…”
- Chapter 3: Methodology and Experimental Setup § 3.2 Models and Precision Conditions: “…family. However, the Llama, Mistral, and Phi additions [−i][+broade][−c][+ the ][+ang][−…”
- Chapter 3: Methodology and Experimental Setup § 3.2 Models and Precision Conditions: “[−E][+Both members of e][+u][+e t][+e s][−r][+m][−s its][− exactly], which is a stronger …”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “Benchmark \| Role \| Sample size per [−a][+mode][+ cond][−as][+tion][−a][+be][−o][+ma][+k…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…ragmatic features that can trigger exaggerated safety. [−Its][+The][−-][+regi][−p][+t][−…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…chmarks are interpreted jointly in discussion, but the [+set of 20 ][+NF4 co][−ult][+par…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “… and compute budget. HarmBench uses 200 behaviours per [−a][+mode][+ cond][−as][+tion], …”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…erator applies the model's native chat template with th[+e model's optional th][+ mode][…”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “…ama Guard. A stratified 200-item human-labelled sample [−gro][+eval][−nd][+ate]s both th…”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “…te 200-item human sample evaluates both mappings. This [−a][+p][+ese][−ang][+v][−ment ke…”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “…n. Llama Guard, however, is intentionally treated as a [−l][+c][+mparis][+n u][+ing a br…”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “Human-validation items are stored outside the public re[+po][−ul][+i][− su][+o][−face][+y…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “The primary NF4 [+set of corrected co][−ult][+par][−plicity family][+sons][−he][+m][−dl][…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…ns follow paired-proportion power analysis [54], [55]. [−A][+The a][−m][+analysis r][−th…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “Llama Guard [−su][−pl][+rov][+d][−the][+an][+comparison for ][−-moder][+ classific][− cro…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…alify the interpretation without silently changing the [+set of 20 ][+NF4 co][−ult][+par…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…d paired difference is then (b−c)/n under the report's [−harmful-ward ][+ for harmful co…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…est answer related but not identical questions, both re[−m][+sults ][−in][+re][−visibl][…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…ses under its assumptions. However, it does not remove [−co][+measureme][−struc][+.][−fr…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “… and alignment improvement is the converse. Capability [+loss with lower harmful ][+mp][…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…repare inputs and calculate benchmark-specific fields. [−M][+The m][−o][−ch][+unn][−st][…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…ode with a 32 GB Tesla V100 GPU, and compute jobs used [−offline ][−m][−d][+fflin][+ set…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…generations and summaries are treated as immutable run [−art][+fil][−fact][−si][+rived s…”
- …and 71 more (see .full.json)

</details>

### Renamed headings (3)
- “2.5 Automatic Evaluators and Human Grounding” → “2.5 Automatic Evaluators and Human Validation”
- “5.3 Human Grounding of Harmful-Compliance Scorers” → “5.3 Human Validation of Harmful-Compliance Scorers”
- “6.3 Capability as the Deployment Cost” → “6.3 Capability Effects and Deployment”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
