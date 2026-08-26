## Rewritten full report (plain-academic line): Rewrite v1 (corpus-informed) → Rewrite v2 (plain-academic pass)

*From:* `docs/rewritten_fyp_report_v1.docx`  
*To:* `docs/rewritten_fyp_report_v2.docx`  
*Decisions:* D57, D58, D59, D63, D65  

**Paragraph/body blocks:** +63 added · −63 removed · ~188 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~4 · front matter blocks 18→18

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
  - ➕ Post-training quantization reduces the resources needed to run instruction-tuned language models, b…

### Sections removed
- ❌ Chapter 6: Discussion § 6.4 Scorer Validity as a Research Result — 6.4 Scorer Validity as a Research Result

### Sections added
- ➕ Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — 6.4 Validity of the Scoring Methods

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Reproducibility Summary | 0 | 0 | 2 | 0 |
| Appendix C: Human-Validation Boundaries | 0 | 0 | 2 | 0 |
| Chapter 1: Introduction | 8 | 8 | 6 | 0 |
| Chapter 2: Literature Review | 7 | 7 | 25 | 0 |
| Chapter 3: Methodology and Experimental Setup | 2 | 2 | 35 | 0 |
| Chapter 4: Results | 11 | 11 | 18 | 0 |
| Chapter 5: Scorer Validation and Robustness | 16 | 16 | 13 | 0 |
| Chapter 6: Discussion | 10 | 10 | 24 | 0 |
| Chapter 7: Conclusion and Future Work | 7 | 7 | 3 | 0 |

### Removed paragraphs (61 body; 2 front matter)

<details><summary>show</summary>

- `670aa556` (94w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact instruction-tuned language models have moved capable generation into a deployment…”
- `edda1567` (83w) Chapter 1: Introduction § 1.1 Background and Motivation — “Post-training quantization addresses that constraint by representing weights, and sometim…”
- `8962c00b` (87w) Chapter 1: Introduction § 1.1 Background and Motivation — “The quantization method papers reviewed here report perplexity, accuracy, latency, or mem…”
- `3582c3db` (89w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour gives a concrete reason for concern. Helpful and harmless assistant tra…”
- `357c7e93` (88w) Chapter 1: Introduction § 1.2 Problem Statement — “Safety evaluation is difficult because refusal is not synonymous with safety. A model tha…”
- `5a09b094` (70w) Chapter 1: Introduction § 1.2 Problem Statement — “This project asks whether quantization changes the balance among harmful compliance, beni…”
- `57196406` (91w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 previews the answer. Each pair is placed by its change in MMLU accuracy and it…”
- `f5e73cbe` (58w) Chapter 1: Introduction § 1.4 Scope — “The capability instruments are a six-subject MMLU subset and ARC-Challenge. They provide …”
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

### Added paragraphs (61)

<details><summary>show</summary>

- `8f15566b` (97w) Chapter 1: Introduction § 1.1 Background and Motivation — “Smaller instruction-tuned language models can now run on consumer GPUs and some edge devi…”
- `e3ac9f09` (79w) Chapter 1: Introduction § 1.1 Background and Motivation — “Post-training quantization reduces memory use by representing weights, and sometimes acti…”
- `62f4bdab` (81w) Chapter 1: Introduction § 1.1 Background and Motivation — “Most quantization studies report perplexity, task accuracy, latency, or memory use [5]-[7…”
- `86b624f4` (88w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour is one reason to test quantized models directly. Training an assistant …”
- `d5f95374` (81w) Chapter 1: Introduction § 1.2 Problem Statement — “Refusal alone is not a sufficient measure of safety. A model that refuses every request m…”
- `5be3f0eb` (65w) Chapter 1: Introduction § 1.2 Problem Statement — “This project examines how quantization affects harmful compliance, refusal of benign prom…”
- `c979201b` (80w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 compares each model pair's change in MMLU accuracy with its change in classifi…”
- `372c216b` (59w) Chapter 1: Introduction § 1.4 Scope — “General capability is measured with a six-subject MMLU subset and ARC-Challenge. These tw…”
- `b042804c` (87w) Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation — “The three broad trustworthiness suites also illustrate a recurring trade-off in benchmark…”
- `e6fcef1d` (72w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “Mechanistic studies examine how refusal is represented inside the model. Safety Layers ar…”
- `ac866b09` (47w) Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation — “Agreement should be reported by model or pair when possible. A pooled agreement score can…”
- `3f95ffc3` (61w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature review identifies five gaps. First, the quantized and full-precision model…”
- `4bbff14f` (62w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The gap is not merely the absence, in Table 2.1, of the same benchmark combination. The m…”
- `54770b66` (81w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature also defines the scope of this study. Quantization-conditioned attacks sho…”
- `f0c9fa40` (76w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `742c828b` (55w) Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design — “The independent variable is therefore the precision configuration, and the dependent vari…”
- `80117cad` (49w) Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control — “Multiplicity correction is performed from the complete 20-row analysis record rather than…”
- `8cb037d5` (41w) Chapter 4: Results § 4.1 Results at a Glance — “The primary analysis contains 20 NF4 comparisons. Three results remain significant after …”
- `f7feffe6` (28w) Chapter 4: Results § 4.1 Results at a Glance — “After correction, the clearest effects concern capability. The only significant over-refu…”
- `bccd3060` (49w) Chapter 4: Results § 4.1 Results at a Glance — “Tables 4.1 and 4.2 report the safety and capability results separately. The baseline and …”
- `f69ef27f` (36w) Chapter 4: Results § 4.1 Results at a Glance — “Positive changes in ASR or over-refusal are undesirable, while negative capability change…”
- `76f7b133` (39w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “The direction of the change differs across model families, but none of the harmful-compli…”
- `94189377` (46w) Chapter 4: Results § 4.4 Capability (RQ3) — “The largest corrected effects concern capability. Qwen3-1.7B MMLU accuracy decreases from…”
- `e8499204` (46w) Chapter 4: Results § 4.4 Capability (RQ3) — “The two capability benchmarks prevent overgeneralisation from a single task. Qwen3-1.7B h…”
- `ba7bd1da` (35w) Chapter 4: Results § 4.4 Capability (RQ3) — “Figure 1.1 places all five pairs in the joint space these results define. The MMLU point …”
- `237c1278` (88w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels are descriptive summaries of the combined changes. Qwen3-1.7B i…”
- `f0396862` (56w) Chapter 4: Results § 4.6 INT8 Precision Point — “The 512-token INT8 rerun also clarifies an earlier result. At 128 tokens, Llama INT8 appe…”
- `9991e044` (83w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ5 tests whether the conclusion changes under different scorers, generation limits, and …”
- `fca8d807` (66w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The sample contains a larger share of scorer disagreements than the full HarmBench datase…”
- `8d494e1d` (51w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The independent three-class refusal judge does not reproduce the Phi-4-mini decrease foun…”
- `5dd7df66` (76w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The human-labelled sample helps explain this disagreement. Under the strict definition, k…”
- `c5f7c1a8` (97w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The XSTest human audit departed from the preregistered procedure in one respect. The blin…”
- `b830de28` (47w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “The broad definition also counts partial refusals. The judge's kappa increases under this…”
- `f6b43071` (46w) Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation — “A separate sensitivity analysis replaces the regex over-refusal score with the strict sem…”
- `c0bfbb18` (57w) Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness — “The longer generation limit also changes the absolute baseline rates because models have …”
- `16044c2d` (52w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “The average Mistral and Phi changes remain close to zero. Their greedy estimates lie just…”
- `fac28864` (34w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Changing the decoding seed does not create an independent model replicate. The model weig…”
- `1cb527ee` (42w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “Figure 5.2 compares the Qwen3-1.7B greedy estimate with the five stochastic-decoding seed…”
- `fecf48d0` (55w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “A non-significant result does not prove that the true effect is exactly zero. With 200 pa…”
- `da5a6bc3` (63w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Several checks support the same qualified conclusion. No ASR comparison remains significa…”
- `48525d6e` (55w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The capability tests have different sensitivity because MMLU and ARC use different sample…”
- `8c08920e` (29w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The evidence shows no statistically significant increase in harmful compliance after corr…”
- `7e72717f` (48w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “The validation analyses address several alternative explanations. Human labels show that …”
- `464e5783` (52w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Disagreement between scorers is examined rather than averaged away. When two scorers disa…”
- `db745a58` (59w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results show why harmful compliance, general capability, and scorer behaviour must be…”
- `7ac59863` (59w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “This conclusion applies only to the evaluated models, quantization settings, direct-reque…”
- `5e6ac865` (60w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results do not support a general claim that quantization improves safety. Changes dif…”
- `bcc6832c` (51w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “Quantization changes some prompt-level classifications in both directions even when the a…”
- `3b7bb1b6` (82w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Table 6.2 compares published results with those of this study rather than leaving the com…”
- `4c8ae25b` (71w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Taken together, the prior studies support a conclusion limited to the evaluated models an…”
- `9b6374c4` (40w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Scorer validation changes the interpretation of the experiment. The HarmBench refusal pro…”
- `901e2fa4` (55w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The original preregistered XSTest analysis and the later validation analysis are both rep…”
- `070cc14b` (37w) Chapter 6: Discussion § 6.5 Threats to Validity — “Human validation uses one annotator on each 200-item sample. The comparisons are therefor…”
- `0aed1309` (36w) Chapter 6: Discussion § 6.6 Practical Implications — “The results do not justify describing NF4 as either unsafe or risk-free. A quantized mode…”
- `ba5014ad` (48w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This project evaluated five matched pairs of smaller language models on HarmBench, XSTest…”
- `9989726a` (102w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “For RQ1, no NF4 change in harmful compliance remains significant after correcting the 20 …”
- `83f4ca5d` (41w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “A refusal count alone is not enough to interpret the safety effect of quantization. Harmf…”
- `3db5f2cc` (41w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The main contribution is the evidence produced by the controlled experiment and the valid…”
- `3d03ff99` (56w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “Under the tested conditions, INT8 preserves capability more consistently than NF4, while …”
- `a9c3be54` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Future experiments should preserve the original run outputs, store revised scorer labels …”
- `1357e5a8` (50w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The first two extensions are the most urgent. An adversarial evaluation would extend the …”

</details>

### Edited paragraphs (188; change window shown)

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
- Chapter 1: Introduction § 1.4 Scope: “…ma-3.2-3B, Mistral-7B, and Phi-4-mini. A separate INT8 [+comp][−m][+ison] uses the same …”
- Chapter 1: Introduction § 1.5 Contributions: “A reproducible benchmark implementation [−wi][+at][+p][−dacted d][+s][−i][−d sidecar][−an…”
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
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…lysis, and agent-status paths. The dashboard reads the [+stored ][− tree][−new user-gene…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “… but they determine whether a rerun loads the intended [−art][+fil][−fact]s instead of s…”
- Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls: “…ever, the exact historical environment of the earliest [−a][−t][+un fil][−fact][−c][+st]…”
- …and 68 more (see .full.json)

</details>

### Renamed headings (3)
- “2.5 Automatic Evaluators and Human Grounding” → “2.5 Automatic Evaluators and Human Validation”
- “5.3 Human Grounding of Harmful-Compliance Scorers” → “5.3 Human Validation of Harmful-Compliance Scorers”
- “6.3 Capability as the Deployment Cost” → “6.3 Capability Effects and Deployment”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
