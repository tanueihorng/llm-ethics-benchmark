## Rewritten full report (plain-academic line): Rewrite v2 (plain-academic pass) → Rewrite v3 (selected plain-academic voice)

*From:* `docs/rewritten_fyp_report_v2.docx`  
*To:* `docs/rewritten_fyp_report_v3.docx`  
*Decisions:* D63, D64, D65  

**Paragraph/body blocks:** +46 added · −46 removed · ~9 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~0 · front matter blocks 18→18

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~1 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Post-training quantization significantly reduces the computational and memory requirements of instr…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Chapter 1: Introduction | 7 | 7 | 7 | 0 |
| Chapter 2: Literature Review | 1 | 1 | 0 | 0 |
| Chapter 4: Results | 7 | 7 | 1 | 0 |
| Chapter 5: Scorer Validation and Robustness | 16 | 16 | 0 | 0 |
| Chapter 6: Discussion | 8 | 8 | 0 | 0 |
| Chapter 7: Conclusion and Future Work | 7 | 7 | 0 | 0 |

### Removed paragraphs (46 body; 0 front matter)

<details><summary>show</summary>

- `8f15566b` (97w) Chapter 1: Introduction § 1.1 Background and Motivation — “Smaller instruction-tuned language models can now run on consumer GPUs and some edge devi…”
- `62f4bdab` (81w) Chapter 1: Introduction § 1.1 Background and Motivation — “Most quantization studies report perplexity, task accuracy, latency, or memory use [5]-[7…”
- `86b624f4` (88w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour is one reason to test quantized models directly. Training an assistant …”
- `d5f95374` (81w) Chapter 1: Introduction § 1.2 Problem Statement — “Refusal alone is not a sufficient measure of safety. A model that refuses every request m…”
- `5be3f0eb` (65w) Chapter 1: Introduction § 1.2 Problem Statement — “This project examines how quantization affects harmful compliance, refusal of benign prom…”
- `c979201b` (80w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 compares each model pair's change in MMLU accuracy with its change in classifi…”
- `372c216b` (59w) Chapter 1: Introduction § 1.4 Scope — “General capability is measured with a six-subject MMLU subset and ARC-Challenge. These tw…”
- `54770b66` (81w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature also defines the scope of this study. Quantization-conditioned attacks sho…”
- `8cb037d5` (41w) Chapter 4: Results § 4.1 Results at a Glance — “The primary analysis contains 20 NF4 comparisons. Three results remain significant after …”
- `f7feffe6` (28w) Chapter 4: Results § 4.1 Results at a Glance — “After correction, the clearest effects concern capability. The only significant over-refu…”
- `bccd3060` (49w) Chapter 4: Results § 4.1 Results at a Glance — “Tables 4.1 and 4.2 report the safety and capability results separately. The baseline and …”
- `f69ef27f` (36w) Chapter 4: Results § 4.1 Results at a Glance — “Positive changes in ASR or over-refusal are undesirable, while negative capability change…”
- `94189377` (46w) Chapter 4: Results § 4.4 Capability (RQ3) — “The largest corrected effects concern capability. Qwen3-1.7B MMLU accuracy decreases from…”
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
- `4c8ae25b` (71w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Taken together, the prior studies support a conclusion limited to the evaluated models an…”
- `9b6374c4` (40w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Scorer validation changes the interpretation of the experiment. The HarmBench refusal pro…”
- `901e2fa4` (55w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “The original preregistered XSTest analysis and the later validation analysis are both rep…”
- `0aed1309` (36w) Chapter 6: Discussion § 6.6 Practical Implications — “The results do not justify describing NF4 as either unsafe or risk-free. A quantized mode…”
- `ba5014ad` (48w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This project evaluated five matched pairs of smaller language models on HarmBench, XSTest…”
- `9989726a` (102w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “For RQ1, no NF4 change in harmful compliance remains significant after correcting the 20 …”
- `83f4ca5d` (41w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “A refusal count alone is not enough to interpret the safety effect of quantization. Harmf…”
- `3db5f2cc` (41w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The main contribution is the evidence produced by the controlled experiment and the valid…”
- `3d03ff99` (56w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “Under the tested conditions, INT8 preserves capability more consistently than NF4, while …”
- `a9c3be54` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Future experiments should preserve the original run outputs, store revised scorer labels …”
- `1357e5a8` (50w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The first two extensions are the most urgent. An adversarial evaluation would extend the …”

</details>

### Added paragraphs (46)

<details><summary>show</summary>

- `378b0d76` (89w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact instruction-tuned language models such as Qwen3 [1], the Llama 3 family [2], Phi-…”
- `f45df69c` (105w) Chapter 1: Introduction § 1.1 Background and Motivation — “Most literature on quantization focuses primarily on perplexity, task accuracy, latency, …”
- `5e1667f2` (86w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour provides a concrete reason to evaluate quantized models directly. Instr…”
- `5106f738` (92w) Chapter 1: Introduction § 1.2 Problem Statement — “Evaluating model safety is challenging because refusal behaviour is not synonymous with t…”
- `c196ec07` (84w) Chapter 1: Introduction § 1.2 Problem Statement — “The objective of this study is to determine how post-training quantization changes harmfu…”
- `48c2ccdf` (95w) Chapter 1: Introduction § 1.2 Problem Statement — “Figure 1.1 places each model pair's change in MMLU accuracy alongside its change in harmf…”
- `96dcab85` (59w) Chapter 1: Introduction § 1.4 Scope — “General capability is evaluated using a representative six-subject MMLU subset and ARC-Ch…”
- `4e51ca35` (84w) Chapter 2: Literature Review § 2.6 Synthesis and Research Gap — “The literature also defines the empirical boundaries of this research. Prior studies on a…”
- `34d0b49f` (56w) Chapter 4: Results § 4.1 Results at a Glance — “The primary statistical family encompasses 20 NF4 comparisons across the five model pairs…”
- `f05446c2` (44w) Chapter 4: Results § 4.1 Results at a Glance — “Applying multiple-comparison correction clarifies that the primary statistically defensib…”
- `71e63bbd` (45w) Chapter 4: Results § 4.1 Results at a Glance — “Tables 4.1 and 4.2 present the safety and capability results. Baseline and NF4 columns di…”
- `8b79307d` (41w) Chapter 4: Results § 4.1 Results at a Glance — “Throughout this report, positive deltas in ASR or over-refusal indicate undesirable incre…”
- `d62e1908` (55w) Chapter 4: Results § 4.4 Capability (RQ3) — “The most pronounced and statistically robust effects of NF4 quantization appear on capabi…”
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
- `3fbc552d` (38w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesizing the published literature reinforces the contextual nature of safety findings…”
- `2960a8c4` (42w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Validating scoring rubrics is central to rigorous empirical evaluation. The HarmBench key…”
- `f2b8a59d` (45w) Chapter 6: Discussion § 6.4 Validity of the Scoring Methods — “Both the preregistered regex analysis and the validated semantic judge evaluation are rep…”
- `31a7a9d7` (33w) Chapter 6: Discussion § 6.6 Practical Implications — “The practical takeaway for system designers is procedural: quantized language models shou…”
- `65a5546d` (36w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “This research conducted a controlled empirical evaluation of five matched compact languag…”
- `c3e72e87` (87w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “In summary, RQ1 is answered by a robust null: no NF4 change in harmful compliance remains…”
- `8c4bac67` (32w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The central methodological insight is that safety evaluations must not rely solely on key…”
- `4c481d63` (28w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “The primary contribution of this work resides in the rigorous empirical evidence and meth…”
- `ab396b5b` (33w) Chapter 7: Conclusion and Future Work § 7.1 Conclusions — “From an engineering standpoint, INT8 offers a highly reliable precision tier for preservi…”
- `3b336751` (24w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “Future investigations should maintain rigorous evaluation standards by preserving immutab…”
- `b1320ed6` (28w) Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work — “The most critical future extensions include extending safety evaluations to adversarial j…”

</details>

### Edited paragraphs (9; change window shown)

<details><summary>show</summary>

- (front matter): “Post-training quantization [+significantly ][+computational and memory ][−so][+q][+i][−c]…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “Post-training quantization [+is one way to add][−duce][+s][−mem][+this c][+nst][−y][+aint…”
- Chapter 1: Introduction § 1.4 Scope: “…hout an optimised jailbreak suffix or attack loop. The [−r][+study do][−ults therefore c…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A c][+C][+.][+The s][−hat][+udy][+model ][+, keeping the checkpoint constant while chan…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A j][+J][+ behavioural][+.][−of][+The study evaluates][+ accuracy][+curacy ac][+ matche…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A p][+P][+.][−w][+The study comb][−th][+nes] bootstrap confidence int…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A s][+S][+. HarmBench results are cross-checked] using the HarmBench clas…”
- Chapter 1: Introduction § 1.5 Contributions: “[−A r][+R][+ing pipeline. The][−that ][+immu][−h][+abl][+raw m][−rigina][+de][− run][+de]…”
- Chapter 4: Results § 4.4 Capability (RQ3): “[−The][+Employing][+ distinct][+s][−ov][+misl][−r][+ading ][−s][+z][+s][−ha][+experience]…”

</details>

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
