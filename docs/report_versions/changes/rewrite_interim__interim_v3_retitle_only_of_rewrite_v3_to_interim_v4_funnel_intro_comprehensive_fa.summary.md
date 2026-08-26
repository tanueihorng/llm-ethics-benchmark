## Rewritten interim report (submission line): Interim v3 (retitle-only of rewrite v3) → Interim v4 (funnel intro, comprehensive fallback)

*From:* `docs/rewritten_fyp_interim_v3.docx`  
*To:* `docs/rewritten_fyp_interim_v4.docx`  
*Decisions:* D65, D68, D70  

**Paragraph/body blocks:** +10 added · −14 removed · ~48 edited · ⟳87 citation-renumber-only · tables: +0 −0 ~3 · front matter blocks 18→18

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~11 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ I declare that this report is my own work and that all sources used have been acknowledged. The res…
  - ❌ Post-training quantization lowers the precision of a model's weights to reduce its memory footprint…
  - ❌ Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance 3
  - ❌ Table 4.2. NF4 capability results 23
  - ❌ Term \| Meaning \|\| ARC \| AI2 Reasoning Challenge \|\| ASR \| Attack Success Rate \|\| BH-FDR \| …

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Reproducibility Summary | 0 | 0 | 1 | 0 |
| Chapter 1: Introduction | 5 | 6 | 1 | 0 |
| Chapter 2: Literature Review | 1 | 1 | 8 | 24 |
| Chapter 3: Methodology and Experimental Setup | 1 | 1 | 2 | 6 |
| Chapter 4: Results | 0 | 0 | 6 | 0 |
| Chapter 5: Scorer Validation and Robustness | 0 | 0 | 8 | 0 |
| Chapter 6: Discussion | 2 | 2 | 6 | 7 |
| Chapter 7: Conclusion and Future Work | 0 | 0 | 4 | 0 |
| References | 1 | 4 | 1 | 50 |

### Removed paragraphs (14 body; 0 front matter)

<details><summary>show</summary>

- `378b0d76` (89w) Chapter 1: Introduction § 1.1 Background and Motivation — “Compact instruction-tuned language models such as Qwen3 [1], the Llama 3 family [2], Phi-…”
- `a5c48389` (85w) Chapter 1: Introduction § 1.1 Background and Motivation — “Post-training quantization is one way to address this constraint: it represents model wei…”
- `f45df69c` (105w) Chapter 1: Introduction § 1.1 Background and Motivation — “Most literature on quantization focuses primarily on perplexity, task accuracy, latency, …”
- `5e1667f2` (86w) Chapter 1: Introduction § 1.1 Background and Motivation — “Refusal behaviour provides a concrete reason to evaluate quantized models directly. Instr…”
- `5106f738` (92w) Chapter 1: Introduction § 1.2 Problem Statement — “Evaluating model safety is challenging because refusal behaviour is not synonymous with t…”
- `c196ec07` (84w) Chapter 1: Introduction § 1.2 Problem Statement — “The objective of this study is to determine how post-training quantization changes harmfu…”
- `46339679` (77w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
- `a384b022` (60w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Cohen's kappa measures chance-corrected agreement between scorers [53]. Sample-size and m…”
- `059f115f` (41w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results reject both common oversimplifications: that quantization inherently improves…”
- `3fbc552d` (38w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesizing the published literature reinforces the contextual nature of safety findings…”
- `e8456ee1` (17w) References — “[4] Gemma Team, Google DeepMind, “Gemma 2: Improving open language models at a practical …”
- `178fcfb6` (21w) References — “[14] Y. Bai et al., “Training a helpful and harmless assistant with reinforcement learnin…”
- `8c43db77` (39w) References — “[17] X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, and P. Henderson, “Fine-tunin…”
- `d08a8717` (12w) References — “[51] A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825.”

</details>

### Added paragraphs (10)

<details><summary>show</summary>

- `c361956e` (148w) Chapter 1: Introduction § 1.1 Background and Motivation — “This project studies five matched model pairs drawn from four compact open-weight model f…”
- `4ab1cd85` (189w) Chapter 1: Introduction § 1.1 Background and Motivation — “Quantization evaluations commonly report perplexity or benchmark accuracy [8], [9], but t…”
- `5fa33275` (149w) Chapter 1: Introduction § 1.1 Background and Motivation — “To address this gap, this study evaluates matched fp16–NF4 versions of five revision-pinn…”
- `076586e7` (69w) Chapter 1: Introduction § 1.2 Problem Statement — “The problem addressed by this study is interpretive rather than merely operational: a cha…”
- `bc8f4d5e` (45w) Chapter 1: Introduction § 1.2 Problem Statement — “This study therefore measures all four outcomes on the same prompts under a single contro…”
- `ce7efbde` (77w) Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour — “HarmLevelBench reports non-uniform safety changes across quantization methods and attack …”
- `f91f5485` (60w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Cohen's kappa measures chance-corrected agreement between scorers [50]. Sample-size and m…”
- `35f26245` (50w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “The results support neither common oversimplification: neither that quantization inherent…”
- `3db5c862` (54w) Chapter 6: Discussion § 6.2 Relationship to Prior Work — “Synthesising the published literature reinforces the contextual nature of safety findings…”
- `a0f6e7f0` (12w) References — “[4] A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825.”

</details>

### Edited paragraphs (48; change window shown)

<details><summary>show</summary>

- (front matter): “… results were computed from the recorded experiment art[−e][+i]facts; no result was fabr…”
- (front matter): “Post-training quantization [+lowers the preci][+on of a model's wei][−nifican][+h][−ly][+…”
- (front matter): “1.2 Problem Statement [−1][+2]”
- (front matter): “1.3 Research Questions [−2][+3]”
- (front matter): “1.5 Contributions [−3][+4]”
- (front matter): “1.6 Report Structure [−3][+4]”
- (front matter): “4.6 INT8 Precision Point 2[−6][+7]”
- (front matter): “6.5 Threats to Validity 4[−1][+0]”
- (front matter): “…nges in MMLU accuracy and HarmBench harmful compliance [−2][+3]”
- (front matter): “Table 4.2. NF4 capability results 2[−2][+3]”
- (front matter): “…ation using LLM.int8 \|\| LLM \| Large language model \|\| M[+DE \\| Minimum detectable …”
- Chapter 1: Introduction § 1.4 Scope: “… evaluate latency, energy use, privacy, monetary cost, [−or ][+, or multimodal behaviour…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…require representative samples and an optimisation pass[+ [5], [6], [22]][+ [24], [25]].…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “… on aggregate tasks, especially as model size grows [7][−-][+–][−13][+28]]. The same con…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…ng and equivalent transformations during calibration [3[−8][+2][−'][+“][−'][+”][−9][+3]]…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…viour can be concentrated in a subset of model layers [[+3][−0][+[35] ][+ [36]], this pr…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…wer different questions. StrongREJECT was motivated by [−'][+“][−'][+”][−41][+38]]. Its …”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…ving finer-grained harmfulness scoring as future work [[−42][+39][−'][+ use][−5][+3][+,]…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…monstrate that targeted worst-case failures can occur [[−12][+33][−9][+7][−3][+0][−gr][+…”
- Chapter 2: Literature Review § 2.7 Implications for Experimental Design: “…many opportunities for an isolated p-value below 0.05. [−T][+K][+arina][− three quantiza…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…te procedure controls multiplicity across that family [[−52][+49][+'s] main conclusion.”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…ror. Exactly three primary contrasts survived Benjamini[−-][+–]Hochberg false-discovery-…”
- Chapter 4: Results § 4.1 Results at a Glance: “…4.1 and 4.2 present the safety and capability results. [−Baselin][+Th][−and ][+fp16→][−m…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…inct capability benchmarks prevents misleading generali[−z][+s]ations from a single task…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…th precision. Figure 4.2 shows that harmful compliance [−i][+doe][− generally][−n-m][+t …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…le Mistral reaches the same 0.565 ASR at INT8 and NF4. [−T][+Qwen3-4B is t][+ only pair …”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…harmful compliance under a 128-token limit, the effect [−completely ][−solv][+app][+ar][…”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “… detects significant NF4 harmful compliance increases. [−H][+Against a single annotator …”
- Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges: “…t agreement occurs on Phi, where Llama Guard's broader [−'][+“][−'][+”] construct differ…”
- Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers: “…purposefully oversamples scorer disagreements to maximi[−z][+s][−li][−n][+ree][+the ][−g…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “…igible. Neither result reaches statistical significance[−,][+.][−c][+Read t][−nf][+gethe…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “…sponse pairs. While this pre-annotation decision minimi[−z][+s][−su][+kappa gap ][−s][+e…”
- Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness: “…e rates by providing models sufficient context to fulfi[−l]l, qualify, or properly r…”
- Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null: “…04 and 0.09 depending on sample discordance. The study [−ha][+i][−ampl][+therefor][+ed][…”
- Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null: “The reported null finding is su[−bstantia][+ppor][−c][+long fou][−oss][−mu][+part][−tiple…”
- Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis: “[−T][+Eac][−e][+ addresse][−systematically elimi][+o][−at][−hy][+ex][+lanati][−theses][+n…”
- Chapter 6: Discussion § 6.1 Answer to the Central Question: “A more accurate characteri[−z][+s]ation of low-bit quantiza…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…le to direct comparison \|\| Decoding Compressed Trust [3[−7][+1][+; IEEE version of rec…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…ble deserve emphasis because they are easy to misread. [−K][+In t][+e IEEE version of re…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…wed here report good average retention at four bits [7][−-][+–][9], but the Qwen3-1.7B M…”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “…. In both cases, model-based semantic evaluators agree [−far][+much][+the ][−gr][+ann][−…”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “…e and withdrawing the claim of full preregistration is [−s][+be][−rong][+t][+r][−vid][+s…”
- Chapter 7: Conclusion and Future Work § 7.1 Conclusions: “…aries substantially by model architecture and precision[−, with][+;][−pre][+stays clo][−…”
- Chapter 7: Conclusion and Future Work § 7.1 Conclusions: “The primary contribution of this work [−re][+i][−ides in][− rigorous][−me][+e acc][−d][+m…”
- Chapter 7: Conclusion and Future Work § 7.1 Conclusions: “From an engineering standpoint, INT8 [+was the m][−ff][+re cons][−s ][+v][− h][+t][−ghly …”
- Chapter 7: Conclusion and Future Work § 7.2 Prioritised Future Work: “… raw outputs, maintaining version-pinned checkpoints, u[−tiliz][+s]ing validated evaluat…”
- References: “[[−62][+59][+in Proc. Conference on Language Modeling (COLM), 2026. ][−, 2025 (preprint)].”
- Appendix A: Reproducibility Summary: “…on schema, generated SLURM scripts, analysis programs, [−r][+th][−sult] derived scoring …”

</details>

### Citation-renumber-only paragraphs: 87 (text identical once [n] markers stripped; details in .full.json)

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
