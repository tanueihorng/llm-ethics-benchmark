## Humanized alternate — interim: Base — interim master → Humanized interim alternate

*From:* `docs/FYP_Interim_2026-07-10.docx`  
*To:* `docs/FYP_Interim_2026-07-10_humanized.docx`  

**Paragraph/body blocks:** +3 added · −2 removed · ~72 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~1 · front matter blocks 43→43

*Front matter (cover/abstract/TOC/lists):* +1 −1 ~7 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ A Matched-Pair, Judge-Validated Study of Safety, Capability Trade-offs in Quantized Compact Languag…
  - ❌ All experimental results reported in this interim report were produced by the open-source benchmark…
  - ❌ Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasin…
  - ❌ It contributes an open, reproducible benchmarking framework built around a matched-pair design (bas…
  - ❌ Three findings result. First, methodologically, the scorer and the generation budget each determine…
  - ❌ Status of this interim report. This report presents the project's completed methodology, system, ex…
  - ❌ Table 6.1 Judge-versus-regex agreement (Cohen's κ) by model family.
  - ❌ Figure 4 Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.
  - ➕ All experimental results reported in this interim report were produced by the open-source benchmark…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A Reproducibility Statement | 0 | 0 | 1 | 0 |
| Chapter 1 Introduction | 0 | 0 | 3 | 0 |
| Chapter 10 Conclusion | 0 | 0 | 2 | 0 |
| Chapter 2 Background and Related Work | 0 | 0 | 5 | 0 |
| Chapter 3 Methodology | 0 | 0 | 5 | 0 |
| Chapter 4 System Design and Implementation | 0 | 0 | 2 | 0 |
| Chapter 5 Experimental Setup | 0 | 0 | 2 | 0 |
| Chapter 6 Results and Analysis | 1 | 0 | 12 | 0 |
| Chapter 7 Discussion and Threats to Validity | 1 | 1 | 2 | 0 |
| Chapter 8 Limitations | 0 | 0 | 3 | 0 |
| Chapter 9 Progress and Remaining Work | 0 | 0 | 3 | 0 |
| References | 0 | 0 | 25 | 0 |

### Removed paragraphs (1 body; 1 front matter)

<details><summary>show</summary>

- `a8695ce9` (517w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “The primary safety construct is scored by the benchmark’s own fine-tuned classifier rathe…”

</details>

### Added paragraphs (2)

<details><summary>show</summary>

- `af0545ff` (250w) Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives — “A refusal-judge sensitivity check bears directly on that one survivor. An independent thr…”
- `72cd149d` (179w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “The primary safety construct is scored by the benchmark's own fine-tuned classifier rathe…”

</details>

### Edited paragraphs (72; change window shown)

<details><summary>show</summary>

- (front matter): “A Matched-Pair, Judge-Validated Study of Safety[−–][+, ]Capability Trade-offs in …”
- (front matter): “…them: a brittle refusal-matching scorer can over-count [−“][+"][−”][+"] and the number o…”
- (front matter): “…rols apply throughout: the safety prompts are HarmBench[−’][+'][−’][+']s own standardize…”
- (front matter): “…harmful set is a near-strict superset of the classifier[−’][+'][−’][+'][−,][+; this][−in…”
- (front matter): “…f this interim report. This report presents the project[−’][+']s completed methodology, …”
- (front matter): “Table 6.1 Judge-versus-regex agreement (Cohen[−’][+']s κ) by model family.”
- (front matter): “…ion: judge ASR vs regex proxy, and judge-vs-proxy Cohen[−’][+']s κ.”
- Chapter 1 Introduction § 1.2 Problem statement: “…ally scored by pattern-matching for refusals, equating [−“][+"][−”][+"][−“][+"][−”][+"] …”
- Chapter 1 Introduction § 1.4 Contributions: “…t is a near-strict superset of the benchmark classifier[−’][+'][−’][+'][−’][+'][−’][+'][…”
- Chapter 1 Introduction § 1.5 Report structure: “…, including the two corrections made during the project[− —][+,][− —][+,] as worked exam…”
- Chapter 2 Background and Related Work § 2.1 Quantization and its behavioural effects: “…ovides the two schemes evaluated here: NF4, a four-bit [−“][+"][−”][+"][−’][+'][−h][+own…”
- Chapter 2 Background and Related Work § 2.2 Safety benchmarks and red-teaming: “…c and transparent. This interim report adopts HarmBench[−’][+']s classifier-as-scorer pr…”
- Chapter 2 Background and Related Work § 2.3 LLM-as-judge evaluation and its validity: “Using one model to judge another[−’][+'][−’][+'][− —][+,] a single-annotator gold …”
- Chapter 2 Background and Related Work § 2.4 The research gap: “… this interim report addresses. First, the quantization[−–][+, ][−–][+, ][−“][+"][−”][+"…”
- Chapter 2 Background and Related Work § 2.4 The research gap: “No prior study, to the author[−’][+']s knowledge, combines all…”
- Chapter 3 Methodology § 3.1 Matched-pair design: “…ur families and roughly 1.7 to 7.2 billion parameters, [−pro][+which gi][−iding][+es] bo…”
- Chapter 3 Methodology § 3.3 Benchmarks and metrics: “…mpliance under direct requests, equivalent to HarmBench[−’][+']s DirectRequest baseline …”
- Chapter 3 Methodology § 3.4 Scoring: classifier-primary, regex-as-foil: “…dependent judge (GPT-4o; run 2026-06-14 — the API alias[−’][+'][+measu][−port][−u][+i][+…”
- Chapter 3 Methodology § 3.5 Capability-anchored interpretation: “…ent shift; if capability collapses, an apparent safety [−“][+"][−”][+"] may merely be th…”
- Chapter 3 Methodology § 3.6 Decoding controls and statistics: “…intervals is prompt sampling. Generation uses HarmBench[−’][+'][−’][+'][−’][+'][−–][+ to…”
- Chapter 4 System Design and Implementation § 4.1 Architecture: “…r batches prompts through the model using the tokenizer[−’][+']s chat template; benchmar…”
- Chapter 4 System Design and Implementation § 4.2 Extensibility and the plugin contract: “…e line; adding a quantization method extends the loader[−’][+'][−’][+']s quickstart and …”
- Chapter 5 Experimental Setup: “…eration budgets: the 512-token primary study (HarmBench[−’][+']s standardized evaluation…”
- Chapter 5 Experimental Setup: “…-4o second judge runs locally against an API. The study[−’][+']s entire analysis is repr…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…ench classifier, and the gap is uneven by family. Cohen[−’][+'][−’][+'][−’][+'][−’][+'][…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “Family \| Cohen[−’][+'][−–][+to][−–][+to][−–][+to][−–][+to] 0.77 \| substantial; like…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “Table 6.1 Judge-versus-regex agreement (Cohen[−’][+']s κ) by model family (fp1…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…i 4-bit) sit just above it. Right: judge-vs-proxy Cohen[−’][+']s κ per model, family-dep…”
- Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget: “…whose confidence interval excludes zero is Llama-3.2-3B[−’][+']s, and it is a decrease (…”
- Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget: “…a significant under the exact paired test is Phi-4-mini[−’][+'][−;][+), and Qwen3-1.7B s…”
- Chapter 6 Results and Analysis § 6.3 The generation-budget artefact (why 512 tokens is the primary budget): “The 128-token budget of the study[−’][+'][−–][+ to ][−–][+ to ][−’][+'][− —][+,] small, z…”
- Chapter 6 Results and Analysis § 6.4 Capability anchoring (RQ3): “Every pair[−’][+'][−’][+'][−’][+'][−’][+'][+. Part of that MMLU gap is a scoring artefact…”
- Chapter 6 Results and Analysis § 6.5 Mechanism: refusal-margin geometry: “…28-token signal, a derived analysis measured each model[−’][+'][−’][+'][+which ][−ing][+…”
- Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5): “…em McNemar contrasts, none significant — far below the [−mult][+s][−pl][+gnif][−ity-robu…”
- Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives: “…or a representative pair (the per-pair values span 0.04[−–][+ to ][−“][+"][−”][+"]), not…”
- Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives: “… Phi-4-mini over-refusal (−0.048, q = 0.012, a decrease[− — regex-scored and scorer-depe…”
- Chapter 7 Discussion and Threats to Validity § 7.1 Internal validity: “Internal validity is the design[−’][+'][−’][+']s own standardized budget…”
- Chapter 7 Discussion and Threats to Validity § 7.3 External and statistical-conclusion validity: “…amily, a minimum-detectable-effect analysis (about 0.04[−–][+ to ]0.09 per pair), and a …”
- Chapter 8 Limitations: “…80% power is about 0.06 for a representative pair (0.04[−–][+ to ][−’][+']s multi-seed s…”
- Chapter 8 Limitations: “…appear), which is controlled here by adopting HarmBench[−’][+'][−–][+ to ]4 percent).”
- Chapter 8 Limitations: “…opping: the generation loop halts only on the tokenizer[−’][+'][−’][+']s completed answe…”
- Chapter 9 Progress and Remaining Work § 9.1 Progress to date: “…ch, XSTest, MMLU, and ARC-Challenge), at both HarmBench[−’][+'][−— ][+(][−’][+'][−–][+ t…”
- Chapter 9 Progress and Remaining Work § 9.1 Progress to date: “…rmful compliance and was demoted in favour of HarmBench[−’][+'][+which ][−ing][+ed][−’][…”
- Chapter 9 Progress and Remaining Work § 9.2 Remaining work toward the final thesis: “The study[−’][+']s scientific claims are e…”
- Chapter 10 Conclusion: “…lity anchor, a multi-precision sweep, and the benchmark[−’][+']s own standardized genera…”
- Chapter 10 Conclusion: “…harmful compliance and, uncorrected, can change a study[−’][+'][−’][+'][−’][+'][−’][+'][…”
- References: “…T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, [−“][+"][−”][+"] in Proc. Advance…”
- References: “…Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, [−“][+"][−”][+"] in Proc. NeurIPS…”
- References: “[3] M. Mazeika et al., [−“][+"][−”][+"] in Proc. International C…”
- References: “[4] A. Kharinaev et al., [−“][+"][−”][+"][−–][+ to ]96793, 2026, doi: 10.1109…”
- References: “[5] Y. Belkhiter et al., [−“][+"][−”][+"] arXiv preprint arXiv:241…”
- References: “[6] R. Jin et al., [−“][+"][−”][+"] in Findings of the Assoc…”
- References: “… K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, [−“][+"][−”][+"] in Proc. Advance…”
- References: “… Kailkhura, D. Hendrycks, D. Song, Z. Wang, and B. Li, [−“][+"][−”][+"] in Proc. Interna…”
- References: “…irk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, [−“][+"][−”][+"] in Proc. NAACL, …”
- References: “[10] P. Liang et al., [−“][+"][−”][+"] Transactions on Machine …”
- References: “[11] J. Gu et al., [−“][+"][−”][+"] arXiv preprint arXiv:241…”
- References: “[12] M. Krumdick et al., [−“][+"][−”][+"] arXiv preprint arXiv:250…”
- References: “…asart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, [−“][+"][−”][+"] in Proc. Interna…”
- References: “…, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, [−“][+"][−”][+"] arXiv preprint a…”
- References: “[16] Q. McNemar, [−“][+"][−”][+"][−–][+ to ]157, 1947.”
- References: “[17] Y. Benjamini and Y. Hochberg, [−“][+"][−”][+"][−–][+-]300, 1995.”
- References: “[18] R. J. Connor, [−“][+"][−”][+"][−–][+-]211, 1987.”
- References: “[19] J. Cohen, [−“][+"][−”][+"][−–][+-]46, 1960.”
- References: “[20] J. R. Landis and G. G. Koch, [−“][+"][−”][+"][−–][+-]174, 1977.”
- References: “…21] I. Proskurina, L. Brun, G. Metzler, and J. Velcin, [−“][+"][−”][+"] in Findings of N…”
- References: “[22] H. Inan et al., [−“][+"][−”][+"] Meta AI, 2023. arXiv:231…”
- References: “[23] J. Pineau et al., [−“][+"][−”][+"] Journal of Machine Learn…”
- References: “… G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, [−“][+"][−”][+"] PLOS Computation…”
- References: “…K. Cranston, J. Kitzes, L. Nederbragt, and T. K. Teal, [−“][+"][−”][+"] PLOS Computation…”
- References: “[26] A. M. Smith et al., [−“][+"][−”][+"] arXiv preprint arXiv:170…”
- Appendix A Reproducibility Statement: “…t [23]: models and algorithms are described (Chapters 3[−–][+ to ]4); datasets, splits, …”

</details>

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
