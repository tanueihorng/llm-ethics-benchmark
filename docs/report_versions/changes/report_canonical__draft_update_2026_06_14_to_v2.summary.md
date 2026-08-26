## Full FYP report (docx-js line): Draft update (2026-06-14) → v2

*From:* `docs/archive/FYP_Report_2026-06-14.docx`  
*To:* `docs/archive/FYP_Report_2026-06-26_v2.docx`  

**Paragraph/body blocks:** +36 added · −13 removed · ~30 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~0 · front matter blocks 29→35

*Front matter (cover/abstract/TOC/lists):* +10 −4 ~2 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ A Matched-Pair, Judge-Validated Study of Safety–Capability Trade-offs in Quantized Compact Language…
  - ❌ Document date: 26 June 2026
  - ❌ Revision: HarmBench ASR scored by the official HarmBench classifier (judge-primary, D16); v2 refusa…
  - ❌ Compact instruction-tuned language models in the one-to-four-billion-parameter range are increasing…
  - ❌ This work designs and implements a research-grade benchmarking framework to study safety–capability…
  - ❌ Each pair is scored on three complementary benchmarks: HarmBench (harmful compliance, measured as A…
  - ➕ Interim report, revised through June 2026. Scope: five matched pairs across four families at three …
  - ➕ Compact instruction-tuned language models are almost always deployed quantized, yet their safety is…
  - ➕ The design loads the baseline and quantized members of each pair from identical weights, applying q…
  - ➕ Under the classifier, four-bit NF4 never significantly reduces harmful compliance and significantly…
  - ➕ Figure 1 Quantization effect on harmful compliance (per-pair judge ΔASR forest plot).
  - ➕ Figure 2 The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).
  - ➕ Figure 3 Multi-seed decoding sensitivity for Qwen3-1.7B (greedy vs stochastic).
  - ➕ Figure 4 Per-category HarmBench ASR for Qwen3-1.7B, fp16 vs NF4.
  - ➕ Figure 5 Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.
  - ➕ Figure 6 Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).

### Sections added
- ➕ Chapter 6 — Results and Analysis § 6.5 Statistical Caveats — 6.5.1 Multiple-comparison correction and statistical power

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Chapter 1 — Introduction | 4 | 4 | 3 | 0 |
| Chapter 2 — Literature Review | 0 | 0 | 6 | 0 |
| Chapter 6 — Results and Analysis | 8 | 0 | 7 | 0 |
| Chapter 9 — Future Work | 0 | 0 | 1 | 0 |
| References | 14 | 5 | 11 | 0 |

### Removed paragraphs (9 body; 4 front matter)

<details><summary>show</summary>

- `858e73b0` (13w) Chapter 1 — Introduction § 1.5 Contributions — “The contributions of this project span experimental design, software engineering, and ana…”
- `841aa2da` (40w) Chapter 1 — Introduction § 1.5 Contributions — “An open, reproducible benchmarking framework consisting of approximately 2,800 lines of p…”
- `2df78202` (41w) Chapter 1 — Introduction § 1.5 Contributions — “SLURM cluster orchestration for the NTU TC1 GPU cluster, including resumable per-model ma…”
- `e1cec630` (53w) Chapter 1 — Introduction § 1.5 Contributions — “A verification suite of 306 automated tests across twenty-three test files, covering sche…”
- `ba487fe2` (19w) References — “The following references are cited or contextually relevant. Final publication-grade form…”
- `29cc726e` (19w) References — “Röttger, P., et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in…”
- `cba88aac` (12w) References — “Hendrycks, D., et al. Measuring Massive Multitask Language Understanding. ICLR 2021. (MML…”
- `cfdd1b24` (15w) References — “Qwen Team. Qwen Technical Report Series (Qwen 2.5 / Qwen 3.x). Alibaba DAMO Academy, 2024…”
- `dd77bb9b` (9w) References — “Meta AI. Llama 3.2 Model Family. Technical Report, 2024.”

</details>

### Added paragraphs (26)

<details><summary>show</summary>

- `dd11f230` (31w) Chapter 1 — Introduction § 1.5 Contributions — “The contributions of this project are, in order of significance, a methodological caution…”
- `a75b2c5b` (79w) Chapter 1 — Introduction § 1.5 Contributions — “A scorer-validity result with consequences beyond this study: the choice of HarmBench sco…”
- `22c564f1` (47w) Chapter 1 — Introduction § 1.5 Contributions — “An open, reproducible benchmarking framework (approximately 2,800 lines of production Pyt…”
- `a39fb8a8` (73w) Chapter 1 — Introduction § 1.5 Contributions — “An empirical characterisation of quantization across five model pairs, four families, and…”
- `79b66941` (34w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Figure 1. Quantization effect on harmful compliance: per-pair judge ΔASR (4-bit − fp16) w…”
- `0a0181bf` (64w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Figure 2. The capability-anchored safety space. Each pair is placed by its capability del…”
- `5bf1d098` (190w) Chapter 6 — Results and Analysis § 6.5 Statistical Caveats — “Rather than only acknowledging the multiplicity problem, we report the corrected view. Ap…”
- `c351f0f7` (113w) Chapter 6 — Results and Analysis § 6.5 Statistical Caveats — “A power analysis quantifies why so few effects reach significance. For a two-sided α = 0.…”
- `83c5cf83` (53w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Figure 3. Qwen3-1.7B judge ΔASR under decoding variation: the single greedy headline (+0.…”
- `a37a050c` (52w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Figure 4. Per-category HarmBench attack success rate for Qwen3-1.7B at fp16 versus NF4 (o…”
- `6b5c69d6` (61w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Figure 5. Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal…”
- `7be3b803` (54w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Figure 6. Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat t…”
- `983de43a` (16w) References — “References are cited in IEEE numbered style; the bracketed numbers below correspond to th…”
- `f974e14f` (32w) References — “P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test…”
- `ebf000ea` (30w) References — “D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measu…”
- `3c1de9a3` (29w) References — “P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Th…”
- `b722f726` (22w) References — “T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multipli…”
- `aaa9fe96` (13w) References — “A. Yang et al. (Qwen Team), “Qwen3 technical report,” Alibaba Group, 2025. arXiv:2505.093…”
- `b632bdf8` (16w) References — “A. Grattafiori et al. (Meta AI), “The Llama 3 herd of models,” Meta AI, 2024. arXiv:2407.…”
- `1f8334ac` (11w) References — “A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825.”
- `66676e38` (10w) References — “M. Abdin et al., “Phi-4 technical report,” Microsoft, 2024. arXiv:2412.08905.”
- `23002927` (10w) References — “J. Gu et al., “A survey on LLM-as-a-judge,” arXiv:2411.15594, 2024.”
- `50e24159` (15w) References — “M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human groundin…”
- `6028040a` (16w) References — “H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversations…”
- `ddd5aa42` (23w) References — “Q. McNemar, “Note on the sampling error of the difference between correlated proportions …”
- `04082c03` (17w) References — “B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall…”

</details>

### Edited paragraphs (30; change window shown)

<details><summary>show</summary>

- (front matter): “A[+ Matched-Pair, Judge-Validated][−4-bit ]Quantized Compact Languag…”
- (front matter): “Document date: 2[−7][+6][−May][+June] 2026”
- Chapter 1 — Introduction § 1.3 Research Questions: “…agnitude of the quantization effect differ between Qwen[− 2][+3-1.7][− ][+3-]4B within t…”
- Chapter 1 — Introduction § 1.5 Contributions: “…ng model_id; the only difference is whether on-the-fly [−NF4 ][− —][+,] two confounds th…”
- Chapter 1 — Introduction § 1.5 Contributions: “A [+capability-ancho][−ule-bas][−lay][+m][−r][+thod][−:][+(][−or ][+), each carried with …”
- Chapter 2 — Literature Review § 2.1 Large-Scale Ethical Benchmarking of LLMs: “…k the ethical performance of open-source LLMs. TrustLLM[+ [1]][+ [2]][+ [3]] each provid…”
- Chapter 2 — Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “…r measuring opposite sides of this trade-off. HarmBench[+ [4]][+ [5]] provides benign pr…”
- Chapter 2 — Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “MMLU[+ [6]][+; ARC-Challenge [7] provides a structurally different second capability axis…”
- Chapter 2 — Literature Review § 2.4 Quantization and Behavioural Effects: “…the NF4 quantization scheme introduced as part of QLoRA[+ [8]][+eight-bit LLM.int8() met…”
- Chapter 2 — Literature Review § 2.4 Quantization and Behavioural Effects: “…ift even when general benchmark scores appear preserved[+ [18]][−(arXiv:2502.][+[][−5799…”
- Chapter 2 — Literature Review § 2.5 Research Gaps Targeted by This Work: “…ct-deployment regime is addressed by the choice of Qwen[− 2][+3-1.7][− ][+3-][− ][+-][− …”
- Chapter 6 — Results and Analysis: “…bels). Detailed per-pair observations are in §6.6 (Qwen[− 2][+3-1.7][− ][+3-][− ][+-][− …”
- Chapter 6 — Results and Analysis § 6.1 Results Table: “…onfidence intervals are computed by a paired bootstrap [+[24] ]over the matched prompt s…”
- Chapter 6 — Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…test. Because HarmBench is paired, McNemar's exact test[+ [23]] (the correct paired-bina…”
- Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis: “…oves in the benign direction. Under the v2 scorer: Qwen[− 2][+3-1.7][− ][+3-]4B −0.008 (…”
- Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…ive ΔOR deltas are not statistically significant — Qwen[− 2][+3-1.7][− ][+3-][− ][+-]3B …”
- Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “… because it is the field-standard reference classifier [+speci][−or][+ied by the][+ pape…”
- Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…The hypothesis under test, following Proskurina et al. [−(2024)][+[18]], is that quantiz…”
- Chapter 9 — Future Work: “…ctivation-space refusal-direction probe (Arditi et al. [−2024][+[19]]) as a second mecha…”
- References: “[+L. ][−, L.,][+,][+“][−L][+l][−L][+l][−M][+m][+,” in Proc. International Conference on M…”
- References: “[+B. ][−, B.,][+,][+“][−C][+c][−A][+a][−T][+t][−M][+m][+,” in Proc][− 2023][+, 2023][+ ar…”
- References: “[+Z. ][−, Z.,][+,][+“][−S][+s][−L][+l][−L][+l][−M][+m][+,” in Proc. ACL, 2024][−, 2023].”
- References: “[+M. ][−, M.,][+,][+“][−S][+s][−E][+e][−F][+f][−A][+a][−R][+r][−T][+t][−R][+r][−R][+r][+,…”
- References: “[+T. ][−T][+A][−,][−,][−A][+and L][−,][−L. ][+“][−F][+f][−Q][+q][+,” in Proc][+,][+. arXi…”
- References: “[+A. ][−A][+V][−,][−V][+E][−,][−, E.,][+,][+“][−I][+i][−Q][+q][−M][+m][−S][+s][−R][+r][−L…”
- References: “[+K. ][−K][+J][−,][−,][−, J.,][+,][+“][−S][+s][−R][+r][−Q][+q][−S][+s][−P][+p][−Q][+q][−L…”
- References: “[+K. ][−K][+M][−,][−M][+R][−,][−R][+J][−,][−J][+and M][−,][−M. ][+“][−Q][+q][+,” in Proc]…”
- References: “[+Y. ][−Y][+G][−,][−G][+and S][−,][−S. ][+“][−H][+h][−L][+l][−C][+c][−I][+i][−Q][+q][−M][…”
- References: “[+A. ][−A][+O][−,][−O][+A][−,][−A][+D][−,][−D][+N][−,][−N][+W][−,][−W][+and N][−,][−N. ][…”
- References: “[+I. ][−I][+L][−,][−L][+G][−,][−G][+and J][−,][−J. ][+“][−Q][+q][−A][+a][−C][+c][−L][+l][…”

</details>

### Renamed headings (3)
- “6.6 Observations: Qwen 2B Pair” → “6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b)”
- “6.8 Observations: Qwen 4B Pair” → “6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b)”
- “6.10 Observations: Llama 3.2 3B Pair” → “6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b)”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
