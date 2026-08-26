## Standalone thesis: Thesis v3 → Thesis v4 (512-primary mirror)

*From:* `docs/archive/FYP_Thesis_2026-06-26_v3.docx`  
*To:* `docs/FYP_Thesis_2026-07-02_v4.docx`  
*Decisions:* D43  

**Paragraph/body blocks:** +47 added · −33 removed · ~32 edited · ⟳17 citation-renumber-only · tables: +1 −1 ~3 · front matter blocks 40→41

*Front matter (cover/abstract/TOC/lists):* +5 −4 ~9 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ DATA SNAPSHOT NOTICE (2026-07-02) - 128-TOKEN-ERA BUILD, HEADLINE SUPERSEDED
  - ❌ This thesis build presents the study at the retired 128-token generation budget. The primary study …
  - ❌ 2 July 2026
  - ❌ Five matched pairs / ten models / four families · three precisions · four benchmarks · HarmBench 51…
  - ❌ All experimental results reported in this thesis were produced by the open-source benchmarking fram…
  - ❌ Date: 2 July 2026
  - ❌ It contributes an open, reproducible benchmarking framework built around a matched-pair design (bas…
  - ❌ Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasin…
  - ❌ Three findings result. First, methodologically, the refusal regex systematically over-counts harmfu…
  - ❌ Table 6.2 Main study at the 512-token reference budget: per-pair deltas and labels (fp16 vs NF4).
  - ❌ Figure 4 Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen’s κ.
  - ❌ Figure 5 The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).
  - ➕ Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasin…
  - ➕ Three findings result. First, methodologically, the scorer and the generation budget each determine…
  - ➕ Figure 1 Study design: matched pairs, the precision arm, the benchmark battery, and the inference p…
  - ➕ Figure 2 System architecture: configuration, execution, benchmark plugins, scoring, and analysis.
  - ➕ Figure 3 TC1 execution model: workstation, head node, GPU compute node, and the return path.

### Sections removed
- ❌ Chapter 6 Results and Analysis § 6.6 Statistical caveats — 6.6 Statistical caveats

### Sections added
- ➕ Chapter 6 Results and Analysis § 6.3 The generation-budget artefact (why 512 tokens is the primary budget) — 6.3 The ge…
- ➕ Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives — 6.7 Statistical r…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A Reproducibility Statement | 0 | 0 | 1 | 0 |
| Appendix B Artefacts and Test Suite | 1 | 1 | 0 | 0 |
| Chapter 1 Introduction | 1 | 1 | 1 | 0 |
| Chapter 10 Conclusion | 1 | 1 | 1 | 0 |
| Chapter 2 Background and Related Work | 1 | 1 | 3 | 0 |
| Chapter 3 Methodology | 4 | 2 | 0 | 1 |
| Chapter 4 System Design and Implementation | 1 | 0 | 1 | 0 |
| Chapter 5 Experimental Setup | 2 | 1 | 2 | 0 |
| Chapter 6 Results and Analysis | 13 | 10 | 8 | 0 |
| Chapter 7 Discussion and Threats to Validity | 4 | 3 | 1 | 0 |
| Chapter 8 Limitations | 6 | 5 | 1 | 0 |
| Chapter 9 Future Work | 2 | 3 | 2 | 0 |
| References | 6 | 1 | 2 | 16 |

### Removed paragraphs (29 body; 4 front matter)

<details><summary>show</summary>

- `83a628ff` (177w) Chapter 1 Introduction § 1.4 Contributions — “This thesis makes four contributions. The first, and the one with consequences beyond thi…”
- `eea4ad98` (85w) Chapter 2 Background and Related Work § 2.3 LLM-as-judge evaluation and its validity — “Using one model to judge another’s output is now common, but its validity is an active co…”
- `ec868612` (101w) Chapter 3 Methodology § 3.4 Scoring: classifier-primary, regex-as-foil — “HarmBench Attack Success Rate is scored by the official HarmBench classifier (cais/HarmBe…”
- `b1268496` (96w) Chapter 3 Methodology § 3.6 Decoding controls and statistics — “All inference uses greedy decoding at temperature 0.0 with a fixed seed, so the only vari…”
- `b27f5be6` (49w) Chapter 5 Experimental Setup — “The five NF4 pairs and the judge validation ran first; the INT8 precision point and its j…”
- `92354323` (110w) Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts — “The first result concerns the instrument. The refusal regex reports far higher Attack Suc…”
- `ffe736a4` (55w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) — “Under the primary classifier, NF4 quantization never significantly reduces harmful compli…”
- `4dfc6c2d` (16w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) — “Table 6.2 Main study: per-pair deltas and interpretation labels (fp16 vs NF4). Bold-zero …”
- `9ea5bf49` (60w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) — “The Qwen3-1.7B effect, though significant, is modest and fragile: it is corroborated by M…”
- `fef2a603` (82w) Chapter 6 Results and Analysis § 6.3 Capability anchoring (RQ3) — “Every pair loses capability under NF4 in direction; the loss is significant on at least o…”
- `fbd209d9` (107w) Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5) — “Adding INT8 shows the effect is not a smooth function of bit-width. On capability the res…”
- `da75eca2` (68w) Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5) — “Pair \| fp16 \| INT8 \| NF4 \| Shape \|\| qwen_2b \| 0.135 \| 0.150 \| 0.190 \| rising; o…”
- `017ced17` (58w) Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5) — “The Llama INT8 effect is real but bounded: it rests on roughly eight to nine prompts, con…”
- `bf39b27f` (90w) Chapter 6 Results and Analysis § 6.6 Statistical caveats — “Three caveats bound the conclusions. With 200 HarmBench prompts the confidence intervals …”
- `34c07a83` (155w) Chapter 6 Results and Analysis § 6.6 Statistical caveats — “Significance is also nominal and, rather than merely noting this, we report the corrected…”
- `4bc6e969` (126w) Chapter 7 Discussion and Threats to Validity — “The results answer the research questions with calibrated, not sweeping, claims. NF4 neve…”
- `7915b4f0` (66w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “The primary safety construct is scored by the benchmark’s own fine-tuned classifier rathe…”
- `e0f07f3b` (56w) Chapter 7 Discussion and Threats to Validity § 7.3 External and statistical-conclusion validity — “External validity is bounded by five pairs, one decoding regime, and two bitsandbytes met…”
- `3c02114d` (17w) Chapter 8 Limitations — “Sample size: 200 HarmBench prompts give wide (±~0.05) confidence intervals; small deltas …”
- `9dbcd33e` (25w) Chapter 8 Limitations — “Multiple comparisons: significance flags are nominal and uncorrected; the one significant…”
- `550a2070` (16w) Chapter 8 Limitations — “Decoding: results condition on a single greedy decode; the multi-seed arm covers only the…”
- `3d52fa61` (20w) Chapter 8 Limitations — “Judge grounding: the HarmBench classifier is treated as reference without independent hum…”
- `6b3e1c5b` (23w) Chapter 8 Limitations — “The INT8 Llama effect rests on ~8–9 prompts on one pair and is non-monotonic; it is repor…”
- `b15cee9f` (36w) Chapter 9 Future Work — “Human-grounded judge validation (a stratified human-labelled subset reporting classifier-…”
- `61b02dc4` (24w) Chapter 9 Future Work — “Replication of the INT8 Llama effect across more models and decode seeds, to establish wh…”
- `edec25e7` (13w) Chapter 9 Future Work — “Extending the refusal-margin probe across all three precisions, not only the behavioural …”
- `40f85485` (104w) Chapter 10 Conclusion — “This thesis set out to determine whether the safety changes attributed to quantization in…”
- `0dfb9165` (12w) References — “[5] K. Egashira et al., “Exploiting LLM quantization,” arXiv preprint arXiv:2405.18137, 2…”
- `0fd87447` (58w) Appendix B Artefacts and Test Suite — “The framework ships 329 automated tests across the schema, loaders (including the fail-lo…”

</details>

### Added paragraphs (42)

<details><summary>show</summary>

- `3a6b45f4` (213w) Chapter 1 Introduction § 1.4 Contributions — “This thesis makes five contributions, and the first two have consequences beyond this stu…”
- `80bea896` (93w) Chapter 2 Background and Related Work § 2.3 LLM-as-judge evaluation and its validity — “Using one model to judge another’s output is now common, but its validity is an active co…”
- `6fab228c` (75w) Chapter 3 Methodology § 3.1 Matched-pair design — “Figure 1. Study design. Five matched model pairs spanning four families are evaluated at …”
- `7c06dc8b` (84w) Chapter 3 Methodology § 3.3 Benchmarks and metrics — “A threat-model boundary applies to the safety axis. The HarmBench standard configuration …”
- `46814c64` (269w) Chapter 3 Methodology § 3.4 Scoring: classifier-primary, regex-as-foil — “HarmBench Attack Success Rate is scored by the official HarmBench classifier (cais/HarmBe…”
- `b59056f3` (227w) Chapter 3 Methodology § 3.6 Decoding controls and statistics — “All inference uses greedy decoding at temperature 0.0 with a fixed seed, so the only vari…”
- `8b330d3a` (82w) Chapter 4 System Design and Implementation § 4.1 Architecture — “Figure 2. System architecture. A YAML configuration is validated by the Pydantic schema i…”
- `9633c0fe` (104w) Chapter 5 Experimental Setup — “The five NF4 pairs, the INT8 precision point, both judge validations, and the multi-seed …”
- `710a33ed` (55w) Chapter 5 Experimental Setup — “Figure 3. Execution model on the TC1 cluster. The head node is restricted to repository s…”
- `55b6e095` (264w) Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts — “The first result concerns the instrument. The refusal regex reports far higher Attack Suc…”
- `f42f4331` (60w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “Under the primary classifier at the 512-token reference budget, no pair shows a significa…”
- `476a5fce` (32w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “Table 6.2 Main study at the 512-token reference budget: per-pair judge deltas and interpr…”
- `7ff03975` (29w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “Exactly 3 primary contrasts survive BH-FDR: Qwen3-1.7B MMLU (−0.090, q = 0.008), Llama-3.…”
- `bcca99ac` (111w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “Over-refusal moves in the benign direction where it moves at all: the only over-refusal d…”
- `d44a121c` (43w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “The Qwen3-1.7B pair, whose +0.055 increase was the study’s original headline at the 128-t…”
- `875c7936` (442w) Chapter 6 Results and Analysis § 6.3 The generation-budget artefact (why 512 tokens is the primary budget) — “The 128-token budget of the study’s first run truncated the majority of generations befor…”
- `2372d948` (374w) Chapter 6 Results and Analysis § 6.4 Capability anchoring (RQ3) — “Every pair’s capability point estimates are non-positive under NF4 at the reference budge…”
- `2c263b72` (163w) Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5) — “Adding INT8 shows the effect is not a smooth function of bit-width. On capability the res…”
- `040163b7` (84w) Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5) — “Pair \| fp16 \| INT8 \| NF4 \| Shape (judge ASR) \|\| qwen_2b \| 0.255 \| 0.245 \| 0.255 …”
- `2bdbbc29` (94w) Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives — “With 200 HarmBench prompts the confidence intervals are wide (about ±0.05), and a power a…”
- `4d852c25` (152w) Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives — “Applying a Benjamini-Hochberg false-discovery-rate correction (q < 0.05) over the family …”
- `22202c4d` (285w) Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives — “Because the over-refusal member of every contrast above is regex-scored, and the §7.2 hum…”
- `43ce19a4` (325w) Chapter 7 Discussion and Threats to Validity — “The results answer the research questions with calibrated, not sweeping, claims. At the r…”
- `21b1fbd5` (357w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “The primary safety construct is scored by the benchmark’s own fine-tuned classifier rathe…”
- `d206339d` (695w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “Over-refusal remains regex-scored, a known weaker construct; the single over-refusal cont…”
- `359177ae` (102w) Chapter 7 Discussion and Threats to Validity § 7.3 External and statistical-conclusion validity — “External validity is bounded by five pairs, one decoding regime (greedy primary, with a f…”
- `9ee0950c` (39w) Chapter 8 Limitations — “Threat model: HarmBench was evaluated in its standard direct-request configuration only; …”
- `47b502f7` (49w) Chapter 8 Limitations — “Sample size and power: 200 HarmBench prompts give wide (about ±0.05) confidence intervals…”
- `c88f1036` (65w) Chapter 8 Limitations — “Decoding: the primary results condition on a single greedy decode; the multi-seed sensiti…”
- `0dd9b34f` (103w) Chapter 8 Limitations — “Judge grounding: the HarmBench classifier is now grounded against human labels on a strat…”
- `db1672fc` (59w) Chapter 8 Limitations — “Generation budget: absolute ASR is budget-dependent by construction (complete generations…”
- `b023c1d9` (94w) Chapter 8 Limitations — “Generation stopping: the generation loop halts only on the tokenizer’s default end-of-seq…”
- `2dbc58c8` (43w) Chapter 9 Future Work — “An adversarial-attack arm: re-run the HarmBench axis under at least one optimisation-base…”
- `569b3f10` (96w) Chapter 9 Future Work — “A second human annotator (inter-rater κ and adjudication of disagreements) extending the …”
- `c4765ddd` (184w) Chapter 10 Conclusion — “This thesis set out to determine whether the safety changes attributed to quantization in…”
- `d39bf7a4` (26w) References — “[7] K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” …”
- `b2024693` (58w) References — “[8] J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Barto…”
- `c2235892` (35w) References — “[17] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and…”
- `b02c53c9` (23w) References — “[18] R. J. Connor, “Sample size for testing differences in proportions for the paired-sam…”
- `8126f726` (21w) References — “[19] J. Cohen, “A coefficient of agreement for nominal scales,” Educational and Psycholog…”
- `01890d49` (24w) References — “[20] J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical …”
- `94c3cd22` (67w) Appendix B Artefacts and Test Suite — “The framework ships an automated verification suite across the schema, loaders (including…”

</details>

### Edited paragraphs (32; change window shown)

<details><summary>show</summary>

- (front matter): “2[−6][−ne][+ly] 2026”
- (front matter): “…/ four families · three precisions · four benchmarks · [−3][+HarmBench 51][−9][+-token r…”
- (front matter): “…k described herein, executed on the NTU TC1 GPU cluster[−, and are reproducible from the…”
- (front matter): “Date: 2[−6][−ne][+ly] 2026”
- (front matter): “…er as the primary judge and cross-checked by a second, [−architecturally ][+. Two scope …”
- (front matter): “Table 6.2 Main study[+ at the 512-token reference budget][−interpretation ]labels (fp16 v…”
- (front matter): “Figure [−1][+4][−'][+’]s κ.”
- (front matter): “Figure [−2][+5] The capability-anchored …”
- (front matter): “Figure [−3][+6] Precision sweep fp16 → I…”
- Chapter 1 Introduction § 1.2 Problem statement: “T[−wo][+hree][+The third is the generation budget: the number of tokens a model may gener…”
- Chapter 2 Background and Related Work § 2.1 Quantization and its behavioural effects: “…-precision formats. The bitsandbytes library provides t[+he t][− widely used][+can ][−s]…”
- Chapter 2 Background and Related Work § 2.2 Safety benchmarks and red-teaming: “HarmBench [[−7][+3][−s][+z][−8][+9][−9][+10][−7][+3][+ One scope boundary is stated at th…”
- Chapter 2 Background and Related Work § 2.4 The research gap: “…rts inconsistent, method- and model-dependent effects [[−3][+4][−6][+5][+, the judge lit…”
- Chapter 4 System Design and Implementation § 4.3 Reliability safeguards: “… an interrupted run continues without double-counting. [−A sui][+h][− of 329][+ ][+uite …”
- Chapter 5 Experimental Setup: “…y run, so cross-condition comparisons are controlled. T[+he full apparatus ran at two ge…”
- Chapter 5 Experimental Setup: “…Benchmarks \| HarmBench, XSTest, MMLU, ARC-Challenge \|\| [+Threat model \\| direct requ…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “… Cohen’s κ (regex vs classifier) \| Reading \|\| Qwen \| 0.[−19][+36][−37][+59][−p][+fai…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…ent (Cohen’s κ) by model family (fp16-vs-NF4 main study[+, 512-token reference budget]).”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “Figure [−1][+4][−v][+ight of t][−ry][+n][+s][−s][+; the two marginal exceptions (Llama ba…”
- Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget: “…SR (95% CI) \| Sig? \| ΔMMLU \| ΔARC \| Label \|\| qwen_2b \| [+0.000 [−0.055, ][− [+0.0…”
- Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget: “Figure [−2][+5][+ at the reference budget][+_512]/analysis/{judge_agreemen…”
- Chapter 6 Results and Analysis § 6.5 Mechanism: refusal-margin geometry: “To probe [−w][+t][+e boundar][+ behaviour behind][+128-token ][−m][+ign][−lest model move…”
- Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5): “…weep: HarmBench ASR at fp16 / INT8 / NF4 (primary judge[+, 512-token reference budget]).”
- Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5): “Figure [−3][+6][+ at the reference budget][−, while][+;][−i][+show][−meth][+n][−d-specifi…”
- Chapter 7 Discussion and Threats to Validity § 7.1 Internal validity: “… cause of a measured delta. The fail-loud loader guard [−ru][+fai][−e][− out][−s][+r][−b…”
- Chapter 8 Limitations: “… bitsandbytes methods (NF4, INT8); GPTQ, AWQ, and GGUF [+(the dominant on-device deploym…”
- Chapter 9 Future Work: “… genuinely different quantization families (GPTQ, AWQ, [+and especially ][+, the deploym…”
- Chapter 9 Future Work: “[−A][+Extending the refusal-margin probe across all three precisions, and a][−and][+with]…”
- Chapter 10 Conclusion: “Its most transferable lesson[+s][−is][+are][−:][+,][+nd there are two. A][−pl][+ch][−c][+…”
- References: “[[−3][+4][−arXi][+IEEE Access, ][+ol. 14,][−re][−r][+. 96771–96793, 2026, do][−nt][+: 10.…”
- References: “[[+2][−4][−arXiv prepr][−t][+ Findings of NAACL, 2024.][−, 2024].”
- Appendix A Reproducibility Statement: “… The project maps to the ML Reproducibility Checklist [[−15][+23][−pinn][+declar][+depen…”

</details>

### Citation-renumber-only paragraphs: 17 (text identical once [n] markers stripped; details in .full.json)

### Renamed headings (4)
- “6.2 Main study (fp16 vs NF4)” → “6.2 Main study (fp16 vs NF4) at the reference budget”
- “6.3 Capability anchoring (RQ3)” → “6.4 Capability anchoring (RQ3)”
- “6.4 Mechanism: refusal-margin geometry” → “6.5 Mechanism: refusal-margin geometry”
- “6.5 Precision point: fp16 → INT8 → NF4 (RQ5)” → “6.6 Precision point: fp16 → INT8 → NF4 (RQ5)”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
