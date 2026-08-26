## Full FYP report (docx-js line): v4 (128-token era) → v5 (512-token primary re-base)

*From:* `docs/archive/FYP_Report_2026-06-30_v4.docx`  
*To:* `docs/FYP_Report_2026-07-01_v5.docx`  
*Decisions:* D41  

**Paragraph/body blocks:** +141 added · −81 removed · ~100 edited · ⟳0 citation-renumber-only · tables: +3 −2 ~9 · front matter blocks 35→52

*Front matter (cover/abstract/TOC/lists):* +18 −3 ~10 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Interim Report
  - ❌ Document date: 2 July 2026
  - ❌ Comprehensive project report, revised through July 2026. Scope: five matched pairs across four fami…
  - ❌ Compact instruction-tuned language models are commonly deployed quantized, yet their safety is typi…
  - ❌ The design loads the baseline and quantized members of each pair from identical weights, applying q…
  - ❌ Under the classifier, four-bit NF4 never significantly reduces harmful compliance and significantly…
  - ❌ Figure 4 Quantization effect on harmful compliance (per-pair judge ΔASR forest plot).
  - ❌ Figure 5 The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).
  - ❌ Figure 6 Multi-seed decoding sensitivity for Qwen3-1.7B (greedy vs stochastic).
  - ❌ Figure 7 Per-category HarmBench ASR for Qwen3-1.7B, fp16 vs NF4.
  - ❌ Figure 8 Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.
  - ❌ Figure 9 Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).
  - ➕ Comprehensive Project Report
  - ➕ I hereby declare that this report is my own work and, to the best of my knowledge and belief, it co…
  - ➕ All experimental results reported here were produced by the open-source benchmarking framework desc…
  - ➕ _______________________________
  - ➕ Tan Uei Horng (UTAN001)
  - ➕ College of Computing and Data Science, Nanyang Technological University
  - ➕ Date: 2 July 2026
  - ➕ In the interest of transparency and in accordance with academic-integrity guidance and emerging ope…
  - ➕ HarmBench attack-success rate is reported at HarmBench's own 512-token reference budget as the prim…
  - ➕ Figure 1 Study design: matched pairs, the precision arm, the benchmark battery, and the inference p…
  - ➕ Figure 2 System architecture: configuration, execution, benchmark plugins, scoring, and analysis.
  - ➕ Figure 3 TC1 execution model: workstation, head node, GPU compute node, and the return path.

### Chapters / parts REMOVED
- ❌ Appendix H: Scorer Incident and Future-Session Handoff

### Chapters / parts ADDED
- ➕ Declaration of Originality
- ➕ Declaration on the Use of Generative AI Tools
- ➕ Appendix H: Scoring-Methodology Correction Record

### Sections removed
- ❌ Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.2 Harmful compliance: one significant delta…
- ❌ Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.4 Capability: directionally consistent wher…
- ❌ Appendix H: Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — H.5 Where the nex…

### Sections added
- ➕ Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.2 Harmful compliance: no significant increa…
- ➕ Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.4 Capability: the robust, budget-invariant …
- ➕ Appendix H: Scoring-Methodology Correction Record § H.5 Remaining construct-validity extension — H.5 Remaining construc…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Final TC1 Configuration | 10 | 0 | 3 | 0 |
| Appendix B: Example Generated SLURM Script | 0 | 0 | 1 | 0 |
| Appendix C: Pydantic Configuration Schema Summary | 1 | 0 | 1 | 0 |
| Appendix D: Test Inventory | 0 | 3 | 0 | 0 |
| Appendix D: Verification Coverage | 3 | 0 | 0 | 0 |
| Appendix E: Repository Layout | 9 | 3 | 2 | 0 |
| Appendix G: Document Revision History | 0 | 0 | 2 | 0 |
| Appendix H: Scorer Incident and Future-Session Handoff | 0 | 3 | 0 | 0 |
| Appendix H: Scoring-Methodology Correction Record | 2 | 0 | 4 | 0 |
| Chapter 10: Conclusion | 2 | 2 | 1 | 0 |
| Chapter 1: Introduction | 2 | 2 | 5 | 0 |
| Chapter 2: Literature Review | 1 | 1 | 4 | 0 |
| Chapter 3: Methodology | 4 | 2 | 9 | 0 |
| Chapter 4: System Design and Implementation | 4 | 3 | 6 | 0 |
| Chapter 5: Experimental Setup | 4 | 1 | 3 | 0 |
| Chapter 6: Results and Analysis | 52 | 42 | 35 | 0 |
| Chapter 7: Discussion and Threats to Validity | 2 | 2 | 5 | 0 |
| Chapter 8: Limitations | 8 | 4 | 1 | 0 |
| Chapter 9: Future Work | 3 | 2 | 5 | 0 |
| References | 16 | 8 | 3 | 0 |

### Removed paragraphs (78 body; 3 front matter)

<details><summary>show</summary>

- `fd520dda` (47w) Chapter 1: Introduction § 1.5 Contributions — “An open, reproducible benchmarking framework (approximately 2,800 lines of production Pyt…”
- `85557a2d` (75w) Chapter 1: Introduction § 1.5 Contributions — “An empirical characterisation of quantization across five model pairs, four families, and…”
- `4f8faa33` (188w) Chapter 2: Literature Review § 2.4 Quantization and Behavioural Effects — “A body of recent work has argued that quantization is not behaviourally neutral. Studies …”
- `dbb9504c` (42w) Chapter 3: Methodology § 3.3 Quantization Approach — “Four-bit loading is forbidden on a CPU runtime: the loader raises a clear error when quan…”
- `fefab43e` (64w) Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier — “The trade-off of deterministic parsing (its potential to miss nuanced or euphemistic refu…”
- `d7489e2e` (60w) Chapter 4: System Design and Implementation § 4.5 Benchmark Plugins — “Refusal detection is shared between HarmBench and XSTest through the benchmarks/utils.py …”
- `a101022a` (139w) Chapter 4: System Design and Implementation § 4.10 Testing — “The repository ships with a verification suite of 329 automated tests across twenty-five …”
- `4ffe4491` (37w) Chapter 4: System Design and Implementation § 4.11 CLI and Operational Interface — “The unified CLI fyp_cli.py exposes seven subcommands, summarised in Table 4.2. Each subco…”
- `1d21c6fa` (66w) Chapter 5: Experimental Setup § 5.7 Reproducibility Notes — “All runs share the same global seed (42), deterministic dataset shuffling, and determinis…”
- `79b66941` (34w) Chapter 6: Results and Analysis § 6.1 Results Table — “Figure 1. Quantization effect on harmful compliance: per-pair judge ΔASR (4-bit − fp16) w…”
- `fb71a3e2` (154w) Chapter 6: Results and Analysis § 6.1 Results Table — “Under the primary (judge) scorer, the statistically significant deltas are: Qwen 1.7B ΔAS…”
- `7d4d4c1a` (123w) Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization …”
- `6e2d05b4` (169w) Chapter 6: Results and Analysis § 6.3 Cross-Family Replication (RQ5) — “The headline finding for RQ5 is that, under the official classifier, NF4 quantization nev…”
- `e42a5553` (314w) Chapter 6: Results and Analysis § 6.4 Capability Anchoring — “The three pairs in this study illustrate why the capability anchor is necessary, even tho…”
- `ae6993c8` (319w) Chapter 6: Results and Analysis § 6.4 Capability Anchoring — “Because the capability anchor rests on a single MMLU subset, a structurally different sec…”
- `4ee6272f` (593w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “Five statistical limitations should be borne in mind when interpreting the results. First…”
- `b828700b` (182w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “Rather than only acknowledging the multiplicity problem, we report the corrected view. Ap…”
- `7e82750c` (112w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “A power analysis quantifies why so few effects reach significance. For a two-sided α = 0.…”
- `fcbad47c` (100w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Under the official HarmBench classifier, the Qwen 1.7B pair is the strongest result in th…”
- `2d64d244` (109w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The ΔASR finding is corroborated by a second, independent significance test. Because Harm…”
- `32138d1d` (105w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Note the sharp contrast with the secondary v2 proxy, which reports a much higher absolute…”
- `7893d472` (68w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The XSTest over-refusal rate declined modestly (0.052 → 0.028, ΔOR = −0.024, CI [−0.052, …”
- `a5c1747c` (100w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Subject-level MMLU breakdown supports the capability-decline reading. Accuracy declined i…”
- `0972413f` (79w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The Qwen 1.7B pair is the study's most consequential deployment finding: a 4-bit-quantize…”
- `83c5cf83` (53w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Figure 3. Qwen3-1.7B judge ΔASR under decoding variation: the single greedy headline (+0.…”
- `dacdb691` (188w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The judge ΔASR is positive in four of the five seeds (per-seed: −0.010, +0.050, +0.055, +…”
- `40a33ab1` (204w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “To check whether the Qwen 1.7B safety regression is concentrated in a single harm type or…”
- `0b460008` (105w) Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b) — “Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under t…”
- `d47f1ebf` (161w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “With all ten models evaluated and HarmBench ASR scored by the official classifier, it is …”
- `b9a7c06b` (139w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “It does not reduce it in any pair, and the one statistically significant move is an incre…”
- `94cbbf62` (203w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “The answer depends strongly on model scale and family. The Qwen3-1.7B model loses 8.7 per…”
- `78adf1ed` (220w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Yes, clearly, and on both axes. On capability, the MMLU delta ratio between Qwen 1.7B and…”
- `515c81b2` (200w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Partially. On capability, both families lose MMLU accuracy under NF4 compression where th…”
- `3215acbf` (162w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Three implications extend beyond the specific numbers. First, surface-level refusal metri…”
- `607de8d3` (155w) Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b) — “Under NF4 4-bit quantization, the Llama pair is classified as broad_degradation. HarmBenc…”
- `52cf9201` (119w) Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b) — “Why broad_degradation? With ΔASR within noise and a significant ΔMMLU, the rule falls thr…”
- `40cc0cb2` (143w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier, only one of the three ΔASR values is statistically signifi…”
- `f96df9c8` (129w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Two of three pairs lose MMLU accuracy significantly: Qwen 1.7B (−8.7 pp, CI [−0.137, −0.0…”
- `9d16ac3b` (217w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier, the three interpretation labels are broad_degradation (Qwe…”
- `ea6e6cbf` (112w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Model \| v2 non-refusal \| judge ASR \| agreement \| Cohen's κ \|\| qwen_2b_base \| 0.600…”
- `cc14d8e3` (151w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 2: the choice of scorer changes the conclusion. Under the judge, the per-pair Harm…”
- `1a9074ff` (274w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 4: the finding is robust to the choice of judge (second-judge cross-check). To rul…”
- `c7ec39bd` (313w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Threats and scope. The judge is itself a model and can err; it is adopted because it is t…”
- `e621f4fd` (219w) Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Mistral-7B is the clearest demonstration of the judge-over-proxy finding (§6.12, D16) in …”
- `2dd3eb7e` (145w) Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Phi-4-mini falls in the opposite (Llama-like) camp, where the proxy and the judge largely…”
- `46af30df` (52w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Figure 6. Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat t…”
- `664cda8e` (217w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Safety: not a single four-bit cliff, but two effects at different precisions. The safety …”
- `df28e876` (168w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Honest bounds on the INT8 safety finding. Two cautions are stated plainly so the effect i…”
- `34353a3b` (132w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The main study generates at most max_new_tokens = 128. HarmBench's own reference evaluati…”
- `cad70bf6` (110w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The longer budget changes the generations as intended. At 512 the HarmBench responses are…”
- `c5a32709` (138w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The generation-length rerun therefore reinforces, rather than threatens, the study's cent…”
- `06d2fcb4` (210w) Chapter 7: Discussion and Threats to Validity § 7.6 Deployment Implications — “The practical takeaway is asymmetric across the scale range studied. The most safety-rele…”
- `1f99a9da` (176w) Chapter 7: Discussion and Threats to Validity § 7.7 Positioning Against Prior Work — “These findings sit consistently within, and extend, the emerging quantization-and-safety …”
- `584a6eb5` (129w) Chapter 8: Limitations — “MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, …”
- `aa1778cc` (74w) Chapter 8: Limitations — “Greedy decoding only (partially addressed). Temperature 0.0 is used throughout the primar…”
- `95755278` (30w) Chapter 8: Limitations — “Sample-size-driven confidence intervals. With 200 HarmBench prompts and 250 benign XSTest…”
- `4c09e84d` (44w) Chapter 8: Limitations — “Qwen baseline provenance. The Qwen baselines are text-only derivatives of a multimodal Qw…”
- `c0016b7e` (71w) Chapter 9: Future Work — “Stochastic-decoding sensitivity arm (partially completed). The load-bearing Qwen 1.7B pai…”
- `06de05ae` (111w) Chapter 9: Future Work — “Fully-independent open-weight second judge. The primary HarmBench classifier has now been…”
- `b2ef02f8` (353w) Chapter 10: Conclusion — “All six original runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 6097…”
- `25773537` (243w) Chapter 10: Conclusion — “The results answer all five research questions. Under the official HarmBench classifier, …”
- `4bb29d20` (25w) References — “A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., “Investigating the impact of quantiza…”
- `b66c7576` (26w) References — “K. Chen, J. Zhang, J. Hu et al., “Q-resafe: Assessing safety risks and quantization-aware…”
- `8b062506` (19w) References — “K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in P…”
- `d456c640` (21w) References — “Y. Belkhiter, G. Zizzo, and S. Maffeis, “HarmLevelBench: Evaluating harm-level compliance…”
- `0fc37f90` (23w) References — “I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence …”
- `c33056a3` (30w) References — “A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusa…”
- `ddd5aa42` (23w) References — “Q. McNemar, “Note on the sampling error of the difference between correlated proportions …”
- `04082c03` (17w) References — “B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall…”
- `ea274bbf` (28w) Appendix D: Test Inventory — “The verification suite comprises 329 automated tests across twenty-five test files. The d…”
- `b0bcdb07` (621w) Appendix D: Test Inventory — “Test file \| Test count \| Coverage area \|\| test_datasets.py \| 27 \| Dataset loaders (…”
- `8f99c803` (8w) Appendix D: Test Inventory — “Table D.1 Distribution of automated tests across modules.”
- `c66809d3` (5w) Appendix E: Repository Layout — “│ └── tc1.yaml (TC1 cluster)”
- `9ff26f80` (7w) Appendix E: Repository Layout — “├── tests/ (329 tests across 25 files)”
- `1da4a784` (9w) Appendix E: Repository Layout — “├── results/ (raw.jsonl, summary.json, v2 score sidecars, analysis outputs)”
- `3b5885bf` (54w) Appendix H: Scorer Incident and Future-Session Handoff — “This appendix is included so that a future agent or fresh chat session can continue the w…”
- `a02ee197` (82w) Appendix H: Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — “The HarmBench judge validation is complete (job 61047) and the official classifier is the…”
- `aaccacb1` (118w) Appendix H: Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — “The most valuable optional extension is a second independent judge (for example LlamaGuar…”

</details>

### Added paragraphs (123)

<details><summary>show</summary>

- `ad4b383a` (45w) Chapter 1: Introduction § 1.5 Contributions — “An open, reproducible benchmarking framework with batched chat-templated generation, per-…”
- `9fb72d09` (96w) Chapter 1: Introduction § 1.5 Contributions — “An empirical characterisation of quantization across five model pairs, four families, and…”
- `257f43f2` (294w) Chapter 2: Literature Review § 2.4 Quantization and Behavioural Effects — “A body of recent work has argued that quantization is not behaviourally neutral. A compre…”
- `6fab228c` (75w) Chapter 3: Methodology § 3.1 Experimental Design — “Figure 1. Study design. Five matched model pairs spanning four families are evaluated at …”
- `516499bf` (71w) Chapter 3: Methodology § 3.3 Quantization Approach — “Four-bit loading is forbidden on a CPU runtime: the loader raises a clear error when quan…”
- `0fc8bada` (192w) Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier — “The trade-off of deterministic parsing (its potential to miss nuanced or euphemistic refu…”
- `457a1eaa` (123w) Chapter 3: Methodology § 3.8 Interpretation Framework — “Boundary convention. The thresholds are inclusive: a ΔASR of exactly +0.02 classifies as …”
- `8b330d3a` (82w) Chapter 4: System Design and Implementation § 4.1 Architecture Overview — “Figure 2. System architecture. A YAML configuration is validated by the Pydantic schema i…”
- `ad791a64` (73w) Chapter 4: System Design and Implementation § 4.5 Benchmark Plugins — “The deterministic refusal layer is shared through benchmarks/utils.py, whose match_refusa…”
- `0bcdb138` (70w) Chapter 4: System Design and Implementation § 4.10 Testing — “The repository ships with an automated verification suite whose live inventory is collect…”
- `81a1e364` (55w) Chapter 4: System Design and Implementation § 4.11 CLI and Operational Interface — “The unified CLI fyp_cli.py exposes nine subcommands, summarised in Table 4.2: seven study…”
- `710a33ed` (55w) Chapter 5: Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints — “Figure 3. Execution model on the TC1 cluster. The head node is restricted to repository s…”
- `6290dbc9` (7w) Chapter 5: Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “Code listing 5.1 Head-node pre-cache invocation (scripts/prefetch_tc1.py).”
- `c35d36e6` (113w) Chapter 5: Experimental Setup § 5.6 Run Plan — “The cross-family extension (run on TC1 on 2026-06-15; results in §6.13) added four more m…”
- `f0706bc5` (239w) Chapter 5: Experimental Setup § 5.7 Reproducibility Notes — “All runs share the same global seed (42), deterministic dataset shuffling, and determinis…”
- `99b03e06` (49w) Chapter 6: Results and Analysis § 6.1 Results Table — “Figure 4. Quantization effect on harmful compliance at the 512-token reference budget: pe…”
- `fa9d24de` (245w) Chapter 6: Results and Analysis § 6.1 Results Table — “Under the primary (judge) scorer at the 512-token reference budget, the HarmBench deltas …”
- `7ff03975` (29w) Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Exactly 3 primary contrasts survive BH-FDR: Qwen3-1.7B MMLU (−0.090, q = 0.008), Llama-3.…”
- `d198f1f6` (171w) Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization …”
- `99965a99` (187w) Chapter 6: Results and Analysis § 6.3 Cross-Family Replication (RQ5) — “The headline finding for RQ5 is that, at the 512-token reference budget under the officia…”
- `009d1059` (39w) Chapter 6: Results and Analysis § 6.4 Capability Anchoring — “Over-refusal regression: HarmBench ASR and MMLU are held within tolerance, but the model …”
- `f4050278` (342w) Chapter 6: Results and Analysis § 6.4 Capability Anchoring — “The five pairs in this study illustrate why the capability anchor is necessary, especiall…”
- `829bae90` (670w) Chapter 6: Results and Analysis § 6.4 Capability Anchoring — “Because the capability anchor rests on a single MMLU subset, a structurally different sec…”
- `f0fc2e90` (1045w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “Five statistical limitations should be borne in mind when interpreting the results. First…”
- `effaac0d` (239w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “Rather than only acknowledging the multiplicity problem, we report the corrected view. Ap…”
- `5cd10fb5` (286w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “Because the over-refusal member of every contrast above is regex-scored, and the §6.12 Re…”
- `e8729792` (195w) Chapter 6: Results and Analysis § 6.5 Statistical Caveats — “A power analysis — Connor's normal-approximation sample-size treatment of the paired McNe…”
- `f90dd79f` (136w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Under the official HarmBench classifier at the 512-token reference budget, the Qwen 1.7B …”
- `2260186f` (156w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “This null on the harm axis is unambiguous under the paired-binary test. Because HarmBench…”
- `9efef659` (70w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The secondary v2 proxy tells a consistent story: it reports a small negative delta (ΔASR …”
- `ef026439` (116w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The XSTest over-refusal rate declined (0.052 → 0.028, ΔOR = −0.024), a non-significant ef…”
- `6e5afea6` (65w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The MMLU decline is the load-bearing effect for this pair: accuracy falls from 0.643 to 0…”
- `e66c092b` (81w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “The Qwen 1.7B pair is thus a capability-only degradation at the reference budget: a 4-bit…”
- `11e56d02` (50w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Figure 6. Qwen3-1.7B judge ΔASR under decoding variation at the 512-token reference budge…”
- `4329d7bb` (532w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “For Qwen 1.7B the per-seed judge ΔASR is 0.000, +0.010, +0.020, 0.000, +0.035 (mean +0.01…”
- `b5a794cc` (206w) Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b) — “Because the aggregate Qwen 1.7B ΔASR is exactly 0.000 at the reference budget, there is n…”
- `de2930dd` (128w) Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b) — “Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under t…”
- `83ac40be` (207w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “With all ten models evaluated and HarmBench ASR scored by the official classifier at its …”
- `d7c62cc7` (185w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “At the 512-token reference budget it does not, and it does not significantly increase it …”
- `5681dd97` (204w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Yes, and this is the robust, budget-invariant cost of NF4. The Qwen3-1.7B model loses 9.0…”
- `5d75092c` (221w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “On capability, yes; on harmful compliance, not at the reference budget. On capability the…”
- `8469c547` (221w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Partially, and consistently in the safety-null / capability-cost direction. On capability…”
- `7bf32641` (249w) Chapter 6: Results and Analysis § 6.9 Research Question Synthesis — “Three implications extend beyond the specific numbers. First, surface-level refusal metri…”
- `763888e7` (136w) Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b) — “Under NF4 4-bit quantization, the Llama pair is classified as capability_collapse_masquer…”
- `634d44c4` (153w) Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b) — “Why capability_collapse_masquerading_as_safety? Harmful compliance decreases and capabili…”
- `f08c906c` (242w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier at the 512-token reference budget, exactly one of the three…”
- `bea1e19f` (203w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Capability degradation, not safety change, is the robust cost of NF4 at the reference bud…”
- `8a9d8e4f` (328w) Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier at the 512-token reference budget, the three interpretation…”
- `e7b3eba1` (112w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Model \| v2 non-refusal \| judge ASR \| agreement \| Cohen's κ \|\| qwen_2b_base \| 0.595…”
- `70ad5a34` (232w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 2: the choice of scorer changes the conclusion. Under the judge, at the 512-token …”
- `1331cfbc` (288w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 4: the finding is robust to the choice of judge (second-judge cross-check). To rul…”
- `64e4dcb1` (590w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 5: the classifier tracks human judgment; the regex does not. The two preceding cro…”
- `fa6341eb` (607w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Threats and scope. The judge is itself a model and can err; it is adopted because it is t…”
- `86a2150c` (507w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 6: the over-refusal axis is scorer-sensitive too, and the one over-refusal survivo…”
- `80355827` (90w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Pair \| Regex ΔOR (primary) \| Judge ΔOR strict (95% CI; p) \| Judge ΔOR broad (95% CI; p…”
- `a30355f9` (86w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Table 6.4 Exploratory per-pair XSTest over-refusal deltas (NF4 − fp16) under the independ…”
- `50828531` (609w) Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “The human grounding for this axis has since been completed, mirroring Result 5. All 200 i…”
- `dada65ef` (220w) Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Mistral-7B is the clearest demonstration of the judge-over-proxy finding (§6.12, D16) in …”
- `2740bcd7` (236w) Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Phi-4-mini falls in the opposite (Llama-like) camp, where the proxy and the judge largely…”
- `21c39014` (63w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Figure 9. Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat t…”
- `2626e538` (207w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Safety: no robust precision effect at the reference budget. The safety axis is the more r…”
- `f8c6b907` (197w) Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Honest reading of the INT8 result. The lesson of the 128-to-512 comparison is stated plai…”
- `08f8b360` (232w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The HarmBench authors show that the number of tokens generated during evaluation can chan…”
- `b452a41a` (305w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The reference budget changes the generations materially, which is the crux of why it must…”
- `a6aff32d` (254w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “A qualitative note on the form of these longer generations, surfaced while inspecting res…”
- `79e2f3da` (16w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “Table 6.5 HarmBench ΔASR across the retained 128-token comparison and the 512-token refer…”
- `2baaa9d7` (231w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The generation-length comparison therefore establishes the study's central dichotomy at t…”
- `0c580771` (413w) Chapter 7: Discussion and Threats to Validity § 7.6 Deployment Implications — “The practical takeaway is asymmetric across the scale range studied, but the asymmetry is…”
- `9ac76b16` (348w) Chapter 7: Discussion and Threats to Validity § 7.7 Positioning Against Prior Work — “These findings sit within, qualify, and extend the emerging quantization-and-safety liter…”
- `47e7dca4` (89w) Chapter 8: Limitations — “Threat model: direct requests only. HarmBench was evaluated in its standard configuration…”
- `e184ba23` (272w) Chapter 8: Limitations — “MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, …”
- `040afcef` (159w) Chapter 8: Limitations — “Generation-budget dependence of the absolute rate (directly evidenced, then controlled). …”
- `84fb1c60` (161w) Chapter 8: Limitations — “Greedy decoding only (stochastic-decoding arm now complete for all five pairs). Temperatu…”
- `37e8a79a` (132w) Chapter 8: Limitations — “Generation not halted at the chat turn boundary. The generation loop stops only on the to…”
- `61a1f32a` (55w) Chapter 8: Limitations — “Benchmark contamination unassessed. Whether benchmark items appear in the evaluated model…”
- `7ed8f08b` (115w) Chapter 8: Limitations — “Sample-size-driven confidence intervals and a bounded null. With 200 HarmBench prompts an…”
- `0452e99b` (102w) Chapter 8: Limitations — “Model-weight provenance (pinned). Each of the five model repositories is pinned by Huggin…”
- `0d58cf72` (61w) Chapter 9: Future Work — “Adversarial-attack arm. Re-run the HarmBench axis under at least one optimisation-based o…”
- `d9a17976` (169w) Chapter 9: Future Work — “Stochastic-decoding sensitivity arm (completed for all five pairs at the reference budget…”
- `5ae94029` (327w) Chapter 9: Future Work — “Fully-independent open-weight second judge, and a second human annotator. The primary Har…”
- `44b07f63` (549w) Chapter 10: Conclusion — “All six original runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 6097…”
- `9e5cc0e2` (340w) Chapter 10: Conclusion — “The results answer all five research questions at HarmBench's 512-token reference budget.…”
- `c63f6abf` (29w) References — “R. Jin, J. Du, W. Huang, et al., “A comprehensive evaluation of quantization strategies f…”
- `0fc37f90` (23w) References — “I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence …”
- `219fd83d` (33w) References — “A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., “Investigating the impact of quantiza…”
- `8b062506` (19w) References — “K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in P…”
- `2e73427a` (31w) References — “Y. Belkhiter, G. Zizzo, and S. Maffeis, “HarmLevelBench: Evaluating harm-level compliance…”
- `ae569425` (31w) References — “K. Chen, J. Zhang, J. Hu et al., “Q-resafe: Assessing safety risks and quantization-aware…”
- `6d586109` (57w) References — “J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Bartoldso…”
- `04082c03` (17w) References — “B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall…”
- `ddd5aa42` (23w) References — “Q. McNemar, “Note on the sampling error of the difference between correlated proportions …”
- `75e23254` (34w) References — “Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powe…”
- `afccfc8e` (20w) References — “J. Cohen, “A coefficient of agreement for nominal scales,” Educational and Psychological …”
- `62606b90` (22w) References — “R. J. Connor, “Sample size for testing differences in proportions for the paired-sample d…”
- `25d8d5f1` (23w) References — “J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,…”
- `43d837ac` (16w) References — “F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no…”
- `f7435a2f` (17w) References — “J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale, NJ: …”
- `c33056a3` (30w) References — “A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusa…”
- `ba15c41c` (8w) Appendix A: Final TC1 Configuration — “revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e # T32: TC1-cached snapshot pinned 2026…”
- `ba15c41c` (8w) Appendix A: Final TC1 Configuration — “revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e # T32: TC1-cached snapshot pinned 2026…”
- `1254f9c3` (8w) Appendix A: Final TC1 Configuration — “revision: 1cfa9a7208912126459214e8b04321603b3df60c # T32: TC1-cached snapshot pinned 2026…”
- `1254f9c3` (8w) Appendix A: Final TC1 Configuration — “revision: 1cfa9a7208912126459214e8b04321603b3df60c # T32: TC1-cached snapshot pinned 2026…”
- `a194fae8` (8w) Appendix A: Final TC1 Configuration — “revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95 # T32: TC1-cached snapshot pinned 2026…”
- `a194fae8` (8w) Appendix A: Final TC1 Configuration — “revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95 # T32: TC1-cached snapshot pinned 2026…”
- `d9ee524b` (8w) Appendix A: Final TC1 Configuration — “revision: c170c708c41dac9275d15a8fff4eca08d52bab71 # T32: TC1-cached snapshot pinned 2026…”
- `d9ee524b` (8w) Appendix A: Final TC1 Configuration — “revision: c170c708c41dac9275d15a8fff4eca08d52bab71 # T32: TC1-cached snapshot pinned 2026…”
- `d10f005d` (8w) Appendix A: Final TC1 Configuration — “revision: cfbefacb99257ffa30c83adab238a50856ac3083 # T32: TC1-cached snapshot pinned 2026…”
- `d10f005d` (8w) Appendix A: Final TC1 Configuration — “revision: cfbefacb99257ffa30c83adab238a50856ac3083 # T32: TC1-cached snapshot pinned 2026…”
- `7df23f85` (37w) Appendix C: Pydantic Configuration Schema Summary — “quant_method : str \| None : quantization method when quantized=true — "nf4" (default, th…”
- `a76e186c` (36w) Appendix D: Verification Coverage — “The test inventory is intentionally not frozen into this document. Run pytest tests/ --co…”
- `80aff6aa` (110w) Appendix D: Verification Coverage — “Coverage area \| What is verified \|\| Data and configuration \| Dataset loaders, schema …”
- `0eb3cd67` (14w) Appendix D: Verification Coverage — “Table D.1 Stable verification coverage areas. The live test inventory is produced by pyte…”
- `55ed3b3f` (7w) Appendix E: Repository Layout — “│ ├── tc1_512.yaml (primary 512-token fp16/NF4 study)”
- `f5f7a68d` (8w) Appendix E: Repository Layout — “│ ├── tc1_int8_512.yaml (INT8 precision point, primary 512-token)”
- `f6dc544f` (6w) Appendix E: Repository Layout — “│ ├── tc1.yaml (retained 128-token comparison)”
- `6dcba367` (7w) Appendix E: Repository Layout — “│ └── tc1_int8.yaml (retained 128-token INT8 comparison)”
- `8ca9de84` (7w) Appendix E: Repository Layout — “├── scripts/ (analysis + build: precision_sweep_analysis.py, judge_agreement.py,”
- `f8a656d4` (5w) Appendix E: Repository Layout — “│ make_figures.py, prefetch_tc1.py, verify_report_claims.py, ...)”
- `7004c1ea` (7w) Appendix E: Repository Layout — “├── tests/ (live inventory: pytest tests/ --collect-only)”
- `6a6abb09` (7w) Appendix E: Repository Layout — “├── results_512/ (primary 512-token results and analysis)”
- `15522870` (5w) Appendix E: Repository Layout — “├── results/ (retained 128-token comparison)”
- `590c0bb1` (54w) Appendix H: Scoring-Methodology Correction Record — “This appendix documents, for transparency and auditability, the scoring-methodology corre…”
- `5de9e47b` (105w) Appendix H: Scoring-Methodology Correction Record § H.5 Remaining construct-validity extension — “The classifier has since been cross-checked three times: by an independent frontier-model…”

</details>

### Edited paragraphs (100; change window shown)

<details><summary>show</summary>

- (front matter): “Document date: 2[−6][−ne][+ly] 2026”
- (front matter): “[−I][+Comprehe][+sive projec][−erim][−ne][+ly][−A][+This is the full technical write-up; …”
- (front matter): “Compact instruction-tuned language models are [−al][+com][−st a][+n][−wa][−s] deployed qu…”
- (front matter): “…dentical weights, applying quantization on the fly, so [+the paired contrast isolates th…”
- (front matter): “Figure [−1][+4] Quantization effect on h…”
- (front matter): “Figure [−2][+5] The capability-anchored …”
- (front matter): “Figure [−3][+6] Multi-seed decoding sens…”
- (front matter): “Figure [−4][+7] Per-category HarmBench A…”
- (front matter): “Figure [−5][+8] Scorer validation: judge…”
- (front matter): “Figure [−6][+9] Precision sweep fp16 → I…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…default. End users encountering these models therefore [+typic][−most a][−wa][−s] intera…”
- Chapter 1: Introduction § 1.3 Research Questions: “…bserved within the Qwen family replicate qualitatively [−in][+across][−a diff][+oth][−en…”
- Chapter 1: Introduction § 1.4 Scope and Boundaries: “…ime. The study targets compact instruction-tuned models[+. The primary matched pairs sit…”
- Chapter 1: Introduction § 1.5 Contributions: “…ond, architecturally independent judge at Cohen's κ 0.6[−0][+8][−relo][+h][−t][+ng][+bot…”
- Chapter 1: Introduction § 1.6 Report Structure: “… Chapter 9 proposes future work. Chapter 10 concludes. [−S][+E][−x][+ght][+, and record …”
- Chapter 2: Literature Review § 2.1 Large-Scale Ethical Benchmarking of LLMs: “…e ethical performance of open-source LLMs. TrustLLM [1][−, DecodingTrust [2], and Safety…”
- Chapter 2: Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “…suring opposite sides of this trade-off. HarmBench [4] [−and AdvBench ][+s][+s] attack s…”
- Chapter 2: Literature Review § 2.2 The Helpfulness–Harmlessness Trade-off: “…Multitask Language Understanding benchmark, has become [+a s][−he ][+an][−e f][−cto][+rd…”
- Chapter 2: Literature Review § 2.5 Research Gaps Targeted by This Work: “…by the choice of Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B[+ (extended by the Mistral-7B an…”
- Chapter 3: Methodology § 3.1 Experimental Design: “…the same chat template, and the same release artifact. [−An][+B][+c][−b][+n][−e][+t][−ve…”
- Chapter 3: Methodology § 3.1 Experimental Design: “…ization state (baseline versus four-bit), model scale ([−two][+1.7][+ in the primary des…”
- Chapter 3: Methodology § 3.2 Model Selection: “Three model pairs [+form the prim][−e][+y][−eva][+contro][−uat][+l][+ design][+ [17]][+[1…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “…ng is deterministic and seed-controlled. Capability is [−a][+reported on][−-be][+ a][−ma…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “…liance benchmark because it is the most widely adopted [−advers][+h][−ia][+mfu][−p][+beh…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “…soning demands without exhausting the time budget. The [+six subjec][+s are p][+oled and…”
- Chapter 3: Methodology § 3.6 Decoding Controls: “…\| top_p \| 1.0 \| Disabled (greedy). \|\| max_new_tokens \| [+5][−8][− Sufficient for s…”
- Chapter 3: Methodology § 3.7 Metrics and Pairwise Analysis: “Bootstrap 95% confidence intervals[+ [19]] are computed for each me…”
- Chapter 3: Methodology § 3.7 Metrics and Pairwise Analysis: “…tbook-correct significance test is McNemar's exact test[+ [20]][+ The same exact paired …”
- Chapter 4: System Design and Implementation § 4.1 Architecture Overview: “…rface: base.py (ABC), harmbench.py, xstest.py, mmlu.py,[+ arc.py, registry.py, and] util…”
- Chapter 4: System Design and Implementation § 4.3 Model Loading and Quantization Path: “…path that BitsAndBytes does not support. Second, after [−the ][− call][−m][+f][+ur suppo…”
- Chapter 4: System Design and Implementation § 4.4 Generation Pipeline: “Generation itself uses torch.[−i][−fe][+o_g][−ence_mo][+a][−e] and a do_sample=False de…”
- Chapter 4: System Design and Implementation § 4.8 SLURM Job Generation: “…, one sbatch is generated per (model, benchmark) pair ([−twen][+for][−-four][−six][+ten]…”
- Chapter 4: System Design and Implementation § 4.8 SLURM Job Generation: “…with the model alias and configuration path passed in. [−Six][+Ten] sbatch files are pro…”
- Chapter 4: System Design and Implementation § 4.11 CLI and Operational Interface: “…. \|\| cluster-submit \| Submit the generated sbatch files[+ (local/dev convenience][+).…”
- Chapter 5: Experimental Setup § 5.2 Software Environment: “…ents.txt) \|\| pytest \| (installed via requirements.txt; [−329][+run pytest][+/][−pass]…”
- Chapter 5: Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache: “…_arc, added as the second capability benchmark), and th[−re][− (Qwen3-1.7B, Qwen3-4B, Ll…”
- Chapter 5: Experimental Setup § 5.6 Run Plan: “With group_by=model, the framework emits [−six][+one][−es, on][+ — six for the original t…”
- Chapter 6: Results and Analysis: “…ed two further matched pairs (Mistral-7B-Instruct-v0.3 [+[21] ][+ [22]]) run on the same…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…h classifier (the primary scorer; see §6.1.1 and §6.12)[+ at HarmBench's own 512-token g…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…(95% CI) \| Sig? \|\| qwen_2b \| HarmBench ASR (judge) \| 0.[−13][+25][−19][+255 \\| 0.0…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “… with paired bootstrap 95% CIs on the deltas. "Sig?" = [−whether ][+ and McNemar's exact…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…b \| mistral_7b \| phi4_mini \|\| HarmBench ΔASR (judge) \| [−+][−55 ★][+00][−25 \\| 0.0…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…v2 proxy row is shown for transparency. ★ marks deltas [−w][+significant under bot][−os]…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…udge) scorer, Qwen3-1.7B's baseline HarmBench ASR is 0.[−13][+25][−06][+11][−7][+14][−20…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…uantization side, read under the primary (judge) scorer[+ at the reference budget][− wor…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “Figure [−2][+5]. The capability-anchored…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “… of the existing artifacts. To estimate how stable the [−headline ][+ result][−th][+all …”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “Figure [−4][+7][+, 512-token reference budget][−+][+ΔASR is ][−55][+00,][−in][+so per-cat…”
- Chapter 6: Results and Analysis § 6.7 Preliminary Scale Observations: Baseline Comparison: “…fety coupling within the Qwen family. At full precision[+ (HarmBench under the official …”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “Under the official classifier[+ at the 512-token reference budget][−06][+11][−090][+155][…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “…otion from the secondary v2 proxy, under which Qwen 4B [−had][+reports][−ppeared][−to be…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “…sal-calibrated on genuine harm (judge baseline ASR = 0.[−06][+11][−move][+shift][−0][+4]…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “Across the five pairs, [−four show ][+ne moves][− change][+ly][+ the][−al][+ing direction…”
- Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b): “The baseline profile of Llama 3.2 3B-Instruct is [−the mo][+rongly][− of the three models…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…nformative. At similar MMLU capability (Qwen3-1.7B: 0.6[−20][+43][+ at the 512-token ref…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “The XSTest over-refusal rate is [−the most][+a][−1.7B (−0.024, CI [−0.052, 0.000]), Qwen3…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…e model refuse a benign prompt?") and remains v2-scored[+ as primary][−l][+with an ind][…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Methodology and setup. [−T][+For the primary 512-token study t][−six][+ten][−original][+f…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “… Agreement and Cohen's κ are over the shared prompt set[+, computed at the 512-token ref…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Figure [−5][+8][−v][+ight of th][−ry][+ ten][+s][−s][+; the two marginal exceptions (Llam…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…efusal rate for the Qwen models (e.g. Qwen 1.7B base 0.[−13][+25][−600][+595][−06][+11][…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “Result 3: the [+null-inc][+ase conc][−ocat][+us][+:][+at 512 tok][−xactly:][+ns][−0][+1][…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…rn holds and strengthens. Under the official classifier[+ at the 512-token reference bud…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…ehavioural results (§6.3, §6.12, §6.13) establish that [+at the 512-token reference budg…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “Central result[+ (computed on the 128-token behavioural flips; the first-token margins ar…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…) is statistically significant for every pair (Wilcoxon[+ signed-rank [30]][+[31] ][−8][…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “… All five INT8 members were run on the NTU TC1 cluster [+at the 512-token reference budg…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…cliff at four-bit. The capability answer is unambiguous[+ and holds at the 512-token ref…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…it does at fp16 and NF4: judge-versus-proxy Cohen κ is [−0.11][+lowest][−,][−ing ASR 0.8…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…ifier) \| ΔASR @512 (gpt-4o) \|\| Qwen3-1.7B \| +0.055 (sig[+ @128]) \| 0.000 \| +0.005 …”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “The headline [−result ][+compa][−eru][+iso][+afety signal seen at 128 tokens was a trunca…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…rast, is essentially unchanged by the generation budget[+, which is why it — not the saf…”
- Chapter 7: Discussion and Threats to Validity: “…l grounding for the threats discussed in this chapter. [−T][+At the 512-token reference …”
- Chapter 7: Discussion and Threats to Validity § 7.1 Internal Validity: “…entical baseline weights, isolates quantization as the [−sole ][+ against model-provenan…”
- Chapter 7: Discussion and Threats to Validity § 7.2 External Validity: “External validity is bounded by [−three][+six][+, the safety axis is measured under direc…”
- Chapter 7: Discussion and Threats to Validity § 7.3 Construct Validity: “…on of "harmful compliance" against a particular set of [+harmful behaviour requests, pre…”
- Chapter 7: Discussion and Threats to Validity § 7.4 Scorer Choice and Remaining Validity Threat: “… strongly per prompt (κ 0.60–0.95 across all ten models[−)][+,][+κ 0.68–0.95 at ][+512-t…”
- Chapter 8: Limitations: “…nt ARC loss MMLU missed, so capability is now reported [−as][+on][−a][+the][−-][+ ][+s][…”
- Chapter 9: Future Work: “…different PTQ algorithms perturb safety and capability.[+ GGUF is the deployment-relevan…”
- Chapter 9: Future Work: “…artly benchmark-specific, notably the Qwen 1.7B drop (−[−8][+9][−7][+0][+, significant,]…”
- Chapter 9: Future Work: “…tivation-space refusal-direction probe (Arditi et al. [[−19][+32][+only the four-b][−is …”
- Chapter 9: Future Work: “…ables 6.1–6.3 and analysed in §6.13: under the judge, n[−either new][+o][−ASR][+harmful-…”
- Chapter 9: Future Work: “… a third precision and run on TC1 across all five pairs[+ at the 512-token reference bud…”
- Chapter 10: Conclusion: “…TC1 cluster with resumable per-model matrix jobs, and a[+n automated][− of 329 automated…”
- References: “[+Y. Huang, ][+,][+Position: ][+PMLR vol. 235, ]2024. arXiv:2401.05561.”
- References: “…uating the safety of large language models,” in Proc. A[+ssociation for Computational Li…”
- References: “[−M][+A][−d][+ouelen][+-Mini][+: Compact yet powerful multimodal language models via Mixt…”
- Appendix A: Final TC1 Configuration: “The full configs/tc1[+_512][+ — the configuration of the primary 512-token study (D41) —]…”
- Appendix A: Final TC1 Configuration: “max_new_tokens: [+5][−8]”
- Appendix A: Final TC1 Configuration: “log_dir: results[+_512]/slurm_logs_tc1”
- Appendix B: Example Generated SLURM Script: “…e generated sbatch script for qwen_2b_4bit, one of the [−six][+ten][−f][+n][−v][+n]e scr…”
- Appendix C: Pydantic Configuration Schema Summary: “quantized : bool : true triggers [−NF4][+on-the-fly quantized] loading.”
- Appendix E: Repository Layout: “…benchmarks/{base.py, harmbench.py, xstest.py, mmlu.py, [+arc.py, ]registry.py, utils.py}”
- Appendix E: Repository Layout: “└── FYP_Report_2026-0[−6][+7][−3][+1][−4][+5].docx (this document)”
- Appendix G: Document Revision History: “When (UTC+8) \| Version \| Change to the report \|\| 2026-0[+7-19 13:30 \\| FYP_Report_20…”
- Appendix G: Document Revision History: “…he source builder script is `scripts/build_fyp_report_v[−3][+5].js`; the docx is a build…”
- Appendix H: Scoring-Methodology Correction Record § H.1 What went wrong in v1: “…he judge, the v2 regex is shown to over-count ASR, and [+— at ][−s][+hen-current 128-tok…”
- Appendix H: Scoring-Methodology Correction Record § H.2 How the correction was made: “…mary.json` in place. That approach was rejected during [−C][+c][−x][−aud][+rev][−t][+ew]…”
- Appendix H: Scoring-Methodology Correction Record § H.3 Current artifact contract: “Artifact \| Status \| [−R][+Handling r][− for future work][+the TC1 judge jobs (512-token…”
- Appendix H: Scoring-Methodology Correction Record § H.4 Verification completed: “…erified in five ways. First, the full test suite passes[− with 329 tests][− v2][− test][…”

</details>

### Renamed headings (1)
- “Appendix D: Test Inventory” → “Appendix D: Verification Coverage”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
