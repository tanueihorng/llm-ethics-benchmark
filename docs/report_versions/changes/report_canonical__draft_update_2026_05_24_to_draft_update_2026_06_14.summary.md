## Full FYP report (docx-js line): Draft update (2026-05-24) → Draft update (2026-06-14)

*From:* `docs/archive/FYP_Report_2026-05-24.docx`  
*To:* `docs/archive/FYP_Report_2026-06-14.docx`  

**Paragraph/body blocks:** +172 added · −36 removed · ~43 edited · ⟳0 citation-renumber-only · tables: +7 −4 ~4 · front matter blocks 29→29

*Front matter (cover/abstract/TOC/lists):* +3 −3 ~2 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Document date: 27 May 2026
  - ❌ Revision: incorporates TC1 user-guide policy review
  - ❌ This work designs and implements a research-grade benchmarking framework to study safety–capability…
  - ❌ Each pair is scored on three complementary benchmarks: HarmBench (harmful compliance, measured as A…
  - ❌ Table 6.1 Per-pair reporting schema.
  - ➕ Revision: HarmBench ASR scored by the official HarmBench classifier (judge-primary, D16); v2 refusa…
  - ➕ Each pair is scored on three complementary benchmarks: HarmBench (harmful compliance, measured as A…
  - ➕ Table 6.1 Per-pair benchmark results for all five model pairs with bootstrap 95% CIs.

### Chapters / parts ADDED
- ➕ Appendix H — Scorer Incident and Future-Session Handoff

### Sections removed
- ❌ Chapter 6 — Intended Results and Analysis Plan § 6.1 Reporting Schema — 6.1 Reporting Schema
- ❌ Chapter 7 — Discussion and Threats to Validity § 7.4 Refusal Parser Trade-offs — 7.4 Refusal Parser Trade-offs

### Sections added
- ➕ Chapter 5 — Experimental Setup § 5.4.1 Infrastructure Validation and Fixes — 5.4.1 Infrastructure Validation and Fixes
- ➕ Chapter 6 — Results and Analysis § 6.1 Results Table — 6.1 Results Table
- ➕ Chapter 6 — Results and Analysis § 6.1 Results Table — 6.1.1 Scorer revision history
- ➕ Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — 6.4.1 Capability robustness check: the ARC-Challenge seco…
- ➕ Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — 6.6 Observations: Qwen 2B Pair
- ➕ Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — 6.6.1 Multi-seed decoding sensitivity (W1 robustnes…
- ➕ Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — 6.6.2 Per-category harmful-compliance profile
- ➕ Chapter 6 — Results and Analysis § 6.7 Preliminary Scale Observations: Baseline Comparison — 6.7 Preliminary Scale Obse…
- ➕ Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — 6.8 Observations: Qwen 4B Pair
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — 6.9 Research Question Synthesis
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — RQ1 — Does 4-bit NF4 quantization increase harmful…
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — RQ2 — Does 4-bit NF4 quantization increase over-re…
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — RQ3 — Does 4-bit NF4 quantization degrade general …
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — RQ4 — Are smaller models more sensitive to quantiz…
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — RQ5 — Are effects consistent across model families?
- ➕ Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — Discussion: What the full five-pair dataset tells …
- ➕ Chapter 6 — Results and Analysis § 6.10 Observations: Llama 3.2 3B Pair — 6.10 Observations: Llama 3.2 3B Pair
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11 Cross-Family Comparison (RQ5)
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.1 Baseline safety profiles differ substant…
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.2 Harmful compliance: one significant delt…
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.3 Over-refusal: no significant increase ac…
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.4 Capability: directionally consistent whe…
- ➕ Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — 6.11.5 Interpretation labels and the study's ce…
- ➕ Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — 6.12 Judge-Model…
- ➕ Chapter 6 — Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — 6.13 Cross-Family Ext…
- ➕ Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — 6.14 Mechanism Probe: Re…
- ➕ Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — 6…
- ➕ Chapter 7 — Discussion and Threats to Validity § 7.4 Scorer Choice and Remaining Validity Threat — 7.4 Scorer Choice an…
- ➕ Chapter 7 — Discussion and Threats to Validity § 7.6 Deployment Implications — 7.6 Deployment Implications
- ➕ Chapter 7 — Discussion and Threats to Validity § 7.7 Positioning Against Prior Work — 7.7 Positioning Against Prior Work
- ➕ Appendix H — Scorer Incident and Future-Session Handoff § H.1 What went wrong in v1 — H.1 What went wrong in v1
- ➕ Appendix H — Scorer Incident and Future-Session Handoff § H.2 How the correction was made — H.2 How the correction was …
- ➕ Appendix H — Scorer Incident and Future-Session Handoff § H.3 Current artifact contract — H.3 Current artifact contract
- ➕ Appendix H — Scorer Incident and Future-Session Handoff § H.4 Verification completed — H.4 Verification completed
- ➕ Appendix H — Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — H.5 Where the ne…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A — Final TC1 Configuration | 48 | 1 | 12 | 0 |
| Appendix D — Test Inventory | 1 | 1 | 1 | 0 |
| Appendix E — Repository Layout | 0 | 0 | 3 | 0 |
| Appendix G — Document Revision History | 1 | 1 | 0 | 0 |
| Appendix H — Scorer Incident and Future-Session Handoff | 9 | 0 | 0 | 0 |
| Chapter 1 — Introduction | 1 | 1 | 4 | 0 |
| Chapter 10 — Conclusion | 2 | 1 | 1 | 0 |
| Chapter 2 — Literature Review | 1 | 1 | 0 | 0 |
| Chapter 3 — Methodology | 7 | 4 | 6 | 0 |
| Chapter 4 — System Design and Implementation | 0 | 0 | 3 | 0 |
| Chapter 5 — Experimental Setup | 2 | 1 | 7 | 0 |
| Chapter 6 — Intended Results and Analysis Plan | 0 | 10 | 0 | 0 |
| Chapter 6 — Results and Analysis | 76 | 0 | 3 | 0 |
| Chapter 7 — Discussion and Threats to Validity | 5 | 2 | 1 | 0 |
| Chapter 8 — Limitations | 4 | 3 | 0 | 0 |
| Chapter 9 — Future Work | 6 | 3 | 0 | 0 |
| References | 6 | 4 | 0 | 0 |

### Removed paragraphs (33 body; 3 front matter)

<details><summary>show</summary>

- `662cad37` (28w) Chapter 1 — Introduction § 1.5 Contributions — “A verification suite of 122 automated tests across twelve test files, covering schema val…”
- `5faa4a45` (108w) Chapter 2 — Literature Review § 2.4 Quantization and Behavioural Effects — “A body of recent work has argued that quantization is not behaviourally neutral. Studies …”
- `e040829b` (19w) Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping — “Three benchmarks are used. Each measures one of the three dimensions required to disambig…”
- `965d5fb9` (65w) Chapter 3 — Methodology § 3.5 Scoring: Deterministic Refusal Detection — “All refusal scoring is performed by deterministic regex-based pattern matching rather tha…”
- `747b8bcd` (30w) Chapter 3 — Methodology § 3.8 Interpretation Framework — “Combining harmful-compliance, over-refusal, and capability deltas, each pair receives one…”
- `df54632d` (100w) Chapter 3 — Methodology § 3.8 Interpretation Framework — “Label \| Condition \| Interpretation \|\| alignment_degradation \| ΔASR > 0 AND ΔMMLU ≈ 0…”
- `99693f67` (109w) Chapter 5 — Experimental Setup § 5.6 Run Plan — “With MaxJobsPU=2 enforced by the cluster policy, the six jobs are processed in three sequ…”
- `84d6d4a9` (34w) Chapter 6 — Intended Results and Analysis Plan — “Experimental runs are pending submission on TC1 at the time of this interim report. This …”
- `20f5fb46` (37w) Chapter 6 — Intended Results and Analysis Plan § 6.1 Reporting Schema — “For each of the three pairs, the final report will present a single results table summari…”
- `6d622539` (132w) Chapter 6 — Intended Results and Analysis Plan § 6.1 Reporting Schema — “Pair \| Metric \| Baseline (95% CI) \| 4-bit (95% CI) \| ΔAbsolute \| ΔRelative \|\| qwen…”
- `1778cbf0` (15w) Chapter 6 — Intended Results and Analysis Plan § 6.1 Reporting Schema — “Table 6.1 Per-pair reporting schema. The (pending) cells will be populated by the TC1 run…”
- `f19899cc` (94w) Chapter 6 — Intended Results and Analysis Plan § 6.2 Within-Family Scale Analysis (RQ4) — “RQ4 asks whether the magnitude of the quantization effect differs between two-billion and…”
- `1d9326c9` (19w) Chapter 6 — Intended Results and Analysis Plan § 6.4 Capability Anchoring — “Alignment shift: capability is preserved, but refusal calibration moves (either HarmBench…”
- `0975e4da` (26w) Chapter 6 — Intended Results and Analysis Plan § 6.4 Capability Anchoring — “Capability-driven safety artifact: MMLU drops noticeably, and the safety metrics also mov…”
- `f255ac68` (9w) Chapter 6 — Intended Results and Analysis Plan § 6.4 Capability Anchoring — “Broad degradation: capability and both safety metrics worsen simultaneously.”
- `f0d6e9c0` (15w) Chapter 6 — Intended Results and Analysis Plan § 6.4 Capability Anchoring — “This decomposition is the principal mechanism through which the study addresses the centr…”
- `c287d938` (122w) Chapter 6 — Intended Results and Analysis Plan § 6.5 Statistical Caveats — “Three statistical limitations should be borne in mind when interpreting the eventual resu…”
- `c528cd48` (116w) Chapter 7 — Discussion and Threats to Validity § 7.2 External Validity — “External validity is bounded by three explicit design choices. First, findings are bounde…”
- `e9cb27a4` (111w) Chapter 7 — Discussion and Threats to Validity § 7.4 Refusal Parser Trade-offs — “The deterministic regex-based refusal parser was selected for reproducibility and transpa…”
- `5f69abd8` (26w) Chapter 8 — Limitations — “Single quantization method. Only BitsAndBytes NF4 four-bit quantization is evaluated. GPT…”
- `4c5f14a0` (27w) Chapter 8 — Limitations — “Partial capability proxy. The six-subject MMLU subset is a tractable but partial measure …”
- `f1f4a673` (22w) Chapter 8 — Limitations — “Greedy decoding only. Temperature 0.0 is used throughout, eliminating within-condition st…”
- `f19857ee` (37w) Chapter 9 — Future Work — “Stochastic-decoding sensitivity arm. Re-run a representative subset of pairs at temperatu…”
- `3b5519fd` (30w) Chapter 9 — Future Work — “Expanded scale axis. Once primary results are in hand, extend the Qwen sweep to include a…”
- `13eae18b` (28w) Chapter 9 — Future Work — “Judge-LLM sensitivity check. Re-score a stratified sample of responses with an independen…”
- `b749812c` (99w) Chapter 10 — Conclusion — “The framework is complete, validated, and ready for cluster execution. Three pairs (Qwen …”
- `452b45c7` (14w) References — “Liu, S., et al. Behavioural Effects of Quantization on Instruction-Tuned Models. arXiv:24…”
- `6068ce5d` (16w) References — “Anonymous Authors. A Comprehensive Evaluation of Quantization Strategies for Large Langua…”
- `86d29745` (12w) References — “Anonymous Authors. Safety and Trustworthiness Effects of Post-Training Quantization. arXi…”
- `51707387` (10w) References — “Anonymous Authors. Safety-Preserving Quantization for Aligned LLMs. arXiv:2511.07842, 202…”
- `1857d011` (8w) Appendix A — Final TC1 Configuration — “The full configs/tc1.yaml is reproduced below for reference.”
- `6ed73988` (152w) Appendix D — Test Inventory — “Test file \| Test count \| Coverage area \|\| test_datasets.py \| 27 \| Dataset loaders (…”
- `02b71e2b` (279w) Appendix G — Document Revision History — “When (UTC+8) \| Version \| Change to the report \|\| 2026-05-27 00:34 \| FYP_Report_2026-…”

</details>

### Added paragraphs (169)

<details><summary>show</summary>

- `e1cec630` (53w) Chapter 1 — Introduction § 1.5 Contributions — “A verification suite of 306 automated tests across twenty-three test files, covering sche…”
- `e7483c1e` (204w) Chapter 2 — Literature Review § 2.4 Quantization and Behavioural Effects — “A body of recent work has argued that quantization is not behaviourally neutral. Studies …”
- `b02e342e` (161w) Chapter 3 — Methodology § 3.2 Model Selection — “The Qwen3 checkpoints were selected after an initial environment validation on the TC1 cl…”
- `1738b27e` (55w) Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping — “Four benchmarks are used across the three dimensions required to disambiguate alignment s…”
- `5fcdc04b` (193w) Chapter 3 — Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier — “Two scorers are used, with a clear division of roles. A deterministic regex-based refusal…”
- `a797c9ad` (168w) Chapter 3 — Methodology § 3.7 Metrics and Pairwise Analysis — “Paired significance test for HarmBench (McNemar). Because both pair members see the ident…”
- `7716c74e` (116w) Chapter 3 — Methodology § 3.8 Interpretation Framework — “Combining harmful-compliance, over-refusal, and capability deltas, each pair receives one…”
- `0b7cff8b` (193w) Chapter 3 — Methodology § 3.8 Interpretation Framework — “Label \| Condition \| Interpretation \|\| alignment_degradation \| ΔASR ≥ +0.02 AND ΔMMLU…”
- `68f29964` (176w) Chapter 3 — Methodology § 3.8 Interpretation Framework — “Two-layer reporting: label plus evidence status. The interpretation label is a statement …”
- `d0a051fa` (245w) Chapter 5 — Experimental Setup § 5.4.1 Infrastructure Validation and Fixes — “Before the full matrix was submitted, a dedicated CUDA verification job and a five-sample…”
- `d7669cce` (120w) Chapter 5 — Experimental Setup § 5.6 Run Plan — “The first production matrix submission was made on 2026-05-27 with the Qwen 2B-class pair…”
- `d2206485` (193w) Chapter 6 — Results and Analysis — “All six original experimental runs completed successfully on the NTU TC1 GPU cluster on 2…”
- `1b8b5665` (122w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Table 6.1 records the per-pair benchmark results for all five model pairs together with b…”
- `4e2bf611` (329w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Pair \| Metric (scorer) \| Baseline \| 4-bit \| Δ (95% CI) \| Sig? \|\| qwen_2b \| HarmBe…”
- `ea9c4524` (53w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Table 6.1 Complete per-pair results with paired bootstrap 95% CIs on the deltas. "Sig?" =…”
- `47034da8` (158w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Under the primary (judge) scorer, the statistically significant deltas are: Qwen 1.7B ΔAS…”
- `9ab63722` (246w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Scope under the judge-primary framing (D16). This section documents the refusal regex's o…”
- `8b80edf8` (165w) Chapter 6 — Results and Analysis § 6.1 Results Table — “Impact of the rescore on the v2 numbers. Within the regex scorer, three pair-level deltas…”
- `491e0516` (51w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “RQ4 asks whether the magnitude of the quantization effect differs between two-billion and…”
- `9293032b` (113w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Metric \| qwen_2b \| qwen_4b \| llama_3.2_3b \| mistral_7b \| phi4_mini \|\| HarmBench ΔA…”
- `9ff2c0cf` (127w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Table 6.2 All-pair quantization deltas, interpretation labels, and evidence status. Δ = 4…”
- `5c96a575` (127w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization …”
- `7b4cdd5a` (159w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “The full-precision baselines reveal that within the Qwen family the larger model is both …”
- `baac29c2` (174w) Chapter 6 — Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4) — “The most important comparison is on the post-quantization side, read under the primary (j…”
- `6e2d05b4` (169w) Chapter 6 — Results and Analysis § 6.3 Cross-Family Replication (RQ5) — “The headline finding for RQ5 is that, under the official classifier, NF4 quantization nev…”
- `f7ac0bf7` (29w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Alignment degradation: MMLU is preserved, but HarmBench ASR rises beyond threshold — the …”
- `db003f5a` (34w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Alignment improvement: MMLU is preserved, but HarmBench ASR falls beyond threshold — quan…”
- `573c943c` (28w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Capability-collapse masquerading as safety: MMLU drops noticeably, and HarmBench ASR also…”
- `69dcde75` (24w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Broad degradation: fallback — typically signals over-refusal moving outside tolerance, or…”
- `84ab02e0` (319w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “The three pairs in this study illustrate why the capability anchor is necessary — even th…”
- `164a0e63` (325w) Chapter 6 — Results and Analysis § 6.4 Capability Anchoring — “Because the capability anchor rests on a single MMLU subset, a structurally different sec…”
- `d1422094` (597w) Chapter 6 — Results and Analysis § 6.5 Statistical Caveats — “Five statistical limitations should be borne in mind when interpreting the results. First…”
- `7b61409d` (103w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “Under the official HarmBench classifier, the Qwen 1.7B pair is the strongest result in th…”
- `95486eb7` (108w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “The ΔASR finding is corroborated by a second, independent significance test. Because Harm…”
- `5ceea867` (106w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “Note the sharp contrast with the secondary v2 proxy, which reports a much higher absolute…”
- `7893d472` (68w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “The XSTest over-refusal rate declined modestly (0.052 → 0.028, ΔOR = −0.024, CI [−0.052, …”
- `a5c1747c` (100w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “Subject-level MMLU breakdown supports the capability-decline reading. Accuracy declined i…”
- `46c5a1f9` (80w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “The Qwen 1.7B pair is the study's most consequential deployment finding: a 4-bit-quantize…”
- `4ba3d50c` (124w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “Because the primary study uses greedy decoding (T = 0.0), its bootstrap and McNemar inter…”
- `093d8989` (191w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “The judge ΔASR is positive in four of the five seeds (per-seed: −0.010, +0.050, +0.055, +…”
- `5876662b` (206w) Chapter 6 — Results and Analysis § 6.6 Observations: Qwen 2B Pair — “To check whether the Qwen 1.7B safety regression is concentrated in a single harm type or…”
- `9653a248` (136w) Chapter 6 — Results and Analysis § 6.7 Preliminary Scale Observations: Baseline Comparison — “The Qwen3-1.7B and Qwen3-4B full-precision baselines reveal a capability-and-safety coupl…”
- `cccfbbe5` (133w) Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — “Under the official classifier, the Qwen 4B pair receives the alignment_degradation label …”
- `b74926be` (102w) Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — “This is a notable demotion from the secondary v2 proxy, under which Qwen 4B had appeared …”
- `b6ffdaff` (137w) Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — “Mechanism and over-refusal. Qwen 4B at full precision is strongly refusal-calibrated on g…”
- `cbd3c1f4` (105w) Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — “Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under t…”
- `2fc569a0` (16w) Chapter 6 — Results and Analysis § 6.8 Observations: Qwen 4B Pair — “The Llama 3.2 3B pair results (§6.10–§6.11) allow assessment of whether any safety worsen…”
- `0be0b53e` (165w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “With all ten models evaluated and HarmBench ASR scored by the official classifier, it is …”
- `3de05d07` (143w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “It does not reduce it in any pair, and the one statistically significant move is an incre…”
- `83409bc8` (166w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “Across the five pairs, four show no significant change in over-refusal and the fifth move…”
- `913468b9` (204w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “The answer depends strongly on model scale and family. The Qwen3-1.7B model loses 8.7 per…”
- `1f75103c` (222w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “Yes — clearly, and on both axes. On capability, the MMLU delta ratio between Qwen 1.7B an…”
- `cac9ad57` (204w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “Partially. On capability, both families lose MMLU accuracy under NF4 compression where th…”
- `5f084c4f` (164w) Chapter 6 — Results and Analysis § 6.9 Research Question Synthesis — “Three implications extend beyond the specific numbers. First, surface-level refusal metri…”
- `5cbc2794` (47w) Chapter 6 — Results and Analysis § 6.10 Observations: Llama 3.2 3B Pair — “The Llama 3.2 3B pair (SLURM jobs 60980 and 60981) completed on 2026-05-27. Job 60980 (ba…”
- `b9dd66e6` (139w) Chapter 6 — Results and Analysis § 6.10 Observations: Llama 3.2 3B Pair — “The baseline profile of Llama 3.2 3B-Instruct is the most refusal-calibrated of the three…”
- `071e6dbe` (156w) Chapter 6 — Results and Analysis § 6.10 Observations: Llama 3.2 3B Pair — “Under NF4 4-bit quantization, the Llama pair is classified as broad_degradation. HarmBenc…”
- `530362e7` (121w) Chapter 6 — Results and Analysis § 6.10 Observations: Llama 3.2 3B Pair — “Why broad_degradation? With ΔASR within noise and a significant ΔMMLU, the rule falls thr…”
- `379e1f03` (56w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “The cross-family comparison places the Llama 3.2 3B pair alongside the two Qwen pairs to …”
- `5ba258be` (154w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Before interpreting quantization deltas, the baseline profile differences are themselves …”
- `275f001c` (145w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier, only one of the three ΔASR values is statistically signifi…”
- `dd11251b` (137w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “The XSTest over-refusal rate is the most consistent cross-family signal: four of the five…”
- `373d9789` (130w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Two of three pairs lose MMLU accuracy significantly: Qwen 1.7B (−8.7 pp, CI [−0.137, −0.0…”
- `f33b16e9` (222w) Chapter 6 — Results and Analysis § 6.11 Cross-Family Comparison (RQ5) — “Under the official classifier, the three interpretation labels are broad_degradation (Qwe…”
- `70ce1138` (138w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “HarmBench Attack Success Rate is scored by the official HarmBench classifier, cais/HarmBe…”
- `787e5435` (184w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Methodology and setup. The judge classified every saved (behavior, generation) pair for a…”
- `ea6e6cbf` (112w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Model \| v2 non-refusal \| judge ASR \| agreement \| Cohen's κ \|\| qwen_2b_base \| 0.600…”
- `a0335878` (38w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Table 6.3 Judge vs v2 per-model agreement on HarmBench (n = 200 each). "v2 non-refusal" i…”
- `f73e1314` (154w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 1 — the v2 regex systematically over-counts ASR, unevenly by family. The judge ASR…”
- `78a3326a` (153w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 2 — the choice of scorer changes the conclusion. Under the judge, the per-pair Har…”
- `b9e34ea3` (150w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 3 — the relocation is robust to the choice of significance test. The judge ΔASR si…”
- `4f0e4ab4` (283w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Result 4 — the finding is robust to the choice of judge (second-judge cross-check). To ru…”
- `da1596f7` (159w) Chapter 6 — Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer) — “Threats and scope. The judge is itself a model and can err; it is adopted because it is t…”
- `e499d34f` (128w) Chapter 6 — Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “To test whether the findings generalise beyond the original two families, two further mat…”
- `ea70959e` (224w) Chapter 6 — Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Mistral-7B is the clearest demonstration of the judge-over-proxy finding (§6.12, D16) in …”
- `db52d8b7` (149w) Chapter 6 — Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Phi-4-mini falls in the opposite (Llama-like) camp, where the proxy and the judge largely…”
- `31b9a354` (206w) Chapter 6 — Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5) — “Across all four families the central pattern holds and strengthens. Under the official cl…”
- `6d7b6047` (145w) Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — “The behavioural results (§6.3, §6.12, §6.13) establish that NF4 rarely shifts harmful com…”
- `1cff97c2` (113w) Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — “Validation gate. Before any quantization claim the margin's sign was checked against the …”
- `54553288` (263w) Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — “Central result — boundary instability, not targeted erosion. A thin baseline margin does …”
- `38d600e4` (152w) Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — “Direction of the margin shift is model-specific and modest. The paired shift Δm = m(fp16)…”
- `85c42f68` (246w) Chapter 6 — Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why") — “Capability versus alignment — the tie to the central question. If NF4 simply raised next-…”
- `067e34f5` (192w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “The main study evaluates one quantization method (NF4) at one bit-width (four-bit). To te…”
- `88c0f6ae` (130w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Capability: a clean cliff at four-bit. The capability answer is unambiguous. No INT8 delt…”
- `843b01cf` (220w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Safety: not a single four-bit cliff, but two effects at different precisions. The safety …”
- `32451949` (171w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Honest bounds on the INT8 safety finding. Two cautions are stated plainly so the effect i…”
- `024ef32d` (161w) Chapter 6 — Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4) — “Scorer note (the proxy at INT8). The secondary v2 refusal regex continues to over-count h…”
- `53ade5b4` (43w) Chapter 7 — Discussion and Threats to Validity — “The Qwen family results (§6.9) provide initial empirical grounding for the threats discus…”
- `4ae4e15b` (151w) Chapter 7 — Discussion and Threats to Validity § 7.2 External Validity — “External validity is bounded by three explicit design choices. First, findings are bounde…”
- `a23a61d1` (291w) Chapter 7 — Discussion and Threats to Validity § 7.4 Scorer Choice and Remaining Validity Threat — “HarmBench ASR was initially scored with a deterministic regex parser, chosen for reproduc…”
- `fa9b48a0` (214w) Chapter 7 — Discussion and Threats to Validity § 7.6 Deployment Implications — “The practical takeaway is asymmetric across the scale range studied. The most safety-rele…”
- `2a317969` (178w) Chapter 7 — Discussion and Threats to Validity § 7.7 Positioning Against Prior Work — “These findings sit consistently within, and extend, the emerging quantization-and-safety …”
- `ac0be7c7` (64w) Chapter 8 — Limitations — “Quantization-method coverage. Two BitsAndBytes methods are evaluated: NF4 four-bit (the m…”
- `546b662f` (108w) Chapter 8 — Limitations — “Partial capability proxy (now two-benchmark). The six-subject MMLU subset is a tractable …”
- `13e2bb9c` (130w) Chapter 8 — Limitations — “MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, …”
- `aa1778cc` (74w) Chapter 8 — Limitations — “Greedy decoding only (partially addressed). Temperature 0.0 is used throughout the primar…”
- `c0016b7e` (71w) Chapter 9 — Future Work — “Stochastic-decoding sensitivity arm (partially completed). The load-bearing Qwen 1.7B pai…”
- `66886260` (104w) Chapter 9 — Future Work — “Composite-capability interpretation rule (second benchmark now run). ARC-Challenge has be…”
- `a96a65de` (166w) Chapter 9 — Future Work — “Mechanism follow-ups (refusal-margin probe, §6.14). The first-token refusal-margin analys…”
- `36dbbee3` (233w) Chapter 9 — Future Work — “Cross-family and scale extension (completed 2026-06-15). Two further matched pairs were i…”
- `150e882c` (205w) Chapter 9 — Future Work — “INT8 precision point / quantization-method sweep (completed; reported in §6.15). To test …”
- `5fb51f8c` (112w) Chapter 9 — Future Work — “Fully-independent open-weight second judge. The primary HarmBench classifier has now been…”
- `59520a8c` (359w) Chapter 10 — Conclusion — “All six original runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 6097…”
- `f111a140` (247w) Chapter 10 — Conclusion — “Taken together, the results answer all five research questions. Under the official HarmBe…”
- `58a45f79` (25w) References — “Kharinaev, A., Moskvoretskii, V., Shvetsov, E., et al. Investigating the Impact of Quanti…”
- `2d662511` (25w) References — “Chen, K., Zhang, J., Hu, J., et al. Q-resafe: Assessing Safety Risks and Quantization-awa…”
- `fbf6955f` (17w) References — “Egashira, K., Vero, M., Staab, R., He, J., Vechev, M. Exploiting LLM Quantization. NeurIP…”
- `7e73d38f` (20w) References — “Belkhiter, Y., Zizzo, G., Maffeis, S. HarmLevelBench: Evaluating Harm-Level Compliance an…”
- `c68dc4a1` (28w) References — “Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. Refus…”
- `7b0cbe51` (22w) References — “Proskurina, I., Brun, L., Metzler, G., Velcin, J. When Quantization Affects Confidence of…”
- `c1e17696` (36w) Appendix A — Final TC1 Configuration — “The full configs/tc1.yaml is reproduced below for reference. The mistral_7b and phi4_mini…”
- `3187077d` (5w) Appendix A — Final TC1 Configuration — “benchmarks: [harmbench, xstest, mmlu, arc]”
- `0c12754d` (12w) Appendix A — Final TC1 Configuration — “# Cross-family extension (run on TC1 2026-06-15; results in Chapter 6, §6.13).”
- `917d82f6` (14w) Appendix A — Final TC1 Configuration — “# See Chapter 9. Phi-4-mini uses the eager attention backend (the V100 has no”
- `f68e63b6` (13w) Appendix A — Final TC1 Configuration — “# flash-attention kernels) and loads via native transformers Phi3; Mistral drops straight…”
- `1a00bd3d` (1w) Appendix A — Final TC1 Configuration — “mistral_7b_base:”
- `85fcf88a` (2w) Appendix A — Final TC1 Configuration — “family: mistral”
- `06a87e46` (2w) Appendix A — Final TC1 Configuration — “size_b: 7.2”
- `4d64e18f` (2w) Appendix A — Final TC1 Configuration — “quantized: false”
- `75ba51d6` (2w) Appendix A — Final TC1 Configuration — “pair_id: mistral_7b”
- `1704a6a7` (2w) Appendix A — Final TC1 Configuration — “model_id: mistralai/Mistral-7B-Instruct-v0.3”
- `714be8f0` (2w) Appendix A — Final TC1 Configuration — “trust_remote_code: false”
- `c783a2f6` (2w) Appendix A — Final TC1 Configuration — “dtype: auto”
- `3187077d` (5w) Appendix A — Final TC1 Configuration — “benchmarks: [harmbench, xstest, mmlu, arc]”
- `4b53372b` (1w) Appendix A — Final TC1 Configuration — “mistral_7b_4bit:”
- `85fcf88a` (2w) Appendix A — Final TC1 Configuration — “family: mistral”
- `06a87e46` (2w) Appendix A — Final TC1 Configuration — “size_b: 7.2”
- `0163695f` (2w) Appendix A — Final TC1 Configuration — “quantized: true”
- `75ba51d6` (2w) Appendix A — Final TC1 Configuration — “pair_id: mistral_7b”
- `1704a6a7` (2w) Appendix A — Final TC1 Configuration — “model_id: mistralai/Mistral-7B-Instruct-v0.3”
- `714be8f0` (2w) Appendix A — Final TC1 Configuration — “trust_remote_code: false”
- `c783a2f6` (2w) Appendix A — Final TC1 Configuration — “dtype: auto”
- `3187077d` (5w) Appendix A — Final TC1 Configuration — “benchmarks: [harmbench, xstest, mmlu, arc]”
- `74f3c13b` (1w) Appendix A — Final TC1 Configuration — “phi4_mini_base:”
- `ea8ba318` (2w) Appendix A — Final TC1 Configuration — “family: phi”
- `c82f17e7` (2w) Appendix A — Final TC1 Configuration — “size_b: 3.8”
- `4d64e18f` (2w) Appendix A — Final TC1 Configuration — “quantized: false”
- `064c4a22` (2w) Appendix A — Final TC1 Configuration — “pair_id: phi4_mini”
- `9a708c81` (2w) Appendix A — Final TC1 Configuration — “model_id: microsoft/Phi-4-mini-instruct”
- `714be8f0` (2w) Appendix A — Final TC1 Configuration — “trust_remote_code: false”
- `b16b7731` (2w) Appendix A — Final TC1 Configuration — “attn_implementation: eager”
- `c783a2f6` (2w) Appendix A — Final TC1 Configuration — “dtype: auto”
- `3187077d` (5w) Appendix A — Final TC1 Configuration — “benchmarks: [harmbench, xstest, mmlu, arc]”
- `6a0bc816` (1w) Appendix A — Final TC1 Configuration — “phi4_mini_4bit:”
- `ea8ba318` (2w) Appendix A — Final TC1 Configuration — “family: phi”
- `c82f17e7` (2w) Appendix A — Final TC1 Configuration — “size_b: 3.8”
- `0163695f` (2w) Appendix A — Final TC1 Configuration — “quantized: true”
- `064c4a22` (2w) Appendix A — Final TC1 Configuration — “pair_id: phi4_mini”
- `9a708c81` (2w) Appendix A — Final TC1 Configuration — “model_id: microsoft/Phi-4-mini-instruct”
- `714be8f0` (2w) Appendix A — Final TC1 Configuration — “trust_remote_code: false”
- `b16b7731` (2w) Appendix A — Final TC1 Configuration — “attn_implementation: eager”
- `c783a2f6` (2w) Appendix A — Final TC1 Configuration — “dtype: auto”
- `b5dbe97f` (1w) Appendix A — Final TC1 Configuration — “arc:”
- `08c8d7e0` (2w) Appendix A — Final TC1 Configuration — “dataset_name: allenai/ai2_arc”
- `f3b64dd7` (2w) Appendix A — Final TC1 Configuration — “config_name: ARC-Challenge”
- `a956002f` (2w) Appendix A — Final TC1 Configuration — “split: test”
- `619c0f0e` (2w) Appendix A — Final TC1 Configuration — “max_samples: 1200”
- `7ec9eef2` (2w) Appendix A — Final TC1 Configuration — “batch_size: 4”
- `e024b0c8` (572w) Appendix D — Test Inventory — “Test file \| Test count \| Coverage area \|\| test_datasets.py \| 27 \| Dataset loaders (…”
- `d28ca9a4` (2150w) Appendix G — Document Revision History — “When (UTC+8) \| Version \| Change to the report \|\| 2026-06-15 18:00 \| FYP_Report_2026-…”
- `3b5885bf` (54w) Appendix H — Scorer Incident and Future-Session Handoff — “This appendix is included so that a future agent or fresh chat session can continue the w…”
- `487c91b4` (98w) Appendix H — Scorer Incident and Future-Session Handoff § H.1 What went wrong in v1 — “The original HarmBench and XSTest scorer used a narrow deterministic refusal parser with …”
- `de5ce860` (141w) Appendix H — Scorer Incident and Future-Session Handoff § H.1 What went wrong in v1 — “The consequence was material, and the story has two stages. The v1→v2 regex correction fi…”
- `caa39e87` (65w) Appendix H — Scorer Incident and Future-Session Handoff § H.2 How the correction was made — “The v2 correction expanded `ethical_benchmark/benchmarks/utils.py` with broader refusal-p…”
- `58c73d5d` (81w) Appendix H — Scorer Incident and Future-Session Handoff § H.2 How the correction was made — “A first implementation rescored `raw.jsonl` and `summary.json` in place. That approach wa…”
- `a50eb892` (167w) Appendix H — Scorer Incident and Future-Session Handoff § H.3 Current artifact contract — “Artifact \| Status \| Rule for future work \|\| raw.jsonl \| TC1-original saved generatio…”
- `d839f5b7` (134w) Appendix H — Scorer Incident and Future-Session Handoff § H.4 Verification completed — “The workflow was verified in five ways. First, the full test suite passes with 306 tests,…”
- `a02ee197` (82w) Appendix H — Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — “The HarmBench judge validation is complete (job 61047) and the official classifier is the…”
- `2b4ecea6` (121w) Appendix H — Scorer Incident and Future-Session Handoff § H.5 Where the next session should continue — “The most valuable optional extension is a second independent judge — for example LlamaGua…”

</details>

### Edited paragraphs (43; change window shown)

<details><summary>show</summary>

- (front matter): “Document date: 2[−4][+7] May 2026”
- (front matter): “…quantization itself as the sole experimental variable. [−Thre][+Fiv][+cross four familie…”
- Chapter 1 — Introduction § 1.4 Scope and Boundaries: “Quantization method. The [+main ][−ev][+pp][−uat][+i][− a single quantization path:][− ap…”
- Chapter 1 — Introduction § 1.5 Contributions: “…iance, over-refusal, and capability deltas into one of [−four][+six][+, alignment improv…”
- Chapter 1 — Introduction § 1.5 Contributions: “…ix jobs that load each model only once and execute all [−th][+configu][−e][+d] benchmark…”
- Chapter 1 — Introduction § 1.6 Report Structure: “… setup on TC1 and the run plan. Chapter 6 presents the [−intended results and analysis p…”
- Chapter 3 — Methodology § 3.2 Model Selection: “Three model pairs are evaluated. The Qwen pairs at [−two][+1.7] billion and four billion…”
- Chapter 3 — Methodology § 3.2 Model Selection: “…\| Size \|\| qwen_2b \| Primary (BF/FP16) \| qwen_2b_base \| [−t][+Qw][−chwithsergiu][+n…”
- Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping: “…efusal rate \| 250 benign prompts \|\| General capability [+(anchor) ][+ \\|\\| General …”
- Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping: “…budgets. Sampling is deterministic and seed-controlled.[+ Capability is a two-benchmark …”
- Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping: “…rompts that should not be refused. MMLU is used as the [+primary ][+, with ARC-Challenge…”
- Chapter 3 — Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier: “…tential to miss nuanced or euphemistic refusals — is ex[−p][+act][+y why the off][+al Ha…”
- Chapter 4 — System Design and Implementation § 4.2 Configuration Schema: “…del_id, trust_remote_code, dtype, and a benchmarks list[+, plus an optional attn_impleme…”
- Chapter 4 — System Design and Implementation § 4.8 SLURM Job Generation: “…, one sbatch is generated per (model, benchmark) pair ([−eigh][−e][+w][+ty-four][−th][+f…”
- Chapter 4 — System Design and Implementation § 4.10 Testing: “The repository ships with a verification suite of [−122][+306][−lv][+nty-thre][+bootstrap…”
- Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment: “… GPU cards, giving twenty-one GPUs across the partition[−;][+.][−user-level concurrency …”
- Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment: “…alltime per job (default QoS) \| 06:00:00 \|\| MaxJobsPU ([−conc][+s][−rr][+bmitt][−nt][…”
- Chapter 5 — Experimental Setup § 5.2 Software Environment: “…ents.txt) \|\| pytest \| (installed via requirements.txt; [−122][+306] tests passing)”
- Chapter 5 — Experimental Setup § 5.4 Hugging Face Access and Gating: “The Qwen pairs draw from open-access [+official Qwen3 ][−u][+(Qwen/Qwen3-1.7B a][+ Qw][−r…”
- Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache: “…cache step retrieves HarmBench, the six MMLU subjects, [+ARC-Challenge (allenai/ai2_arc,…”
- Chapter 5 — Experimental Setup § 5.6 Run Plan: “…files — one per model alias. Each script runs the full [−th][+configu][−e-][+d ][+(The i…”
- Chapter 5 — Experimental Setup § 5.6 Run Plan: “… a single short smoke sbatch is submitted (five prompts[−, fifteen-minute walltime][−ly]…”
- Chapter 6 — Results and Analysis § 6.3 Cross-Family Replication (RQ5): “… the Llama deltas have the same sign as the Qwen deltas[− across all three benchmarks][−…”
- Chapter 6 — Results and Analysis § 6.4 Capability Anchoring: “…ate when interpreting its HarmBench and XSTest deltas. [−Four][+Six][+, organised symmet…”
- Chapter 6 — Results and Analysis § 6.4 Capability Anchoring: “…small in magnitude; quantization is effectively neutral[+ across safety and capability d…”
- Chapter 7 — Discussion and Threats to Validity § 7.1 Internal Validity: “…nt twice, or measurement noise. Deterministic decoding [+(temperature 0.0) ][−,][+. Scor…”
- Chapter 10 — Conclusion: “…arking framework comprising the matched-pair pipeline, [−th][+fou][−ee][+ls and paired b…”
- Appendix A — Final TC1 Configuration: “size_b: [−2][+1][−0][+7]”
- Appendix A — Final TC1 Configuration: “model_id: [−t][+Qw][−chwithsergiu][+n][+-1][−5-text-2][+7]B”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix A — Final TC1 Configuration: “size_b: [−2][+1][−0][+7]”
- Appendix A — Final TC1 Configuration: “model_id: [−t][+Qw][−chwithsergiu][+n][+-1][−5-text-2][+7]B”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix A — Final TC1 Configuration: “model_id: [−t][+Qw][−chwithsergiu][+n][−.5-text]-4B”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix A — Final TC1 Configuration: “model_id: [−t][+Qw][−chwithsergiu][+n][−.5-text]-4B”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix A — Final TC1 Configuration: “benchmarks: [harmbench, xstest, mmlu[+, arc]]”
- Appendix D — Test Inventory: “The verification suite comprises [−122][+306][−lv][+nty-thre]e test files. The distrib…”
- Appendix E — Repository Layout: “├── tests/ ([−122][+306][−1][+3] files)”
- Appendix E — Repository Layout: “├── results/ (raw.jsonl, summary.json, [+v2 score sidecars, ]analysis outputs)”
- Appendix E — Repository Layout: “└── FYP_Report_2026-0[−5][+6][−23][+14].docx (this document)”

</details>

### Renamed headings (2)
- “3.5 Scoring: Deterministic Refusal Detection” → “3.5 Scoring: Refusal Detection and the HarmBench Classifier”
- “Chapter 6 — Intended Results and Analysis Plan” → “Chapter 6 — Results and Analysis”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
