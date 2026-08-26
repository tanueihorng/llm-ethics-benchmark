## Humanized alternate — full report: Base — report v5 → Humanized report alternate

*From:* `docs/FYP_Report_2026-07-01_v5.docx`  
*To:* `docs/FYP_Report_2026-07-01_humanized.docx`  
*Decisions:* D41  

**Paragraph/body blocks:** +0 added · −8 removed · ~145 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~3 · front matter blocks 52→43

*Front matter (cover/abstract/TOC/lists):* +0 −7 ~4 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ A Matched-Pair, Judge-Validated Study of Safety, Capability Trade-offs in Quantized Compact Languag…
  - ❌ Comprehensive project report, revised through July 2026. Scope: five matched pairs across four fami…
  - ❌ I hereby declare that this report is my own work and, to the best of my knowledge and belief, it co…
  - ❌ All experimental results reported here were produced by the open-source benchmarking framework desc…
  - ❌ _______________________________
  - ❌ Tan Uei Horng (UTAN001)
  - ❌ College of Computing and Data Science, Nanyang Technological University
  - ❌ Date: 2 July 2026
  - ❌ In the interest of transparency and in accordance with academic-integrity guidance and emerging ope…
  - ❌ The design loads the baseline and quantized members of each pair from identical weights, applying q…
  - ❌ HarmBench attack-success rate is reported at HarmBench's own 512-token reference budget as the prim…

### Chapters / parts REMOVED
- ❌ Declaration of Originality
- ❌ Declaration on the Use of Generative AI Tools

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A: Final TC1 Configuration | 0 | 0 | 1 | 0 |
| Appendix C: Pydantic Configuration Schema Summary | 0 | 0 | 3 | 0 |
| Appendix G: Document Revision History | 0 | 0 | 1 | 0 |
| Appendix H: Scoring-Methodology Correction Record | 0 | 0 | 2 | 0 |
| Chapter 10: Conclusion | 0 | 0 | 3 | 0 |
| Chapter 1: Introduction | 0 | 0 | 5 | 0 |
| Chapter 2: Literature Review | 0 | 0 | 4 | 0 |
| Chapter 3: Methodology | 0 | 1 | 9 | 0 |
| Chapter 4: System Design and Implementation | 0 | 0 | 2 | 0 |
| Chapter 5: Experimental Setup | 0 | 0 | 8 | 0 |
| Chapter 6: Results and Analysis | 0 | 0 | 56 | 0 |
| Chapter 7: Discussion and Threats to Validity | 0 | 0 | 7 | 0 |
| Chapter 8: Limitations | 0 | 0 | 6 | 0 |
| Chapter 9: Future Work | 0 | 0 | 5 | 0 |
| References | 0 | 0 | 29 | 0 |

### Removed paragraphs (1 body; 7 front matter)

<details><summary>show</summary>

- `457a1eaa` (123w) Chapter 3: Methodology § 3.8 Interpretation Framework — “Boundary convention. The thresholds are inclusive: a ΔASR of exactly +0.02 classifies as …”

</details>

### Edited paragraphs (145; change window shown)

<details><summary>show</summary>

- (front matter): “A Matched-Pair, Judge-Validated Study of Safety[−–][+, ]Capability Trade-offs in …”
- (front matter): “…anion documents present the same study at other lengths[− —][+,] a shorter Interim Repor…”
- (front matter): “…rs across four families (Qwen, Llama, Mistral, Phi; 1.7[−–][+ to ][− —][+,][− —][+,][−–]…”
- (front matter): “…y losses (MMLU, ARC) and a single over-refusal decrease[+,][−—][+and][+at over-refusal d…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…default. End users encountering these models therefore [−typic][+most a][+wa][+s] intera…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…hts, even one that preserves task accuracy on average, [−h][+c][−s the pote][−tial to] p…”
- Chapter 1: Introduction § 1.5 Contributions: “…nd, architecturally independent judge at Cohen's κ 0.68[−–][+ to ]0.95) changes both whi…”
- Chapter 1: Introduction § 1.5 Contributions: “…ange under four-bit NF4 after a multiplicity correction[− —][+,][− —][+,] and the effect…”
- Chapter 1: Introduction § 1.6 Report Structure: “…re on large-scale ethical benchmarking, the helpfulness[−–][+, ]harmlessness trade-off, …”
- Chapter 2: Literature Review § 2.1 Large-Scale Ethical Benchmarking of LLMs: “…ies such as Llama, Mistral, Qwen, and Falcon, and have [−mean][−g][+l][−lly shift][−d th…”
- Chapter 2: Literature Review § 2.2 The Helpfulness, Harmlessness Trade-off: “…Multitask Language Understanding benchmark, has become [+the de f][− s][+c][−andard][+o]…”
- Chapter 2: Literature Review § 2.4 Quantization and Behavioural Effects: “…ation-aware patching method to repair it. Closest to th[−e pre][+i][−ent][−report][+find…”
- Chapter 2: Literature Review § 2.5 Research Gaps Targeted by This Work: “…of compact (<4B) instruction-tuned models in the safety[−–][+, ]quantization context. Mo…”
- Chapter 3: Methodology § 3.1 Experimental Design: “…the same chat template, and the same release artifact. [−B][+An][+observed delta in benc…”
- Chapter 3: Methodology § 3.2 Model Selection: “… pairs at 1.7 billion and four billion parameters [17] [−s][+ar][−rve as][−serves a][+i]…”
- Chapter 3: Methodology § 3.3 Quantization Approach: “…ved runtime device is not cuda. After load, the loader [−v][+ch][+cks fou][+ s][−fie][+g…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “…efore measures harmful compliance under direct requests[− —][+,][− —][+,] not robustness…”
- Chapter 3: Methodology § 3.4 Benchmark Selection and Mapping: “… from the pool, so the per-subject counts are not equal[− —][+,] they range from 25 to 9…”
- Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier: “…get and an additional moving part, so it is controlled [+r][−cc][+ther than av][−r][+i][…”
- Chapter 3: Methodology § 3.5 Scoring: Refusal Detection and the HarmBench Classifier: “…ther than a released fine-tuned classifier, this study [−reta][+stead keep][− as the pri…”
- Chapter 3: Methodology § 3.7 Metrics and Pairwise Analysis: “…d test is applied to every other paired-binary endpoint[− —][+,][− —][+,][−t][+suc][−e][…”
- Chapter 3: Methodology § 3.9 Reproducibility Controls: “Reproducibility is treated as a [−fi][+co][−st-class][+e] engineering requirement.…”
- Chapter 4: System Design and Implementation § 4.6 Matrix Orchestration and Memory Management: “Memory management is [−cri][+ghtly ][+onstr][−l][+ined] on TC1, where each job i…”
- Chapter 4: System Design and Implementation § 4.11 CLI and Operational Interface: “…. \|\| cluster-submit \| Submit the generated sbatch files[− (local/dev convenience][−).…”
- Chapter 5: Experimental Setup § 5.1 Hardware and Cluster Environment: “…. The compute partition consists of seven nodes (TC1N01[−–][+, ]TC1N07), each equipped w…”
- Chapter 5: Experimental Setup § 5.1 Hardware and Cluster Environment: “…; confirm on first login \|\| Account window \| March 2026[− –][+,] November 2026 \|\| B…”
- Chapter 5: Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints: “…cause the GPU cards reside on the compute nodes (TC1N01[−–][+, ]TC1N07) and not on the h…”
- Chapter 5: Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache: “…OFFLINE=1 exported in the sbatch setup_commands block, [−en][−uring][+o] that any cache …”
- Chapter 5: Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache: “…s explicitly demonstrated in the TC1 user guide (page 7[−–][+ to ]9, where conda install…”
- Chapter 5: Experimental Setup § 5.6 Run Plan: “…el, the framework emits one sbatch file per model alias[− —][+,] six for the original th…”
- Chapter 5: Experimental Setup § 5.6 Run Plan: “… §6.13) added four more matrix scripts of the same form[− —][+,][− —][+,] submitted pair…”
- Chapter 5: Experimental Setup § 5.7 Reproducibility Notes: “…del repositories are pinned by Hugging Face commit hash[− —][+,][− —][+,][− —][+,][− —][…”
- Chapter 6: Results and Analysis: “…the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976[−–][+ to ][−–][+ to ]6.3 and ana…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…ples prompt indices and recomputes the delta directly. [−(One resolution detail for full…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…05); at the corrected over-refusal scoring no contrast [+s][+t][+ in the][−—][+margin wh…”
- Chapter 6: Results and Analysis § 6.1 Results Table: “…whose 95% CI excludes zero is Llama-3.2-3B's, at −0.040[− —][+,][+w][−l][+t][−nger ex][+…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “…exact p < 0.05); at the corrected over-refusal scoring [+Qwe][−o][+3-1.7B's][− borderlin…”
- Chapter 6: Results and Analysis § 6.2 Within-Family Scale Analysis (RQ4): “Table 6.2 [−reveal][+show][−st][+la][−ikin][+e][+;][−u][+o][−d][+ this b][+nchma][+k, sca…”
- Chapter 6: Results and Analysis § 6.3 Cross-Family Replication (RQ5): “…only significant delta (Phi, −0.048) is a decrease and [−the previously-borderline ][+'s…”
- Chapter 6: Results and Analysis § 6.4 Capability Anchoring: “…l five pairs lose on MMLU, and four of five lose on ARC[− —][+,][−primary ][−is apparent…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “… (with one honest nuance for the two cross-family pairs[+,][−di][−c][+pe][−os][+l][+ out…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “[−R][+This section reports the corrected view r][−, we report the corrected view][−wherea…”
- Chapter 6: Results and Analysis § 6.5 Statistical Caveats: “…alidation-informed parallel correction is also reported[+.][−—][+It is][−— ][+(][− —][+)…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…refusal-counting, is the appropriate scorer (see §6.12)[− —][+,] but here both scorers a…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…significant effect: the bootstrap CI [−0.048, 0.000] no[+w][−l][+t][−nger ex][+u][−lud][…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…e. The capability anchor is what makes this unambiguous[− —][+,] the significant ΔMMLU e…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…le fixed stochastic setting (temperature 0.7, top-p 0.8[− —][+,] Qwen3's published non-t…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “… ≈0.06, with the five pairs spanning 0.044–0.086, §6.5)[− —][+,][−c][+s][−mpleting cover…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…ncrease to attribute. Per-category counts are small (19[−–][+ to ]58 prompts), so the ra…”
- Chapter 6: Results and Analysis § 6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b): “…the reference budget. Per-category counts are small (19[−–][+ to ]58 prompts each), so t…”
- Chapter 6: Results and Analysis § 6.7 Preliminary Scale Observations: Baseline Comparison: “… v2 non-refusal proxy the baseline ASR gap looks larger[−,][−refl][+b][−ting][+ause][−'s…”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “…harm (judge baseline ASR = 0.115). XSTest over-refusal [−sh][−ft][−marg][+essent][−n][+ …”
- Chapter 6: Results and Analysis § 6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b): “The Llama 3.2 3B pair results (§6.10[−–][+, ]§6.11) allow assessment o…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…16), and one over-refusal decrease, Phi-4-mini (−0.048)[− —][+,][−n][+t][− longer ex][+u…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…, Llama 3.2 3B −0.040 (CI [−0.075, −0.010], significant[− —][+,] a decrease), Mistral-7B…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…→ 0.028, not significant: bootstrap CI [−0.048, 0.000] [−n][+t][− longer ex][+u][−lud][+…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…ce are Llama (−0.032, which also survives the Benjamini[−–][+, ][− —][+,][−–][+, ]Hochbe…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “… no within-family safety-sensitivity contrast to report[− —][+,][− —][+,] on ARC the wit…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “…significant delta, Phi's −0.048, is a decrease and the [−previously-borderline ]Qwen 1.7…”
- Chapter 6: Results and Analysis § 6.9 Research Question Synthesis: “… conclusion-level correction rather than a cosmetic one[− —][+,][+suppl][+e][+ two of][−…”
- Chapter 6: Results and Analysis § 6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b): “…the v2 regex and the judge agree most closely (κ ≈ 0.71[−–][+ to ]0.84; §6.12): Llama mo…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…+0.016), Mistral-7B (0.000), and Qwen3-1.7B (−0.024, no[−n-][+t ][−n][+t][− longer ex][+…”
- Chapter 6: Results and Analysis § 6.11 Cross-Family Comparison (RQ5): “…re broad_degradation (Qwen 1.7B), alignment_degradation[−–][+, ][− —][+,] no ASR contras…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…ich is family-dependent: low for Qwen/Mistral (κ ≈ 0.25[−–][+ to ][−–][+ to ]0.84). Sour…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…ndingly fair-to-moderate for Qwen and Mistral (κ ≈ 0.25[−–][+ to ][−–][+ to ]0.84). The …”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…ss (ΔMMLU −0.090), with the harm axis flat (ΔASR 0.000)[− —][+,] it is a capability-only…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “… 1.7B p = 1.000, Qwen 4B p = 0.096, and Llama p = 0.021[− —][+,][−–][+, ]regex relations…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…er than the regex-versus-classifier agreement (κ ≈ 0.25[−–][+ to ][−crucially ][−–][+ to…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “… a stratified subset of 200 saved HarmBench generations[− —][+,][− —][+,][−single ][− (t…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…), where an independent frontier judge agrees at κ 0.68[−–][+ to ][− —][+,][− —][+,][+e.…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…test whether that regex-scored finding is scorer-robust[− —][+,][− —][+,][− —][+:][− —][…”
- Chapter 6: Results and Analysis § 6.12 Judge-Model Validation of HarmBench ASR (the primary scorer): “…ng Result 5. All 200 items of a pre-registered gold set[− —][+,][− —][+,][+,][−—][+at][−…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…The proxy and the judge disagree in sign, and the judge[−–][+, ][++][−0][+4], n.s.), so …”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…I [−0.015, +0.055], not significant; McNemar p = 0.424)[−. That ΔASR][+,][−its exactly][…”
- Chapter 6: Results and Analysis § 6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5): “…ts severely for Qwen and Mistral (judge-vs-proxy κ 0.25[−–][+ to ][−–][+ to ]0.84)), so …”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…ated position and therefore generation-length-invariant[−:][+,][−the margin value][+o][+…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “…it discriminates refusal from compliance well (AUC 0.86[−–][+ to ][−–][+ to ]13 prompts)…”
- Chapter 6: Results and Analysis § 6.14 Mechanism Probe: Refusal-Margin Geometry (the "why"): “Central result[− (computed on the 128-token behavioural flips; the first-token margins ar…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…T8 and falls only at four-bit (a cliff, not a gradient)[− —][+,] the one robust, budget-…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “… (ΔASR = +0.040, McNemar p = 0.008/0.022, non-monotonic[− —][+,][−–][+ to ][− —][+,] eit…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…INT8 increase rested on a small absolute flip count (≈8[−–][+ to ][− —][+,][−–][+ to ]4%…”
- Chapter 6: Results and Analysis § 6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4): “…hereas the safety axis shows no robust precision effect[− —][+,] neither bit-width-grade…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “… primary run regenerates the entire study at 512 tokens[− —][+,] all five pairs, all fou…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…ling proxy puts the truncation rate at about 62 percent[− —][+,][−–][+ to ][−–][+ to ][−…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…l continues, and often fabricates a synthetic next turn[− —][+,][− —][+,] so a single re…”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…crease. The two judges agree per prompt at Cohen κ 0.68[−–][+ to ][−–][+ to ]0.95 range …”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…lly unchanged by the generation budget, which is why it[− —][+,][− —][+,] is the robust …”
- Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun: “…tive, a sub-MDE directional signal disclosed in §6.6.1)[− —][+,] so at 512 there is no d…”
- Chapter 7: Discussion and Threats to Validity § 7.1 Internal Validity: “…entical baseline weights, isolates quantization as the [+sole ][− against model-provenan…”
- Chapter 7: Discussion and Threats to Validity § 7.2 External Validity: “…ety axis is measured under direct harmful requests only[− —][+,][− —][+,] so the harmful…”
- Chapter 7: Discussion and Threats to Validity § 7.3 Construct Validity: “…y stated in Chapter 3); other operationalisations exist[− —][+,][− —][+,] and may yield …”
- Chapter 7: Discussion and Threats to Validity § 7.4 Scorer Choice and Remaining Validity Threat: “… D25). The two judges agree strongly per prompt (κ 0.60[−–][+ to ][−–][+ to ][− —][+,][+…”
- Chapter 7: Discussion and Threats to Validity § 7.5 Cross-Family Comparison Caveat: “…s but does not support causal claims about quantization[−–][+, ]family interactions.”
- Chapter 7: Discussion and Threats to Validity § 7.6 Deployment Implications: “…el tolerates quantization far better on the MMLU anchor[− —][+,][− —][+,][−–][+ to ]9).”
- Chapter 7: Discussion and Threats to Validity § 7.7 Positioning Against Prior Work: “…hat, like this one, isolates quantization at four bits [−—][+is][+, which][− —][+,] repo…”
- Chapter 8: Limitations: “…. HarmBench was evaluated in its standard configuration[− —][+,][− —][+,][−lever][+imp][…”
- Chapter 8: Limitations: “…e asymmetry even more strongly for this pair (52.3% of [+i][−he][+s][− member's][+ARC's …”
- Chapter 8: Limitations: “…utside a narrow seed range that straddles zero (§6.6.1)[− —][+,][−,][+rather tha][−ot] a…”
- Chapter 8: Limitations: “…the model sometimes generates synthetic follow-up turns[− —][+,][− —][+,] which skip_spe…”
- Chapter 8: Limitations: “…line HarmBench null should therefore be read as bounded[− —][+,][− —][+,] and pre-regist…”
- Chapter 8: Limitations: “…odel repositories is pinned by Hugging Face commit hash[− —][+,][− —][+,] to the exact s…”
- Chapter 9: Future Work: “…nch [14] and Egashira et al. [13]. This is the highest-[−lever][+imp][−ge][+ct] extensio…”
- Chapter 9: Future Work: “…unt and extending the arm beyond judge-scored HarmBench[− —][+,][− —][+,] would complete…”
- Chapter 9: Future Work: “… model in the study. Results are reported in Tables 6.1[−–][+ to ][−n.s.; a ][− on the t…”
- Chapter 9: Future Work: “…very INT8 contrast is non-significant under both judges[− —][+,][− —][+,] so the safety …”
- Chapter 9: Future Work: “…cross-checked against a second frontier judge (gpt-4o, [−pro][+sa][−pt][−d with a conden…”
- Chapter 10: Conclusion: “This Final Year Project investigates safety[−–][+, ]capability trade-offs in …”
- Chapter 10: Conclusion: “…the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976[−–][+ to ][− — the last regex-sc…”
- Chapter 10: Conclusion: “…cant and the fifth (Phi-4-mini) significantly decreases[− under the primary regex scorer…”
- References: “Y. Huang, L. Sun, et al., [−“][+"][−”][+"] in Proc. International C…”
- References: “B. Wang et al., [−“][+"][−”][+"] in Proc. NeurIPS Dataset…”
- References: “Z. Zhang et al., [−“][+"][−”][+"] in Proc. Association for…”
- References: “M. Mazeika et al., [−“][+"][−”][+"] in Proc. International C…”
- References: “…irk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, [−“][+"][−”][+"] in Proc. NAACL, …”
- References: “…asart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, [−“][+"][−”][+"] in Proc. Interna…”
- References: “…, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, [−“][+"][−”][+"] arXiv:1803.05457…”
- References: “…Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, [−“][+"][−”][+"] in Proc. NeurIPS…”
- References: “T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, [−“][+"][−”][+"] in Proc. NeurIPS,…”
- References: “R. Jin, J. Du, W. Huang, et al., [−“][+"][−”][+"] in Findings of the Assoc…”
- References: “I. Proskurina, L. Brun, G. Metzler, and J. Velcin, [−“][+"][−”][+"] in Findings of NAACL,…”
- …and 25 more (see .full.json)

</details>

### Renamed headings (1)
- “2.2 The Helpfulness–Harmlessness Trade-off” → “2.2 The Helpfulness, Harmlessness Trade-off”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
