## Humanized alternate — thesis: Base — thesis v4 → Humanized thesis alternate

*From:* `docs/FYP_Thesis_2026-07-02_v4.docx`  
*To:* `docs/FYP_Thesis_2026-07-02_humanized.docx`  
*Decisions:* D43  

**Paragraph/body blocks:** +1 added · −3 removed · ~56 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~1 · front matter blocks 41→41

*Front matter (cover/abstract/TOC/lists):* +1 −1 ~3 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ A Matched-Pair, Judge-Validated Study of Safety, Capability Trade-offs in Quantized Compact Languag…
  - ❌ All experimental results reported in this thesis were produced by the open-source benchmarking fram…
  - ❌ Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasin…
  - ❌ Three findings result. First, methodologically, the scorer and the generation budget each determine…
  - ➕ All experimental results reported in this thesis were produced by the open-source benchmarking fram…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A Reproducibility Statement | 0 | 0 | 1 | 0 |
| Chapter 1 Introduction | 0 | 0 | 1 | 0 |
| Chapter 10 Conclusion | 0 | 0 | 1 | 0 |
| Chapter 2 Background and Related Work | 0 | 0 | 4 | 0 |
| Chapter 3 Methodology | 0 | 0 | 3 | 0 |
| Chapter 6 Results and Analysis | 0 | 1 | 10 | 0 |
| Chapter 7 Discussion and Threats to Validity | 0 | 1 | 3 | 0 |
| Chapter 8 Limitations | 0 | 0 | 3 | 0 |
| Chapter 9 Future Work | 0 | 0 | 2 | 0 |
| References | 0 | 0 | 25 | 0 |

### Removed paragraphs (2 body; 1 front matter)

<details><summary>show</summary>

- `d44a121c` (43w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget — “The Qwen3-1.7B pair, whose +0.055 increase was the study’s original headline at the 128-t…”
- `21b1fbd5` (357w) Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity — “The primary safety construct is scored by the benchmark’s own fine-tuned classifier rathe…”

</details>

### Edited paragraphs (56; change window shown)

<details><summary>show</summary>

- (front matter): “A Matched-Pair, Judge-Validated Study of Safety[−–][+, ]Capability Trade-offs in …”
- (front matter): “…them: a brittle refusal-matching scorer can over-count [−“][+"][−”][+"] and the number o…”
- (front matter): “…ity losses and one benign-direction over-refusal change[+.][−—][+That][−th][+lon][−latt]…”
- Chapter 1 Introduction § 1.2 Problem statement: “…ally scored by pattern-matching for refusals, equating [−“][+"][−”][+"][−“][+"][−”][+"] …”
- Chapter 2 Background and Related Work § 2.1 Quantization and its behavioural effects: “…ovides the two schemes evaluated here: NF4, a four-bit [−“][+"][−”][+"][+own ][−— ][+(][…”
- Chapter 2 Background and Related Work § 2.2 Safety benchmarks and red-teaming: “HarmBench [3] provides a standardi[−z][+s]ed red-teaming benchmark …”
- Chapter 2 Background and Related Work § 2.3 LLM-as-judge evaluation and its validity: “… human-label validation on a stratified 200-item subset[− —][+,] a single-annotator gold…”
- Chapter 2 Background and Related Work § 2.4 The research gap: “… the gap this thesis addresses. First, the quantization[−–][+, ][−–][+, ][−“][+"][−”][+"…”
- Chapter 3 Methodology § 3.4 Scoring: classifier-primary, regex-as-foil: “…m) re-scores the same generations as a robustness check[−; independent validation of jud…”
- Chapter 3 Methodology § 3.5 Capability-anchored interpretation: “…ent shift; if capability collapses, an apparent safety [−“][+"][−”][+"] may merely be th…”
- Chapter 3 Methodology § 3.6 Decoding controls and statistics: “…representative pair, with the five pairs spanning 0.044[−–][+ to ]0.086); and a multi-se…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…(moderate) while the regex agrees at only 0.11 (slight)[− —][+,][− —][+,] so refusal-cou…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…hen’s κ (regex vs classifier) \| Reading \|\| Qwen \| 0.36 [−–][+to][−–][+to][−–][+to][−…”
- Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) at the reference budget: “…hi-4-mini’s decrease (ΔOR = −0.048, CI [−0.076, −0.020][−,][+)][−;][+, and][−,][−and][+(…”
- Chapter 6 Results and Analysis § 6.3 The generation-budget artefact (why 512 tokens is the primary budget): “…cent a floor). Truncation is family-heterogeneous (93.5[−–][+ to ][−–][+ to ][−rat][− th…”
- Chapter 6 Results and Analysis § 6.4 Capability anchoring (RQ3): “…ic within-Qwen scale gap looks partly MMLU-specific. Th[−is][+at][+e][−ent diverge][+a][…”
- Chapter 6 Results and Analysis § 6.5 Mechanism: refusal-margin geometry: “…en boundary instability than a targeted alignment shift[−,][+. This][−ing][+es] the capa…”
- Chapter 6 Results and Analysis § 6.6 Precision point: fp16 → INT8 → NF4 (RQ5): “…em McNemar contrasts, none significant — far below the [−two mult][+s][−pl][+gnif][−ity-…”
- Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives: “…an-discordant) pair, with the five pairs spanning 0.044[−–][+ to ][−“][+"][−”][+"]), not…”
- Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives: “…i-4-mini over-refusal (−0.048, q = 0.012, a decrease). [−Not one HarmBench ASR contrast …”
- Chapter 6 Results and Analysis § 6.7 Statistical robustness: multiplicity, power, and what survives: “[−B][+Ev][−cause the][+ry][−every con][−r][+h][−s][+f][−bove][+mily][−subs][+lat][−quentl…”
- Chapter 7 Discussion and Threats to Validity: “…e results answer the research questions with calibrated[−, not sweeping,][− —][+,][+i][−…”
- Chapter 7 Discussion and Threats to Validity § 7.2 Construct validity: “[+The primary safety construct is scored by the benchmark’s own fine-tuned classifier rat…”
- Chapter 7 Discussion and Threats to Validity § 7.3 External and statistical-conclusion validity: “… analysis (about 0.06 at the representative pair; 0.044[−–][+ to ]0.086 across the five)…”
- Chapter 8 Limitations: “…% power is about 0.06 at the representative pair (0.044[−–][+ to ]0.086 across the five)…”
- Chapter 8 Limitations: “…ond-judge cross-check, and the XSTest refusal judge is [−like][+no][−ise][+ the same way…”
- Chapter 8 Limitations: “…-run greedy divergence, since Llama truncates at only 3[−–][+ to ]4 percent).”
- Chapter 9 Future Work: “…ization-versus-safety-under-attack result. This is the [−highe][+mo][−-leve][+ impo][+t]…”
- Chapter 9 Future Work: “…mpleted single-annotator human validations on both axes[− —][+:][+ which is][−:][+, with…”
- Chapter 10 Conclusion: “…t plausibly a measurement artifact of the regex scorer [−— not reproduced by ][+(][+does…”
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
