## Standalone thesis: Thesis v2 → Thesis v3

*From:* `docs/archive/FYP_Thesis_2026-06-26_v2.docx`  
*To:* `docs/archive/FYP_Thesis_2026-06-26_v3.docx`  

**Paragraph/body blocks:** +0 added · −0 removed · ~26 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~1 · front matter blocks 40→40

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~5 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ FINAL YEAR PROJECT: THESIS
  - ❌ I hereby declare that this Final Year Project thesis is my own work and, to the best of my knowledg…
  - ❌ Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasin…
  - ❌ It contributes an open, reproducible benchmarking framework built around a matched-pair design (bas…
  - ❌ Three findings result. First, methodologically, the refusal regex systematically over-counts harmfu…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A Reproducibility Statement | 0 | 0 | 1 | 0 |
| Chapter 1 Introduction | 0 | 0 | 3 | 0 |
| Chapter 10 Conclusion | 0 | 0 | 1 | 0 |
| Chapter 2 Background and Related Work | 0 | 0 | 3 | 0 |
| Chapter 4 System Design and Implementation | 0 | 0 | 2 | 0 |
| Chapter 6 Results and Analysis | 0 | 0 | 9 | 0 |
| Chapter 7 Discussion and Threats to Validity | 0 | 0 | 1 | 0 |
| Chapter 9 Future Work | 0 | 0 | 1 | 0 |

### Edited paragraphs (26; change window shown)

<details><summary>show</summary>

- (front matter): “FINAL YEAR PROJECT[− —][+:] THESIS”
- (front matter): “…e in the text. The intellectual content of this thesis [−— ][+(][− —][+)] is the product…”
- (front matter): “…loyed on edge and consumer hardware, where quantization[− —][+,][− —][+,] has become the…”
- (front matter): “…chmarking framework built around a matched-pair design [−— ][+(][− —][+)] so that quanti…”
- (front matter): “…tistically significant effect from one model to another[− —][+,][− —][+:] a second, both…”
- Chapter 1 Introduction § 1.1 Motivation: “…act models in the one-to-seven-billion-parameter range [−— ][+(][− —][+),] four-bit quan…”
- Chapter 1 Introduction § 1.2 Problem statement: “…e for reasons that have nothing to do with quantization[− —][+:] a different published c…”
- Chapter 1 Introduction § 1.4 Contributions: “…nting scorer systematically over-counts attack success [−— ][+(][− —][+)][− —][+,] and t…”
- Chapter 2 Background and Related Work § 2.2 Safety benchmarks and red-teaming: “…aming benchmark with an explicit behaviour taxonomy and[−, crucially,][− —][+,] a delibe…”
- Chapter 2 Background and Related Work § 2.3 LLM-as-judge evaluation and its validity: “…tically biased [11]. This thesis follows that guidance [−— ][+(][− —][+)] and is explici…”
- Chapter 2 Background and Related Work § 2.4 The research gap: “…ter-judge agreement is often left unreported [10], [11][− —][+,] yet the benchmark’s own…”
- Chapter 4 System Design and Implementation § 4.2 Extensibility and the plugin contract: “…s keeps the framework reusable beyond the present study[− —][+,] a design property docum…”
- Chapter 4 System Design and Implementation § 4.4 Privacy discipline: “…s, boolean labels, scalar scores, and run metadata only[− —][+,] never prompt or respons…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…x vs classifier) \| Reading \|\| Qwen \| 0.19 – 0.37 \| poor[− —][+;][− —][+;][− —][+;][…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…sus the regex 'non-refusal' proxy, one marker per model[− —][+:][− —][+,] family-depende…”
- Chapter 6 Results and Analysis § 6.3 Capability anchoring (RQ3): “…r is what allows the safety deltas to be read correctly[− —][+:] the smallest Qwen pair …”
- Chapter 6 Results and Analysis § 6.4 Mechanism: refusal-margin geometry: “…nly weakly within a family, and the flips are symmetric[− —][+:] quantization destabilis…”
- Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5): “… (p = 0.022 classifier, 0.008 GPT-4o) yet non-monotonic[− —][+:] it reverts to baseline …”
- Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5): “…sentially flat through INT8 and falls only at four-bit [−— ][+(][− —][+),] while the saf…”
- Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5): “…the LLM.int8 algorithm on this model, not a general law[− — the honest, calibrated readi…”
- Chapter 6 Results and Analysis § 6.6 Statistical caveats: “…B effect (+0.055) has post-hoc power of only about 0.67[− —][+,] so the predominance of …”
- Chapter 6 Results and Analysis § 6.6 Statistical caveats: “…lysis/multiple_comparisons.json), exactly three survive[− —][+,] and all three are capab…”
- Chapter 7 Discussion and Threats to Validity: “…reas its effect on safety is method- and model-specific[− —][+:][+.][−— and f][+F]our-bi…”
- Chapter 9 Future Work: “Human-grounded judge validation [−— ][+(][− —][+)] and an open-weight guard…”
- Chapter 10 Conclusion: “…lace the study’s significant finding on the wrong model[− —][+,] so safety evaluation sh…”
- Appendix A Reproducibility Statement: “…pository follows recognised research-software practice [−— ][+(][− —][+)] and is package…”

</details>

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
