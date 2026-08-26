## Full FYP report (docx-js line): v3 → v4 (128-token era)

*From:* `docs/archive/FYP_Report_2026-06-26_v3.docx`  
*To:* `docs/archive/FYP_Report_2026-06-30_v4.docx`  

**Paragraph/body blocks:** +6 added · −0 removed · ~3 edited · ⟳0 citation-renumber-only · tables: +1 −0 ~1 · front matter blocks 35→35

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~1 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Under the classifier, four-bit NF4 never significantly reduces harmful compliance and significantly…

### Sections added
- ➕ Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — 6.16 Generation-Length Robus…

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix E: Repository Layout | 0 | 0 | 1 | 0 |
| Appendix G: Document Revision History | 0 | 0 | 1 | 0 |
| Chapter 6: Results and Analysis | 6 | 0 | 0 | 0 |

### Added paragraphs (6)

<details><summary>show</summary>

- `34353a3b` (132w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The main study generates at most max_new_tokens = 128. HarmBench's own reference evaluati…”
- `cad70bf6` (110w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The longer budget changes the generations as intended. At 512 the HarmBench responses are…”
- `235e6290` (54w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “Pair \| ΔASR @128 (classifier) \| ΔASR @512 (classifier) \| ΔASR @512 (gpt-4o) \|\| Qwen3…”
- `bd24ab1e` (181w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The headline result of the rerun is that the study's single significant safety regression…”
- `0ff70574` (69w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “Capability degradation, in contrast, is essentially unchanged by the generation budget. T…”
- `c5a32709` (138w) Chapter 6: Results and Analysis § 6.16 Generation-Length Robustness: the 512-Token Rerun — “The generation-length rerun therefore reinforces, rather than threatens, the study's cent…”

</details>

### Edited paragraphs (3; change window shown)

<details><summary>show</summary>

- (front matter): “…orer-sensitive, and concentrated in the smallest model.[+ A generation-length robustness…”
- Appendix E: Repository Layout: “└── FYP_Report_2026-06-[−26][+30][−3][+4].docx (this document)”
- Appendix G: Document Revision History: “…en (UTC+8) \| Version \| Change to the report \|\| 2026-06-[+30 11:30 \\| FYP_Report_202…”

</details>

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
