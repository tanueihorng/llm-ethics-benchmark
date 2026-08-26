## Standalone thesis: Thesis v1 (early) → Thesis v2

*From:* `docs/archive/FYP_Thesis_2026-06-18.docx`  
*To:* `docs/archive/FYP_Thesis_2026-06-26_v2.docx`  

**Paragraph/body blocks:** +13 added · −3 removed · ~5 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~1 · front matter blocks 37→40

*Front matter (cover/abstract/TOC/lists):* +3 −0 ~2 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ 26 June 2026
  - ❌ Date: 26 June 2026
  - ➕ Figure 1 Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.
  - ➕ Figure 2 The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).
  - ➕ Figure 3 Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Chapter 1 Introduction | 1 | 1 | 0 | 0 |
| Chapter 3 Methodology | 0 | 0 | 1 | 0 |
| Chapter 6 Results and Analysis | 5 | 1 | 2 | 0 |
| Chapter 9 Future Work | 1 | 1 | 0 | 0 |
| References | 3 | 0 | 0 | 0 |

### Removed paragraphs (3 body; 0 front matter)

<details><summary>show</summary>

- `0c3980e6` (114w) Chapter 1 Introduction § 1.4 Contributions — “This thesis makes three contributions. First, a methodological one: a capability-anchored…”
- `18806867` (120w) Chapter 6 Results and Analysis § 6.6 Statistical caveats — “Three caveats bound the conclusions. With 200 HarmBench prompts the confidence intervals …”
- `f4f182b3` (13w) Chapter 9 Future Work — “Human-grounded judge validation, to remove the residual construct threat on the primary s…”

</details>

### Added paragraphs (10)

<details><summary>show</summary>

- `d4fed0a8` (180w) Chapter 1 Introduction § 1.4 Contributions — “This thesis makes four contributions. The first, and the one with consequences beyond thi…”
- `0160c611` (44w) Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts — “Figure 1. Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal…”
- `a6c95079` (41w) Chapter 6 Results and Analysis § 6.2 Main study (fp16 vs NF4) — “Figure 2. The capability-anchored safety space. Each pair is placed by its capability del…”
- `5f7f2e72` (47w) Chapter 6 Results and Analysis § 6.5 Precision point: fp16 → INT8 → NF4 (RQ5) — “Figure 3. Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat t…”
- `fea734a2` (91w) Chapter 6 Results and Analysis § 6.6 Statistical caveats — “Three caveats bound the conclusions. With 200 HarmBench prompts the confidence intervals …”
- `9ac0f590` (156w) Chapter 6 Results and Analysis § 6.6 Statistical caveats — “Significance is also nominal and, rather than merely noting this, we report the corrected…”
- `72b778d7` (38w) Chapter 9 Future Work — “Human-grounded judge validation — a stratified human-labelled subset reporting classifier…”
- `051188cf` (24w) References — “[19] Q. McNemar, “Note on the sampling error of the difference between correlated proport…”
- `5c904812` (18w) References — “[20] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman &…”
- `324e94ca` (17w) References — “[21] H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversa…”

</details>

### Edited paragraphs (5; change window shown)

<details><summary>show</summary>

- (front matter): “[−18][+26] June 2026”
- (front matter): “Date: [−18][+26] June 2026”
- Chapter 3 Methodology § 3.6 Decoding controls and statistics: “… Deltas carry paired-bootstrap 95% confidence intervals[+ [20]][+ [19]][−n][+rep][−minal…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “….79 \| good — little ambiguous middle ground \|\| Phi \| 0.[−59 – 0.][−m][+go][+ — lik][…”
- Chapter 6 Results and Analysis § 6.1 Scoring validity: the regex over-counts: “…udge-versus-regex agreement (Cohen’s κ) by model family[+ (fp16-vs-NF4 main study)].”

</details>

### Renamed headings (1)
- “List of Tables” → “List of Tables and Figures”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
