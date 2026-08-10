# Claim-to-source entailment coverage — rewritten_fyp_report_v1

Closes the "cannot be independently reconstructed" boundary: the full inventory is here.

| Unit | Count | Audited |
|---|---|---|
| Prose sentences carrying a citation | 85 | 85 (88 in `claims.json`, incl. 3 later superseded by fixes) |
| Table rows carrying a citation | 8 | 8 |

**Prose pass** (`claims.json`, 88 entries): 12 candidates, 3 confirmed after adversarial
refutation (75% refuted), all fixed 2026-07-28.

**Table-row pass** (`table_rows.json`, `table_row_findings.json`): 4 rows had never been
audited by any pass — table cells were the blind spot named by the completeness critic.
5 non-supported cells found across 3 of those rows, all 5 verified against the primary
source by hand before editing, all fixed.

Two were factual contradictions, both about SCORER:

- **Kharinaev et al. [10]** marked `Automated`. Its headline contribution is 4,268
  human-annotated question-answer pairs across benchmarks "including human-centric
  evaluations" (arXiv:2502.15799). Corrected to `Human / mixed`.
- **HarmLevelBench [35]** marked `Automated`. "Human ASR" is a reported column in every
  results table alongside String ASR and a GPT judge (9 occurrences in the PDF).
  Corrected to `Human / mixed`.

Two credited a paper with a check it never performed:

- **Kharinaev [10]** over-refusal `Related` -> `Not reported`. Zero occurrences of
  over-refusal / exaggerated safety / false refusal / benign / XSTest; SafetyBench is
  used "to assess a model's understanding of safety concepts rather than its refusal
  behavior".
- **HarmLevelBench [35]** capability `Limited` -> `Not reported`. Zero occurrences of
  perplexity or MMLU; the paper reports only ASR.

One was phrasing:

- **HarmBench [24]** boundary "Threat model must be stated" -> "ASR conditional on the
  selected threat model". HarmBench defines its threat model (Sec. 3.1, Appendix B.3
  "Supported Threat Models") and ships a DirectRequest baseline, so the imperative read
  as a deficiency it does not have and broke the column's parallel form.

**Note both Kharinaev corrections strengthen the report's own argument.** Labelling the
closest prior human-grounded scorer study as merely "Automated" erased the nearest
methodological precedent for this study's own human gold sets; and conceding "Related"
over-refusal coverage gave away a gap the paper never actually occupies.

## Independent Codex verification (2026-07-28)

Codex re-verified the five table-cell repairs against the primary papers rather than
relying on the preceding audit narrative:

- **Kharinaev et al. [10]:** Section 4.3 says the four benchmarks combine automated
  evaluation and human assessment; Section 4.3.1 reports five linguistic experts,
  more than 20,000 evaluations, and 4,268 unique labelled question-answer pairs.
  Section 4.3.2 explicitly says SafetyBench measures safety concepts rather than
  refusal behaviour. Full-text searches returned no occurrence of `over-refusal` or
  `XSTest`. This supports `Human / mixed` and `Not reported`.
- **HarmLevelBench [35]:** Section 2.2 defines human annotation, string detection and
  an LLM judge as the three ASR methods. Tables 2–4 report `Human ASR` alongside
  `String ASR` and `GPT Judge`. Its limitations state that performance and accuracy
  trade-offs remain to be investigated, while full-text searches returned no
  occurrence of `MMLU` or `perplexity`. This supports `Human / mixed` and
  `Not reported`.
- **HarmBench [24]:** Appendix B.3 is titled “Supported Threat Models” and explicitly
  specifies attacker-access and defender assumptions, while Appendix C includes a
  Direct Request baseline. This supports `ASR conditional on the selected threat
  model` and rules out the earlier wording that implied HarmBench omitted a threat
  model.

`tests/test_rewritten_report_v1.py` now pins the exact corrected HarmBench,
Kharinaev and HarmLevelBench rows, in addition to the three prose/source repairs,
so a future rebuild that restores any old cell fails the focused report test.
