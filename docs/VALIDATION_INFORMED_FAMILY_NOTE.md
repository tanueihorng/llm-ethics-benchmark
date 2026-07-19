# Pre-analysis note — the validation-informed parallel BH-FDR family

**Status:** composition LOCKED 2026-07-19, before the family was computed.
This is a **post-hoc, clearly-labeled parallel analysis**, added after the T36
human-grounding outcome (`docs/XSTEST_GOLD_PREREG.md` §7, Outcome J). It is
**not** blind and does not claim to be: the per-pair judge-strict tests it
aggregates were already known from T35 (`results_512/analysis/
xstest_judge_agreement.json`, committed 2026-07-12). The purpose of this note
is **composition transparency** — fixing exactly which 20 p-values enter the
family and how they are corrected, before the computation runs, so the family
cannot later be suspected of having been assembled to produce a desired count.

## Why this analysis exists

The T36 human audit (200 items, blinded single annotator, disagreement-enriched
draw) found the gpt-4o 3-class judge substantially better aligned with the
annotator than the v2 refusal regex on the over-refusal construct (STRICT
κ 0.485 vs −0.006; BROAD 0.662 vs 0.054; regex recall 2/63). The original
regex-scored analysis remains the **original pre-specified scorer-of-record**
(D45) and its registered 20-test family (`multiple_comparisons.json`) is
retained unchanged for traceability. This parallel family gives the
substantive over-refusal conclusion a multiplicity-controlled footing on the
better-validated instrument, instead of resting it on verbal caveats alone.

## Locked composition

- **Family size 20**, mirroring the registered family exactly: 5 pairs ×
  {HarmBench ASR, MMLU accuracy, ARC accuracy, XSTest over-refusal}.
- **Only the over-refusal column changes scorer**: XSTest OR uses the
  **judge-STRICT** mapping (full-refusal only), the replication definition
  pre-registered in `docs/XSTEST_JUDGE_PREREG.md` (T35/D45). HarmBench ASR
  stays the official classifier; MMLU/ARC stay exact-match. Those 15 p-values
  are taken verbatim from the registered `multiple_comparisons.json`.
- The 5 judge-strict OR p-values are taken verbatim from the committed
  `xstest_judge_agreement.json` `per_pair[*].judge_strict.mcnemar_p_value`
  (exact McNemar, same test as the registered family).
- **Correction:** Benjamini–Hochberg at q < 0.05 over the 20 p-values —
  identical procedure and threshold to the registered family.
- **Output:** `results_512/analysis/multiple_comparisons_judge_strict.json`,
  a NEW sibling artifact; the registered artifact is never modified.

## Expectations stated before computing

Because every input p-value is already known, the outcome is arithmetically
anticipated, and it is **purely deflationary**: no judge-strict OR contrast is
individually significant (smallest p = 0.087), so the parallel family can only
*remove* the regex-scored Phi over-refusal survivor, never add a survivor.
Anticipated result: **2 BH survivors** (qwen_2b MMLU, llama_3_2_3b ARC — both
capability contrasts). If the computation deviates from this expectation in
any direction, that is a surprise to be investigated and disclosed, not
silently accepted.

## Reporting rules

- The registered family remains **the family of record**; report both families
  side by side; the substantive RQ2 conclusion may rest on this
  validation-informed family.
- Headline formulation: "no scorer finds a **statistically significant**
  over-refusal increase" (point estimates do move upward in places, e.g.
  judge-strict Qwen3-1.7B +0.040, n.s.).
- The prereg-committed clause for the Phi contrast — "most plausibly a
  (regex) measurement artifact" — is used intact, neither hardened nor
  softened.
- Disclosures carried wherever this family is presented: the T36 reference
  set is single-annotator and disagreement-enriched (not population ground
  truth); the annotator shared the judge's 3-class taxonomy (the benchmark's
  own); D45's no-swap rule was judge-outcome-blind but regex-outcome-aware;
  this parallel family was added after both scorers' results were known.
