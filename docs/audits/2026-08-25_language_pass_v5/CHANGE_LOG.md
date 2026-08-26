# Change Log — v5 Full-Report Language Cleanup (2026-08-25)

Base: `docs/rewritten_fyp_interim_v4.docx` / `docs/rewritten_fyp_report_v4.docx`
(the verified v4 rewrite line, SHA-256 `e25cf6…cdf08` interim). Output:
`docs/rewritten_fyp_report_v5.docx` + `docs/rewritten_fyp_interim_v5.docx`,
built by the new non-destructive overlay builder
`scripts/build_rewritten_fyp_report_v5.py` (137 exact-substring corrections)
and `scripts/build_rewritten_fyp_interim_v5.py` (`make rewrite-interim-v5`).

Every correction preserves all numbers (build-verified token-by-token), all
citation markers in their original order (build-verified), all research
questions verbatim, all headings, all section numbering, all tables and figures
(except the two prose table cells listed below), and every conclusion's
direction and strength.

## 1. Survive / survivor shorthand → significance-after-correction language

| Where | Before | After |
|---|---|---|
| §4.2 | "No NF4 harmful-compliance contrast survives multiplicity correction." | "None of the five NF4 harmful-compliance comparisons remains statistically significant after correction for multiple comparisons." |
| §4.2 | "Its individually significant ΔASR is −0.040, a decrease that does not survive the 20-test correction." | "The estimated change in harmful compliance is −0.040. This difference is statistically significant before multiple-comparison correction, but it is no longer statistically significant after the 20 primary comparisons are adjusted using the Benjamini–Hochberg procedure." |
| §3.5 / §4.2 / §4.4 / §4.7 / §5.4 | "survives correction", "survived correction", "sole surviving", "survive BH-FDR", "ARC survives correction" | "remains statistically significant after Benjamini–Hochberg correction", "the only over-refusal effect that remains significant", "the ARC result remains statistically significant after correction" |
| §4.2 | "exactly two contrasts survived correction, both capability losses" (validation-informed family) | "exactly two comparisons remained statistically significant after correction, both capability decreases" |
| Table 6.2 cell | "none surviving correction" | "none statistically significant after correction" |

## 2. Contrast → comparison

Noun uses of "contrast" for statistical comparisons became
"comparison(s)" (§1.2 objective, §2.7, §3.5 ×2, §4.2 ×3, §4.3, §4.4, §4.5,
§4.6 ×2, §5.4, §6.6, §7.1, Table 6.2). Discourse connectives ("In contrast,",
"By contrast,") were kept — they are ordinary English, not statistical
shorthand. Two table row labels ("Contrast set only", "unsafe items are a
calibration contrast") were left unchanged as table labels.

## 3. Multiplicity → multiple-comparison correction

"multiplicity control/correction" → "correction for multiple comparisons" /
"multiple-comparison correction" (§1.5, §2.6 ×2, §3.5, §3.8, §4.2, §4.4, §5.8,
§6.6). §2.6's "separated from … multiplicity" became "…or the number of
comparisons tested".

## 4. Corrected family / statistical family

"The primary statistical family encompasses 20 NF4 comparisons" → "The primary
analysis covers 20 NF4 comparisons"; "The primary NF4 set of corrected
comparisons contains 20 contrasts" → "The primary NF4 analysis contains 20
comparisons"; "controls multiplicity across that family" → "corrects for
multiple comparisons across that set".

## 5. Unresolved → stated evidence status

- §3.3: "significant on ARC and unresolved on HarmBench" → "statistically
  significant on ARC yet inconclusive on HarmBench".
- §4.4: "These unresolved cases show why…" → "In these cases the evidence is
  inconclusive: …".
- §4.7: "the remaining estimates are mixed or unresolved" → "the remaining
  estimated differences vary in direction or do not provide clear statistical
  evidence of a change".
- Appendix B lead-in (§4.4): non-significant effects below the detection
  threshold are now explicitly "inconclusive rather than … evidence that fp16
  and NF4 are equivalent".

## 6. Construct → plain statement of what is measured

Nine prose occurrences rewritten (§2.1, §2.3 ×2, §2.4, §2.5 ×3, §3.5 ×2, §3.8,
§4.3, §5.1, §5.2, §6.4 ×3): e.g. "These two audits address different
constructs" → "These two validation exercises evaluate different behaviours and
scoring criteria"; "requires construct validation" → "requires validation that
the scoring method is measuring the intended behaviour". "Construct validity"
is RETAINED as a named measurement concept in §6.5.2 and is now defined at
first use: "Construct validity asks whether each benchmark and scorer measures
the behaviour it is intended to measure."

## 7. Scorer-dependent / scorer-sensitivity phrasing

"Phi-4-mini's −0.048 … is scorer-dependent" → "That result depends on the
scoring method, because the independent semantic judge does not reproduce it."
(§4.3, echoed in the Abstract). "proxy error can itself become
treatment-dependent" → "the proxy's error can itself depend on the precision
condition". "superior measurement fidelity" → "its closer agreement with the
human annotations" (no scorer is called correct).

## 8. Precision contrast / inferential threshold / point estimate / treatment effect

- "single controlled precision contrast" → "…precision comparison"; "in the
  same precision contrast" → "in the same fp16–NF4 comparison".
- "crosses the inferential threshold" → "reaches the threshold for statistical
  significance"; "its question is inferential:" → "to answer that question:".
- "point estimate(s)" replaced by "estimated change(s)" in seven result
  statements (§1.2, §4.2 ×2, §4.4 ×3, §4.5, §4.7); e.g. "Qwen3-4B is the
  largest point estimate for increased harmful compliance" → "Qwen3-4B shows
  the largest estimated increase in harmful compliance, at 0.040."
- "treatment effect(s)" ×4, "treatment member", "the primary treatment",
  "randomised treatments", "treatment-dependent" → quantization/condition
  wording; "Paired deltas remain comparable as treatment effects" → "The paired
  differences remain useful for comparing the effect of NF4 quantization within
  each model".

## 9. Heterogeneity / one shared path

"This heterogeneity is why results are reported as paired within-model
contrasts" → "This variation across models is why results are reported as
paired within-model comparisons"; "the evidence supports heterogeneity, not one
shared response to bit-width" → "the effect of lower precision differs across
models and benchmarks rather than following one consistent pattern"; "does not
follow one shared path" → "does not change consistently".

## 10. Interpretation labels defined where used

§3.9 label definitions rewritten so each label states its meaning in ordinary
language; §4.5 now reads: "Qwen3-1.7B shows a clear capability reduction while
harmful compliance is unchanged; the report labels this pattern broad
degradation. For Llama-3.2-3B, both harmful compliance and capability decrease
under NF4; because capability also declines, the lower harmful-compliance rate
should not be read as a safety improvement, and the report labels this pattern
capability loss with lower harmful compliance." Plus: "The interpretation
labels summarise the observed combination of safety and capability changes;
they do not establish a causal mechanism."

## 11. Tone words removed

"clean alignment gain" → "clear evidence of improved alignment" (negated);
"earlier signal" → "earlier apparent increase"; Table 5.2 cell "signal
disappears at 512" → "the effect disappears at 512"; "reveals" ×3 →
"shows"/"identifies"; "misleading narrative" → "misleading impression"; "far
more pessimistic" → "reports much higher attack-success rates"; "evidence
against a large, consistent increase" → "provide no indication of a large,
consistent increase"; "evidence favours" ×2 → "results support" / "validation
supports"; "human ground-truth annotation" → "human annotation"; "graded
fp16–INT8–NF4 curve" / "graded with precision" → "move gradually across
fp16, INT8, and NF4" / "changes gradually with precision".

## 12. Sentence splitting and audit/anchor/gap wording

Abstract: scorer-validation sentence split into result + consequence; "caused
measurable capability costs" → "produced measurable capability decreases for
specific model and benchmark combinations" (no causal over-reach). §1: "The
interpretive problem" → "The difficulty"; "practical methodological gap" →
"leaves developers without evidence…"; "useful anchors" → "useful reference
points"; "second anchor" → "second capability benchmark". §5.3: "human audit
sample … diagnostic power" → "human-labelled sample deliberately oversamples
the items on which the two scorers disagree"; "audit sample" →
"human-labelled sample"; "human audit" → "human annotation sample"/"human
annotation". §5.8: "supported along four partly independent dimensions" →
"Four partly independent checks support the null result"; "artifact does not
survive the move" → "anomaly does not reappear". §6.2: "underscore the
necessity of jointly evaluating … capability preservation" → "show why harmful
compliance, capability, and scorer behaviour need to be evaluated together".
§6.5: "causal attribution to the load-time precision change" → "attribution of
the observed differences to the load-time precision change"; "diagnostic, not
confirmatory" → "exploratory: they qualify the interpretation but do not
confirm it". §6.6: "declare multiplicity" → "state how many comparisons were
corrected".

## 13. Infrastructure changes (harness)

- New builders registered in `configs/claim_surfaces.yaml` (discovery +
  surfaces) and `configs/artifact_policy.yaml` (`epistemic_hedge.scan_paths`,
  `stale_text.scan_paths`, `report_worthy.changed_file_patterns`).
- `tests/test_rewritten_report_v5.py` added: rendered-document locks for the
  banned shorthand (each with a must-bite self-test), v4↔v5 numeric/citation/
  research-question/heading/table equality, and the two declared table-cell
  prose edits.
- `tests/test_ieee_citation_order.py` and `tests/test_citation_completeness.py`
  extended to cover the v5 documents/builder.
- `scripts/verify_claim_surfaces.py`: the three wording-coupled `docx_primary`
  checks (survivor-count, vi-survivor-count, no-asr-survivor) now also accept
  the meaning-identical cleared phrasing ("exactly N comparisons remained
  statistically significant after Benjamini–Hochberg correction"), and the
  contradiction detectors were extended symmetrically so wrong/hedged counts
  and un-negated post-correction ASR claims still fail in the new vocabulary.
  Four must-bite fixtures added to `tests/test_claim_registry.py`.
- `Makefile`: new `rewrite-interim-v5` target.
- `scripts/report_tools/check_pagination.py`: `render_pdf` now retries once
  with an isolated throwaway LibreOffice profile when the default profile is
  broken (environment failure observed 2026-08-25); this unblocked
  `tests/test_report_pagination.py` without weakening any check.
