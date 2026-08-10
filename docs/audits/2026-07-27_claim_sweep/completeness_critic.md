# Completeness critic — what the claim-enumeration method still cannot see

Produced 2026-07-27 by the gap-closing run (`wf_6deb7927-123`). This is the component
that four earlier audit rounds never got to run. Its headline file citation points at
`build_rewritten_fyp_report_v1.py`, NOT the v6 builder — but the defect CLASS was
verified to apply to v6's own Table 2.1 (11 data rows x 6 columns) and partially
remediated in the 2026-07-27 13:30 changelog row. The remaining items below are open.

## The single most likely remaining defect: Table 2.1

`scripts/build_rewritten_fyp_report_v1.py:1084–1104`. Every audited unit in `docs/audits/2026-07-27_claim_sweep/confirmed_42.json` is a chapter or the abstract (`ch1`, `ch2`, …, `appendixI`) — **zero table cells, zero captions, zero reference-list entries**. Table 2.1 has 9 rows × 7 columns = 63 cells, and roughly 50 of them are uncited empirical assertions about other researchers' papers:

- `("TrustLLM / DecodingTrust [20], [21]", "No", "Broad", "Partial", "Yes", "Automated", …)` — "Partial" over-refusal coverage is a substantive characterisation of two large benchmark suites, defensible or not, with no supporting text anywhere.
- `("StrongREJECT / SORRY-Bench [41], [42]", …, "No matched precision intervention")` — a class-(f) absence claim in a cell.
- `("XSTest / OR-Bench [18], [19]", "No", "No", "Yes", "No", …)` — four claims, four cells, no prose.

And the novelty claim itself is carried structurally, not lexically. The prose says only "Table 2.1 compares representative studies…" and "Blank or partial cells reveal the gap" — both innocuous to a sentence auditor. The exclusivity claim (class c) is made by the *visual contrast* between eight rows of "No" and the final `("Present study", "NF4 + INT8", "Yes", "Yes", "MMLU + ARC", "Multi-judge + human", …)` row. No sentence says "we are the first"; the table says it, and the table is unauditable by the method used. This is the same shape-vs-words root cause the four rounds kept hitting, moved into a different carrier.

## Defect classes absent from the six

- **(g) Scope drift in quantifiers of the study's own design** — "five pairs across four families" reads as coverage of *families*; n=1 model per family supports no family-level inference. Not prevalence, not universal, not superlative; a sample-to-population slide inside the authors' own numbers.
- **(h) Construct substitution** — the report's own honest phrase, "direct requests", degrading to "safety" between mention and restatement. Each word individually correct.
- **(i) Ordering/salience** — "the results favour the latter two explanations" (§6.1) enumerates three explanations, then reports two. Nothing false; the ranking implies a comparative test that was not run.
- **(j) Temporal/version claims** — "the installed transformers", "current" model releases. True at writing, silently false at submission.
- **(k) Negative-space claims from a citation-set that was itself selected** — the whole §2.6 "five gaps follow" argument depends on the *completeness* of a 64-item reference list, which no per-sentence check can assess.

## Non-prose carriers, ranked by residual risk

1. **Table cells** (above) — highest.
2. **Table headers**. `"Main boundary"` presupposes each prior study has one and that this report identified it correctly. `"Harmful compliance" / "Over-refusal" / "Capability"` presupposes those are the right three axes — an argued position rendered as neutral structure.
3. **The `add_table_caption`/`add_figure_caption` strings**. `"Table 5.1. Per-pair HarmBench deltas under three judges"` — "three judges" implies three comparable instruments; §5 elsewhere explains Llama Guard answers a different question. The caption drops the caveat and captions are what examiners read first.
4. **Section titles**. `"6.1 Answer to the Central Question"` asserts an answer exists; `"5.8 Robustness Synthesis"` asserts robustness. Titles are read as verdicts.
5. **The reference list as a claim**. Fifteen archived PDFs, 64 entries — the *composition* claims the field's relevant work is represented.
6. **Abbreviations table**, `line 675`: `("TC1", "NTU GPU compute cluster used in this project")` — the definite article claims TC1 is *the* cluster. Trivially fixable, invisible to prose scanning.

## Cross-sentence and cross-section defects

Neither half is defective alone, so both survive independent adversarial refutation:

- **§6.1 pair**: "The results favour the latter two explanations" is followed by "This conclusion is deliberately bounded." The hedge follows the claim. A reader who stops at the paragraph break gets the unhedged version — and abstract/conclusion readers always stop early.
- **Chapter 1 → Chapter 6**: `line 719` "It is a plausible deployment form of the model" (correctly hedged, cited to AWQ/GPTQ) licenses the abstract's `"commonly deployed quantized"` — which *was* flagged (finding #1). But the reverse case, where the hedged original survives and only the restatement is checked in isolation and passes, is invisible. The method audits sentences, so it structurally cannot see a *pair*.
- **§5.8 → §6.1**: "make a large hidden harmful-ward effect less plausible" (correct) versus "no corrected harmful-ward increase" (correct). Read together they license "NF4 is safe", which appears nowhere. Emergent claims have no sentence to flag.

## Where this auditor produces false positives

The counting is diagnostic: **~210 candidates raised, ~50 survived — a ~76% false-positive rate.** Each refutation costs judgement, and the pressure is toward weakening correct claims:

- Class (f) fires on *correct* statistical language. "No NF4 harmful-compliance contrast survives multiplicity correction" is exactly right, and "The defensible claim is the absence of a corrected increase large enough to be detected in this design" (§5.7) is the textbook correct formulation — but both pattern-match "absence stated as evidence of absence."
- Class (a) fires on the report's own *measured* prevalence figures. "60.3% direct-prefix truncation" is a computed number from `genlen_robustness.json`, not an adoption claim.
- Class (c) fires on legitimate comparatives grounded in committed artifacts: "classifier κ 0.5908 versus regex κ 0.1095."
- The generic risk: over-hedging the Discussion until the report has no thesis. An examiner penalises "this study cannot conclude anything" as heavily as over-claiming, and the D56 rule ("prefer the fix that adds rigour") only helps when a computation is available.

## What a human examiner sees that none of this catches

- **Whether the research question was worth asking.** Five pairs, one hardware platform, direct requests only, no attacks — sentence-level correctness is orthogonal to "is this an FYP-scale contribution?"
- **The rendered artifact.** Table 2.1 sets `font_size=7.3` with column widths `[3.0, 1.7, 2.0, 1.6, 1.8, 2.1, 3.4]`. Whether cells wrap illegibly, whether Figure 4.1's error bars are visible in print, whether the Greek and the − sign render — the builder is not the document.
- **Tone mismatch.** Chapter 2 reads as a careful survey; Chapter 6 reads more confident. Nobody wrote a false sentence; the register shifts.
- **The absence of the obvious control.** No examiner asks "is this sentence supported?" — they ask "why didn't you run fp16 twice to get a floor on run-to-run variance?" An audit enumerating existing claims can never surface a missing experiment. This is D54's symmetry rule, unapplied at the level of the study design rather than the bibliography.
- **Whether the honesty is load-bearing or performative.** The report discloses a pre-registration deviation and labels preprints. An examiner will ask whether the disclosed deviation actually changes the conclusion — a question about the *analysis*, not the *prose*.

**Recommendation:** run one pass whose unit is a table cell, one whose unit is a caption, and one whose unit is a section title, before another prose round. Table 2.1's ~50 uncited cells about other people's papers are the highest-yield target in the document, and no round so far has looked at them.