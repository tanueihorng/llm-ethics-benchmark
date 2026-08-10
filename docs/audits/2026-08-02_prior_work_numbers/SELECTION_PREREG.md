# Pre-registered selection rule — importing published figures into §6.2

Date: 2026-08-02 (Asia/Singapore)
Task: T47 gap 6 — "prior work is never given a number"
Target artifact: `docs/rewritten_fyp_report_v1.docx` §6.2, via `scripts/build_rewritten_fyp_report_v1.py`

**This file was written before the extracted figures were read.** It exists because
importing published numbers is the operation that produced the 26 claim-fitness defects
behind D56, and because the first completed extraction showed how cherry-pickable these
result tables are: Q-Misalign's headline `0.00 → 0.97` belongs to an attacker-prepared
checkpoint, while its *control* column (unmodified model, `0.07 → 0.07`) is the figure that
is actually commensurable with this study. Choosing between those two after seeing which
one flatters the null is not a defensible procedure. So the rule is fixed first.

## 1. What gets imported

One primary figure per prior study named in the gap (five studies: [10], [35], [36], [37],
[39]). Where a study genuinely has no single defensible number, that is recorded as an
exclusion rather than papered over with a manufactured one.

## 2. Priority order for choosing that figure

Applied in order; the first applicable condition wins.

1. A quantized-versus-baseline **delta** beats an absolute level.
2. The paper's own **unmodified / control** condition beats an attacked, adversarially
   prepared, or safety-patched one.
3. The condition **closest to this study's setting** wins: direct harmful request over
   attack-augmented; load-time post-training quantization over calibrated methods;
   four-bit over other widths.
4. Where several models are reported, the paper's own headline or most-discussed model
   wins, and the report names which model it is.

## 3. Direction blindness

The figure is selected by the rule above **before** checking whether it agrees with this
study's null. A figure that contradicts this study is reported with the same prominence
as one that supports it. If applying the rule yields a number that undercuts the report's
conclusion, that number is the one that goes in.

## 4. Mandatory accompanying fields

No figure is imported bare. Each carries, in the table or its caption: the metric **as the
source defines it**, the scorer, the models, the quantization method, the bit-width, the
sample size where stated, and an explicit statement of why it is not directly comparable
to this study's contrast.

## 5. Transcription discipline

Values are transcribed **as printed**. No unit conversion (a printed `9.2` is not silently
turned into `0.092`), no sign flipping, no rounding beyond the source's own precision. Where
a paper's sign convention differs from this report's, the convention is stated rather than
the number adjusted.

## 6. Permitted exclusions

A study may be excluded from the numeric table only for:

- the primary source could not be reached in full text, or
- the paper's relevant result is genuinely qualitative with no defensible single number.

Every exclusion is logged with its reason in `PROVENANCE.md`, and the study keeps its
existing qualitative treatment in the prose.

## 7. Verification

Every imported figure is verified by an independent agent that did not extract it, under two
distinct lenses (entailment: does the source report this value with this meaning?
comparability: does placing it beside our figure mislead?), with no sibling figures visible.
This follows D57: batching is for reading, never for judgement. Verdicts are recorded in
`PROVENANCE.md` and the surviving numbers are pinned by a regression test.
