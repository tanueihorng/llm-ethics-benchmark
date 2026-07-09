# Session notes — 2026-07-10

A short narrative recap of this working session. The authoritative record is
`docs/PROJECT_LOG.md` (§4 changelog, §3 decisions); this file is just a readable
summary of what was discussed and decided.

## Pre-submission audit (D44)
Ran a full adversarial audit of the report and thesis before submission. 27
findings raised, 11 confirmed, 15 refuted. Nothing invalidated a result. The main
fix: the significance criterion was unified on McNemar's exact test, so the
Qwen3-1.7B XSTest over-refusal delta (−0.028) is now reported as borderline
(p = 0.065), not significant. This resolved a report-vs-thesis contradiction.
Claim lock rebuilt; report and thesis regenerated.

## T32 — model revisions pinned
Pinned all five study models to the exact Hugging Face commit hashes cached on
TC1, on all ten config entries. Read the hashes from the snapshot directory names
after noticing the `refs/main` paste had clipped a character on both Qwen hashes.
Changes no result; closes the "weights not commit-pinned" caveat.

## T30 — human-label gold set (the main piece)
Labelled 200 stratified HarmBench responses by hand to ground the primary scorer
against human judgment.

Result: the HarmBench classifier agrees with the human at Cohen's κ = 0.59
(moderate); the refusal regex agrees at κ = 0.11 (negligible). Against the human,
the regex made 101 false-harmful calls to the classifier's 18. The human flagged
the fewest as harmful (37 of 200, vs the classifier's 46 and the regex's 132), so
both automated scorers over-flag relative to a person; the classifier's errors
look like balanced noise (18 over, 9 under), the regex's like systematic
over-counting (101 over, 6 under). This upgrades the scorer story from
"disagreement" to "correctness." κ triple-verified (script, claim lock,
independent recompute). Folded into report §6.12 and thesis §6.1, claim-locked.
Stated with honest bounds throughout: single annotator, moderate not perfect.

Two things surfaced during the labelling:
- A "self-continuation" artifact: generation halts only on the default
  end-of-sequence token, so some responses run past the model's answer into a
  synthetic follow-up turn (the model role-playing both sides), saved as a run-on
  by `skip_special_tokens`. Documented in report §6.16 and Chapter 8. It does not
  bias any matched-pair delta.
- A labelling-criterion critique (verified correct against `validation.py`): the
  classifier's rubric has no correctness or functionality condition, so a
  fabricated recipe or buggy code still counts as harmful if the content is
  concrete. Checked the actual labels: the drift was not systematic. Fixed the
  annotation tool's wording ("usable / working code") for any future annotator.

A "strict" wording confusion was corrected: the human is not "stricter than the
classifier" on disagreements (there she was the lenient one, 18 vs 9); she simply
sets the highest overall bar (flags the fewest). The report now states the plain
counts instead of the ambiguous word.

## Report, thesis, interim
Cross-checked the report and thesis: all ten shared findings align, no
contradictions. Explained the difference between the two documents (the report is
the long, thorough interim-milestone document; the thesis is the concise
standalone research document). Built a third document, an interim-milestone report
(`make interim`, ~25 to 28 pages), derived from the thesis so it reuses the same
claim-locked numbers, with progress framing and a "Progress and Remaining Work"
chapter. The full report and thesis were left untouched.

## Humanized variants
Produced humanized copies of all three documents (removing AI-writing tells) as
separate files, originals untouched. Removed every em/en dash and straightened
curly quotes, then had three editor agents strip AI vocabulary, copula avoidance,
filler, and cadence tells in a neutral academic register. Verified that every
number is byte-identical to the original and no dashes remain. Build with
`make report-humanized` / `thesis-humanized` / `interim-humanized`.

## Fable reasoning skills
Wrote two reusable reasoning skills authored by the Fable 5 model: `fable-reasoning`
and `fable-coding` (installed under `~/.claude/skills/`). Three more were planned
but paused when that model hit its usage limit.

## What remains (all optional, not blocking submission)
- Rotate the OpenAI key that was pasted in chat earlier.
- T1: send the July progress email to the supervisor (drafted).
- T3: run `MyTCinfo` on TC1 for the exact storage quota.
- T30b: a second annotator for an inter-rater κ.

The report and thesis are claim-locked and submission-ready.
