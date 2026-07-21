# T44 Phase 2B — statistics-appropriateness panel results

Workflow `wf_0cdecbd1-16c` (S3, 2026-07-21): 6 examiner seats (Opus 4.8, effort xhigh), one
method-choice question each; every finding attacked by 2 adversarial refuters (Opus 4.8 xhigh);
kill rule = both refute; Fable orchestrator adjudicated and re-verified the survivor's evidence
against primary artifacts. 38 agents, 0 errors, ~2.9M subagent tokens. Execution arithmetic was
out of scope (already proven by Phase 2A's 84/85 scripted recompute); this phase judged METHOD
CHOICE only.

## Seat verdicts — all six: `appropriate_with_caveats`

| Seat | Question | Verdict (one line) |
|---|---|---|
| mcnemar | exact McNemar right for paired binaries? | Yes — standard conditional-binomial exact test, correct two-sided doubling (null symmetric), correct zero-discordant handling, decoding-variance scope disclosed scrupulously |
| bh-family | BH family composition vs registration? | Yes — 20-contrast family matches its notes; D51 parallel family composition-locked; exploratory arms (INT8, multiseed, §6.14) correctly outside; BH-vs-BY acceptable |
| sidedness | one- vs two-sided consistent? | Yes — every test two-sided, uniformly applied both directions; "no significant increase" wording backed by disclosed two-sided tests + sidedness-invariant BH headline |
| ci-vs-test | CI vs test disagreements handled? | Yes — precedence rule stated (§6.5: per-comparison flags never read in isolation); the two uncorrected-only signals (qwen_4b ARC q=0.169, llama ΔASR q=0.107) each carry "uncorrected only" caveats |
| dz-effect-size | dz right for §6.14? | Yes — dz correctly named, paired nature disclosed; rank-biserial would be a nicety, not a correction |
| mde | MDE assumptions sound? | Yes for the ASR axis (formula, α/power basis fine) — but see FS-8: the INT8 *capability* verdict has no such machinery behind it |

## Findings: 15 raised → 14 refuted (unanimous) → 1 survived

**Survivor → FS-8 (P3, verified).** INT8 "capability-lossless" is an unbounded equivalence claim
from non-significance, with a misattributed basis:
- report v5:942 states the basis as "**(paired bootstrap)**" — `results_512/analysis/precision_sweep.json`
  contains **no bootstrap and no CI on any axis**; its only significance tests are ASR-axis McNemar.
  No capability-axis significance test existed in the pipeline at all (until Phase 2A's direct
  recompute manufactured exactly that evidence: 10 McNemar contrasts from 14,720 paired records,
  none significant, max |Δ| 1.33pp — so the VERDICT is true; the stated BASIS is wrong).
- No capability-axis MDE exists (`scripts/multiple_comparisons.py:265` filters MDE/power to
  `harmbench_asr_judge` only), yet thesis v4:303 says the INT8 deltas sit "below the study's
  detection floor" — importing the n=200 ASR MDE onto different metrics and n.
- Unqualified "capability-lossless" / "free precision point" (report:942/945) is equivalence-from-
  non-significance, inconsistent with the report's own bounded-null discipline on the safety axis
  (report:1015; multiple_comparisons.py:174). Both refuters: real, evidence-accurate, P3 (hedged by
  "within the study's resolution"; cliff-at-four-bit conclusion robust regardless).

**Refuted (kept for the record; each killed by both refuters — misread evidence, standard practice,
or already disclosed):** iid/super-population under-disclosure (McNemar is a conditional
randomization test; needs no super-population; anti-conservative direction can't manufacture the
nulls) · mid-p variant (preference, not defect) · pooled-family granularity (disclosed; robust) ·
BH positive-dependence note (independent-ish contrasts; BY not required) · §6.14 exploratory
labeling (already labeled) · one-sided-would-flip on two borderline contrasts (real arithmetic,
but tests are uniformly two-sided and disclosed; BH headline sidedness-invariant) · thesis
sidedness wording · report:833 ARC "significantly" without inline BH caveat (caveated at :849 and
§6.5.1) · evidence_status two-layer badge · percentile-vs-BCa bootstrap · dz-with-Wilcoxon
pairing · AUCs without CIs · MDE basis sentence · Hoenig–Heisey post-hoc power (report already
uses MDE-not-power framing where it matters).

## Optional polish list (NOT findings — refuted as defects, real as copy-edits; Phase 9 may spend them)

1. report:808 "family-wise-surviving" → "FDR-surviving" (one word; correct term used everywhere else).
2. report:833 add inline "(uncorrected only, §6.5.1)" to the ARC sentence (currently caveated by cross-reference).
3. One sentence in §3.7: benchmarks treated as exchangeable prompt samples; HarmBench/XSTest/ARC are
   fixed curated sets; clustering direction is anti-conservative for the nulls.
4. One sentence noting the two borderline contrasts (qwen_4b ASR p=.096, judge-strict qwen_2b OR p=.087)
   would cross α one-sided but nowhere near BH survival.

Overall Phase 2B verdict: **the statistical method choices are sound and honestly presented**; the
one defect is a basis/bounding slip on the secondary INT8 capability claim (FS-8 → Phase 9 R1,
where Phase 2A's direct McNemar evidence can become the corrected citation).
