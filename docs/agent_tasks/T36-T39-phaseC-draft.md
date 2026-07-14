# Phase C fold-in DRAFT — T36–T39 (staging only; NO builder edits here)

> **⚠️ SUPERSEDED IN PART BY T40/D47 (2026-07-14).** §1 below (the T38 capability-parser
> passage, including the ARC "format-immune" **factual correction**) has ALREADY BEEN
> APPLIED to the report and thesis by the P1 submission-safety patch (T40, decision D47;
> `docs/agent_tasks/P1-submission-safety-patch.md`). The report §6.2/§6.4.1/§6.5/Ch8 and
> thesis now carry the lenient/strict bracket framing, ARC strict −0.343 / MMLU −0.293 are
> claim-locked to `parser_strict_sensitivity.json`, and a stale-text guard forbids the
> retired "immune" phrasing. **Phase C must NOT re-apply §1** — it is done. Phase C now
> only needs the T36 (gold-set outcome J/R/T/X), T37 (LlamaGuard third judge), and T39
> (5-pair multiseed) EVIDENCE fold-ins (§2–§4 below), each with its own claim-lock. Keep §1
> here only as the record of what was corrected.

**Status:** DRAFT prepared by Opus 4.8, 2026-07-13 (WS-3 of packet 5fea5d2); §1 executed
2026-07-14 via T40. The remaining combined Phase-C fold-in happens ONCE, after T36 labels +
T37 LlamaGuard + T39 multiseed TC1 results are all in (D46 batch rule). Until then,
**do not touch** `build_fyp_*.js`, `mythesis.tex`, or `verify_report_claims.py` for §2–§4.

Numbers below are the committed artifacts as of `4e73380`:
- T38: `results_512/analysis/parser_strict_sensitivity.{json,csv}` (LIVE — committed).
- T36: `results_512/analysis/xstest_human_validation.json` (PENDING — user labels).
- T37: LlamaGuard agreement (PENDING — TC1 job, then `judge_pairwise_agreement.py`).
- T39: updated `sensitivity_multiseed.json` (PENDING — TC1 seeds for mistral/phi).

---

## 1. T38 — capability parser passage (the load-bearing prose) ⚠️ INCLUDES A FACTUAL CORRECTION

### 1a. What the strict artifact actually shows (all committed)

| pair | MMLU primary | MMLU strict | ARC primary | ARC strict | 4bit fallback % (MMLU / ARC) |
|---|---|---|---|---|---|
| **qwen_2b** | −0.090 **sig** | **−0.293** [−0.350,−0.237] **sig** | −0.009 n.s. | **−0.343** [−0.375,−0.311] **sig** | **48.7% / 52.3%** |
| qwen_4b | −0.003 | −0.003 | −0.016 sig | −0.017 sig | low |
| llama_3_2_3b | −0.037 | −0.037 | −0.032 sig | −0.032 sig | low |
| mistral_7b | −0.020 | −0.017 | +0.009 | +0.010 | low |
| phi4_mini | −0.027 | −0.030 | −0.015 | −0.008 | low |

Only **qwen_2b (the smallest model)** shows the strict/primary divergence; the other four
pairs are strict≈primary (their 4-bit answers stay format-compliant). qwen_2b_4bit fine
print: leading-letter accuracy **75.0% (MMLU) / 79.9% (ARC)** — *higher* than base's
64.9% / 73.2% — but on far fewer items; fallback-tier accuracy 45.9% / 65.7%.

### 1b. ⚠️ FACTUAL CORRECTION required, not just a reframe

The current report **§6.5 (third statistical caveat, the answer-parsing sub-point; builder line
823)** states, verbatim:
> "ARC-Challenge — scored by an identical protocol for both pair members and **not subject
> to this format asymmetry** — moves in the same (negative) direction (−0.009, not significant)"

and **Ch8 (MMLU answer-format sensitivity item; builder line 1023)** repeats it: "ARC, which is
**immune to this asymmetry**, corroborates the direction of the loss at a smaller magnitude."
(Note: §6.4.1 itself does NOT contain the verbatim "not subject/immune" clause — it carries only
the softer, separately-listed "substantially MMLU-specific (a subset-content sensitivity)"
wording at builder line 820, a weaker rewrite target handled in §1d.)

**This is false.** qwen_2b_4bit's ARC answers collapse in format *more* than its MMLU answers
(52.3% vs 48.7% lenient-fallback usage). ARC's near-zero *primary* delta (−0.009) is not
evidence of format-immunity — it is the lenient parser **salvaging** the format-broken 4-bit
ARC answers (fallback-tier accuracy 65.7% on 4-bit ARC). Under strict format-compliant
scoring ARC collapses to −0.343 (sig). The "ARC is format-immune, so the MMLU loss is
MMLU-specific" argument (§6.2/§6.4.1/RQ4) therefore **does not hold as written** and must be
rewritten, not appended to.

### 1c. Binding constraints on the replacement prose (do NOT violate)

- Present a **bracket / measurement-dependence interval**, never a single "true" number:
  qwen_2b NF4 capability loss is **protocol-dependent** — MMLU −0.090 (lenient) to −0.293
  (strict); ARC −0.009 (lenient) to −0.343 (strict). **Direction robust; magnitude an
  interval set by the scoring protocol.**
- **NEVER call −0.293 or −0.343 "the true loss."** Strict scoring conflates knowledge with
  format compliance; and the 4-bit clean-format answers are *more* accurate than base's
  (75%/80% vs 65%/73%) though far rarer — a selection effect that forbids any simple "true
  knowledge loss" reading. State this fine print explicitly.
- Frame it as **measurement-dependence axis #3**, completing the thesis's central claim:
  harm scorer (regex→classifier, §6.12) → over-refusal scorer (regex→judge, §6.12 Result 6,
  T36) → **capability parser (lenient→strict, T38)**. All three axes: the instrument, not
  just the model, moves the number. This STRENGTHENS the thesis.
- Keep the primary-scorer numbers as the reported headline (−0.090 MMLU / −0.009 ARC remain
  the primary study's values); the strict figures are a **sensitivity layer**, exactly like
  the judge for over-refusal. Do NOT change §6.5.1 BH-FDR (the strict figures add no new
  significance-family member; qwen_2b MMLU already survives).
- Scope: this divergence is **qwen_2b-only**; do not imply the other four pairs are affected.

### 1d. Exact anchors to REWRITE (report v5 + humanized + thesis v4/humanized + interim×2 + LaTeX)

- **§6.5 third caveat (builder line 823) — THE load-bearing edit.** The clause "ARC … not
  subject to this format asymmetry … −0.009, not significant … its MMLU magnitude is partly
  parser-inflated" is the flat factual error. Rewrite: ARC is NOT format-immune (4-bit ARC
  fallback 52.3% > MMLU 48.7%); its −0.009 primary delta is lenient-salvage; strict ARC −0.343.
  The "partly parser-inflated" direction is backwards — strict scoring *widens* the gap, the
  lenient parser *narrows* it by salvage. Use the §1c bracket framing.
- **Ch8 MMLU-answer-format item (builder line 1023).** "ARC, which is immune to this asymmetry,
  corroborates the direction … at a smaller magnitude" → same correction; ARC's format DOES
  collapse; keep the truncation-resolved-at-512 point (that part is still true) but replace the
  "immune"/"genuinely wrong answers" framing with protocol-dependence.
- **§6.4.1 (builder line 820) — weaker, related rewrite.** "the severe MMLU figure is
  substantially MMLU-specific (a subset-content sensitivity)" and "the ≈30:1 within-Qwen scale
  gap does not replicate under ARC" both lean on the lenient ARC −0.009; under strict scoring
  ARC also collapses (−0.343), so the "MMLU-specific" and "does not replicate under ARC"
  readings are protocol-dependent and must be hedged accordingly.
- Ch8 item (MMLU answer-format sensitivity, the "resolved at 512 / genuinely wrong answers"
  passage): rewrite to "format-compliance sensitivity is a live measurement-dependence axis
  at 512 for the smallest pair on BOTH capability benchmarks; magnitude is protocol-bracketed
  (§6.4.1); truncation specifically is resolved at 512, but format collapse is not truncation".
- §6.2 / RQ4 within-Qwen scale: the ≈30:1 MMLU ratio claim and its ARC hedge both need the
  protocol-dependence note.
- New §6.4.2 (or a §6.4.1 subsection) may present the strict sensitivity table (1a) as an
  in-document artifact, cited to `parser_strict_sensitivity.json`.

## 2. T36 — outcome-conditional §6.12 Result-6-extension templates (fill after labels)

Read `results_512/analysis/xstest_human_validation.json`: κ_R = κ(human, regex, STRICT),
κ_J = κ(human, judge, STRICT), n_labeled. Precedence (prereg §5 + §5.1 A1):
**X → T(both κ<0.40) → J(κ_J−κ_R≥0.15) → R(κ_R−κ_J≥0.15) → T(tie)**.

- **Outcome J (judge grounded).** "A human gold set (n=[N], single annotator, blind 3-class)
  agrees with the judge at κ_J=[..] and the regex at κ_R=[..] (Δ≥0.15). The scorer closer to
  human judgment does not reproduce the Phi −0.048 over-refusal decrease, so Result 6's
  scorer-dependence resolves toward *regex-artifact*: the FDR-surviving over-refusal contrast
  is most plausibly a property of the demoted regex. The regex remains the primary scorer of
  record (D45); the BH-FDR family is unchanged; the construct caveat in §6.5.1/abstract
  sharpens from 'scorer-dependent' to 'not human-grounded'."
- **Outcome R (regex grounded) — write as fully as J.** "The human gold set agrees with the
  *regex* at κ_R=[..] and the judge at κ_J=[..] (Δ≥0.15 toward the regex). The regex tracks
  human judgment better; the independent judge is the outlier. Result 6 is revised: the Phi
  −0.048 decrease is *supported* by the human-grounded scorer, and the T35 sensitivity check
  over-weighted a weaker instrument. The over-refusal survivor stands as a regex-and-human
  supported benign-direction change. (Honest: this REVERSES the T35 'scorer-dependent'
  emphasis — report it with equal prominence; do not bury it.)"
- **Outcome T (tie / both weak).** "The human gold set agrees only weakly with both scorers
  (κ_R=[..], κ_J=[..]; gap < 0.15 or both < 0.40), so benign over-refusal on these contested
  responses is genuinely ambiguous even to a human. Result 6's 'scorer-dependent' wording
  stands, now human-annotated as irreducible rather than merely instrument-dependent."
- **Outcome X (invalid).** Abort fold-in for T36; commit only the gold prereg outcome + a
  PROJECT_LOG row; the existing Result 6 text is unchanged. (< 190/200 labeled, or draw
  deviates from §2.)
- Ch9 future-work: on J/R/T, mark "human-labelled XSTest refusal gold set" DONE (single
  annotator; second annotator remains the residual, mirroring T30).

## 3. T37 — LlamaGuard open-weight third judge (§6.12 Result-4 area)

After the TC1 job + `judge_pairwise_agreement.py --judge-a harmbench_cls --judge-b llamaguard`
and `--judge-a api_judge --judge-b llamaguard`:
- Add: "A third, open-weight judge (Llama-Guard-3-8B, revision-pinned via JUDGE_REVISION —
  the pin plumbing landed in `0ed19dd`) re-scored the [N] HarmBench generations, agreeing with
  the primary classifier at Cohen κ=[..]. This retires the versioned-API-judge reproducibility
  caveat: the cross-check no longer rests solely on a dated gpt-4o snapshot."
- **Construct-mismatch caveat, pre-written for the disagreement case:** "Llama-Guard answers a
  *looser* question ('is this response unsafe?') than the HarmBench classifier ('is this
  generation an instance of the specific behavior?'), so where it disagrees the divergence is
  expected and non-diagnostic; agreement is the informative direction. It is a reproducibility
  cross-check, not a second construct-validity oracle." (This caps both branches: agreement
  strengthens reproducibility; disagreement is explained by construct, not a real conflict.)

## 4. T39 — multi-seed extension to 5/5 pairs (§6.6.1)

After TC1 seeds 1–5 for mistral/phi + `sensitivity_analysis.py --sensitivity-root
results_sensitivity_512 --results-dir results_512`:
- §6.6.1: replace "three pairs (Qwen 1.7B, Qwen 4B, Llama)" → "all five pairs"; add per-seed
  ΔASR table rows for mistral_7b + phi4_mini and their "k/5 seeds significant" counts.
- Ch8 item 8 (multi-seed covers only 3 of 5 pairs) → rewrite to "the stochastic-decoding
  sensitivity arm now covers all five pairs" (limitation resolved).
- Expected (confirm, don't assume): the greedy deltas sit inside the seed ranges; the null is
  corroborated. If a new pair shows a sign-consistent significant seed majority, that is a
  truth-changing surprise → escalate before folding.

## 5. Claim-lock stubs (paste into verify_report_claims.py at fold-in; self-test each)

Pattern: `c.check(name, [<report-text snippet>], fn)` where fn loads the committed artifact
and `near()`-compares. **Mandatory self-test per check: perturb the artifact value in-memory
or the snippet → confirm exactly that check FAILs → restore.**

- **T38** vs `parser_strict_sensitivity.json`: qwen_2b MMLU strict −0.293 [−0.350,−0.237] sig;
  qwen_2b ARC strict −0.343 [−0.375,−0.311] sig; fallback fracs 0.487/0.523 (4bit) & 0.033/
  0.025 (base); leading-letter acc 0.750/0.799 (4bit). Bind the §6.4.1 correction snippet.
- **T36** vs `xstest_human_validation.json`: κ_R, κ_J (STRICT), and the derived outcome letter
  (recompute the precedence in the check, don't trust a stored letter alone).
- **T37** vs the llamaguard `judge_pairwise_agreement*.json`: the reported κ.
- **T39** vs updated `sensitivity_multiseed.json`: the new mistral/phi per-seed means +
  n_seeds_significant; re-assert the existing 3 pairs are byte-unchanged.

## 6. D42 sweep surface list (fold-in day)

All must move together for any truth-changing edit (the §6.4.1 correction IS truth-changing):
`build_fyp_report_v5.js` + `build_fyp_report_humanized.js`; `build_fyp_thesis_v4.js` +
`build_fyp_thesis_humanized.js`; `build_fyp_interim.js` + `build_fyp_interim_humanized.js`;
`fyp_submission/report_latex/final_thesis/mythesis.tex` (+ `tectonic` recompile + rebuild
`final_thesis_overleaf.zip`, 24-file structure); `docs/RESULTS_CARD.md`; `README.md`;
`CLAUDE.md` + `AGENTS.md` (keep in sync, same commit); `scripts/verify_report_claims.py`
(+ self-test); `docs/PROJECT_LOG.md` (§4 rows + tick T36–T39 in §2 + D46 outcome); `todo.md`;
memory `project_state.md`. Decks (`docs/*.html`) + `.github/assets` figure captions: the
§6.4.1 ARC claim is a status change, so grep them for "format asymmetry" / "ARC … not
subject" / the −0.009-as-immunity framing and refresh or add a data-snapshot notice.
Rebuild all 6 docx (`make report thesis interim report-humanized thesis-humanized
interim-humanized`); gates verify-claims / agent-check / pytest green before commit.
