# T41 — Phase-C audit remediation packet (for Opus 4.8)

**Prepared:** 2026-07-19 (Fable 5, verification pass). **Status: plan only — no content edited yet.**
**Trigger:** external post-Phase-C audit (user-pasted, 2026-07-19). Per project rule, every finding
was independently re-verified against primary artifacts before entering this packet.
**Baseline:** `9b572a2` (all gates green: verify-claims 85/85, surfaces 264/264, pytest 435, agent-check 8/8).

---

## 0. Verification table (Fable 5 — every finding checked against sources)

| # | Audit finding | Verdict | Evidence checked |
|---|---|---|---|
| 1 | T36 draw violated the locked prereg (≤10 (alias,prompt) exclusions; deviation ⇒ Outcome X) | **CONFIRMED, with nuance** | Prereg §2 says "the specific (alias, prompt) items … ≤10 items"; §5 says a §2 draw deviation ⇒ X. The execution packet (`T36-T39-hardening-batch.md` WS-A) fixed the list as ~5 prompts × **both pair members** (≈10, compliant). The implementation (`xstest_human_label_audit.py` `EXCLUDE_PROMPT_IDS`, applied per-alias in the draw filter) excludes the 5 prompt_ids across **all ten** aliases (≈50 pool items). The code comment "Excluded across BOTH pair members" is **inaccurate about the code's own behavior**. Deviation is real; it was baked into committed code **before drawing and before labeling**, and is blindness-conservative in direction. κ arithmetic unaffected. |
| 2 | Human audit not independently reproducible (aggregates only; alias/prompt_id discarded at scoring) | **CONFIRMED as a gap, with nuance** | `xstest_human_validation.json` is aggregates-only — but that was itself pre-specified (prereg §4: "aggregates and counts only"). So this is an *enhancement* (add a redacted per-item trail), not a violation fix. The gitignored `human_labels/xstest_label_sheet.csv` (with labels) exists locally on the Mac. |
| 3 | Shipped report still says the multi-seed arm covers "three of the five pairs" | **CONFIRMED** | `build_fyp_report_v5.js:992` + `build_fyp_report_humanized.js:974` (the Ch7 practical-takeaway paragraph). Both docx were regenerated after, so the delivered reports carry it. Thesis/interim builders and both `.tex` are clean of this phrase (grep verified). |
| 4 | RESULTS_CARD says "no pair is sign-consistent" — false | **CONFIRMED** | `docs/RESULTS_CARD.md:99`. Artifact marks qwen_2b, qwen_4b, llama `sign_consistent: true`. (Fable's own part-B edit introduced this — the exact error the subagents were warned against.) Correct claim: "no pair is robustly positive; the 4 individually significant seeds split 1 increase vs 3 decreases". |
| 5 | ≈0.06 treated as a universal detection floor; "below 0.06 ⇒ noise" invalid | **CONFIRMED (framing, not numbers)** | `multiple_comparisons.json` per-pair `mde_delta_asr`: 0.0443–0.0864 (representative 0.0594). Phi seed 5 ΔASR −0.055 IS individually significant (p 0.0433) yet < 0.06 — so "sub-MDE ⇒ noise" is not a valid inference. §6.6.1's greedy-vs-seed "noise" conclusion is fine on its own terms (gaps ~0.02–0.03, sign-unstable, zero-straddling ranges) but must not lean on a universal 0.06 floor. |
| 6 | LlamaGuard sidecars carry the HarmBench-classifier `scorer_version` and label the metric `attack_success_rate` | **CONFIRMED** | `summary.judge.llamaguard.json` AND every `scores.judge.llamaguard.jsonl` line carry `judge_harmbench_cls_v1_2026-05-28`. Numbers correct; provenance/construct labels wrong (shared-runner stamp). |
| 7 | "fully reproducible" LlamaGuard re-scoring overstated | **CONFIRMED** | `build_fyp_report_v5.js:1526` ("whose fully reproducible re-scoring — unlike the versioned API judge"). The phrase family also appears in the humanized report, thesis v4/humanized, and both `.tex` — each occurrence's context must be checked (some may be about the pipeline generally, which is a different claim). |
| 8 | Field guide substantially stale | **CONFIRMED** | `project_field_guide.html:371` (three-pair multi-seed), `:415` ("no human gold set exists yet … T30 closes" — stale since 2026-07-09, worse now), no LlamaGuard. Registered `lifecycle: current` in `claim_surfaces.yaml:103`, so it MUST be refreshed (or formally reclassified). |
| 9 | PROJECT_LOG §2 contradictions; T38 unticked; HANDOFF/AGENT_DASHBOARD stale | **CONFIRMED** | T37/T39 entries ticked but retain "Remaining: …" clauses inside the preserved original text; T38 (`PROJECT_LOG.md:78`) unticked despite the committed 2026-07-13 execution; HANDOFF.md last refreshed 07-16; AGENT_DASHBOARD stale. |

Also verified green (audit agrees): D51 recomputation, T37 κ/McNemar, T39 statistics, T36 arithmetic,
immutability, redaction, all machine gates. **The science is not in question; the protocol status,
audit trail, and several prose surfaces are.**

---

## 1. USER DECISIONS REQUIRED (ask before executing WS-1; the rest can proceed)

**D-1 (WS-1): how to treat the T36 prereg deviation.** Options:
- **(A) Deviation-disclosure — RECOMMENDED.** Keep the analysis and the substantive conclusion
  (judge aligns better with the annotator), but stop presenting it as a *clean* "pre-registered
  Outcome J". Every surface that says "pre-registered Outcome J" changes to Outcome J **with one
  disclosed protocol deviation**: the blindness exclusion was applied prompt-wide (5 prompts ×
  all 10 aliases, ≈50 candidate items of 2,450) instead of the pre-specified ≤10 (alias,prompt)
  items; the widening was fixed in committed code before drawing and before labeling and is
  blindness-conservative; under the prereg's strict letter (§5) any §2 draw deviation is
  classified Outcome X, so the confirmatory badge is withdrawn and the result is reported as a
  disclosed-single-deviation validation. This is standard practice (CONSORT-style deviation
  reporting) and is the honest option that preserves the evidence.
- **(B) Counterfactual-draw sensitivity — RECOMMENDED AS ADDITION to A.** Re-run the draw code
  with the packet-literal exclusion (the 5 prompts × their two originating pair members only) and
  report the overlap between the counterfactual 200 and the labeled 200 (expected high). Commit
  the overlap number in the validation artifact. Quantifies the deviation's footprint without
  unblinding anything. (Do NOT label new items — that would require unblind re-contact with the
  annotator and a new protocol.)
- **(C) Full redraw + relabel** under a corrected draw. Clean prereg status, but costs the user a
  new ~200-item blind labeling session and discards none of the deviation issues retroactively
  (the first draw still happened). Only if the user wants the confirmatory badge at any cost.

**D-2 (WS-5): field guide** — refresh it to current (recommended; it is registered
`lifecycle: current`) or formally reclassify it as a dated snapshot with a data-notice banner
and change its registration. Refresh is ~10 focused edits; recommend refresh.

---

## 2. Workstreams (ordered; WS-1/2/3 are the submission blockers)

### WS-1 — T36 protocol status (after user picks D-1; assume A+B unless told otherwise)
1. `docs/XSTEST_GOLD_PREREG.md` is **append-only**: add a dated §8 "Protocol deviation record"
   stating exactly what §0/#1 above says (what deviated, when it was fixed in code, direction,
   footprint, and the resulting classification under §5's letter). Do not edit §2/§5.
2. Fix the inaccurate code comment in `scripts/xstest_human_label_audit.py` (say what the code
   actually does: prompt-wide exclusion across all aliases; reference the deviation record).
3. Run the (B) counterfactual-draw overlap; emit the overlap stats into the validation artifact
   (WS-2's regeneration) and cite it in the deviation disclosure.
4. Sweep EVERY surface for "pre-registered Outcome J" / "prereg" claims about T36 (D42 rule: grep
   the claim, not filenames — builders ×6, both `.tex`, README, RESULTS_CARD, CLAUDE/AGENTS,
   PROJECT_LOG §1, memory is Fable's job not yours, decks if they carry it) and apply the agreed
   wording. Keep the substantive Outcome-J-consistent conclusion; change only the protocol badge.
5. Update the T36 lock (`verify_report_claims.py t36_outcome`): it currently recomputes the
   precedence and asserts "pre-registered" prose implicitly via snippets — adjust snippets to the
   new wording and ADD an assertion that the deviation disclosure sentence is present. Perturb-test.

### WS-2 — Redacted per-item audit trail (reproducibility)
1. Extend `xstest_human_label_audit.py --score` to also emit
   `results_512/analysis/xstest_human_validation_items.jsonl`: one line per labeled item with
   `model_alias, prompt_id, stratum (disagreement|agreement), human_label (3-class),
   regex_refusal, judge_strict, judge_broad` — **no prompt/response text**. Plus extend the
   aggregate JSON with draw metadata: excluded prompt_ids, priority/other alias weighting,
   n per stratum, sheet-CSV sha256, protocol/prereg version tag, and the WS-1 overlap stats.
2. Regenerate both artifacts from the local `human_labels/` sheet (verify the sheet hash matches
   the one recorded in the 2026-07-18 log rows before trusting it).
3. `configs/artifact_policy.yaml`: add the new sidecar to redaction-scanned globs; self-test by
   planting text in a field and asserting the scan fires.
4. Add a lock: recompute the committed aggregate κs/counts FROM the per-item sidecar and assert
   they equal `xstest_human_validation.json` (this is the independent-reproduction path the audit
   asked for). Perturb-test both directions (mutate a sidecar line → FAIL; mutate aggregate → FAIL).
5. Tests in `tests/test_xstest_human_audit.py`: sidecar redaction, aggregate↔sidecar consistency,
   deterministic regeneration.

### WS-3 — Report contradictions and framing (the delivered-docx blockers)
1. **"three of the five pairs"** — `build_fyp_report_v5.js:992` and `build_fyp_report_humanized.js:974`:
   rewrite to five pairs, folding in the honest cross-family clause (greedy-outside-narrow-range
   for Mistral/Phi, pointer to §6.6.1). Keep "no pair is seed-robust in the harmful direction"
   (true). Then grep ALL current builders + both `.tex` for `three of the five|three-pair` in
   multi-seed context to confirm no other site (Fable found none, verify anyway).
2. **MDE framing** — in §6.6.1 (both report builders; thesis §6.3 mirror if it copied the clause)
   replace "every quantity involved is below the study's minimum detectable effect (≈0.06)" with
   pair-specific truth: per-pair MDEs span 0.044–0.086 (`multiple_comparisons.json
   power_analysis.per_pair_harmbench_asr[].mde_delta_asr`); state that the greedy-vs-seed
   discrepancies (≤0.03) are small, sign-unstable, and inside zero-straddling seed ranges — and
   do NOT use "below MDE" as a noise proof (Phi's −0.055 seed is individually significant and is
   already disclosed as such). Anywhere else "≈0.06" is used as a universal floor, scope it as
   the representative/median-pair MDE.
3. **"fully reproducible" (LlamaGuard sentence)** — `build_fyp_report_v5.js:1526` + the same
   sentence in the humanized report, thesis v4/humanized, and both `.tex`: change to
   "revision-pinned, open-weight and locally rerunnable" (audit's suggested register). Check each
   other `fully reproducible` hit's context first — occurrences about the analysis pipeline
   replay are a different (already-hedged) claim; touch only the LlamaGuard-scoring ones.
   Archived builders (v2/v3/v4, thesis v2/v3) are historical — do not touch.
4. **RESULTS_CARD.md:99** — replace "no pair is sign-consistent" with "no pair is robustly
   positive (three pairs are sign-consistent — Qwen 1.7B non-negative, Qwen 4B all positive,
   Llama all negative); the 4 individually significant seeds split 1 increase vs 3 decreases".
5. Rebuild everything: `make report report-humanized thesis thesis-humanized interim
   interim-humanized`, recompile both PDFs (tectonic), refresh named copies + Overleaf zips.

### WS-4 — LlamaGuard provenance metadata
1. Fix the runner (`scripts/run_judge_validation.py` / shared scoring path) so `scorer_version`
   is backend-specific (e.g. `judge_llamaguard_v1_2026-07-13`) for future runs.
2. The committed sidecars: apply a **documented metadata correction** — they are derived
   sidecars, not raw artifacts, so correction is permitted with logging. Rewrite
   `scorer_version` in the 15 `summary.judge.llamaguard.json` + the per-line field in the 15
   `scores.judge.llamaguard.jsonl` via a one-shot script; leave every other byte identical
   (assert: only that field differs, `judge_harmful`/IDs untouched — write the assertion into
   the correction script and print the per-file diff counts). Add a `provenance_note` field to
   the summaries recording the correction date + reason. Log as a §3 decision (metadata-only
   correction of derived sidecars; raw artifacts untouched).
3. Construct label: keep the `attack_success_rate` field name (schema compatibility) but add
   `"metric_construct": "llamaguard_unsafe_content_rate"` to the summary metadata, and make sure
   the report's construct caveat (already present) is what carries the reader-facing truth.
4. Soften the report's provenance claim per WS-3.3; if any surface claims full environment
   capture for the LlamaGuard run, scope it to "revision + precision pinned; job id recorded".
5. Re-run the T37 lock (it pins `judge_revision`, `parse_error_count`, `num_samples` — all
   unaffected) and extend it to assert the corrected `scorer_version` prefix so regression fires.

### WS-5 — Stale documentation layer
1. `docs/project_field_guide.html` (assuming D-2 = refresh): five-pair multi-seed (line ~371
   region), the human-gold-set paragraph (line ~415: both T30 HarmBench and T36 XSTest gold sets
   now EXIST — rewrite the "residual" story to T30b second annotator), add the LlamaGuard third
   judge to the judge-stack description, and sweep the file for "three pairs", "no human",
   "pending" in Phase-C-adjacent contexts.
2. `docs/PROJECT_LOG.md` §2: trim the contradictory retained clauses inside the ticked T37/T39
   entries (keep provenance pointers, delete the stale "Remaining:" sentences); tick **T38** with
   a completion note pointing at the 2026-07-13 00:16 row and the committed
   `parser_strict_sensitivity.json`.
3. Regenerate `docs/HANDOFF.md` (`make agent-handoff`) and `docs/AGENT_DASHBOARD.md`
   (`make agent-dashboard`).

### WS-6 — Guards, gates, ledger
1. New stale-text patterns in `configs/artifact_policy.yaml` (self-test each fires):
   multi-seed "three of the five pairs" / "covering three of the five"; "no pair is
   sign-consistent"; the LlamaGuard "fully reproducible re-scoring" phrasing; unscoped clean
   "pre-registered Outcome J" (pattern must allow the deviation-disclosed wording — build the
   negative-lookahead the same way the "not immune" guard did).
2. Full gates: `make verify-claims` (expect 85 + WS-1/WS-2 lock changes), `python3
   scripts/verify_claim_surfaces.py` (264+, 0 FAIL), `pytest`, `make agent-check`.
3. PROJECT_LOG: one §4 row per landed commit; a §3 decision row for the T36 deviation treatment
   (user's D-1 choice) and the WS-4 metadata correction.
4. Suggested commit slicing: (1) WS-1+WS-2 (T36 protocol + audit trail), (2) WS-3+WS-6 report
   fixes + guards, (3) WS-4 provenance, (4) WS-5 docs. Push after all green.

---

## 3. Binding constraints (do not violate)

- **Raw artifacts and primary-scorer sidecars are immutable.** WS-4's correction touches ONLY the
  `scorer_version` field (+ added metadata keys) of the `*.judge.llamaguard.*` derived sidecars,
  with a printed proof that nothing else changed.
- **The prereg file is append-only.** Deviations are recorded, never rewritten.
- **Do not soften or harden the substantive T36 conclusion.** The κ numbers, the "most plausibly
  a regex measurement artifact" carry (prereg Outcome-J clause), and the scorer-invariant RQ2
  headline stay as-is; only the *protocol badge* changes per D-1.
- **Never claim greedy-in-range for all five pairs; never claim "no pair sign-consistent".**
- **Every new/changed lock gets a perturbation self-test (mutate → FAIL → restore → PASS) before
  commit.**
- **D42 sweep discipline:** grep for the CLAIMS ("three of the five", "fully reproducible",
  "pre-registered Outcome J", "no pair is sign-consistent", "0.06"), not just the cited files.
- The gitignored `human_labels/` sheet is the only source of per-item labels — verify its
  recorded hash before regenerating anything from it, and never commit its text columns.

## 4. Acceptance

- All four blocker fixes verifiable: (a) deviation disclosed on every T36 surface + prereg §8;
  (b) per-item sidecar committed + aggregate↔sidecar lock green; (c) no current surface says the
  multi-seed arm covers three pairs; (d) delivered docx/PDFs regenerated after all edits.
- All gates green; every new guard/lock demonstrated to fire on perturbation.
- PROJECT_LOG rows + decisions written; no contradiction left in §2.
