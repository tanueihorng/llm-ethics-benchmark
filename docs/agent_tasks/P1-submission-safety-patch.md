# P1 submission-safety patch — verified external-audit findings (2026-07-14)

**Status:** PLANNED (Fable 5, verified against repo 2026-07-14). Executor: Opus 4.8.
**Scope decision:** this patch pulls the **truth-correcting** fixes forward, ahead of the
Phase-C evidence fold-in (which still waits on T36 labels + T37 LlamaGuard + T39 multiseed,
per D46). Rationale: every fix below is anchored on *already-committed* artifacts, so none
of it needs new evidence — while the current shipped report contains a demonstrably false
sentence (ARC "immune") and several stale/overclaimed passages. Phase C remains a separate
pass; `docs/agent_tasks/T36-T39-phaseC-draft.md` must be updated to mark what this patch
already did.

**Executor freedom:** every claim in this packet was verified by Fable 5 against the repo,
but you must re-verify each number against the committed artifact before writing it into a
builder, and you may correct this plan where it is wrong — log any deviation in the
PROJECT_LOG row. Do NOT trust this packet's numbers as a source; trust the artifacts.

---

## 0. Verification record (what Fable 5 confirmed, with anchors)

External audit findings, verdicts after independent verification:

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| P1.1 | ARC "immune / not subject to format asymmetry" is false | **CONFIRMED** | `build_fyp_report_v5.js` L823 ("not subject to this format asymmetry"), L1023 ("immune to this asymmetry"); **thesis** `build_fyp_thesis_v4.js` L299 ("immune to this asymmetry"); refuted by committed `results_512/analysis/parser_strict_sensitivity.json` (qwen_2b ARC strict Δ ≈ −0.343, fallback 52.3% NF4 vs 2.5% fp16; independently re-derived to 6 dp in the Phase-B verification) |
| P1.2 | Report stale vs executed T37/T39 | **CONFIRMED (deliberate deferral)** | L840 "Mistral and Phi remain greedy-only… still outstanding", L1026 "Extending… would complete", L927 & L1000 "LlamaGuard… future check". T37/T39 were computed (PROJECT_LOG, commit e2ef67a) but sidecars deferred to Phase C — so numbers must NOT be quoted; only the "unperformed" framing is fixed |
| P1.3 | κ "lower bound" / "population κ would sit higher" unjustified | **CONFIRMED** | L926, two clauses: truncation-direction stated as certain; population-κ direction asserted. Neither is identified by the design (truncation can also create spurious agreement; κ under enriched sampling is not monotone) |
| P1.4 | Causal language too strong | **CONFIRMED** | L328 "quantization is the only variable", L457 "therefore attributable… rather than", L991 "no plausible alternative explanation" + "resume logic prevents partial-run contamination". Conflicts with the report's own Appendix-G provenance caveat (2026-07-11 row) and the T33 audit record |
| P1.5 | Novelty/scale overclaims | **PARTIALLY CONFIRMED** | "almost always" L327+L369; "de facto" L369+L427 (defensible usage, hedge lightly or keep with citation); L800 "single largest predictor" is *already* scoped "observed in this study" and hedged MMLU-specific in the same sentence — tighten, don't rewrite; **"most studies" does NOT appear anywhere** (audit error, no action) |
| P1.6 | TC1 forbidden path advertised | **CONFIRMED (pre-existing T33)** | `docs/USER_GUIDE.md:51` advertises `fyp_cli.py cluster-submit`; Makefile `cluster-all` chains it; report Table 4.2 (L640) lists it without a TC1 caveat; `docs/TC1_CLUSTER_RUNBOOK.md:243` forbids it |
| P2.a | "11 discordant" wrong | **CONFIRMED** | artifact `results_512/analysis/multiple_comparisons.json`: qwen_2b xstest **b=2, c=8, n_discordant=10**. Report says 11 at L530, L749, L833. The quoted p=0.109 is exactly the n=10 McNemar value (2·P(Bin(10,.5)≤2)=0.109), so only the count is wrong |
| P2.b | Llama baseline OR 0.036 vs 0.032 | **CONFIRMED** | L879 says 0.036 = the retired v1 `summary.json` value; the v2 scorer of record (`summary.v2.json`) says **0.032**, which L887 correctly uses |
| P2.c | "339 tests / 26 files" stale | **CONFIRMED** | live: **382 tests, 29 test files** (pytest --collect-only; ls). Report: L406, L628, L693, L1438, L1549 + Appendix D distribution table. Thesis builder: 5 hits of 339/twenty-six. `docs/REPRODUCIBILITY.md` "339-test suite". README to be swept |
| P2.d | "classifier- and human-anchored" category error | **CONFIRMED** | L826 — the two capability survivors (Qwen MMLU, Llama ARC) are exact-match scored, not classifier/human-anchored. Intended meaning: "do not rest on the demoted regex" |
| P2.e | List of Figures and Tables omits Tables 6.2–6.4 | **CONFIRMED** | L341–359 lists only Tables 3.1–3.4, 4.1–4.2, 5.1–5.2, 6.1, D.1. Tables 6.2 (scale), 6.3 (judge), 6.4 (XSTest judge) exist but are unlisted |
| P2.f | TOC renders blank | **CONFIRMED (mechanism)** | builder uses docx-js `new TableOfContents(...)` (L337) — a Word *field*, empty until fields are updated; renders blank in non-Word converters |
| P2.g | Revision history stale | **CONFIRMED** | Appendix G top row is 2026-07-11 "(current)"; the report *content* now includes T35 (Result 6, Table 6.4, 2026-07-12) and later passes — no rows for them |
| P2.h | REPRODUCIBILITY.md `quant_method` claim false | **CONFIRMED** | actual raw.jsonl keys: benchmark, chat_templated, family, generation_config, model_alias, model_id, pair_id, prompt_id, prompt_text, quantized, response, score_fields, seed, size_b, timestamp — **no `quant_method`** |
| P2.i | Repro commands default to `results/` | **CONFIRMED** | REPRODUCIBILITY step 3 runs `scripts/judge_agreement.py` (argparse default `results`) and `scripts/precision_sweep_analysis.py` (default `ROOT/"results"`) with no `--results-dir`, yet claims byte-for-byte reproduction of `results_512/analysis/*` |
| P2.j | Resume trusts prompt IDs alone | **CONFIRMED (pre-existing T33)** | `run_quant_benchmark.py` ~L411: `get_processed_prompt_ids(raw_path)` — no seed/gen-config/model-identity fingerprint check on resume |
| P2.k | Pair validation permits >1 member / mixed model_id | **CONFIRMED** | `config_schema.py` L203–209 enforces only ≥1 baseline + ≥1 quantized; no model_id-equality or exactly-one check |
| P2.l | "significant" defined differently in code vs prose | **CONFIRMED (no numeric impact today)** | prose (L749): CI excludes 0 AND McNemar p<.05; `compare_quant_pairs.py` evidence_status docstrings: CI-only. No current contrast flips between the two definitions |
| P2.m | T36 sheet blank | **NOT A DEFECT** | user labeling pending (WS-1 done; awaiting labels) |
| — | 111 pages vs 40–65 guideline | **USER DECISION** | comprehensive report vs standalone thesis are distinct deliverables; not for the executor to resolve |

---

## WS-A — Report builder truth corrections (`scripts/build_fyp_report_v5.js`)

Work in this order; after all edits, `make report` once.

### A1. ARC format-asymmetry correction (the one truth-changing edit; triggers full D42 sweep)

Use the prepared framing in `docs/agent_tasks/T36-T39-phaseC-draft.md` §1/§1b (anchors already
verified there: report L823 §6.5 + L1023 Ch8; also §6.4.1 L820 "substantially MMLU-specific
(a subset-content sensitivity)" and §6.2 L800's 30:1 discussion). Required content:

- **Delete/replace** every claim that ARC is immune to / not subject to the format asymmetry
  (report L823, L1023; thesis L299 — thesis handled in WS-D2).
- **Replace with the bracket framing:** the answer-format asymmetry affects BOTH capability
  benchmarks for the smallest pair. Read exact values from
  `results_512/analysis/parser_strict_sensitivity.json` before writing (expected ballpark:
  qwen_2b lenient MMLU −0.090 → strict ≈ −0.293 [−0.350, −0.237]; lenient ARC −0.009 →
  strict ≈ −0.343 [−0.375, −0.311]; fallback shares ≈ MMLU 48.7% vs 3.3%, ARC 52.3% vs 2.5%).
- **Binding interpretation constraints** (from the Phase-C draft; do not weaken):
  1. Direction of the capability loss is parser-robust (negative under both protocols).
  2. Magnitude is protocol-dependent on BOTH benchmarks; report as a bracket
     [lenient, strict], never as "the true loss is X".
  3. ARC's near-zero lenient −0.009 is the lenient parser *salvaging* format-broken answers,
     not immunity; under strict scoring ARC's loss is comparable to (slightly larger than)
     MMLU's. ARC still corroborates the *direction*; it no longer corroborates the
     *magnitude-smallness*.
  4. This is measurement-dependence axis #3 (harm scorer → over-refusal scorer → capability
     parser); wire it into the report's central-thesis sentences where ARC-as-clean-anchor
     was used.
- **Sweep, don't spot-fix:** grep the builder for `immune`, `not subject`, `corroborat`,
  `MMLU-specific`, `subset-content`, `30:1`, `parser-inflated` and re-read each hit in
  context. Known hits needing rework beyond L823/L1023: L820 (§6.4.1 — "substantially
  MMLU-specific (a subset-content sensitivity)" must be re-scoped: the *lenient-parser
  gap pattern* is MMLU-specific, the *strict-parser loss* is not), L800 (§6.2 — the 30:1
  MMLU ratio survives as a lenient-protocol observation but gains a strict-parser caveat),
  and any Abstract/Ch10/RQ3 sentence leaning on ARC's small magnitude.
- **Claim-locks:** add `verify_report_claims.py` checks binding the strict-parser deltas,
  CIs and fallback percentages quoted in the report to
  `results_512/analysis/parser_strict_sensitivity.json`.

### A2. κ human-validation hedges (L926, §6.12 Result 5)

Two rewrites, keep everything else:
- Truncation clause: "the direction that makes the classifier's κ a lower bound, not the
  reverse" → state it as the *plausible predominant* direction, not a certainty (truncation
  can also manufacture spurious agreement when a harmful-looking prefix precedes a benign
  full response). E.g. "which plausibly inflates measured disagreement; the observed κ is
  therefore best read as conservative, though the direction is not strictly identified."
- Population clause: "over a representative population both scorers' agreement with the
  human would sit higher" → do not assert direction. The defensible claim: absolute κ values
  are computed on a deliberately disagreement-enriched slice and are **not population
  estimates**; what is robust is the *ordering and its size* on the contested cases
  (classifier ≫ regex) and the one-directional character of the regex's error. (The existing
  final sentence already says most of this — make it the only claim.)

### A3. Causal / internal-validity language

- L328 (Abstract): "so quantization is the only variable" → "so the paired contrast isolates
  the quantization step under the recorded matched-pair configuration" (or similar).
- L457 (§3.x): "Any observed delta … is therefore attributable to the quantization step
  itself rather than to …" → keep the design argument but scope it: attribution holds for
  checkpoint-provenance confounds by construction; environment/version provenance is a
  documented caveat (cross-reference the existing reproducibility caveat).
- L991 (Ch7): "There is no plausible alternative explanation…" → "the design leaves few
  plausible alternative explanations… beyond measurement/protocol choices, which Chapters
  6.5/6.12 show are material" — this is the honest tie-in: the study's OWN finding is that
  scorer/parser choice moves measured deltas, so "no alternative explanation" was
  self-contradictory. Also fix the same paragraph's "The resume logic prevents partial-run
  contamination" → state what it guarantees (metrics computed from a complete raw.jsonl with
  exactly the configured prompt count; per-record provenance fields carried) and what it
  does not (no config-fingerprint check on resume; see WS-C1 / limitation).
- "best practical isolation under the recorded matched-pair configuration" is the audit's
  suggested register — acceptable phrasing.

### A4. Novelty/scale wording

- L327 + L369 "almost always" → "typically" / "commonly".
- L369/L427 "de facto": keep at most one, hedged ("arguably the de facto…" or cite);
  executor's judgment.
- L800: tighten "making scale the single largest predictor of capability preservation under
  quantization observed in this study" → "the largest single capability contrast observed in
  this study (a two-pair, within-Qwen, MMLU-anchored comparison)" — the existing MMLU-specific
  hedge in the same sentence stays and now also gains the A1 strict-parser context.
- "most studies": confirmed absent — no action.

### A5. T37/T39 staleness (numbers must NOT appear)

Reword four spots so completed work is not called unperformed, without quoting any result:
- L840 (§6.6.1): "…Mistral and Phi remain greedy-only, so a full-matrix stochastic estimate
  is still outstanding" → "…the same multi-seed arm has since been executed for Mistral and
  Phi; those results are undergoing verification and will be folded into a future revision,
  so this section reports the three verified pairs."
- L1026 (Ch8 limitation): same adjustment ("run executed, verification pending" rather than
  "extending… would complete").
- L927 + L1000 (LlamaGuard "future check"): "…an independent open-weight guard (Llama Guard)
  cross-check has since been run; its verification and fold-in are pending, so this report
  carries the two-judge + human grounding only."
- Constraint: no deltas, no κ, no counts from T37/T39 anywhere (their sidecars are
  intentionally not committed yet; verify-claims must not reference them).

### A6. P2 numeric/wording fixes

- "11 discordant" → "10 discordant" at L530, L749, L833 (b=2, c=8; p=0.109 unchanged —
  re-derive once to confirm).
- L879: baseline Llama XSTest over-refusal 0.036 → **0.032** (v2 scorer of record; 0.036 is
  the retired v1 value).
- Test counts: 339→382, twenty-six→twenty-nine at L406, L628, L693, L1438, L1549; refresh the
  Appendix D distribution table from live `pytest --collect-only` output (regenerate the
  per-file counts; do not hand-guess).
- L826: "The two capability survivors are classifier- and human-anchored" → "The two
  capability survivors rest on deterministic exact-match scoring and involve no refusal
  scorer, whereas the over-refusal survivor rests on the demoted regex…".

### A7. Structure/format

- List of Figures and Tables (L341–359): add entries for Table 6.2, Table 6.3, Table 6.4
  (copy exact captions from the builder’s table caption strings).
- TOC: check whether the `Document` constructor sets `features: { updateFields: true }`;
  if not, add it so Word refreshes the TOC on open. Verify by opening/rendering. If a
  non-Word render path is used for submission PDFs, note in REPRODUCIBILITY or the log that
  fields must be updated in Word/LibreOffice before export. (A static TOC is the fallback
  option if updateFields proves unreliable — executor's call, but don't hand-maintain a
  static TOC silently: generate it from the same heading constants the builder already has.)
- Appendix G revision history: append new top rows — (i) 2026-07-12 T35 fold-in (Result 6 /
  Table 6.4 / scorer-dependence caveats; this row is currently missing), (ii) this patch,
  dated, describing the ARC correction + audit remediation. Move "(current)" to the new top
  row and mark the 2026-07-11 row superseded.

### A8. Table 4.2 cluster-submit row (ties to WS-B2)

Amend the row's description: "Submit the generated sbatch files (dev/local convenience;
**not used on TC1**, where head-node Python is disallowed — jobs are submitted with direct
`sbatch`, §5.x)."

## WS-B — Docs corrections

- **B1 `docs/REPRODUCIBILITY.md`:** (i) remove `quant_method` from the per-record field list
  (list only fields actually present; `quantized`, `generation_config`, `seed`, `pair_id`,
  `model_id`, `timestamp` are real); optionally note `quant_method` lives in config +
  summaries only. (ii) "339-test suite" → 382. (iii) Step-3 commands: add
  `--results-dir results_512` to `scripts/judge_agreement.py` and
  `scripts/precision_sweep_analysis.py` invocations — then **actually run all three step-3
  commands** and confirm they reproduce `results_512/analysis/*` without diffs before
  claiming byte-for-byte (if they don't, describe what they do reproduce).
- **B2 `docs/USER_GUIDE.md:51`:** annotate the `cluster-submit` example with the TC1
  prohibition + point to the direct-sbatch recipe (README L255 / TC1_CLUSTER_RUNBOOK).
  Optionally have `fyp_cli.py cluster-submit` print a one-line TC1 policy warning. Do NOT
  delete the subcommand (it's legitimate off-TC1); this narrows T33's item but doesn't
  close T33.

## WS-C — Optional code hardening (each: small, test-covered; descope with justification if risky)

- **C1 resume fingerprint (T33 item):** on `--resume` with an existing raw.jsonl, read the
  first record and fail-closed if `seed`, `model_id`, `model_alias`, or `generation_config`
  mismatch the current run (clear error telling the operator to use `--force_restart` or a
  different results dir). Add tests (mismatch → SystemExit/ValueError; match → resumes).
  This only guards *future* runs; no historical artifact changes.
- **C2 pair-validation tightening:** first check every shipped config
  (`configs/*.yaml`, incl. tc1_int8 and multiseed variants) for pairs with >2 members or
  mixed `model_id`. Then enforce: all members of a pair share `model_id` (safe), and —
  only if all shipped configs comply — exactly one baseline. Tests for both. If any shipped
  config legitimately violates (e.g. an INT8 third member sharing pair_id), enforce
  model_id-equality only and record why.
- **C3 significance-definition note:** cheapest fix is one disclosure sentence in §3.7/§3.8:
  the evidence_status layer uses the CI criterion; the significance markers in Tables 6.1/6.2
  use CI ∧ McNemar; at current data no contrast distinguishes them. (Aligning the code is
  out of scope pre-submission.)

## WS-D — D42 claim-surface sweep + gates + logs

1. **Grep the whole repo** for each corrected token before and after: `11 discordant`,
   `0.036` (XSTest context), `339`, `twenty-six`, `immune to this asymmetry`,
   `not subject to this format asymmetry`, `only variable`, `no plausible alternative`,
   `lower bound` (κ context), `almost always`. Classify each hit historical-vs-current
   per D42. Known current-facing hits outside the report builder: **thesis builder**
   (`scripts/build_fyp_thesis_v4.js` L299 "immune" + five 339/twenty-six hits — apply the
   same A1/A6 fixes there), README.md (at least one hit), REPRODUCIBILITY.md (WS-B1).
   Check RESULTS_CARD.md, CLAUDE.md/AGENTS.md (sync rule!), dashboard/data layer, decks
   (decks may carry a data-snapshot notice instead of edits, per D42).
2. `make report` **and** `make thesis`; if interim/humanized docs carry the false ARC claim,
   do NOT regenerate them — they are snapshot/alternate artifacts; instead flag to the user
   that the humanized alternates are now stale vs the masters.
3. **Gates:** extend `verify_report_claims.py` (A1 strict-parser locks; discordant n=10;
   llama base OR 0.032; consider a test-count lock reading pytest collect output or the
   policy note), update `configs/artifact_policy.yaml` `expected_test_count_note` if WS-C
   adds tests, add stale-text patterns for the retired claims (`immune to this asymmetry`,
   `11 discordant` — self-test that each fires on the pre-fix text). Run
   `make verify-claims`, `pytest -q`, `make agent-check`.
   Note: `agent-check` project-log discipline currently trips on the pre-existing untracked
   `docs/fyp-report-defense-deck-2026-07.html` — do not delete it; surface to the user.
4. **PROJECT_LOG:** §4 rows for each commit; §3 decision (Dxx): "truth-correcting audit
   fixes pulled forward ahead of the Phase-C evidence fold-in; ARC-immunity error corrected
   against the committed T38 artifact; causal/κ/novelty register standardized". §1 snapshot
   updated. Tick/annotate the T33 sub-items this patch narrows (cluster-submit docs, resume
   guard if C1 done) without closing T33.
5. **Update `docs/agent_tasks/T36-T39-phaseC-draft.md`:** mark the ARC correction as DONE by
   this patch; Phase C retains only the T36/T37/T39 evidence fold-ins + their claim-locks.
6. **Adversarial verification before commit** (established pattern): independent re-read of
   the rebuilt report sections vs artifacts (at minimum: every number newly written by this
   patch re-derived from its artifact by a fresh context), then commit + push.

## Explicitly OUT of scope (user decisions — do not act)

1. **Page count / which document is submitted** (111-page comprehensive report vs 40–65-page
   guideline vs standalone thesis) — the user must confirm the submission category with the
   supervisor.
2. **T36 labeling** — user's task (sheet ready; localStorage reset instructions in
   PROJECT_LOG/WS-1 notes).
3. **Phase C evidence fold-in** — waits on T36 + T37 + T39 per D46.
4. **Humanized alternates** — flag staleness, don't regenerate.
5. Deleting/committing the untracked defense deck.

## Done criteria

- No occurrence of the ARC-immunity claim anywhere current-facing; capability loss reported
  as a parser-bracket in report + thesis; claim-locks bind every new number.
- All P2 numerics fixed and locked; List of Tables complete; TOC updates on open (verified);
  revision history current.
- REPRODUCIBILITY step-3 commands actually reproduce what they claim (executed, not assumed).
- `make verify-claims`, `pytest -q` green; `agent-check` green except the pre-existing
  untracked-deck discipline trip (surfaced to user).
- PROJECT_LOG rows + decision recorded; Phase-C draft updated; pushed.
