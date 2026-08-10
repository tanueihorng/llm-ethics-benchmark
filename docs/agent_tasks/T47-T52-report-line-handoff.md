# Handoff — the FYP report line (T47–T52)

## Mode

Fresh-session recovery.

## Objective

Decide which report is submitted, finish the two structural items on it, and freeze.
No experimental work remains; this is all document work.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md` (or `CLAUDE.md` — identical below the header).
- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md`, §2 open actions and the §4
  changelog rows dated **2026-07-25 through 2026-07-28**. That is ~15 rows and it is the
  real record; this packet is only a bridge.
- Decisions **D56** (claim-to-source fitness) and **D57** (batch for reading, never for
  judgement) in §3 are the two rules this work established. Follow them.
- Run `git status -sb` for live Git state. Nothing in this work has been committed.

## Current State

Three report candidates exist. All three build and pass the harness.

| File | Words | Status |
|---|---|---|
| `docs/FYP_Report_2026-07-01_v5.docx` | ~34k body | **Frozen, byte-identical to HEAD.** Deliberately untouched all session for side-by-side comparison. Still carries ~19 of the original 26 claim defects. |
| `docs/FYP_Report_2026-07-25_v6.docx` | ~45k | Corpus-voice rewrite of v5, fully remediated, full lock coverage, evidence trail in Appendix I (~8.4k words, **zero tables**). |
| `docs/rewritten_fyp_report_v1.docx` | ~16k | Independent rewrite by Codex. Corpus-shaped, numerically verified, entailment-audited. **Currently the strongest candidate.** |

Builders: `scripts/build_fyp_report_v5.js`, `scripts/build_fyp_report_v6.js`,
`scripts/build_rewritten_fyp_report_v1.py` (Python — note the different language).

**v6 is registered but NOT canonical.** `make report`, `CLAUDE.md` and `AGENTS.md` still
name v5. That swap touches the build/submission path and was deliberately left undone.

## What Was Verified From The Repo

Everything below was checked directly, not taken from a prior agent's report.

- **Numeric fidelity** of `rewritten_fyp_report_v1`: every results-table cell cross-checked
  against `generated/claim_registry.json` and `results_512/analysis/capability_mde.csv` —
  **0 mismatches**, re-run after every edit round.
- **Entailment coverage is closed**: all 85 citation-bearing prose sentences and all 8
  citation-bearing table rows checked. Inventory persisted at
  `docs/audits/2026-07-28_rewrite_fitness/` (`COVERAGE.md`, `claims.json`,
  `references.json`, `table_rows.json`, `table_row_findings.json`).
- **8 claim-to-source defects found and fixed** in that document (3 prose, 5 table cells),
  each re-verified against the primary paper before editing.
- **Two new analyses** added to the project and used by both v6 and the rewrite:
  `scripts/capability_equivalence.py` → `results_512/analysis/capability_equivalence.{json,csv}`
  and `capability_mde.csv`. TOST equivalence: all ten INT8 capability contrasts equivalent
  to fp16 at ±0.05. Capability MDE: only **2 of 10** contrasts clear their own detection
  floor, and they are exactly the two BH survivors.
- Harness at time of writing: **319/319 surface checks, 514 tests, 9/9 agent-check gates**,
  v5 byte-identical to HEAD.

## Verification To Run First

```bash
git status -sb
make verify-claims        # expect 319/319
make agent-check          # expect 9/9 including the epistemic-hedge gate
git status --porcelain -- scripts/build_fyp_report_v5.js docs/FYP_Report_2026-07-01_v5.docx
```

The last command must print **nothing**. If v5 has changed, something has gone wrong —
restore it with `git checkout HEAD -- <path>` before doing anything else.

## Risks / Things To Distrust

1. **A green harness has been a poor predictor of semantic correctness here — four times.**
   Four independent audit rounds each found real defects in text the previous round had
   just cleared, and the `epistemic_hedge` gate once cleared text it had a pattern for
   (line-level `allow` suppressed it; now span-scoped). Do not freeze on a green gate.
2. **Presence ≠ fitness.** A passing citation check proves references exist and are ordered.
   It cannot show a source supports the sentence citing it. That distinction produced 26
   defects in v6 and 8 more in the rewrite.
3. **Concurrent agents.** Codex has been rebuilding `rewritten_fyp_report_v1.docx`
   independently. A rebuild from a stale builder would revert the 5 table-cell corrections
   (Kharinaev/HarmLevelBench `Scorer` and `Over-refusal`/`Capability`, HarmBench boundary).
   Verify those cells survive any rebuild — `tests/test_rewritten_report_v1.py` pins the
   prose fixes but **not** the table cells.
4. **Do not trust agent claims about file contents.** Two separate deep agents this session
   misreported which builders contained a string. Grep it yourself.
5. Pagination ("56 A4 pages") has never been verified by any check here — it rests on
   visual inspection only.

## Next Actions

1. **T52 — pick the line.** Recommendation: `rewritten_fyp_report_v1`. It is inside corpus
   norms on 12 of 13 measured dimensions (lit review 23% of body vs v6's ~7%; 9.0
   citations per 1,000 words vs a corpus median of 7.1). The single argument for v6 is that
   it carries the full evidence trail *inside* the document rather than citing the repo.
   Once chosen, archive the other two and stop maintaining both.
2. **T49 — Appendix I tables.** Only if v6 wins: it is ~8.4k words of relocated prose with
   zero tables, against a "compact appendix" intent. Convert I.1 / I.3 / I.10 / I.12.
   Tooling is in `scripts/report_tools/` (see its README — the gate battery, and why G3
   must run against the RENDERED docx, not the builder source).
3. **T50 — freeze checklist.** Open the chosen docx in Word, `Ctrl+A` then `F9`, **confirm
   the Table of Contents populates**, then export the PDF. Both v6 and the rewrite use a
   real `TOC` field, which is blank in every non-Word renderer. Then freeze the builder
   hash and run one final enumeration pass.
4. **T51 — method, if another audit is run.** Enumerate, do not pattern-match. Batch the
   *finding* stage (amortises reading); never batch the *verification* stage — measured
   4% false-negative rate when judgement was batched (D57). The known blind spots that
   remain are in `docs/audits/2026-07-27_claim_sweep/completeness_critic.md`: claims carried
   by figure captions, and claims split across two sections where neither half is defective
   alone.

## Do Not Touch / Do Not Revert

- `scripts/build_fyp_report_v5.js` and `docs/FYP_Report_2026-07-01_v5.docx` — frozen for
  comparison at the student's explicit request.
- `results/` and `results_512/` `raw.jsonl` / `summary.json` — immutable TC1 originals.
- The `epistemic_hedge` patterns in `configs/artifact_policy.yaml`. **Clear a hit by
  rewording or by an `allow` entry that quotes the exempted span — never by weakening a
  pattern.** Every pattern traces to a real audit finding.
- `scripts/report_tools/renumber_refs.py`'s references wrapper. Emitting bare strings there
  silently shipped v6 with **no bibliography at all** for an entire session while every
  source-level citation gate stayed green.

## Privacy / Artifact Guardrails

No raw HarmBench prompt or response text in any document, log, or message. Use IDs, counts,
labels and the redacted `scores.judge.*` sidecars. Raw outputs and v2 sidecars are never
modified.
