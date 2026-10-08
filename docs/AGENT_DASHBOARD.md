# Agent Dashboard

Generated: 2026-10-01 13:27:08 UTC+8

## Live State

- Git: `## main...origin/main [ahead 1]`
- Changed files: 64
- PROJECT_LOG: 2026-10-01 13:27 (UTC+8) by Codex
- Report: 2026-09-27 16:44:21 +08 (1397734 bytes)
- Handoff: 2026-10-01 13:27:07 +08
- Immutable manifest: 2026-07-19 11:46:32 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

- T80 tables/figures follow-up - 2026-09-28 09:56, supplied Downloads copy (`d88b2f66…`).
- T80. Resolve selected final-report audit findings before submission (opened 2026-09-27).
- T80 progress - 2026-09-27 14:31 (student-directed fixes, fresh audit 2026-09-27 02:44, IDs A01-A15).
- T81. Tighten the T73 post-hoc gate to paragraph scope (opened 2026-09-27, D96).
- T79. Set up a standing off-machine backup for the local-only report and presentation sources (opened 2026-09-26, D94).
- T74. Validate Jev for report auditing on a separately verified audit set (opened 2026-09-20, D86).
- T75. Fold the validated Jev studies into the selected final-report line (base chosen in D95; opened 2026-09-20, D86/D87).
- T71. Migrate the rewritten-line LaTeX mirrors to the selected final-report base (base-choice portion complete; opened 2026-08-25, D75).
- T68. Optional: acquire local copies of the seven statistical-methods classics cited by the interim progress docx, and clean the fake PDFs in `tmp/source_audit_external/`.
- T66. Extend the `epistemic_hedge` scan with a certainty-verb pattern (from the 2026-08-22 v3 audit).
- T73. Reconcile post-hoc equivalence disclosure across the historical report inventory (opened 2026-09-07; counts below are the September snapshot).
- T61. Capture the TC1 conda environment once, so "the cluster environment is fixed" becomes a record rather than an assertion (opened 2026-08-11).
- T3. Run `MyTCinfo` on TC1.
- T15. Submit the final report - DUE Mon 19 Oct 2026 (CCDS deadline, Sem 1 Wk 10).
- T54. Submit the amended final report + deposit to DR-NTU - DUE Fri 13 Nov 2026 (CCDS deadline, Sem 1 Wk 13).
- T55. Oral presentation - Fri 4 / Mon 7 / Tue 8 Dec 2026 (one half-hour slot: 20 min presentation + 10 min Q&A).
- T63. No gate ties the rewrite-line LaTeX mirrors to their builder.
- T34. Upload the 512 LaTeX bundles to Overleaf.
- T33. Residual pre-submission code hardening from the 2026-07-10 audit (original P1 findings remediated 2026-07-11/14/15; not a current scientific/report blocker).
- T19.
- T42. External venue plan (post-FYP workshop paper / preprint) - SCOPED 2026-07-19, blocked on T15 + supervisor conversation.
- T72. `build_rewritten_fyp_interim_progress.py::build_interim` no longer runs (found 2026-09-02).
- T24. Minor `sensitivity_analysis.py` provenance hardening (audit nit, 2026-06-14).
Full task details and superseding decisions: `docs/PROJECT_LOG.md` section 2.

## Recent Decision IDs

- D102. 2026-10-01 - Staleness updates preserve the existing design
- D101. 2026-09-27 - Heading numbers are native Word list numbering; citations stay typed
- D100. 2026-09-27 - The final report's navigation and numbering are native Word fields
- D99. 2026-09-27 - Keep both report-audit skills: the staged version and the original as `-old`
- D98. 2026-09-27 - Restore the original report-audit skill text
- D97. 2026-09-27 - Every audit is a fresh, independent audit

## Analysis Artifacts

- `results_512/analysis/judge_agreement.json`: 2026-08-10 12:28:56 +08
- `results_512/analysis/judge_agreement.csv`: 2026-08-10 12:28:56 +08
- `results_512/analysis/multiple_comparisons.json`: 2026-08-10 12:32:32 +08
- `results_512/analysis/multiple_comparisons_judge_strict.json`: 2026-07-19 11:16:52 +08

## Harness Checks

- Not run for this dashboard.

## Commands

```bash
python fyp_cli.py agent-status
make agent-check
python scripts/generate_handoff.py
python scripts/generate_tc1_checklist.py
```
