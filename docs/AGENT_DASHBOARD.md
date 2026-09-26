# Agent Dashboard

Generated: 2026-09-26 18:10:24 UTC+8

## Live State

- Git: `## main...origin/main [ahead 5]`
- Changed files: 6
- PROJECT_LOG: 2026-09-26 18:06 (UTC+8) by Claude Code
- Report: 2026-09-26 18:03:37 +08 (1397734 bytes)
- Handoff: 2026-09-26 18:10:24 +08
- Immutable manifest: 2026-07-19 11:46:32 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

- T79. Set up a standing off-machine backup for the local-only report and presentation sources (opened 2026-09-26, D94).
- T74. Validate Jev for report auditing on a separately verified audit set (opened 2026-09-20, D86).
- T75. Fold the validated Jev studies into the selected final-report line after T71 chooses the base (opened 2026-09-20, D86/D87).
- T71. Decide the final-report base line and regenerate the rewritten-line LaTeX mirrors accordingly (opened 2026-08-25, D75).
- T68. Optional: acquire local copies of the seven statistical-methods classics cited by the interim progress docx, and clean the fake PDFs in `tmp/source_audit_external/`.
- T66. Extend the `epistemic_hedge` scan with a certainty-verb pattern (from the 2026-08-22 v3 audit).
- T53. Submit the interim report to Dr. Zhang - DUE Mon 24 Aug 2026 (CCDS deadline, Sem 1 Wk 3).
- T1. Send the staged July follow-up to Dr. Zhang.
- T73. The INT8 equivalence claim is undisclosed as post hoc in the SELECTED submission copy and 16 other documents (opened 2026-09-07).
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

- D94. 2026-09-26 - Keep reports and presentations local-only, going forward
- D93. 2026-09-26 - Fold the Chapter 1 revision and two foundational quantization sources into interim v3
- D92. 2026-09-26 - Restructure the interim literature review fundamentals-first as a separate v3 copy
- D91. 2026-09-22 - Combined Jev quantization-effect judge fails; atomic subconstructs remain candidates
- D90. 2026-09-22 - Validate quantization effects with atomic paired Choices
- D89. 2026-09-22 - Do not expand the failed Jev response-quality rubric

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
