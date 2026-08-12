# Handoff

Last refreshed: 2026-08-13 00:37:29 UTC+8 by Codex harness.

## Mode

Fresh-session recovery for Codex, Claude Code, or another coding agent.

## Objective

Start from repo truth, verify live state, and continue the FYP without relying on chat memory.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`.
- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md`.
- Run `python fyp_cli.py agent-status` for live state.
- Run `make agent-check` before finishing changes.
- Treat this file as a bridge, not as a replacement for `docs/PROJECT_LOG.md`.

## Current State

- Git: `## main...origin/main`.
- PROJECT_LOG last updated: 2026-08-13 (UTC+8) by Claude (Opus 5).
- Judge sidecars: 15 score files and 15 summary files.
- Report artifact: docs/FYP_Report_2026-07-25_v6.docx modified 2026-08-02 02:08:35 +08.
- Immutable manifest: results/raw_artifact_manifest.sha256 modified 2026-07-19 11:46:32 +08.

## Verification To Run

```bash
python fyp_cli.py agent-status
make agent-check
python scripts/generate_handoff.py
python scripts/generate_agent_dashboard.py
```

## Harness Check Summary

- Not run during this handoff generation.

## Risks / Things To Distrust

- Stale prose can survive in docs, report appendices, and historical changelog rows.
- Do not trust old v2-headline claims over `results/analysis/judge_agreement.{json,csv}`.
- Do not mutate `raw.jsonl` or `summary.json`; new scoring must use derived sidecars.
- Do not print raw HarmBench prompt or response text in handoffs, diagnostics, or chat.

## Next Actions

1. Next project action from PROJECT_LOG.md: - [ ] **T53. Submit the interim report to Dr. Zhang — DUE Mon 24 Aug 2026 (CCDS deadline, Sem 1 Wk 3).** School-set date, recorded 2026-08-10 from the CCDS FYP timeline; see the deadline table in §1. The deliverable exists and is current: `docs/FYP_Interim_2026-07-10.docx` (registry-backed; LaTeX mirror in `fyp_submission/report_latex/interim_report/` compiles clean, named PDF copy byte-matches). **The artefact to send is now `docs/rewritten_fyp_interim_v1.docx`** (built 2026-08-10 from `scripts/build_rewritten_fyp_interim_v1.py`; same body as the submission candidate, titled "Interim Report"), which supersedes `docs/FYP_Interim_2026-07-10.docx` for this purpose and sidesteps T56. **This is a send action, not a writing action** — decide docx vs PDF (if PDF, open in Word and update fields first so the TOC populates) and whether the July email (T1) carries it or a separate submission message does. **Ordering constraint:** the drafted August email (T1b in `todo.md`) opens by asserting the interim report has been submitted, so its send gate opens only after T53 is done; the July email (T1) should go first or be folded in. User-owned action.
2. If editing report-worthy content, edit `scripts/build_fyp_report_v5.js` and run `make report`.
3. Regenerate this handoff and the dashboard after meaningful harness or state changes.

## Privacy / Artifact Guardrails

- Use IDs, counts, labels, aggregate metrics, and redacted sidecars only.
- Preserve `raw.jsonl` and `summary.json` as TC1-original artifacts.
- Keep `docs/PROJECT_LOG.md` as the durable decision and changelog record.
