# Handoff

Last refreshed: 2026-07-02 11:50:21 UTC+8 by Codex harness.

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

- Git: `## main...origin/main [ahead 2]`.
- PROJECT_LOG last updated: 2026-07-02 (UTC+8) by Claude.
- Judge sidecars: 15 score files and 15 summary files.
- Report artifact: docs/FYP_Report_2026-07-01_v5.docx modified 2026-07-02 11:50:20 +08.
- Immutable manifest: results/raw_artifact_manifest.sha256 modified 2026-07-02 09:37:43 +08.

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

1. Run make agent-check, review the diff, then commit or hand off the current change.
2. If editing report-worthy content, edit `scripts/build_fyp_report_v5.js` and run `make report`.
3. Regenerate this handoff and the dashboard after meaningful harness or state changes.

## Privacy / Artifact Guardrails

- Use IDs, counts, labels, aggregate metrics, and redacted sidecars only.
- Preserve `raw.jsonl` and `summary.json` as TC1-original artifacts.
- Keep `docs/PROJECT_LOG.md` as the durable decision and changelog record.
