---
name: fyp-report-audit
description: Use in fyp_quant when auditing or strengthening the FYP report, Chapter 6/7 results discussion, judge-vs-v2 claims, report/log consistency, or stale current-facing result text.
---

# FYP Report Audit

Use this skill only inside `/Users/tanueihorng/fyp_quant` for report-facing
work. Keep `docs/PROJECT_LOG.md` as the source of truth and treat this skill as
a small task router, not a replacement for the log.

## Start

1. Read `AGENTS.md`, then `docs/PROJECT_LOG.md`.
2. Run `python fyp_cli.py agent-start --task T21 --agent fyp-report-auditor`.
3. Read `docs/agent_tasks/T21-strengthen-results-discussion.md`.

## Scope

- Verify judge-primary claims against `results/analysis/judge_agreement.{json,csv}`.
- Verify pair labels and deltas against `results/analysis/pairwise_deltas.{json,csv}` and `results/analysis/pair_interpretations.{csv,json}`.
- For report-worthy edits, change `scripts/build_fyp_report_v3.js` first, then run `make report`.
- Update `docs/PROJECT_LOG.md` for every change.

## Guardrails

- Do not mutate `results/*/*/raw.jsonl` or `results/*/*/summary.json`.
- Do not copy raw HarmBench prompt, behavior, generation, or response text into docs, logs, or chat.
- Use IDs, counts, labels, aggregate metrics, sidecar filenames, and redacted summaries.

## Finish

Run:

```bash
python fyp_cli.py agent-status
make agent-check
```
