# Harness Self-Evaluation

## Scope

This self-evaluation covers the agent harness added on 2026-06-08. It is focused on whether the harness can catch the main failure modes in this repo, not on whether the FYP experimental conclusions are scientifically complete.

## Implemented Controls

- `make agent-check` runs the full harness gate.
- `python fyp_cli.py agent-status` prints live repo status for fresh agents.
- `configs/artifact_policy.yaml` declares immutable artifacts, allowed derived sidecars, stale-text patterns, redaction scans, and report-worthy file patterns.
- `results/raw_artifact_manifest.sha256` hashes ignored raw artifacts so mutation can be detected.
- `docs/agent_tasks/` stores bounded task packets.
- `scripts/generate_handoff.py` refreshes the session bridge.
- `scripts/generate_agent_dashboard.py` refreshes a scan-friendly status page.
- `scripts/generate_tc1_checklist.py` refreshes a TC1-safe operation checklist.
- `scripts/harness_eval.py` runs the focused harness failure-mode tests.

## Tested Failure Modes

The harness eval suite creates temporary Git repos and intentionally breaks one rule at a time. It currently verifies that the harness catches:

- `AGENTS.md` / `CLAUDE.md` body drift.
- Repo changes without a `docs/PROJECT_LOG.md` update.
- Stale current-facing handoff text.
- Redaction leaks in handoff content.
- Mutation of immutable raw artifacts after the hash manifest is written.
- Status and generated Markdown render without raw response leaks.

## Known Limits

- The redaction scanner is conservative; it catches explicit leak patterns and schema-like raw response snippets, but it cannot prove no natural-language paraphrase of a raw HarmBench response exists.
- The report-freshness check works on current Git changes. After a commit, historical verification still depends on the changelog and report artifact history.
- The immutable manifest protects this local workspace's ignored raw artifacts. A fresh clone without raw artifacts will need the raw files restored before the check can pass.
- `make agent-check` includes pytest and can take longer than lightweight status checks.

## Self-Test Commands

```bash
python scripts/harness_eval.py
python scripts/agent_check.py --skip-pytest
make agent-check
python fyp_cli.py agent-status
```

## Evaluation

The harness materially improves the repo because it moves from instruction-only guidance to executable verification. The strongest controls are the immutable artifact manifest, stale-text scanner, redaction scanner, and project-log discipline check. The remaining frontier is deeper semantic validation, such as checking that report claims match numeric analysis outputs.
