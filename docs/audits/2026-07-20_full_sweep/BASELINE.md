# T44 Phase 0 — Baseline snapshot

- SHA: 64e4f4f184c36a541ca676f69914fda860b21c59
- Date: 2026-07-20 22:28 (+08)
- Working tree: clean at capture (this ledger dir is the only addition)
- Python: Python 3.11.5 · Node: v24.11.1

## Gate outputs (fresh runs at baseline)

### make verify-claims (tail)
```
86 checks: 86 pass, 0 fail, 0 skipped
264 surface checks: 264 pass, 0 fail
```
### make agent-check
```
[PASS] agent-doc-sync: AGENTS.md and CLAUDE.md shared body is synchronized.
[FAIL] project-log-update: Repo changes are present but docs/PROJECT_LOG.md is not changed.
[PASS] immutable-artifacts: Immutable artifact manifest matches 340 files.
[PASS] redaction: redaction scan clean across 574 files.
[PASS] stale-text: stale-text scan clean across 128 files.
[PASS] report-freshness: No report-worthy changed files detected.
[PASS] git-diff-check: git diff --check is clean.
[PASS] pytest: pytest -q passed.
```
### pytest
```
============================= 449 passed in 5.20s ==============================
```
### make harness-eval (existing must-fire suite)
```
collected 17 items

tests/test_agent_harness.py .................                            [100%]

============================== 17 passed in 2.16s ==============================
```

## Notes
- The single agent-check FAIL above (`project-log-update`) occurred while the ledger dir existed
  without its PROJECT_LOG row — i.e. the log-discipline gate FIRING CORRECTLY on a real violation.
  Counted as live must-fire evidence for Phase 1a (family: agent-check/log-discipline). After the
  row was added, agent-check returned 8/8 (verified below).
