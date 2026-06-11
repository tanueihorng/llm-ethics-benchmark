# Agent Task Packet: Harness Maintenance

## Objective

Keep the agent harness itself trustworthy as the repo evolves.

## Likely Files

- `configs/artifact_policy.yaml`
- `ethical_benchmark/harness/agent.py`
- `scripts/agent_check.py`
- `scripts/generate_handoff.py`
- `scripts/generate_agent_dashboard.py`
- `tests/test_agent_harness.py`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/PROJECT_LOG.md`

## Allowed Actions

- Add new checks when a recurring failure mode appears.
- Convert prose rules from `AGENTS.md` into machine checks.
- Regenerate `docs/HANDOFF.md`, `docs/AGENT_DASHBOARD.md`, and `docs/TC1_AGENT_CHECKLIST.md`.
- Update the immutable manifest only after verifying raw artifacts are intended and unchanged.

## Forbidden Actions

- Do not make harness checks silently ignore failures for convenience.
- Do not remove privacy or immutability checks without recording a decision.
- Do not make `docs/HANDOFF.md` a second source of truth.

## Verification

```bash
python scripts/harness_eval.py
make agent-check
python scripts/generate_handoff.py
python scripts/generate_agent_dashboard.py
```

## Done Criteria

- New checks have tests that intentionally fail on bad fixtures.
- `make agent-check` passes in the real repo.
- `docs/PROJECT_LOG.md` records the change.
