---
name: fyp-harness-maintenance
description: Use in fyp_quant when changing agent harness checks, task packets, skills, Codex hooks, custom subagents, dashboards, handoffs, or artifact-policy enforcement.
---

# FYP Harness Maintenance

Use this skill for changes to the agent operating system around the FYP repo:
checks, skills, hooks, custom agents, task packets, dashboards, handoffs, and
artifact-policy enforcement.

## Start

1. Read `AGENTS.md`, then `docs/PROJECT_LOG.md`.
2. Run `python fyp_cli.py agent-start --task harness-maintenance --agent fyp-artifact-guardian`.
3. Read `docs/agent_tasks/harness-maintenance.md`.

## Scope

- Convert repeated prose rules into checks, hooks, or small task packets.
- Prefer on-demand skills over adding long always-loaded instructions.
- Regenerate generated harness docs after changing their source.
- Add tests for any new behavior that is meant to catch future agents.

## Guardrails

- Do not weaken privacy, redaction, or immutability checks for convenience.
- Do not make `docs/HANDOFF.md` a second source of truth.
- Do not mutate raw TC1-original artifacts.

## Finish

Run:

```bash
python scripts/harness_eval.py
make agent-check
python scripts/generate_handoff.py
python scripts/generate_agent_dashboard.py
```
