---
name: fyp-meetup-brief
description: Use in fyp_quant when preparing the Codex meetup story, USP, demo script, or explanation of the repo-native agent harness and memory strategy.
---

# FYP Meetup Brief

Use this skill when the output is a presentation explanation, demo script, or
USP narrative about how Codex is used in this FYP.

## Start

1. Read `AGENTS.md`, then `docs/PROJECT_LOG.md`.
2. Run `python fyp_cli.py agent-start --task codex-meetup-prep --agent fyp-meetup-story`.
3. Read `docs/agent_tasks/codex-meetup-prep.md`.

## Story To Preserve

- Codex is used as a repo-native research harness, not just as autocomplete.
- Memory is not magic; make the repo remember.
- `docs/PROJECT_LOG.md` is durable state, `docs/HANDOFF.md` is a bridge, skills/task packets are on-demand context, and checks enforce rules.
- The strongest example is the scoring correction: raw outputs stayed immutable while v2 and judge sidecars carried revised scoring.

## Guardrails

- Keep the story grounded in concrete repo files and commands.
- Do not expose raw HarmBench prompt or response text.
- Do not claim an automation exists unless it is present in the repo or current Codex setup.

## Finish

Run:

```bash
python fyp_cli.py agent-status
make agent-check
```
