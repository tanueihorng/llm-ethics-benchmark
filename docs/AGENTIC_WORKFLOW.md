# Agentic Workflow Guide

This repo uses small, on-demand context instead of one giant startup prompt.
The operating rule is:

> `AGENTS.md` gives the law, `docs/PROJECT_LOG.md` gives current state, skills
> and task packets give task-specific context, subagents isolate review, hooks
> refresh/check the work, and `make agent-check` is the finish gate.

For the main all-in-one poster, open
`docs/architecture/fyp_quant_integrated_agentic_stack.svg`. For narrower views,
use the top-down repo map (`docs/architecture/fyp_quant_repo_hierarchy.svg`),
the agent harness control plane
(`docs/architecture/fyp_quant_agent_harness_architecture.svg`), or the full
agentic flow view (`docs/architecture/fyp_quant_agentic_architecture.svg`). All
have editable draw.io sources in `docs/architecture/`.

## Start A Session

For a general session:

```bash
python fyp_cli.py agent-status
```

For a specific task:

```bash
python fyp_cli.py agent-start --task T21 --agent fyp-report-auditor
python fyp_cli.py agent-start --task T22 --agent fyp-judge-reviewer
python fyp_cli.py agent-start --task codex-meetup-prep --agent fyp-meetup-story
python fyp_cli.py agent-start --task harness-maintenance --agent fyp-artifact-guardian
```

The same command is available through Make:

```bash
make agent-start TASK=T21 AGENT=fyp-report-auditor
```

## Use Skills

Repo-scoped Codex skills live in `.agents/skills/`.

Use them explicitly when you want Codex to load only the relevant workflow:

```text
$fyp-report-audit
$fyp-second-judge
$fyp-meetup-brief
$fyp-harness-maintenance
```

Each skill points back to the relevant `docs/agent_tasks/*.md` packet and
harness commands.

## Use Context-Isolation Subagents

Project-scoped Codex custom agents live in `.codex/agents/`.

Recommended usage pattern:

```text
Spawn/use the fyp-report-auditor subagent for T21.
Give it only AGENTS.md, docs/PROJECT_LOG.md, the T21 task packet, and the report/analysis files named in the packet.
Wait for findings. The main agent decides edits and runs make agent-check.
```

Available subagents:

- `fyp-report-auditor` - report/log/result consistency.
- `fyp-artifact-guardian` - raw artifact immutability and redaction.
- `fyp-tc1-ops` - SLURM, TC1 policy, offline-mode workflow.
- `fyp-judge-reviewer` - HarmBench judge sidecars and second-judge work.
- `fyp-meetup-story` - Codex meetup explanation and demo story.

Use subagents for audit, exploration, and focused review. Keep final edits and
project-log updates in the main session unless you explicitly delegate
implementation.

## Hooks

Project hooks live in `.codex/hooks.json` and `.codex/hooks/`.

They are intentionally lightweight:

- `PreToolUse` warns if a tool call appears to touch immutable raw artifacts.
- `PreCompact` regenerates `docs/HANDOFF.md` and `docs/AGENT_DASHBOARD.md` when the working tree already has local changes.
- `Stop` runs `agent-status` plus a lightweight `agent_check.py --skip-pytest`.

Codex requires hook review/trust before project hooks run. In the CLI, use:

```text
/hooks
```

Review the hooks, trust them for this repo, and then continue normally.

## Finish A Change

Before ending any implementation session:

```bash
python fyp_cli.py agent-status
make agent-check
python scripts/generate_handoff.py
python scripts/generate_agent_dashboard.py
```

If the change is report-worthy, update `scripts/build_fyp_report_v5.js` and run:

```bash
make report
```

Then update `docs/PROJECT_LOG.md` with the changelog row and any decision or
open-action changes.
