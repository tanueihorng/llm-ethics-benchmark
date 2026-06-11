# Codex Meetup Brief

## One-Line Positioning

I use Codex as a repo-native research harness for my FYP, not just as a coding assistant.

## Core Message

The distinctive part is not that Codex writes code. The distinctive part is that the repo has been designed so agents can orient, act, verify, recover, and hand off across long-running research work. The harness makes the repo remember.

## What The Harness Does

1. It gives agents orientation.
   - `AGENTS.md` and `CLAUDE.md` define the shared operating contract.
   - `docs/PROJECT_LOG.md` is the source of truth for status, decisions, tasks, and changelog.
   - `docs/HANDOFF.md` is only a bridge for fresh sessions.

2. It gives agents constraints.
   - Raw TC1 outputs are immutable.
   - New scoring layers must be derived sidecars.
   - Raw HarmBench prompt and response text must not be copied into handoffs, diagnostics, or chat.
   - TC1 work must respect the head-node and `sbatch` policy.

3. It gives agents actionability.
   - `make smoke`, `make analyze`, `make report`, and `make cluster-generate` are stable commands.
   - `python fyp_cli.py agent-status` gives a fresh agent live state.
   - `docs/agent_tasks/` gives bounded task packets.

4. It gives agents verification.
   - `make agent-check` validates docs sync, log discipline, artifact integrity, stale text, redaction, report freshness, whitespace, and tests.
   - `results/raw_artifact_manifest.sha256` locks the ignored raw artifacts by hash.
   - `python scripts/harness_eval.py` proves the harness catches broken mini-repos.

5. It gives agents recoverability.
   - `python scripts/generate_handoff.py` refreshes the next-session bridge.
   - `python scripts/generate_agent_dashboard.py` refreshes a scan-friendly status page.
   - `python scripts/generate_tc1_checklist.py` refreshes the cluster-operation checklist.

## Best Example To Tell

The strongest example is the HarmBench scoring correction.

The raw experiment outputs were preserved as TC1-original evidence. When the scoring layer needed correction, the repo moved revised scoring into v2 sidecars, then later into official HarmBench judge sidecars. The report and project log record the methodological change. This shows Codex acting as a research-integrity assistant: it can audit, repair, and document a scientific workflow without overwriting evidence.

## Suggested Talk Track

"Most people think of Codex as a pair programmer. In my FYP, I use it more like a research operations layer. The repo tells Codex what is true, what is allowed, what must be verified, and how to hand off to another agent. Memory is not magic; I make the repo remember."

"The harness has five jobs: orient the agent, constrain the agent, give it safe commands, verify its work, and recover across sessions. That matters because this FYP includes cluster jobs, generated reports, scoring corrections, and immutable research artifacts. A normal chat memory would not be enough."

"The USP is repo-native harness engineering. I am not just prompting better. I am shaping the environment so agents can do more while being trusted less."

## Demo Sequence

```bash
python fyp_cli.py agent-status
make harness-eval
make agent-check
python scripts/generate_handoff.py
```

## Likely Questions

**Is this just documentation?**
No. The latest step turns the documentation into executable checks: `make agent-check` can fail when the repo violates harness rules.

**Why not rely on chat memory?**
Because long-running research needs durable state, audit trails, and reproducibility. The source of truth is in files, tests, Git, and sidecars.

**What is the human still responsible for?**
The human owns decisions, authentication, supervisor communication, and final claims. The agent handles repeatable repo operations and verification.

**What is the biggest improvement?**
The artifact policy and `make agent-check`. They convert prose rules into machine-checkable rails.

## USP Summary

My USP is using Codex to build and operate a self-verifying research harness:

- repo-native memory
- cross-agent continuity
- immutable evidence
- derived sidecars for scoring changes
- report-as-code
- cluster-aware workflow
- executable harness checks
- generated handoffs and dashboards
