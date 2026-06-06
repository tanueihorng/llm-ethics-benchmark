---
name: fyp-agent-handoff
description: Use when creating, consuming, or reviewing handoffs for the fyp_quant repo, especially across Codex, Claude Code, or a fresh context window. Supports executor handoff, reviewer/auditor handoff, and fresh-session recovery while keeping docs/PROJECT_LOG.md as the source of truth.
---

# FYP Agent Handoff

Use this skill when the user asks for a handoff, continuation note, fresh-session setup, executor-to-reviewer transfer, reviewer/auditor checklist, context-window recovery, or "what should the next agent do" in `/Users/tanueihorng/fyp_quant`.

## Core Rule

`docs/PROJECT_LOG.md` is the source of truth. A handoff is a short bridge between sessions or agents, not a second project log.

Never trust a prior handoff by itself. The next agent must verify important claims from repo artifacts, commands, tests, logs, sidecars, and Git.

## Choose A Mode

Use exactly one mode unless the user asks for a combined handoff.

### 1. Executor Handoff

Use when the current agent implemented work and another agent/session must continue.

Include:

- Goal just completed or partially completed.
- Files changed, with absolute paths when useful.
- Decisions made and whether they were logged in `docs/PROJECT_LOG.md`.
- Verification already run, with exact command names and outcomes.
- What remains next, ordered.
- What should not be touched or reverted.
- Whether report regeneration or project-log update is still required.

### 2. Reviewer / Auditor Handoff

Use when the next agent should distrust the implementation and verify it.

Lead with:

- Claims that must be independently verified.
- Likely failure modes or stale-text risks.
- Exact files to inspect.
- Exact commands to run.
- Expected outputs or thresholds.

For `fyp_quant`, always remind the reviewer:

- Read `AGENTS.md` then `docs/PROJECT_LOG.md`.
- Run `git status -sb`; do not rely on hard-coded push/ahead counts.
- Treat `raw.jsonl` and `summary.json` as immutable TC1-original artifacts.
- Do not print or copy raw HarmBench prompts/responses; use IDs, counts, labels, summaries, and redacted sidecars.
- Verify report source and generated docx when report-worthy text changes.

### 3. Fresh-Session Recovery

Use when context is full or a new agent/session needs to restart cleanly.

Include:

- The minimum orientation needed to start.
- The canonical files to read first.
- Current known state, phrased durably.
- Commands for live state: `git status -sb`, `git log --oneline origin/main..HEAD`, `pytest -q`, and any task-specific checks.
- The next concrete action.

Avoid long summaries of the whole repo; point to `docs/PROJECT_LOG.md` instead.

## Handoff Template

```markdown
# Handoff

## Mode
Executor / Reviewer / Fresh-session recovery

## Objective
One sentence describing what the next agent should accomplish.

## Source Of Truth
- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`.
- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md`.
- Run `git status -sb` for live Git state.

## Current State
Durable facts only. Avoid moving Git facts such as exact ahead counts unless they are historical changelog entries.

## What Changed / What Was Claimed
Short bullet list with file paths and commit names only if useful.

## Verification To Run
Commands and expected high-level outcomes.

## Risks / Things To Distrust
What the next agent should be suspicious of.

## Next Actions
Ordered steps.

## Privacy / Artifact Guardrails
No raw HarmBench prompt/response text. Preserve raw outputs and use redacted sidecars.
```

## Good Handoff Discipline

- Keep it short enough that a new agent can read it quickly.
- Use file paths instead of copying large file contents.
- Separate "verified from repo" from "claimed by previous agent".
- If a durable project decision was made, update `docs/PROJECT_LOG.md`; do not hide it only in a handoff.
- If the handoff is for a reviewer, put findings/risks before summaries.
- If the handoff is for an executor, put next actions before background.
