---
name: fyp-todo-capture
description: Use to capture the work just done (or in progress) into the repo's persistent, dated todo.md backlog so the next session or agent can resume and execute immediately — without re-doing research. Triggers include "add this to todo", "capture next steps", "save this so the next session can take over", "I have no time, let the next agent continue", "log what we decided". For the fyp_quant repo. This maintains a durable task queue; it is NOT the one-shot fyp-agent-handoff note, and it never replaces docs/PROJECT_LOG.md as source of truth.
---

# FYP Todo Capture

Capture the current/just-completed work into `todo.md` (repo root) as a **dated,
self-contained, resumable** entry. The goal: a future session or a different
agent reads the entry cold and executes the next step with **zero re-research**.

## When to use this (and when not)

Use it when you want to **bank progress into a durable queue** — you paused
mid-task, finished a research/feasibility step, or want the next session to pick
up exactly where you stopped.

Do NOT confuse it with the two neighbours:

| Tool | What it is | Lifetime |
|---|---|---|
| **`fyp-todo-capture`** (this) | Persistent, dated backlog of resumable work items in `todo.md` | Durable queue; prune items as they finish |
| `fyp-agent-handoff` | A one-shot bridge note at the moment of transfer (executor/reviewer/fresh-session) | Ephemeral; written once per transfer |
| `docs/PROJECT_LOG.md` §2 | The canonical, numbered (T<N>) open-actions ledger — strategic/milestone level | Permanent record |

Rule of thumb: **todo.md is the tactical "how to resume right now" buffer;
PROJECT_LOG.md is the strategic record.** When a todo item is *finished*, record
the durable outcome in PROJECT_LOG.md (§4 changelog, plus §2/§3 if it closes an
action or sets a decision) and then remove/tick it here. Never let todo.md become
a second source of truth.

## Core rule

`docs/PROJECT_LOG.md` is the source of truth. `todo.md` points to it and never
duplicates its decisions — it holds the granular execution context that is too
fine-grained for the project log (exact commands, decided-vs-rejected options,
verification already run).

## Procedure

1. **Read the recent work.** Look back over the latest response(s) / the task you
   just did. Extract the concrete substance — don't summarise vaguely.
2. **Open or create `todo.md`** at the repo root. If it has the working-backlog
   header already, keep it; otherwise add the header from the template below.
3. **Write one dated entry** using the Entry Template. Date it with today's
   absolute date (`YYYY-MM-DD`, local UTC+8 to match the repo). Convert any
   "today/tomorrow/next" into absolute dates.
4. **Apply the Specificity Checklist** — an entry that fails it is not resumable.
5. **Keep it short and current.** Prune finished items (push their durable result
   to PROJECT_LOG.md first). todo.md should read in under a minute.

## Entry Template

```markdown
## [YYYY-MM-DD] ACTIVE: <one-line objective>

**Why:** <1–2 lines: the goal and why it matters.>

**Decided (don't re-litigate):**
- <choice> — <one-line reason>

**Rejected (don't re-litigate):**
- <option> — <why it was ruled out, with the blocking fact>

**Verification already done:**
- <what was checked/run> → <result> (so the next agent doesn't repeat it)

**Next steps (ordered, concrete):**
1. <exact action — file paths, exact commands, exact IDs>
2. ...

**Watch items / guardrails:**
- <gotchas, things NOT to touch, repo guardrails that apply>

**Ready-to-paste artifacts (optional):**
- <config blocks, commands, snippets the next agent can use verbatim>
```

## Specificity Checklist (the whole point)

A captured entry must let the next agent act without re-deriving anything. Before
finishing, confirm the entry has:

- [ ] **Absolute date** on the item.
- [ ] **Exact file paths** for every change (`ethical_benchmark/...`, `configs/...`).
- [ ] **Exact commands** (the actual `make`/`sbatch`/`pytest`/`git` lines), not "run the tests".
- [ ] **Exact identifiers** (model IDs, job IDs, flags, config keys) — no "the new model".
- [ ] **Decided AND rejected** options, each with the *reason* — so nobody re-researches a dead end.
- [ ] **Verification already run**, separated from what's still claimed/unverified.
- [ ] **What NOT to touch** (immutable artifacts, things deliberately left as-is).
- [ ] A pointer to **`docs/PROJECT_LOG.md`** as source of truth.

## fyp_quant guardrails to carry into every entry

- `raw.jsonl` / `summary.json` are immutable TC1 originals — never edit; use redacted sidecars.
- No raw HarmBench prompt/response text in todo.md — IDs, counts, labels, summaries only.
- TC1: no Python on the head node for compute; submit via `sbatch`; offline mode needs prefetch.
- When the captured work itself changed the repo, the change still owes a
  PROJECT_LOG.md §4 row — capturing it in todo.md does not discharge that.

## Maintenance

- One entry per coherent task; newest at the top.
- Tick/remove an item only after its durable result is in PROJECT_LOG.md.
- If todo.md grows past a handful of active items, the backlog is stale — prune.
