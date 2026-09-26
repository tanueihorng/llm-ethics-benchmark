# Handoff — five Claude Code sessions, 2026-09-25 to 2026-09-26

## Mode
Fresh-session recovery. A handoff is a bridge, not a second log. `docs/PROJECT_LOG.md` is authoritative.

## Objective
Resume FYP work after five parallel Claude app sessions. The next concrete step is the student's T71 answer (the canonical/final-report base), then the T15 final report due **Mon 19 Oct 2026**.

## Source of truth
- Read `AGENTS.md`.
- Search `docs/PROJECT_LOG.md` for D92, D93, D94, T71, T73, T75 and T79. Do not load its full history.
- Live state: `git status -sb`, `python fyp_cli.py agent-status`, `make agent-check`.

## The five sessions
| Session (app title) | What it did | State |
|---|---|---|
| **Repo cleanup and D94** (this session) | Committed the month-long backlog in grouped commits. Reports and presentations are local-only (D94). T53 and T1 closed. `uv.lock` ignored. Pushed. | Done; logged (2026-09-26 18:06 and 18:12 rows) |
| **Literature review structure feedback** | Built interim progress v3: fundamentals-first Chapter 2 and revised Chapter 1 (D92/D93), plus 2 foundational sources. The citation/bibliography audit found no hallucinated references. | Done; logged (17:35, 17:47, 17:59 rows). v3 is a `comparison-draft` |
| **Repository root cleanup review** | Read-only proposal to delete or move root clutter, stop tracking `.understand-anything/` and `.hallmark/`, reorganise `docs/`, and prune `tmp/`. | **Proposal only; nothing changed.** It conflicts with D84 (no deletes or moves) unless the student approves and a new §3 decision supersedes D84. It advises holding the `docs/` reorganisation until T71. |
| **FYP motion design video** | Rendered a 66 s 1080p60 explainer video with numbers taken from `results_512/analysis/`. Three overclaiming captions were softened. | Moved to `docs/media/` (local-only, D94) on 2026-09-26 21:12. |
| **Writing ability** | Short Q&A: offered report-chapter writing help. | Nothing to resume |

## Current state (verified 2026-09-26)
- `main` matched `origin/main` after the push; recheck with `git status -sb`.
- **D94:** report and deck builders, rendered documents, `docs/**/*.html`, `docs/viva/`, `docs/drafts/` and `world3d/` are gitignored and local-only.
  - CI runs the core suite only; report tests run locally.
  - The `local-only-untracked` check fails if any of them becomes tracked again.
  - Commits before D94 still contain older versions, because the history purge was declined.
  - The repository is private.
- `make agent-check` passed (806 tests) before the last commit.
- Canonical is still the old docx-js `docs/FYP_Report_2026-07-01_v5.docx` (`make report`).
  - This session recommended **`docs/rewritten_fyp_report_v5.docx`** as the new canonical/final base: it's the only full-length candidate that rebuilds from source.
  - The September improvements from interim progress v3 (Chapters 1–2, new sources, the T73 post-hoc INT8 disclosure) and the Jev results (T75) would then be ported into it.
  - **The student has not said go.** Do not switch without an explicit yes. Repointing touches `Makefile`, `configs/report_versions.yaml`, AGENTS/CLAUDE, `configs/artifact_policy.yaml` report triggers, `configs/claim_surfaces.yaml`, `scripts/verify_report_claims.py` and about 6 tests. Relabel the old v5 as archived; delete nothing.

## Risks / things to distrust
- Explainer videos now live in `docs/media/` (gitignored under D94, included in `make backup`).
- **T79 is open:** this Mac is the only current copy of the builders. Nothing is backed up until the student picks a destination for `make backup DEST=<path>`.
- `docs/~$written_fyp_interim_progress_v3.docx` is a Word lock file, so v3 was open in Word. Don't rebuild v3 while it is open.
- **T1b** (August email) status is unconfirmed; don't assume it was sent.
- The root-cleanup proposal judged files by date, not by content; verify before acting on any of it.

## Next actions
1. Get the student's go or no-go on canonical = `rewritten_fyp_report_v5.docx` (T71). If go, repoint it, record a §3 decision, and run `make agent-check`.
2. T79 backup, as soon as the student gives a destination.
3. Final-report work (T15): port v3's Chapters 1–2, fix T73, fold in Jev (T75), then `make verify-claims` and a pagination check.
4. Root cleanup only with explicit approval and a decision superseding D84; do the `docs/` reorganisation after T71.

## Privacy / artifact guardrails
No raw HarmBench prompts or responses anywhere; use IDs, labels and redacted sidecars. `raw.jsonl` and `summary.json` are immutable. Never `git add -f` a D94 path.
