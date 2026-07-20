# T44 Full-Sweep Audit — STATE

> The sweep's single source of truth. Every session reads this FIRST (before PROJECT_LOG
> orientation), executes the NEXT-ACTION, and updates this file + FINDINGS.md before ending.
> Plan: `docs/agent_tasks/T44-full-sweep-audit-plan.md`. Findings: append-at-discovery to
> `FINDINGS.md` — never held in-context for an end-of-session write-up.

| Field | Value |
|---|---|
| **Baseline SHA** | `64e4f4f184c36a541ca676f69914fda860b21c59` (2026-07-20; post-D54, post-T43) |
| **Started** | 2026-07-20 22:28 (+08) |
| **Model policy** | Fable 5 orchestrator · Opus 4.8 xhigh heavy verifiers · Opus 4.8 low mechanical · scripts for 2A |
| **T1 status** | **DEFERRED by user decision 2026-07-20** (user launched T44 without sending; sending later = drift-log entry, low impact: gitignored email + one log row) |

## Phase status

| Phase | Status | Session(s) | Notes |
|---|---|---|---|
| 0 Freeze & baseline | in-progress | S1 | this session |
| 1 Gate self-audit | pending | S1 | 1a must-fire + 1b coverage map |
| 2A Recompute (scripted) | pending | — | |
| 2B Stats appropriateness | pending | — | needs 1b's unlocked-claims map |
| 3 Citations (4 axes) | pending | — | |
| 4 Cross-document consistency | pending | — | |
| 5 Omission audit | pending | — | |
| 6 Fresh-clone reproducibility | pending | — | |
| 7 Audit-of-audits | pending | — | |
| 8 Examiner panel (4 lenses) | pending | — | after 1–7 |
| 9 Synthesis & closure | pending | — | last |

## Drift log

(none — content work frozen during the sweep; T1 send, if it happens, gets an entry here)

## NEXT-ACTION

S1 in progress: Phase 0 baseline capture → Phase 1a perturbation sweep → Phase 1b coverage-map
workflow → end-of-session protocol. If S1 dies mid-run: re-read BASELINE.md (if present, Phase 0
is done); check FINDINGS.md for recorded 1a results; re-run any 1a perturbation not yet recorded;
1b workflow results land in `phase1_coverage_map.md` (absent = re-run 1b).
