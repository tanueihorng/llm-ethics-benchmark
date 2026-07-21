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
| 0 Freeze & baseline | **done** | S1 | BASELINE.md; committed 680e7bc |
| 1 Gate self-audit | **done** | S1 | 1a: 12 families fired, 0 fail-to-fire (phase1_must_fire.md). 1b: coverage map done (phase1_coverage_map.md + phase1_coverage_data.json); 18 high / ~103 medium / ~87 low unlocked claims; findings FS-1..FS-6 (FS-1 = verified content defect 300-vs-340). Workflow wf_15d80ba0-845: 10/11 agents (diff:thesis died on USER SESSION LIMIT, resets 02:40; completed INLINE by Fable orchestrator — model-policy deviation recorded) |
| 2A Recompute (scripted) | pending | S2 | targets now include FS-2 (§6.14 numbers) + FS-3 (INT8 capability verdicts, Table 6.5 cells) |
| 2B Stats appropriateness | pending | — | needs 1b's unlocked-claims map |
| 3 Citations (4 axes) | pending | — | |
| 4 Cross-document consistency | pending | — | |
| 5 Omission audit | pending | — | |
| 6 Fresh-clone reproducibility | pending | — | |
| 7 Audit-of-audits | pending | — | |
| 8 Examiner panel (4 lenses) | pending | — | after 1–7 |
| 9 Synthesis & closure | pending | — | last |

## Remediation policy (user decision 2026-07-21)

**Batch-at-Phase-9 (default):** findings accumulate in FINDINGS.md and are remediated once, at
Phase 9, grouped by THEME (content fixes vs lock/gate extensions), with one re-verify loop.
Rationale: keeps the frozen baseline audit-stable, lets fix clusters complete before design
(FS-2..FS-6 are one lock-extension change), and avoids N fix-verify-drift cycles.
**Hotfix exception (only):** a P1 content defect on a deliverable that will leave the repo
before the sweep ends → fix immediately + drift-log entry + targeted re-check of done phases.
FS-1 does NOT qualify while nothing is being sent; if the user decides to send thesis/report
mid-sweep, FS-1 is fixed first.

## Drift log

(none — content work frozen during the sweep; T1 send, if it happens, gets an entry here)

## NEXT-ACTION

**S1 COMPLETE (Phase 0 + Phase 1). Next: S2 = Phase 2A (scripted recompute — cheap session).**
S2 plan: write deterministic recompute scripts against results_512 sidecars/artifacts for
(a) every FS-2 §6.14 number (refusal_margin.json + per-item margins), (b) FS-3 INT8 MMLU/ARC
significance (precision_sweep + per-item summaries; McNemar on paired records), (c) Table 6.5
cells vs headline_512_vs_128.json + genlen_robustness.json, then extend to the full locked set
(re-derive the 86 locks' values independently). Findings → FINDINGS.md at discovery. Inputs:
phase1_coverage_data.json (unlocked map), BASELINE.md. FS-1 (300-vs-340) awaits remediation —
either a quick fix session (builders + regen + log) or Phase 9; if fixed mid-sweep, add a
drift-log entry here.
