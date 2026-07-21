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
| 2A Recompute (scripted) | **done** | S2 | phase2a_recompute.py + phase2a_results.md: **84/85 pass**. §6.14 verified (27 checks) except FS-7 (dz +1.8 should be +1.75/+1.7, mis-rounding). INT8 capability verdict VERIFIED by first-ever direct test (10 McNemar contrasts from 14,720 paired items, none sig, max 1.33pp). Table 6.5 all cells match. Independent re-derivations: own BH reproduces all 20 q-values; own exact McNemar all 20 p-values; own κ over raw sidecars reproduces all 30 published κs to 1e-6; human-val κs 0.59/0.11 from confusion cells; XSTest gold κs (−0.006/0.485/0.054/0.662), 2-of-63, 61-of-63, 0.695 all reproduce from the 200-item trail; MDE formula reproduces all per-pair MDEs |
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

**S2 COMPLETE (Phase 2A). Next: S3 = Phase 2B (stats appropriateness, heavy — needs full window)
or Phase 7 (audit-of-audits, medium — fits a partial window). Both have their inputs ready:**
2B plan: Opus-xhigh panel judging METHOD CHOICE (not execution): McNemar-exact vs alternatives for
paired binaries; BH family composition vs its preregistration; one- vs two-sided consistency;
CI-vs-test disagreement handling; dz appropriateness; MDE formula assumptions. Inputs:
phase1_coverage_data.json + phase2a_results.md.
7 plan: iterate every ledger under docs/audits/* + audit-shaped PROJECT_LOG rows; verdict per finding
(remediated-link / tracked-open / waived / LOST); any LOST finding = new P1 process finding.
FS-1 + FS-7 await Phase 9 R1 (content); FS-2..FS-6 lock gaps await Phase 9 R2.
