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
| 2B Stats appropriateness | **done** | S3 | Workflow wf_0cdecbd1-16c: 6 Opus-xhigh seats + 2× adversarial refutation (38 agents, 0 errors). All six seats: `appropriate_with_caveats` — method choices sound. 15 findings → 14 unanimously refuted → 1 survivor = **FS-8** (P3: INT8 "capability-lossless" cites a "(paired bootstrap)" test that was never run; no capability-axis MDE; thesis borrows ASR MDE as "detection floor"; verdict true per Phase 2A, basis misattributed). phase2b_results.md has seat verdicts, kill list, + 4-item optional polish list for Phase 9 |
| 3 Citations (4 axes) | **done** | S4 | Deterministic: D54 gate 12/12; numbering integrity clean on all 8 surfaces (2 orchestrator false alarms self-caught pre-filing). Workflow wf_8cb7af9e-b96 (47 agents, 0 errors): 31 fitness + 2×3b + 3c + 3d seats; **30/35 clean, zero hallucinated citations** (2nd consecutive clean fitness audit; 5 new stats refs + renumber cascades all correct). 6 findings → 2 killed → **FS-9** (tex-only Jin misattribution), **FS-10** (Proskurina venue inconsistency ×6 surfaces), **FS-11** (MDE/Connor-Lachin formula uncited; thesis has NO power ref; lexicon blind — D54-class catch by the absence-direction pass). All P3. phase3_results.md |
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

**S4 COMPLETE (Phase 3). Next: S5 — pick by window:**
- **Phase 4 (cross-document consistency — heavy):** diff the load-bearing story across ALL
  surfaces (report v5, thesis v4, interim, 3 humanized, 2 tex mirrors, decks, README,
  RESULTS_CARD, dashboard data layer, CLAUDE/AGENTS, PROJECT_LOG §1); both directions —
  same claim stated differently, and claim absent where a reader needs it. Point-in-time
  decks exempt but must carry their snapshot notice.
- **Phase 5 (omission audit — heavy):** what SHOULD the documents say that they don't.
- **Phase 6 (fresh-clone reproducibility — cheap-medium)** and **Phase 7 (audit-of-audits —
  medium)** both fit partial windows.
Content findings awaiting Phase 9 R1: FS-1, FS-7, FS-8, FS-9, FS-10, FS-11.
Lock/gate gaps awaiting Phase 9 R2: FS-2..FS-6 + FS-11's lexicon extension.
Optional polish: 4-item list in phase2b_results.md.
