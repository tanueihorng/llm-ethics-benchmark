# T44 Phase 7 — audit-of-audits results

Session S7, 2026-07-21. Two workflow rounds (`wf_3a3ac433-4d2` + rerun `wf_60ac08f2-7f8`):
7 trace seats (one per past audit source), every recorded defect assigned a disposition
∈ {remediated, tracked-open, waived, LOST}; each LOST candidate attacked by xhigh refuters
hunting for the disposition the seat missed; orchestrator adjudicated + re-verified. 16 agents
total (~1.5M subagent tokens).

## QC incident (recorded per the sweep's honesty rule)
First round: 2 of 7 Opus-LOW seats returned PLACEHOLDER output ("notes":"test", items "x"/"y")
and 1 hit the structured-output retry cap. Caught by count reconciliation (traced ≠ recorded −
OK). Re-run at effort MEDIUM with anti-placeholder + completeness constraints succeeded. Lesson
for future harness work: Opus-low is unreliable for large structured traces; reconcile counts
before trusting any structured fan-out result. (One refuter also died on the retry cap,
leaving FS-21 with a single vote — compensated by direct orchestrator verification.)

## Disposition census (all 7 sources)

| Source | Recorded | OK rows | Remediated | Tracked-open | Waived | LOST |
|---|---|---|---|---|---|---|
| 2026-07-10 adversarial academic | 29 | 17 | 7 | 3 (T33/T15/T34) | 1 | **1 → FS-21** |
| 2026-07-11 clean-room (Codex) | 27 | 15 | 10 | 1 (FS-15 track) | 1 | 0 |
| 2026-07-11 fable5 full review | 88 | 61 | 17 | 7 | 1 | **2 → FS-22, FS-23 (cluster residual)** |
| 2026-07-15 fable5 full audit | 45 | 20 | 26 | 0 | 0 | 0 |
| 2026-07-16 code review | 25 | 14 | 1 | 10 | 0 | 0 |
| REPORT_CLAIM_AUDIT_v5.md | 58 | 34 | 25 | 0 | 0 | 0 |
| PROJECT_LOG-row audits (07-19 citation etc.) | 10 | 2 | 6 | 1 (T30b) | 1 | 0 |

The 25th-verifier BH+κ item traced explicitly: **LOST→recovered (D54/T43), now remediated** —
the historical instance that motivated this phase.

## The three recovered LOST findings (all from the pre-ledger-discipline 07-10/07-11 era)

- **FS-21 (P3)** — report:604 says generation uses `torch.inference_mode`; the code uses
  `torch.no_grad()` (functionally equivalent here, factually wrong). Untracked 11 days.
- **FS-22 (P2)** — the embedded HarmBench classifier prompt is the HF model-card template, not
  the GitHub-harness variant; fork undisclosed → absolute ASRs not literature-comparable
  (matched-pair deltas immune). Untracked 10 days.
- **FS-23 (P3, cluster residual)** — of the ~9-item methods-framing cluster, T40/D47 fixed 3
  named members; 6 sub-items remain present and untracked (incl. "cancels in every matched-pair
  delta" asserted-unmeasured, the single-run claim vs the qwen_2b_base smoke+resume composite,
  benchmark-contamination silence, val+test pooling). Each gets fix-or-waive at Phase 9.

## The P1 process finding

**FS-24** — the no-loss guarantee failed for the 07-10/07-11 audits (P2/P3s had no tracking
mechanism; only P1s got remediation rows). Everything from 07-15 onward traces clean. Phase 9
adopts the standing rule: every audit close leaves each recorded P2/P3 with a tracker or an
explicit waiver — the T44 ledger pattern becomes mandatory.

**Phase 7 verdict:** ~282 recorded items traced across 7 sources; 92 defect items dispositioned;
**3 losses recovered** (now FS-tracked), 0 losses after 2026-07-15 — the ledger discipline
demonstrably works, and the pre-discipline era is now fully reconciled.
