# T44 Findings Ledger (append-only, IDs assigned AT DISCOVERY TIME)

Format per finding:
`FS-<n> | phase | severity (P1/P2/P3) | status (open/verified/refuted/remediated/waived) | evidence (file:line) | one-line claim | tracking link`

Statuses change in place (append a dated status-change note under the entry); entries are never deleted.

---
FS-1 | phase1b | P2 | **verified** (live check 2026-07-20 23:0x) | scripts/build_fyp_thesis_v4.js:388 + scripts/build_fyp_interim.js:393 | CONTENT DEFECT: both Appendix-A texts claim the checksum manifest pins "300 immutable raw files"; `wc -l results/raw_artifact_manifest.sha256` = **340** (120 results_512 + 100 sensitivity + 120 results). Stale count on an unlocked claim. | remediation → Phase 9 (or earlier quick fix; report builder does not carry the number)

FS-2 | phase1b | P2 | open | scripts/build_fyp_report_v5.js:934-937 (§6.14) | Mechanism-probe results have ZERO lock coverage: Wilcoxon p<0.001 verdicts, margin shifts (+1.2/+4.0/−1.1/−0.3), dz range, AUC 0.76/0.64/0.61/0.54, 92-flip 50/42 split, gate AUCs 0.86-0.90/0.69. | Phase 2A recompute target + candidate new locks

FS-3 | phase1b | P2 | open | scripts/build_fyp_report_v5.js:942,954-960 (§6.15/§6.16) + thesis:303 | INT8 "capability-lossless" significance verdict (no INT8 MMLU/ARC delta sig; all within ~1.3pp) and Table 6.5's non-Qwen @128 cells + qwen_4b/mistral/phi gpt-4o cells are unchecked by any lock. | Phase 2A recompute target + candidate new locks

FS-4 | phase1b | P2 | open | scripts/build_fyp_interim.js (9 sites: 133,283-286,297,307-311,317,322,358) | Interim surface systematically under-locked: BH 3-survivor verdict + q-values, validation-informed 2-survivor verdict, RQ block, precision-sweep table cells, Table 6.1 family κs, Phi −0.048 CI — all duplicated from report/thesis but gates read those builders, not the interim. | candidate: extend lock targets to interim (mirror the thesis treatment)

FS-5 | phase1b | P3 | open | scripts/build_fyp_thesis_v4.js:278,321,325,326 | Thesis verdict-bearing blocks unpinned on this surface (scorer-relocation §6.1, RQ block ch7, LlamaGuard §7.2, XSTest-gold κs §7.2) — values locked on the report builder only. | candidate: thesis-side gates

FS-6 | phase1b | P3 | open | scripts/verify_report_claims.py (gate design) | Duplicate-instance blindness: presence gates are whole-file string searches, so a drifted duplicate instance (abstract/summary restatements) passes while any one pinned instance survives. ~85 medium unlocked claims are largely this class. | candidate: instance-counting gates for headline numbers (e.g. assert N occurrences)

FS-7 | phase2a | P3 | **verified** (recompute 2026-07-21) | scripts/build_fyp_report_v5.js:936 (§6.14) + report_humanized mirror | CONTENT DEFECT (mis-rounding): "effect sizes spanning Cohen dz ≈ −0.74 to +1.8" — artifact max m1 dz = +1.7498 (qwen_4b), which rounds to +1.7 at the sentence's own precision; "+1.8" is a double-rounding (1.7498→1.75→1.8). Min −0.74 (llama −0.7402) is exact. Thesis/interim do not carry dz. | remediation → Phase 9 R1 (change "+1.8" → "+1.75" or "+1.7")

> STATUS NOTE (2026-07-21, post-2A) on FS-2: §6.14 VALUES now verified correct against refusal_margin.json (27 checks; sole exception = FS-7 rounding). The LOCK GAP remains open → Phase 9 R2.
> STATUS NOTE (2026-07-21, post-2A) on FS-3: INT8 capability verdict now VERIFIED by direct recompute — 10 McNemar contrasts (5 pairs × MMLU/ARC) from 14,720 paired per-item records: no p < 0.05, max |Δ| = 1.33pp; Table 6.5 cells all match artifacts. The LOCK GAP remains open → Phase 9 R2.

FS-8 | phase2b | P3 | **verified** (panel survivor, 2× adversarial confirm + orchestrator artifact re-check 2026-07-21) | scripts/build_fyp_report_v5.js:942,945 + scripts/build_fyp_thesis_v4.js:303 | CONTENT DEFECT (misattributed basis + unbounded equivalence): the INT8 capability claim cites "(paired bootstrap)" as its significance basis but precision_sweep.json contains NO bootstrap/CI on any axis (its only tests are ASR McNemar); no capability-axis MDE exists (multiple_comparisons.py:265 filters to harmbench_asr_judge), yet thesis:303 says the deltas sit "below the study's detection floor" (ASR-only MDE imported onto MMLU/ARC); unqualified "capability-lossless"/"free precision point" = equivalence-from-non-significance, contra the report's own bounded-null discipline (report:1015). The VERDICT itself is true (Phase 2A direct test: 10 McNemar contrasts, none sig, max 1.33pp) — the stated BASIS and bounding are wrong. | remediation → Phase 9 R1: fix basis attribution (cite exact McNemar on paired per-item records — Phase 2A's evidence; consider committing it as a proper analysis artifact), bound the claim (≈±3pp CI), qualify "lossless", fix thesis detection-floor sentence
