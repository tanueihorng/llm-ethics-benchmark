# T44 Full-Sweep Audit — Phase 9 Synthesis

Sessions S1–S8, 2026-07-20 → 2026-07-22. Baseline `64e4f4f` (post-D54/T43), frozen throughout
(drift log: empty — zero content commits landed mid-sweep). ~200 agents across 9 phases;
Fable 5 orchestrator / Opus 4.8 tiered executors per the model policy; every finding
adversarially refuted (2× xhigh) and orchestrator-verified against primary artifacts before
filing. Ledger: FINDINGS.md FS-1..FS-25, append-at-discovery, zero losses.

## Headline

**No finding in the entire sweep touches a research conclusion.** The RQ1 power-bounded safety
null, the capability cliff at four-bit, the scorer-validity contribution, the BH survivor set,
every κ, every p, every q — all reproduce from committed artifacts (Phase 2A: 84/85 + Phase 1b
spot-set; sole exception the FS-7 rounding), the method choices are sound (Phase 2B: 6/6 seats),
the citation surface has zero hallucinations (Phase 3, 2nd consecutive clean audit), the JS
document surfaces are in sync (Phase 4: thesis diff fully clean), there are no material
omissions (Phase 5), the scientific core is portable from a fresh clone (Phase 6: 85/86), no
past audit finding remains lost (Phase 7: 282 items reconciled, 3 recovered), and all four
hostile examiner lenses returned **pass_with_minor_corrections** (Phase 8).

## Findings census (25 findings, all verified, none refuted post-filing)

**Content defects → R1 batch (16 items; by surface):**
| Surface | Items |
|---|---|
| Report v5 (+ mirrors) | FS-7 dz "+1.8"→"+1.7"; FS-8 INT8 basis "(paired bootstrap)"→exact-McNemar + bound ±~3pp + qualify "lossless"; FS-11 cite Connor 1987 at §6.5.1; FS-20 gpt-4o run-date note at Result 4; FS-21 "inference_mode"→"no_grad"; FS-22 classifier-template-fork disclosure sentence; FS-25 align RQ4 with §6.4.1; FS-23 fix-items (a,b,c,d,f — see dispositions below) |
| Thesis v4 (+ tex + humanized) | FS-1 "300"→"340 immutable raw files"; FS-8 "detection floor" sentence; FS-11 add power reference |
| Interim (+ tex + humanized) | FS-1; FS-14 revision-note under the 10-July date |
| Tex mirrors only | FS-9 Jin clause reword; FS-12 κ-table re-grade + 0.28→0.29; FS-13 front-matter re-base (planned→done) |
| All non-report surfaces | FS-10 Proskurina + Egashira venue normalization |
| Humanized report | FS-15 restore the two abstract hedges |
| Docs | FS-19 doc-half: npm install + real interpreter in setup docs |

**FS-23 residual dispositions (proposed):** (a) "cancels in every matched-pair delta" → "is held
fixed by design" [fix]; (b) single-run claim → disclose the qwen_2b_base smoke+resume composite
[fix]; (c) add one benchmark-contamination limitation sentence [fix]; (d) soften the Kharinaev
gap-clause [fix]; (e) StrongREJECT/Souly engagement → **waive** (related-work preference; scorer
scope already bounded; waiver recorded in ledger); (f) disclose HarmBench val+test pooling [fix].

**Optional polish (8 items, killed-as-defects/real-as-improvements):** phase2b ×4
("FDR-surviving" one-worder; §833 inline caveat; §3.7 sampling sentence; one-sided note) +
phase8 ×4 (thinking-mode 6th bound; abstract "independently supports" soften; multiseed
paragraph reconcile; "regenerating the numbers" doc section).

**Lock/tooling gaps → R2 batch (6 clusters):** FS-2/FS-3 (§6.14 + INT8/Table 6.5 locks —
recompute values now known-good to pin); FS-4/FS-5 (interim + thesis surface locks); FS-6
(instance-counting for headline numbers); FS-11-lexicon ("power analysis|minimum detectable");
FS-16 (tex↔JS sync gate: stale-label patterns + κ-range sync check); FS-17 (__dirname paths in
13 builders); FS-18 (email surfaces → local_optional; mtime → content-hash freshness); FS-19
(constraints file). Every new gate ships with a must-fire self-test (D42).

**Process → FS-24 (P1):** adopt the audit-close no-loss standing rule as a §3 decision:
closing any audit requires every recorded P2/P3 to carry a §2 tracker or an explicit waiver.

## Scope limits of this sweep (what was NOT checked)
- TC1/GPU-side reproducibility (model loading, inference, judge runs) — stub backends only.
- Unknown-unknowns beyond the four Phase-8 lenses (mitigated, not eliminated).
- Paywalled full texts verified via abstracts/secondary knowledge (e.g. Landis & Koch body).
- Gitignored-only artifacts (raw generations, human label sheets) — process-audited, not re-read.
- Humanized mirrors diffed for numbers/verdicts/caveats, not full-sentence equivalence.
- The two decks' visual/aesthetic layer; the dashboard's Streamlit runtime behaviour.

## Process lessons (for the next sweep)
1. Structured fan-out outputs REQUIRE mechanical QC: 3 incidents (2 placeholder traces, 1
   degenerate verdict) caught only by count-reconciliation / coherence checks. Opus-low is not
   reliable for large structured traces; medium+ with anti-placeholder rules held.
2. The adversarial-refuter layer killed 47 of 62 raised findings — the false-positive rate of
   even xhigh finders on a heavily-hedged corpus is ~75%, and unverified audit output would have
   produced a remediation list 4× too long.
3. The absence-direction (D54) keeps paying: FS-11 (uncited power method) is the same class as
   the uncited-BH that motivated it, caught by design this time.
4. All real cross-surface drift lives in hand-maintained mirrors (tex) — generated surfaces
   stayed in sync; gates must cover the hand-maintained edges (R2/FS-16).

## Remediation execution plan (pending user approval)
- **R1 content batch:** edit the six builders + both tex mirrors per the census above (FS-11
  requires web-verifying Connor 1987's bibliographic details first; every citation insertion
  keeps the first-use lock green); regenerate all six docx; recompile both PDFs; refresh
  Overleaf zips; named copies. One re-verify loop: make verify-claims + surfaces + citation
  gate + pytest + targeted re-greps of every FS site.
- **R2 lock batch:** implement the six gate clusters; each new gate gets a deliberate-perturbation
  must-fire self-test before being trusted (D42); make harness-eval extended accordingly.
- **Close-out:** FS-24 D-row; ledger statuses → remediated (with row links); reconcile vs
  baseline `64e4f4f`; tick T44 in §2; final commit + push; HANDOFF regenerated.
