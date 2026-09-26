# Report Versions — Complete Map

> **Generated file** (`make report-versions`); edit `configs/report_versions.yaml`, not this file.
> Provenance: statuses and decision IDs were transcribed from `docs/PROJECT_LOG.md` §1/§3. Rendered docx are **gitignored local artifacts** (D60) — the tracked source of every document is its builder script.

Every FYP report/thesis/interim version ever built is listed below, grouped into lineages. Each consecutive pair in a lineage has a change record under [`docs/report_versions/changes/`](report_versions/changes/) stating what was added, **removed**, or edited down to individual chapters, sections and paragraphs.

## Full FYP report (docx-js line)  (`report_canonical`)

The original nine-chapter full report, built by the successive build_fyp_report*.js docx-js builders. v5 is the canonical build path (D41); v6 is the remediated comparison draft; earlier versions are archived history.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **First full draft (2026-05-23)** | 🗄 archived | 2026-05-23 | `scripts/build_fyp_report.js` *(name-inferred)* | — | ✅ 46,915 B |
| **Draft update (2026-05-24)** | 🗄 archived | 2026-05-24 | `scripts/build_fyp_report.js` *(name-inferred)* | — | ✅ 51,679 B |
| **Draft update (2026-06-14)** | 🗄 archived | 2026-06-14 | `scripts/build_fyp_report.js` *(name-inferred)* | — | ✅ 104,378 B |
| **v2** | 🗄 archived | 2026-06-26 | `scripts/build_fyp_report_v2.js` *(name-inferred)* | — | ✅ 677,454 B |
| **v3** | 🗄 archived | 2026-06-26 | `scripts/build_fyp_report_v3.js` *(name-inferred)* | — | ✅ 677,737 B |
| **v4 (128-token era)** | 🗄 archived | 2026-06-30 | `scripts/build_fyp_report_v4.js` *(name-inferred)* | — | ✅ 680,810 B |
| **v5 (512-token primary re-base) — Canonical `make report` output; 512-token-primary study.** | ⭐ canonical | 2026-07-01 | `scripts/build_fyp_report_v5.js` | D41 | ✅ 1,397,734 B |
| **v6 (remediated comparison draft) — Live claim surface kept beside canonical v5 for comparison.** | 🔍 comparison draft | 2026-07-25 | `scripts/build_fyp_report_v6.js` | — | ✅ 1,402,732 B |

**Change records:**

- [`First full draft (2026-05-23)` → `Draft update (2026-05-24)`](report_versions/changes/report_canonical__first_full_draft_2026_05_23_to_draft_update_2026_05_24.summary.md) — +37 / −6 / ~10¶
- [`Draft update (2026-05-24)` → `Draft update (2026-06-14)`](report_versions/changes/report_canonical__draft_update_2026_05_24_to_draft_update_2026_06_14.summary.md) — +172 / −36 / ~43¶
- [`Draft update (2026-06-14)` → `v2`](report_versions/changes/report_canonical__draft_update_2026_06_14_to_v2.summary.md) — +36 / −13 / ~30¶
- [`v2` → `v3`](report_versions/changes/report_canonical__v2_to_v3.summary.md) — +0 / −0 / ~133¶
- [`v3` → `v4 (128-token era)`](report_versions/changes/report_canonical__v3_to_v4_128_token_era.summary.md) — +6 / −0 / ~3¶
- [`v4 (128-token era)` → `v5 (512-token primary re-base)`](report_versions/changes/report_canonical__v4_128_token_era_to_v5_512_token_primary_re_base.summary.md) — +141 / −81 / ~100¶
- [`v5 (512-token primary re-base)` → `v6 (remediated comparison draft)`](report_versions/changes/report_canonical__v5_512_token_primary_re_base_to_v6_remediated_comparison_draft.summary.md) — +350 / −132 / ~128¶

## Standalone thesis  (`thesis_line`)

The separate full research thesis (not touched by `make report`). Thesis v4 is the 512-primary mirrored master (D43).

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Thesis v1 (early)** | 🗄 archived | 2026-06-18 | `scripts/build_fyp_thesis.js` *(name-inferred)* | — | ✅ 30,827 B |
| **Thesis v2** | 🗄 archived | 2026-06-26 | `scripts/build_fyp_thesis_v2.js` *(name-inferred)* | — | ✅ 435,540 B |
| **Thesis v3** | 🗄 archived | 2026-06-26 | `scripts/build_fyp_thesis_v3.js` *(name-inferred)* | — | ✅ 435,490 B |
| **Thesis v4 (512-primary mirror) — Output of `make thesis`; standalone thesis master.** | ⭐ canonical | 2026-07-02 | `scripts/build_fyp_thesis_v4.js` | D43 | ✅ 1,140,379 B |

**Change records:**

- [`Thesis v1 (early)` → `Thesis v2`](report_versions/changes/thesis_line__thesis_v1_early_to_thesis_v2.summary.md) — +13 / −3 / ~5¶
- [`Thesis v2` → `Thesis v3`](report_versions/changes/thesis_line__thesis_v2_to_thesis_v3.summary.md) — +0 / −0 / ~26¶
- [`Thesis v3` → `Thesis v4 (512-primary mirror)`](report_versions/changes/thesis_line__thesis_v3_to_thesis_v4_512_primary_mirror.summary.md) — +47 / −33 / ~32¶

## Interim report (docx-js line)  (`interim_classic`)

The registry-backed interim master of 2026-07-10. Superseded for the T53 send action by the rewritten interim line, but still a live claim surface and LaTeX mirror source.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Interim master (2026-07-10) — Superseded for submission by the rewritten interim line (T53).** | 📘 current reference | 2026-07-10 | `scripts/build_fyp_interim.js` | — | ✅ 1,137,174 B |

## Rewritten full report (plain-academic line)  (`rewrite_report`)

The corpus-informed then plain-academic rewrite of the full report, built by Python (python-docx) builders. v3 carries the user-selected voice (D65); v4 adds the HW0288 funnel introduction (D66/D68).

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Rewrite v1 (corpus-informed) — Demoted to reference by the user's D65 voice choice.** | 📎 reference | 2026-07-27 | `scripts/build_rewritten_fyp_report_v1.py` | D57, D58, D59, D65 | ✅ 819,354 B |
| **Rewrite v2 (plain-academic pass) — Dominated by v3 (430/480 shared paragraphs); kept for comparison.** | 🔍 comparison draft | 2026-08-21 | `scripts/build_rewritten_fyp_report_v2.py` | D63, D65 | ✅ 819,149 B |
| **Rewrite v3 (selected plain-academic voice) — The user-selected prose register for the submission line.** | ✅ selected | 2026-08-21 | `scripts/build_rewritten_fyp_report_v3.py` | D64, D65 | ✅ 819,891 B |
| **Rewrite v4 (HW0288 funnel introduction) — v3 plus the author's funnel Chapter 1 and completed remediation.** | ✅ selected | 2026-08-22 | `scripts/build_rewritten_fyp_report_v4.py` | D66, D67, D68, D69 | ✅ 904,524 B |
| **Rewrite v5 (language-cleanup line) — Language-cleanup rewrite of the verified v4 line (T70 closed by its creating session): 137 exact-substring corrections replacing compressed statistical shorthand with conventional academic English. Numbers, citation markers, research questions, headings, tables and figures are asserted equal to v4 at build time; see docs/audits/2026-08-25_language_pass_v5/. Not selected for T53 (progress copy remains the sent artifact); candidate base for the T15 final report.** | 📎 reference | 2026-08-25 | `scripts/build_rewritten_fyp_report_v5.py` | D75 | ✅ 820,335 B |

> ⚠️ Chain note: No v0 exists: this line starts at rewritten_fyp_report_v1.docx (T52/D57).

**Change records:**

- [`Rewrite v1 (corpus-informed)` → `Rewrite v2 (plain-academic pass)`](report_versions/changes/rewrite_report__rewrite_v1_corpus_informed_to_rewrite_v2_plain_academic_pass.summary.md) — +63 / −63 / ~188¶
- [`Rewrite v2 (plain-academic pass)` → `Rewrite v3 (selected plain-academic voice)`](report_versions/changes/rewrite_report__rewrite_v2_plain_academic_pass_to_rewrite_v3_selected_plain_academic_voice.summary.md) — +46 / −46 / ~9¶
- [`Rewrite v3 (selected plain-academic voice)` → `Rewrite v4 (HW0288 funnel introduction)`](report_versions/changes/rewrite_report__rewrite_v3_selected_plain_academic_voice_to_rewrite_v4_hw0288_funnel_introduction.summary.md) — +10 / −14 / ~48¶
- [`Rewrite v4 (HW0288 funnel introduction)` → `Rewrite v5 (language-cleanup line)`](report_versions/changes/rewrite_report__rewrite_v4_hw0288_funnel_introduction_to_rewrite_v5_language_cleanup_line.summary.md) — +16 / −16 / ~132¶

## Rewritten interim report (submission line)  (`rewrite_interim`)

The interim-submission derivatives of the rewrite line. The selected T53 artifact is docs/rewritten_fyp_interim_progress.docx (D72/D73); the others remain as dated fallbacks.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Interim v1 (retitle-only of rewrite v1) — Word-verified fallback (58 pages); rebuild side effect rewrites the v1 report.** | 🧯 fallback | 2026-08-10 | `scripts/build_rewritten_fyp_interim_v1.py` | D59 | ✅ 820,378 B |
| **Interim v3 (retitle-only of rewrite v3) — Superseded for sending by interim v4 (D70).** | 🧯 fallback | 2026-08-22 | `scripts/build_rewritten_fyp_interim_v3.py` | D65 | ✅ 820,905 B |
| **Interim v4 (funnel intro, comprehensive fallback) — Comprehensive 59-page fallback after D70 promotion and T67 remediation.** | 🧯 fallback | 2026-08-22 | `scripts/build_rewritten_fyp_interim_v4.py` | D68, D70 | ✅ 820,919 B |
| **Interim compact (progress-centred) — 36-page build; superseded for sending by the progress copy (D72).** | 🧯 fallback | 2026-08-24 | `scripts/build_rewritten_fyp_interim_compact.py` | D71, D72 | ✅ 800,058 B |
| **Interim progress (SELECTED T53 artifact) — Uses supplied Ch1 + full v4 Ch2-4, replaces detailed scorer/discussion chapters with Ch5 Progress and Remaining Work; repaired in place 2026-08-25 (D73).** | ✅ selected | 2026-08-24 | `scripts/build_rewritten_fyp_interim_progress.py` | D72, D73 | ✅ 735,270 B |
| **Interim v5 (retitle-only of rewrite v5) — Same retitle-only contract as the other interim derivatives, over the v5 language-cleanup line (D75). Comprehensive fallback; the D73 progress copy remains the selected T53 artifact.** | 🧯 fallback | 2026-08-25 | `scripts/build_rewritten_fyp_interim_v5.py` | D75 | ✅ 821,271 B |
| **Interim progress (humanized prose variant) — /humanizer prose pass over the submitted progress copy. 84 exact-substring rewrites thin the machine-prose markers ("therefore" 26 to 8, "rather than" 26 to 12, provides/supplies cycling, participle tails) and simplify ornamental vocabulary at the user's request, holding the academic register. Numbers (663), citation sequence (159), headings (32), table cells (245), references (57) and the TOC assert unchanged at build time; renders to the same 33 pages. Does not supersede the submitted copy, which stays the selected T53 artifact.** | variant | 2026-09-02 | `scripts/build_rewritten_fyp_interim_progress_humanized.py` | D72, D73 | ✅ 735,120 B |
| **Submitted interim (literature review revised) — Argument-led Chapter 2 rewrite, followed by authorised prose edits to the abstract and Chapters 1, 3, 4 and 5. Preserves citations and tables; submitted source untouched. Cached contents refreshed after pagination changes.** | 🔍 comparison draft | 2026-09-05 | `scripts/rewrite_submitted_literature_review.py` | D78 | ✅ 734,033 B |
| **Interim progress v2 (plain academic full-report revision) — Full-report prose edit from the literature-review-revised sibling. Simplifies language, trims repeated explanations and clarifies methods and results. Preserves numerical tables and original citation occurrences; adds four verified method sources and renumbers 61 references by first use. Clarifies two literature-table labels and marks technical-report preprints. Figures are regenerated from verified results with cautious interpretation labels. Evidence audit: docs/audits/2026-09-08_interim_v2_evidence_check/AUDIT.md. Earlier versions and the selected submission copy remain unchanged.** | 🔍 comparison draft | 2026-09-07 | `scripts/build_rewritten_fyp_interim_progress_v2.py` | D81, D82, D83 | ✅ 845,617 B |
| **Interim progress v3 (restructured literature review) — Rebuilds Chapter 2 of v2 fundamentals-first (new 2.1 Quantization Fundamentals with Figure 2.1, merged measurement section, new Kharinaev et al. paragraph, RQ-linked synthesis, new 2.7 Chapter Summary) with section bridges, linking-word prose and explicit links to this study; revises Chapter 1 (motivation, problem statement, joint RQ1-RQ3 framing, MMLU subset justification, contributions). Adds two foundational sources (Jacob et al. 2018; Gholami et al. 2022): 63 references renumbered by first use. Result figures relabelled FP16 with identical data. Chapters 3-5 unchanged apart from citation numbers. TOC pages from a LibreOffice render; Word remains the authority.** | 🔍 comparison draft | 2026-09-26 | `scripts/build_rewritten_fyp_interim_progress_v3.py` | D92, D93 | ✅ 939,967 B |

> ⚠️ Chain note: Verified absence (2026-08-25): no rewritten_fyp_interim_v2.docx was ever built or logged — the interim chain jumps v1 to v3 because the v2 editorial pass was applied to the report line only (D63).

**Change records:**

- [`Interim v1 (retitle-only of rewrite v1)` → `Interim v3 (retitle-only of rewrite v3)`](report_versions/changes/rewrite_interim__interim_v1_retitle_only_of_rewrite_v1_to_interim_v3_retitle_only_of_rewrite_v3.summary.md) — +64 / −64 / ~191¶
- [`Interim v3 (retitle-only of rewrite v3)` → `Interim v4 (funnel intro, comprehensive fallback)`](report_versions/changes/rewrite_interim__interim_v3_retitle_only_of_rewrite_v3_to_interim_v4_funnel_intro_comprehensive_fa.summary.md) — +10 / −14 / ~48¶
- [`Interim v4 (funnel intro, comprehensive fallback)` → `Interim compact (progress-centred)`](report_versions/changes/rewrite_interim__interim_v4_funnel_intro_comprehensive_fa_to_interim_compact_progress_centred.summary.md) — +19 / −166 / ~74¶
- [`Interim compact (progress-centred)` → `Interim progress (SELECTED T53 artifact)`](report_versions/changes/rewrite_interim__interim_compact_progress_centred_to_interim_progress_selected_t53_artifact.summary.md) — +66 / −92 / ~52¶
- [`Interim progress (SELECTED T53 artifact)` → `Interim v5 (retitle-only of rewrite v5)`](report_versions/changes/rewrite_interim__interim_progress_selected_t53_artifact_to_interim_v5_retitle_only_of_rewrite_v5.summary.md) — +199 / −26 / ~86¶
- [`Interim v5 (retitle-only of rewrite v5)` → `Interim progress (humanized prose variant)`](report_versions/changes/rewrite_interim__interim_v5_retitle_only_of_rewrite_v5_to_interim_progress_humanized_prose_variant.summary.md) — +40 / −213 / ~98¶
- [`Interim progress (humanized prose variant)` → `Submitted interim (literature review revised)`](report_versions/changes/rewrite_interim__interim_progress_humanized_prose_variant_to_submitted_interim_literature_review_revi.summary.md) — +63 / −63 / ~63¶
- [`Submitted interim (literature review revised)` → `Interim progress v2 (plain academic full-report revision)`](report_versions/changes/rewrite_interim__submitted_interim_literature_review_revi_to_interim_progress_v2_plain_academic_full_.summary.md) — +100 / −96 / ~42¶
- [`Interim progress v2 (plain academic full-report revision)` → `Interim progress v3 (restructured literature review)`](report_versions/changes/rewrite_interim__interim_progress_v2_plain_academic_full__to_interim_progress_v3_restructured_literat.summary.md) — +51 / −51 / ~37¶

## Humanized alternate — full report  (`humanized_report`)

Submission-alternate re-render of canonical v5 with the humanized styling pass.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Base — report v5** | ⭐ canonical | 2026-07-01 | `scripts/build_fyp_report_v5.js` | D41 | ✅ 1,397,734 B |
| **Humanized report alternate — Separate submission alternate (PROJECT_LOG §1 Current documents).** | 🧯 fallback | 2026-07-01 | `scripts/build_fyp_report_humanized.js` | — | ✅ 1,393,682 B |

**Change records:**

- [`Base — report v5` → `Humanized report alternate`](report_versions/changes/humanized_report__base_report_v5_to_humanized_report_alternate.summary.md) — +0 / −8 / ~145¶

## Humanized alternate — thesis  (`humanized_thesis`)

Submission-alternate re-render of thesis v4 with the humanized styling pass.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Base — thesis v4** | ⭐ canonical | 2026-07-02 | `scripts/build_fyp_thesis_v4.js` | D43 | ✅ 1,140,379 B |
| **Humanized thesis alternate** | 🧯 fallback | 2026-07-02 | `scripts/build_fyp_thesis_humanized.js` | — | ✅ 1,140,076 B |

**Change records:**

- [`Base — thesis v4` → `Humanized thesis alternate`](report_versions/changes/humanized_thesis__base_thesis_v4_to_humanized_thesis_alternate.summary.md) — +1 / −3 / ~56¶

## Humanized alternate — interim  (`humanized_interim`)

Submission-alternate re-render of the 2026-07-10 interim with the humanized styling pass.

| Version | Status | Date | Builder | Decisions | Artifact |
|---|---|---|---|---|---|
| **Base — interim master** | 📘 current reference | 2026-07-10 | `scripts/build_fyp_interim.js` | — | ✅ 1,137,174 B |
| **Humanized interim alternate** | 🧯 fallback | 2026-07-10 | `scripts/build_fyp_interim_humanized.js` | — | ✅ 1,137,739 B |

**Change records:**

- [`Base — interim master` → `Humanized interim alternate`](report_versions/changes/humanized_interim__base_interim_master_to_humanized_interim_alternate.summary.md) — +3 / −2 / ~72¶

## Standalone documents (inventoried, no version chain)

| Document | Note |
|---|---|
| `docs/drafts/final_report_visuals_pack.docx` | Review pack of 15 proposed final-report figures built by docs/drafts/build_visuals_pack.py from docs/drafts/visuals/; not a report version. |
| `docs/drafts/final_report_missing_parts_draft.docx` | Review draft of the missing final-report chapters (5-7) built by docs/drafts/build_final_chapters_draft.py; not a report version and not the final-report base (T71). |
| `docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx` | Word architecture guide for the agent harness. |
| `docs/Codex_Meetup_Agent_Harness_Report.docx` | Meetup report on the repo-native agent harness. |
| `docs/Agentic_AI_Workflow_Report_2026-06-26.docx` | Agentic workflow report deck companion. |
| `docs/archive/FYP_TC1_Full_Session_Report_2026-04-08.docx` | Archived TC1 session record. |
| `docs/archive/FYP_Decision_Register_and_Compressed_Work_Plan_2026-05-13.docx` | Archived early decision register. |
| `docs/archive/FYP_Decisions_and_Plan_2026-05-13.docx` | Archived early decisions and plan. |

## How to read / regenerate

```bash
make report-versions                    # regenerate this map, the inventory and all change records
python scripts/report_tools/report_version_diff.py extract docs/FYP_Report_2026-07-01_v5.docx
python scripts/report_tools/report_version_diff.py diff <old.docx> <new.docx>
```

- `changes/<edge>.summary.md` — committed overview; removed/added/edited paragraph lists with ≤90-char quotes and content hashes.
- `changes/<edge>.full.md` / `.full.json` — **gitignored local files** with complete prose (kept out of git so the public repo never republishes submission text, same rationale as D60).
- Paragraph identity: `sha8` is the first 8 hex of SHA-1 over the whitespace-normalised paragraph text; `NNw` is word count.
- Consistency between this map, the YAML registry and the artifacts on disk is enforced by `tests/test_report_versions.py`.
