# Comprehensive audit of `docs/rewritten_fyp_interim_progress.docx` — 2026-08-25

**Audited artifact:** `docs/rewritten_fyp_interim_progress.docx` (the submitted interim copy, SHA-era D73 repair of the student-edited document).
**Trigger:** user concern about (a) hallucinated claims, (b) non-existent or wrong in-text citations, (c) wrongly included reference-list entries, (d) TOC correctness.
**Method note (D54/D56 symmetry):** this audit ran BOTH directions — inward from every claim to its source (fitness), and outward from every bibliography entry to authoritative records (identity/presence) — plus an absence pass (uncited entries, phantom citations, dangling cross-references). Every finding below was re-verified against primary artifacts by the delegating session before acceptance.

## Verdict

**No hallucinated claims, no non-existent sources, no wrong-source citations, no uncited or phantom references, and a fully correct TOC were found.** Two defects exist, one substantive-but-small (a dangling cross-reference to deleted appendix material) and one bibliographic ambiguity (defensible author order on [33]). Details below.

## 1. TOC — PASS
- 32 static TOC entries ↔ 32 bookmarked headings: text identical, order identical.
- Every entry is a working internal hyperlink (`w:hyperlink w:anchor="toc_NNNN"`) to an existing bookmark.
- All 32 page numbers match an independent LibreOffice render (`tmp/docx_renders/progress_audit_20260825/rewritten_fyp_interim_progress.pdf`): body restarts at Chapter 1 = page 1; front matter i–iii; 33 physical = 3 front matter + 30 numbered pages, matching D73's claim.
- Standing caveat (D62): Microsoft Word remains the pagination authority; this verifies the shipped numbers against an independent render, not against Word.

## 2. Citation mechanics — PASS
- Bibliography: exactly 57 entries, numbered [1]–[57], contiguous, no duplicates.
- In-text: all 57 cited at least once; zero uncited entries; zero phantom citations; all ranges ascending+contiguous; first-use order strictly [1]…[57] (verified programmatically over `tmp/audit_progress_20260825/docx_dump.txt`).
- No tracked changes, no comments, no external hyperlinks in the body XML.

## 3. Claim-to-source fitness — PASS (106/106 pairs SUPPORTED)
Three parallel auditors opened all linked source PDFs via pdftotext and judged each (sentence, citation) pair:
- Chunk 1 (Ch.1 + §2.1–2.2, refs [1]–[28]): 45/45 SUPPORTED. Exact hits confirmed inside papers: AWQ "0.1%–1% salient weights" + explicit mixed-precision rejection; InstructGPT 1.3B-preferred-over-175B; Marchisio Japanese automatic-vs-human loss gap; Proskurina low-confidence concentration. Two claims flagged honestly as resting partly on absence-of-evidence ([24]/[25] "without a task-specific calibration set" — zero 'calibration' mentions in QLoRA; LLM.int8 "loaded, converted, used immediately").
- Chunk 2 (§2.3–2.6 + Table 2.1 rows, refs [29]–[46]): 47/47 SUPPORTED. Numbers verified inside papers: SORRY-Bench 44 topics / 440 instructions / 20 augmentations / 7,040 annotations ("440 * (8 ID + 8 OOD) = 7,040 human annotations"); OR-Bench 80k prompts / 10 categories / ~1,000 hard / 600 toxic / 32 models × 8 families; Do-Not-Answer 939 prompts / 5,634 responses / ≤600M classifier; MT-Bench >80% agreement + four biases; XSTest three-class labels; Table 2.1 "Not reported" cells checked as true absences (e.g., zero safety/refusal mentions in AWQ/GPTQ/SmoothQuant).
- Chunk 3 (Ch.3 methods, refs [7], [47]–[57]): 14/14 SUPPORTED; stats-classic identities resolved against Crossref/Semantic Scholar (URLs recorded in `chunk3_results.json`).
Delegator spot-checks that confirmed the riskiest items personally: SORRY-Bench 7,040 figure; OR-Bench 600-toxic/80K/Hard-1K/32×8; Kharinaev IEEE identifiers; HarmBench PMLR 235 stamp; six stats DOIs re-resolved directly via Crossref API.

## 4. Reference identities — PASS (0 MAJOR_MISMATCH)
Full 57-ref sweep (`identity_results.json`; arXiv API ×31 IDs, Crossref/OpenAlex for all DOIs, PMLR v70/v202/v235/v267 + MLSys + NeurIPS indexes, OpenReview v1+v2, DBLP tie-breaker, page-1 pdftotext of all 50 held PDFs): **56 MATCH, 1 MINOR_ISSUE ([2] GPTQ proceedings title variant "OPTQ…" vs cited arXiv title "GPTQ…" — same work/venue/year; common practice), 0 MAJOR_MISMATCH, 0 UNVERIFIABLE.**
Special cases resolved:
- [33] Q-Misalign: exact ICLR 2025 camera-ready title; venue genuine. Author-order ambiguity documented under Finding F2.
- [42] OR-Bench ICML 2025 CONFIRMED (PMLR 267:11515–11542 + own PDF footer "PMLR 267").
- [10] Kharinaev IEEE Access vol. 14 / pp. 96771–96793 / 2026 / DOI 10.1109/ACCESS.2026.3703899 verified in the held IEEE version-of-record PDF itself.
- [13] TrustLLM "Y. Huang, L. Sun" matches the printed byline (Yue Huang, Lichao Sun, …).
- [38] StrongREJECT DOI resolves (Adv. NeurIPS 37, 2024); [43], [44], [28] page ranges exact.

## 5. Numeric/factual claims vs artifacts — PASS
84/84 table cells (rates, deltas, CIs, p/q across Tables 4.1–4.2) verified against `multiple_comparisons.json`, `judge_agreement.json`, `pairwise_deltas.json`, `summary.v2.json` + judge summaries; prose discordant counts (16/16; 13/5 p=0.0963; 9/1 p=0.0215); INT8 paths (Qwen3-4B 0.115→0.125→0.155; Mistral INT8=NF4=0.565; Llama INT8 cls +0.005 p=1.000 / API +0.010 p=0.688; all 10 INT8 ASR contrasts n.s. under both scorers); equivalence ±0.05 all True; MDE exactly Qwen-1.7B MMLU + Llama ARC above threshold; validation-informed family = 2 capability survivors; Phi judge ΔOR +0.016 strict / −0.004 broad; κ 0.59 vs 0.11; XSTest human κ 0.485 vs −0.006; multiseed 5 pairs × seeds 1–5; parse errors independently recounted = 0 across 45 judge summaries; §3.4's "gpt-4o snapshot not recorded" is ACCURATE (HarmBench-side judge ran under undated alias `gpt-4o`, `scripts/run_judge_validation.py:57`). Registry gate: `verify_report_claims.py` 93/93.

## Findings

### F1 (P2, content) — dangling cross-reference "Table A.1"
§3.1 (docx ¶P0138): "Table A.1 records the quantization settings and the declared dependency constraints." The appendices were removed from this build; no Table A.1 exists. The builder's stale-text assertion blocks "Appendix A/B/C"/"Table 5.3"/"Table 6.1" but not "Table A.", so both the build and the D73 repair missed it. **Fix:** reword to point at the revision-pinned configs (as the build already did for the other Table-A.1 sentence), then rebuild/re-render. Not a hallucination or citation defect; a leftover from the appendix removal.

### F2 (P3, bibliographic ambiguity) — [33] author order
Printed ICLR 2025 byline: Haowei Li* , Peiran Dong* , Song Guo (equal-contribution asterisks). DBLP official record: "Peiran Dong, Haowei Li, Song Guo". The document cites "P. Dong, H. Li, and S. Guo" — matching DBLP, differing from the printed byline. Defensible either way; if strict byline order is preferred at final-report stage, swap to "H. Li, P. Dong, and S. Guo". Work/title/venue/year are correct; not a wrong source.

### Cosmetic notes (no action required for correctness)
- [2] GPTQ: ICLR 2023 proceedings metadata records the title as "OPTQ: Accurate Quantization for Generative Pre-trained Transformers"; entry uses the arXiv/camera-ready "GPTQ…" title with resolving arXiv ID. Common practice.
- Cover logo uses INCLUDEPICTURE with a remote logo.wine URL (cached image embedded). Harmless unless Word field-updates with network access.
- Unrelated repo-hygiene item noticed en route: `verify_claim_surfaces.py` reports `docs/fyp_showcase_latentspace.html` as an unregistered surface (the only failing check in `make verify-claims`, 558/559).

## Enumeration basis & blind spots (declared)
- Fitness coverage: every paragraph containing bracket citations (39 paragraphs) + all 13 citation-bearing Table 2.1 rows → 106 (unit, ref) pairs; every one of the 57 refs appears in ≥1 audited pair.
- Identity coverage: all 57 entries, all fields. Limitation: web_search was unavailable to the delegated sessions (auth errors), so corroboration rests on direct arXiv/Crossref/OpenAlex/PMLR/OpenReview/DBLP fetches and the 50 held PDFs; [48]/[56]'s local scans are HTML-as-PDF, so their identity rests on Crossref alone; [51]/[54] end pages and [47]'s print year could not be confirmed beyond start-page/e-edition records (flagged, immaterial).
- Pagination verified against LibreOffice render only (D62: Word is the submission authority).

## Evidence trail
`tmp/audit_progress_20260825/`: `docx_dump.txt` (full structural dump), `citation_units.json`, `ref_pdf_mapping.json`, `chunk{1,2,3}.json` inputs, `chunk{1,2,3}_results.json`, `identity_results.json`; render at `tmp/docx_renders/progress_audit_20260825/`.

## Remediation (2026-08-25, post-audit)

**F1 — FIXED.** `scripts/build_rewritten_fyp_interim_progress.py` now (a) replaces the dangling sentence in both paths (`build_interim()` replacements + `repair_current_document()` repairs) with: "The quantization settings and declared dependency constraints are specified in the versioned configuration and requirements files used for every run."; (b) blocks the whole class by adding `"Table A."` to the `_assert_structure` stale-text list.
Applied via `repair_current_document()` to the student-edited document (Word edits preserved). Post-fix verification: "Table A." occurrences = 0; new sentence renders exactly once; 57 references intact; strict first-use citation order re-verified; physical pages still 33; TOC page numbers still 32/32 correct against a fresh LibreOffice render (`tmp/docx_renders/progress_audit_20260825/rewritten_fyp_interim_progress.pdf`, re-generated). Targeted suites pass (126 tests: rewritten-report v3/v4 + claim-fitness + citation-completeness).
**F2** left unchanged at user discretion ([33] entry follows DBLP official record; printed byline lists equal-contribution Li first).
