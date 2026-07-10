# Report, Thesis & Interim — Format and Academic-Writing Audit (2026-07-10)

Documents audited (all rendered text extracted from the committed .docx; builders inspected for geometry):
`FYP_Report_2026-07-01_v5.docx`, `FYP_Thesis_2026-07-02_v4.docx`, `FYP_Interim_2026-07-10.docx`, and the three `*_humanized.docx` variants.

Rubric basis: the repo's own captured `fyp_submission/report_latex/SCHOOL_GUIDELINES.md` (NTU SCSE/CCDS "Guidelines for Writing the FYP Report", captured by the student + Chng Eng Siong's thesis-writing notes) — **40–65 pp final / body ≤10,000 words / interim ≈50% of final / 1.5 line spacing / A4 / mandatory title page + chart-form schedule**. Where the school brief is not in-repo, IEEE/common conventions are used and labelled as such. **The student must still confirm the current CCDS brief — treat the captured guide as the best available evidence, not the authoritative live rubric.**

## Format findings (docx channel)

| Sev | Finding | Evidence |
|---|---|---|
| P1 | **Page size US Letter, not A4.** All six docx are 12240×15840 twips (8.5×11") with 1" margins. NTU standard is A4 (11906×16838) with 35 mm/30 mm margins. | build_fyp_report_v5.js PAGE_W=12240; thesis/interim builders size {12240,15840} |
| P1 | **Report body length far over the guide.** Report v5 ≈29.8k body words / ~100 rendered pp vs the guide's 10,000-word main-body expectation. (The thesis/interim are ~7.6k/8.1k — in range for the guide's body but see the identity confusion below.) | word counts of extracted text |
| P1 | **No Declaration of Originality anywhere in the report; no AI-use declaration in the report** (master or humanized), despite the report carrying self-documented AI-agent scaffolding (Appendix H addressed "to a future agent"). Thesis + interim DO carry an AI-use declaration. | grep of extracted text; §Appendix H |
| P2 | **Stale running header "24 May 2026"** on every page of report v5 + humanized, contradicting the 2 July cover date. | header extraction |
| P2 | **Line spacing 1.15–1.25**, guide requires 1.5. | builder spacing params |
| P2 | **TOC renders blank** in the delivered docx (Word TOC field with no cached entries / no updateFields) — a direct PDF export shows an empty contents page. List of Figures/Tables is hand-typed without page numbers. | TOC field in extracted text |
| P2 | **Cover pages omit degree title and "submitted in partial fulfilment" line** (guide's title-page sample is mandatory). | cover extraction |
| P3 | Report body is Calibri 11pt; thesis/interim Times New Roman 12pt (template convention). Abstracts 505–619 words (over one page). Reference list renders "1." decimal while the preamble promises bracketed [1]. Page numbering is a single arabic run starting on the cover (no roman front matter). | builder fonts; extracted text |

## Document-identity confusion (P1 — the highest-value format finding)

The deliverable set does not map cleanly to CCDS deliverable names:
- The 36.7k-word **"FYP_Report … v5"** is titled/handled as the **Interim Report** on its own cover, yet is thesis-scale.
- The 7.6k-word **"FYP_Thesis … v4"** is a *condensation* (shorter than the interim it is supposedly senior to).
- The 8.1k-word **"FYP_Interim"** is derived from the thesis builder.
- "FYP: THESIS" is not a CCDS deliverable name; two documents self-identify as the interim.

**Consequence:** it is currently ambiguous which file the student submits as the interim vs the final report, and the labels contradict the sizes. This must be resolved before T15.

## Cross-channel hazard (P1, main-agent finding — outside every gate)

The school guidance repeatedly names the **Overleaf LaTeX PDFs** (`fyp_submission/report_latex/**`, gitignored) as "the submission deliverable." Those LaTeX sources are **128-era stale** (`max_new_tokens 128`; headline `qwen_2b +0.055 … yes … broad_degradation`; "only significant increase is in the smallest model"), last edited 2026-06-29 — before the 512-primary promotion — and carry **no AI-use declaration** (deleted 2026-06-27). Because `fyp_submission/` is gitignored, no stale-text/claim-lock gate covers it. If the student follows the school's own Overleaf guidance, they would submit the retired truncation-artefact headline as their finding.

## Academic-writing / prose

- Structure, chapter-overview framing, RQ-driven organisation, and limitations discipline are **strong** — well above typical FYP level; the DirectRequest threat model and power/MDE null-framing are disclosed repeatedly and correctly.
- Overreach to fix: causal phrasing in Ch10/Table 3.4 beyond the correlational label rule; "budget-invariant" applied to §6.14 flip numbers; "reproducible from the committed configuration"/"pinned dependencies" in signed Declarations (see RESULTS ledger + P1 list).
- Humanized variants: dash-removal converted en-dash **ranges** to comma **lists** in κ table cells and a page range ("pp. 153 to 157") — a semantic corruption; and no record fixes which variant (master vs humanized) is canonical for submission.

## Bottom line (format)
The intellectual content is presented to a high scholarly standard, but the **packaging is not yet submission-conformant**: A4/margins/spacing, the blank TOC, the report's missing originality+AI declarations, the length-vs-guide gap, and above all the **document-identity confusion + the stale gitignored LaTeX submission channel** must be resolved. None of these touch a result; all are pre-submission blockers of the presentation layer.
