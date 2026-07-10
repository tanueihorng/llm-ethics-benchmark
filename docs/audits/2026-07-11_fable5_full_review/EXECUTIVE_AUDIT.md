# Executive Audit — fyp_quant (CCDS25-1136)
**Date:** 2026-07-10 (UTC+8) · **Lead:** Fable 5 (main agent) + 24 fresh-context workflow verifiers · **Mode:** adversarial, evidence-led, read-only diagnostic.

> **Convergence note:** an independent Codex audit earlier the same day (`docs/audits/2026-07-10_adversarial_academic_audit/`) reached the **same verdict** ("Defensible only after specified P1 fixes") by a separate path. This Fable-5 review ran without reading Codex's findings first; the agreement (no P0; results reproduce; P1s are report-integrity + construct-validity + packaging) is corroboration. Overlapping findings (phi float label, §6.14 flips, stale 128-era CIs, document identity, A4/format) are double-confirmed.

## VERDICT: Defensible only after specified P0/P1 fixes.
There is **no P0** finding and **no fatal validity threat**. Every headline result reproduced **exactly** from the committed primary artifacts under independent recomputation (main agent + multiple verifiers). The science is sound, the protocol is faithfully executed and substantially standards-conformant, and the disclosure discipline is well above typical FYP level. What blocks a "submission-ready" verdict is a set of **P1 report-integrity contradictions, human-grounding construct-validity caveats, and presentation/packaging defects** — all fixable without re-running any experiment, and none of which changes a conclusion.

"Submission-ready" is withheld solely because open P1s remain, per the audit's own gate.

## Coverage
- In-scope tracked files: **838**. Reviewed: **146 fully**, **512 executed/scripted-scan**, **81 partial**, **99 not reviewed** (archive, vendored skill examples, historical task packets, binary figures, tool caches — enumerated in `UNRESOLVED_OR_UNVERIFIED.md` and `CODE_COVERAGE_MANIFEST.csv`).
- Method: 24 of 25 fresh-context verifier agents returned evidence ledgers (the 25th, semantic citation-support, was cut by a session-limit and its load-bearing checks were run inline by the main agent). Every P1 below was re-verified by the main agent against primary artifacts. This project's history shows expert auditors confabulate (two refuted in the 2026-07-02 round), so nothing here is accepted on an agent's say-so alone.
- Machine gates re-run this audit: `make agent-check` 8/8 (339 tests, 300/300 immutable hashes clean), `verify_report_claims.py` 61/61. **These verify machine-checkable consistency, not correctness** — see the P1s the gates do not cover.

## What is SOUND (independently reproduced — see REPRODUCTION_MATRIX.md)
- **Every headline ΔASR**: qwen_2b 0.000 (p=1.000, 16/16), llama −0.040 (p=0.0215, the only individually-significant, a *decrease*), qwen_4b +0.040 n.s., mistral/phi n.s. — exact.
- **BH-FDR**: exactly 3 survivors (qwen_2b MMLU, llama ARC, phi over-refusal); **no ASR contrast survives** — reproduced by own step-up over all 20 contrasts.
- Judge-vs-regex κ (10), gpt-4o κ 0.68–0.95, human gold-set 2×2 (κ 0.59/0.11), multiseed per-seed, 128-era +0.055/INT8 +0.040 and their vanishing at 512, sample counts, manifest hashes, the HarmBench "N=512" quote — all exact/verbatim.
- Matched-pair fairness invariant holds (shared model_id/seed/decoding; quant is the only variable); NF4/INT8 genuinely applied with a fail-loud guard; redaction airtight; the 21 references are **all real (zero fabricated)**.

## P1 findings (fix before submission) — grouped
**A. Report internal contradictions (results correct; the prose/labels disagree with the artifacts):**
1. **RQ3 answer overclaims FDR survival** — states Qwen-4B ARC loss "survive[s] multiplicity correction"; it does not (p=0.042 uncorrected; only 3 survivors, Qwen-4B ARC is not one). [RESULTS ledger]
2. **§6.14 refusal-margin flip numbers are 128-era but labelled "budget-invariant"** — `refusal_margin.json` in the 512 tree is a byte-copy of the 128 file; flip counts (92; qwen 16-vs-5) contradict §6.3/Ch10's 16/16 at the primary budget. Margins are budget-invariant; the flip *set* is not. Conclusion (boundary instability) survives.
3. **phi4_mini label `robust_preservation` is a float-rounding artifact** — exact ΔASR = +0.02 ≥ tol ⇒ the report's own Table 3.4 rule gives `alignment_degradation`; the stored float 0.0199…9 gives `robust_preservation`. (n.s. either way.)
4. **Stale 128-era MMLU CI** `[−0.137,−0.037]` in the 512-scoped §6.3 prose, contradicting the same report's Table 6.1 `[−0.140,−0.043]`.
5. **Test-count self-contradiction** — "339 automated tests" then "All 329 tests pass" (also in the humanized twin).

**B. Human-grounding construct validity (caps what "human-validated" may claim):**
6. **Annotator saw responses truncated to 2000 chars** while both scorers saw full text (114/200 items), and the tool told the annotator the shown text equalled what the scorers saw — false; concentrated in the disagreement cells that define κ.
7. **Stratified-sample κ published without reweighting**, with the deflation caveat applied one-sidedly to the classifier; on a population-weighted basis the 0.59-vs-0.11 contrast narrows (≈0.8 vs ≈0.5). Single annotator = the author (identity undisclosed).

**C. Presentation / packaging / integrity-of-record:**
8. **Document-identity confusion** — the 36.7k-word "Report v5" is cover-labelled *Interim Report*, the "Thesis" is a 7.6k-word condensation shorter than the interim; CCDS deliverable mapping is ambiguous.
9. **Stale gitignored LaTeX submission channel** — the school-recommended Overleaf PDFs (`fyp_submission/report_latex/**`) are 128-era (`+0.055` as current), outside every gate; would submit the retired truncation artefact as the finding. (main-agent finding)
10. **No Declaration of Originality / AI-use declaration in the report** (master + humanized), despite self-documented AI-agent authorship; signed Declarations elsewhere overclaim "reproducible from the committed configuration"/"pinned dependencies".
11. **Format non-conformance** — US Letter not A4, 1.15–1.25 not 1.5 spacing, blank Word TOC field, report body ≈3× the guide's 10k-word main-body expectation.
12. **`REPRODUCIBILITY.md` still describes the retired 128-era study** as what a third party reproduces.
13. **Humanized variants corrupt en-dash ranges into comma lists** in κ table cells (semantic change; masters unaffected).

## Selected P2 (explain or fix)
Classifier uses the model-card template not the GitHub-harness variant (undisclosed; matched-pair delta immune, absolute ASRs not literature-comparable) · **MMLU parser asymmetry**: 49% of qwen_2b_4bit items scored by the weakest fallback vs 3% for its fp16 twin — undisclosed, and qwen_2b MMLU −0.090 is a headline survivor (direction corroborated by ARC) · committed `pair_interpretations.json` (v2-proxy) contradicts the headline (Mistral "alignment_degradation, significant") with no in-file superseded marker · INT8@512 judge-agreement artifact is all-null and §6.15's κ-ordering claim is partly false (Phi not among the highest) · qwen_2b_base @512 HarmBench is a smoke+resume two-job composite, contradicting "every delta computed within a single run" · environment versions machine-recorded nowhere; revision pins retroactive · closest prior work (Hong et al., *Decoding Compressed Trust*, ICML 2024) uncited · CITATION.bib is pre-pivot init metadata ("[Your University]", toxicity/bias/factuality) · `make smoke` still defaults to force_restart into the protected results tree. Full list in the ledgers.

## Research-gap assessment
The hedged conjunction contribution ("no prior study combines matched-pair on-the-fly quantization + judge-validated primary scorer + over-refusal axis + capability anchor + multi-precision on compact models") was **adversarially hunted and not falsified**; no unhedged "first/only/novel" claim exists. The contribution is correctly a **methodological** one (scorer choice and generation budget each determine the safety conclusion; capability is the robust cost), which is the right framing given a mostly-null primary result. Four *supporting* claims overreach (see METHODS_VS_STANDARDS.md) and Hong et al. should be cited, but the core gap is real and correctly stated.

## Submission recommendation
Fix the 13 P1s (est. focused effort: report-builder prose/label edits + one config disclosure + A4/declarations/TOC in the builders + decide the submission channel and refresh or retire the LaTeX + re-score or re-scope the human gold set + regenerate). None require TC1 compute. After the P1 fixes and a re-verification pass, the work moves to **"Defensible with stated limitations"** and, with the packaging conformed, is a strong FYP submission. The underlying study does not need to be re-run.

## Deliverables in this folder
`CODE_COVERAGE_MANIFEST.csv` (838 rows) · `METHODS_VS_STANDARDS.md` · `RESULTS_CLAIM_LEDGER.csv` (27 claims) · `CITATION_FORENSICS.csv` (22 refs) · `REPORT_FORMAT_AND_STYLE.md` · `REPRODUCTION_MATRIX.md` (18 recomputations) · `UNRESOLVED_OR_UNVERIFIED.md`.
