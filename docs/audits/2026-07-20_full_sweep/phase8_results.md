# T44 Phase 8 — adversarial examiner panel results

Session S8, 2026-07-21/22. Workflows `wf_9be3417f-b7e` + domain re-run `wf_eb7ab800-5b9`:
4 hostile lenses (Opus xhigh, no enumeration basis by design, run on the post-Phase-1..7
state), every finding → 2 adversarial refuters calibrated to undergraduate-FYP standards.
31 agents total (~2.5M subagent tokens).

## QC incident #3 (recorded)
The first domain-examiner run returned degenerate output — verdict "at_risk" with ZERO findings
and a placeholder viva question ("q1") — internally incoherent and discarded. Re-run with
anti-placeholder constraints + a verdict-must-match-findings rule returned a coherent
`pass_with_minor_corrections`. Pattern now seen 3× this sweep: structured fan-out outputs must
be sanity-checked (count reconciliation, verdict-findings coherence) before use.

## Examiner verdicts — all four lenses: `pass_with_minor_corrections`

| Lens | Verdict | One-line rationale |
|---|---|---|
| Stats (inference chains) | pass_with_minor_corrections | "exceptionally hedged, heavily audited"; residual wounds all interpretive-layer wording, none touches the two load-bearing conclusions |
| Safety-ML domain | pass_with_minor_corrections (re-run) | "central results sound and unusually well-validated"; residuals are text-level over-reaches, no P1 |
| FYP examiner | pass_with_minor_corrections | contribution clear, RQ traceability good; wounds are wording-level with prepared answers |
| Reproducibility | pass_with_minor_corrections | artifacts self-contained; residuals are documentation-of-entry-points, already FS-tracked classes |

## Findings: 13 raised → 12 killed → 1 survived

**FS-25 (P3, verified)** — internal contradiction: §6.4.1 concludes the Qwen scale-gap is
"parser-dependent rather than MMLU-specific" (strict parser replicates it on ARC −0.343) while
RQ4 calls it "an MMLU-specific result, not a benchmark-independent scale law". One-sentence
alignment fix → Phase 9 R1.

**Kill highlights (the documents defended themselves):** the "unearned causal attribution"
attack died because the report DOES carry the causal disclaimer; the "hidden parser artifact"
attack died because parser-dependence is disclosed 3× and the strict parser makes the loss
LARGER (−0.090 is the conservative end); the "scorer-of-record epistemics" attack died on the
D51 parallel family + scorer-invariant phrasing; the deployment-sentence and data-availability
attacks died on existing disclosures.

## Optional polish (killed as defects at severity, real as one-line improvements — Phase 9 discretion)
1. §7.2: add a SIXTH external-validity bound — Qwen3 pairs run with enable_thinking=false
   (non-default mode; quantization × thinking-mode interaction untested). Both refuters: real, P3.
2. Abstract: soften "a refusal-margin probe independently supports" (body concedes the probe is
   near-chance within the one moving pair; "is consistent with" is the earned strength).
3. §6.6.1: reconcile "corroborates the null across all five pairs" with the same paragraph's
   "directional non-null" reading of Qwen-4B (one clause).
4. README/docs: a short "regenerating the headline numbers" section naming the analysis entry
   points (scripts/multiple_comparisons.py etc.) — addresses the repro examiner's GAP questions.

## Viva-prep sheet (the panel's real deliverable — ~19 real questions; GAP = documents lack a good answer)

**Hardest, with answers available (prep from the cited sections):**
- Why does a scorer you demonstrated broken (regex, 2/63 recall) stay the over-refusal scorer-of-record, and is the 3-survivor count inflated? → §6.5.1 + VALIDATION_INFORMED_FAMILY_NOTE (pre-registration + scorer-invariant restatement + D51 parallel family).
- What rules out a genuine Llama alignment improvement? → §6.10/§6.11.5 + the imported disclaimer; concede correlational.
- Qwen-4B multiseed: one seed shows a significant harmful-ward +0.050 — why isn't the greedy null a decode-config artifact? → §6.6.1: not a majority, sub-MDE, disclosed as directional non-null.
- The −0.090 vs strict −0.293 bracket: which number should a deployer believe? → §6.4.1/§6.5: the bracket IS the answer; lenient = conservative end.
- Regex misses 97% of true refusals — what does the reported over-refusal rate measure? → §3.5 + Result 6: a coarse template-phrase construct, which is exactly why the judge-strict parallel family exists.
- Two-point precision ladder confounds bit-width with method — how is "not bit-width-graded" earned? → §6.15 frames this exactly; concede descriptive-only.

**GAPs (no good in-document answer — prepare a spoken answer or add a line):**
- Thinking-mode: both Qwen3 pairs run non-default (enable_thinking=false); quantization × thinking-mode interaction untested (polish item 1 closes this).
- Abstract's "independently supports" for the margin probe (polish item 2 closes this).
- Hong et al. as "nearest analogue" evaluates GPTQ/AWQ, which §7.2 says may differ — reconcile the corroboration's weight.
- "Regenerate the headline in front of me": no documented command chain for the BH table (polish item 4 closes this).
- The power-bounded-null framing: be ready to state plainly what effect sizes the study CANNOT rule out (MDE 0.044–0.086) and why that's disclosed, not hidden.

**Phase 8 verdict: no new P1/P2 content defects — the unknown-unknowns pass found one P3
internal contradiction and a viva-prep map. The documents largely defend themselves.**
