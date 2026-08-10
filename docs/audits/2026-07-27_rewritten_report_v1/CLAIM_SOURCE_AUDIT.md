# Claim and Source Audit — `rewritten_fyp_report_v1`

Date: 2026-07-27; remediation verified 2026-07-28 (Asia/Singapore)  
Audited artifact: `docs/rewritten_fyp_report_v1.docx`  
Source: `scripts/build_rewritten_fyp_report_v1.py`  
Status: evidence-bounded submission candidate; separate from the canonical v5 report

## 2026-07-28 independent-review remediation

The six document-level concerns recorded after the initial audit are closed:

1. All numeric negatives use the Unicode mathematical minus sign; structural hyphens remain ordinary hyphens.
2. Table 2.1 identifies its compilation basis, uses “Not reported” where the reviewed source does not state a feature, and explicitly disclaims evidence of absence, uniqueness, or priority.
3. References is an unnumbered Heading 1.
4. The populated contents list is stored as the cached result of a genuine Word `TOC` field, and the document requests field refresh on open.
5. `tests/test_citation_completeness.py` now executes the Python builder, reads its `REFERENCES`, and includes a must-fire omission test; `tests/test_rewritten_report_v1.py` guards the report-specific document mechanics.
6. The rebuilt DOCX rendered as 56 A4 pages with the bundled LibreOffice renderer; all 56 page images were visually inspected with no clipping, overlap, blank TOC, or pagination defect.

These corrections change presentation and verification coverage only. No project result, statistical verdict, bibliography item, or scientific claim changed.

## 2026-07-28 entailment-verification addendum

An independent Codex pass rechecked the three confirmed source-fitness findings
in `docs/audits/2026-07-27_claim_sweep/rewrite_fitness_confirmed.json` against
the report and the primary sources:

- **Jin et al. [8].** The official ACL record describes knowledge and capacity,
  alignment, and efficiency as three critical evaluation dimensions. The
  report therefore correctly limits the narrower
  perplexity/accuracy/latency/memory statement to [5]-[7], while retaining
  [8] only for the separate four-bit task-retention claim.
- **SORRY-Bench [42].** The archived paper defines a 44-class taxonomy of unsafe
  topics and explicitly formulates refusal evaluation as binary. The report
  now attributes finer response granularity to Do-Not-Answer's six action
  categories [45] and XSTest's full-compliance/partial-refusal/full-refusal
  labels [18], rather than to SORRY-Bench.
- **XSTest / OR-Bench [18], [19].** The archived XSTest paper includes 200
  unsafe contrasts and reports their three-way outcomes; OR-Bench includes 600
  toxic prompts and reports toxic-prompt rejection. Table 2.1 therefore uses
  “Contrast set only” rather than “Not reported” for harmful compliance and
  keeps the over-refusal/calibration boundary explicit.

The corrected wording is present in both the Python builder and the generated
DOCX. `tests/test_rewritten_report_v1.py` now pins all three corrections,
including the absence of their former unsupported wording.

### Coverage boundary

The earlier audit reported 88 citation-bearing sentences, 12 candidates, and
three confirmed findings, but only the three confirmed records were persisted.
The 88-item inventory and the nine refuted candidates are not present, so that
coverage total cannot be reconstructed independently from the repository.
A fresh extraction of the current DOCX finds 85 citation-bearing prose
sentences before References plus eight citation-bearing Table 2.1 rows
(93 citation-bearing units); the difference is attributable to sentence
splitting and whether table rows are counted. This addendum therefore verifies
the three remediations and their sources. It does **not** convert the
confirmed-only ledger into proof that every citation-bearing unit has received
an independently reproducible entailment verdict.

## Audit standard

This audit asks two different questions:

1. Does each load-bearing project claim match the committed result artifact that produced it?
2. Does each literature citation support the sentence in which it is used, at the stated scope?

A passing string or citation-presence check is not treated as proof of source entailment. Verification reduces hallucination risk; it does not turn an empirical report into an absolute guarantee about untested models, attacks, languages, or effects below the design's resolution.

## Project-result claims

The report loads its primary pair results and multiplicity verdicts from `generated/claim_registry.json`. It loads validation and sensitivity details from redacted analysis sidecars rather than copying raw model output into the report.

| Claim family | Authoritative artifact(s) | Verified report boundary |
|---|---|---|
| Primary fp16–NF4 rates, deltas, intervals, p/q values | `generated/claim_registry.json`; `results_512/analysis/multiple_comparisons.json` | Twenty registered contrasts; exactly three BH-FDR survivors; no HarmBench ASR survivor |
| HarmBench scorer choice and human grounding | `results_512/analysis/judge_agreement.json`; `human_validation.json` | Classifier κ 0.5908 versus regex κ 0.1095 on the 200-item audit sample; sample prevalence is not presented as population ASR |
| Independent HarmBench judges | `judge_pairwise_agreement.json`; `judge_pairwise_agreement_llamaguard.json` | API-judge per-alias κ 0.6780–0.9539; Llama Guard κ 0.3615–0.9178; no corrected harmful-ward NF4 conclusion is claimed |
| XSTest human validation | `xstest_human_validation.json`; `xstest_judge_agreement.json` | Judge-strict κ 0.4848 versus regex κ −0.0063; 103/34/63 three-class counts; disclosed prompt-wide blindness-exclusion deviation and ≤14/200 displacement bound retained |
| Scorer-dependent Phi result | claim registry; XSTest judge sidecars | Regex ΔOR −0.048 is named as a registered survivor, but the semantic judge gives +0.016 strict and −0.004 broad, both non-significant |
| Generation budget | `genlen_robustness.json` | The 128-token Qwen3-1.7B +0.055 result is explicitly historical; the 512-token result is 0.000; 60.3% direct-prefix truncation is scoped to the 2,000 HarmBench responses |
| Multi-seed sensitivity | `sensitivity_multiseed.json` | Five seeds per pair; no stable harmful-ward effect is claimed; seeds are not described as independent model replicates |
| INT8 precision point and equivalence | `precision_sweep.json`; `capability_equivalence.json` | All ten INT8 capability contrasts satisfy the post-hoc TOST analysis at the stated ±0.05 margin; the report does not call INT8 lossless or quantify an unmeasured efficiency benefit |
| Capability interpretation | claim registry; `capability_mde.csv` | Qwen3-1.7B MMLU −0.090 and Llama ARC −0.0316 are the corrected capability survivors; other nulls are not converted into equivalence claims |

The generated DOCX is registered as a live `docx_primary` surface. The claim-surface verifier checks the headline result locks, the source/DOCX freshness relationship, and stale generation-budget language. The builder itself is also a registry-consumer surface.

## Literature claims

The rewrite incorporates the earlier literature-review draft and its evidence matrix:

- `Examples of Past FYP Papers/Verified Literature Sources - Quantization and LLM Safety/Literature_Review_Expansion_Draft_Quantization_and_LLM_Safety.docx`
- `Examples of Past FYP Papers/Verified Literature Sources - Quantization and LLM Safety/Verified_Source_Evidence_Matrix_Quantization_and_LLM_Safety.docx`
- the fifteen archived primary PDFs under the adjacent `papers/` directory
- the earlier claim-to-source ledger at `report_semantic_audit/Report_Claim_Source_Fitness_Ledger_v5_v6_snapshot.csv`

The fifteen archived papers cover GPTQ, AWQ, SmoothQuant, OmniQuant, Q-Misalign, shallow safety alignment, Safety Layers, StrongREJECT, JailbreakBench, SORRY-Bench, WildGuard, OR-Bench, Do-Not-Answer, MT-Bench, and Prometheus 2. Load-bearing facts were rechecked against their full text, including:

| Reported fact | Primary-source check |
|---|---|
| GPTQ performs one-shot weight quantization and reports models up to 175B | GPTQ paper |
| AWQ's analysis identifies roughly 0.1%–1% salient weights; the implemented protection is activation-aware scaling | AWQ paper |
| Q-Misalign is an adversarially prepared, quantization-conditioned failure rather than an estimate of ordinary PTQ prevalence | Q-Misalign paper |
| SORRY-Bench uses 44 unsafe topics, 440 instructions, 20 augmentations, and 7,040 human annotations in its evaluator study | SORRY-Bench paper |
| OR-Bench contains 80,000 prompts, ten categories, an approximately 1,000-prompt hard subset, 600 toxic controls, and evaluates 32 models from eight families in the final ICML study | OR-Bench paper |
| MT-Bench reports over 80% agreement in its tested setting and documents position, verbosity, self-enhancement, and reasoning limitations | MT-Bench paper |
| Prometheus 2 treats judge agreement as indirect and recommends pairing model-based evaluation with human evaluation | Prometheus 2 paper |
| JailbreakBench measures a target together with a specified attack/evaluation procedure | JailbreakBench paper |
| WildGuard's moderation constructs differ from HarmBench's behaviour-specific completion criterion | WildGuard and HarmBench papers |

All 64 bibliography entries are cited. Reference numbering, first-use order `[1]` through `[64]`, and exact-duplicate checks pass. Items without an archival venue are labelled as preprints rather than presented as peer-reviewed evidence.

## Semantic corrections applied

The rewrite resolves the 26 source-fitness issues recorded for the earlier v5/v6 snapshot. The consequential corrections are:

- removed unsupported claims about what users “typically” receive, default runtime settings, market prevalence, and automatic privacy/latency/cost benefits;
- corrected OR-Bench from 25 to 32 evaluated models;
- described the XSTest regex as an adapted pre-specified scorer, not a reproduction of XSTest's primary human protocol;
- replaced undisclosed architectural-independence claims with “separately operated API judge” and described Llama Guard by its different construct;
- labelled refusal-boundary and safety-layer explanations as hypotheses rather than causal findings about NF4;
- replaced “lossless” INT8 language with the explicit post-hoc TOST result and stated margin;
- retained the registered Phi over-refusal survivor while distinguishing the validation-informed family, whose two survivors are both capability outcomes;
- replaced universal or superlative statements with model-, benchmark-, threat-model-, and detection-bound language;
- stated that pairing controls checkpoint exposure history but does not prove immunity to benchmark contamination;
- removed any suggestion that direct requests establish jailbreak robustness.

## Remaining claim boundaries

The report does not claim:

- exact safety equivalence or absence of effects smaller than the study's detectable range;
- adversarial or jailbreak robustness;
- universal behaviour across all quantizers, formats, languages, checkpoints, or model scales;
- causal identification of an internal refusal mechanism;
- population prevalence from disagreement-enriched human samples;
- measured memory, speed, energy, privacy, or cost benefit;
- a multi-annotator human gold standard.

These boundaries are deliberate. They are the conditions under which the report's claims remain fitted to the available evidence.
