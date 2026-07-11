# Independent clean-evidence audit — 2026-07-11

## Verdict

**Defensible only after specified P0/P1 fixes.**

No P0 was independently established. Four P1 findings remain unresolved; P1-1 and P1-2 are independently sufficient to prevent a submission-ready verdict. P1-1 is an internal agent handoff in the document labelled a “Comprehensive Project Report”; P1-2 is incomplete historical provenance. The current configuration pins main NF4 revisions, but it cannot retrospectively prove the provenance of artifacts generated before those keys were persisted.

## Evidence-led findings

| ID | Severity | Category | Conclusion and evidence | Falsifiable remediation |
|---|---|---|---|---|
| P1-1 | P1 | Submission document | **READ.** Rendered page 108 of `docs/FYP_Report_2026-07-01_v5.docx` contains “H.5 Where the next session should continue”, agent instructions, task IDs, paths and TC1 operational instructions. This is incompatible with a polished submission master. Builder: `scripts/build_fyp_report_v5.js` (Appendix H). | Remove the internal handoff appendix from the submission builder; rebuild and visually inspect the resulting document. |
| P1-2 | P1 | Reproducibility/provenance | **READ/INFERRED.** Direct inspection of historical `results_512/*/*/raw.jsonl` and `summary.json` found no resolved revision, package-version, dataset-fingerprint, quantisation-configuration or dtype field. Separately, `ethical_benchmark/pipeline/run_quant_benchmark.py:443-464` writes current `quant_method`/`torch_dtype` but still no resolved revision, package-version, dataset fingerprint or raw hash. A present config cannot prove the state that made an earlier artifact. | Obtain a signed TC1 provenance sidecar (checkpoint/tokenizer SHAs, dataset hashes, exact dependency/CUDA/GPU state, BnB config, raw hashes) or qualify/re-run the headline experiment. |
| P1-3 | P1 | Precision reproducibility | **READ.** All executed INT8 entries in `configs/tc1_int8_512.yaml:20-125` omit `revision`, while `configs/tc1_512.yaml:4-114` pins them. The INT8 comparison is therefore not reproducibly tied to the fp16/NF4 checkpoint snapshots. | Pin identical per-model revisions in both INT8 configs; assert this schema invariant in tests. |
| P1-4 | P1 | Primary scoring provenance | **READ.** `ethical_benchmark/judges/validation.py:265-301` loads the classifier and tokenizer from a mutable model ID; `slurm/judge_validation_512.sbatch:54-64` does not supply a revision. | Add a judge revision parameter, pass it to both tokenizer/model loads, persist it in the judge output, and test all three behaviors. |
| P2-1 | P2 | Run integrity | **READ.** Resume logic at `run_quant_benchmark.py:383-440` trusts existing IDs and does not fingerprint a partial run; `zip(batch, prompts, responses)` can silently truncate a short generator output. | Persist/validate a run fingerprint, expected-ID coverage and response cardinality; add negative tests. |
| P2-2 | P2 | Construct validity | **READ/EXTERNAL.** HarmBench uses its intended classifier, but XSTest over-refusal remains regex-scored (`benchmarks/xstest.py`, `benchmarks/utils.py`) rather than independently validated. XSTest’s safe-prompt construct is documented by [Röttger et al.](https://aclanthology.org/2024.naacl-long.301/). | Blindly human-label a stratified XSTest sample or validate against an independent refusal judge. |
| P2-3 | P2 | Power/generalisation | **READ.** `results_512/analysis/multiple_comparisons.json` reports 200 HarmBench items and per-pair ASR MDEs of 0.044–0.086. Multi-seed coverage is only three pairs (`configs/tc1_sensitivity_512.yaml`). | State a smallest effect of interest and bounded-null conclusion; extend sample/seed coverage, including Mistral and Phi. |
| P2-4 | P2 | CI/reproducibility | **READ.** `requirements.txt` contains lower bounds and `.github/workflows/agent-harness.yml:19-25` does not run the experiment unit suite. | Commit a TC1 constraints/lock file and a dependency-compatible pytest CI job. |
| P2-5 | P2 | Data governance | **READ.** When invoked, the API backend at `judges/validation.py:463-538` sends behavior/generation text externally; redacted output sidecars mitigate storage, not egress. It is not evidence that this backend was used for the headline scorer. | Retain an approval/retention record or use a local independent judge. |

## Positive controls

- **READ.** The main `tc1_512.yaml` matches baseline/NF4 model IDs and revisions within every pair; the loader uses on-the-fly quantisation.
- **READ.** `multiple_comparisons.json` applies paired exact McNemar tests and BH-FDR across 20 contrasts; three non-ASR contrasts survive.
- **READ (audit command transcript, not retained in package).** `python scripts/verify_report_claims.py` reported 62/62 checks, `pytest -q` reported 339 tests, and `scripts/agent_check.py` reported eight checks. These validate coded assertions and selected repository invariants, not historical execution provenance, scientific validity, or institutional submission acceptance.

## Safe checks run

| Check | Result | Validates | Does not validate |
|---|---|---|---|
| `git status --porcelain` | empty | no in-flight tracked remediation | completeness/correctness of past remediation |
| `verify_report_claims.py` | audit transcript reported 62 pass | selected builder/artifact numerical consistency | all prose, figures, historical provenance |
| `pytest -q` | audit transcript reported 339 pass | unit/integration behavior covered by tests | GPU execution, archival results, methods validity |
| `agent_check.py` | audit transcript reported 8 pass | repository policy/integrity checks | submission fitness or claim truth beyond coded rules |
| raw JSONL duplicate-ID scan | audit transcript reported 60 files; 0 duplicate-ID findings | duplicate `prompt_id` within current primary raw files | correct prompt universe, provenance, scorer correctness |
