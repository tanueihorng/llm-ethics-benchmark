# Adversarial academic audit — 10 July 2026

## Verdict

**Defensible only after specified P1 fixes.**

The 512-token, judge-primary numerical conclusion is internally supported: no HarmBench ASR contrast survives BH-FDR and committed sidecars reproduce the paired statistics. Current reports nevertheless overstate exact from-scratch reproducibility, and the canonical report is visibly identified as an old interim document.

## Evidence-led conclusion

- **READ:** `make verify-claims` passed 61/61 checks; `pytest tests/` passed 339 tests; `make agent-check` passed eight gates on 2026-07-10.
- **READ:** `results_512/analysis/multiple_comparisons.json` has 20 contrasts, three BH survivors (Qwen-2B MMLU −0.090, Llama ARC −0.0316, Phi XSTest over-refusal −0.044), and no ASR survivor.
- **INFERRED:** this supports a bounded direct-request result, not a universal NF4 safety claim or an exact rerun claim.
- **EXTERNAL:** 512 tokens matches HarmBench's standardized generation budget; DirectRequest-only scope remains essential.

## P1 findings

| ID | Finding | Evidence | Falsifiable remediation |
|---|---|---|---|
| P1-01 | Historical model, dataset and environment provenance is insufficient for exact reruns; current reports overstate it. | `results_512` retains model IDs and generation settings, but lacks immutable model revisions, dataset revision/fingerprint, environment lock and config digest; dependencies are lower-bounded and model pins entered Git after the run. | Correct current claims to analysis replayability; add a future-run manifest + schema tests; claim historical equality only with contemporaneous TC1 evidence. |
| P1-02 | Resume can combine changed conditions. | `run_quant_benchmark.py:334-345,400-425` resumes by prompt ID only. | Require matching manifest fingerprint; test seed, revision, device, data and decoding mismatches. |
| P1-03 | Current TC1 entrypoints contradict policy. | `Makefile:62-71`, `fyp_cli.py:150-153,303-305`, `docs/USER_GUIDE.md:49-52` expose head-node Python/ignored-manifest submission. | Remove/hard-fail entrypoints; document direct `sbatch`; test public instructions. |
| P1-04 | Canonical report identity is inconsistent. | Rendered cover/footer and `build_fyp_report_v5.js:253,1553,1623` say “Interim Report” / 24 May 2026. | Align filename, cover, footer, metadata and date in both report builders; regenerate, claim-lock and render-QA. |
| P1-05 | CI skips the full tests and claim lock. | `.github/workflows/agent-harness.yml:19-28` uses `--skip-pytest`. | Add supported CI jobs for `pytest -q` and `make verify-claims`. |

## P2 findings

- Generic pairwise artifacts use the refusal-regex proxy while the dashboard uses judge-primary analysis; make the default judge-primary or suffix proxy artifacts.
- Report says `torch.inference_mode`; implementation uses `torch.no_grad()` (`build_fyp_report_v5.js:579`; `models/generation.py:158-160`).
- Cite the Phi-4-Mini technical report for Phi-4-mini-instruct.
- Align interim cover/declaration date; use strict IEEE bibliography presentation or call it adapted numbered style.

## Research-gap assessment

**INFERRED:** The contribution is useful when narrowly stated: a matched, judge-primary, multi-family comparison shows that a short-budget safety signal did not persist at HarmBench's 512-token reference budget, while capability/over-refusal contrasts did. It does not establish a general causal account of quantization or jailbreak robustness.

## Submission recommendation

Do not submit the current package. Resolve P1-01–P1-05, rerun the claim lock/harness, then obtain a fresh audit of corrected reports, provenance and CI.
