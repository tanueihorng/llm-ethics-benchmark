# Jev full-corpus validation results

**Date:** 2026-09-20 (UTC+8)  
**Protocol:** `PROTOCOL.md` in this directory  
**Model:** requested `jev-1.13.0`; observed `jev-1.13.0` on every successful request  
**Interpretation:** additional probabilistic sensitivity judge; official HarmBench classifier remains primary

## Executive result

Jev successfully scored all 3,000 saved HarmBench DirectRequest responses across the five fp16, five NF4 and five INT8 aliases with zero errors. It showed high agreement with the official HarmBench classifier (96.23%, κ=0.890) and the existing GPT-4o API judge (94.83%, κ=0.841). On the existing 200-item human-labelled validation set, Jev achieved 87.0% agreement, κ=0.602 and F1=0.683—the strongest κ and F1 of the four automated judges on this particular single-annotator, disagreement-enriched sample, though only slightly above the official classifier (κ=0.591, F1=0.675).

The principal project conclusion is not reversed: no Jev-scored fp16→NF4 pair shows a statistically confirmed **increase** in harmful compliance. Jev does make the Llama-3.2-3B decrease stronger: ΔASR −0.050, 95% paired-bootstrap interval [−0.085, −0.020], McNemar p=0.00195, and BH q=0.0098 in the explicitly separate Jev sensitivity family. This is a harmful-compliance decrease accompanying the already documented capability decline, not evidence of improved alignment or a safety degradation.

## Execution and coverage

| Item | Result |
|---|---:|
| Model aliases | 15 |
| Records per alias | 200 |
| Full-corpus requests | 3,000 |
| Answered | 3,000 |
| Errors / skipped | 0 / 0 |
| Decision threshold | `P(harmful compliance) >= 0.50` |
| Jev probability range | 0.01–0.98 |
| Mean / median probability | 0.202 / 0.01 |
| Within ±0.05 of threshold | 51 |
| Within ±0.10 of threshold | 113 |

The later repeatability check made 113 additional calls. A separate two-item `jev-latest` smoke check confirmed that the moving alias resolved to `jev-1.13.0` on the run date. The fixed identifier was retained for the scientific artifacts so later reruns cannot silently mix provider versions.

## Full-corpus comparison with existing judges

Jev is the candidate column in the directional counts below.

| Reference judge | Shared n | Agreement | Cohen's κ | Both harmful | Both safe | Reference harmful / Jev safe | Reference safe / Jev harmful | Reference rate | Jev rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official HarmBench classifier | 3,000 | 96.23% | 0.890 | 604 | 2,283 | 72 | 41 | 22.53% | 21.50% |
| GPT-4o API judge | 3,000 | 94.83% | 0.841 | 534 | 2,311 | 44 | 111 | 19.27% | 21.50% |
| LlamaGuard | 3,000 | 87.43% | 0.673 | 579 | 2,044 | 311 | 66 | 29.67% | 21.50% |

The marginal label rates differ significantly over 3,000 paired records (Jev vs classifier McNemar p=0.00455; vs GPT-4o p=7.28×10⁻⁸; vs LlamaGuard p=3.89×10⁻³⁹). These tests show systematic threshold/construct differences despite high item-level agreement; they do not identify which judge is correct.

### Agreement with the official classifier by precision

| Precision | Shared n | Agreement | Cohen's κ | Classifier rate | Jev rate |
|---|---:|---:|---:|---:|---:|
| fp16 | 1,000 | 97.20% | 0.919 | 22.50% | 21.70% |
| NF4 | 1,000 | 96.10% | 0.888 | 22.50% | 22.40% |
| INT8 | 1,000 | 95.40% | 0.864 | 22.60% | 20.40% |

Per-alias classifier agreement ranges from 92.0% to 98.5%; per-alias κ ranges from 0.781 to 0.928. The full machine-readable table is in `results_512/analysis/jev_full_evaluation.json`.

## Human-labelled validation subset

The existing sample has 200 single-annotator labels, 37 marked harmful. It deliberately overrepresents prior classifier-versus-regex disagreements, so these metrics compare judges on a difficult validation sample rather than estimate corpus-wide accuracy.

| Judge | Agreement | κ | Precision | Recall | F1 | Harmful calls | False alarms vs human | Misses vs human |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Jev** | **87.0%** | **0.602** | 0.622 | 0.757 | **0.683** | 45 | 17 | 9 |
| Official classifier | 86.5% | 0.591 | 0.609 | 0.757 | 0.675 | 46 | 18 | 9 |
| GPT-4o API judge | **87.0%** | 0.569 | **0.649** | 0.649 | 0.649 | 37 | 13 | 13 |
| LlamaGuard | 72.5% | 0.322 | 0.371 | 0.703 | 0.486 | 70 | 44 | 11 |

Jev and the classifier identify the same number of human-positive cases (28/37) and miss the same nine; Jev makes one fewer false-positive call. That one-record difference explains the small κ/F1 advantage and should not be overstated. Jev's probability metrics on this subset are Brier score 0.0856 and log loss 0.2654. Calibration interpretation is limited by the non-representative sample composition.

## Threshold analysis

| Jev threshold | Harmful calls / 3,000 | κ vs human | F1 vs human | κ vs classifier | κ vs GPT-4o |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 689 | 0.591 | 0.675 | 0.900 | 0.843 |
| 0.45 | 664 | 0.602 | 0.683 | 0.890 | 0.838 |
| **0.50** | **645** | **0.602** | **0.683** | **0.890** | **0.841** |
| 0.55 | 613 | 0.610 | 0.684 | 0.879 | 0.833 |
| 0.60 | 585 | 0.615 | 0.685 | 0.861 | 0.830 |

The human-subset metrics change only slightly across 0.45–0.60. The numerically highest human κ/F1 occurs at 0.60, but choosing it after inspecting this same small sample would overfit. The transparent 0.50 rule remains the reported operating point; future calibration should use a separately frozen validation/calibration split.

## Borderline repeatability

All 113 records within ±0.10 of 0.50 were immediately re-judged with the same pinned model and rubric:

| Metric | Result |
|---|---:|
| Answered / errors | 113 / 0 |
| Binary agreement | 88.50% |
| Threshold flips | 13 |
| Mean absolute probability change | 0.0215 |
| Median absolute probability change | 0.0200 |
| Maximum absolute probability change | 0.0700 |

This is good probability-level stability but only moderate hard-label stability near an arbitrary boundary. Borderline records should therefore be routed for review or interpreted probabilistically rather than treated as certain Boolean facts.

## Matched fp16→NF4 results under Jev

| Pair | fp16 ASR | NF4 ASR | ΔASR | 95% paired-bootstrap interval | McNemar p | Jev-family BH q |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.225 | 0.245 | +0.020 | [−0.035, +0.075] | 0.5966 | 0.7458 |
| Qwen3-4B | 0.115 | 0.145 | +0.030 | [−0.010, +0.070] | 0.2101 | 0.4307 |
| Llama-3.2-3B | 0.100 | 0.050 | **−0.050** | **[−0.085, −0.020]** | **0.00195** | **0.0098** |
| Mistral-7B | 0.570 | 0.595 | +0.025 | [−0.035, +0.085] | 0.5114 | 0.6818 |
| Phi-4-mini | 0.075 | 0.085 | +0.010 | [−0.030, +0.050] | 0.8036 | 0.9454 |

The explicitly named 20-test Jev sensitivity family has four BH survivors: the existing Qwen MMLU decline, Llama ARC decline, Phi XSTest over-refusal decrease, and the Jev Llama harmful-compliance decrease. Only the HarmBench axis was substituted; the primary registered family is untouched. No harmful-compliance **increase** survives.

## Review queue

`results_512/analysis/jev_review_queue.jsonl` contains 530 unique redacted records:

- 113 within ±0.10 of the Jev threshold;
- 493 with disagreement among Jev, classifier, GPT-4o or LlamaGuard;
- 26 where Jev disagrees with the available human label.

The categories overlap. Each row contains only model alias, prompt ID, family, precision, probability, judge labels, optional human label and reason codes. Human review has not yet been performed.

## Usage and cost record

Before the full run, the TypeSafe Usage page showed 44 requests, 34,974 tokens and estimated spend of $0.0014. After the full run and repeatability/latest checks, the page warned that statistics may be delayed and showed 1,277 requests, 927,497 tokens and $0.0377. The completed local artifacts account for 3,000 full-corpus requests, 113 repeatability requests, two `jev-latest` smoke requests and the earlier 44 pilot requests (3,159 successful requests total). Because the dashboard had not caught up, $0.0377 is an interim provider estimate, not the final billed total.

## Artifacts

- Fifteen `results_512/<alias>/harmbench/scores.judge.jev.jsonl` sidecars.
- Fifteen matching `summary.judge.jev.json` files.
- `results_512/analysis/jev_full_evaluation.json`.
- `results_512/analysis/multiple_comparisons_jev_sensitivity.json`.
- `results_512/analysis/jev_repeatability.jsonl` and `jev_repeatability_summary.json`.
- `results_512/analysis/jev_review_queue.jsonl`.

No raw benchmark text, generation text, provider response body or API key is present in these artifacts.

## Decision and next use

This evaluation is sufficient to designate Jev as the project's fourth operational HarmBench judge and first probability-preserving judge. It is not sufficient to replace the official classifier or erase the limitations of the single-annotator validation set.

The next planned application is a **separate report-audit evaluation**. It should use atomic Noul/Choice questions for source entailment, numerical support, causal overstatement and limitation presence, evaluated against a manually verified audit set. HarmBench success is evidence that the integration works, not evidence that the report-audit rubric is valid.

Report integration of these validation results is tracked separately as T75 after T71 selects the final-report base. This avoids mixing the new evidence into unrelated dirty report-builder work or implicitly choosing a deliverable line.
