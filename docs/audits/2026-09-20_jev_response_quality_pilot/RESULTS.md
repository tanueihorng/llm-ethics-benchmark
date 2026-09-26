# Jev paired response-quality pilot results

Completed 2026-09-22 under the protocol frozen on 2026-09-20. This result is a
validation decision for one paired-response rubric, not a population estimate
of fp16 versus NF4 response quality.

## Execution

- Human annotation was completed first: 60 blinded pairs × 7 dimensions = 420
  valid A/B/tie choices.
- The labels were locked into a redacted artifact before any Jev
  response-quality output was generated or inspected.
- Pinned model requested and observed: `jev-1.13.0`.
- Initial Jev coverage: 60/60 pairs, zero request or parse errors.
- Pre-specified repeatability coverage: 20/20 lowest-confidence overall choices,
  zero request or parse errors.
- Raw benign prompts, paired responses and the hidden A/B condition key remain
  local and gitignored. Result artifacts contain only identifiers, choices,
  probabilities, confidence values and aggregates.

## Human agreement

| Dimension | Exact agreement | Cohen's κ | Human ties | Jev ties |
|---|---:|---:|---:|---:|
| Correctness | 37/60 (61.7%) | 0.423 | 20 | 3 |
| Task fulfilment | 32/60 (53.3%) | 0.343 | 32 | 12 |
| Relevance | 25/60 (41.7%) | 0.216 | 35 | 8 |
| Completeness | 34/60 (56.7%) | 0.376 | 25 | 6 |
| Clarity | 31/60 (51.7%) | 0.336 | 30 | 4 |
| Safe helpfulness | 18/60 (30.0%) | 0.114 | 41 | 6 |
| Overall quality | 40/60 (66.7%) | 0.438 | 10 | 3 |

For overall quality, the human annotator selected A/B/tie 24/26/10 times and
Jev selected them 24/33/3 times. The disagreement matrix shows that Jev often
forced a winner where the human annotator saw no material difference: eight of
the ten human overall ties became A or B under Jev. This is a construct-level
rubric mismatch, especially on safe helpfulness and relevance, rather than an
API reliability problem.

## Repeatability and frozen gate

The 20 lowest-confidence overall judgments were repeated once. Eighteen of 20
labels were stable (90.0%), passing the frozen 80% threshold. Overall-choice
confidence across the initial run ranged from 0.11 to 1.00, with median 0.65.

| Frozen requirement | Result | Pass? |
|---|---:|:---:|
| Overall exact agreement ≥ 0.70 | 0.667 | No |
| Overall Cohen's κ ≥ 0.40 | 0.438 | Yes |
| Every component agreement ≥ 0.55 | minimum 0.300 | No |
| Repeat stability ≥ 0.80 | 0.900 | Yes |

**Decision: the pilot fails its pre-registered validity gate.** The rubric does
not license a full 1,250-pair XSTest response-quality run. Do not tune on these
same 60 labels and present the retuned result as held-out validation. A future
retry would require a newly frozen rubric and an independent held-out human
sample, with special attention to material-tie calibration.

## Descriptive condition choices

After labels were locked, the hidden condition mapping was decoded for aggregate
description only. On overall quality, the human annotator preferred fp16/NF4/tie
30/20/10 times; Jev preferred them 34/23/3 times. Because the sample was enriched
for scorer changes, includes only 60 pairs and was designed for judge validation,
these counts are not an fp16-versus-NF4 effect estimate and must not be used as a
study conclusion.

## Artifacts

- `results_512/analysis/response_quality_pilot_human_labels.jsonl`
- `results_512/analysis/response_quality_jev_scores.jsonl`
- `results_512/analysis/response_quality_jev_repeatability.jsonl`
- `results_512/analysis/response_quality_jev_validation.json`
- `scripts/run_response_quality_jev.py`
- `tests/test_response_quality_jev.py`

The canonical report was not regenerated. T75 still governs integration of the
validated and negative Jev studies after the final-report base is selected.
