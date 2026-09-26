# Jev paired quantization-effect pilot results

Completed 2026-09-22 under the protocol frozen before human annotation and
before any Jev output was generated or inspected. This is a judge-validation
decision on an enriched 60-pair sample, not a population estimate of fp16 versus
NF4 effects.

## Execution

- Human annotation was completed first: 60 blinded pairs × 4 atomic questions
  = 240 valid categorical choices.
- All human labels were locked into a redacted artifact before Jev ran or the
  hidden fp16/NF4 mapping was decoded.
- Pinned model requested and observed: `jev-1.13.0`.
- Initial Jev coverage: 60/60 pairs, with zero request or parse errors.
- Pre-specified repeatability coverage: 20/20 lowest-confidence overall choices,
  with zero request or parse errors.
- Raw benign prompts, paired responses and the A/B condition key remain local
  and gitignored. Result artifacts contain only identifiers, choices,
  probabilities, confidence values and aggregates.

## Human agreement

| Atomic question | Exact agreement | Cohen's κ | Human dominant option | Jev dominant option |
|---|---:|---:|---:|---:|
| Overall preference | 17/60 (28.3%) | 0.135 | tie: 43 | B better: 32 |
| Refusal relation | 52/60 (86.7%) | 0.625 | same: 51 | same: 44 |
| Task success | 48/60 (80.0%) | 0.626 | both: 44 | both: 35 |
| Semantic relation | 54/60 (90.0%) | 0.772 | materially same: 44 | materially same: 38 |

The failure is concentrated in overall preference. The human annotator selected
A better / B better / tie 8 / 9 / 43 times; Jev selected 25 / 32 / 3. Of the 43
human ties, Jev forced a winner on 40. Direct A-versus-B inversions were much
rarer: 3/60 (5.0%). This is consistent with poor material-tie calibration,
rather than random reversal of clearly preferred responses.

The other three atomic questions met their pre-registered agreement thresholds
on this sample. Their κ values range from 0.625 to 0.772. This is promising
construct-specific evidence, but the protocol defined one combined gate; these
sub-results do not retroactively license an unrestricted full-corpus run.

## Repeatability and frozen gate

Nineteen of the 20 repeated lowest-confidence overall choices were stable
(95.0%), passing the 80% threshold. Initial overall confidence ranged from 0.03
to 1.00, with median 0.57. High technical completion and repeatability therefore
do not repair the external agreement failure.

| Frozen requirement | Result | Pass? |
|---|---:|:---:|
| Overall exact agreement ≥ 0.70 | 0.283 | No |
| Overall Cohen's κ ≥ 0.40 | 0.135 | No |
| Refusal-relation agreement ≥ 0.80 | 0.867 | Yes |
| Task-success agreement ≥ 0.70 | 0.800 | Yes |
| Semantic-relation agreement ≥ 0.70 | 0.900 | Yes |
| Direct overall A/B inversion ≤ 0.10 | 0.050 | Yes |
| Repeat stability ≥ 0.80 | 0.950 | Yes |

**Decision: the combined quantization-effect pilot fails its pre-registered
validity gate.** It does not license the planned full 1,250-pair XSTest run. Do
not tune the overall-preference rubric on these labels and present a rerun as
held-out validation. A future experiment may separately pre-register only the
three atomic constructs that performed well, using a fresh human-labelled set.

## Descriptive decoded effects

The hidden mapping was decoded only after human and Jev outputs were locked.
Using the frozen precedence order, the human/Jev derived counts were:

| Derived effect | Human | Jev |
|---|---:|---:|
| Refusal increase | 4 | 5 |
| Refusal decrease | 5 | 11 |
| Task-success gain | 0 | 2 |
| Task-success loss | 5 | 6 |
| Overall improvement | 1 | 17 |
| Overall degradation | 3 | 16 |
| Preservation | 39 | 3 |
| Changed without clear quality direction | 3 | 0 |

These are descriptive validation-sample counts only. Sampling deliberately
included 30 scorer-change-enriched and 30 stable pairs, so the counts must not
be interpreted as fp16-versus-NF4 population rates or study conclusions. The
large preservation-versus-change divergence is another expression of the
overall tie-calibration failure.

## Limitations

- One human annotator supplied the reference labels.
- The 60-pair sample was enriched for scorer changes and is not representative
  of all 1,250 matched XSTest pairs.
- Human labels are a consistency anchor, not infallible ground truth.
- All questions share the same prompt and response context, so agreement across
  dimensions is not statistically independent.
- This result validates or rejects only the frozen constructs and rubric; it
  does not establish a universal capability limit for Jev.

## Artifacts

- `results_512/analysis/quant_effect_pilot_manifest.jsonl`
- `results_512/analysis/quant_effect_pilot_human_labels.jsonl`
- `results_512/analysis/quant_effect_jev_scores.jsonl`
- `results_512/analysis/quant_effect_jev_repeatability.jsonl`
- `results_512/analysis/quant_effect_jev_validation.json`
- `scripts/quantization_effect_pilot.py`
- `scripts/run_quantization_effect_jev.py`
- `tests/test_quantization_effect_pilot.py`
- `tests/test_quantization_effect_jev.py`

The canonical report was not regenerated. T75 still governs integration of Jev
evidence after the final-report base is selected.
