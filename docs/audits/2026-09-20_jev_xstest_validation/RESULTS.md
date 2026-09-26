# Jev XSTest validation — complete run log

Run date: 2026-09-20 (UTC+8)  
Protocol: [PROTOCOL.md](PROTOCOL.md)  
Jev model: `jev-1.13.0` (pinned and observed)  
Role: sensitivity evaluation only

## Outcome

The integration succeeded technically, but Jev did **not** validate as an
operational XSTest refusal judge under this rubric. Jev completed all 3,750
saved responses without an API or parse error, yet its human-anchor refusal
recall was much lower than GPT-4o's. The existing GPT-4o three-class judge
remains the better validated XSTest sensitivity scorer; the prespecified v2
regex remains the scorer of record under the existing no-swap decision.

This is useful negative evidence about domain transfer. Strong HarmBench
harmful-compliance performance did not transfer automatically to nuanced benign
refusal classification.

## Execution record

- Frozen protocol: three-class Choice (`full_compliance`, `partial_refusal`,
  `full_refusal`) plus parallel Nouls for substantive compliance and refusal.
- Synthetic gate: 3/3 requests returned successfully. The deliberately mixed
  case was labelled full compliance with confidence 0.45, correctly exercising
  the uncertainty path even though it did not select the intended middle class.
- Full run: 15 aliases × 250 records = 3,750/3,750 answered; zero errors.
- First two aliases ran sequentially. The interrupted third alias wrote no
  partial sidecar; resumable execution then completed the 13 remaining aliases
  in four non-overlapping process groups.
- Inputs were the existing immutable XSTest `raw.jsonl` records. Only new
  `scores.judge.xstest_jev.jsonl` and `summary.judge.xstest_jev.json` sidecars
  were written.
- The API key was supplied in memory through the environment and was not
  written to any repository artifact or command output.

## Full-corpus comparison with GPT-4o

| Mapping | Agreement | Cohen's κ | Jev positives matching GPT-4o | GPT-4o positives missed by Jev |
|---|---:|---:|---:|---:|
| Three class | 87.04% | 0.405 | — | — |
| Strict refusal | 87.52% | 0.385 | 178 | 463 |
| Broad refusal | 87.49% | 0.421 | 209 | 465 |

The high raw agreement is dominated by the majority full-compliance class.
Directional disagreement is strongly asymmetric: 449 records were Jev full
compliance versus GPT-4o full refusal, while only two went in the opposite
direction. Treating GPT-4o as the comparator, Jev strict refusal precision was
0.973 but recall only 0.278. These are comparator statistics, not ground-truth
accuracy.

Per-alias three-class agreement ranged from 81.6% to 94.0%, while κ ranged from
0.078 to 0.677. This variability and the low-prevalence class imbalance are why
agreement alone is insufficient.

## Human-labelled validity anchor (n=200)

| Judge/scorer | Strict κ | Strict precision | Strict recall | Strict F1 | Broad κ | Broad F1 | Three-class agreement | Three-class κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | 0.210 | 0.722 | 0.206 | 0.321 | 0.119 | 0.271 | 57.0% | 0.154 |
| GPT-4o | 0.485 | 0.540 | 0.968 | 0.693 | 0.662 | 0.841 | 69.5% | 0.496 |
| v2 regex | −0.006 | 0.286 | 0.032 | 0.057 | 0.054 | 0.115 | n/a | n/a |

Jev caught 13 of 63 human-labelled full refusals, compared with GPT-4o's 61 of
63 and the regex's 2 of 63. It produced only five strict false positives, so its
failure mode is conservative under-detection rather than indiscriminate
over-flagging. Jev is better than the regex on strict κ and F1, but far behind
GPT-4o and therefore does not earn operational-judge status for XSTest.

The human set is single-annotator and disagreement-enriched. Its metrics are a
validity contrast on contested cases, not population performance estimates.

## fp16–NF4 contrasts under Jev

| Pair | Strict ΔOR | Strict 95% CI | Broad ΔOR | Broad 95% CI |
|---|---:|---:|---:|---:|
| Qwen 1.7B | +0.012 | [−0.016, +0.040] | +0.008 | [−0.020, +0.036] |
| Qwen 4B | −0.012 | [−0.032, +0.004] | −0.012 | [−0.032, +0.008] |
| Llama 3.2 3B | +0.004 | [−0.020, +0.028] | +0.004 | [−0.020, +0.028] |
| Mistral 7B | +0.004 | [−0.008, +0.020] | −0.008 | [−0.024, +0.008] |
| Phi-4-mini | 0.000 | [−0.028, +0.028] | −0.012 | [−0.044, +0.016] |

No Jev XSTest contrast is significant by the paired bootstrap, and all exact
McNemar p-values are at least 0.375. In particular, Jev does not reproduce the
regex Phi decrease: strict ΔOR is exactly zero and broad ΔOR is −0.012 with a
zero-crossing interval. This agrees directionally with the earlier conclusion
that the regex survivor is scorer-dependent, but Jev's weak human validity
means it supplies no stronger basis for that conclusion than GPT-4o already
does.

## Probability and uncertainty diagnostics

- Choice confidence: mean 0.952, median 1.00; 161/3,750 below 0.60.
- Review queue: 552 records, triggered by low confidence or GPT-4o disagreement.
- Choice versus Noul-derived strict mapping: 98.77% agreement, κ=0.860.
- Choice versus Noul-derived broad mapping: 98.53% agreement, κ=0.858.

The internal Jev questions are coherent with each other, but this does not
establish external validity. The combination of high internal confidence and
low human refusal recall is a warning against using uncertainty alone to detect
systematic construct mismatch.

## Artifacts

- Per-alias: `results_512/<alias>/xstest/scores.judge.xstest_jev.jsonl`
- Per-alias summaries: `summary.judge.xstest_jev.json`
- Aggregate: `results_512/analysis/xstest_jev_validation.json`
- Redacted review queue: `results_512/analysis/xstest_jev_review_queue.jsonl`
- Regex agreement and paired deltas:
  `results_512/analysis/xstest_judge_agreement_jev.{json,csv}`

All per-item outputs contain IDs, labels, probabilities and provenance only.
They contain no prompt, response, generation or provider-response text.

## Decision

Do not designate Jev as an XSTest operational judge from this pilot. Retain the
existing GPT-4o judge for validation-informed XSTest sensitivity work and the
v2 regex as the historically prespecified scorer of record. Jev remains useful
for domains where it has separate construct-specific validation, including the
completed HarmBench role. Any future XSTest retry must change and pre-freeze the
rubric, then revalidate on held-out human labels rather than tune on this same
200-item set.
