# Methods versus standards

## Reconstructed protocol

**READ:** `configs/tc1_512.yaml` defines five matched fp16/NF4 pairs, greedy decoding, seed 42 and 512 new tokens. `slurm/jobs_tc1_512/` uses this config. Canonical evidence is `results_512/` plus redacted judge sidecars; 128-token `results/` is historical comparison evidence.

| Question | Audit status | Evidence and assessment |
|---|---|---|
| Matched pair | Controlled | Aligned prompt IDs, same model IDs, seed and stored generation settings were observed for five fp16/NF4 pairs. |
| HarmBench budget/scorer | Controlled | **EXTERNAL + READ:** HarmBench standardizes N=512 ([paper](https://arxiv.org/abs/2402.04249)); classifier role matches [official card](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls). DirectRequest-only. |
| XSTest scope | Controlled | **EXTERNAL + READ:** [XSTest](https://aclanthology.org/2024.naacl-long.301/) has 250 safe prompts; the project evaluates those 250 benign prompts. |
| NF4 settings | Controlled | **EXTERNAL + READ:** loader matches documented [NF4/double-quant/fp16 settings](https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit). |
| Primary ASR inference | Controlled | 20 BH contrasts; three non-ASR survivors. Exact McNemar is paired significance; bootstrap intervals estimate uncertainty. |
| Scorer construct validity | P2 | One-annotator, disagreement-enriched human validation makes classifier-vs-human κ=0.59 informative but not a gold standard. |
| Exact historical rerun | P1 | Post-run model pins; lower-bound packages; unpinned datasets; no run manifest. Offline cache does not prove fresh-clone identity. |
| Generalisation | P2 | Bound to these models, NF4/INT8, greedy decoding, four benchmarks, DirectRequest prompts and finite samples. |

## Design interpretation

The design can answer whether recorded matched NF4 contrasts co-occur with judge-primary harmful compliance and capability differences. It cannot identify a mechanism or general safety effect. The research contribution is defensible only as this bounded, budget-sensitive empirical result.

## Analysis naming risk

`pairwise_deltas.json` and `pair_interpretations.json` are regex-proxy HarmBench artifacts; `multiple_comparisons.json` and dashboard reconstruction are judge-primary. Canonical naming/output must be repaired before external reuse.
