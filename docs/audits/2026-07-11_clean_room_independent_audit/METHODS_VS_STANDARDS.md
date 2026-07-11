# Methods versus standards

| Dimension | Assessment | Evidence |
|---|---|---|
| Matched NF4 comparison | Controlled risk | **READ.** Pair members share model ID/revision in `configs/tc1_512.yaml`; main limitation is retrospective artifact provenance. |
| NF4 method | Controlled risk | **READ/EXTERNAL.** Loader requests NF4/double quantisation. This is a recognised bitsandbytes path ([official documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes)). Persisted method flags are nevertheless absent. |
| HarmBench scoring | Controlled risk | **READ/EXTERNAL.** Official HarmBench classifier sidecars are used; the model card identifies the classifier ([official card](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)). Its revision is not pinned. |
| XSTest scoring | P2 | **READ.** Regex scoring is deterministic but unvalidated against a reference labeler. |
| Statistical inference | Controlled risk/P2 | **READ.** Paired McNemar + BH family of 20 are appropriate for paired binary contrasts. The declared MDE means non-significance is not equivalence. |
| Seeds/decoding | P2 | **READ.** Main run is greedy/seed 42; only three of five pairs receive stochastic multi-seed coverage. |
| Reproducibility | P1 | **READ.** Historical outputs lack resolved model/data/software provenance. Hugging Face advises commit-hash revisions for reproducibility ([Transformers documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/models.md)). |
| Ethics/data handling | P2 | **READ.** Redacted sidecars are good storage practice; API judging creates a separate transfer surface. |

The evidence supports only direct-request HarmBench behavior at the stated budget and model/checkpoint conditions; it does not justify jailbreak robustness, zero effect, or broad deployment safety claims.
