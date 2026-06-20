# Methodology

## 1. Experimental Design
This study uses a matched-pair comparative design to isolate the impact of 4-bit quantization.

Independent variables:
- quantization state / precision (baseline fp16 vs INT8 vs NF4 4-bit)
- model size (1.7B vs 4B in Qwen; 1.7B–7.2B across the study)
- model family (Qwen, Llama, Mistral, Phi — 5 matched pairs / 10 models / 4 families)

Dependent variables:
- harmful compliance
- over-refusal
- general capability

## 2. Benchmark Mapping
- Harmful compliance: HarmBench (primary scorer = official HarmBench classifier `cais/HarmBench-Llama-2-13b-cls`; the v2 refusal regex is a secondary non-refusal-rate proxy)
- Over-refusal: XSTest (benign prompts)
- Capability: MMLU subset (primary anchor), corroborated by ARC-Challenge (second capability benchmark)

The capability benchmarks are included to test whether apparent safety gains are genuine or artifacts of capability collapse.

## 3. Pipeline Steps
1. Load validated quantization matrix config.
2. Select one model and one benchmark.
3. Generate responses using fixed decoding settings.
4. Score responses with deterministic benchmark-specific logic.
5. Save per-response records and aggregate summary metrics.
6. Repeat across matrix.
7. Compute baseline-vs-quantized deltas per pair.
8. Produce scale-sensitivity and cross-family summaries.

## 4. Prompting and Decoding Controls
To maintain fairness, all runs use shared decoding parameters from `configs/default.yaml`:
- fixed `temperature`
- fixed `max_new_tokens`
- fixed `top_p`
- fixed `repetition_penalty`
- deterministic seed

Thinking-style variability is controlled by deterministic generation settings in the core evaluation flow.

## 5. Matched Pair Analysis
Pair-level comparisons are computed per `pair_id`:
- absolute delta = quantized - baseline
- relative delta = (quantized - baseline) / baseline

Interpretation labels are rule-based and derived from combined benchmark deltas (each paired with a two-layer `evidence_status`: confirmed / directional / null):
- alignment degradation
- alignment improvement
- capability collapse masquerading as safety
- over-refusal regression
- robust preservation
- broad degradation
