# Methodology

## 1. Experimental Design
This study uses a matched-pair comparative design to isolate the impact of 4-bit quantization.

Independent variables:
- quantization state (baseline vs 4-bit)
- model size (0.8B vs 4B in Qwen)
- model family (Qwen vs Llama)

Dependent variables:
- harmful compliance
- over-refusal
- general capability

## 2. Benchmark Mapping
- Harmful compliance: HarmBench
- Over-refusal: XSTest (benign prompts)
- Capability: MMLU subset

The capability benchmark is included to test whether apparent safety gains are genuine or artifacts of capability collapse.

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

Interpretation labels are rule-based and derived from combined benchmark deltas:
- alignment degradation
- capability collapse masquerading as safety
- robust preservation
- broad degradation
