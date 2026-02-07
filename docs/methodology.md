# Methodology

## 1. End-to-End Benchmarking Pipeline
The benchmarking workflow is implemented in `ethical_benchmark/pipeline/run_benchmark.py` and follows the sequence below:

1. Load experiment configuration from YAML.
2. Set deterministic seeds for reproducibility.
3. Resolve model alias and load tokenizer/model via Hugging Face.
4. Load task dataset split (toxicity, bias, or factuality).
5. Generate model outputs in batches under fixed decoding settings.
6. Evaluate outputs using task-specific evaluators.
7. Persist per-sample raw records (JSONL) and aggregate summaries (JSON + CSV).
8. Export radar-compatible CSV for downstream visualization.

## 2. Model Selection Rationale
The framework targets open-source models below 10B parameters because:

- They are feasible for single-GPU and CPU-only environments.
- They align with edge deployment constraints.
- They provide a realistic setting for academic comparison under bounded resources.

Supported families include LLaMA, Gemma, Phi, and DeepSeek distilled variants, with concrete aliases defined in `configs/default.yaml`.

## 3. Prompting and Decoding Strategy
To maintain fair comparisons, decoding parameters are controlled globally through `configs/default.yaml`:

- `temperature` (default `0.0`) for deterministic decoding.
- `max_new_tokens` to cap output length.
- `top_p` and `repetition_penalty` for controlled sampling behavior.
- `use_chat_template` to apply tokenizer-native chat formatting when available.

Multiple-choice tasks (BBQ and TruthfulQA) instruct the model to return only an option letter to reduce parsing ambiguity.

## 4. Experimental Controls
The implementation includes explicit controls for methodological rigor:

- **Decoding consistency:** same decoding config for all models.
- **Split control:** evaluation uses benchmark splits only (no training).
- **Seed control:** unified random seeding for Python, NumPy, and Torch.
- **Resume support:** partial runs continue without re-evaluating completed samples.
- **Auditability:** raw prompt-response-score records are persisted.
- **Metric separation:** subjective judge metrics (if enabled) are reported separately from objective metrics.

## 5. Task-Specific Execution
### Toxicity
Implemented end-to-end with RealToxicityPrompts loader, generation, toxicity classifier scoring, per-sample persistence, and aggregate reporting.

### Social Bias
Implemented as multiple-choice QA on BBQ with accuracy, bias gap, and demographic-axis confusion diagnostics.

### Factuality
Implemented using TruthfulQA multiple-choice with objective MC accuracy and optional subjective LLM-as-judge scoring.
