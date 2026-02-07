# Datasets

## 1. RealToxicityPrompts
### Source
- Hugging Face dataset ID: `allenai/real-toxicity-prompts`

### Purpose
Provides naturally occurring prompts with associated toxicity context for evaluating harmful continuation tendencies.

### Usage in This Framework
- Task: toxicity
- Default split: `train` (used as benchmark prompt set, not for model training)
- Loader: `ethical_benchmark/datasets/toxicity.py`

## 2. BBQ (Bias Benchmark for QA)
### Source
- Hugging Face dataset ID (default in config): `heegyu/bbq`

### Purpose
Measures social bias in question answering across demographic dimensions.

### Usage in This Framework
- Task: bias
- Default split: `test`
- Loader: `ethical_benchmark/datasets/bias.py`

### Notes
BBQ variants can differ in schema. The loader includes schema normalization for common field patterns (e.g., `ans0`-`ans2` vs list-based choices).

## 3. TruthfulQA (Multiple Choice)
### Source
- Hugging Face dataset ID: `truthful_qa`
- Config: `multiple_choice`

### Purpose
Evaluates tendency to produce truthful answers on known misconception-prone questions.

### Usage in This Framework
- Task: factuality
- Default split: `validation`
- Loader: `ethical_benchmark/datasets/factuality.py`

## 4. Licensing Notes
Dataset licensing and usage conditions may vary by source and version. Users should verify:

- redistribution permissions
- academic/commercial usage terms
- attribution requirements

before publishing benchmark outputs.

## 5. Sampling Strategy
The framework applies deterministic shuffling with a fixed seed and optional `max_samples` truncation. This supports:

- reproducible subset creation
- balanced runtime across models
- controlled comparison under equal sample budgets

## 6. Ethical Considerations
- Toxicity datasets can include offensive language; handling requires safe storage and restricted exposure.
- Bias benchmarks may encode demographic assumptions and cultural framing bias.
- Factuality benchmarks are sensitive to annotation design and may not capture all truth conditions.
