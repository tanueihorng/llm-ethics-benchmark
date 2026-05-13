# Safety-Capability Trade-offs in 4-Bit Quantized Small Language Models

## Project Overview
This repository implements a research-grade benchmarking framework for a focused quantization study:

- harmful compliance under unsafe prompts
- over-refusal under benign prompts
- capability under general knowledge evaluation

The core objective is to compare baseline and 4-bit checkpoints as matched pairs, then analyze whether observed safety changes reflect true alignment shifts or capability degradation.

## Documentation
- Full FYP operational guide: `docs/FYP_REPO_GUIDE.md`
- TC1 cluster step-by-step runbook: `docs/TC1_CLUSTER_RUNBOOK.md`
- Quick start guide: `docs/USER_GUIDE.md`
- Methodology notes: `docs/methodology.md`
- Metric definitions: `docs/evaluation_metrics.md`
- Dataset notes: `docs/datasets.md`
- Limitations: `docs/limitations.md`
- Extensibility: `docs/extensibility.md`

## Study Focus
The framework is explicitly scoped to the following research questions:

- RQ1: Does 4-bit quantization increase harmful compliance?
- RQ2: Does 4-bit quantization increase over-refusal on benign prompts?
- RQ3: Does 4-bit quantization degrade general capability?
- RQ4: Are smaller models more sensitive than larger models within the same family?
- RQ5: Are effects consistent across model families?

## System Overview
The framework keeps model loading/generation modular, and refactors orchestration around quantization pair analysis.

```mermaid
flowchart TD
    A[run_quant_benchmark.py] --> B[Quant Config Validation]
    B --> C[HF Model Loader]
    C --> D[Shared Text Generator]
    B --> E[Benchmark Plugin]
    E --> F[HarmBench or XSTest or MMLU]
    D --> G[Per-prompt Response]
    F --> H[Deterministic Scoring]
    G --> H
    H --> I[Raw JSONL]
    H --> J[Summary JSON/CSV]
    K[run_quant_matrix.py] --> A
    L[compare_quant_pairs.py] --> J
    L --> M[Pair Deltas + Labels]
```

## Repository Layout
```text
ethical_benchmark/
├── benchmarks/
│   ├── base.py
│   ├── harmbench.py
│   ├── xstest.py
│   ├── mmlu.py
│   └── registry.py
├── pipeline/
│   ├── run_quant_benchmark.py
│   └── run_quant_matrix.py
├── analysis/
│   └── compare_quant_pairs.py
├── cluster/
│   ├── generate_jobs.py
│   ├── submit_jobs.py
│   └── check_runs.py
├── models/
│   ├── loader.py
│   └── generation.py
└── quant/
    └── config_schema.py
```

Legacy toxicity/bias/factuality modules are retained as non-default paths for backward compatibility, but the default study flow is now quantization-focused.

## Model Matrix
Configured in `configs/default.yaml`:

- `techwithsergiu/Qwen3.5-text-0.8B` (baseline)
- `techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit` (4-bit)
- `techwithsergiu/Qwen3.5-text-4B` (baseline)
- `techwithsergiu/Qwen3.5-text-4B-bnb-4bit` (4-bit)
- `meta-llama/Llama-3.2-3B-Instruct` (baseline)
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit)

Each model entry includes `family`, `size_b`, `quantized`, `pair_id`, and benchmark coverage.

## Benchmarks and Metrics
### Harmful Compliance: HarmBench
- Primary metric: `attack_success_rate`
- Additional outputs: harmful compliance count, safe refusal count, malformed output count

### Over-refusal: XSTest
- Primary metric: `over_refusal_rate`
- Additional outputs: benign answer rate, malformed output count
- Scoring method: deterministic refusal parser (no judge model in v1)

### Capability: MMLU Subset
- Primary metric: `accuracy`
- Additional outputs: answered rate and subject-level breakdown

## Quantization Analysis Outputs
`compare_quant_pairs.py` computes:

- baseline vs 4-bit deltas per `pair_id`
- absolute and relative deltas
- Qwen scale sensitivity (0.8B vs 4B delta magnitudes)
- cross-family sign consistency (Qwen vs Llama)
- interpretation labels:
  - `alignment_degradation`
  - `capability_collapse_masquerading_as_safety`
  - `robust_preservation`
  - `broad_degradation`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run
### 1) One model x benchmark
```bash
python run_quant_benchmark.py \
  --config configs/default.yaml \
  --model qwen_0_8b_bf16 \
  --benchmark harmbench \
  --output_dir results
```

### 2) Full matrix
```bash
python run_quant_matrix.py \
  --config configs/default.yaml \
  --output_dir results
```

### Easier unified CLI
```bash
python fyp_cli.py smoke
python fyp_cli.py matrix
python fyp_cli.py analyze
```

### Fast shortcut commands
```bash
make smoke
make matrix DEVICE=cuda
make analyze
make cluster-generate
make cluster-submit
make cluster-check
```

### 3) Pairwise analysis
```bash
python compare_quant_pairs.py \
  --config configs/default.yaml \
  --results_dir results \
  --output_dir results/analysis
```

## SLURM Workflow
### Generate sbatch scripts
```bash
python generate_slurm_jobs.py \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

### Submit jobs
```bash
python submit_slurm_jobs.py --jobs_dir slurm/jobs
```

### Check run status
```bash
python check_slurm_runs.py \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

## Output Structure
Results are organized by model and benchmark to simplify matched comparisons:

```text
results/
  qwen_0_8b_bf16/
    harmbench/
      raw.jsonl
      summary.json
    xstest/
      raw.jsonl
      summary.json
    mmlu/
      raw.jsonl
      summary.json
  qwen_0_8b_4bit/
  qwen_4b_bf16/
  qwen_4b_4bit/
  llama_3_2_3b_bf16/
  llama_3_2_3b_4bit/
  summary/
    harmbench_runs.csv
    xstest_runs.csv
    mmlu_runs.csv
  analysis/
    pairwise_deltas.json
    pairwise_deltas.csv
    pair_interpretations.json
    pair_interpretations.csv
    quantization_analysis_summary.json
```

## Reproducibility Controls
- Fixed global seed
- Shared decoding configuration across all models
- Prompt-level resume logic via `prompt_id`
- Per-response raw logs for auditability

## Sample Summary Table
| Pair | Benchmark | Baseline | 4-bit | Delta |
|---|---|---:|---:|---:|
| qwen_0_8b | HarmBench ASR | 0.31 | 0.38 | +0.07 |
| qwen_0_8b | XSTest over-refusal | 0.12 | 0.19 | +0.07 |
| qwen_0_8b | MMLU accuracy | 0.44 | 0.36 | -0.08 |

## Documentation
- `docs/methodology.md`
- `docs/evaluation_metrics.md`
- `docs/datasets.md`
- `docs/limitations.md`
- `docs/extensibility.md`

## Mapping to FYP Report
- Chapter 1: Introduction -> README Overview
- Chapter 2: Related Work -> `docs/datasets.md`
- Chapter 3: Methodology -> `docs/methodology.md`
- Chapter 4: Experiments -> `docs/evaluation_metrics.md`
- Chapter 5: Limitations -> `docs/limitations.md`
