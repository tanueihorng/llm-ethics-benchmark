# Safety-Capability Trade-offs in 4-Bit Quantized Small Language Models

> **Coding agents:** read [`AGENTS.md`](AGENTS.md) (or [`CLAUDE.md`](CLAUDE.md) — they are duplicates) for the working conventions, then [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) for current state.

## Project Overview
This repository implements a research-grade benchmarking framework for a focused quantization study:

- harmful compliance under unsafe prompts
- over-refusal under benign prompts
- capability under general knowledge evaluation

The core objective is to compare baseline and 4-bit checkpoints as matched pairs, then analyze whether observed safety changes reflect true alignment shifts or capability degradation.

## Documentation

> **Start here:** [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) — single source of truth for current status, open tasks, decisions, and the changelog. Update this file whenever something changes; everything else listed below is permanent reference material that does not track day-to-day state.

**Permanent reference material:**
- Current FYP interim report: `docs/FYP_Report_2026-05-27.docx`
- Full FYP operational guide: `docs/FYP_REPO_GUIDE.md`
- TC1 cluster step-by-step runbook: `docs/TC1_CLUSTER_RUNBOOK.md`
- Quick start guide: `docs/USER_GUIDE.md`
- Methodology notes: `docs/methodology.md`
- Metric definitions: `docs/evaluation_metrics.md`
- Dataset notes: `docs/datasets.md`
- Limitations: `docs/limitations.md`
- Extensibility: `docs/extensibility.md`

**Archived (superseded; kept for traceability):** `docs/archive/`

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

- `Qwen/Qwen3-1.7B` (baseline and on-the-fly BNB NF4 4-bit)
- `Qwen/Qwen3-4B` (baseline and on-the-fly BNB NF4 4-bit)
- `meta-llama/Llama-3.2-3B-Instruct` (baseline and on-the-fly BNB NF4 4-bit)

Each model entry includes `family`, `size_b`, `quantized`, `pair_id`, and benchmark coverage.

## Benchmarks and Metrics
### Harmful Compliance: HarmBench
- Primary metric: `attack_success_rate`
- Additional outputs: harmful compliance count, safe refusal count, malformed output count
- Scoring method: deterministic v2 refusal parser. The TC1-original `raw.jsonl` / `summary.json` files are preserved; corrected v2 scoring is stored in `scores.v2.jsonl` / `summary.v2.json` sidecars.

### Over-refusal: XSTest
- Primary metric: `over_refusal_rate`
- Additional outputs: benign answer rate, malformed output count
- Scoring method: deterministic v2 refusal parser (judge-model validation is tracked as T20 follow-up work and must use separate judge sidecars)

### Capability: MMLU Subset
- Primary metric: `accuracy`
- Additional outputs: answered rate and subject-level breakdown

## Quantization Analysis Outputs
`compare_quant_pairs.py` computes:

- baseline vs 4-bit deltas per `pair_id`
- absolute and relative deltas
- Qwen compact-scale sensitivity (2B vs 4B delta magnitudes)
- cross-family sign consistency (Qwen vs Llama)
- interpretation labels:
  - `alignment_degradation`
  - `alignment_improvement`
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
  --model qwen_2b_base \
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
make cluster-generate   # regenerate sbatch scripts (Mac only)
make cluster-check      # poll squeue for job status
# Note: do NOT use make cluster-submit on TC1 — use direct sbatch instead (see SLURM Workflow below)
```

### 3) Pairwise analysis
```bash
python compare_quant_pairs.py \
  --config configs/default.yaml \
  --results_dir results \
  --output_dir results/analysis
```

## SLURM Workflow (TC1)
Active sbatch scripts live in `slurm/jobs_tc1/` (matrix) and `slurm/jobs_tc1_smoke/` (5-sample smoke).
These are tracked in git — `git pull` on TC1 is sufficient to get them.

### Submit jobs (TC1 head node)
`make cluster-submit` is **not suitable for TC1**: it reads `manifest.json` (gitignored, absent after
`git pull`) and runs Python on the head node (TC1 policy forbids user code there). Use direct `sbatch`:
```bash
# Submit the current pair only. TC1 currently runs one GPU job at a time;
# the second job can queue with QOSMaxGRESPerUser until the first clears.
sbatch slurm/jobs_tc1/qwen_2b_base__matrix.sbatch
sbatch slurm/jobs_tc1/qwen_2b_4bit__matrix.sbatch
squeue -u utan001   # wait for both to finish, then next pair
sbatch slurm/jobs_tc1/qwen_4b_base__matrix.sbatch
sbatch slurm/jobs_tc1/qwen_4b_4bit__matrix.sbatch
squeue -u utan001
sbatch slurm/jobs_tc1/llama_3_2_3b_base__matrix.sbatch
sbatch slurm/jobs_tc1/llama_3_2_3b_4bit__matrix.sbatch
```

### Check run status
```bash
squeue -u utan001
seff <JOBID>
ls results/*/*/summary.json
```

## Output Structure
Results are organized by model and benchmark to simplify matched comparisons:

```text
results/
  qwen_2b_base/
    harmbench/
      raw.jsonl
      summary.json
      scores.v2.jsonl
      summary.v2.json
    xstest/
      raw.jsonl
      summary.json
      scores.v2.jsonl
      summary.v2.json
    mmlu/
      raw.jsonl
      summary.json
  qwen_2b_4bit/
  qwen_4b_base/
  qwen_4b_4bit/
  llama_3_2_3b_base/
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
- Immutable raw-output contract: post-hoc scoring corrections and future judge validation write derived sidecars only; they do not overwrite `raw.jsonl` or `summary.json`.

## Current v2 Headline Results
| Pair | Benchmark | Baseline | 4-bit | Delta |
|---|---|---:|---:|---:|
| qwen_2b | HarmBench ASR | 0.600 | 0.575 | -0.025 |
| qwen_4b | HarmBench ASR | 0.240 | 0.305 | +0.065 |
| llama_3_2_3b | HarmBench ASR | 0.060 | 0.060 | 0.000 |

Interpretation labels under the current v2 scorer: `qwen_2b=capability_collapse_masquerading_as_safety`, `qwen_4b=alignment_degradation`, `llama_3_2_3b=broad_degradation`. See `docs/PROJECT_LOG.md` and `docs/FYP_Report_2026-05-27.docx` for the full v1→v2 scorer audit trail.

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
