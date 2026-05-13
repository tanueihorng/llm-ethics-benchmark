# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**fyp_quant** is a research benchmarking framework that studies safety-capability trade-offs in 4-bit quantized small language models (0.8B–4B parameters). It compares **baseline vs 4-bit checkpoint pairs** across three dimensions:

- **HarmBench** — harmful compliance under unsafe prompts (Attack Success Rate)
- **XSTest** — over-refusal on benign prompts (Over-Refusal Rate)
- **MMLU subset** — general capability (Accuracy)

The core research question: do observed safety changes in 4-bit models reflect true alignment shifts, or just capability degradation?

---

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Local Development (via Makefile)
```bash
make smoke              # One model × one benchmark, 20 samples (fast sanity check)
make run                # One model × one benchmark, full samples
make matrix DEVICE=cuda # All 6 models × 3 benchmarks (18 runs)
make analyze            # Compute pairwise deltas and interpretations
```

### Direct CLI (fyp_cli.py)
```bash
python fyp_cli.py run -m qwen_0_8b_bf16 -b harmbench -d cuda -n 400
python fyp_cli.py matrix --model qwen_0_8b_bf16 --benchmark harmbench -d cuda
python fyp_cli.py analyze --config configs/default.yaml --output_dir results/analysis
```

### SLURM Cluster (TC1)
```bash
make cluster-generate   # Generate sbatch scripts → slurm/jobs_tc1/
make cluster-submit     # Submit jobs (use --dry_run to validate first)
make cluster-check      # Poll squeue for job status
```

### Tests
```bash
pytest tests/                                    # All tests
pytest tests/test_quant_smoke.py                 # Smoke: end-to-end pipeline
pytest tests/test_quant_analysis.py              # Pairwise delta computation
pytest tests/test_refusal_parser.py              # Refusal detection patterns
pytest tests/test_slurm_helpers.py               # SLURM script generation
pytest tests/test_matrix_reuse.py                # Model reuse logic
```

---

## Architecture

### Directory Map

```
ethical_benchmark/
├── benchmarks/          # Benchmark plugins (HarmBench, XSTest, MMLU)
├── models/              # HF model loading + batched generation
├── pipeline/            # Run orchestration (single run, full matrix)
├── cluster/             # SLURM job generation, submission, status
├── analysis/            # Pairwise delta computation and interpretation
├── quant/               # Pydantic config schema
├── metrics/             # JSONL/CSV I/O, bootstrap CI helpers
└── evaluators/          # Legacy toxicity/bias/factuality judges (not default)
```

### Execution Flow

**Single run** (`run_quant_benchmark.py`):
1. Load YAML config → resolve model + benchmark
2. Set seed (Python, NumPy, Torch)
3. `HFModelLoader` loads model + tokenizer (dtype/device resolution)
4. `TextGenerator` batches prompts through model
5. Benchmark plugin scores each response
6. Aggregate metrics → write `results/<model>/<benchmark>/raw.jsonl` + `summary.json`

**Matrix run** (`run_quant_matrix.py`):
- Loads each model once, runs all its benchmarks sequentially (avoids 2 extra load cycles per model)
- Calls `gc.collect()` + `torch.cuda.empty_cache()` between model loads

**Analysis** (`compare_quant_pairs.py`):
- Groups results by `pair_id` → baseline vs 4-bit
- Computes absolute and relative deltas per benchmark
- Assigns interpretation label: `alignment_degradation`, `capability_collapse_masquerading_as_safety`, `robust_preservation`, or `broad_degradation`
- Outputs to `results/analysis/`

### Benchmark Plugins

All plugins implement `BenchmarkPlugin` ABC in `ethical_benchmark/benchmarks/base.py`. Each plugin handles its own dataset loading, prompt formatting, and per-response scoring. Refusal detection is **deterministic regex** (no judge models) for reproducibility.

| Benchmark | Dataset | Primary Metric | Batch Size |
|-----------|---------|----------------|------------|
| HarmBench | walledai/HarmBench | `attack_success_rate` | 4 |
| XSTest | allenai/xstest-response | `over_refusal_rate` | 4 |
| MMLU | cais/mmlu | `accuracy` | 4 |

### Configuration

Configs are YAML files validated by Pydantic (`ethical_benchmark/quant/config_schema.py`):
- `configs/default.yaml` — local development
- `configs/tc1.yaml` — TC1 SLURM cluster (partition `UGGPU-TC1`, work dir `/tc1home/FYP/utan001/fyp_quant/repo`)

Key constraint enforced by schema: each `pair_id` must have ≥1 baseline + ≥1 quantized model, and model `benchmarks` lists must reference top-level benchmark keys.

### Model Pairs

| pair_id | Baseline alias | 4-bit alias |
|---------|----------------|-------------|
| qwen_0_8b | qwen_0_8b_bf16 | qwen_0_8b_4bit |
| qwen_4b | qwen_4b_bf16 | qwen_4b_4bit |
| llama_3_2_3b | llama_3_2_3b_bf16 | llama_3_2_3b_4bit |

All models use `dtype: auto` (resolves to float16 on CUDA, float32 on CPU) and `temperature: 0.0` (greedy decoding for determinism).

### Output Structure

```
results/
├── <model_alias>/<benchmark>/raw.jsonl     # Per-prompt: prompt, response, score_fields
├── <model_alias>/<benchmark>/summary.json  # Aggregated metrics
├── summary/<benchmark>_runs.csv            # Flattened summaries across all runs
└── analysis/
    ├── pairwise_deltas.{json,csv}
    ├── pair_interpretations.{csv}
    └── quantization_analysis_summary.json
```

`raw.jsonl` records include `pair_id`, `quantized`, `model_alias`, `seed`, `generation_config`, and `timestamp` for full auditability. The resume logic (`--resume`, default) skips already-processed `prompt_id`s to allow interrupted runs to continue.
