# FYP Repository Guide
## Safety-Capability Trade-offs in 4-Bit Quantized Small Language Models

## 1. Purpose of This Guide
This document is a full operational manual for using this repository in a Final Year Project workflow. It is written in report style and is intended to support:

- reproducible experiment execution,
- transparent methodology reporting,
- consistent result aggregation and interpretation,
- alignment between implementation and dissertation chapters.

The guide focuses on the repository's current scope:

- harmful compliance (HarmBench),
- over-refusal (XSTest),
- capability (MMLU subset),
- matched baseline vs 4-bit pair analysis.

## 2. Study Scope Encoded in the Repository
The repository operationalizes a controlled quantization study where each baseline model is compared against its 4-bit counterpart. The current matrix includes:

- Qwen 0.8B baseline vs Qwen 0.8B 4-bit,
- Qwen 4B baseline vs Qwen 4B 4-bit,
- Llama 3.2 3B baseline vs Llama 3.2 3B 4-bit.

The study does **not** design quantization algorithms. It evaluates publicly available checkpoints under fixed evaluation conditions.

## 3. Repository Structure and Responsibilities
### 3.1 Core execution modules
- `ethical_benchmark/models/`
  - shared Hugging Face loader and text generation controls.
- `ethical_benchmark/benchmarks/`
  - benchmark plugin contract and concrete benchmark implementations.
- `ethical_benchmark/pipeline/run_quant_benchmark.py`
  - single run (`model x benchmark`) execution engine.
- `ethical_benchmark/pipeline/run_quant_matrix.py`
  - matrix orchestration runner.
- `ethical_benchmark/analysis/compare_quant_pairs.py`
  - pairwise baseline-vs-quantized delta analysis.
- `ethical_benchmark/cluster/`
  - SLURM script generation, submission helper, and status checker.

### 3.2 Primary command-line entrypoints
- `run_quant_benchmark.py`
- `run_quant_matrix.py`
- `compare_quant_pairs.py`
- `generate_slurm_jobs.py`
- `submit_slurm_jobs.py`
- `check_slurm_runs.py`
- `fyp_cli.py` (single entrypoint wrapper for smoke/run/matrix/analyze/cluster tasks)
- `Makefile` (shortcut wrappers for frequent commands)

## 4. Environment Setup
## 4.1 Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional shortcut layer:
```bash
make smoke
make matrix DEVICE=cuda
make analyze
```

## 4.2 Minimum software requirements
- Python 3.10+
- `transformers`, `datasets`, `torch`, `pydantic`, `pyyaml`, `numpy`, `tqdm`

## 4.3 Hardware assumptions
- CPU-only execution is supported (slow but functional).
- Single-GPU execution is supported and recommended for practical runtime.
- The framework is designed for compact models and edge-oriented constraints.

## 5. Configuration Specification
The default experiment specification is in `configs/default.yaml`.

## 5.1 Top-level keys
- `study_name`: identifier written into result metadata.
- `models`: model matrix with pair metadata.
- `decoding`: shared generation controls.
- `benchmarks`: benchmark-specific dataset and execution settings.
- `slurm`: cluster execution defaults.

## 5.2 Model entry schema
Each model entry contains:

- `family`
- `size_b`
- `quantized`
- `pair_id`
- `model_id`
- `benchmarks`
- optional loading fields (`trust_remote_code`, `dtype`, `revision`)

This schema is validated through `ethical_benchmark/quant/config_schema.py` before execution.

## 5.3 Benchmark entry schema
Each benchmark block defines:

- `dataset_name`
- `split`
- `max_samples`
- `batch_size`
- benchmark-specific extras (for example MMLU `subjects`)

## 6. Standard Experimental Workflow
## 6.1 Stage A: single-run sanity check
Run a small execution to verify environment and output structure.

```bash
python run_quant_benchmark.py \
  --config configs/default.yaml \
  --model qwen_0_8b_bf16 \
  --benchmark harmbench \
  --max_samples 20 \
  --output_dir results
```

Expected artifacts:
- `results/qwen_0_8b_bf16/harmbench/raw.jsonl`
- `results/qwen_0_8b_bf16/harmbench/summary.json`
- `results/summary/harmbench_runs.csv`

## 6.2 Stage B: full matrix execution
```bash
python run_quant_matrix.py \
  --config configs/default.yaml \
  --output_dir results
```

Optional filters:
- restrict benchmarks: repeat `--benchmark` (for example `--benchmark mmlu`)
- restrict models: repeat `--model`

## 6.3 Stage C: pairwise quantization analysis
```bash
python compare_quant_pairs.py \
  --config configs/default.yaml \
  --results_dir results \
  --output_dir results/analysis
```

Outputs include:
- pairwise deltas (JSON + CSV),
- interpretation labels,
- scale-sensitivity and cross-family summaries.

## 7. Metric Definitions in Implementation
## 7.1 HarmBench
Primary metric:
- `attack_success_rate`

Additional tracked values:
- harmful compliance count,
- safe refusal count,
- malformed output count.

## 7.2 XSTest
Primary metric:
- `over_refusal_rate`

Additional tracked values:
- benign answer rate,
- malformed output count.

Refusal detection uses a deterministic parser (regex-based), not LLM-as-judge scoring.

## 7.3 MMLU subset
Primary metric:
- `accuracy`

Additional tracked values:
- answered rate,
- subject-level breakdown.

## 8. Result Schema and Auditability
Each response record saved to `raw.jsonl` includes the required fields:

- `benchmark`,
- `prompt_id`,
- `prompt_text`,
- `response`,
- `score_fields`,
- `family`,
- `size_b`,
- `quantized`,
- `pair_id`,
- `model_id`,
- `generation_config`,
- `seed`,
- `timestamp`.

The runner supports resume mode by skipping prompts that already exist in raw logs.

## 9. Pairwise Interpretation Logic
The analysis layer computes baseline vs quantized deltas and maps each pair to one label:

- `alignment_degradation`
- `capability_collapse_masquerading_as_safety`
- `robust_preservation`
- `broad_degradation`

This interpretation is rule-based and explicitly tied to measured deltas from HarmBench, XSTest, and MMLU.

## 10. SLURM Execution Protocol
## 10.1 Generate job scripts
```bash
python generate_slurm_jobs.py \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

## 10.2 Submit generated scripts
```bash
python submit_slurm_jobs.py --jobs_dir slurm/jobs
```

For verification without submission:
```bash
python submit_slurm_jobs.py --jobs_dir slurm/jobs --dry_run
```

## 10.3 Check run completion and queue state
```bash
python check_slurm_runs.py \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

To skip queue probing (`squeue`):
```bash
python check_slurm_runs.py \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs \
  --skip_squeue
```

## 11. Reproducibility Checklist (for Dissertation Methods Chapter)
Before reporting final findings, verify:

1. same config file used for all compared runs,
2. fixed seed policy documented,
3. decoding settings identical across models,
4. sample limits and benchmark splits fixed,
5. resume behavior did not mix incompatible settings,
6. pairwise analysis generated from the same result root.

## 12. Troubleshooting
## 12.1 Missing summary files in analysis
Cause: incomplete benchmark runs for one or more model-benchmark combinations.

Action:
- run `check_slurm_runs.py` to identify missing combinations,
- rerun only missing combinations using `run_quant_benchmark.py`.

## 12.2 Empty or malformed outputs inflate “safe” behavior
Cause: degradation can produce gibberish/invalid responses.

Action:
- inspect `malformed_output_count` in benchmark summaries,
- interpret ASR and refusal changes jointly with capability deltas.

## 12.3 Cluster scripts generated but not submitted
Cause: only generation stage executed.

Action:
- run `submit_slurm_jobs.py` after `generate_slurm_jobs.py`.

## 13. Extending the Study
## 13.1 Add new models
Add entries under `models` in config using the same pair metadata fields.

## 13.2 Add new benchmark plugins
Implement the plugin contract in `ethical_benchmark/benchmarks/base.py` and register in `ethical_benchmark/benchmarks/registry.py`.

## 13.3 Add new analysis metrics
Extend extraction and delta computation in `ethical_benchmark/analysis/compare_quant_pairs.py`.

## 14. Suggested Appendix Material for FYP Submission
Recommended appendix exports:

- `configs/default.yaml`
- one representative `raw.jsonl` excerpt per benchmark
- full `summary.json` files for each run
- `results/analysis/pairwise_deltas.csv`
- `results/analysis/pair_interpretations.csv`
- `slurm/jobs/manifest.json`

## 15. Mapping to Dissertation Chapters
- Chapter 1 (Introduction): use problem framing from README and Section 2 of this guide.
- Chapter 2 (Literature/Benchmarks): use `docs/datasets.md`.
- Chapter 3 (Methodology): use Sections 5-10.
- Chapter 4 (Experiments/Results): use Sections 7-9 and generated analysis files.
- Chapter 5 (Limitations/Future Work): use `docs/limitations.md` and Section 13.
