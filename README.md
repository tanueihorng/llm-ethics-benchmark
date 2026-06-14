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
- Current FYP interim report: `docs/FYP_Report_2026-06-14.docx`
- Integrated agentic stack poster: `docs/architecture/fyp_quant_integrated_agentic_stack.svg`
- Visual repo hierarchy: `docs/architecture/fyp_quant_repo_hierarchy.svg`
- Visual agent harness architecture: `docs/architecture/fyp_quant_agent_harness_architecture.svg`
- Visual agentic architecture: `docs/architecture/fyp_quant_agentic_architecture.svg`
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
- `mistralai/Mistral-7B-Instruct-v0.3` ⏳ (baseline and on-the-fly BNB NF4 4-bit)
- `microsoft/Phi-4-mini-instruct` ⏳ (baseline and on-the-fly BNB NF4 4-bit; `trust_remote_code` + `attn_implementation: eager`)

⏳ = T26 cross-family extension (config/loader/sbatch/judge/tests complete, TC1 run pending; results not yet in the report).

Each model entry includes `family`, `size_b`, `quantized`, `pair_id`, and benchmark coverage, plus an optional `attn_implementation`.

## Benchmarks and Metrics
### Harmful Compliance: HarmBench
- Primary metric: `attack_success_rate`
- Additional outputs: harmful compliance count, safe refusal count, malformed output count
- Scoring method: **primary scorer is the official HarmBench classifier** (`cais/HarmBench-Llama-2-13b-cls`, judge; decision D16, run on TC1 job 61047). The deterministic v2 refusal regex is retained as a secondary non-refusal-rate proxy. Judge labels live in redacted `scores.judge.harmbench_cls.jsonl` / `summary.judge.harmbench_cls.json` sidecars; v2 in `scores.v2.jsonl` / `summary.v2.json`; the TC1-original `raw.jsonl` / `summary.json` are preserved unchanged.

### Over-refusal: XSTest
- Primary metric: `over_refusal_rate`
- Additional outputs: benign answer rate, malformed output count
- Scoring method: deterministic v2 refusal parser (the HarmBench classifier covers HarmBench only; a refusal-style judge for XSTest is future work)

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

### Agent harness commands
```bash
python fyp_cli.py agent-start --task T21 --agent fyp-report-auditor
python fyp_cli.py agent-status
make agent-check
make harness-eval
make agent-handoff
make agent-dashboard
make agent-tc1-checklist
```

The agent harness turns repo rules into checks: `configs/artifact_policy.yaml`
declares immutable raw artifacts, allowed sidecars, stale-text scans, redaction
scans, and report-worthy file patterns. `docs/agent_tasks/` holds bounded task
packets for future agents, and generated files such as `docs/HANDOFF.md`,
`docs/AGENT_DASHBOARD.md`, and `docs/TC1_AGENT_CHECKLIST.md` provide fresh
session recovery without replacing `docs/PROJECT_LOG.md`.

Agentic workflow helpers:
- Repo skills: `.agents/skills/`
- Codex custom subagents: `.codex/agents/`
- Codex hooks: `.codex/hooks.json` and `.codex/hooks/`
- Usage guide: `docs/AGENTIC_WORKFLOW.md`
- Integrated architecture poster: `docs/architecture/fyp_quant_integrated_agentic_stack.svg`
- Harness architecture visual: `docs/architecture/fyp_quant_agent_harness_architecture.svg`

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
# T26 extension (run after the originals; smoke first, Mistral is HF-gated):
squeue -u utan001
sbatch slurm/jobs_tc1/mistral_7b_base__matrix.sbatch
sbatch slurm/jobs_tc1/mistral_7b_4bit__matrix.sbatch
squeue -u utan001
sbatch slurm/jobs_tc1/phi4_mini_base__matrix.sbatch
sbatch slurm/jobs_tc1/phi4_mini_4bit__matrix.sbatch
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

## Current Headline Results (judge-primary, D16)

HarmBench ASR is scored by the **official HarmBench classifier** (`cais/HarmBench-Llama-2-13b-cls`, run on TC1 job 61047). The v2 refusal regex is shown only as a secondary non-refusal-rate proxy.

| Pair | HarmBench ASR (judge) | Δ (95% CI) | Sig? | Label |
|---|---:|---|:--:|---|
| qwen_2b | 0.135 → 0.190 | +0.055 [+0.010, +0.100] | **yes** | **broad_degradation** |
| qwen_4b | 0.065 → 0.090 | +0.025 [−0.000, +0.055] | no | alignment_degradation (directional) |
| llama_3_2_3b | 0.040 → 0.040 | 0.000 [−0.020, +0.020] | no | broad_degradation |

v2 proxy ΔASR for reference (over-counts; see report §6.12): qwen_2b −0.025, qwen_4b +0.065, llama 0.000.

Under the official classifier, NF4 quantization never reduces true harmful compliance in any pair; the one statistically significant ΔASR is the smallest model (Qwen 1.7B), which also loses significant capability — a confirmed degradation on both axes. The judge validation relocated the significant finding from Qwen 4B (regex proxy) to Qwen 1.7B; a second independent judge (gpt-4o) and a multi-seed sensitivity arm both corroborate the Qwen 1.7B direction while showing the effect is modest and borderline-significant. See `docs/PROJECT_LOG.md` (D16/D23/D26) and `docs/FYP_Report_2026-06-14.docx` §6.12 for the full audit trail.

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
