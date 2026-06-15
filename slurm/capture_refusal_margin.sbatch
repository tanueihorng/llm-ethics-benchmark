#!/bin/bash
#SBATCH --job-name=capture_refusal_margin
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/capture_refusal_margin.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/capture_refusal_margin.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Refusal-margin MECHANISM capture (D33). For all ten model aliases, runs
# teacher-forced forward passes over the already-saved HarmBench prompts (+ a
# neutral MMLU/XSTest control sample) and records scalar refusal-margin /
# confidence diagnostics into NEW redacted sidecars
# (scores.margin.<precision>.jsonl). It performs NO new generation and never
# touches raw.jsonl / summary.json / scores.judge.* — same immutability contract
# as the judge validation.
#
# Models are already cached from the original + T26 runs, so no prefetch is
# needed (offline mode loads them from the HF cache). Precision tag is derived
# per model (fp16 for *_base, nf4 for *_4bit). The INT8 point is a later run:
#   ... --models <int8 aliases> --precision-tag int8   (after INT8 entries land).
#
# Forward-only over 10 models is light (no autoregressive decoding); 16G host
# RAM + a V100 is ample and it finishes well inside the 4h walltime. If a
# single 10-model job is undesirable, split with --models pair-by-pair.

python - <<'PY'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] GPU={p.name}  VRAM={p.total_memory/1024**3:.1f} GiB", flush=True)
else:
    print("[probe] no CUDA device visible", flush=True)
PY

python /tc1home/FYP/utan001/fyp_quant/repo/scripts/capture_refusal_margin.py \
    --config configs/tc1.yaml \
    --results-dir results \
    --device cuda \
    --models qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit \
             llama_3_2_3b_base llama_3_2_3b_4bit \
             mistral_7b_base mistral_7b_4bit phi4_mini_base phi4_mini_4bit \
    --control-n 50
