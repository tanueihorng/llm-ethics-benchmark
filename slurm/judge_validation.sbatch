#!/bin/bash
#SBATCH --job-name=judge_harmbench_cls
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/judge_harmbench_cls.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/judge_harmbench_cls.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# HarmBench classifier judge validation over all six models' saved HarmBench
# generations. Derived sensitivity check only — does NOT modify raw.jsonl,
# summary.json, or the v2 sidecars, and runs NO new target-model inference.
#
# Prerequisite (one-time, on the HEAD node, before submitting this job):
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge
# which caches cais/HarmBench-Llama-2-13b-cls into the HF cache so this
# offline job can load it.
#
# PRECISION: the judge runs in fp16 (full precision) by default so the judge's
# OWN quantization does not confound the validation of the target models.
#   - fp16 13B needs ~26 GB VRAM  -> requires a 32 GB V100.
#   - Batch size is kept conservative to leave activation/KV-cache headroom.
#   - If this job OOMs, reduce --batch-size first; if fp16 still does not fit,
#     resubmit with --precision 8bit (~13-14 GB, far closer to fp16 than nf4).
#   - --precision 4bit is the last resort.
# The runner logs the actual GPU name + total VRAM + chosen precision at startup
# (see the .out log) and records them in summary.judge.harmbench_cls.json for
# reproducibility.

# Quick hardware probe on the compute node (allowed inside sbatch; NOT on the
# head node). Records which V100 variant we landed on before the heavy load.
python - <<'PY'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] GPU={p.name}  VRAM={p.total_memory/1024**3:.1f} GiB", flush=True)
else:
    print("[probe] no CUDA device visible", flush=True)
PY

python /tc1home/FYP/utan001/fyp_quant/repo/scripts/run_judge_validation.py \
    --results-dir results \
    --models qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit \
             llama_3_2_3b_base llama_3_2_3b_4bit \
    --benchmark harmbench \
    --backend harmbench_cls \
    --model-id cais/HarmBench-Llama-2-13b-cls \
    --device cuda \
    --precision fp16 \
    --batch-size 4
