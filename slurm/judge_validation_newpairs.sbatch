#!/bin/bash
#SBATCH --job-name=judge_harmbench_cls_newpairs
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/judge_harmbench_cls_newpairs.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/judge_harmbench_cls_newpairs.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# HarmBench classifier judge validation over ONLY the two NEW pairs' saved
# HarmBench generations (mistral_7b + phi4_mini, T26). Kept separate from
# slurm/judge_validation.sbatch on purpose: re-scoring is deterministic, but
# pointing only at the new aliases means the committed old-6 sidecars
# (scores.judge.harmbench_cls.* for qwen/llama) are never reopened or rewritten
# — zero drift risk for the already-published results.
#
# This is the PRIMARY HarmBench ASR scorer (D16) for the new pairs: their ASR
# must be judge-scored exactly like the existing pairs, NOT read off the v2
# regex, or the cross-pair comparison is invalid.
#
# Derived sensitivity check only — does NOT modify raw.jsonl, summary.json, or
# the v2 sidecars, and runs NO new target-model inference.
#
# Prerequisite (one-time, on the HEAD node, before submitting this job):
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge
# which caches cais/HarmBench-Llama-2-13b-cls into the HF cache so this
# offline job can load it. (Already cached from the D16 run; re-running the
# prefetch is a no-op.)
#
# PRECISION: fp16 (full precision) so the judge's OWN quantization does not
# confound the validation of the target models — identical to the D16 run.
#   - fp16 13B needs ~26 GB VRAM -> requires a 32 GB V100.
#   - If this job OOMs, reduce --batch-size first; then --precision 8bit; then
#     --precision 4bit as a last resort.

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
    --models mistral_7b_base mistral_7b_4bit phi4_mini_base phi4_mini_4bit \
    --benchmark harmbench \
    --backend harmbench_cls \
    --model-id cais/HarmBench-Llama-2-13b-cls \
    --device cuda \
    --precision fp16 \
    --batch-size 4
