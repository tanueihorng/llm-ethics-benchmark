#!/bin/bash
#SBATCH --job-name=judge_harmbench_cls_int8_512
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_harmbench_cls_int8_512.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_harmbench_cls_int8_512.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T31/D39 -- PRIMARY HarmBench ASR for the 512-token INT8 precision point.
# Scores ONLY the five *_8bit aliases' 512-token HarmBench generations in
# results_512/, so the 512 fp16/NF4 sidecars are never reopened. Run AFTER the
# INT8-512 matrix jobs complete. fp16 classifier on a 32 GB V100 -- identical
# instrument to the fp16/NF4 runs, so the 512 fp16 -> INT8 -> NF4 ASR sweep is
# measured by one consistent scorer. Reads results_512/<alias>/harmbench/raw.jsonl.
# Derived only: never modifies raw.jsonl/summary.json/v2 sidecars.

python - <<'PY'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] GPU={p.name}  VRAM={p.total_memory/1024**3:.1f} GiB", flush=True)
else:
    print("[probe] no CUDA device visible", flush=True)
PY

python /tc1home/FYP/utan001/fyp_quant/repo/scripts/run_judge_validation.py \
    --results-dir results_512 \
    --models qwen_2b_8bit qwen_4b_8bit llama_3_2_3b_8bit mistral_7b_8bit phi4_mini_8bit \
    --benchmark harmbench \
    --backend harmbench_cls \
    --model-id cais/HarmBench-Llama-2-13b-cls \
    --judge-revision "${JUDGE_REVISION:-}" \
    --device cuda \
    --precision fp16 \
    --batch-size 4
