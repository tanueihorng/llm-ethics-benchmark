#!/bin/bash
#SBATCH --job-name=judge_llamaguard_512
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_llamaguard_512.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_llamaguard_512.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T37/D46 -- OPEN-WEIGHT THIRD HarmBench judge (Llama-Guard-3-8B).
# Re-scores the 512-token HarmBench generations (results_512/) with the
# open-weight Llama Guard classifier, all FIFTEEN aliases (5 pairs x {fp16, NF4,
# INT8}) in one job. Purpose: replace the "the second judge is a versioned API
# model" reproducibility caveat with a fully-reproducible open-weight cross-check
# of the primary HarmBench classifier. Derived sidecars only -- does NOT modify
# raw.jsonl/summary.json and runs NO target-model inference.
# Writes scores.judge.llamaguard.* + summary.judge.llamaguard.* under results_512/.
# The harmbench_cls and api_judge sidecars are never touched.
#
# CONSTRUCT NOTE: Llama Guard judges *content safety*, not "attack success" under
# the HarmBench rubric, so its per-prompt kappa vs the classifier reflects an
# instrument-family difference as much as error -- read it as a reproducibility
# cross-check, not a tie-breaker (see docs/agent_tasks/T36-T39-hardening-batch.md).
#
# PREREQUISITE (one-time, HEAD node): Llama-Guard-3-8B is HF-GATED. Accept the
# licence on huggingface.co and refresh the token login FIRST, then prefetch:
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml \
#       --judge --judge-model-id meta-llama/Llama-Guard-3-8B
#
# PRECISION: fp16 (~16 GB for 8B) fits the 32 GB V100 comfortably. If it OOMs,
# reduce --batch-size, then fall back to --precision 8bit.
#
# WALLTIME NOTE: 15 aliases x 200 prompts. The runner writes sidecars per-model as
# each alias finishes and has NO skip logic; if this nears the 6h cap, split the
# --models list across two jobs, and on a timeout resubmit with only the aliases
# that have no summary.judge.llamaguard.json yet (the finished ones are saved).

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
    --models qwen_2b_base qwen_2b_4bit qwen_2b_8bit \
             qwen_4b_base qwen_4b_4bit qwen_4b_8bit \
             llama_3_2_3b_base llama_3_2_3b_4bit llama_3_2_3b_8bit \
             mistral_7b_base mistral_7b_4bit mistral_7b_8bit \
             phi4_mini_base phi4_mini_4bit phi4_mini_8bit \
    --benchmark harmbench \
    --backend llamaguard \
    --judge-revision "${JUDGE_REVISION:-}" \
    --device cuda \
    --precision fp16 \
    --batch-size 4
