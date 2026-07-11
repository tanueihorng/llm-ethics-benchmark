#!/bin/bash
#SBATCH --job-name=judge_harmbench_cls_512
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_harmbench_cls_512.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_harmbench_cls_512.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T31/D39 -- PRIMARY HarmBench scorer for the 512-token rerun.
# Scores the 512-token HarmBench generations (results_512/) with the official
# HarmBench classifier, all TEN aliases in one job. Derived sidecars only --
# does NOT modify raw.jsonl/summary.json, and runs NO target-model inference.
# Reads from results_512/, writes scores.judge.harmbench_cls.* under results_512/.
# The 128-token judge sidecars in results/ are never touched.
#
# Prerequisite (one-time, HEAD node, already done for the 128 run -- the 13B
# classifier is cached; only re-run if the cache was cleared):
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge
#
# PRECISION: fp16 (full precision) so the judge's own quantization does not
# confound the validation. fp16 13B ~26 GB -> needs the 32 GB V100. If it OOMs,
# reduce --batch-size; then fall back to --precision 8bit.
#
# WALLTIME NOTE: 512-token responses are ~4x longer than the 128-token run, so
# the classifier's encode cost per item is higher. Sidecars are written per-model
# as each alias finishes, but the runner has NO skip logic. If this 10-model job
# nears the 6h cap, run it as two jobs (split the --models list 5/5); and on a
# timeout, resubmit with --models listing only the aliases that have no
# scores.judge.harmbench_cls.json yet (the finished ones are already saved).

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
    --models qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit \
             llama_3_2_3b_base llama_3_2_3b_4bit \
             mistral_7b_base mistral_7b_4bit \
             phi4_mini_base phi4_mini_4bit \
    --benchmark harmbench \
    --backend harmbench_cls \
    --model-id cais/HarmBench-Llama-2-13b-cls \
    --judge-revision "${JUDGE_REVISION:-}" \
    --device cuda \
    --precision fp16 \
    --batch-size 4
