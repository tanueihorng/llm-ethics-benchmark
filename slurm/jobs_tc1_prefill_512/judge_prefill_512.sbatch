#!/bin/bash
#SBATCH --job-name=judge_prefill_512
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/judge_prefill_512.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/judge_prefill_512.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p /tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T87/D105: score the prefill-arm generations with the primary HarmBench
# classifier (fp16), exactly as the primary study (slurm/judge_validation_512.sbatch).
# Reads results_prefill_512/<alias>/harmbench/raw.jsonl; writes redacted
# scores.judge.harmbench_cls.* sidecars beside them. Runs only after BOTH
# generation jobs have finished. The runner has no skip logic: after a timeout,
# edit --models to list only aliases without a summary.judge.harmbench_cls.json.
python /tc1home/FYP/utan001/fyp_quant/repo/scripts/run_judge_validation.py \
    --results-dir results_prefill_512 \
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

# Hashes only (no text) of the raw generations, which stay on TC1 (PREFILL_PREREG §8).
sha256sum results_prefill_512/*/harmbench/raw.jsonl > results_prefill_512/raw_sha256.txt
cat results_prefill_512/raw_sha256.txt
