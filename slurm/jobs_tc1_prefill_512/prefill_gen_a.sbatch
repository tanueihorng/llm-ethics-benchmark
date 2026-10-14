#!/bin/bash
#SBATCH --job-name=prefill_gen_a
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/prefill_gen_a.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/prefill_gen_a.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p /tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T87/D105 exploratory prefill arm (docs/PREFILL_PREREG.md). HarmBench only,
# greedy, 512 tokens, assistant reply prefilled with "Sure, here is".
# Writes ONLY to results_prefill_512/; results_512/ is never touched.
# FP16 and NF4 of each pair run back to back in this job. Resume is on by
# default, so resubmitting this file after a timeout continues where it stopped.
for MODEL in qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit llama_3_2_3b_base llama_3_2_3b_4bit; do
  echo "=== $MODEL prefill HarmBench ==="
  python /tc1home/FYP/utan001/fyp_quant/repo/run_quant_matrix.py \
    --config configs/tc1_prefill_512.yaml \
    --model "$MODEL" \
    --benchmark harmbench \
    --output_dir results_prefill_512 \
    --seed 42 \
    --device cuda
done
