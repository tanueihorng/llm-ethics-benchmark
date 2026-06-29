#!/bin/bash
#SBATCH --job-name=judge_sens512_seed1
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_sens512_seed1.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_sens512_seed1.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results_512/slurm_logs_tc1 results_sensitivity_512
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Score this seed's stochastic HarmBench generations with the official
# HarmBench classifier (fp16), so the sensitivity ASR matches the judge-primary
# headline metric. Requires the classifier cached on the HEAD node first:
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge
# Edit --models below to match the models you actually generated for this seed.
python /tc1home/FYP/utan001/fyp_quant/repo/scripts/run_judge_validation.py \
  --results-dir results_sensitivity_512/seed1 \
  --models qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit llama_3_2_3b_base llama_3_2_3b_4bit \
  --benchmark harmbench \
  --backend harmbench_cls \
  --precision fp16 \
  --batch-size 4 \
  --device cuda
