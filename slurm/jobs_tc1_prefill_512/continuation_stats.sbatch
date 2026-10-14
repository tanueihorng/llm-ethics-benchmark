#!/bin/bash
#SBATCH --job-name=prefill_cont_stats
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/continuation_stats.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1/continuation_stats.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p /tc1home/FYP/utan001/fyp_quant/repo/results_prefill_512/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T87 post hoc follow-up: numbers-only description of each prefill-arm reply
# (continuation length, approximate token count, near-cap flag, refusal-wording
# flag and offset). Reads results_prefill_512/<alias>/harmbench/raw.jsonl
# read-only and writes scores.continuation_stats.jsonl beside it. No GPU
# computation; loads only the cached tokenizers at their pinned revisions.
python /tc1home/FYP/utan001/fyp_quant/repo/scripts/prefill_continuation_stats.py --results-dir results_prefill_512
