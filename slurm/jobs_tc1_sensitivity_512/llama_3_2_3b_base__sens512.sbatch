#!/bin/bash
#SBATCH --job-name=llama_3_2_3b_base__sens512
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/llama_3_2_3b_base__sens512.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/llama_3_2_3b_base__sens512.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results_512/slurm_logs_tc1 results_sensitivity_512
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# T18 sensitivity: HarmBench under stochastic decoding (T=0.7) across seeds.
# Each seed writes to a separate root so resume logic does not collide.
for SEED in 1 2 3 4 5; do
  echo "=== llama_3_2_3b_base seed $SEED (T=0.7 HarmBench sensitivity) ==="
  python /tc1home/FYP/utan001/fyp_quant/repo/run_quant_matrix.py \
    --config configs/tc1_sensitivity_512.yaml \
    --model llama_3_2_3b_base \
    --benchmark harmbench \
    --seed "$SEED" \
    --output_dir results_sensitivity_512/seed"$SEED" \
    --device cuda
done
