#!/bin/bash
#SBATCH --job-name=qwen_4b_base__matrix
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/qwen_4b_base__matrix.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results/slurm_logs_tc1/qwen_4b_base__matrix.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python /tc1home/FYP/utan001/fyp_quant/repo/run_quant_matrix.py --config configs/tc1.yaml --model qwen_4b_base --output_dir results --seed 42 --device cuda
