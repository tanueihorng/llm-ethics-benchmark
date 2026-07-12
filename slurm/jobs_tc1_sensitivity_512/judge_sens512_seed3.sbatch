#!/bin/bash
#SBATCH --job-name=judge_sens512_seed3
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_sens512_seed3.out
#SBATCH --error=/tc1home/FYP/utan001/fyp_quant/repo/results_512/slurm_logs_tc1/judge_sens512_seed3.err

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
# --models lists exactly the aliases emitted by this generator invocation (the
# --pairs selection). The runner has NO skip logic, so it re-scores every alias
# listed; an incremental extension must regenerate with --pairs so this line
# names ONLY the new aliases and never re-scores (or requires the raw of) pairs
# already judged in an earlier run.
python /tc1home/FYP/utan001/fyp_quant/repo/scripts/run_judge_validation.py \
  --results-dir results_sensitivity_512/seed3 \
  --models mistral_7b_base mistral_7b_4bit phi4_mini_base phi4_mini_4bit \
  --benchmark harmbench \
  --backend harmbench_cls \
  --precision fp16 \
  --batch-size 4 \
  --device cuda
