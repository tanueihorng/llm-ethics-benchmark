# T23 ARC-Challenge sbatch files (second capability benchmark)

Six per-model ARC-only jobs. Each runs `run_quant_matrix.py --benchmark arc`
(greedy, T=0.0, same as the main MMLU capability run) and writes
`results/<model>/arc/`. The existing harmbench/xstest/mmlu artifacts are not
touched.

## One-time prefetch (TC1 HEAD node)

ARC-Challenge is a new dataset, so cache it before running offline jobs:
```bash
git -C /tc1home/FYP/utan001/fyp_quant/repo pull --ff-only
python scripts/prefetch_tc1.py --config configs/tc1.yaml   # now includes allenai/ai2_arc (ARC-Challenge)
```

## Run order (one GPU job at a time; MaxJobsPU=2)

Submit two at a time, wait for both to clear, then the next two:
```bash
sbatch slurm/jobs_tc1_arc/qwen_2b_base__arc.sbatch
sbatch slurm/jobs_tc1_arc/qwen_2b_4bit__arc.sbatch
# wait, then:
sbatch slurm/jobs_tc1_arc/qwen_4b_base__arc.sbatch
sbatch slurm/jobs_tc1_arc/qwen_4b_4bit__arc.sbatch
# wait, then:
sbatch slurm/jobs_tc1_arc/llama_3_2_3b_base__arc.sbatch
sbatch slurm/jobs_tc1_arc/llama_3_2_3b_4bit__arc.sbatch
```

## After the runs

SCP `results/*/arc/` back to the Mac, then `make analyze` (the pairwise-delta
pipeline already treats any benchmark with an `accuracy` metric as a capability
axis — verify the ARC ΔACC appears alongside MMLU), and fold the ARC ΔACC into
the report's capability sections (RQ3, §6.x, Ch8 partial-capability limitation).
`raw.jsonl`/`summary.json` under `results/<model>/arc/` are TC1-original
immutable artifacts once produced.
