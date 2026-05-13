# TC1 Cluster Runbook (Step-by-Step)
## Using This Repository on Your School GPU Cluster

## 0. Approved Account Notes (From CCDS Support Email)
- Cluster: `TC1`
- TC1 IP: `10.96.189.11`
- First login is mandatory to create your home directory.
- Inactive users (never logged in) may be revoked after 3 months without prior notice.
- No platform backup is provided; maintain your own backups.
- Account and data are scheduled for removal at end of **November 2026**.
- Support contact: `ccdsgpu-tc@ntu.edu.sg`

## 1. Purpose
This runbook is a separate operational guide for running the quantization study on the TC1 school GPU cluster.

It is designed to be a checklist-style document you can follow during actual experiment runs and include in FYP appendix material.

## 2. What This Runbook Assumes
- You already have approved access to the TC1 cluster.
- You can SSH into the TC1 login/head node.
- You have read the TC1 user guide (`CCDSGPU-TC1-UG-UserGuide`) and will use its official values for:
  - login host,
  - partition,
  - QoS,
  - account/project code,
  - walltime limits,
  - GPU constraints.

Important policy assumption from TC1 guide:
- Do not run heavy training/inference directly on the head/login node.
- Use SLURM job submission (`sbatch`) for benchmark execution.

## 3. One-Time Setup on TC1
## 3.1 SSH to TC1
```bash
ssh <YOUR_TC1_USERNAME>@10.96.189.11
```

If this is your first login, do this immediately so the home directory is created and the account is marked active.

## 3.2 Create project workspace on cluster storage
```bash
mkdir -p ~/fyp_quant
cd ~/fyp_quant
```

## 3.3 Get repository onto TC1
Option A: clone from remote git origin
```bash
git clone <YOUR_REPO_URL> repo
cd repo
```

Option B: copy from local machine
```bash
scp -r /path/to/local/repo <YOUR_TC1_USERNAME>@10.96.189.11:~/fyp_quant/repo
ssh <YOUR_TC1_USERNAME>@10.96.189.11
cd ~/fyp_quant/repo
```

## 3.4 Prepare Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If TC1 requires module loading first, follow the TC1 guide and run required module commands before creating/activating `.venv`.

Conda-style alternative (matching the provided sample script pattern):
```bash
module load anaconda
source activate <YOUR_ENV_NAME>
pip install -r requirements.txt
```

## 3.5 Quick local sanity check on login node (very small only)
Run only a tiny smoke test to verify setup, then stop using login node for inference:
```bash
python fyp_cli.py smoke --max_samples 5 --device cpu
```

## 4. Configure TC1 SLURM Parameters
`configs/default.yaml` is now prefilled with the TC1 sample script defaults:

- `partition: UGGPU-TC1`
- `qos: normal`
- `gpus: 1`
- `cpus_per_task: 1`
- `mem: 10G`
- `time: "360"` (minutes)
- `log_dir: results/slurm_logs`

For TC1, the generated sbatch scripts can also bootstrap your shell before running Python. This is useful because TC1 jobs often need module loading and environment activation before `python` is available with the right packages.

Current block:
```yaml
slurm:
  partition: UGGPU-TC1
  qos: normal
  account:
  gpus: 1
  cpus_per_task: 1
  mem: 10G
  time: "360"
  log_dir: results/slurm_logs
  work_dir: /tc1home/FYP/utan001/fyp_quant/repo
  setup_commands:
    - module load slurm
    - module load anaconda
    - source activate fyp-tc1
```

Adjust `work_dir` and `source activate ...` to match your actual cluster folder and environment name.

If the latest TC1 guide for your cohort specifies different values, replace this block accordingly before large runs.

## 5. Easiest CLI Workflow (Recommended)
This repository now includes a unified CLI for easier operations:

- `python fyp_cli.py smoke`
- `python fyp_cli.py run ...`
- `python fyp_cli.py matrix ...`
- `python fyp_cli.py analyze ...`
- `python fyp_cli.py cluster-generate ...`
- `python fyp_cli.py cluster-submit ...`
- `python fyp_cli.py cluster-check ...`

For even less typing, use the `Makefile` shortcuts:
- `make smoke`
- `make matrix`
- `make analyze`
- `make cluster-generate`
- `make cluster-submit`
- `make cluster-check`

Example with overrides:
```bash
make matrix DEVICE=cuda SEED=42
make cluster-generate DEVICE=cuda JOBS_DIR=slurm/jobs
```

## 6. Generate SLURM Jobs (Model x Benchmark)
From repository root:
```bash
python fyp_cli.py cluster-generate \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

This creates:
- one `.sbatch` per model-benchmark combination,
- `slurm/jobs/manifest.json`.

Expected job count for current matrix:
- 6 models x 3 benchmarks = 18 jobs.

## 7. Validate Jobs Before Submission
## 7.1 Dry run submit
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs --dry_run
```

## 7.2 Inspect one generated script
```bash
sed -n '1,200p' slurm/jobs/qwen_0_8b_bf16__harmbench.sbatch
```

Confirm:
- partition/qos/account values,
- memory/time,
- command path and arguments.

## 8. Submit Jobs
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs
```

This writes `slurm/jobs/submitted_jobs.json` with `job_id` records.

## 9. Monitor Progress
## 9.1 Check queue status + output completeness
```bash
python fyp_cli.py cluster-check \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

Produces report file:
- `slurm/check_status.json`

## 9.2 If `squeue` is unavailable in your shell
```bash
python fyp_cli.py cluster-check \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs \
  --skip_squeue
```

## 10. Resume and Recovery Strategy
If runs fail or timeout:

1. Fix SLURM resource settings (`time`, `mem`, etc.) in config.
2. Regenerate jobs.
3. Resubmit only missing/incomplete combinations.
4. Keep `--resume` enabled (default) so completed prompts are not recomputed.

For a clean rerun of one combination:
```bash
python fyp_cli.py run \
  --model qwen_0_8b_bf16 \
  --benchmark harmbench \
  --force_restart
```

## 11. Faster Matrix Execution Notes
The matrix runner is now optimized to reuse one loaded model across multiple benchmarks per model by default.

Equivalent command:
```bash
python fyp_cli.py matrix --config configs/default.yaml --results_dir results
```

If you need legacy behavior (load model each run), disable reuse:
```bash
python run_quant_matrix.py --no-reuse_loaded_model
```

## 12. Post-Run Analysis on TC1
After benchmark summaries are generated:
```bash
python fyp_cli.py analyze \
  --config configs/default.yaml \
  --results_dir results \
  --output_dir results/analysis
```

Key outputs:
- `results/analysis/pairwise_deltas.csv`
- `results/analysis/pair_interpretations.csv`
- `results/analysis/quantization_analysis_summary.json`

## 13. Move Results Back to Local Machine
From local machine:
```bash
scp -r <YOUR_TC1_USERNAME>@10.96.189.11:~/fyp_quant/repo/results ./results_tc1
```

## 14. Recommended Experiment Order
1. smoke test (`max_samples <= 20`)
2. one full model across all 3 benchmarks
3. full matrix (18 jobs)
4. analysis export
5. rerun any missing combinations only

## 15. Troubleshooting
## 15.1 Jobs submit but no outputs
- Check SLURM logs in `results/slurm_logs`.
- Verify Python environment activation inside sbatch script if required by your TC1 environment.

## 15.2 Out-of-memory or timeout
- Increase `mem` and/or `time`.
- Reduce benchmark `batch_size` in config.
- For quick debug, reduce `max_samples`.

## 15.3 Missing Hugging Face model access
- Ensure credentials are configured on cluster if model requires gated access.

## 16. Reproducibility Notes for FYP Write-Up
Always report:
- exact config file used,
- seed value,
- benchmark sample limits,
- decoding settings,
- SLURM resource settings,
- date/time and cluster environment summary.

## 17. Minimal Command Cheat Sheet
```bash
# Generate jobs
python fyp_cli.py cluster-generate

# Dry-run submit
python fyp_cli.py cluster-submit --dry_run

# Submit
python fyp_cli.py cluster-submit

# Status
python fyp_cli.py cluster-check

# Analyze after completion
python fyp_cli.py analyze
```

Equivalent `make` shortcuts:
```bash
make cluster-generate
make cluster-dry
make cluster-submit
make cluster-check
make analyze
```
