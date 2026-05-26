# TC1 Cluster Runbook (Step-by-Step)
## Using This Repository on Your School GPU Cluster

## Approved Account Notes (From CCDS Support Email)
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

## 3. TC1 Cluster Policy (verified against the TC1 user guide, Dec 2025)

The following policies are enforced by the cluster operator and shape every
step in this runbook. They are not optional.

**Forbidden on the head node (CCDS-TC1):**
- Running any user code that consumes meaningful CPU/RAM. The guide states:
  *"DO NOT execute your coding on TC1 Head Node ... will be terminated with
  no prior notification. Repeated offenders will be banned from TC1."* (p.16)
- Running `nvidia-smi`, `nvcc --version`, or any GPU/CUDA verification — the
  GPU cards live on the compute nodes (TC1N01–07), not on the head node. (p.3, p.11)
- Using `srun` for real workloads. Use `sbatch` only: *"all users are advised
  to use the command 'sbatch' for job submission. Then exit from the session,
  access later to see the result."* (p.19)

**Allowed on the head node (explicitly demonstrated in the user guide):**
- SSH sessions, script editing (`vi`/`nano`/`ne`), file transfers (SFTP).
- Conda environment management: `module load anaconda`, `conda create`,
  `conda install`, `pip install`. (p.7–9)
- Hugging Face downloads — same activity category as `pip install`. Used by
  this study's pre-cache step (see §4.6 below).
- SLURM administration: `squeue`, `scontrol`, `sacct`, `MyTCinfo`,
  `MyJobHistory`, `seff`, `TC1RunningJob`.

**Resource limits (QoS "normal"):**
- 1 GPU per job, up to 20 CPU cores and 64 GB RAM total across all jobs.
- 2 concurrent jobs per user (MaxJobsPU).
- 6-hour walltime per job (MaxWall). Extensions to 8/12/24/48h available on
  request to `ccdsgpu-tc@ntu.edu.sg` with justification.
- 100 GB minimum storage quota; this project was approved for ~300 GB.
  Confirm actual quota with `MyTCinfo` on first login.

**Other rules:**
- No backups. You are responsible for your own data.
- Do not access any directory other than your home and assigned shared folder.
- Account revoked after 3 months of inactivity. First login activates the
  account; subsequent inactivity is measured from your last job submission.

## 4. One-Time Setup on TC1
## 4.1 SSH to TC1
```bash
ssh <YOUR_TC1_USERNAME>@10.96.189.11
```

If this is your first login, do this immediately so the home directory is created and the account is marked active.

## 4.2 Create project workspace on cluster storage
```bash
mkdir -p ~/fyp_quant
cd ~/fyp_quant
```

## 4.3 Get repository onto TC1
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

## 4.4 Prepare Python environment
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

## 4.5 Do NOT smoke-test on the head node

Earlier versions of this runbook suggested a quick CPU-only smoke test on the
login node. This is now revised: per the TC1 user guide (p.16), running user
code on the head node is forbidden and may result in account termination.
All validation, including the initial smoke test, must go through `sbatch`.
See §4.7 below for the proper smoke-validation procedure.

## 4.6 Pre-cache datasets and model weights (head node, one-time)

This study runs in strict offline mode on the compute nodes (which may not have
outbound internet). Cache everything on the head node first.

```bash
# Confirm your QoS and quota assignment
MyTCinfo

# Activate the conda env
module load anaconda
source activate fyp-tc1
cd /tc1home/FYP/utan001/fyp_quant/repo

# One-time HF authentication (required for Llama and gated HarmBench access)
python - <<'PY'
from getpass import getpass
from huggingface_hub import login
login(token=getpass("HF token: "), add_to_git_credential=False)
print("HF login saved.")
PY

# Pre-cache HF datasets (HarmBench, 6 MMLU subjects) + model weights
# (Qwen 2B, Qwen 4B, Llama 3.2 3B). XSTest is bundled locally.
# Observed 2026-05-26: ~25.5 GB total, completed in under 5 minutes.
python scripts/prefetch_tc1.py
# Equivalently: make prefetch CONFIG=configs/tc1.yaml
```

The pre-cache step is policy-compliant because it performs only HTTP file
transfer — equivalent to the `pip install` / `conda install` activities that
the TC1 user guide explicitly demonstrates on the head node (p.7–9). It does
not run any user code that consumes CPU or memory.

After this completes, every SLURM job runs with `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` exported (these are
baked into the generated sbatch files via `configs/tc1.yaml`'s
`setup_commands`). Any cache miss inside a job will then fail immediately
with a clear error rather than hang waiting on network.

## 4.7 Smoke-validate on a compute node (via sbatch, not srun)

Before submitting the full 6-job matrix, run one small sbatch to verify the
offline-cache path on a real GPU:

```bash
# On the head node, regenerate the smoke sbatch
make cluster-smoke CONFIG=configs/tc1.yaml

# Submit it
sbatch slurm/jobs_tc1_smoke/qwen_2b_base__harmbench.sbatch

# Check the queue, then come back later for the output
squeue -u utan001
seff <jobid>    # after completion: shows CPU and memory efficiency
```

A clean smoke run will produce `results/qwen_2b_base/harmbench/summary.json`
containing an `attack_success_rate`. If that file exists and is well-formed,
the full matrix is safe to launch.

## 5. Configure TC1 SLURM Parameters
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

## 6. Easiest CLI Workflow (Recommended)
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

## 7. Generate SLURM Jobs (Model x Benchmark)
From repository root:
```bash
python fyp_cli.py cluster-generate \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs \
  --group_by model \
  --script run_quant_matrix.py
```

This creates:
- one `.sbatch` per model alias, with the loaded model reused across benchmarks,
- `slurm/jobs/manifest.json`.

Expected job count for current matrix:
- 6 model jobs. Each job runs that model across its configured benchmarks.

## 8. Validate Jobs Before Submission
## 8.1 Dry run submit
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs --dry_run
```

## 8.2 Inspect one generated script
```bash
sed -n '1,200p' slurm/jobs/qwen_2b_base__matrix.sbatch
```

Confirm:
- partition/qos/account values,
- memory/time,
- command path and arguments.

## 9. Submit Jobs
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs
```

This writes `slurm/jobs/submitted_jobs.json` with `job_id` records.

## 10. Monitor Progress
## 10.1 Check queue status + output completeness
```bash
python fyp_cli.py cluster-check \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs
```

Produces report file:
- `slurm/check_status.json`

## 10.2 If `squeue` is unavailable in your shell
```bash
python fyp_cli.py cluster-check \
  --config configs/default.yaml \
  --results_dir results \
  --jobs_dir slurm/jobs \
  --skip_squeue
```

## 11. Resume and Recovery Strategy
If runs fail or timeout:

1. Fix SLURM resource settings (`time`, `mem`, etc.) in config.
2. Regenerate jobs.
3. Resubmit only missing/incomplete combinations.
4. Keep `--resume` enabled (default) so completed prompts are not recomputed.

For a clean rerun of one combination:
```bash
python fyp_cli.py run \
  --model qwen_2b_base \
  --benchmark harmbench \
  --force_restart
```

## 12. Faster Matrix Execution Notes
The matrix runner is now optimized to reuse one loaded model across multiple benchmarks per model by default.

Equivalent command:
```bash
python fyp_cli.py matrix --config configs/default.yaml --results_dir results
```

If you need legacy behavior (load model each run), disable reuse:
```bash
python run_quant_matrix.py --no-reuse_loaded_model
```

## 13. Post-Run Analysis on TC1
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

## 14. Move Results Back to Local Machine
From local machine:
```bash
scp -r <YOUR_TC1_USERNAME>@10.96.189.11:~/fyp_quant/repo/results ./results_tc1
```

## 15. Recommended Experiment Order
1. smoke test (`max_samples <= 20`)
2. one full model across all 3 benchmarks
3. full matrix (18 jobs)
4. analysis export
5. rerun any missing combinations only

## 16. Troubleshooting
## 16.1 Jobs submit but no outputs
- Check SLURM logs in `results/slurm_logs`.
- Verify Python environment activation inside sbatch script if required by your TC1 environment.

## 16.2 Out-of-memory or timeout
- Increase `mem` and/or `time`.
- Reduce benchmark `batch_size` in config.
- For quick debug, reduce `max_samples`.

## 16.3 Missing Hugging Face model access
- Ensure credentials are configured on the cluster if a model or dataset requires gated access.
- For this study, Llama 3.2 3B and HarmBench both require accepted Hugging Face access conditions.

## 17. Reproducibility Notes for FYP Write-Up
Always report:
- exact config file used,
- seed value,
- benchmark sample limits,
- decoding settings,
- SLURM resource settings,
- date/time and cluster environment summary.

## 18. Minimal Command Cheat Sheet
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
