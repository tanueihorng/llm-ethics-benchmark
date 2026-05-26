# TC1 Activity Log — 2026-04-08

This log records the setup and verification work carried out for the TC1 GPU cluster on 2026-04-08.

## Scope

- Activate and verify TC1 account access.
- Prepare the local repository for TC1 execution.
- Sync a trimmed copy of the repository to TC1.
- Create and verify a TC1 conda environment.
- Submit an initial small Qwen job on TC1.

## Activity Log

### 1. Repository preparation

- Reviewed the TC1 approval form, user guide, and sample job script.
- Confirmed the assigned TC1 home directory is `/tc1home/FYP/utan001`.
- Confirmed assigned QoS is `normal`.
- Confirmed QoS limits:
  - `cpu=20`
  - `gres/gpu=1`
  - `mem=64G`
  - `MaxSubmit=2`
  - `MaxWall=06:00:00`
- Added TC1-oriented SLURM bootstrap support to generated job scripts.
- Added a TC1-specific config file at `configs/tc1.yaml`.
- Generated initial TC1 job scripts locally in `slurm/jobs_tc1`.

### 2. TC1 account activation

- Connected to `10.96.189.11` over SSH.
- Accepted the server host key on first connection.
- Logged in successfully to TC1.
- Verified that the first login created the home directory.
- Verified `pwd` resolved to `/tc1home/FYP/utan001`.
- Ran `MyTCinfo` successfully to confirm account and QoS information.

### 3. TC1 home directory status

- Confirmed the home directory exists and is active.
- Confirmed the directory was essentially fresh at inspection time.
- Created `/tc1home/FYP/utan001/fyp_quant` as a dedicated workspace folder.

### 4. Pending actions

- Repository sync to TC1.
- TC1 conda environment creation.
- First small benchmark job submission.

## Notes

- No password or secret values are recorded in this file.
- TC1 guide restrictions still apply: heavy inference should run through SLURM jobs on compute nodes, not on the head node.
