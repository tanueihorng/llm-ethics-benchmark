# TC1 Activity Log — 2026-04-08

This file records the work performed today for the TC1 onboarding and FYP pipeline setup.

## Local repo work completed

- Reviewed the TC1 application form, the TC1 user guide, and the sample job script.
- Confirmed approved TC1 account details:
  - username: `utan001`
  - home directory: `/tc1home/FYP/utan001`
  - QoS: `normal`
  - partition: `UGGPU-TC1`
  - quota: `300 GB`
  - usage window: `03/2026` to `11/2026`
- Added TC1-specific cluster configuration:
  - `configs/tc1.yaml`
- Generated TC1 SLURM job scripts:
  - `slurm/jobs_tc1/`
- Improved SLURM generation so scripts can:
  - `cd` into a configured working directory
  - run environment bootstrap commands before Python execution
- Updated TC1 runbook documentation to reflect the new SLURM bootstrap support.
- Fixed local import-time crashes by making evaluator/model package exports lazy:
  - `ethical_benchmark/evaluators/__init__.py`
  - `ethical_benchmark/evaluators/toxicity_eval.py`
  - `ethical_benchmark/models/__init__.py`
  - `ethical_benchmark/models/generation.py`
  - `ethical_benchmark/models/loader.py`
- Updated tests to validate the lazy-loading behavior and SLURM helper behavior.
- Verified the full local test suite passes:
  - `pytest -q`
  - Result: `113 passed`

## TC1 access work completed

- Connected to TC1 successfully over SSH.
- Completed first successful login.
- Verified that TC1 created the home directory automatically on first login.
- Verified current home path:
  - `/tc1home/FYP/utan001`
- Verified account status with `MyTCinfo`:
  - QoS `normal`
  - `cpu=20, gres/gpu=1, mem=64G`
  - `MaxSubmit=2`
  - `MaxWall=06:00:00`
- Created a clean project workspace on TC1:
  - `/tc1home/FYP/utan001/fyp_quant`
- Checked the home directory contents and current size:
  - top-level project folder currently present: `fyp_quant`
  - home directory usage at that time: about `41K`

## In progress after this point

- Sync a trimmed copy of the current repo to:
  - `/tc1home/FYP/utan001/fyp_quant/repo`
- Create the TC1 conda environment:
  - `fyp-tc1`
- Install project requirements on TC1.
- Run a first small Qwen smoke/download test on TC1.

