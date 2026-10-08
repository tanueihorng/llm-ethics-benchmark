# User Guide (Quantization Study)

The completed primary study is `results_512/` (`configs/tc1_512.yaml`); `results_scratch/` is the 128-token comparison. Examples below are for a new local study in `results_scratch/`, preserving the existing evidence. For installation profiles, use `docs/REPO_GUIDE.md`. Run evaluation and script generation locally or through `sbatch`; TC1 head-node use is limited to the runbook's setup, transfer and SLURM administration.

For full dissertation-style documentation, see `docs/FYP_REPO_GUIDE.md`.
For full TC1 cluster execution steps, see `docs/TC1_CLUSTER_RUNBOOK.md`.

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Sanity Run (One Combination)
```bash
python fyp_cli.py smoke --results_dir results_scratch
```
or:
```bash
python run_quant_benchmark.py --config configs/default.yaml --model qwen_2b_base --benchmark harmbench --max_samples 5 --output_dir results_scratch
```

## 3. Full Matrix
```bash
python fyp_cli.py matrix --config configs/default.yaml --results_dir results_scratch
```
or:
```bash
python run_quant_matrix.py --config configs/default.yaml --device cuda --output_dir results_scratch
```

## 4. Pairwise Analysis
```bash
python fyp_cli.py analyze \
  --config configs/default.yaml \
  --results_dir results_scratch \
  --output_dir results_scratch/analysis
```

## 5. SLURM Workflow
Generate scripts:
```bash
python fyp_cli.py cluster-generate --config configs/default.yaml
```
or:
```bash
make cluster-generate DEVICE=cuda
```

Submit (local/dev clusters only):
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs
```
> **Not for NTU TC1.** `cluster-submit` runs Python on the head node and reads a
> gitignored `manifest.json`, both of which TC1 policy forbids. On TC1, submit each
> generated sbatch directly with `sbatch slurm/jobs_tc1/<model>__matrix.sbatch`
> (see `docs/TC1_CLUSTER_RUNBOOK.md` and the README TC1 section).

Check status:
```bash
python fyp_cli.py cluster-check --config configs/default.yaml
```

## 6. Key Outputs
- Per-run raw records: `results_scratch/<model>/<benchmark>/raw.jsonl`
- Per-run summary: `results_scratch/<model>/<benchmark>/summary.json`
- Pairwise deltas: `results_scratch/analysis/pairwise_deltas.csv`
- Interpretation labels: `results_scratch/analysis/pair_interpretations.csv`
