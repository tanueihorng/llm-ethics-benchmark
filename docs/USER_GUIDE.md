# User Guide (Quantization Study)

For full dissertation-style documentation, see `docs/FYP_REPO_GUIDE.md`.
For full TC1 cluster execution steps, see `docs/TC1_CLUSTER_RUNBOOK.md`.

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Sanity Run (One Combination)
```bash
python fyp_cli.py smoke
```
or:
```bash
make smoke
```

## 3. Full Matrix
```bash
python fyp_cli.py matrix --config configs/default.yaml
```
or:
```bash
make matrix DEVICE=cuda
```

## 4. Pairwise Analysis
```bash
python fyp_cli.py analyze \
  --config configs/default.yaml \
  --results_dir results \
  --output_dir results/analysis
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

Submit:
```bash
python fyp_cli.py cluster-submit --jobs_dir slurm/jobs
```

Check status:
```bash
python fyp_cli.py cluster-check --config configs/default.yaml
```

## 6. Key Outputs
- Per-run raw records: `results/<model>/<benchmark>/raw.jsonl`
- Per-run summary: `results/<model>/<benchmark>/summary.json`
- Pairwise deltas: `results/analysis/pairwise_deltas.csv`
- Interpretation labels: `results/analysis/pair_interpretations.csv`
