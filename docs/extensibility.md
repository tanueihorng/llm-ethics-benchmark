# Extensibility

## 1. Add a New Model
Update `configs/default.yaml` under `models` with:
- `family`
- `size_b`
- `quantized`
- `pair_id`
- `model_id`
- `benchmarks`

## 2. Add a New Benchmark Plugin
Implement the plugin contract in `ethical_benchmark/benchmarks/base.py`:
- `load_items(max_samples, seed)`
- `build_prompt(item)`
- `score_response(item, response)`
- `aggregate(records)`

Register the plugin in `ethical_benchmark/benchmarks/registry.py` and add benchmark config in YAML.

## 3. Add a New Pairwise Metric
Extend `ethical_benchmark/analysis/compare_quant_pairs.py`:
- include metric extraction logic
- add delta computation
- update CSV/JSON export fields

## 4. Add Cluster Policies
`ethical_benchmark/cluster/generate_jobs.py` reads SLURM defaults from config. Extend config and template rendering for cluster-specific directives as needed.

## 5. Backward Compatibility
Legacy toxicity/bias/factuality modules remain in the repository but are not default in the quantization study workflow.
