PYTHON ?= python
CONFIG ?= configs/default.yaml
RESULTS_DIR ?= results
JOBS_DIR ?= slurm/jobs
MODEL ?= qwen_2b_base
BENCHMARK ?= harmbench
SEED ?= 42
DEVICE ?= auto
MAX_SAMPLES ?=
BATCH_SIZE ?=
ANALYSIS_DIR ?= results/analysis
GROUP_BY ?= model
SCRIPT ?= run_quant_matrix.py

.PHONY: smoke run matrix analyze prefetch report cluster-generate cluster-submit cluster-dry cluster-check cluster-all cluster-smoke

smoke:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) smoke -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

run:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) run -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

matrix:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) matrix -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

analyze:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) analyze -o $(ANALYSIS_DIR)

cluster-generate:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-generate -j $(JOBS_DIR) -s $(SEED) -d $(DEVICE) --group_by $(GROUP_BY) --script $(SCRIPT) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

cluster-submit:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-submit -j $(JOBS_DIR)

cluster-dry:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-submit -j $(JOBS_DIR) --dry_run

cluster-check:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-check -j $(JOBS_DIR)

cluster-all: cluster-generate cluster-submit cluster-check

# Pre-cache datasets and model weights on the TC1 HEAD node so subsequent
# SLURM jobs can run with HF_*_OFFLINE=1. Run once after first login.
# Per the TC1 user guide (page 16), package installs and downloads on the
# head node are explicitly demonstrated and allowed.
prefetch:
	$(PYTHON) scripts/prefetch_tc1.py --config $(CONFIG)

# Regenerate the FYP interim report docx. Run after any "report-worthy"
# change (methodology, scope, results, anything cited in Chapters 3-6 of
# the report). Output: docs/FYP_Report_<date>.docx.
# Requires docx-js installed globally: npm install -g docx
report:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_report.js

# Generate small per-(model,benchmark) smoke sbatch files (5 samples each)
# for verifying the offline-cache path on a real compute node before
# submitting the full matrix. Pick ONE and submit it; you do not need to
# submit all of them. Always use sbatch (not srun) per TC1 user guide p.19.
SMOKE_JOBS_DIR ?= slurm/jobs_tc1_smoke
cluster-smoke:
	rm -rf $(SMOKE_JOBS_DIR)
	$(PYTHON) fyp_cli.py cluster-generate --config $(CONFIG) --results_dir $(RESULTS_DIR) -j $(SMOKE_JOBS_DIR) -s $(SEED) -d cuda --group_by benchmark --script run_quant_benchmark.py -n 5
	@echo ""
	@echo "Smoke sbatch files written to $(SMOKE_JOBS_DIR)/ (one per model x benchmark)."
	@echo "Submit ONE to validate the offline cache before running the full matrix, e.g.:"
	@echo "    sbatch $(SMOKE_JOBS_DIR)/qwen_2b_base__harmbench.sbatch"
