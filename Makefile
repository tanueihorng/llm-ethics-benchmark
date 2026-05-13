PYTHON ?= python
CONFIG ?= configs/default.yaml
RESULTS_DIR ?= results
JOBS_DIR ?= slurm/jobs
MODEL ?= qwen_0_8b_bf16
BENCHMARK ?= harmbench
SEED ?= 42
DEVICE ?= auto
MAX_SAMPLES ?=
BATCH_SIZE ?=
ANALYSIS_DIR ?= results/analysis

.PHONY: smoke run matrix analyze cluster-generate cluster-submit cluster-dry cluster-check cluster-all

smoke:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) smoke -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

run:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) run -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

matrix:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) matrix -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

analyze:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) analyze -o $(ANALYSIS_DIR)

cluster-generate:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-generate -j $(JOBS_DIR) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

cluster-submit:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-submit -j $(JOBS_DIR)

cluster-dry:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-submit -j $(JOBS_DIR) --dry_run

cluster-check:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) cluster-check -j $(JOBS_DIR)

cluster-all: cluster-generate cluster-submit cluster-check
