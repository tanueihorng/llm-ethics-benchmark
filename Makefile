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
PYTEST_ARGS ?= -q
TASK ?=
AGENT ?=

.PHONY: smoke run matrix analyze prefetch report architecture-diagram cluster-generate cluster-submit cluster-dry cluster-check cluster-all cluster-smoke agent-start agent-status agent-check agent-manifest agent-handoff agent-dashboard agent-tc1-checklist harness-eval dashboard

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

smoke:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) smoke -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

run:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) run -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

matrix:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) matrix -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),) $(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),)

analyze:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(RESULTS_DIR) analyze -o $(ANALYSIS_DIR)

agent-status:
	$(PYTHON) fyp_cli.py agent-status

agent-start:
	$(PYTHON) fyp_cli.py agent-start $(if $(TASK),--task $(TASK),) $(if $(AGENT),--agent $(AGENT),)

agent-check:
	$(PYTHON) scripts/agent_check.py --pytest-args $(PYTEST_ARGS)

agent-manifest:
	$(PYTHON) scripts/agent_check.py --write-immutable-manifest

agent-handoff:
	$(PYTHON) scripts/generate_handoff.py

agent-dashboard:
	$(PYTHON) scripts/generate_agent_dashboard.py

agent-tc1-checklist:
	$(PYTHON) scripts/generate_tc1_checklist.py

harness-eval:
	$(PYTHON) scripts/harness_eval.py

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
# docx is pinned in package.json — `npm install` once on a fresh clone.
# (NODE_PATH is kept as a fallback for machines with only a global install;
# local node_modules wins when present.)
report:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_report_v5.js

# Deterministic claim lock (D43): every load-bearing number in the canonical
# report builder is asserted against the committed analysis artifacts. Also
# runs inside pytest via tests/test_report_claims.py.
verify-claims:
	$(PYTHON) scripts/verify_report_claims.py

# Standalone full thesis (separate from the interim report; not overwritten by
# `make report`). v4 = the 512-primary mirror (D41/D43); v1-v3 builders are
# retained as 128-era history and their banner-marked docx live in docs/archive/.
thesis:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_thesis_v4.js

# Interim milestone report (~25-30 pp): a shorter, progress-framed derivative of
# the thesis that REUSES the same 512-primary, claim-locked prose/numbers. The
# full report (make report) stays as the final-thesis-grade master; this never
# touches it. Output: docs/FYP_Interim_2026-07-10.docx.
interim:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_interim.js

# Humanized variants (separate deliverables; AI-writing-tells removed, prose only;
# every number byte-identical to the originals, verified). Originals stay the
# claim-locked masters. Outputs: docs/FYP_*_humanized.docx.
report-humanized:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_report_humanized.js
thesis-humanized:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_thesis_humanized.js
interim-humanized:
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_interim_humanized.js

# Standalone agentic-AI workflow / methods report (how agentic AI tools were used to
# accelerate the project — separate deliverable; touches neither report nor thesis).
# Regenerates figures (local @resvg/resvg-js → run `npm install` once) + the docx.
# Output: docs/Agentic_AI_Workflow_Report_<date>.docx.
agentic-report:
	NODE_PATH=$$(npm root -g) node scripts/build_agentic_report.js

architecture-diagram:
	$(PYTHON) scripts/generate_architecture_diagram.py
	$(PYTHON) scripts/generate_repo_hierarchy_diagram.py
	$(PYTHON) scripts/generate_agent_harness_architecture_diagram.py
	$(PYTHON) scripts/generate_integrated_architecture_diagram.py

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
