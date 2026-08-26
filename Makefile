PYTHON ?= python
CONFIG ?= configs/default.yaml
RESULTS_DIR ?= results
# `make smoke` force-restarts (deletes prior raw). Default it to a scratch tree so a
# sanity check can never delete committed evidence under results/ or results_512/
# (the pipeline also refuses such deletes; see run_quant_benchmark._guard_protected_delete).
SMOKE_RESULTS_DIR ?= results_smoke
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

.PHONY: backup smoke run matrix analyze prefetch report thesis interim rewrite rewrite-v2 rewrite-interim rewrite-interim-v3 rewrite-interim-v5 rewrite-interim-compact rewrite-interim-progress rewrite-latex viva report-humanized thesis-humanized interim-humanized claim-registry claim-surfaces verify-claims verify-claim-surfaces architecture-diagram diagrams cluster-generate cluster-submit cluster-dry cluster-check cluster-all cluster-smoke agent-start agent-status agent-check agent-manifest agent-handoff agent-dashboard agent-tc1-checklist harness-eval dashboard

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

smoke:
	$(PYTHON) fyp_cli.py --config $(CONFIG) --results_dir $(SMOKE_RESULTS_DIR) smoke -m $(MODEL) -b $(BENCHMARK) -s $(SEED) -d $(DEVICE) $(if $(MAX_SAMPLES),-n $(MAX_SAMPLES),)

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
claim-registry:
	$(PYTHON) scripts/claim_registry.py --write

claim-surfaces: claim-registry
	$(PYTHON) scripts/sync_claim_surfaces.py --write

# Schematic diagrams embedded in Chapters 3-5 of every document (study design,
# system architecture, TC1 execution). Hand-authored SVG → PNG via
# @resvg/resvg-js; deterministic, so a rebuild is a no-op unless the script
# changed. Data plots stay in scripts/make_figures.py.
# Output: docs/figures/{study_design,system_architecture,tc1_execution}.png
diagrams:
	node scripts/make_diagrams.js

report: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_report_v5.js
	@touch .local_build_state

# The corpus-informed rewrite — the SUBMISSION CANDIDATE selected in T52/D59.
# `make report` above still builds v5, which remains the canonical build path until
# a deliberate promotion (a T15 decision); these targets exist because until
# 2026-08-10 the Makefile had no way at all to rebuild the document being submitted,
# so the only route was knowing the script path by heart.
# Python builders, not docx-js: use $(PYTHON), no NODE_PATH. Needs python-docx.
# Output: docs/rewritten_fyp_report_v1.docx
rewrite: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_report_v1.py
	@touch .local_build_state

# Second, non-destructive editorial pass: replaces the repository/audit dialect
# with conventional academic prose while preserving v1's citations and numbers.
# Output: docs/rewritten_fyp_report_v2.docx
rewrite-v2: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_report_v2.py
	@touch .local_build_state

# Third, comprehensive natural human-academic essay pass: full essay flow across all
# seven chapters while strictly locking numerical tokens and citation sequences.
# Output: docs/rewritten_fyp_report_v3.docx
rewrite-v3: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_report_v3.py
	@touch .local_build_state

# Fourth rewrite line: preserves the v3 report while replacing the Chapter 1
# opening with the author's HW0288 funnel introduction.
# Output: docs/rewritten_fyp_report_v4.docx
rewrite-v4: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_report_v4.py
	@touch .local_build_state

# Interim-submission build of the same prose (T53, due 2026-08-24). NOTE: this
# imports the report builder and calls build_report(), so it rewrites
# docs/rewritten_fyp_report_v1.docx as a side effect before writing its own output.
# Output: docs/rewritten_fyp_interim_v1.docx
rewrite-interim: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_v1.py
	@touch .local_build_state

# Interim-submission build of the V3 prose line (T53; the user selected the v3
# plain-academic voice for submission after reviewing v1/v2/v3 side by side).
# Same retitle-only contract as rewrite-interim above.
# Output: docs/rewritten_fyp_interim_v3.docx
rewrite-interim-v3: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_v3.py
	@touch .local_build_state

# Interim v4 copy of the v4 report line. The v3 interim remains unchanged.
# Output: docs/rewritten_fyp_interim_v4.docx
rewrite-interim-v4: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_v4.py
	@touch .local_build_state

# Language-cleanup rewrite of the v4 line (2026-08-25). Building the interim
# also regenerates docs/rewritten_fyp_report_v5.docx as a side effect, exactly
# like the rewrite-interim contract above.
# Outputs: docs/rewritten_fyp_report_v5.docx + docs/rewritten_fyp_interim_v5.docx
rewrite-interim-v5: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_v5.py
	@touch .local_build_state

# Compact progress-centred interim derived non-destructively from verified v4.
# Output: docs/rewritten_fyp_interim_compact.docx
rewrite-interim-compact: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_compact.py
	@touch .local_build_state

# User-directed interim: supplied Chapter 1, full v4 Chapters 2-4, then a
# progress chapter that defers the detailed validation and discussion writing.
# Output: docs/rewritten_fyp_interim_progress.docx
rewrite-interim-progress: claim-surfaces
	$(PYTHON) scripts/build_rewritten_fyp_interim_progress.py
	@touch .local_build_state

# Report version map: regenerate docs/REPORT_VERSIONS.md, the machine inventory
# (docs/report_versions/inventory.json) and a chapter/paragraph-level change
# record for every consecutive pair in every report lineage, driven by
# configs/report_versions.yaml. The .full.md/.full.json siblings quote report
# prose wholesale and stay gitignored (D60 rationale); summaries are committed.
# Outputs: docs/REPORT_VERSIONS.md + docs/report_versions/
report-versions:
	$(PYTHON) scripts/report_tools/report_version_diff.py all

# Overleaf/LaTeX mirrors of the rewrite line, GENERATED from the docx builder rather
# than hand-maintained: the builder's content helpers are swapped for recorders, so
# the tex carries the same prose and the same registry-derived numbers as the docx.
# The two older mirrors (final_thesis/, interim_report/) track the superseded
# JS builders and are left untouched. --compile verifies with tectonic if installed.
# Outputs: fyp_submission/report_latex/rewritten_{final,interim}/ + _overleaf.zip
rewrite-latex: claim-surfaces
	$(PYTHON) scripts/build_rewritten_latex.py --compile

# Oral-defence deck (T55, 4/7/8 Dec 2026). Two renderers over ONE source: the HTML
# builder imports the pptx builder and records its slides, so the showcase copy
# cannot drift from the graded one. Numbers are read from results_512/analysis at
# build time and the build FAILS if an ASR contrast ever survives BH correction.
# Outputs: docs/viva/FYP_Viva_Deck.{pptx,html}
viva:
	$(PYTHON) scripts/build_viva_deck.py
	$(PYTHON) scripts/build_viva_deck_html.py

# Archive the parts of this project that exist ONLY on this machine: the raw model
# outputs (~97 MB, unreproducible without TC1 GPU hours and gated weights), the two
# hand-labelled gold sets, the supervisor correspondence, and the submission tree.
# Those are correctly NOT in git -- .gitignore says the correspondence is "never
# commit", D60 untracked the documents, and the raw outputs hold unredacted harmful
# text -- which is exactly why a laptop failure would lose them and git will not help.
# DEST must be outside the repo; the script refuses otherwise.
#   make backup DEST=~/Dropbox/fyp_backup
backup:
	@test -n "$(DEST)" || { echo "usage: make backup DEST=/path/outside/the/repo"; exit 2; }
	$(PYTHON) scripts/backup_local.py --dest $(DEST)

# Deterministic claim lock (D43): every load-bearing number in the canonical
# report builder is asserted against the committed analysis artifacts. Also
# runs inside pytest via tests/test_report_claims.py.
verify-claims:
	$(PYTHON) scripts/claim_registry.py --check
	$(PYTHON) scripts/verify_report_claims.py
	$(PYTHON) scripts/verify_claim_surfaces.py

verify-claim-surfaces:
	$(PYTHON) scripts/verify_claim_surfaces.py

# Standalone full thesis (separate from the interim report; not overwritten by
# `make report`). v4 = the 512-primary mirror (D41/D43); v1-v3 builders are
# retained as 128-era history and their banner-marked docx live in docs/archive/.
thesis: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_thesis_v4.js
	@touch .local_build_state

# Interim milestone report (~25-30 pp): a shorter, progress-framed derivative of
# the thesis that REUSES the same 512-primary, claim-locked prose/numbers. The
# full report (make report) stays as the final-thesis-grade master; this never
# touches it. Output: docs/FYP_Interim_2026-07-10.docx.
interim: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_interim.js
	@touch .local_build_state

# Humanized variants (separate deliverables; AI-writing-tells removed, prose only;
# every number byte-identical to the originals, verified). Originals stay the
# claim-locked masters. Outputs: docs/FYP_*_humanized.docx.
report-humanized: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_report_humanized.js
	@touch .local_build_state
thesis-humanized: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_thesis_humanized.js
	@touch .local_build_state
interim-humanized: claim-surfaces
	NODE_PATH=$$(npm root -g) node scripts/build_fyp_interim_humanized.js
	@touch .local_build_state

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
