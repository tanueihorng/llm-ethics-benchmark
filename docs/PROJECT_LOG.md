# PROJECT_LOG.md — Single Source of Truth

| Field | Value |
|---|---|
| **Project** | CCDS25-1136 — Benchmarking Ethical Performance of Open-Source LLMs |
| **Student** | TAN UEI HORNG (UTAN001) — UTAN001@e.ntu.edu.sg |
| **Supervisor** | Dr. Zhang Jiehuang — jiehuang.zhang@ntu.edu.sg |
| **School** | College of Computing and Data Science, Nanyang Technological University |
| **Last updated** | 2026-05-26 (22:30 UTC+8) |
| **Last updated by** | TAN UEI HORNG |

> Whenever you edit this file, bump the **Last updated** date (and name, if a collaborator other than the student) to reflect the most recent change. This is the cheapest way to see at a glance whether the log is current.

This file is the single source of truth for project state, decisions, open tasks, and changes. **Update it whenever something changes.** Every other markdown/docx in the repo is either (a) permanent reference material (methodology, runbooks, the FYP report itself) or (b) historical archive — none of them are authoritative for "what's happening now." Only this file is.

If you're a future Claude session or a collaborator reading this for the first time, start here. Status, open tasks, and the recent changelog below are the fastest way to get oriented.

---

## 1. Status snapshot (updated 2026-05-24)

| Dimension | State |
|---|---|
| **Scope (locked)** | 4-bit NF4 quantization study, matched-pair design, on-the-fly quantization from identical baseline weights. Three pairs: Qwen 2B, Qwen 4B, Llama 3.2 3B. Three benchmarks: HarmBench, XSTest, MMLU (6-subject subset). |
| **Framework** | Complete. 122 tests passing. CLI + Makefile + SLURM orchestration in place. |
| **TC1 access** | Approved (account `utan001`, partition `UGGPU-TC1`, QoS `normal`, window 03/2026–11/2026). `fyp-tc1` conda environment created. Repo synced to `/tc1home/FYP/utan001/fyp_quant/repo` as of 2026-04-08 (needs re-sync to pick up the May changes — see open action). |
| **SLURM scripts** | 6 per-model `*_matrix.sbatch` files generated under `slurm/jobs_tc1/`, with offline env vars (`HF_*_OFFLINE=1`) in setup_commands. 18 short smoke sbatch files under `slurm/jobs_tc1_smoke/` for cache validation. |
| **Hugging Face gating** | Pending. Llama 3.2 3B is gated; need `huggingface-cli login` on TC1 with an HF token whose account has accepted the Meta Llama community license. |
| **Datasets / model weights** | Not yet pre-cached on TC1. Pre-cache step (head node) is the next action. |
| **Experiments** | Not yet run. 0 of 6 model-level jobs submitted. |
| **Current FYP report** | `docs/FYP_Report_2026-05-24.docx` (interim report, framework-complete + experiments-pending framing). |
| **Last supervisor update** | 2026-03-09 (~2.5 months ago). Re-engagement email drafted in conversation but not yet sent. |

---

## 2. Open actions (work to do, in priority order)

Tasks are numbered globally as `T<N>` so they can be referenced unambiguously in conversation and in changelog rows. Tick `[x]` when done and move the item to the changelog with a brief note. **Do not delete unchecked items without recording a decision.** When adding a new task, give it the next available `T<N>` number; do not renumber existing tasks.

### 2.1 Immediate (this week)

- [ ] **T1. Send progress email to Dr. Zhang.** Draft exists in conversation history (2026-05-23). Cover: framework complete, TC1 environment provisioned, experiments pending submission. Include the corrected scope (2B + 4B + Llama 3B, on-the-fly quantization).
- [x] ~~**T2. Re-sync repo to TC1.**~~ Done 2026-05-26: pushed commit `63b7a3e` from Mac; TC1 `git pull` confirms repo at that commit. Going forward, sync is `git push` (Mac) + `git pull` (TC1).
- [ ] **T3. Run `MyTCinfo` on TC1.** Confirm actual storage quota (assumed ~300 GB in the report; verify and update Table 5.1 if different).
- [x] ~~**T4. One-time HF login on TC1 head node.**~~ Done 2026-05-26: `huggingface_hub.login()` ran from inline Python; token `tc1-fyp-read` (read scope) registered for account `ueihorng`. Llama 3.2 license acceptance still TBD — will surface as a clear 401 in T5 if not accepted.
- [ ] **T5. Pre-cache datasets + model weights on TC1 head node.** `make prefetch CONFIG=configs/tc1.yaml`. Takes ~15–30 minutes, ~20 GB total. Pure HTTP I/O (policy-compliant per TC1 user guide p.7–9).

### 2.2 After pre-cache (validation)

- [ ] **T6. Run one smoke sbatch.** `sbatch slurm/jobs_tc1_smoke/qwen_2b_base__harmbench.sbatch`. Verify `results/qwen_2b_base/harmbench/summary.json` appears with a sensible `attack_success_rate`.
- [ ] **T7. Check job efficiency.** `seff <jobid>` on the smoke job. If memory efficiency is borderline (>80%), bump `mem: 10G` to `16G` in `configs/tc1.yaml` and regenerate sbatch files before launching the matrix.

### 2.3 After smoke passes (run the matrix)

- [ ] **T8. Submit the 6-job matrix.** `make cluster-submit CONFIG=configs/tc1.yaml JOBS_DIR=slurm/jobs_tc1`. With MaxJobsPU=2, expect 3 sequential pairs.
- [ ] **T9. Monitor.** `squeue -u utan001`, `MyJobHistory`, `seff <jobid>` after each completes.
- [ ] **T10. Run analysis.** `make analyze CONFIG=configs/tc1.yaml` to compute pairwise deltas and interpretation labels. Outputs to `results/analysis/`.

### 2.4 After results are in (write-up)

- [ ] **T11. Populate Chapter 6 of the FYP report.** Replace the `(pending)` cells in Table 6.1 with actual values. Adjust narrative tense from "intended" to past tense throughout the chapter.
- [ ] **T12. Update Chapter 7 (Discussion).** Add empirical observations: which pairs landed in which interpretation labels, surprises vs expectations.
- [ ] **T13. Update Chapter 10 (Conclusion).** Restate findings in present tense.
- [ ] **T14. Rebuild the docx** with the populated content (`make report`).
- [ ] **T15. Submit final report** to Dr. Zhang.

### 2.5 Backlog / nice-to-have (not blocking)

- [ ] **T16.** Commit the current changes (configs/tc1.yaml, scripts/prefetch_tc1.py, Makefile additions, regenerated sbatch files, CLI fix, test fixture update, runbook revision, FYP report) as a coherent git commit.
- [ ] **T17.** Consider migrating from `techwithsergiu/Qwen3.5-text-*` baselines to official `Qwen/Qwen2.5-*-Instruct` baselines with on-the-fly quantization. Stronger external validity. Trade-off: not the same exact weights you've been planning around.
- [ ] **T18.** Add a multi-seed (T=0.7) sensitivity arm for one pair, to provide an independent variance estimate. Future-work item; not required for the interim report.
- [ ] **T19.** Stochastic-decoding sensitivity, multi-method quantization (GPTQ/AWQ/GGUF), 0.5B/7B scale extension — all listed in Chapter 9 of the FYP report as future work; track here when scoped.

---

## 3. Decisions (with rationale, append-only)

Decisions are numbered globally as `D<N>` for cross-reference (e.g. "see D3"). Each decision is dated and includes the rationale at the time. When a decision is later reversed, add a new entry referencing the old one — never edit history.

### D1. 2026-05-23 — Drop Qwen 0.8B; finalise the three-pair design

**Decision:** Study uses three pairs: Qwen 2B, Qwen 4B, and Llama 3.2 3B (cross-family). Qwen 0.8B is dropped.

**Rationale:** At 0.8B, the BF16 baseline itself is at or barely above MMLU chance (~25%), making the alignment-versus-capability disambiguation unreliable (the capability anchor degenerates). Removing it eliminates the largest scientific risk in the study. The 2× scale gap (2B → 4B) is sufficient to test scale sensitivity within the compact-deployment regime, even though the original 5× gap (0.8B → 4B) would have been more dramatic. The scope is reframed from "scale-sensitivity across small-to-compact models" to "quantization effects in the compact-deployment regime (2B–4B)."

### D2. 2026-05-23 — On-the-fly NF4 quantization from identical baseline weights

**Decision:** Both members of every pair load from the same Hugging Face `model_id`. The only difference is whether `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` is injected at `from_pretrained` time.

**Rationale:** Eliminates publisher- and conversion-asymmetry as confounds. Earlier plan used a pre-quantized `*-bnb-4bit` checkpoint, which conflated quantization effect with the specific publisher's NF4 pipeline. On-the-fly quantization is the strongest practical isolation of quantization as the experimental variable. Loader (`models/loader.py:_build_bnb_4bit_config`) and config schema both updated to support this. Test `test_models.TestPromptFormatting` confirms `enable_thinking=False` is also handled correctly.

### D3. 2026-05-23 — Group SLURM jobs by model, not by benchmark

**Decision:** `slurm/jobs_tc1/` contains 6 sbatch files (one per model alias), each running all three benchmarks sequentially with the model loaded only once. Previous config produced 18 per-(model, benchmark) jobs.

**Rationale:** Loading a model is the most expensive operation in the pipeline; reusing the load across three benchmarks per job is a ~3× saving on cold-load overhead. Fits well within the 6h walltime budget. With MaxJobsPU=2, scheduling pressure is reduced from 18 → 6 submissions. Implementation via `--group_by model` flag on `cluster-generate`; matrix runner uses `try/finally` to release GPU memory between models if needed.

### D4. 2026-05-24 — Strict offline mode on TC1 compute nodes

**Decision:** All datasets and model weights are pre-cached on the TC1 head node via `make prefetch`. SLURM jobs export `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` in setup_commands.

**Rationale:** Compute nodes may not have outbound internet (architecture diagram in TC1 user guide is ambiguous). Even if they do, runtime downloads risk burning the 6h walltime budget on slow HuggingFace mirrors, and gating failures (Llama) would surface mid-job rather than at setup. Offline mode converts any cache miss into an immediate clear error. Pre-cache on the head node is policy-compliant — equivalent to `pip install` / `conda install` which the user guide explicitly demonstrates on the head node (p.7–9).

### D5. 2026-05-24 — Smoke validation via sbatch, not srun

**Decision:** The initial cluster smoke test is submitted as an sbatch job (5 prompts, qwen_2b_base on HarmBench). No interactive `srun` sessions for validation work.

**Rationale:** TC1 user guide p.19 explicitly says "all users are advised to use the command 'sbatch' for job submission." srun ties up an SSH session and risks killing the job on disconnect. sbatch is the documented path and survives session loss.

### D6. 2026-05-24 — Cross-agent file convention + FYP report regeneration rule

**Decision:** The repo carries two agent-instruction files (`CLAUDE.md` for Claude Code, `AGENTS.md` for Codex / Cursor / Aider / others) with intentionally identical content. The FYP report docx (`docs/FYP_Report_<date>.docx`) is regenerated from `scripts/build_fyp_report.js` via `make report` whenever a change affects report-worthy content (methodology, system design, experimental setup, results, limitations, or appendices).

**Rationale:** User explicitly stated they would use multiple coding agents (Codex + Claude). Different agents auto-load different files, so a single canonical file would only be read by one of them. Duplication with a sync discipline is more reliable than tooling. For the report: handing a stale docx to the supervisor undermines the deliverable; making `make report` a one-second rebuild removes the friction of keeping it current. Source-of-truth is the JS builder; the docx is an artifact and should never be hand-edited.

---

## 4. Changelog (one row per change; newest first)

Every change to the repo gets one row. Timestamps are local time (UTC+8 / Asia/Singapore) in `YYYY-MM-DD HH:MM` format. The **Files** column lists the files touched; the **Change** column summarises what was done; the **Why / Notes** column captures the rationale or the user request that triggered it. The **Report?** column shows whether the FYP report docx was regenerated (`yes` = `make report` ran; `no` = change was not report-worthy; `n/a` = pre-convention entry).

| When (UTC+8) | Files | Change | Why / Notes | Report? | Who |
|---|---|---|---|---|---|
| 2026-05-26 22:30 | `data/xstest_v2_prompts.csv` (new, 450 rows), `ethical_benchmark/benchmarks/xstest.py`, `ethical_benchmark/quant/config_schema.py`, `scripts/prefetch_tc1.py`, `configs/tc1.yaml`, `configs/default.yaml` | Bundled the canonical XSTest v2 prompts CSV directly in the repo at `data/xstest_v2_prompts.csv` (38 KB, 250 safe + 200 unsafe). Added `local_csv` field to `BenchmarkEntry` schema and `XSTestConfig` dataclass; XSTest plugin now loads from the CSV when `local_csv` is set (lazy-imports HF `datasets` only if it isn't). Updated `scripts/prefetch_tc1.py` to skip HF prefetch for benchmarks that use `local_csv`. Both configs now point at the local CSV. Verified locally: plugin returns exactly 250 benign / 450 total items. | Two HF dataset attempts failed — `allenai/xstest-response` is response data with wrong splits, `paul-rottger/xstest-prompts` doesn't exist on the Hub (the data lives in the GitHub repo as a CSV, not as an HF dataset). Bundling the canonical CSV is the most reproducible approach: zero HF dependency for XSTest at runtime, fully versioned in git, ~40 KB cost. Source URL: https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv. | yes | TAN UEI HORNG |
| 2026-05-26 21:15 | `configs/tc1.yaml`, `configs/default.yaml`, `ethical_benchmark/benchmarks/xstest.py` | Switched XSTest dataset from `allenai/xstest-response` to `paul-rottger/xstest-prompts` (the canonical original XSTest dataset from Röttger et al. 2023). Changed split from `test` to `prompts`. Strengthened `_is_benign_prompt` to handle XSTest v2's `contrast_` prefix convention on the `type` field (unsafe prompts are categorised as `contrast_<category>` in v2) and an id-based fallback. | TC1 smoke test surfaced two problems with `allenai/xstest-response`: (1) splits are `response_harmfulness` / `response_refusal`, not `test`; (2) more fundamentally, the dataset contains pre-generated model responses for evaluation, not the original prompts. We generate responses ourselves, so we want the prompts-only dataset. `paul-rottger/xstest-prompts` is ungated, canonical, and used by the original paper. | yes | TAN UEI HORNG |
| 2026-05-26 21:00 | `ethical_benchmark/benchmarks/harmbench.py`, `ethical_benchmark/benchmarks/xstest.py`, `ethical_benchmark/quant/config_schema.py`, `scripts/prefetch_tc1.py`, `configs/tc1.yaml`, `configs/default.yaml` | Added `config_name` field to `BenchmarkEntry` schema and to HarmBench/XSTest plugin dataclasses; both plugins now pass the config to `load_dataset()` when set. Updated `scripts/prefetch_tc1.py` to forward `config_name` (with a fallback default of `"standard"` for `walledai/HarmBench` even if the YAML omits it). Set `config_name: standard` on the HarmBench entry in both `configs/tc1.yaml` and `configs/default.yaml`. | TC1 dataset-load smoke test failed: `walledai/HarmBench` has three configs (`standard`, `contextual`, `copyright`) and refuses to load without one. We need the canonical 400-behaviour `standard` set. Bug would have killed the full matrix at the very first dataset load. T2 (re-sync) and T4 (HF login) confirmed complete on TC1 — see notes below. | no | TAN UEI HORNG |
| 2026-05-26 20:50 | (TC1 session) | T2 confirmed: TC1 repo at commit `63b7a3e` via `git pull --ff-only`. T4 confirmed: HF login written via `huggingface_hub.login(token=...)`; `whoami()` returns account `ueihorng` (TAN UEI HORNG) with token `tc1-fyp-read` (read scope). Upgraded `huggingface_hub` 1.9.1 → 1.16.1 in the `fyp-tc1` conda env. | TC1-side checkpoint — no code changes; just status verification. | no | TAN UEI HORNG |
| 2026-05-24 01:50 | `scripts/build_fyp_report.js`, `docs/FYP_Report_2026-05-24.docx` | Fixed numbered-list numbering in the FYP report docx. All five numbered lists (Ch 1.5 Contributions, Ch 2.5 Research Gaps, Ch 8 Limitations, Ch 9 Future Work, References) were sharing one global counter — Ch 2.5 started at 6, Ch 8 at 10, Ch 9 at 19, References at 25. Refactored the builder: introduced `numberedList(items)` helper that mints a fresh `numlist<N>` numbering reference per call; pre-declared 20 such references in `numbering.config`; replaced all 5 list blocks with `...numberedList([...])`. Each list now restarts at 1. | User reported: "the issue I mentioned earlier is this Word doc" — the numbering bug in the docx was the actual issue, not the markdown numbering. | yes | TAN UEI HORNG |
| 2026-05-24 01:35 | `docs/TC1_CLUSTER_RUNBOOK.md`, `docs/PROJECT_LOG.md` | Renumbered TC1 runbook sections so they start at §1 cleanly: promoted the prior `## 0. Approved Account Notes` to unnumbered preamble; renumbered the inserted "Policy" section as §3 and shifted subsequent sections §3–§17 down to §4–§18 (including sub-sections 3.1–3.7 → 4.1–4.7, 6.1–6.2 → 8.1–8.2, etc.); updated internal cross-references that pointed to the old §3.6 / §3.7. Added `T<N>` ordinal IDs to all 19 open actions in PROJECT_LOG.md §2 and `D<N>` IDs (D1–D6) to all decisions in §3 so they can be referenced unambiguously in conversation. | User request: "fix the numbering format in the docs it should start with 1, 2, 3 and so on for individual part". | no | TAN UEI HORNG |
| 2026-05-24 01:20 | `docs/PROJECT_LOG.md`, `scripts/build_fyp_report.js`, `AGENTS.md`, `CLAUDE.md`, `docs/FYP_Report_2026-05-24.docx` | Restructured changelog as a row-per-change table with timestamp + files + change + reason columns. Added Document Revision History appendix to the FYP report. Updated AGENTS.md / CLAUDE.md / §6 Maintenance contract to mandate the new format. | User request: "every new update / change / new task will update the project log table and the word document, and each row will show the date and time and what changes it makes to what files". | yes | TAN UEI HORNG |
| 2026-05-24 00:55 | `docs/PROJECT_LOG.md`, `scripts/build_fyp_report.js`, `docs/FYP_Report_2026-05-24.docx` | Added prominent metadata table (name, date, supervisor, "Last updated") to PROJECT_LOG.md. Strengthened FYP report cover page: bold author + date at 26pt; supervisor email added; running header now shows project code + author on the left and report type + date on the right. | User request: "put date and name on the log and word doc as well". | yes | TAN UEI HORNG |
| 2026-05-24 00:40 | `AGENTS.md` (new), `CLAUDE.md`, `scripts/build_fyp_report.js` (moved in from /tmp), `Makefile` (+ `report` target), `README.md`, `docs/PROJECT_LOG.md` | Created `AGENTS.md` as cross-agent instruction file (identical to `CLAUDE.md`). Moved docx builder into the repo. Added `make report` target. Codified the rule that report-worthy changes regenerate the docx. | User request: also using Codex (not just Claude); needs AGENTS.md so both agents bootstrap with same context. Plus: "every time new changes / task / update should update the log AND word docs". | yes | TAN UEI HORNG |
| 2026-05-24 00:30 | `docs/PROJECT_LOG.md` (new), `docs/archive/*` (9 files moved), `README.md`, `CLAUDE.md` | Created `docs/PROJECT_LOG.md` as single source of truth. Archived 9 superseded / redundant docs (3 day-of TC1 logs from 04-08, FYP plan duplicates, the 05-23 report version, HTML exports of docx files). Reduced `docs/` from 14 active files to 12 + `archive/`. | User request: "I also need a single source of truth ... a lot of documents scattered across the repo". | no | TAN UEI HORNG |
| 2026-05-24 00:15 | `configs/tc1.yaml`, `scripts/prefetch_tc1.py` (new), `Makefile` (+`prefetch`, +`cluster-smoke`), `docs/TC1_CLUSTER_RUNBOOK.md`, `slurm/jobs_tc1/*` (6 regenerated), `slurm/jobs_tc1_smoke/*` (18 new), `docs/FYP_Report_2026-05-24.docx` (new) | TC1 policy review + offline-mode hardening. Added `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` to sbatch setup_commands. Built head-node prefetch script. Rewrote runbook policy section with verbatim quotes from the user guide. Added FYP report §5.3 (Cluster Usage Policy), §5.5 (Offline-Mode Strategy and Pre-Cache). | After reading the authoritative TC1 PDFs (UG-UserGuide, FYP form): confirmed strict head-node policy, sbatch-not-srun preference, QoS limits. Compute-node internet status is ambiguous; offline mode is the robust default. | yes | TAN UEI HORNG |
| 2026-05-23 evening | `configs/tc1.yaml`, `configs/default.yaml`, `ethical_benchmark/models/loader.py`, `ethical_benchmark/models/generation.py`, `ethical_benchmark/cluster/generate_jobs.py`, `fyp_cli.py`, `tests/test_quant_pipeline_utils.py`, `docs/FYP_Report_2026-05-23.docx` (now archived) | Three-pair scope finalisation. Dropped Qwen 0.8B; both pair members now share the same `model_id` (on-the-fly NF4). 4-bit compute dtype now follows model dtype. `enable_thinking=False` passed to chat template. Added `--group_by {benchmark, model}` flag (default `model` for TC1). Fixed CLI argparse bug (subparser `_add_common_options` silently overwrote root `--config`). Generated initial FYP report docx. | 0.8B baseline floors on MMLU → alignment-vs-capability disambiguation degenerates. On-the-fly NF4 from identical baseline weights eliminates publisher / conversion asymmetry. Group-by-model SLURM saves 2× model loads per job. Argparse fix unblocks Makefile-style `--config X cmd` invocation. | n/a | TAN UEI HORNG |
| 2026-05-13 | `ethical_benchmark/*` (full framework, ~2800 LOC), `tests/*` (113 tests), `docs/*.md` (6 reference docs) | Built the full quantization benchmarking framework: three benchmark plugins (HarmBench, XSTest, MMLU) with deterministic regex scoring; HF model loader with dtype/device resolution and 4-bit BNB integration; pairwise delta analysis with interpretation labels; CLI + Makefile; SLURM orchestration with resume logic. Wrote methodology / datasets / metrics / limitations / extensibility / runbook docs. | Pivot from broad ethical benchmarking (overlap with TrustLLM/DecodingTrust/SafetyBench) to focused quantization-safety study. Commits `59c81a4` (framework), `e793ad3` (docs). | n/a | TAN UEI HORNG |
| 2026-04-08 | `configs/tc1.yaml` (new), `ethical_benchmark/cluster/generate_jobs.py` (work_dir + setup_commands), `ethical_benchmark/evaluators/__init__.py`, `ethical_benchmark/models/__init__.py` (lazy imports), TC1 remote: `fyp-tc1` conda env, repo synced to `/tc1home/FYP/utan001/fyp_quant/repo` | TC1 access activation + initial environment. First SSH login (account activated). Confirmed QoS via `MyTCinfo` (cpu=20, gres/gpu=1, mem=64G, MaxSubmit=2, MaxWall=06:00:00). Created `fyp-tc1` conda env (Python 3.11.15, torch 2.11.0+cu130, transformers 5.5.0). Locally: added TC1 config, refactored model/evaluator imports to be lazy. | TC1 onboarding. (Smoke run was the next step at end of day but was not actioned for six weeks — see 2026-05-24 entries for renewed plan.) Consolidated here from the three (now-archived) day-of logs: `TC1_ACTIVITY_*.md`, `TC1_WORK_LOG_*.md`. | n/a | TAN UEI HORNG |
| 2026-02-08 | (initial repo skeleton) | Repository initialised. Commit `c0684e3`. | Project kick-off. | n/a | TAN UEI HORNG |
| Pre-2026-02 | (not in repo) | Literature review: ~21 papers (TrustLLM, DecodingTrust, SafetyBench, HarmBench, XSTest, MMLU, quantization-safety literature). Identified the four research gaps. Email to Dr. Zhang on 2026-03-09 proposing the focused scope. | Pre-onboarding work. | n/a | TAN UEI HORNG |

---

## 5. Reference (permanent docs — not duplicated here)

Each of these has a single, well-defined purpose. They are stable references, edited only when their topic itself changes.

| Document | Purpose |
|---|---|
| `README.md` | Top-level repo intro and quick orientation. |
| `CLAUDE.md` | Guidance for Claude Code sessions working on this repo. Mandates maintaining this log. |
| `docs/methodology.md` | Methodology reference (experimental design, scoring, interpretation labels). |
| `docs/datasets.md` | Per-benchmark dataset notes. |
| `docs/evaluation_metrics.md` | Metric definitions and formulas. |
| `docs/limitations.md` | Standing limitations of the design. |
| `docs/extensibility.md` | How to add a new model / benchmark / metric. |
| `docs/USER_GUIDE.md` | Quick-start guide for local development. |
| `docs/FYP_REPO_GUIDE.md` | Full operational manual (dissertation-style). |
| `docs/TC1_CLUSTER_RUNBOOK.md` | Step-by-step TC1 operations, with verified policy block and pre-cache procedure. |
| `docs/CCDSGPU-TC1-UG-UserGuide (1).pdf` | Authoritative TC1 user guide (operator-provided). |
| `docs/CCDSGPU-TC1-Cluster-FYP-Form.pdf` | TC1 application form (record). |
| `docs/FYP_Report_2026-05-24.docx` | **Current** FYP interim report. |
| `docs/archive/` | Superseded documents (kept for traceability, not authoritative). |

---

## 6. Maintenance contract (the rules that keep this repo coherent)

These are the standing conventions. They apply to every agent session (Claude, Codex, Cursor, Aider, or a human) and to every commit.

1. **Every change writes one row to the §4 changelog table.** Columns: `When (UTC+8) | Files | Change | Why / Notes | Report? | Who`. Timestamp is `YYYY-MM-DD HH:MM` local time. `Files` lists the actual paths touched. `Why / Notes` captures the rationale or the user request that triggered it. `Report?` is `yes` if `make report` was run, `no` if the change wasn't report-worthy. `Who` is the human author or the agent name (e.g., "Claude" or "Codex"). Append at the top.
2. **Every report-worthy change regenerates the FYP report** (`make report`) and bumps the `Report?` column to `yes`. A change is report-worthy if it touches methodology, system design, experimental setup, results, limitations / future work, or appendices. Edit `scripts/build_fyp_report.js`, never the docx. The docx has its own "Document Revision History" appendix that mirrors the report-affecting subset of this changelog.
3. **Update the §1 metadata table when you edit this file.** Bump `Last updated` to today's date and `Last updated by` to your name.
4. **Update §2 (open actions) when work appears or finishes.** Tick `[x]` when done and move the item (one line) into the new changelog row.
5. **Decisions go to §3, append-only.** A decision is a choice between alternatives with rationale. Never edit past decisions; if reversing one, append a new entry referencing the old.
6. **`AGENTS.md` and `CLAUDE.md` stay identical** (apart from the agent-name header). Edit both in the same commit. `diff AGENTS.md CLAUDE.md` should show only the header differences.
7. **No hand-edits to `.docx` files.** Generated docx files (FYP report) are artifacts; edit the source builder script.
8. **Never push or commit without a corresponding PROJECT_LOG.md entry.** Treat the log as part of the change, not a separate chore.
9. **Superseded files go to `docs/archive/`**, not the trash. Keep historical artifacts for traceability.

## 7. How to use this log

**When you (or a future Claude session) make a change to the repo:**
1. Update the relevant section here in the same change. Don't push code without a log entry.
2. If the change is just a file edit (typo, small refactor), add one line to the changelog.
3. If the change involves a real decision (choosing approach A over B, dropping scope, adopting a new convention), add an entry to §3 (Decisions) with the rationale.
4. If the change creates work for later, add to §2 (Open Actions).
5. If the change affects current state (e.g., experiments now running, environment broken, quota exhausted), update §1 (Status snapshot) in place.

**When you finish a task:**
1. Tick `[x]` next to the item in §2.
2. Move it (with a one-line note) into the latest changelog entry, or create a new dated changelog entry if it's substantial.

**When you make a decision:**
1. Append to §3 with date, decision, and rationale.
2. Never edit prior decisions. If reversing one, add a new entry that references the old.

**When in doubt, write it down here.** This file is the only thing that survives session resets, supervisor handovers, and your own forgetful future self.
