## Full FYP report (docx-js line): First full draft (2026-05-23) → Draft update (2026-05-24)

*From:* `docs/archive/FYP_Report_2026-05-23.docx`  
*To:* `docs/archive/FYP_Report_2026-05-24.docx`  

**Paragraph/body blocks:** +37 added · −6 removed · ~10 edited · ⟳0 citation-renumber-only · tables: +1 −0 ~3 · front matter blocks 27→29

*Front matter (cover/abstract/TOC/lists):* +3 −1 ~1 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Supervisor: Dr. Zhang Jiehuang (jiehuang.zhang@ntu.edu.sg)
  - ❌ 23 May 2026
  - ➕ Document date: 24 May 2026
  - ➕ Revision: incorporates TC1 user-guide policy review
  - ➕ Code listing 5.1 Head-node pre-cache invocation (scripts/prefetch_tc1.py).

### Chapters / parts ADDED
- ➕ Appendix G — Document Revision History

### Sections added
- ➕ Chapter 5 — Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints — 5.3 Cluster Usage Policy and Workf…
- ➕ Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — 5.5 Offline-Mode Strategy and Pre-Cache

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix A — Final TC1 Configuration | 7 | 0 | 2 | 0 |
| Appendix B — Example Generated SLURM Script | 3 | 0 | 0 | 0 |
| Appendix G — Document Revision History | 3 | 0 | 0 | 0 |
| Chapter 10 — Conclusion | 0 | 0 | 1 | 0 |
| Chapter 3 — Methodology | 0 | 0 | 2 | 0 |
| Chapter 5 — Experimental Setup | 19 | 3 | 4 | 0 |
| Chapter 8 — Limitations | 2 | 2 | 0 | 0 |

### Removed paragraphs (5 body; 1 front matter)

<details><summary>show</summary>

- `0fc652e2` (12w) Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment — “Table 5.1 TC1 cluster hardware and policy parameters relevant to this study.”
- `750c7b7f` (92w) Chapter 5 — Experimental Setup § 5.3 Hugging Face Access — “The Qwen pairs draw from open-access checkpoints under the techwithsergiu namespace and r…”
- `4e0c6232` (63w) Chapter 5 — Experimental Setup § 5.4 Run Plan — “With MaxSubmit=2 enforced by the cluster policy, the six jobs are processed in three sequ…”
- `e5a0e9eb` (27w) Chapter 8 — Limitations — “Sample-size-driven confidence intervals. With 400 prompts per safety benchmark, binomial-…”
- `f318c92a` (34w) Chapter 8 — Limitations — “Llama gated access. The Llama 3.2 3B pair depends on accepted gating on Hugging Face and …”

</details>

### Added paragraphs (34)

<details><summary>show</summary>

- `1def1819` (38w) Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment — “Table 5.1 TC1 cluster hardware and policy parameters relevant to this study. Higher-wallt…”
- `8db04bae` (109w) Chapter 5 — Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints — “The TC1 user guide imposes strict workflow rules that shape how this study is executed. T…”
- `34952474` (51w) Chapter 5 — Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints — “By contrast, package installation, file transfers, conda environment management, dataset …”
- `dc142faf` (57w) Chapter 5 — Experimental Setup § 5.3 Cluster Usage Policy and Workflow Constraints — “The guide also recommends sbatch over srun for all real job submission: "Avoid using the …”
- `67031cb4` (83w) Chapter 5 — Experimental Setup § 5.4 Hugging Face Access and Gating — “The Qwen pairs draw from open-access checkpoints under the techwithsergiu namespace and r…”
- `a3bab9dc` (82w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “Because the compute nodes may not have outbound internet access, and because runtime down…”
- `1330cc7c` (45w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “The repository provides a helper script (scripts/prefetch_tc1.py) that reads the configur…”
- `227f189d` (3w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “module load anaconda”
- `310ca685` (3w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “source activate fyp-tc1”
- `3d5a5016` (2w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “cd /tc1home/FYP/utan001/fyp_quant/repo”
- `b54f4d42` (7w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “python - <<'PY' # one-time HF login”
- `2c8244ce` (4w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “from getpass import getpass”
- `b11821d1` (4w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “from huggingface_hub import login”
- `4474e39d` (4w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “login(token=getpass("HF token: "), add_to_git_credential=False)”
- `8cabb1fe` (1w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “PY”
- `9c74aed9` (7w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “python scripts/prefetch_tc1.py # or: make prefetch CONFIG=configs/tc1.yaml”
- `7394b036` (117w) Chapter 5 — Experimental Setup § 5.5 Offline-Mode Strategy and Pre-Cache — “The pre-cache step retrieves HarmBench, the six MMLU subjects, and three model repositori…”
- `3433ec2a` (72w) Chapter 5 — Experimental Setup § 5.6 Run Plan — “Before submitting the full matrix, a single short smoke sbatch is submitted (five prompts…”
- `99693f67` (109w) Chapter 5 — Experimental Setup § 5.6 Run Plan — “With MaxJobsPU=2 enforced by the cluster policy, the six jobs are processed in three sequ…”
- `95755278` (30w) Chapter 8 — Limitations — “Sample-size-driven confidence intervals. With 200 HarmBench prompts and 250 benign XSTest…”
- `a566da42` (43w) Chapter 8 — Limitations — “Gated-access dependency. The Llama 3.2 3B pair and HarmBench dataset depend on accepted H…”
- `208acf1a` (2w) Appendix A — Final TC1 Configuration — “config_name: standard”
- `5abd9109` (2w) Appendix A — Final TC1 Configuration — “local_csv: data/xstest_v2_prompts.csv”
- `56427da3` (10w) Appendix A — Final TC1 Configuration — “# Force offline mode (compute nodes may lack outbound internet).”
- `14b0d82b` (9w) Appendix A — Final TC1 Configuration — “# Pre-cache via scripts/prefetch_tc1.py on the HEAD node first.”
- `56817988` (3w) Appendix A — Final TC1 Configuration — “- export HF_HUB_OFFLINE=1”
- `6b059892` (3w) Appendix A — Final TC1 Configuration — “- export HF_DATASETS_OFFLINE=1”
- `c33a5eb6` (3w) Appendix A — Final TC1 Configuration — “- export TRANSFORMERS_OFFLINE=1”
- `404bf4e8` (2w) Appendix B — Example Generated SLURM Script — “export HF_HUB_OFFLINE=1”
- `72095b35` (2w) Appendix B — Example Generated SLURM Script — “export HF_DATASETS_OFFLINE=1”
- `277a4f94` (2w) Appendix B — Example Generated SLURM Script — “export TRANSFORMERS_OFFLINE=1”
- `3bd40285` (49w) Appendix G — Document Revision History — “This appendix records the revision history of this FYP report. It mirrors the report-affe…”
- `02b71e2b` (279w) Appendix G — Document Revision History — “When (UTC+8) \| Version \| Change to the report \|\| 2026-05-27 00:34 \| FYP_Report_2026-…”
- `f9ef82f1` (24w) Appendix G — Document Revision History — “Table G.1 Revision history of this FYP report. The source builder script is `scripts/buil…”

</details>

### Edited paragraphs (10; change window shown)

<details><summary>show</summary>

- (front matter): “Supervisor: Dr. Zhang Jiehuang[+ (jiehuang.zhang@ntu.edu.sg)]”
- Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping: “…g \|\| Harmful compliance \| HarmBench (walledai/HarmBench[+, standard][−4][+2][+loc][−l…”
- Chapter 3 — Methodology § 3.4 Benchmark Selection and Mapping: “…k success. The walledai release on Hugging Face is used[+ with the standard config][+is …”
- Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment: “…h workloads. Account access was approved in March 2026 [+under QoS "normal" ][+ The comp…”
- Chapter 5 — Experimental Setup § 5.1 Hardware and Cluster Environment: “…tan001 \|\| Partition \| UGGPU-TC1 \|\| QoS \| normal \|\| GPU [−r][+model \\| NVIDIA T]…”
- Chapter 5 — Experimental Setup § 5.2 Software Environment: “…L stack. The relevant versions are listed in Table 5.2.[+ Per the TC1 user guide, the ba…”
- Chapter 5 — Experimental Setup § 5.2 Software Environment: “…txt) \|\| datasets \| (installed via requirements.txt) \|\| [+huggingface_hub \\| (insta…”
- Chapter 10 — Conclusion: “…ramework is complete, validated, and ready for cluster [−s][+exec][−bmiss][+t][−and the …”
- Appendix A — Final TC1 Configuration: “dataset_name: [+p][+u][−l][+-rottg][−nai][+r][+p][−e][+ompt][−ponse]”
- Appendix A — Final TC1 Configuration: “split: [+promp][−e][−t]”

</details>

### Renamed headings (3)
- “5.3 Hugging Face Access” → “5.4 Hugging Face Access and Gating”
- “5.4 Run Plan” → “5.6 Run Plan”
- “5.5 Reproducibility Notes” → “5.7 Reproducibility Notes”

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
