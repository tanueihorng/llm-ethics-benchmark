# TODO — working backlog (not the source of truth)

> **Source of truth is `docs/PROJECT_LOG.md`.** This file is a short, executable
> working queue so the next session/agent can resume mid-task without re-doing
> research. When an item here is finished, record it in PROJECT_LOG.md (§4
> changelog + §2/§3 as needed) and tick/remove it here.

---

## [2026-06-15] ACTIVE: T29 — RUN INT8 precision point on TC1 (groundwork DONE + pushed; run pending)

**Source of truth:** `docs/PROJECT_LOG.md` D34 + §4 row (2026-06-15 21:15). This is the "how to resume the RUN" buffer only.

**Status:** groundwork DONE, committed **`780dc00`**, pushed to branch **`int8-precision-point`** (NOT merged). 282 tests pass; `make agent-check` 8/8. ZERO regression to the evaluated study (main `tc1.yaml` / tests / base-vs-4bit pipeline untouched; existing fp16/NF4 load byte-identical). Only the TC1 run + sweep analysis + report fold-in remain.

**Why:** add INT8 (bitsandbytes LLM.int8 — a SECOND method, NOT "8-bit NF4"; NF4 is inherently 4-bit) as a 3rd precision so the study spans fp16→INT8→NF4 (addresses the single-method / single-bit-width weakness). Open question: is degradation **graded** with bit-width or a **cliff** at 4-bit?

**Decided (don't re-litigate):**
- **Separate `configs/tc1_int8.yaml`** (NOT int8 in the main tc1.yaml) — `compare_quant_pairs._select_pair_members` picks the alphabetically-first quantized alias, so an `_8bit` member would silently hijack the base-vs-4bit analysis. INT8 is a separate precision dimension analysed by `scripts/precision_sweep_analysis.py`.
- **`quant_method` field**: None defaults to nf4 (existing entries byte-identical); int8 → LLM.int8; rejected on a baseline.
- **Baselines in tc1_int8.yaml are NOT re-run** (present only for pair-validity + as the sweep's fp16 reference; their results already exist). Submit ONLY the `*_8bit` aliases.
- **Judge for int8 = fp16 classifier** (same instrument as fp16/NF4), scores ONLY the 5 int8 aliases (`slurm/judge_validation_int8.sbatch`).

**Verification already done (don't repeat):** 282 tests (16 int8 + 5 sweep new); backward-compat + int8/nf4 branch selection verified without GPU/bitsandbytes; agent-check 8/8; int8 matrix sbatch byte-consistent (alias + `--config tc1_int8.yaml` only).

**Next steps — RUN (TC1):**
1. `cd /tc1home/FYP/utan001/fyp_quant/repo && git fetch origin && git checkout int8-precision-point && git pull --ff-only origin int8-precision-point` (expect `780dc00`). No prefetch / HF-login (same model IDs, already cached).
2. **Smoke (GATE):** `sbatch slurm/jobs_tc1_int8_smoke/qwen_2b_8bit__harmbench.sbatch` → confirm `.err` clean, `load_in_8bit` loads on the V100, `raw.jsonl` coherent. STOP if it OOMs/errors.
3. **Matrix (5, pair-by-pair, MaxJobsPU=2):** `sbatch slurm/jobs_tc1_int8/qwen_2b_8bit__matrix.sbatch`, then `qwen_4b_8bit`, `llama_3_2_3b_8bit`, `mistral_7b_8bit`, `phi4_mini_8bit`. Wait COMPLETED (`squeue -u utan001`; `seff <id>`).
4. **Judge (after ALL 5 matrix COMPLETE):** `sbatch slurm/judge_validation_int8.sbatch`.
5. **SCP back (Mac):** `rsync -avz utan001@10.96.189.11:/tc1home/FYP/utan001/fyp_quant/repo/results/{qwen_2b_8bit,qwen_4b_8bit,llama_3_2_3b_8bit,mistral_7b_8bit,phi4_mini_8bit} /Users/tanueihorng/fyp_quant/results/`

**Next steps — POST-RUN (Mac; ping Claude):**
6. `python scripts/precision_sweep_analysis.py` → `results/analysis/precision_sweep.{json,csv}`.
7. (optional) INT8 margin capture: a margin job with `--config configs/tc1_int8.yaml --models <int8 aliases> --precision-tag int8`, to extend the §6.14 mechanism sweep across precisions.
8. **ADVERSARIALLY VERIFY the trend BEFORE writing** (lesson from T28 — don't present before verifying), then fold fp16→INT8→NF4 into the report (`build_fyp_report.js`) + `make report`; PROJECT_LOG run-results D-decision + §4 row; `make agent-check`; merge `int8-precision-point` → main.

**Watch items:** `load_in_8bit` on V100 sm_70 (smoke is the gate); INT8 expected to sit between fp16 and NF4 (verify, don't assume); `summary.json` is gitignored, so the int8 summaries MUST be SCP'd back for the sweep.

## [2026-06-15] ✅ DONE: T28 — refusal-margin mechanism probe (the "why"); merged to main `c67dbe8` (D33, report §6.14)

Honest finding: **boundary instability, NOT targeted erosion** (within-family AUC 0.64; symmetric 50 harmful-ward / 42 safe-ward flips; flip-driving Qwen-1.7B = generic softening → capability-driven, supports the dichotomy). Overclaimed first; the adversarial-verify workflow deflated it → reported caveated. Durable record: PROJECT_LOG D33. Nothing left to do.

## [2026-06-15] ✅ DONE: T26 — Mistral-7B + Phi-4-mini RUN COMPLETE + folded into report (commit `19a3345`, merged to main; durable record = PROJECT_LOG D32)

**Source of truth:** `docs/PROJECT_LOG.md` — D30 + §4 row (2026-06-14 23:45). This entry is only the "how to resume the RUN" buffer.

**Status:** ✅ COMPLETE 2026-06-15. Ran on TC1 (matrix 61121/61122/61123/61125 + HarmBench classifier 61134) + gpt-4o 2nd judge (local). Judge-primary: Mistral ΔASR **−0.040** (n.s.; v2 proxy +0.055 sign-flip, κ 0.11–0.19; gpt-4o concurs −0.030/κ 0.60–0.63), Phi ΔASR **0.000** (robust_preservation; ΔOR −0.028 SIG decrease). No new significant ΔASR (Qwen 1.7B stays the only one); D16's judge-over-proxy finding now spans 4 families. Folded into report (new §6.13 + Tables 6.1–6.3 + Abstract/RQ2/RQ5/§6.11/§6.4.1/§6.12/Ch10); manifest 48→80; 246 tests + `make agent-check` 8/8 + 3-agent adversarial verification (36/36 numeric checks) all green; Phi via native Phi3 (D31). Committed `19a3345`, **merged to main + pushed**. **Durable record: PROJECT_LOG.md D32.** Everything below is retained for reference only — nothing left to do for T26.

**Why:** 3 pairs/2 families → 5 pairs/4 families (add `mistral_7b`, `phi4_mini`) for cross-family generality (RQ5) + small-model deployment. IDENTICAL methodology to the old 3 (NF4, greedy, seed 42, 4 benchmarks incl. ARC, HarmBench classifier as PRIMARY ASR).

**Decided (don't re-litigate):**
- **gpt-4o 2nd judge → YES** on both new pairs (full W3 parity). Runs LOCALLY on the Mac (needs internet/`OPENAI_API_KEY`), NOT on TC1 (offline).
- **Mistral-7B 6h/10G fit → decide at smoke time** (measure, then bump `slurm.time` only if it times out).
- **cross-family → all-pairs matrix** (done; `compute_cross_family_consistency`).
- **v2 refusal proxy → KEEP and run on the new pairs.** It is the flawed-baseline FOIL that proves the judge-validation contribution (§6.12 / Table 6.3 κ), not a metric we trust. User asked whether to delete it (even from the old 3) → decided NO: deleting removes the evidence for the study's headline methodological finding ("the regex over-counts; the judge relocated the significant result"). Per-family κ on the new pairs is itself a finding. Optional later: demote its prominence in tables (presentation only), never delete.
- **attn_implementation validator** → added, fail-loud (`eager`/`sdpa`/`flash_attention_2` only).

**Rejected (don't re-litigate):**
- Gemma-3-4b / Ministral-3-3b — multimodal, not `AutoModelForCausalLM`.
- AWQ — V100 sm_70 can't run AutoAWQ GEMM kernels.
- **Multi-seed sensitivity for the new pairs up front** — it was only ever run on Qwen 1.7B (the headline pair); Qwen 4B + Llama are greedy-only. **RULE:** run multi-seed on a new pair ONLY IF its results make it a headline (significant/borderline ΔASR), mirroring why Qwen 1.7B got it. `tc1_sensitivity.yaml` stays at its current pairs (do not add the new pairs now).

**Verification already done (don't repeat):** configs load → 10 models/5 pairs/4 families + correct per-family flags; loader injects `attn_implementation` only when set (omitted-when-None test); 246 tests + `make agent-check` 8/8; 4 matrix + 4 smoke sbatch are byte-consistent with the committed `qwen_2b_base` templates (alias-only diff); `judge_validation_newpairs.sbatch` targets only the 4 new aliases; existing config entries + NF4/decode/seed/chat-template code byte-unchanged; immutable artifacts not reopened.

**Next steps — RUN (TC1 head node; identical copy-paste guide in CLAUDE.md / README):**
0. PRE-FLIGHT (browser): accept the Mistral-7B-Instruct-v0.3 license on huggingface.co under account `ueihorng` (gated). Phi-4-mini = MIT/ungated.
1. `cd /tc1home/FYP/utan001/fyp_quant/repo && git fetch origin && git checkout t26-add-mistral-phi-pairs && git pull --ff-only origin t26-add-mistral-phi-pairs` (expect `60c0acc`).
2. `huggingface-cli login` (re-login harmless; needed for the gated Mistral download).
3. `make prefetch CONFIG=configs/tc1.yaml` (fetches the 2 new model_ids; judge classifier already cached from job 61047).
4. **Smoke (gate):** `sbatch slurm/jobs_tc1_smoke/mistral_7b_base__harmbench.sbatch` + `…/phi4_mini_base__harmbench.sbatch`. Verify `.err` clean + `summary.json` has metrics + a response looks coherent (`head -c 600 results/<m>/harmbench/raw.jsonl`). STOP if Phi errors on eager/trust_remote_code or anything OOMs.
5. **Matrix Mistral:** `sbatch slurm/jobs_tc1/mistral_7b_base__matrix.sbatch` + `…/mistral_7b_4bit__matrix.sbatch`. Wait both `COMPLETED` (`squeue -u utan001`; `seff <jobid>` — watch the 6h walltime on the 7.2B base).
6. **Matrix Phi:** `sbatch slurm/jobs_tc1/phi4_mini_base__matrix.sbatch` + `…/phi4_mini_4bit__matrix.sbatch`.
7. **Judge (PRIMARY ASR):** after ALL 4 matrix jobs COMPLETE → `sbatch slurm/judge_validation_newpairs.sbatch` (fp16; scores ONLY the 4 new aliases → old-6 sidecars untouched). Must run AFTER the matrix (reads `raw.jsonl`; crashes if absent).

**Next steps — POST-RUN (Mac, after SCP back; ping Claude to drive this):**
8. `rsync` the 4 new `results/{mistral_7b_base,mistral_7b_4bit,phi4_mini_base,phi4_mini_4bit}/` dirs back from TC1.
9. `make analyze` + `python scripts/{judge_agreement,judge_pairwise_agreement,harmbench_category_breakdown,mmlu_subject_breakdown,rescore_harmbench}.py`.
10. gpt-4o 2nd judge (local, needs `OPENAI_API_KEY` from `~/.zshrc`): `python scripts/run_judge_validation.py --backend api_judge --models mistral_7b_base mistral_7b_4bit phi4_mini_base phi4_mini_4bit`, then `python scripts/judge_pairwise_agreement.py`.
11. Fold REAL numbers into `scripts/build_fyp_report.js` (Tables 6.1/6.2/6.3, §6.11 cross-family, §6.4.1 ARC, Abstract, RQ5/Ch10) → `make report`. Add PROJECT_LOG run-results D-decision + §4 row. `python scripts/agent_check.py --write-immutable-manifest` (ADD new raw hashes, never overwrite). `make agent-check`. Then merge branch → main.

**Watch items / guardrails:**
- **Mistral 7.2B vs 6h walltime / 10G mem** — the binding risk. If `TIMEOUT`, bump `slurm.time` (+ regenerate, or hand-edit the 2 mistral sbatch) — decoding/seed/NF4/n unaffected, so fairness holds.
- **Phi-4-mini** eager + trust_remote_code only exercise on the V100 — the smoke is the gate.
- **Mistral HF-gated** → license + login before prefetch (offline compute nodes fail closed).
- **Judge AFTER matrix** (`run_judge_validation` FileNotFoundError on missing `raw.jsonl`).
- **DON'T touch:** old `raw.jsonl`/`summary.json`/`scores.*` sidecars (immutable TC1 originals); `tc1_sensitivity.yaml` (separate D23 study, stays as-is); no raw HarmBench prompt/response text in any doc.
- **Expected non-bug:** post-run `make analyze` restructures `quantization_analysis_summary.json`'s `cross_family` section to the all-pairs shape (`<famA>__vs__<famB>` + `overall_sign_consistency`); the existing qwen-vs-llama leaf numbers are byte-identical.
