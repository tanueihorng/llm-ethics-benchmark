# TODO — working backlog (not the source of truth)

> **Source of truth is `docs/PROJECT_LOG.md`.** This file is a short, executable
> working queue so the next session/agent can resume mid-task without re-doing
> research. When an item here is finished, record it in PROJECT_LOG.md (§4
> changelog + §2/§3 as needed) and tick/remove it here.

---

## [2026-06-18] ACTIVE: submission wrap-up — the study is COMPLETE; only the two submission tasks remain

**Source of truth:** `docs/PROJECT_LOG.md` §1 status + D35 (INT8) + D36 (audit). All experiments, code, and report work are done and on `main` (`67fb292`+). 295 tests; `make agent-check` 8/8.

**Do next, in order:**
1. **T1 — email Dr. Zhang.** Draft is `docs/email_drZhang_2026-06-13.md` (gitignored, local-only). Highest non-research priority; last supervisor contact was 2026-03-09. Results are in hand (5 pairs / 4 families / 3 precisions, judge-validated).
2. **T15 — submit the interim report.** `docs/FYP_Report_2026-06-14.docx` (regenerate with `make report` if edited). Now carries §6.15 (INT8 precision point) and the family-wise caveat (§6.5).
3. **T3 — `MyTCinfo`** on TC1 (storage quota). Quick, optional.

**Optional disclosure polish (low; already partly covered in §7.5/Ch8 — do only if tightening for the viva):**
- One-line XSTest-ΔOR construct-validity note where ΔOR is reported (the over-refusal regex was never judge-validated; same over-counting risk as HarmBench). Esp. the Phi −0.028 claim.
- "HarmBench classifier adopted as reference without independent human-label validation" disclosure (the gpt-4o 2nd judge κ 0.69–0.94 is the partial cross-check).
- `max_new_tokens=128` ASR-floor note (MMLU-truncation already disclosed in Ch8).

**Optional research follow-ups (Ch9 future work; not blocking submission):**
- Replicate the §6.15 Llama-3B INT8 ASR effect across more models + decode seeds (it rests on ≈8–9 prompts, one pair, non-monotonic — establish if it's a general LLM.int8 phenomenon).
- Trace the §6.14 first-token refusal-margin probe across all three precisions (fp16/INT8/NF4), not just the behavioural metrics.
- Add a genuinely different quant family (GPTQ / AWQ / GGUF) beyond the two bitsandbytes methods.
- Arditi activation-direction probe; paired neutral-margin control with a significance test.

**Reuse/dissemination deliverables — DONE 2026-06-18 (research-grounded):** `pip install -e .` packaging (`pyproject.toml`) + `CITATION.cff` + `docs/QUICKSTART.md` (framework reuse), `docs/paper_outline.md` (workshop paper/poster), `docs/REPRODUCIBILITY.md` + `docs/RESULTS_CARD.md`, `docs/THESIS_OUTLINE.md`. ⚠️ **ACTION (user decision): add a `LICENSE`** — the repo is currently all-rights-reserved, which blocks reuse. Recommend MIT or Apache-2.0 for the code, but **confirm against NTU FYP intellectual-property policy first** (FYP IP may be co-owned). Then set `license` in `pyproject.toml` + `CITATION.cff`, and optionally archive a release to Zenodo for a citable DOI. See `docs/REPRODUCIBILITY.md` §6.

**Housekeeping (optional):** stale local branches already merged to main can be pruned — `t26-add-mistral-phi-pairs`, `mechanism-refusal-margin`, `int8-precision-point`, `harden/latent-six-audit`, `backup/pre-v2-scorer-final` (keep `backup/*` if wanted as a safety net).

## [2026-06-18] ✅ DONE: full-repo scorer-integrity + consistency audit (D36)

6-dimension adversarial audit (the student's worry: did v1/v2 + the classifier-primary switch corrupt the old/NF4 results?). **Verdict: NOTHING invalidates the results** — every primary HarmBench ASR is classifier-scored and independent of the regex; the main pipeline + `pairwise_deltas.json` carry the V2 (not v1) proxy; 120 raw artifacts hash-match; redaction + matched-pair integrity clean; reproduced Qwen-1.7B +0.055 and Llama INT8 +0.040 byte-for-byte. Applied the non-invalidating fixes (§6.5 family-wise caveat, stale §6.1.1 opening, dir-tree 22→23, κ 0.69→0.68, sensitivity `_v2_asr`→v2 + loader raise guard, +6 tests → 295, PROJECT_LOG §1). Durable record: PROJECT_LOG D36. Nothing left to do.

## [2026-06-18] ✅ DONE: T29 — INT8 precision point RUN COMPLETE + merged to main `48330d4` (D35, report §6.15)

Ran 5 `*_8bit` matrix + INT8 classifier judge on TC1 + gpt-4o 2nd judge (0 parse errors). **Finding (NOT bit-width-graded):** capability = clean cliff at 4-bit; safety = two-peaked/method-specific — Qwen-1.7B @ NF4 (+0.055, judge-dependent) and **Llama-3B @ INT8 (+0.040, both-judge + McNemar sig, non-monotonic** — most judge-robust move, caveated). Full-parity INT8 diagnostics (scoped `*_int8`, zero drift); fixed a latent v1/v2 sweep-column bug. Durable record: PROJECT_LOG D35. Nothing left to do.

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
