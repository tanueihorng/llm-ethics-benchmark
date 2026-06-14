# TODO — working backlog (not the source of truth)

> **Source of truth is `docs/PROJECT_LOG.md`.** This file is a short, executable
> working queue so the next session/agent can resume mid-task without re-doing
> research. When an item here is finished, record it in PROJECT_LOG.md (§4
> changelog + §2/§3 as needed) and tick/remove it here.

---

## ACTIVE: extend the study with 2 new model pairs (small-model deployment / cross-family generality)

**Why:** turn the study from 3 models / 2 families into 5 pairs / 4 families
(Qwen, Llama, Microsoft, Mistral), strengthening RQ5 cross-family generality and
deployment relevance. NF4 on-the-fly, same rigorous protocol. Runs on the TC1
V100 (no kernel issues for these two).

### Decided models (feasibility FINAL-VERIFIED via fresh HF-page re-read, 2026-06-14)
> Both GO on the exact stack (transformers 5.5.0 + AutoModelForCausalLM + AutoTokenizer + NF4 + V100). **The only new code is the `attn_implementation` loader field** (Phi needs `eager`); Mistral drops straight in.
| Model | Size | pair_id | Loads | Special handling |
|---|---|---|---|---|
| `microsoft/Phi-4-mini-instruct` | 3.8B | `phi4_mini` | `AutoModelForCausalLM`, std `AutoTokenizer`, has chat_template | **`trust_remote_code: true`** + **`attn_implementation: "eager"`** (V100 has no flash-attn). MIT, ungated. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7.2B | `mistral_7b` | `AutoModelForCausalLM`, std `AutoTokenizer`, chat_template | none — drops straight in. Apache-2.0, ungated. 7B = upper edge of "small" (note in limitations). |

- Both text-only; both `pad_token` = None → already handled by loader (`pad_token = eos_token`).
- **TC1 env is ready:** `transformers 5.5.0`, `bitsandbytes 0.49.2`, `accelerate 1.13.0` — clears every version floor (Phi ≥4.49). **No upgrade needed.**

### Rejected candidates (do NOT re-litigate)
- `google/gemma-3-4b-it` — **multimodal**, needs `Gemma3ForConditionalGeneration`, not `AutoModelForCausalLM`. Blocked.
- `mistralai/Ministral-3-3B-Instruct-2512` — multimodal + non-standard `MistralCommonBackend` tokenizer + bleeding-edge deps + FP8. Blocked (worse than Gemma).
- `google/gemma-3-1b-it` — works (text-only) but gated + only 1B; deprioritised.
- AWQ quantization — **AutoAWQ GEMM kernels require sm_75+; the V100 is sm_70 → cannot run.** GGUF/GPTQ are feasible but are the separate quant-variant axis (large engineering lift).

### Next steps (ordered, concrete)

**LOCAL WORK DONE 2026-06-14 (D30) — all of steps 1–4 + tests + report/docs landed; 246 tests pass.**
1. ✅ **Loader:** `attn_implementation` field added to `ModelEntry` (+fail-loud validator) → threaded through `ModelSpec` + `model_kwargs` (truthiness-guarded) + both pipeline constructors + `build_model_spec`. Tests added (injected-when-set / omitted-when-None).
2. ✅ **Configs:** 4 entries in `configs/tc1.yaml` + `configs/default.yaml`. New `tests/test_quant_config_schema.py` validates 10 models / 5 pairs / 4 families + flags.
3. ✅ **Prefetch:** no edit needed — `scripts/prefetch_tc1.py` reads model_ids from the config generically (auto-caches the 2 new ones).
4. ✅ **sbatch:** 4 matrix + 4 harmbench-smoke sbatch templated byte-consistent from the committed files (did NOT use `make cluster-generate` — the live generator drifted to absolute log paths and would have rewritten the old 6). Plus `slurm/judge_validation_newpairs.sbatch` (scores only the 4 new aliases). Judge/analysis scripts + cross-family function updated. Report (additive "run pending") + CLAUDE/AGENTS/README + PROJECT_LOG (D30) done.

**REMAINING — TC1 (user runs; commands in CLAUDE.md/README TC1 block):**
5. **On TC1 (head node):** `git pull --ff-only`; **Mistral-7B is HF-gated → re-run the HF token login FIRST**; `make prefetch CONFIG=configs/tc1.yaml`; then **5-sample smoke per new model** (`slurm/jobs_tc1_smoke/{mistral_7b_base,phi4_mini_base}__harmbench.sbatch`) — confirm load on transformers 5.x, native chat template (not raw-prompt fallback), Phi loads with eager — BEFORE the full matrix; then `sbatch` the 4 matrix jobs (Mistral first, pair-by-pair). **Mistral 7.2B vs 6h/10G: decide any resource bump at smoke time (user decision).**
6. **Judge (PRIMARY ASR for new pairs):** `sbatch slurm/judge_validation_newpairs.sbatch` (fp16).
7. **After results land (SCP back):** `make analyze` + `scripts/{judge_agreement,judge_pairwise_agreement,harmbench_category_breakdown,mmlu_subject_breakdown,rescore_harmbench}.py` + **gpt-4o second judge** (`run_judge_validation.py --backend api_judge --models mistral_7b_* phi4_mini_*` — user chose full W3 parity). Then fold REAL numbers into `scripts/build_fyp_report.js` (Tables 6.1/6.2/6.3, §6.11 all-pairs cross-family, §6.4.1 ARC) + `make report`; update PROJECT_LOG D-decision (run results) + changelog; `make agent-check`. Verify `git diff results/analysis/` shows old pairs' **numbers** unchanged. **Expected (not a bug):** `quantization_analysis_summary.json`'s `cross_family` section will restructure from the old flat metric-keyed shape to the new `<famA>__vs__<famB>` all-pairs shape (+ a new `overall_sign_consistency` key) — the existing qwen-vs-llama leaf values are byte-identical, only the JSON shape changes (D30 rewrite).

### Watch items / guardrails
- Smoke-test on transformers **5.x** (a major version; neither card explicitly notes 5.x).
- Record `transformers==5.5.0` + `bitsandbytes==0.49.2` in methodology for reproducibility; if the original 6 models were run on 4.x, note it (within-pair deltas stay valid).
- `raw.jsonl`/`summary.json` are immutable; redaction rules apply; PROJECT_LOG discipline per AGENTS.md.

### Ready-to-paste config entries
```yaml
  phi4_mini_base:   {family: phi,    size_b: 3.8, quantized: false, pair_id: phi4_mini, model_id: microsoft/Phi-4-mini-instruct,      trust_remote_code: true,  attn_implementation: eager, dtype: auto, benchmarks: [harmbench, xstest, mmlu]}
  phi4_mini_4bit:   {family: phi,    size_b: 3.8, quantized: true,  pair_id: phi4_mini, model_id: microsoft/Phi-4-mini-instruct,      trust_remote_code: true,  attn_implementation: eager, dtype: auto, benchmarks: [harmbench, xstest, mmlu]}
  mistral_7b_base:  {family: mistral, size_b: 7.2, quantized: false, pair_id: mistral_7b, model_id: mistralai/Mistral-7B-Instruct-v0.3, trust_remote_code: false, dtype: auto, benchmarks: [harmbench, xstest, mmlu]}
  mistral_7b_4bit:  {family: mistral, size_b: 7.2, quantized: true,  pair_id: mistral_7b, model_id: mistralai/Mistral-7B-Instruct-v0.3, trust_remote_code: false, dtype: auto, benchmarks: [harmbench, xstest, mmlu]}
```
(Expand to the block form used by existing entries when pasting.)
