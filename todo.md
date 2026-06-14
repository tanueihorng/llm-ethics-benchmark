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
1. **Loader:** add optional `attn_implementation: Optional[str]` field to `ModelEntry`
   (`ethical_benchmark/quant/config_schema.py`) → thread through `ModelSpec`
   (`ethical_benchmark/models/loader.py`) → set in `model_kwargs` only when present.
   Add a unit test (eager set vs unset).
2. **Configs:** add the 4 entries below to `configs/tc1.yaml` and `configs/default.yaml`.
3. **Prefetch:** add the 2 model_ids to `scripts/prefetch_tc1.py` targets.
4. **sbatch:** `make cluster-generate CONFIG=configs/tc1.yaml` → new per-model sbatch.
5. **On TC1 (head node):** `git pull --ff-only`; prefetch the 2 models; then **5-sample smoke per new base model** (confirm load on transformers 5.x; Phi needs eager + trust_remote_code) before the full matrix; then `sbatch` the 4 new jobs.
6. **After results land (SCP back):** `make analyze`; this is results-worthy → edit `scripts/build_fyp_report.js` (Table 6.1/6.2, cross-family §) + `make report`; add PROJECT_LOG.md D-decision + changelog row; run `make agent-check`.

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
