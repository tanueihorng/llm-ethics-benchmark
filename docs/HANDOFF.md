# Handoff

Last refreshed: 2026-06-14 (~21:00 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Executor — the current agent did the feasibility work for a new enhancement track (T26); the next session implements it. The core study is otherwise submission-ready.

## Objective

Two tracks:
1. **Active executor track — T26:** extend the study with 2 new model pairs (Phi-4-mini, Mistral-7B). Feasibility is **FINAL-VERIFIED**; integration is **PENDING**. Full execution detail + ready-to-paste config is in **`todo.md`** (repo root).
2. **Submission-critical (unchanged):** T1 (supervisor email) and T15 (submit the report). The study is complete, judge-validated, and double-robustness-checked — these two are non-compute and remain the priority for the deadline.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 open actions, §3 decisions **D1–D28**, §4 changelog).
- For the T26 executor detail, read **`/Users/tanueihorng/fyp_quant/todo.md`** (the working backlog; tactical buffer, not source of truth).
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.

## Current State (durable)

- **Study complete + judge-validated (D16) + double-robustness-checked (T18/D23, T22/D26).** Judge-primary headline (`results/analysis/judge_agreement.json`): Qwen 1.7B = broad_degradation (ΔASR +0.055; ΔMMLU −0.087), but T18 multi-seed (mean +0.024) and the gpt-4o second judge (+0.045, κ 0.69–0.94) show the effect is **real but modest/borderline**. Qwen 4B = alignment_degradation (directional); Llama = broad_degradation (capability-only).
- **Code-quality hardening done (D22, latent-six):** κ degenerate-case, MMLU parser, taxonomy symmetry (now 6 labels incl. `over_refusal_regression`), loader quant-guard, XSTest schema-drift. Verified to change **no** committed number (empty `git diff results/analysis/`).
- **Productivity harness added (D28):** `todo.md` working-backlog convention + `fyp-todo-capture` skill (distinct from this one-shot handoff). CLAUDE/AGENTS pointer synced.
- **T26 feasibility done (this session):** see the executor track below.
- **223 tests pass; `make agent-check` 8/8.** Report = `docs/FYP_Report_2026-06-14.docx`.

## T26 Executor Track — what to implement (verify each from the repo / HF before running)

**Decided models** (feasibility re-confirmed against the live HF pages 2026-06-14):
- `microsoft/Phi-4-mini-instruct` (3.8B, MIT, ungated) — needs `trust_remote_code: true` **+ a NEW `attn_implementation: eager` loader flag** (V100 is Volta sm_70, no flash-attn).
- `mistralai/Mistral-7B-Instruct-v0.3` (7.2B, Apache-2.0, ungated) — drops straight in, no special handling. 7B = upper edge of "small" (note in limitations).
- Both text-only, std `AutoTokenizer` + chat template, `pad_token=None` already handled by the loader's eos fallback.

**Rejected — do NOT re-litigate:** Gemma-3-4b-it & Ministral-3-3B (multimodal / non-`AutoModelForCausalLM`, non-standard tokenizer); Gemma-3-1b-it (gated, only 1B); AWQ (V100 sm_70 can't run its kernels).

**Env is ready** (checked on TC1): `transformers 5.5.0`, `bitsandbytes 0.49.2`, `accelerate 1.13.0` — clears every floor; **no upgrade needed**.

**Ordered steps (the single code change is step 1):**
1. Add optional `attn_implementation: Optional[str]` to `ModelEntry` (`ethical_benchmark/quant/config_schema.py`) → `ModelSpec` (`ethical_benchmark/models/loader.py`) → set in `model_kwargs` only when present. Add a unit test (set vs unset). **This is the only new code.**
2. Add the 4 config entries (in `todo.md`) to `configs/tc1.yaml` and `configs/default.yaml`.
3. Add the 2 model_ids to `scripts/prefetch_tc1.py` targets.
4. `make cluster-generate CONFIG=configs/tc1.yaml` → new sbatch.
5. On TC1: `git pull --ff-only`; prefetch; **5-sample smoke per new base model** (Phi needs eager + trust_remote_code; confirm load on transformers 5.x) before the full matrix; then `sbatch` the 4 new jobs.
6. After results land: `make analyze`; results-worthy → edit `scripts/build_fyp_report.js` (cross-family tables) + `make report`; PROJECT_LOG D-decision + changelog; `make agent-check`.

## Verification To Run

```bash
git status -sb && git log --oneline -8     # live Git state
pytest -q                                   # expect 223 passed
make agent-check                            # expect 8/8 pass
```

## Risks / Things To Distrust

- **T26 is NOT implemented yet** — the config block + `attn_implementation` field are designed, not coded. Implement + smoke-test before any full matrix run.
- **Smoke-test the 2 new models on transformers 5.x** — neither model card explicitly notes 5.x; only the 2 new models are unverified on it.
- The headline Qwen 1.7B effect is **modest/borderline** — do not over-state +0.055 as large/certain (report already framed honestly).
- `raw.jsonl`/`summary.json` immutable; all judge/v2/sensitivity layers are redacted sidecars.
- Repo is PUBLIC; email drafts (`docs/email_*.md`) gitignored — never commit them.

## Next Actions (ordered)

1. **(Active track) T26** — implement the executor steps above (start with the `attn_implementation` loader field + test), on a branch.
2. **T1 — send the supervisor email** (`docs/email_drZhang_2026-06-13.md`, gitignored). Submission-critical.
3. **T15 — submit `docs/FYP_Report_2026-06-14.docx`.** Submission-critical.
4. Optional/later: T23 (ARC), T24/T25 (audit nits / stale Appendix D), LlamaGuard cross-check.

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits — IDs, counts, labels, aggregates, redacted sidecars only. Preserve `raw.jsonl`/`summary.json` as TC1-original. `results_sensitivity/` stays gitignored. Email drafts gitignored — never commit.
