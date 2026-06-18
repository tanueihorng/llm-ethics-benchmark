# Handoff

Last refreshed: 2026-06-15 (~21:30 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Executor — the current agent finished and merged the **T26 cross-family run** and the **T28 mechanism probe**, and built the **T29 INT8 groundwork** (run pending). The next session/agent runs INT8 on TC1 and folds it in; the study is otherwise submission-ready.

## Objective

Run the INT8 precision point on TC1 and fold the fp16 → INT8 → NF4 trend into the report; then the submission-critical email (T1) and report submission (T15).

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 actions, §3 decisions **D1–D34**, §4 changelog — INT8 is **D34** / **T29**; mechanism is **D33** / T28; cross-family is **D32** / T26).
- For the INT8 run/resume detail, read **`/Users/tanueihorng/fyp_quant/todo.md`** (tactical buffer, not source of truth).
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.

## Current State (durable, verify before acting)

- **Active branch `int8-precision-point` @ `780dc00` (pushed; NOT merged).** INT8 groundwork only — no INT8 numbers exist yet.
- **`main` carries the evaluated study:** T26 (5 pairs / 4 families) merged at `19a3345` → report §6.13; the T28 mechanism probe merged at `c67dbe8` → report §6.14. Both adversarially verified.
- **Headline (unchanged, judge-primary D16):** Qwen-1.7B is the **only** significant ΔASR (+0.055, modest/borderline); Mistral/Phi add no significant ΔASR; the mechanism probe reads the small-model effect as **capability-driven boundary instability, not targeted erosion** (§6.14, fully caveated).
- **282 tests pass; `make agent-check` 8/8.** Report = `docs/FYP_Report_2026-06-14.docx`.

## What Changed (INT8 track) — file paths

- Backward-compatible `quant_method` field: `ethical_benchmark/quant/config_schema.py` (+validator, +baseline-consistency check) → `ethical_benchmark/models/loader.py` (`ModelSpec`, `_build_bnb_8bit_config`, int8/nf4 branch, `is_loaded_in_8bit` detection, `build_model_spec`) → both pipeline constructors. **`None` ⇒ NF4, byte-identical to before** (verified).
- `configs/tc1_int8.yaml` (5 `*_8bit` + fp16 baselines; baselines NOT re-run). Main `tc1.yaml` untouched.
- `slurm/jobs_tc1_int8/*_8bit__matrix.sbatch` (5), `slurm/jobs_tc1_int8_smoke/*_8bit__harmbench.sbatch` (5), `slurm/judge_validation_int8.sbatch` (scores only the 5 int8 aliases).
- `scripts/precision_sweep_analysis.py` → `results/analysis/precision_sweep.{json,csv}` (graded/cliff/non-monotonic; int8-pending graceful).
- Tests: `tests/test_quant_int8.py` (16) + `tests/test_precision_sweep.py` (5). Report: Ch9 INT8 entry + count 261→282.

## Verification To Run

```bash
git status -sb && git log --oneline -3     # expect int8-precision-point @ 780dc00, in sync with origin
pytest -q                                   # expect 282 passed
make agent-check                            # expect 8/8 pass
```

## Risks / Things To Distrust

- **No INT8 numbers exist yet** — the report's results are fp16/NF4 only; INT8 is "run pending" (Ch9).
- **Smoke each `*_8bit` first (the gate):** confirm `load_in_8bit` actually loads on the V100 (sm_70) — like the Phi remote-code issue, a smoke catches a load failure before a wasted matrix. STOP if it OOMs/errors.
- **Run the int8 judge AFTER the int8 matrix** (`run_judge_validation` needs the `raw.jsonl`).
- **INT8 is a different METHOD (LLM.int8), not "8-bit NF4"** — do not describe it as a lower-bit NF4.
- **Do not merge `int8-precision-point` to main until the INT8 results are run + folded in** (mirrors how T26/T28 were merged only after results).
- The mechanism finding (§6.14) is deliberately modest/caveated — do not re-inflate it to "targeted erosion."

## Next Actions (ordered)

1. **(Active) T29 — run INT8 on TC1.** Per `todo.md`: checkout `int8-precision-point` on TC1 → smoke each `*_8bit` (gate) → `sbatch slurm/jobs_tc1_int8/<pair>_8bit__matrix.sbatch` (pair-by-pair, MaxJobsPU=2) → `sbatch slurm/judge_validation_int8.sbatch` (after the matrix). Then on the Mac: rsync the 5 `*_8bit` results dirs back → `python scripts/precision_sweep_analysis.py` → **adversarially verify the trend before writing** → fold fp16→INT8→NF4 into the report (`build_fyp_report.js`) + `make report` → PROJECT_LOG run-results + §4 row → `make agent-check` → merge branch to main. No new prefetch / HF login (same model IDs, cached).
2. **T1 — send the supervisor email** (`docs/email_drZhang_2026-06-13.md`, gitignored). Submission-critical.
3. **T15 — submit the report.** Submission-critical.
4. **T3 — `MyTCinfo`** on TC1 (storage quota). Optional later: Arditi activation-direction probe / paired neutral-margin control (§6.14 follow-ups).

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits — IDs, counts, labels, aggregates, redacted sidecars only. `raw.jsonl`/`summary.json` are immutable TC1 originals (gitignored; hash-pinned in `results/raw_artifact_manifest.sha256`) — never reopen. The new `scores.margin.*` / `summary.margin.*` and the INT8 `scores.margin.*` are redacted (IDs + scalars) and committable. `tc1_sensitivity.yaml` stays as-is. Email drafts (`docs/email_*.md`) gitignored — never commit.
