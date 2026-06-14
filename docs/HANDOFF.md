# Handoff

Last refreshed: 2026-06-15 (~00:30 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Executor — the current agent completed the **T26 local integration** (add Mistral-7B + Phi-4-mini pairs); the next session/agent supports the **TC1 run** and the **post-run local write-up**. The core study is otherwise submission-ready.

## Objective

Run the 2 new pairs on TC1 and fold the real numbers into the report:
1. **Active executor track — T26 RUN:** the integration is DONE, committed, and pushed on a branch; only the 4 new TC1 jobs + the new-pairs judge run remain, then the post-run local analysis/report. Full ordered run guide is in **`todo.md`** (repo root) and **`CLAUDE.md`** (TC1 block).
2. **Submission-critical (unchanged):** T1 (supervisor email, `docs/email_drZhang_2026-06-13.md`, gitignored) and T15 (submit the report). Non-compute; still the deadline priority.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 actions, §3 decisions **D1–D30**, §4 changelog — T26 is **D30** + the 2026-06-14 23:45 row).
- For the T26 run/resume detail, read **`/Users/tanueihorng/fyp_quant/todo.md`** (tactical buffer, not source of truth).
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.

## Current State (durable)

- **T26 local integration DONE + committed `60c0acc`, pushed to branch `t26-add-mistral-phi-pairs` (NOT merged to main).** Adds `mistral_7b` + `phi4_mini` → 5 pairs / 4 families, on the **identical** methodology as the old 3 (NF4, greedy, seed 42, 4 benchmarks incl. ARC, HarmBench classifier as PRIMARY ASR). The **existing 3-pair results & setup are byte-unchanged** (additive config; NF4/decode/seed/chat-template code untouched; immutable artifacts not reopened).
- **246 tests pass; `make agent-check` 8/8 green.** An 8-auditor repo audit + a 5-reviewer adversarial diff review (verdict: SHIP) backed the change; the one latent bug they found (cross-family `_sign(None)` collision) is fixed + tested.
- **Core study (unchanged):** judge-primary (D16) + double-robustness-checked (T18/D23 multi-seed, T22/D26 gpt-4o). Qwen 1.7B = broad_degradation but **modest/borderline**; Qwen 4B directional; Llama capability-only. Report = `docs/FYP_Report_2026-06-14.docx`.

## What Changed (this track) — file paths

- New code (one field): `attn_implementation` in `ethical_benchmark/quant/config_schema.py` (+ fail-loud validator) → `ethical_benchmark/models/loader.py` (`ModelSpec` + truthiness-guarded `model_kwargs` injection) → both pipeline constructors (`run_quant_matrix.py`, `run_quant_benchmark.py`) → `build_model_spec`.
- Config: 4 entries in `configs/{tc1,default}.yaml` (Phi: `trust_remote_code`+`eager` on both members; Mistral: neither).
- Eval parity: the 6 judge/analysis scripts → 10 aliases/5 pairs; new `slurm/judge_validation_newpairs.sbatch` (only new aliases); `compute_cross_family_consistency` → all-pairs matrix; 4 matrix + 4 smoke sbatch templated byte-consistent.
- Tests: new `tests/test_quant_config_schema.py` + loader attn tests + cross-family None-guard test.
- Report/docs: additive "run pending" (Appendix A config, Appendix C field, Ch9, count 223→246) + CLAUDE/AGENTS/README + PROJECT_LOG **D30**.

## Verification To Run

```bash
git status -sb && git log --oneline -3      # expect branch t26-add-mistral-phi-pairs @ 60c0acc, in sync with origin
pytest -q                                    # expect 246 passed
make agent-check                             # expect 8/8 pass
```

## Risks / Things To Distrust

- **The TC1 RUN is NOT done** — only the local integration. Nothing about the 2 new pairs is empirically validated yet; the report's Ch6 results are still the original 3 pairs (correctly marked "run pending").
- **Mistral-7B (7.2B)** is the largest model yet — watch the **6h walltime / 10G** envelope; if `TIMEOUT`, bump `slurm.time` (fairness unaffected).
- **Phi-4-mini** `eager` + `trust_remote_code` only exercise on the V100 (sm_70) — the 5-sample smoke is the gate; STOP if it errors.
- **Run the judge AFTER the matrix** (`run_judge_validation` crashes on a missing `raw.jsonl`).
- **v2 refusal proxy = KEEP** (decided): it is the flawed-baseline foil that proves the judge contribution (§6.12), not a trusted metric — do not delete it from old or new pairs.
- The old Qwen 1.7B effect is **modest/borderline** — don't overstate +0.055.

## Next Actions (ordered)

1. **(Active track) T26 RUN** — execute the TC1 steps in `todo.md` / `CLAUDE.md`: accept Mistral HF license → checkout the branch on TC1 → `huggingface-cli login` → `make prefetch` → smoke the 2 base models → matrix (Mistral first, pair-by-pair) → `judge_validation_newpairs.sbatch`. Then on the Mac: rsync results back → `make analyze` + breakdown scripts + gpt-4o 2nd judge → fold real numbers into the report (`build_fyp_report.js`) + `make report` → PROJECT_LOG run-results decision + §4 row → `--write-immutable-manifest` → `make agent-check` → merge branch to main.
2. **T1 — send the supervisor email** (`docs/email_drZhang_2026-06-13.md`, gitignored). Submission-critical.
3. **T15 — submit the report.** Submission-critical.
4. Optional/later: T24 (`sensitivity_analysis.py` provenance nit), LlamaGuard cross-check; multi-seed on a new pair **only if** it becomes a headline.

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits — IDs, counts, labels, aggregates, redacted sidecars only. `raw.jsonl`/`summary.json`/`scores.*` sidecars for the old pairs are immutable TC1 originals — never reopen. `tc1_sensitivity.yaml` stays as-is (separate study). `results_sensitivity/` + email drafts (`docs/email_*.md`) gitignored — never commit. Branch `t26-add-mistral-phi-pairs` is not merged to main yet.
