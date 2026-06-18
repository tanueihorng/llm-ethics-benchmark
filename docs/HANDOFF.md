# Handoff

Last refreshed: 2026-06-18 by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Wrap-up — the experimental study is **complete**. The current agent ran and merged the **T29 INT8 precision point** (D35), the last enhancement. All research/code/report work is done; only the two submission-critical, non-research tasks remain (email the supervisor, submit the report).

## Objective

Send the progress email to Dr. Zhang (T1) and submit the interim report (T15). No experiments, code, or report edits are outstanding.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 actions, §3 decisions **D1–D35**, §4 changelog — INT8 run is **D35** / **T29**; mechanism is **D33** / T28; cross-family is **D32** / T26).
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.

## Current State (durable, verify before acting)

- **`main` carries the complete study, including INT8.** T29 INT8 precision point merged at `48330d4` → report §6.15; T26 (5 pairs / 4 families) → §6.13; T28 mechanism probe → §6.14. All adversarially verified.
- **289 tests pass; `make agent-check` 8/8.** Report = `docs/FYP_Report_2026-06-14.docx` (regenerate with `make report`).
- **Headline (judge-primary, D16):** under four-bit NF4, Qwen-1.7B is the only significant ΔASR (+0.055, modest/borderline). The INT8 precision point (§6.15) shows the quantization effect is **not bit-width-graded**: capability loss is a clean **cliff at four-bit**, while safety is **two-peaked and method-specific** — a second, both-judge + McNemar-significant ASR move appears on **Llama-3B specifically at INT8** and reverts at NF4 (the most judge-robust safety move in the study, fully caveated: ≈8–9 prompts, one pair, LLM.int8-specific).
- ⚠️ **Wording:** "Qwen-1.7B is the *only* significant ΔASR" is correct ONLY for the fp16-vs-NF4 main study; the precision sweep adds Llama @ INT8. This is already scoped in the Abstract, §6.5 and §10 — keep it that way.

## What Changed (T29 INT8 track) — file paths

- INT8 results (redacted sidecars only; raw gitignored): `results/{qwen_2b,qwen_4b,llama_3_2_3b,mistral_7b,phi4_mini}_8bit/`. Manifest 80→120.
- Precision sweep + full-parity INT8 diagnostics: `results/analysis/{precision_sweep,*_int8}.{json,csv}`.
- Backward-compatible `--models` + `--out-suffix` added to `scripts/{rescore_harmbench,judge_agreement,judge_pairwise_agreement,harmbench_category_breakdown,mmlu_subject_breakdown}.py` (defaults byte-identical; regression-guarded so committed old-model artifacts never drift).
- `scripts/precision_sweep_analysis.py`: v2-proxy column now reads `summary.v2.json` (was reading `summary.json` = v1 for the original models — a latent bug, now fixed with a fallback).
- Report: new **§6.15** + Abstract/Ch3/§6.5/§6.14/Ch8/Ch9/§10/Appendix D.
- Tests: `tests/test_int8_scoping.py` (5) + `tests/test_precision_sweep.py` (5→7) → **289**.

## Verification To Run

```bash
git status -sb && git log --oneline -3     # expect main in sync with origin, top = 48330d4 merge
pytest -q                                   # expect 289 passed
make agent-check                            # expect 8/8 pass
```

## Risks / Things To Distrust

- **Do not re-inflate the INT8 Llama finding** — it is real (both judges + McNemar) but caveated: ≈8–9 prompts, one pair, non-monotonic, LLM.int8-specific. The §6.15 framing is deliberately "honest middle."
- **INT8 is a different METHOD (bitsandbytes LLM.int8), not "8-bit NF4."** Never describe it as a lower-bit NF4.
- **The v2 regex is the demoted foil, never a primary result.** All HarmBench ASR in the report is classifier-scored (audited study-wide: all 15 aliases classifier-primary). If you touch the sweep, the v2 column must read `summary.v2.json`, not `summary.json`.
- The mechanism finding (§6.14) is deliberately modest/caveated — do not re-inflate it to "targeted erosion."

## Next Actions (ordered)

1. **T1 — send the supervisor email** (`docs/email_drZhang_2026-06-13.md`, gitignored, local-only). Submission-critical; highest non-research priority.
2. **T15 — submit the interim report** (`docs/FYP_Report_2026-06-14.docx`, now carries §6.15). Submission-critical.
3. **T3 — `MyTCinfo`** on TC1 (storage quota). Optional.
4. Optional research follow-ups (Ch9): replicate the Llama INT8 effect across more models/seeds; trace the §6.14 refusal-margin probe across all three precisions; add a genuinely different quant family (GPTQ/AWQ/GGUF); Arditi activation-direction probe; paired neutral-margin control.

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits — IDs, counts, labels, aggregates, redacted sidecars only. `raw.jsonl`/`summary.json` are immutable TC1 originals (gitignored; hash-pinned in `results/raw_artifact_manifest.sha256`) — never reopen. The INT8 `scores.judge.*` / `summary.judge.*` / `scores.v2` / `summary.v2` sidecars are redacted (IDs + booleans/scalars) and committed. Email drafts (`docs/email_*.md`) gitignored — never commit.
