# Agent Dashboard

Generated: 2026-06-09 00:49:45 UTC+8

## Live State

- Git: `## main...origin/main`
- Changed files: 60
- PROJECT_LOG: 2026-06-09 (00:48 UTC+8) by Codex
- Report: 2026-06-06 16:53:55 +08 (79546 bytes)
- Handoff: 2026-06-09 00:49:45 +08
- Immutable manifest: 2026-06-08 17:26:57 +08
- Judge sidecars: 6 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

- [ ] **T1. Send progress email to Dr. Zhang.** Draft exists in conversation history (2026-05-23). Cover: framework complete, TC1 environment provisioned, experiments pending submission. Include the corrected scope (2B + 4B + Llama 3B, on-the-fly quantization).
- [ ] **T3. Run `MyTCinfo` on TC1.** Confirm actual storage quota (assumed ~300 GB in the report; verify and update Table 5.1 if different). Effectively confirmed by the successful 25.5 GB prefetch but should still run `MyTCinfo` to record the exact number.
- [ ] **T15. Submit final report** to Dr. Zhang.
- [ ] **T18.** Add a multi-seed (T=0.7) sensitivity arm for one pair, to provide an independent variance estimate. Future-work item; not required for the interim report.
- [ ] **T19.** Stochastic-decoding sensitivity, multi-method quantization (GPTQ/AWQ/GGUF), 0.5B/7B scale extension - all listed in Chapter 9 of the FYP report as future work; track here when scoped.
- [ ] **T21. Strengthen Results & Discussion (Ch6/Ch7) robustness.** User's stated next priority. Make the judge-vs-v2 disagreement the analytical centrepiece (why the scoring layer changed while raw outputs stayed valid). Concrete, mostly-free additions from existing data: (a) HarmBench harm-category breakdown - `raw.jsonl` already stores `category`, so per-category judge ASR needs no new runs; (b) effect-size framing + clearer bootstrap-CI interpretation; (c) family-dependent judge agreement (kappa) as a finding; (d) deployment implications of the Qwen 1.7B dual degradation; (e) position findings against the quantization-and-safety literature. Report-worthy -> edit `scripts/build_fyp_report.js`, `make report`.
- [ ] **T22. Second independent judge for HarmBench (robustness cross-check).** Add a second judge backend (LlamaGuard or a frontier API classifier with a strict yes/no rubric) to cross-check `cais/HarmBench-Llama-2-13b-cls` - the one remaining construct-validity threat on the HarmBench metric (report §7.4, Ch9). Infrastructure exists: add a backend in `ethical_benchmark/judges/validation.py`, run via `scripts/run_judge_validation.py` into `scores.judge.<name>.*` sidecars, compare with `scripts/judge_agreement.py`. Constraints: never mutate raw/v2/judge sidecars; redacted outputs only; verify the resolved model id + prompt/rubric (do not reuse the HarmBench default); on TC1 run via `sbatch`, not the head node. ("Add more model pairs / scale points" is the lower-priority T19.)

## Analysis Artifacts

- `results/analysis/judge_agreement.json`: 2026-06-06 15:15:57 +08
- `results/analysis/judge_agreement.csv`: 2026-06-06 15:15:57 +08
- `results/analysis/pairwise_deltas.json`: 2026-05-27 23:23:30 +08
- `results/analysis/pair_interpretations.csv`: 2026-05-27 23:23:30 +08
- `results/analysis/quantization_analysis_summary.json`: 2026-05-27 23:23:30 +08

## Harness Checks

- Not run for this dashboard.

## Commands

```bash
python fyp_cli.py agent-status
make agent-check
python scripts/generate_handoff.py
python scripts/generate_tc1_checklist.py
```
