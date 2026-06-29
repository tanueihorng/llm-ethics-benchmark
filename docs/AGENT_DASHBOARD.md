# Agent Dashboard

Generated: 2026-06-29 17:20:02 UTC+8

## Live State

- Git: `## main...origin/main [ahead 1]`
- Changed files: 13
- PROJECT_LOG: 2026-06-29 (UTC+8) by Claude
- Report: 2026-06-29 16:14:50 +08 (677737 bytes)
- Handoff: 2026-06-29 17:20:02 +08
- Immutable manifest: 2026-06-18 11:33:39 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Finish the current change, update docs/PROJECT_LOG.md, then run make agent-check.

## Open Actions

- [ ] **T1. Send the staged July follow-up to Dr. Zhang.** The 2026-05-23 re-engagement email **was sent**, and the **June progress update was SENT 2026-06-13** (user's own edited wording; as-sent record at `docs/email_drZhang_june.md`) - completed 3-pair judge-primary headline (Qwen 1.7B dual degradation; Llama capability-only; Qwen 4B intact) + regex->13B-classifier self-correction, with the 5-pair/INT8/2nd-judge/ARC work framed as *upcoming*, and the 6-13 status deck attached. **Open item:** the staged FOLLOW-UP, `docs/email_drZhang_july.md` (drafted in the June email's format/voice), which reports those extensions as *completed*; hold for a July send (deliberate pacing). Its companion July status deck is built: `docs/fyp_status_2026-07.html` (completed 5-pair/INT8/mechanism state, editorial style) - attach THAT, not the June 3-pair deck. (Earlier drafts `email_drZhang_2026-06-09/13.md` + `..._2026-06-26.md` superseded/renamed.)
- [ ] **T3. Run `MyTCinfo` on TC1.** Confirm actual storage quota (assumed ~300 GB in the report; verify and update Table 5.1 if different). Effectively confirmed by the successful 25.5 GB prefetch but should still run `MyTCinfo` to record the exact number.
- [ ] **T15. Submit final report** to Dr. Zhang.
- [ ] **T30. Human-label gold-set validation of the HarmBench primary scorer (the deferred "label later" half of D38).** Groundwork is in place: `scripts/human_label_audit.py --make-sheet` builds a stratified, gitignored local annotation sheet (oversamples judge-vs-regex disagreements + Qwen-1.7B/Mistral; smoke-tested at 120 rows), and `--score` emits redacted classifier-vs-human / regex-vs-human kappa/precision/recall to `results/analysis/human_validation.json`. Protocol + rubric in `docs/HUMAN_LABEL_PROTOCOL.md`. **Next:** annotate the sheet (ideally 2 annotators on a ≥40-prompt overlap slice), run `--score`, then upgrade report §6.12 / thesis §6.1 from the current "scorer choice changes the conclusion" (disagreement) framing to "classifier agrees with humans at kappa=X; regex does not" (correctness), and rebuild the v2 docx. Closes the single deepest construct-validity gap a methods reviewer would demand (no GPU needed). **UPDATE 2026-06-29:** the sampler was upgraded to a representative stratified mix (D-row 2026-06-29 00:47), single-annotator decided, and an in-browser labelling tool was built (`--make-html`/`--apply-labels`; D-row 01:05). **PAUSED 2026-06-29 pending T31:** the user chose to make the 512-token rerun the new primary study, so the gold set should be built from the **512** HarmBench responses (the text those scorers will have judged), not the 128 ones. The 128 sheet + tool stay as a secondary "scorer-validity holds at 128 too" artifact. Resume T30 on `results_512/` after T31 phase B. See T31.
- [ ] **T31. Full 512-token generation-length rerun (D39) - retain the 128-token study, run everything again at HarmBench's 512 reference budget.** Motivation: a reviewer/examiner could ask why the study used `max_new_tokens=128` when HarmBench's reference pipeline uses 512 (generation length materially affects ASR; HarmBench docs). Decision (D39): rerun the **entire** study at 512 (all 4 benchmarks, main fp16+NF4 + INT8 sweep + multi-seed), **retaining all 128-token data** by writing to a parallel `results_512/` root (proven by the `results_sensitivity/` pattern; nothing in `results/` is touched). The 128-vs-512 contrast becomes a built-in robustness result. **Phased plan:** A. main matrix 10 models × 4 benchmarks @512 -> `results_512/`; B. judge re-score HarmBench @512 (HarmBench classifier on TC1 + gpt-4o on Mac), all 10; C. INT8 sweep @512 (needs A's fp16 refs); D. multi-seed @512; E. re-analysis (`make analyze RESULTS_DIR=results_512`) + 128-vs-512 comparison; F. redo T30 on 512 responses; G. regenerate report/thesis @512 + add the budget-robustness section. **DONE so far (2026-06-29, local, no GPU):** `configs/tc1_512.yaml` (max_new_tokens 512, log_dir results_512/, schema-validated); `slurm/jobs_tc1_512/` (10 matrix, `--output_dir results_512 --device cuda`) + `slurm/jobs_tc1_512_smoke/` (40, 5-sample); `slurm/judge_validation_512.sbatch` (all 10 aliases, results_512); `.gitignore` mirrors the results/ policy for results_512. **NEXT:** commit+push -> TC1 `git pull` -> prefetch (already cached) -> smoke one model -> run the 10 matrix jobs pair-by-pair (QoS: 1 GPU job at a time; resume on timeout) -> judge -> SCP back -> analyse. C/D/E/F/G scaffolding still to build. Detail in `todo.md`.
- [ ] **T19.** Stochastic-decoding sensitivity, multi-method quantization (GPTQ/AWQ/GGUF), 0.5B/7B scale extension - all listed in Chapter 9 of the FYP report as future work; track here when scoped.
- [ ] **T24. Minor `sensitivity_analysis.py` provenance hardening (audit nit, 2026-06-14).** The 3-way correctness audit of the T18 work found one low-severity latent issue: `seeds_used` is appended only inside the v2 branch (`scripts/sensitivity_analysis.py` ~L127/L134), so if a future seed had a judge sidecar but no/partial `summary.json`, the reported `seeds_used` would undercount the judge column. Harmless in the current run (v2 + judge co-populated) and changes **no** committed number. Fix when next touching that script (track `seeds_used` per scorer, or union both branches); also two cosmetic nits (docstring wording on the judge-ASR source path; `_f` redefined in the print loop). Aligns with the D22 "harden latent issues before extending" philosophy - do before any full-matrix T18 extension.

## Analysis Artifacts

- `results/analysis/judge_agreement.json`: 2026-06-20 23:00:03 +08
- `results/analysis/judge_agreement.csv`: 2026-06-20 23:00:03 +08
- `results/analysis/pairwise_deltas.json`: 2026-06-15 18:13:09 +08
- `results/analysis/pair_interpretations.csv`: 2026-06-15 18:13:09 +08
- `results/analysis/quantization_analysis_summary.json`: 2026-06-15 18:13:09 +08

## Harness Checks

- Not run for this dashboard.

## Commands

```bash
python fyp_cli.py agent-status
make agent-check
python scripts/generate_handoff.py
python scripts/generate_tc1_checklist.py
```
