# Agent Dashboard

Generated: 2026-07-02 11:50:20 UTC+8

## Live State

- Git: `## main...origin/main [ahead 2]`
- Changed files: 342
- PROJECT_LOG: 2026-07-02 (UTC+8) by Claude
- Report: 2026-07-02 11:50:20 +08 (679098 bytes)
- Handoff: 2026-07-02 11:40:47 +08
- Immutable manifest: 2026-07-02 09:37:43 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

- [ ] **T1. Send the staged July follow-up to Dr. Zhang.** The 2026-05-23 re-engagement email **was sent**, and the **June progress update was SENT 2026-06-13** (user's own edited wording; as-sent record at `docs/email_drZhang_june.md`) - completed 3-pair judge-primary headline (Qwen 1.7B dual degradation; Llama capability-only; Qwen 4B intact) + regex->13B-classifier self-correction, with the 5-pair/INT8/2nd-judge/ARC work framed as *upcoming*, and the 6-13 status deck attached. **Open item:** the staged FOLLOW-UP, `docs/email_drZhang_july.md` (drafted in the June email's format/voice), which reports those extensions as *completed*; hold for a July send (deliberate pacing). Its companion July status deck exists (`docs/fyp_status_2026-07.html`) **but is 128-era and banner-marked as a stale snapshot (D41/D42) - DO NOT ATTACH until it is refreshed to the 512-primary numbers (`results_512/analysis` / `scripts/number_bible_512.py`), and re-base the email body numbers at the same time.** (Earlier drafts `email_drZhang_2026-06-09/13.md` + `..._2026-06-26.md` superseded/renamed.)
- [ ] **T3. Run `MyTCinfo` on TC1.** Confirm actual storage quota (assumed ~300 GB in the report; verify and update Table 5.1 if different). Effectively confirmed by the successful 25.5 GB prefetch but should still run `MyTCinfo` to record the exact number.
- [ ] **T15. Submit final report** to Dr. Zhang.
- [ ] **T30. Human-label gold-set validation of the HarmBench primary scorer (the deferred "label later" half of D38).** Groundwork is in place: `scripts/human_label_audit.py --make-sheet` builds a stratified, gitignored local annotation sheet (oversamples judge-vs-regex disagreements + Qwen-1.7B/Mistral; smoke-tested at 120 rows), and `--score` emits redacted classifier-vs-human / regex-vs-human kappa/precision/recall to `results/analysis/human_validation.json`. Protocol + rubric in `docs/HUMAN_LABEL_PROTOCOL.md`. **Next:** annotate the sheet (ideally 2 annotators on a ≥40-prompt overlap slice), run `--score`, then upgrade report §6.12 / thesis §6.1 from the current "scorer choice changes the conclusion" (disagreement) framing to "classifier agrees with humans at kappa=X; regex does not" (correctness), and rebuild the v2 docx. Closes the single deepest construct-validity gap a methods reviewer would demand (no GPU needed). **UPDATE 2026-06-29:** the sampler was upgraded to a representative stratified mix (D-row 2026-06-29 00:47), single-annotator decided, and an in-browser labelling tool was built (`--make-html`/`--apply-labels`; D-row 01:05). **PAUSED 2026-06-29 pending T31:** the user chose to make the 512-token rerun the new primary study, so the gold set should be built from the **512** HarmBench responses (the text those scorers will have judged), not the 128 ones. The 128 sheet + tool stay as a secondary "scorer-validity holds at 128 too" artifact. Resume T30 on `results_512/` after T31 phase B. See T31.
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
