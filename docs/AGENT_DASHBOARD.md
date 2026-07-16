# Agent Dashboard

Generated: 2026-07-16 14:06:29 UTC+8

## Live State

- Git: `## main...origin/main`
- Changed files: 42
- PROJECT_LOG: 2026-07-16 (UTC+8) by Codex
- Report: 2026-07-16 13:28:30 +08 (696682 bytes)
- Handoff: 2026-07-16 14:05:26 +08
- Immutable manifest: 2026-07-02 09:37:43 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

- [ ] **T36. XSTest human-labelled refusal gold set** (hardening batch WS-A, D46). Pre-reg LOCKED at `docs/XSTEST_GOLD_PREREG.md` (200 items, blind 3-class labeling, J/R/T/X outcome matrix); execution packet `docs/agent_tasks/T36-T39-hardening-batch.md` §WS-A. Build sibling `scripts/xstest_human_label_audit.py` (mirror T30's tool), user labels blind (~3-5 h), score -> `results_512/analysis/xstest_human_validation.json`. Resolves "scorer-dependent but which scorer is right" - the strongest remaining defense upgrade. No compute.
- [ ] **T37. LlamaGuard open-weight third judge, HarmBench @512** (WS-B, D46). **COMPUTE IS DONE - do NOT submit a TC1 job.** Job `61854` (pinned rerun, superseding the unpinned `61844`) COMPLETED: Llama-Guard-3-8B over all 15 aliases, 0 parse errors; the two `judge_pairwise_agreement.py` runs (`_llamaguard`, `_api_vs_llamaguard`) are done and kappa is computed. Sidecars are **deliberately not committed** (scored with `--out-suffix _llamaguard` so they can never overwrite the committed api_judge file); they live on TC1 (`results_512/*/harmbench/*.judge.llamaguard.*` + `~/t37_t39_sidecars.tgz`) with the re-SCP command in the §4 changelog. **Remaining: the Phase-C fold-in only**, deliberately held until T36 labels land so the D42 claim-surface sweep is paid once (todo.md 2026-07-13; plan `docs/agent_tasks/T36-T39-phaseC-draft.md`). Resolves the versioned-API-judge reproducibility caveat.
- [ ] **T38. Strict-parser capability sensitivity (MMLU+ARC)** (WS-C, D46). New `scripts/rescore_capability_strict.py` (mirror rescore_harmbench.py; tiers 1-2 only, unparsed->incorrect, rule pre-locked in the packet); sidecars `scores/summary.parser_strict.*` (policy globs REQUIRED - not covered) + `results_512/analysis/parser_strict_sensitivity.json`; first committed artifact behind the 48.7%/3.3% tier-usage claim. Local only.
- [ ] **T39. Multi-seed completion: mistral_7b + phi4_mini @512, seeds 1-5** (WS-D, D46). **COMPUTE IS DONE - do NOT submit TC1 jobs.** All 5 sensitivity seed jobs COMPLETED (40 sidecars; each of seeds 1-5 now carries 10/10 aliases), SCP'd to the Mac, and the 5-pair aggregate is computed. The committed `results_512/analysis/sensitivity_multiseed.{json,csv}` was **deliberately restored (git checkout) to the 3-pair version** so verify-claims stays consistent with the current 3-pair §6.6.1 during the T36 wait - the artifact and the report must move together, in Phase C. **Remaining: regenerate the 5-pair artifact alongside the §6.6.1 edit in the Phase-C fold-in** (held for T36; plan `docs/agent_tasks/T36-T39-phaseC-draft.md`), keeping existing pairs byte-identical. ⚠️ Still to do in that pass: TWO hardcoded lists must change in sync (`generate_sensitivity_jobs.py MODELS_BY_PAIR` + `sensitivity_analysis.py PAIRS` - add a sync test); fix T24 while in the file; verify/add the two pairs to `configs/tc1_sensitivity_512.yaml` (predates T26). Extends §6.6.1 to 5/5 pairs.
- [ ] **T1. Send the staged July follow-up to Dr. Zhang.** The 2026-05-23 re-engagement email **was sent**, and the **June progress update was SENT 2026-06-13** (user's own edited wording; as-sent record at `docs/email_drZhang_june.md`) - completed 3-pair judge-primary headline (Qwen 1.7B dual degradation; Llama capability-only; Qwen 4B intact) + regex->13B-classifier self-correction, with the 5-pair/INT8/2nd-judge/ARC work framed as *upcoming*, and the 6-13 status deck attached. **Open item:** the staged FOLLOW-UP, `docs/email_drZhang_july.md` (drafted in the June email's format/voice), which reports those extensions as *completed*; hold for a July send (deliberate pacing). **READY (2026-07-02):** the email body is re-based to 512-primary in the user's framing (spotted the 128 truncation issue -> reran everything at 512 -> 512 is HarmBench's official budget) and BOTH July status decks (`docs/fyp_status_2026-07{,_v2}.html`) are refreshed to 512 (banners removed; they pass the stale scan). User: review wording, pick a deck design, send in July. (Earlier drafts `email_drZhang_2026-06-09/13.md` + `..._2026-06-26.md` superseded/renamed.)
- [ ] **T3. Run `MyTCinfo` on TC1.** Confirm actual storage quota (assumed ~300 GB in the report; verify and update Table 5.1 if different). Effectively confirmed by the successful 25.5 GB prefetch but should still run `MyTCinfo` to record the exact number.
- [ ] **T15. Submit final report** to Dr. Zhang.
- [ ] **T34. Upload the 512 LaTeX bundles to Overleaf.** `fyp_submission/report_latex/{final_thesis,interim_report}_overleaf.zip` were rebuilt 2026-07-16 from the registry-aligned `.tex` mirrors and BOTH compile clean under Tectonic/pdfLaTeX (only pre-existing cosmetic overflow/annotation warnings). Overleaf -> **New Project -> Upload Project** -> pick the zip (pdfLaTeX; the `% !TEX program = pdflatex` magic comment sets it). **Manual user action** - there is no Overleaf git remote and `fyp_submission/` is gitignored, so an agent cannot push it. The `.tex`, compiled PDFs, named PDF copies, and bundles on disk are current; only the web upload remains.
- [ ] **T33. Residual pre-submission code hardening from the 2026-07-10 audit (original P1 findings remediated 2026-07-11/14/15; not a current scientific/report blocker).** Remaining: add a future-run manifest/resume fingerprint so a changed config cannot silently resume into old prompt IDs; hard-fail `cluster-submit` on TC1 rather than relying only on the current documentation warning; and strengthen pair validation/significance-note plumbing. The report accurately discloses existing behavior, CI runs the full test suite, the claim gate is active, all six documents are aligned and visually inspected, and the current audit found no P1. Evidence history: `docs/audits/2026-07-10_adversarial_academic_audit/EXECUTIVE_AUDIT.md`; current ledger: `docs/audits/2026-07-15_fable5_full_audit/FINDINGS.md`.
- [ ] **T19.** Stochastic-decoding sensitivity, multi-method quantization (GPTQ/AWQ/GGUF), 0.5B/7B scale extension - all listed in Chapter 9 of the FYP report as future work; track here when scoped.
- [ ] **T24. Minor `sensitivity_analysis.py` provenance hardening (audit nit, 2026-06-14).** The 3-way correctness audit of the T18 work found one low-severity latent issue: `seeds_used` is appended only inside the v2 branch (`scripts/sensitivity_analysis.py` ~L127/L134), so if a future seed had a judge sidecar but no/partial `summary.json`, the reported `seeds_used` would undercount the judge column. Harmless in the current run (v2 + judge co-populated) and changes **no** committed number. Fix when next touching that script (track `seeds_used` per scorer, or union both branches); also two cosmetic nits (docstring wording on the judge-ASR source path; `_f` redefined in the print loop). Aligns with the D22 "harden latent issues before extending" philosophy - do before any full-matrix T18 extension.

## Analysis Artifacts

- `results/analysis/judge_agreement.json`: 2026-07-11 14:58:45 +08
- `results/analysis/judge_agreement.csv`: 2026-07-11 14:58:45 +08
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
