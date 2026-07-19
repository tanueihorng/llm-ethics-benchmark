# Agent Dashboard

Generated: 2026-07-19 13:02:33 UTC+8

## Live State

- Git: `## main...origin/main`
- Changed files: 66
- PROJECT_LOG: 2026-07-19 (UTC+8) by Claude
- Report: 2026-07-19 12:50:46 +08 (703036 bytes)
- Handoff: 2026-07-19 13:02:33 +08
- Immutable manifest: 2026-07-19 11:46:32 +08
- Judge sidecars: 15 score files

## Suggested Next Action

Run make agent-check, review the diff, then commit or hand off the current change.

## Open Actions

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
