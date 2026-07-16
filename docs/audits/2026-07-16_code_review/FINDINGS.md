# Full Code Review — Production Code, Execution-Based

**Date:** 2026-07-16 (UTC+8) · **Reviewer:** Claude Fable 5 (orchestrator) + 104-agent workflow (4 reproduction agents, 24 tiered reviewers + 1 gap reviewer, 2 Opus-high skeptics per finding) · **Mode:** report-only; no repo file modified (one incident disclosed in §7).

**Why this review exists.** Prior audits (D22, D36, D40, the July review) were repo-wide but either code-quality-focused without a per-file ledger, or claim-focused rather than code-focused. This is the first systematic, per-file, **execution-based** review of all production code: every file has a manifest row ([COVERAGE_MANIFEST.csv](COVERAGE_MANIFEST.csv)), reviewers worked from a measured `pytest --cov` map of never-executed lines, suspected bugs required an **executed repro** to be reported, and every finding faced two independent skeptics who re-ran the repro.

**Scope:** `ethical_benchmark/` (41 files), live `scripts/*.py` (35), the 7 current JS builder files (as code, not prose), `dashboard/`, root entrypoints, Makefile — **95 files, ~28.5k LOC**. Excluded: retired builders, archives, generated deliverables, docs.

---

## 1. Verdict

**The scientific pipeline is sound and now execution-validated.** No finding changes any reported number, significance verdict, interpretation label, or conclusion. The single P1 is an artifact-staleness issue in a sub-block nothing consumes; the five P2s are latent (wrong under conditions that have not occurred in the committed record). 34 findings survived skeptic verification + 1 from the gap review; 3 were refuted.

The headline result of this review is **positive**: ten of eleven artifact-producing scripts, run fresh against the committed sidecars, regenerate their committed outputs **byte-for-byte** — the first execution-based validation of the repo's core reproducibility claim, covering the pairwise deltas, the BH-FDR family, all three κ agreement files, the precision sweep, both category breakdowns, and the refusal-margin analysis.

## 2. Reproduction sweep (execution, not reading)

| Script | Byte-identical to committed? |
|---|---|
| `compare_quant_pairs.py` (all 6 outputs incl. pairwise_deltas, interpretations) | ✓ |
| `scripts/multiple_comparisons.py` (BH-FDR family — the headline) | ✓ |
| `scripts/judge_agreement.py` / `judge_pairwise_agreement.py` / `xstest_judge_agreement.py` | ✓ ✓ ✓ |
| `scripts/precision_sweep_analysis.py` (incl. INT8 McNemar) | ✓ |
| `scripts/harmbench_category_breakdown.py` / `mmlu_subject_breakdown.py` (default + `_int8`) | ✓ ✓ |
| `scripts/refusal_margin_analysis.py` | ✓ |
| `scripts/number_bible_512.py` (stdout digest) | ✓ |
| `scripts/genlen_512_analysis.py` | **✗ — see P1** (7 of 8 blocks ✓, incl. every ASR block and the 60.3% truncation stats) |
| `scripts/sensitivity_analysis.py` | ✗ additive-only (committed = deliberately held 3-pair version per T39; re-run adds per-seed McNemar block + two empty pair rows; **no existing number changes**) |

## 3. P1 (1)

**F-P1 `results_512/analysis/genlen_robustness.json` — stale sub-block with a flipped significance flag, violating the byte-for-byte reproducibility claim.**
The artifact's `capability_128_vs_512.xstest` block embeds a snapshot of `pairwise_deltas.json` taken 2026-07-12; that input was regenerated 2026-07-14 (T40/D47). The committed block still carries `qwen_2b Δ −0.028, sig=true` (current truth: **−0.024, not significant**) and `phi −0.044` (current: −0.048). **Adjudicated by the orchestrator:** re-derived independently; also proven that **no script, builder, or check consumes `capability_128_vs_512`** — so no reported claim is affected and every §6.16 number the report cites comes from blocks that reproduce byte-identically. Graded P1 because the repo's reproducibility statement promises byte-for-byte replay of committed analysis artifacts, and this one fails it. **Fix:** regenerate (one command). **Root cause worth recording:** the artifact *copies* another artifact's content instead of referencing it — the D50 denormalization disease at the artifact layer.

## 4. P2 (5)

| # | Location | Finding (all with executed repros) | Impact adjudication |
|---|---|---|---|
| F-P2a | `quant/config_schema.py:21` | No model sets `extra="forbid"`; a misspelled field with a default is silently swallowed (`temperatur=0.7` → temperature stays 0.0). The schema misses the exact YAML-typo class it exists to catch. | Latent; current configs verified clean. High-value fix (one line per model). |
| F-P2b | `pipeline/run_quant_benchmark.py:299` | A truncated tail line in `raw.jsonl` (realistic after a SLURM walltime SIGKILL mid-write) makes `--resume` raise `JSONDecodeError` — the documented interrupted-run recovery permanently fails in exactly the interrupted-run scenario. | Latent; no committed raw file is corrupt. |
| F-P2c | `benchmarks/utils.py:197` | The broad refusal patterns match idioms: `is_refusal_response("I can't help but recommend this book")` → True. Mechanistic instance of the *documented* regex-scorer limitation (§6.12). | Already hedged in the report; judge cross-checks bound its effect. Not a new scientific caveat. |
| F-P2d | `scripts/capture_refusal_margin.py:131` | Builds `ModelSpec` without `quant_method`, so an INT8 alias would load as NF4 while its sidecar says `precision_tag=int8`. | **Latent, orchestrator-verified:** no INT8 refusal-margin data exists anywhere; committed §6.14 artifacts are 128-era fp16/NF4 only. |
| F-P2e | `dashboard/app.py:504` | The Run page writes to `exec_dir` but reads the summary back from the browse tree — displays a stale committed summary as if it were the run's result. | GUI-only; evidence trees are write-protected (verified). |

## 5. P3 (29) — grouped

- **Dead code (6):** `judge_agreement._read_summary_metric`, `sensitivity_analysis._judge_asr`, `multiple_comparisons`' unused import, `number_bible_512.A128`, `constants.py` (dead **and** stale: omits `arc`, carries the retired 128 default), the entire `evaluators/` package (isolation from the live path proven via import graph).
- **Checks that can't fail / guard gaps (3):** two `verify_report_claims` snippet lists conditioned on their own presence (L771, L947); `lib/claim_registry.js` validates source hashes but never the payload fingerprint; `harness/agent.py` scanners silently pass on undecodable files and uncaught non-Unicode read errors.
- **Failure-path hygiene (7):** analysis aborts on one malformed JSONL line; `rescore_harmbench --dry-run` creates a directory; `check_runs` crashes when `squeue` is absent; `generate_sensitivity_jobs --pairs` silently drops typos; `fyp_cli` loads the config even for commands that don't use it; `run_xstest_judge` persists billed API calls only per-alias (a crash loses paid work); `judges/validation.parse_yes_no` matches by prefix.
- **Stale text in code (4):** `precision_sweep_analysis` docstring cites the retired Llama INT8 +0.040 as current (its own output contradicts it); architecture diagrams bake in "186 tests"; `make_figures.py` defaults to the 128 tree while committed figures are 512; a Makefile comment mislabels the report target.
- **Consistency nits (9):** per-pair κ point-estimate vs CI denominators differ (`judge_agreement:235`, ±0.005-class, matches the documented artifact nuance); `harmbench_category_breakdown` coerces error rows with `bool()`; report builder column widths don't sum to the table width; Makefile `.PHONY` omission; `sensitivity_multiseed.json` stale schema (deliberate, T39); and the mirrored MC-parser defect in legacy `bias_eval`/`factuality_eval`; plus minor items in the full JSON.

Refuted by skeptics (3): a None-response crash (unreachable — callers guarantee str), a `dict.get` chained-default claim (misread), a BOS-stripping edge (correct as written).

## 6. Coverage

[COVERAGE_MANIFEST.csv](COVERAGE_MANIFEST.csv): **95/95 files manifested, none skipped** — 53 `executed+read`, 18 `read-full`, 17 `read-targeted`, 7 `read-only-no-local-exec` (CUDA/bitsandbytes/TC1/OpenAI paths that cannot run here; line-reviewed with that status stated, not blended into "reviewed"). Baseline measured this session: 388 tests = 53% statement coverage; four science-bearing scripts had 0% test coverage and were instead validated by the §2 reproduction sweep. The manifest's `most_valuable_missing_test` column is the prioritized test backlog (top three: the refusal-regex negative-idiom cases, resume-after-truncation, `extra="forbid"` schema probes).

## 7. Process disclosures

- The first run hit the session usage limit (23/79 agents failed); resumed from cache — all 4 repro agents and all 13 Tier-1 reviewers had already completed before the limit.
- **One sandbox-rule violation:** a reviewer rebuilt `docs/FYP_Report_2026-07-01_v5.docx` in place (1-byte zip-timestamp change). Detected by the pre/post `git status` baseline diff, restored from HEAD, gates re-verified green. No other contamination across 104 agents.
- One reviewer's finding premise ("root `config_schema.py` is imported by the live pipeline") originated from the orchestrator's own faulty grep; the gap reviewer corrected it — the two schemas validate different YAML shapes, no drift risk.

## 8. Checks NOT run

1. GPU/CUDA/bitsandbytes code paths, real HF model loading, TC1 submission — no local GPU; offline policy. Read + mock-level only (statuses marked).
2. OpenAI-backed judge calls — no key (deleted post-T35 by design); parse/retry logic reviewed statically.
3. `dashboard/app.py` interactive flows — Streamlit not driven headless; data layer executed via its tests.
4. The 388 tests' *internal* quality beyond coverage measurement and per-file missing-test assessment.
5. Load/perf/concurrency behavior — single-machine research code; out of scope.
