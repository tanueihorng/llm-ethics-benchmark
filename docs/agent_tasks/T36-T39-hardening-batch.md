# T36–T39 — Post-T35 hardening batch (execution packet)

**Planned by:** Claude (Fable 5), 2026-07-12. **Executor:** Opus 4.8 (+ the user
for hand-labeling and TC1 submissions). **Decision:** D46 (PROJECT_LOG §3).

Four independent workstreams chosen by the user from the weakness assessment,
run as ONE batch with a SINGLE combined documentation fold-in at the end
(one D42 claim sweep, one commit set), unless a workstream stalls > 1 week —
then fold in what is done and log the split.

| WS | Task | What it hardens | Compute |
|---|---|---|---|
| A | **T36** XSTest human gold set | "scorer-dependent, but who's right?" (Ch8 gap) | none (user hours) |
| B | **T37** LlamaGuard third judge | versioned-API-judge reproducibility caveat | 1 TC1 job |
| C | **T38** strict-parser capability sensitivity | MMLU parser-inflation caveat | none (local) |
| D | **T39** multi-seed completion (Mistral + Phi) | greedy-single-decode caveat, 2 uncovered pairs | ~7 TC1 jobs |

**Global rules (same as T35):**
1. Adversarial code review (multi-agent) on all new/modified code BEFORE any TC1
   submission and BEFORE the labeling sheet is finalized.
2. Immutability: never touch `raw.jsonl`/`summary.json`/existing sidecars; all
   new outputs are additive derived sidecars or analysis-dir aggregates.
3. Every new committed artifact class needs `configs/artifact_policy.yaml` globs
   in BOTH `allowed_derived_artifacts` and `redaction.scan_paths`, plus a
   synthetic-leak self-test proving the new glob fires (T35 lesson).
4. Pre-registered interpretation rules are read off mechanically; outcome
   sections are append-only.
5. Phase C fold-in: new verify-claims checks for every new reported number, each
   perturb-self-tested (perturb → FAIL → restore **by re-writing the original
   string — NEVER `git checkout` a file that carries other work**).
6. PROJECT_LOG row per phase; tick T36–T39 in §2 as each completes.

---

## Phase A — build all infra locally (no runs, no labels)

### WS-A (T36) — XSTest human gold set tooling

Mirror of T30's `scripts/human_label_audit.py` (do NOT modify it — it
hard-couples HarmBench field names; sibling script per the T35 lesson).

1. **Prereg is already locked:** `docs/XSTEST_GOLD_PREREG.md`. Read it first;
   the sample design, blindness protocol, and outcome matrix are binding.
2. New `scripts/xstest_human_label_audit.py`, four modes mirroring the sibling
   (`--make-sheet`, `--make-html`, `--apply-labels`, `--score`;
   `--results-dir` default `results_512`, `--n` 200, `--disagree-frac 0.6`):
   - Data loading: reuse the loader pattern from
     `scripts/xstest_judge_agreement.py` (`_load_v2_refusal` ~L79–95 reads
     `scores.v2.jsonl` → `score_fields.is_refusal`; `_load_judge_refusal` ~L98
     reads `scores.judge.xstest_api.jsonl` → `judge_refusal_strict`, filter
     `status == "ok"`). Raw text comes from `xstest/raw.jsonl`
     (`prompt_id`, `prompt_text`, `response`).
   - Stratified draw: mirror `human_label_audit.py` `make_sheet`/`_draw`
     (deterministic weighted round-robin, no RNG). Disagreement bucket =
     regex `is_refusal` != judge `judge_refusal_strict`. PRIORITY aliases
     (double weight) = phi4_mini_base/4bit + qwen_2b_base/4bit; OTHER = the six
     remaining NF4 aliases; **exclude all 8-bit aliases** (prereg §2).
   - **Blindness exclusion list** (prereg §2): before drawing, exclude these
     items discussed with labels in chat during T35 — grep the T35 packet +
     PROJECT_LOG for quoted prompt_ids and hard-code the list as a module
     constant with a comment; known members: the phi pair items for the
     "what constitutes rape" definition, the mosquito-methods item, the
     telekinesis item, the checkers-deflection item (find their prompt_ids by
     text-matching `prompt_text` in raw.jsonl; both pair members).
   - Sheet: `human_labels/xstest_label_sheet.csv` (gitignored dir — verify
     `.gitignore:76` covers it) with columns
     `model_alias, prompt_id, prompt, response, regex_refusal, judge_label, human_label`
     — scorer columns present for `--score` but the HTML tool renders ONLY
     prompt + response (blind; mirror `human_label_audit.py` L463–467).
   - HTML tool: 3-class buttons (`1_full_compliance` / `2_partial_refusal` /
     `3_full_refusal`) + keyboard 1/2/3, progress counter, download-CSV and
     paste-back JSON export, rubric legend = the frozen judge rubric semantics
     (alternative-only → full refusal; lecture+answer → compliance).
   - `--score`: emits `results_512/analysis/xstest_human_validation.json`
     (aggregates only): n_labeled, human 3-class counts, and for STRICT and
     BROAD mappings κ/precision/recall/over-flag/under-flag for
     regex-vs-human and judge-vs-human, plus 3-class human-vs-judge agreement.
     Kappa: local `_kappa` like the sibling (house pattern). No text in output.
3. Policy: `results_512/analysis/*.json` globs already cover the artifact
   (scout-verified); `human_labels/` already gitignored. Confirm both with the
   redaction self-test anyway.
4. Tests `tests/test_xstest_human_audit.py`: deterministic draw (same input →
   same sheet), disagreement fraction honored, 8-bit + exclusion-list items
   never sampled, blind HTML contains no scorer strings, `--score` math on a
   synthetic sheet (hand-computed κ), redaction (plant text in a field → assert
   absent from JSON), apply-labels merge keyed `alias||prompt_id`.

### WS-B (T37) — LlamaGuard third judge (HarmBench, 512 tree)

The backend EXISTS and is complete (`LlamaGuardJudgeBackend`,
`ethical_benchmark/judges/validation.py:384–467`; CLI
`scripts/run_judge_validation.py --backend llamaguard`; model
`meta-llama/Llama-Guard-3-8B`; sidecars `scores.judge.llamaguard.jsonl` +
`summary.judge.llamaguard.json`). Build only:

1. `slurm/judge_validation_llamaguard_512.sbatch`, copied from
   `slurm/judge_validation_512.sbatch`: `--backend llamaguard`, drop the
   `--model-id` override (class default applies), `--models` = **all 15 aliases**
   (the 512 template lists only 10 — add the five `*_8bit`), new job name +
   log paths under `results_512/slurm_logs_tc1/`. Keep the bootstrap block
   (offline env vars) byte-consistent. 8B fp16 ≈ 16 GB — fits the 32 GB V100;
   keep the 8-bit-fallback-on-OOM comment convention.
2. Agreement: run `scripts/judge_pairwise_agreement.py` TWICE (it is two-judge
   only, scout-verified): `--judge-a harmbench_cls --judge-b llamaguard
   --out-suffix _llamaguard` and `--judge-a api_judge --judge-b llamaguard
   --out-suffix _api_vs_llamaguard`, both `--results-dir results_512`.
   No script changes expected; verify `--judge-b llamaguard` resolves sidecar
   names generically (it keys on the judge name string).
3. Policy: `results_512/*/harmbench/scores.judge.*.jsonl` +
   `summary.judge.*.json` globs already cover llamaguard sidecars
   (scout-verified). Confirm via the redaction self-test.
4. Tests: extend `tests/test_judge_validation.py` with a stub-backend
   llamaguard-name test (sidecar naming + redaction) if not already covered.

**Trap:** `meta-llama/Llama-Guard-3-8B` is HF-GATED. The user must accept the
license on huggingface.co AND have a valid token login on the TC1 head node
BEFORE prefetch. Prefetch command (head node, allowed):
`python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge --judge-model-id meta-llama/Llama-Guard-3-8B`.

### WS-C (T38) — strict-parser capability sensitivity (MMLU + ARC, local)

**Pre-locked decision rule (no post-hoc discretion):** the STRICT parser =
cascade tiers 1–2 only (leading-letter + "answer is X",
`ethical_benchmark/benchmarks/utils.py:246–262`); tier-3 lenient scan and
tier-4 numeric fallback are disabled; an unparsed response scores
`is_answered=false` → incorrect (same convention as the primary scorer's
parse-failure path). Primary numbers do NOT change; this is a sensitivity
sidecar. The reportable question, fixed now: **does qwen_2b's ΔMMLU remain
negative and significant under the strict parser, and what is the strict
magnitude?** (Direction+significance survive → caveat softens to "magnitude
partly parser-inflated, sign robust"; significance lost → the Ch8 caveat
hardens and §6.5.1's MMLU survivor gains an explicit parser-sensitivity flag.
Either way the finding is reported.)

1. New `scripts/rescore_capability_strict.py`, mirroring
   `scripts/rescore_harmbench.py` (immutability contract, `--dry-run`,
   `--results-dir` default `results_512`, `--models` default = 10 NF4 aliases,
   `--benchmarks mmlu arc`):
   - Re-parse each `raw.jsonl` `response` with a strict variant of
     `parse_choice_index` (implement as a new function in
     `ethical_benchmark/benchmarks/utils.py` — e.g.
     `parse_choice_index_strict()` sharing tier-1/2 regexes with the existing
     cascade rather than duplicating them; do NOT alter `parse_choice_index`).
     num_choices: MMLU raw lacks it — infer by counting option lines in
     `prompt_text` (scout note), with ARC using the same inference; assert
     inferred count ∈ {2..6} and equals len-of-options where derivable.
   - Emit per alias/benchmark: `scores.parser_strict.jsonl` (prompt_id +
     recomputed `{predicted_index, is_answered, is_correct, parse_tier}` +
     scorer_version + passthrough metadata, NO text) and
     `summary.parser_strict.json` (strict accuracy, tier-usage counts — this
     becomes the FIRST committed artifact behind the 48.7%/3.3% claim;
     cross-check those two numbers match the prose, they are claim-locked).
   - Aggregate: `results_512/analysis/parser_strict_sensitivity.json` (+ .csv):
     per-alias strict vs primary accuracy, per-pair strict ΔACC with paired
     bootstrap CI (2,000 resamples, seed 42 — match Table 6.1's capability
     convention) + exact McNemar, tier-usage per alias.
2. Policy globs (scout-verified NOT covered — add 8 entries):
   `results_512/*/{mmlu,arc}/scores.parser_strict.jsonl` and
   `summary.parser_strict.json` in BOTH policy blocks + synthetic-leak
   self-test extension.
3. Tests `tests/test_parser_strict.py`: strict parser on the canonical variants
   (leading letter, "answer is", lenient-only case → unanswered, numeric-only →
   unanswered), num_choices inference, sidecar redaction, immutability
   (raw untouched), aggregate math on synthetic records, ΔACC/CI plumbing.

### WS-D (T39) — multi-seed completion: mistral_7b + phi4_mini @512

**Trap (scout-verified): TWO hardcoded lists must change in sync** —
`scripts/generate_sensitivity_jobs.py` `MODELS_BY_PAIR` (~L33) and
`scripts/sensitivity_analysis.py` `PAIRS` (~L31). Changing only one silently
drops the pair. Add a cross-check test that imports both and asserts the pair
sets are identical.

1. Add mistral_7b + phi4_mini to both lists. Fix **T24** while in
   `sensitivity_analysis.py` (PROJECT_LOG §2: `seeds_used` appended only in the
   v2 branch → track per scorer or union; plus the two cosmetic nits) — it is a
   pre-condition for extending this script.
2. `configs/tc1_sensitivity_512.yaml`: verify it contains model entries for
   mistral_7b_base/4bit + phi4_mini_base/4bit (it predates T26 — likely NOT).
   Add them mirroring `configs/tc1.yaml` (Phi: `attn_implementation: eager`,
   `trust_remote_code: false` on both members; Mistral neither), decoding
   `max_new_tokens: 512, temperature: 0.7, top_p: 0.8`, harmbench only.
3. Regenerate `slurm/jobs_tc1_sensitivity_512/` for the two new pairs
   (gen sbatch loops seeds 1–5 → `results_sensitivity_512/seed$SEED`; judge
   sbatch per seed reads the same). Do NOT regenerate/re-run the existing
   qwen/llama files — additive only; diff to confirm existing files unchanged.
4. `.gitignore`: `results_sensitivity_512/` already ignored (raw harmful text);
   only the redacted `results_512/analysis/sensitivity_multiseed.{json,csv}`
   is committed (it will be REGENERATED to include the new pairs — this
   overwrites a committed analysis artifact, which is allowed for analysis-dir
   files but must be flagged in the changelog row; existing qwen/llama numbers
   must be byte-identical after the rerun — assert this).
5. Tests: pair-list sync test (above); generator smoke for the new pairs;
   T24 regression test.

**Interpretation, fixed now:** both pairs' greedy judge ΔASR are null
(Mistral −0.020 n.s., Phi +0.020 n.s.). The reportable outcome is whether the
greedy value sits inside the 5-seed range and how many seeds are individually
significant (expected: ~0). Any seed showing a significant INCREASE is reported
prominently, not smoothed over. §6.6.1 extends from 3 to 5 pairs either way.

### Phase A exit gate
Full `pytest` green (expect ≈ 353 + ~25 new), `make agent-check` green,
`make verify-claims` 67/67 (no claims changed yet), then the **adversarial
multi-agent code review** over all Phase A diffs; fix findings; only then
Phase B. Commit Phase A as its own commit (infra-only, no results) and push —
TC1 needs the sbatch/config files via `git pull`.

---

## Phase B — runs (parallel tracks)

**Track 1 (user, no compute) — T36 labeling:**
`python scripts/xstest_human_label_audit.py --make-sheet` →
`--make-html` → user labels ~200 items blind (~3–5 h; may split across
sittings; the tool preserves progress via export/import) → `--apply-labels` →
`--score`. THEN read `docs/XSTEST_GOLD_PREREG.md` §5 and append the outcome
letter (J/R/T/X) to its §7 mechanically.

**Track 2 (TC1; ONE track — QoS runs 1 GPU job at a time, order B before D):**
Every step below is one command; the user runs them in order.
1. Head node: `git pull --ff-only` in the TC1 repo.
2. Head node: accept the Llama-Guard-3-8B license on huggingface.co (browser)
   and refresh the token login.
3. Head node: `python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge --judge-model-id meta-llama/Llama-Guard-3-8B`
4. `sbatch slurm/judge_validation_llamaguard_512.sbatch`  (T37; if it hits the
   6h wall, split the `--models` list into two sbatch files per the template's
   own comment)
5. T39 gen jobs, pair by pair (each loops seeds 1–5 internally):
   `sbatch slurm/jobs_tc1_sensitivity_512/mistral_7b_base__sens512.sbatch` (+ 4bit),
   wait, then phi4_mini pair. (Exact filenames from the Phase A generation.)
6. T39 judge jobs: `sbatch slurm/jobs_tc1_sensitivity_512/judge_sens512_seed{1..5}.sbatch`
   (re-generated to cover the new pairs' outputs).
7. SCP back to the Mac: the llamaguard sidecars
   (`results_512/*/harmbench/{scores,summary}.judge.llamaguard.*`) and the
   `results_sensitivity_512/seed*/{mistral,phi}*` trees (local-only).

**Track 3 (local Mac, anytime) — T38:**
`python scripts/rescore_capability_strict.py --results-dir results_512` →
verify aggregate → done (results wait for Phase C).

**Post-run analyses (local, no GPU):**
- T37: the two `judge_pairwise_agreement.py` runs (out-suffixes above).
- T39: `python scripts/sensitivity_analysis.py --results-dir results_512 --sensitivity-root results_sensitivity_512`
  → assert existing qwen/llama entries byte-identical, new pairs appended.

## Phase C — ONE combined fold-in (after all four tracks land)

Scope per result, all in one pass over: `build_fyp_report_v5.js` +
`build_fyp_report_humanized.js`, `build_fyp_thesis_v4.js` + humanized, interim
mirrors (condensed treatment), `fyp_submission/report_latex/final_thesis/mythesis.tex`
(+ recompile + rebuild Overleaf zip — REMEMBER: gitignored, invisible to
git status), `docs/RESULTS_CARD.md`, `README.md`, `CLAUDE.md` + `AGENTS.md`
(same-commit sync), `scripts/verify_report_claims.py`, `docs/PROJECT_LOG.md`
(tick T36–T39, D46 outcome, §1 scorer-validity paragraph, changelog rows,
meta bump), `todo.md` prune, Appendix D test count, revision-history rows.

- **T36** → §6.12 Result 6 extension (or "Result 7"): human-grounded κ pair,
  outcome letter per prereg §5; Ch8 XSTest limitation updated per outcome;
  abstract only if outcome J or R (it changes the survivor's status).
- **T37** → §6.12 Result 4 extension: open-weight third judge κ vs both;
  Ch7 "versioned API judge" residual caveat resolved or bounded; Ch9
  future-work item closed.
- **T38** → §6.5 third statistical limitation + Ch8 bullet updated with the
  strict-parser answer; §6.4/§6.5.1 only if the MMLU survivor's significance
  changes; tier-usage numbers (48.7/3.3, 46/75) now cite the committed artifact.
- **T39** → §6.6.1 extended to 5 pairs; Ch8 greedy-decode caveat updated.
- New verify-claims checks: ≥2 per workstream (headline numbers pinned to the
  new artifacts), each perturb-self-tested. Expect ≈ 75+ checks total.
- Full gates: `make verify-claims`, `make agent-check`, full pytest, then one
  commit series (results commit + docs commit is fine) and push.

## What can go wrong (pre-answered)

- **T36 outcome R** (regex human-grounded): the biggest doc change — Result 6
  must be honestly revised, abstract wording too. Budget extra fold-in time.
- **LlamaGuard construct mismatch:** Llama-Guard classifies *content safety*,
  not "attack success" per the HarmBench rubric — κ vs the classifier may be
  mediocre for construct reasons, not error. Report it as an
  instrument-family difference (the packet's framing), don't over-read.
- **Mistral 7B × 5 seeds walltime:** the T26 full matrix ran 1:19 for four
  benchmarks; 5 × harmbench-only fits 6h comfortably. If not, split seeds.
- **TC1 queue:** ~7 jobs single-file; expect several days elapsed. Do not let
  Track 1/3 results sit unfolded past a week — apply the split rule.
