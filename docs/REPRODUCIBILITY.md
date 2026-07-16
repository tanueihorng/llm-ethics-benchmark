# Reproducibility Kit

> Maps this project against recognized standards so a third party can re-run and
> trust the results: the **ML Reproducibility Checklist** (Pineau et al. 2021,
> *Improving Reproducibility in ML Research*, JMLR / NeurIPS 2019 reproducibility
> program), **Ten Simple Rules for Reproducible Computational Research** (Sandve
> et al. 2013, PLOS Comp Biol 10.1371/journal.pcbi.1003285), **Good Enough
> Practices in Scientific Computing** (Wilson et al. 2017, PLOS Comp Biol
> 10.1371/journal.pcbi.1005510), and the **JOSS** software-review criteria
> (Smith et al. 2018, arXiv:1707.02264).

## 1. ML Reproducibility Checklist — coverage

| Checklist item (Pineau et al.) | Where it lives | Status |
|---|---|---|
| Models & algorithms described | `docs/methodology.md`, report Ch3–4 | ✅ |
| Datasets used (sources, splits, sample counts) | `docs/datasets.md`; per-run `summary.json` records dataset/split/`max_samples` | ✅ |
| Code released with dependency specs | public repo; `requirements.txt` (minimum-version floors) + `pyproject.toml` | ✅ |
| All reported results + full experimental setup | `results_512/analysis/*.json` (the primary 512-token study), redacted judge sidecars, report Ch6; `results/analysis/*` retained as the 128-token generation-length comparison | ✅ |
| Hyper-parameters / decoding controls | `generation_config` in every `summary.json` (greedy, `temperature=0.0`, seed 42, `max_new_tokens=512` primary — `128` in the retained comparison tree) | ✅ |
| Evaluation-measure definitions | `docs/evaluation_metrics.md`; ASR / over-refusal / accuracy defined | ✅ |
| Compute infrastructure | `summary.judge.*.json` records GPU + precision (Tesla V100-32GB); `docs/TC1_CLUSTER_RUNBOOK.md` | ✅ |
| Number of runs / variance | greedy is deterministic; a multi-seed (T=0.7) sensitivity arm at the 512-token budget quantifies generation variance for three of the five pairs | ✅ |
| Statistical significance | paired bootstrap 95% CIs (2 000 resamples, seed 42) + McNemar exact test; Benjamini-Hochberg FDR correction over the 20-contrast NF4 family (report §6.5.1) | ✅ |
| Central tendency + variation | deltas with CIs in `pairwise_deltas.json` / report Tables 6.1–6.2 | ✅ |

## 2. Determinism & provenance (Sandve "Ten Simple Rules")

- **Program execution, not manual steps** — every metric is produced by a script
  (`run_quant_*`, `compare_quant_pairs`, the `scripts/*` analysis) from a complete
  `raw.jsonl`; no hand-editing.
- **Fixed seeds** — Python / NumPy / Torch seeded (default 42); `temperature=0.0`.
- **Resume safety** — `--resume` skips processed `prompt_id`s; reported metrics
  come from a complete record set with exactly the configured prompt count.
- **Immutable raw artefacts** — `raw.jsonl` / `summary.json` are TC1-originals,
  never overwritten; all corrections/judging write *derived sidecars*. Integrity
  is pinned in `results/raw_artifact_manifest.sha256` (300 files across the 128, 512, and multi-seed-512 trees) and checked by
  `make agent-check`.
- **Provenance in every record** — `raw.jsonl` carries `pair_id`, `quantized`,
  `model_id`, `model_alias`, `seed`, `generation_config`, and `timestamp` (the
  quantization *method* is fixed per alias in the config and recorded in each
  run's `summary.json`, not repeated on every raw row).

## 3. Project structure (Wilson "Good Enough Practices")

The repo follows the recommended `README / LICENSE / CITATION / data / doc /
results / src` shape: `README.md`, `CITATION.cff`, `data/`, `docs/`, `results/`,
and the `ethical_benchmark/` package as `src`. Dependencies are explicit
(`requirements.txt`), code is decomposed into plugins/modules, and the automated
suite guards behaviour plus artifact-derived document claims. Run
`pytest --collect-only` for the live inventory.

**Environment caveat (retroactive pinning).** `requirements.txt` declares
*minimum-version floors*, and those floors were recorded *after* the TC1 runs
rather than solved into an exact lockfile at run time; the precise resolved package
versions (`transformers`, `torch`, `bitsandbytes`) and the CUDA/driver build are not
machine-recorded in the run artifacts. Reproduction therefore *replays* the committed
analysis deterministically (seed 42) from the redacted sidecars, rather than
re-solving the original environment bit-for-bit. An exact `pip freeze` lockfile is a
documented future-hardening step; it does not affect the committed results, which are
derived from the saved generations, not re-generated.

Going forward, the run pipeline now records the resolved package versions, the
Python/platform string, and the pinned checkpoint revision into every new
`summary.json` (an `env_provenance` field); the INT8 precision configs pin the same
per-model revisions as the NF4 study (`configs/tc1_int8_512.yaml`,
`configs/tc1_int8.yaml`); and the HarmBench judge exposes a `--judge-revision` pin
that is persisted in its summary. New artifacts therefore carry the provenance the
pre-2026-07 historical artifacts lack; the historical gap itself is unchanged, since
those artifacts are immutable.

## 4. One-command reproduction

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Verify the framework (no GPU needed)
pytest -q                                   # expect all green
make agent-check                            # docs/artifact/redaction gates + tests

# 3. Re-derive the committed analysis from the (redacted) sidecars (no GPU).
# NOTE: judge_agreement.py and precision_sweep_analysis.py default to the 128-token
# tree (results/); pass --results-dir results_512 so they read the primary tree.
python compare_quant_pairs.py --config configs/default.yaml \
  --results_dir results_512 --output_dir results_512/analysis
python scripts/judge_agreement.py --results-dir results_512           # judge-vs-regex agreement (κ)
python scripts/precision_sweep_analysis.py --results-dir results_512  # fp16 → INT8 → NF4 sweep

# 4. Re-run experiments from scratch (GPU; see docs/TC1_CLUSTER_RUNBOOK.md)
make matrix DEVICE=cuda
```

Re-running step 3 reproduces the pairwise deltas and the judge-agreement analysis
(`pairwise_deltas.*`, `judge_agreement.*`) byte-for-byte from the committed sidecars,
and the ASR and capability columns of the precision sweep — the cells that carry the
reported claims. One known exception: the committed `precision_sweep.json` predates the
analyzer's switch to preferring the `summary.v2.json` refusal sidecars, so re-running
`precision_sweep_analysis.py` updates only its (uncited, unlocked) regex-derived
over-refusal column to the current v2 values; regenerating that artifact cleanly is
tracked as a separate follow-up. Step 4 regenerates the raw generations and requires the
gated model weights (HF licence + login).

## 5. Privacy / data-release note

Raw HarmBench/XSTest prompts and model responses are **not** redistributed.
Committed artefacts are *redacted sidecars* — prompt IDs, boolean labels, scalars,
and run metadata only — so the study is reproducible without releasing harmful
text. `raw.jsonl`/`summary.json` are gitignored and hash-pinned.

## 6. License (ACTION REQUIRED — your decision)

The repo currently has **no `LICENSE`**, which defaults to *all rights reserved*
and **blocks reuse** — the one thing standing between "others can read it" and
"others can use it." Recommended: a permissive OSI licence (**MIT** or
**Apache-2.0**; Apache-2.0 adds an explicit patent grant) for the *code*.

> ⚠️ **Confirm against NTU's FYP intellectual-property policy first.** University
> FYP IP terms can constrain or co-own student work; clear the licence choice with
> your supervisor/school before adding it. Note also that the *model weights* and
> *datasets* carry their own licences (Llama, Mistral, Phi gating; HarmBench,
> MMLU, ARC, XSTest terms) — your licence covers the framework code only.

Once decided: add `LICENSE`, set `license` in `pyproject.toml` + `CITATION.cff`,
and (optional) archive a release to **Zenodo** for a citable DOI (Wilson et al.;
JOSS practice).

## 7. AI-usage disclosure (JOSS convention)

Per current open-source-software norms (JOSS), disclose AI assistance: parts of
this framework's code, tests, analysis, and documentation were developed with
AI coding assistants under human direction; the author reviewed, validated, and
made the core design and scientific decisions. (Adapt and place this in the thesis
acknowledgements / a repo `AI_DISCLOSURE.md` as your school requires.)
