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
| Code released with dependency specs | public repo; `requirements.txt` (pinned) + `pyproject.toml` | ✅ |
| All reported results + full experimental setup | `results/analysis/*.json`, redacted judge sidecars, report Ch6 | ✅ |
| Hyper-parameters / decoding controls | `generation_config` in every `summary.json` (greedy, `temperature=0.0`, seed 42, `max_new_tokens=128`) | ✅ |
| Evaluation-measure definitions | `docs/evaluation_metrics.md`; ASR / over-refusal / accuracy defined | ✅ |
| Compute infrastructure | `summary.judge.*.json` records GPU + precision (Tesla V100-32GB); `docs/TC1_CLUSTER_RUNBOOK.md` | ✅ |
| Number of runs / variance | greedy is deterministic; a multi-seed (T=0.7) sensitivity arm quantifies generation variance for the load-bearing pair | ✅ |
| Statistical significance | paired bootstrap 95% CIs (2 000 resamples, seed 42) + McNemar exact test; *uncorrected* multiple comparisons disclosed (report §6.5) | ✅ (disclosed) |
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
  is pinned in `results/raw_artifact_manifest.sha256` (120 files) and checked by
  `make agent-check`.
- **Provenance in every record** — `raw.jsonl` carries `pair_id`, `quantized`,
  `quant_method`, `seed`, `generation_config`, `timestamp`.

## 3. Project structure (Wilson "Good Enough Practices")

The repo follows the recommended `README / LICENSE / CITATION / data / doc /
results / src` shape: `README.md`, `CITATION.cff`, `data/`, `docs/`, `results/`,
and the `ethical_benchmark/` package as `src`. Dependencies are explicit
(`requirements.txt`), code is decomposed into plugins/modules, and a 306-test
suite guards behaviour.

## 4. One-command reproduction

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Verify the framework (no GPU needed)
pytest -q                                   # expect all green
make agent-check                            # docs/artifact/redaction gates + tests

# 3. Re-derive the committed analysis from the (redacted) sidecars (no GPU)
python compare_quant_pairs.py --config configs/default.yaml \
  --results_dir results --output_dir results/analysis
python scripts/judge_agreement.py           # judge-vs-regex agreement (κ)
python scripts/precision_sweep_analysis.py  # fp16 → INT8 → NF4 sweep

# 4. Re-run experiments from scratch (GPU; see docs/TC1_CLUSTER_RUNBOOK.md)
make matrix DEVICE=cuda
```

Re-running step 3 reproduces `results/analysis/*` byte-for-byte from the committed
sidecars (the audit, PROJECT_LOG D36, confirms this). Step 4 regenerates the raw
generations and requires the gated model weights (HF licence + login).

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
