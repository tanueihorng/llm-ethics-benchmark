# Quickstart & Reuse Guide

> Run your own safety–capability quantization study in three steps, or extend the
> framework to new models, benchmarks, and quantization methods. Structured after
> research-software documentation conventions (Wilson et al. 2017, *Good Enough
> Practices in Scientific Computing*, PLOS Comp Biol; JOSS review criteria).

## Statement of need

Studies of how quantization affects LLM *safety* are easy to get wrong: a baseline
and a quantized model can differ for reasons other than quantization (different
checkpoints, decoding, or scoring), and a brittle refusal-regex can over-count
"attack success." This framework removes those confounds with a **matched-pair
design** (both members load from identical weights; quantization is applied on the
fly), **deterministic decoding**, and a **judge-validated** scorer (the official
HarmBench classifier as primary, a second judge as cross-check). It is built to be
*reused*: add a model in YAML, add a benchmark via a 4-method plugin contract, and
the loader, runner, SLURM generation, and analysis pick it up.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # editable install (uses the new pyproject.toml)
pytest -q                 # optional: confirm the install (expect all green)
```

## Three-step run (your own study)

**1. Configure** — point `configs/default.yaml` at your model pair(s):

```yaml
benchmarks:
  harmbench: { dataset_name: walledai/HarmBench, split: train, max_samples: 200 }
models:
  - alias: my_model_base                # fp16 baseline
    model_id: org/my-model
    family: myfamily
    size_b: 3.0
    quantized: false
    pair_id: my_model
    benchmarks: [harmbench, xstest, mmlu, arc]
  - alias: my_model_4bit                # NF4 4-bit member of the SAME weights
    model_id: org/my-model
    quantized: true                     # quant_method defaults to nf4
    pair_id: my_model
    benchmarks: [harmbench, xstest, mmlu, arc]
```

For an **INT8 precision point**, add a member with `quant_method: int8` in a
*separate* config (`configs/tc1_int8.yaml` is the template) so it does not enter
the base-vs-4bit pairwise pipeline — see "Add a precision" below.

**2. Run** the matched pair across all its benchmarks:

```bash
python run_quant_matrix.py --config configs/default.yaml --output_dir results
# or: make matrix DEVICE=cuda
```

**3. Analyse** — pairwise deltas + capability-anchored interpretation labels:

```bash
python compare_quant_pairs.py --config configs/default.yaml \
  --results_dir results --output_dir results/analysis
# or: make analyze
```

Outputs land in `results/<alias>/<benchmark>/` (per-prompt `raw.jsonl` +
`summary.json`) and `results/analysis/` (`pairwise_deltas.{json,csv}`,
`pair_interpretations.csv`). HarmBench Attack Success Rate from `summary.json` is
the *secondary* regex proxy; the **primary** classifier ASR comes from the judge
step (next).

## Judge-validated HarmBench ASR (recommended)

The refusal regex over-counts ASR. For the primary metric, re-score the saved
generations with the official HarmBench classifier (GPU; or any HF classifier):

```bash
python scripts/run_judge_validation.py --results-dir results \
  --models my_model_base my_model_4bit --backend harmbench_cls --precision fp16
python scripts/judge_agreement.py --models my_model_base my_model_4bit --out-suffix _mine
```

Optionally cross-check with a second judge (`--backend api_judge`, needs
`OPENAI_API_KEY`) and `scripts/judge_pairwise_agreement.py`.

## Extending the framework

| To add… | Do this | Touch |
|---|---|---|
| **A model** | add a model entry in YAML (`family`, `size_b`, `quantized`, `pair_id`, `model_id`, `quant_method`, `benchmarks`) | `configs/*.yaml` |
| **A benchmark** | implement `load_items`, `build_prompt`, `score_response`, `aggregate` and register one line | `ethical_benchmark/benchmarks/{your_plugin}.py` + `registry.py` |
| **A precision/method** | set `quant_method: int8` (or extend the loader's method branch) in a *separate* config; analyse with `scripts/precision_sweep_analysis.py` | `configs/*_int8.yaml`, `ethical_benchmark/models/loader.py` |
| **A pairwise metric** | add extraction + delta computation + export fields | `ethical_benchmark/analysis/compare_quant_pairs.py` |
| **Cluster jobs** | regenerate per-model sbatch from config | `make cluster-generate` |

The diagnostic scripts (`judge_agreement`, `judge_pairwise_agreement`,
`harmbench_category_breakdown`, `mmlu_subject_breakdown`, `rescore_harmbench`)
accept `--models` and `--out-suffix`, so you can scope them to *your* aliases
without editing the source or touching anyone else's committed artifacts.

See [`docs/extensibility.md`](extensibility.md) for the contract details,
[`docs/methodology.md`](methodology.md) for the scientific design, and
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the reproducibility checklist.

## What's deliberately *not* automatic

Honesty for reusers (per JOSS documentation expectations):

- **Gated weights** (Llama, Mistral, Phi) need a Hugging Face licence acceptance +
  `huggingface-cli login` before download.
- **The interpretation labels** are tuned to the matched-pair quantization design;
  a different study design should review `classify_pair_change` thresholds.
- **The HarmBench classifier is treated as the reference scorer** without
  independent human-label validation (a known limitation; see methodology).
