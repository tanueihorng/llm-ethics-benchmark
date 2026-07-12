"""Generate TC1 sbatch files for the T18/T39 multi-seed sensitivity arm.

Emits, into the chosen ``--out-dir``:

* one *generation* sbatch per model (``<alias>__<suffix>.sbatch``) that loops over
  the seed set, re-running HarmBench under stochastic decoding (T=0.7) into
  ``<sens-root>/seed<N>/``; and
* one *judge* sbatch per seed (``judge_<suffix>_seed<N>.sbatch``) that scores that
  seed's generations with the official HarmBench classifier, so the sensitivity
  metric matches the judge-primary headline.

Two budgets share this generator via flags (128-token defaults preserved for
back-compat):

* 128-token arm (default): ``--out-dir slurm/jobs_tc1_sensitivity`` with the
  default config/root/log-dir/suffix.
* 512-token arm (T31/T39): ``--out-dir slurm/jobs_tc1_sensitivity_512
  --config configs/tc1_sensitivity_512.yaml --sens-root results_sensitivity_512
  --log-dir results_512/slurm_logs_tc1 --suffix sens512``.

T39 (D46) extended the model set from the original three pairs to all five (adds
mistral_7b + phi4_mini). ``MODELS_BY_PAIR`` MUST stay in sync with
``sensitivity_analysis.PAIRS`` (asserted by tests). ``--pairs`` restricts which
pairs are emitted (default: all).

This is a deterministic code-generation step (no GPU, no network); run it on the
Mac and commit the result, then ``git pull`` on TC1 and ``sbatch`` directly
(TC1 policy forbids running Python on the head node).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

WORK_DIR = "/tc1home/FYP/utan001/fyp_quant/repo"

# Grouped by pair so the user can run the load-bearing Qwen 1.7B pair first. T39
# (D46) appends mistral_7b + phi4_mini. Keep in sync with sensitivity_analysis.PAIRS.
MODELS_BY_PAIR = [
    ("qwen_2b", ["qwen_2b_base", "qwen_2b_4bit"]),
    ("qwen_4b", ["qwen_4b_base", "qwen_4b_4bit"]),
    ("llama_3_2_3b", ["llama_3_2_3b_base", "llama_3_2_3b_4bit"]),
    ("mistral_7b", ["mistral_7b_base", "mistral_7b_4bit"]),
    ("phi4_mini", ["phi4_mini_base", "phi4_mini_4bit"]),
]


def _bootstrap(log_dir: str, sens_root: str) -> str:
    return f"""set -euo pipefail

cd {WORK_DIR}
mkdir -p {log_dir} {sens_root}
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"""


def _gen_sbatch(alias: str, seeds: List[int], *, config: str, sens_root: str,
                log_dir: str, suffix: str) -> str:
    seed_list = " ".join(str(s) for s in seeds)
    return f"""#!/bin/bash
#SBATCH --job-name={alias}__{suffix}
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output={WORK_DIR}/{log_dir}/{alias}__{suffix}.out
#SBATCH --error={WORK_DIR}/{log_dir}/{alias}__{suffix}.err

{_bootstrap(log_dir, sens_root)}
# T18 sensitivity: HarmBench under stochastic decoding (T=0.7) across seeds.
# Each seed writes to a separate root so resume logic does not collide.
for SEED in {seed_list}; do
  echo "=== {alias} seed $SEED (T=0.7 HarmBench sensitivity) ==="
  python {WORK_DIR}/run_quant_matrix.py \\
    --config {config} \\
    --model {alias} \\
    --benchmark harmbench \\
    --seed "$SEED" \\
    --output_dir {sens_root}/seed"$SEED" \\
    --device cuda
done
"""


def _judge_sbatch(seed: int, models: List[str], *, sens_root: str, log_dir: str,
                  suffix: str) -> str:
    model_args = " ".join(models)
    return f"""#!/bin/bash
#SBATCH --job-name=judge_{suffix}_seed{seed}
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output={WORK_DIR}/{log_dir}/judge_{suffix}_seed{seed}.out
#SBATCH --error={WORK_DIR}/{log_dir}/judge_{suffix}_seed{seed}.err

{_bootstrap(log_dir, sens_root)}
# Score this seed's stochastic HarmBench generations with the official
# HarmBench classifier (fp16), so the sensitivity ASR matches the judge-primary
# headline metric. Requires the classifier cached on the HEAD node first:
#   python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge
# Edit --models below to match the models you actually generated for this seed.
python {WORK_DIR}/scripts/run_judge_validation.py \\
  --results-dir {sens_root}/seed{seed} \\
  --models {model_args} \\
  --benchmark harmbench \\
  --backend harmbench_cls \\
  --precision fp16 \\
  --batch-size 4 \\
  --device cuda
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate T18/T39 sensitivity sbatch files.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--out-dir", default="slurm/jobs_tc1_sensitivity")
    parser.add_argument("--config", default="configs/tc1_sensitivity.yaml",
                        help="Sensitivity config (128-token default; 512 arm uses "
                             "configs/tc1_sensitivity_512.yaml).")
    parser.add_argument("--sens-root", default="results_sensitivity",
                        help="Per-seed output root (512 arm: results_sensitivity_512).")
    parser.add_argument("--log-dir", default="results/slurm_logs_tc1",
                        help="SLURM log dir (512 arm: results_512/slurm_logs_tc1).")
    parser.add_argument("--results-dir", default="results",
                        help="Analysis tree the aggregation writes to / reads the greedy "
                             "baseline from (512 arm: results_512). Only used to render a "
                             "fully-formed run command in the emitted README.")
    parser.add_argument("--suffix", default="sens",
                        help="Job-name/file suffix (512 arm: sens512).")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Restrict to these pair_ids (default: all in MODELS_BY_PAIR).")
    args = parser.parse_args()

    selected = MODELS_BY_PAIR
    if args.pairs:
        wanted = set(args.pairs)
        selected = [(pid, al) for pid, al in MODELS_BY_PAIR if pid in wanted]
    all_models = [alias for _, aliases in selected for alias in aliases]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for alias in all_models:
        path = out_dir / f"{alias}__{args.suffix}.sbatch"
        path.write_text(_gen_sbatch(alias, args.seeds, config=args.config,
                                    sens_root=args.sens_root, log_dir=args.log_dir,
                                    suffix=args.suffix), encoding="utf-8")
        written.append(str(path))
    for seed in args.seeds:
        path = out_dir / f"judge_{args.suffix}_seed{seed}.sbatch"
        path.write_text(_judge_sbatch(seed, all_models, sens_root=args.sens_root,
                                      log_dir=args.log_dir, suffix=args.suffix),
                        encoding="utf-8")
        written.append(str(path))

    readme = out_dir / "README.md"
    pair_names = ", ".join(pid for pid, _ in selected)
    readme.write_text(
        "# Multi-seed sensitivity-arm sbatch files (generated)\n\n"
        "Generated by `scripts/generate_sensitivity_jobs.py` — do not hand-edit; "
        "regenerate instead. Seeds: " + ", ".join(str(s) for s in args.seeds) + ". "
        "Pairs: " + pair_names + ".\n\n"
        "Run order on TC1 (one GPU job at a time):\n"
        "1. Generation, load-bearing pair first: "
        f"`sbatch {out_dir}/qwen_2b_base__{args.suffix}.sbatch` then "
        f"`qwen_2b_4bit__{args.suffix}.sbatch` (repeat per pair).\n"
        "2. After generation for a seed completes, judge it: "
        f"`sbatch {out_dir}/judge_{args.suffix}_seed1.sbatch` (repeat per seed; edit "
        "`--models` if you only generated some pairs).\n"
        "3. SCP the per-seed results back to the Mac and run the FULLY-FORMED command "
        f"`python scripts/sensitivity_analysis.py --sensitivity-root {args.sens_root} "
        f"--results-dir {args.results_dir}`. Do NOT drop `--results-dir`: its default "
        "is the 128-era `results/` tree, so omitting it here would silently write the "
        "512 seed aggregate into the wrong-budget analysis dir against the 128 greedy "
        "baseline.\n",
        encoding="utf-8",
    )
    written.append(str(readme))

    print(f"Wrote {len(written)} files to {out_dir}/:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
