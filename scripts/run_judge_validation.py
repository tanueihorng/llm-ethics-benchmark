"""CLI runner for HarmBench-classifier judge validation (TC1 sbatch entry point).

Runs the official HarmBench Llama-2-13b classifier over the saved HarmBench
generations for the requested model aliases and writes redacted judge
sidecars (``scores.judge.harmbench_cls.jsonl`` + ``summary.judge.harmbench_cls.json``).

This is a *derived sensitivity check* — it never mutates raw.jsonl,
summary.json, or the v2 sidecars (see D11/D12/D13 in docs/PROJECT_LOG.md).
No new target-model inference is performed; only the saved generations are
re-scored by the judge model.

Typical TC1 usage (inside an sbatch job, GPU node):
    python scripts/run_judge_validation.py \
        --results-dir results \
        --models qwen_2b_base qwen_2b_4bit qwen_4b_base qwen_4b_4bit \
                 llama_3_2_3b_base llama_3_2_3b_4bit \
        --benchmark harmbench \
        --precision fp16

Console output is redacted: only IDs counts and rates are printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.judges.validation import (  # noqa: E402
    HarmBenchJudgeBackend,
    LlamaGuardJudgeBackend,
    run_judge_validation,
)

DEFAULT_MODELS = [
    "qwen_2b_base",
    "qwen_2b_4bit",
    "qwen_4b_base",
    "qwen_4b_4bit",
    "llama_3_2_3b_base",
    "llama_3_2_3b_4bit",
]


_DEFAULT_MODEL_ID_BY_BACKEND = {
    "harmbench_cls": "cais/HarmBench-Llama-2-13b-cls",
    "llamaguard": "meta-llama/Llama-Guard-3-8B",
}


def _build_backend(args: argparse.Namespace):
    # Resolve the model id per backend: only override the backend default when
    # the user explicitly passed --model-id (otherwise --model-id's own default
    # of the HarmBench classifier would wrongly be applied to LlamaGuard).
    model_id = args.model_id or _DEFAULT_MODEL_ID_BY_BACKEND[args.backend]
    if args.backend == "harmbench_cls":
        return HarmBenchJudgeBackend(
            model_id=model_id,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
    if args.backend == "llamaguard":
        return LlamaGuardJudgeBackend(
            model_id=model_id,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HarmBench-classifier judge validation.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", default="harmbench",
                        help="HarmBench classifier only supports 'harmbench'.")
    parser.add_argument("--backend", default="harmbench_cls",
                        choices=["harmbench_cls", "llamaguard"])
    parser.add_argument("--model-id", default=None,
                        help="HF model id for the judge backend. Defaults per backend: "
                             "cais/HarmBench-Llama-2-13b-cls (harmbench_cls), "
                             "meta-llama/Llama-Guard-3-8B (llamaguard).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "8bit", "4bit"],
                        help="Judge precision. Default fp16 (full precision — the judge's "
                             "own quantization must not confound the validation). Use 8bit "
                             "if the 13B classifier does not fit in fp16 on a 16 GB GPU; "
                             "4bit is a last resort.")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.benchmark != "harmbench" and args.backend == "harmbench_cls":
        print("ERROR: the HarmBench classifier validates the 'harmbench' benchmark only.",
              file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir).resolve()
    backend = _build_backend(args)

    print("=" * 72)
    resolved_model_id = getattr(backend, "model_id", args.model_id)
    print(f"Judge validation — backend={args.backend} model_id={resolved_model_id}")
    print(f"benchmark={args.benchmark}  precision={args.precision}  batch_size={args.batch_size}")
    print("=" * 72)
    print(f"{'model_alias':<22} {'n':>5} {'judge ASR':>10} {'parse_err':>10}")
    print("-" * 72)

    try:
        for alias in args.models:
            summary = run_judge_validation(
                results_dir, alias, args.benchmark, backend, judge_name=backend.name
            )
            m = summary["metrics"]
            asr = m["attack_success_rate"]
            print(f"{alias:<22} {m['num_samples']:>5} "
                  f"{(asr if asr is not None else 0):>10.3f} "
                  f"{m['parse_error_count']:>10}")
    finally:
        backend.close()

    print("-" * 72)
    print("Wrote scores.judge.<name>.jsonl + summary.judge.<name>.json per model.")
    print("Run scripts/judge_agreement.py (no GPU) to compare against the v2 scorer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
