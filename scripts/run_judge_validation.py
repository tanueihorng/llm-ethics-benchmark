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
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.judges.validation import (  # noqa: E402
    HarmBenchJudgeBackend,
    JevJudgeBackend,
    LlamaGuardJudgeBackend,
    OpenAICompatibleJudgeBackend,
    run_judge_validation,
)

DEFAULT_MODELS = [
    "qwen_2b_base",
    "qwen_2b_4bit",
    "qwen_4b_base",
    "qwen_4b_4bit",
    "llama_3_2_3b_base",
    "llama_3_2_3b_4bit",
    "mistral_7b_base",
    "mistral_7b_4bit",
    "phi4_mini_base",
    "phi4_mini_4bit",
]


_DEFAULT_MODEL_ID_BY_BACKEND = {
    "harmbench_cls": "cais/HarmBench-Llama-2-13b-cls",
    "llamaguard": "meta-llama/Llama-Guard-3-8B",
    "api_judge": "gpt-4o",
    "jev": "jev-1.13.0",
}


def _build_backend(args: argparse.Namespace):
    # Resolve the model id per backend: only override the backend default when
    # the user explicitly passed --model-id (otherwise --model-id's own default
    # of the HarmBench classifier would wrongly be applied to the other backends).
    model_id = args.model_id or _DEFAULT_MODEL_ID_BY_BACKEND[args.backend]
    if args.backend == "harmbench_cls":
        return HarmBenchJudgeBackend(
            model_id=model_id,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            revision=getattr(args, "judge_revision", None) or None,
        )
    if args.backend == "llamaguard":
        return LlamaGuardJudgeBackend(
            model_id=model_id,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            revision=getattr(args, "judge_revision", None) or None,
        )
    if args.backend == "api_judge":
        # API judge (T22): runs locally, no GPU. The OpenAI SDK reads the key
        # from the OPENAI_API_KEY environment variable when api_key is None, so
        # the secret never passes through argv or this process's source.
        # NOTE: this backend SENDS the benchmark prompt + model response to an
        # external API (egress) — a deliberate decision (see D25); never the
        # report's primary scorer (that is harmbench_cls, D16).
        return OpenAICompatibleJudgeBackend(
            model=model_id,
            base_url=args.base_url,
            api_key=None,
        )
    if args.backend == "jev":
        # The SDK reads TYPESAFE_API_KEY when api_key is None. Keep secrets out
        # of argv, logs, summaries, and source. This sends behavior + generation
        # text to TypeSafe; only redacted IDs, labels and probabilities persist.
        return JevJudgeBackend(
            model_id=model_id,
            base_url=args.base_url,
            api_key=None,
            threshold=args.jev_threshold,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HarmBench-classifier judge validation.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", default="harmbench",
                        help="HarmBench classifier only supports 'harmbench'.")
    parser.add_argument("--backend", default="harmbench_cls",
                        choices=["harmbench_cls", "llamaguard", "api_judge", "jev"])
    parser.add_argument("--model-id", default=None,
                        help="Model id for the judge backend. Defaults per backend: "
                             "cais/HarmBench-Llama-2-13b-cls (harmbench_cls), "
                             "meta-llama/Llama-Guard-3-8B (llamaguard), "
                             "gpt-4o (api_judge), jev-1.13.0 (jev).")
    parser.add_argument("--base-url", default=None,
                        help="Optional API base URL for api_judge or jev. Defaults to the "
                             "provider API. Keys are read from OPENAI_API_KEY or "
                             "TYPESAFE_API_KEY, never from argv.")
    parser.add_argument("--judge-revision", default=None,
                        help="HF commit SHA/tag pinning the local judge weights + tokenizer "
                             "(harmbench_cls or llamaguard) to an immutable snapshot (audit "
                             "P1-4). Passed to both from_pretrained calls and persisted as "
                             "judge_revision in the summary. Ignored by hosted API backends "
                             "(their version belongs in --model-id). Omit to use the mutable "
                             "default branch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "8bit", "4bit"],
                        help="Judge precision. Default fp16 (full precision — the judge's "
                             "own quantization must not confound the validation). Use 8bit "
                             "if the 13B classifier does not fit in fp16 on a 16 GB GPU; "
                             "4bit is a last resort.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jev-threshold", type=float, default=0.5,
                        help="Binary threshold for Jev P(harmful compliance). The raw "
                             "probability is retained in the redacted sidecar.")
    parser.add_argument("--acknowledge-external-egress", action="store_true",
                        help="Required for Jev: confirms that saved behavior and generation "
                             "text will be sent transiently to TypeSafe.")
    parser.add_argument("--skip-complete", action="store_true",
                        help="Skip an alias when both target sidecars already exist and its "
                             "summary reports full coverage with zero errors. Useful for "
                             "resuming hosted-judge runs without duplicate spend.")
    args = parser.parse_args()

    if args.benchmark != "harmbench" and args.backend == "harmbench_cls":
        print("ERROR: the HarmBench classifier validates the 'harmbench' benchmark only.",
              file=sys.stderr)
        return 2
    if args.backend == "jev" and not args.acknowledge_external_egress:
        parser.error(
            "--backend jev requires --acknowledge-external-egress because saved "
            "behavior and generation text is sent to TypeSafe"
        )

    results_dir = Path(args.results_dir).resolve()
    backend = _build_backend(args)

    print("=" * 72)
    resolved_model_id = getattr(backend, "model_id", None) or getattr(backend, "model", args.model_id)
    print(f"Judge validation — backend={args.backend} model_id={resolved_model_id}")
    print(f"benchmark={args.benchmark}  precision={args.precision}  batch_size={args.batch_size}")
    print("=" * 72)
    print(f"{'model_alias':<22} {'n':>5} {'judge ASR':>10} {'parse_err':>10}")
    print("-" * 72)

    try:
        for alias in args.models:
            run_dir = results_dir / alias / args.benchmark
            scores_path = run_dir / f"scores.judge.{backend.name}.jsonl"
            summary_path = run_dir / f"summary.judge.{backend.name}.json"
            if args.skip_complete and scores_path.exists() and summary_path.exists():
                try:
                    prior = json.loads(summary_path.read_text(encoding="utf-8"))
                    metrics = prior.get("metrics") or {}
                    n = int(metrics.get("num_samples", -1))
                    answered = int(metrics.get("answered_count", -2))
                    errors = int(metrics.get("parse_error_count", -3))
                    skipped = int(metrics.get("skipped_count", -4))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    n, answered, errors, skipped = -1, -2, -3, -4
                if n >= 0 and answered == n and errors == 0 and skipped == 0:
                    print(f"{alias:<22} {n:>5} {'skip':>10} {0:>10}")
                    continue
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
    print("Compare two judges with scripts/judge_pairwise_agreement.py (no GPU);")
    print("or against the v2 scorer with scripts/judge_agreement.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
