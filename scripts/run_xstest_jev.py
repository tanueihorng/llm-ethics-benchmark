#!/usr/bin/env python3
"""Score saved XSTest responses with pinned TypeSafe Jev.

External egress is explicit. Outputs are additive, redacted sidecars; original
and existing scorer artifacts are never modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.judges.validation import (  # noqa: E402
    JevXSTestRefusalJudgeBackend,
    run_xstest_judge_validation,
)
from scripts.run_xstest_judge import DEFAULT_MODELS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned Jev XSTest refusal sensitivity run.")
    parser.add_argument("--results-dir", default="results_512")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model-id", default="jev-1.13.0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--acknowledge-external-egress", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_external_egress:
        parser.error("Jev sends saved prompt/response text to TypeSafe; pass --acknowledge-external-egress.")
    if not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY is required and must be supplied via the environment.")

    results_dir = Path(args.results_dir).resolve()
    backend = JevXSTestRefusalJudgeBackend(model_id=args.model_id)
    print(f"Jev XSTest sensitivity — model={args.model_id}; outputs are redacted")
    try:
        for alias in args.models:
            summary_path = results_dir / alias / "xstest" / "summary.judge.xstest_jev.json"
            if args.skip_complete and summary_path.exists():
                import json
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing.get("metrics", {}).get("num_samples") == 250:
                    print(f"{alias}: complete, skipped")
                    continue
            summary = run_xstest_judge_validation(
                results_dir,
                alias,
                backend,
                judge_name=backend.name,
                max_samples=args.max_samples,
                extra_metadata={
                    "evaluation_role": "sensitivity_only",
                    "external_egress_acknowledged": True,
                },
            )
            metrics = summary["metrics"]
            print(
                f"{alias}: n={metrics['num_samples']} answered={metrics['answered_count']} "
                f"strict={metrics['over_refusal_rate_strict']:.3f} "
                f"broad={metrics['over_refusal_rate_broad']:.3f} "
                f"errors={metrics['parse_error_count']}"
            )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
