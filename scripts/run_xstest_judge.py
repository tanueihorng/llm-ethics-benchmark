"""CLI runner for the XSTest refusal-judge sensitivity check (T35 / D45).

Labels the saved XSTest generations for the requested model aliases with an
independent API judge using the XSTest paper's 3-class refusal taxonomy
(full compliance / partial refusal / full refusal), and writes redacted judge
sidecars (``scores.judge.xstest_api.jsonl`` + ``summary.judge.xstest_api.json``).

This is a *derived sensitivity check* — the deterministic v2 refusal regex
remains the study's PRIMARY XSTest over-refusal scorer and the BH-FDR family is
untouched (see docs/XSTEST_JUDGE_PREREG.md and D45 in docs/PROJECT_LOG.md). No
new target-model inference is performed; only the saved generations are re-scored.
Like the HarmBench ``api_judge`` this SENDS the (benign) prompt + model response
to an external API (egress); the OpenAI key is read from ``OPENAI_API_KEY`` and
never passes through argv or this process's source.

Typical usage (local Mac, no GPU):
    # Pilot (gate before the full run):
    python scripts/run_xstest_judge.py --results-dir results_512 \
        --models phi4_mini_base phi4_mini_4bit --max-samples 20
    # Full run (15 aliases x 250 benign prompts):
    python scripts/run_xstest_judge.py --results-dir results_512

Console output is redacted: only IDs, counts, and rates are printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.judges.validation import (  # noqa: E402
    XSTestRefusalJudgeBackend,
    run_xstest_judge_validation,
)

# All fifteen aliases that have XSTest generations in results_512/ (base / 4bit
# NF4 / 8bit INT8 for each of the five pairs). The INT8 members are scored and
# reported descriptively — they add no new significance family (D45 pre-reg).
DEFAULT_MODELS = [
    "qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit",
    "qwen_4b_base", "qwen_4b_4bit", "qwen_4b_8bit",
    "llama_3_2_3b_base", "llama_3_2_3b_4bit", "llama_3_2_3b_8bit",
    "mistral_7b_base", "mistral_7b_4bit", "mistral_7b_8bit",
    "phi4_mini_base", "phi4_mini_4bit", "phi4_mini_8bit",
]

# Pinned dated snapshot for reproducibility (recorded in every summary as
# judge_model_id). If the account lacks access to this snapshot, pass
# --model-id gpt-4o; the run still records exactly which id produced the labels.
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"


def main() -> int:
    parser = argparse.ArgumentParser(description="XSTest refusal-judge sensitivity check (T35).")
    parser.add_argument("--results-dir", default="results_512",
                        help="Results root holding the XSTest generations (default: results_512).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model-id", default=DEFAULT_JUDGE_MODEL,
                        help="OpenAI model name for the judge (default: a pinned dated gpt-4o "
                             "snapshot for reproducibility). Recorded as judge_model_id in the "
                             "summary. Pass gpt-4o if the dated snapshot is unavailable.")
    parser.add_argument("--base-url", default=None,
                        help="Optional OpenAI-compatible base URL. Defaults to the OpenAI API. "
                             "The API key is read from OPENAI_API_KEY, never from argv.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Score only the first N records per alias (pilot use). Omit for the "
                             "full 250-prompt run.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    backend = XSTestRefusalJudgeBackend(model=args.model_id, base_url=args.base_url, api_key=None)

    print("=" * 84)
    print(f"XSTest refusal-judge sensitivity check — model_id={backend.model_id}")
    print(f"results_dir={results_dir}  max_samples={args.max_samples}")
    print("=" * 84)
    print(f"{'model_alias':<22} {'n':>5} {'OR strict':>10} {'OR broad':>10} {'parse_err':>10}")
    print("-" * 84)

    try:
        for alias in args.models:
            summary = run_xstest_judge_validation(
                results_dir, alias, backend,
                judge_name=backend.name, max_samples=args.max_samples,
            )
            m = summary["metrics"]
            ors = m["over_refusal_rate_strict"]
            orb = m["over_refusal_rate_broad"]
            print(f"{alias:<22} {m['num_samples']:>5} "
                  f"{(ors if ors is not None else 0):>10.3f} "
                  f"{(orb if orb is not None else 0):>10.3f} "
                  f"{m['parse_error_count']:>10}")
    finally:
        backend.close()

    print("-" * 84)
    print("Wrote scores.judge.xstest_api.jsonl + summary.judge.xstest_api.json per model.")
    print("Compute agreement vs the v2 regex with scripts/xstest_judge_agreement.py (no GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
