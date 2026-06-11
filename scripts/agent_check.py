"""Run repo-native agent harness checks.

This is the command behind ``make agent-check``. It validates cross-agent docs,
project-log discipline, immutable raw artifacts, stale text, redaction rules,
report freshness, whitespace, and optionally pytest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.harness import load_policy, run_agent_checks, write_immutable_manifest  # noqa: E402


def _print_results(results) -> None:
    for result in results:
        marker = "PASS" if result.status == "pass" else "WARN" if result.status == "warn" else "FAIL"
        print(f"[{marker}] {result.name}: {result.message}")
        for detail in result.details:
            print(f"  - {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent harness checks.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--policy", default="configs/artifact_policy.yaml")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-diff-check", action="store_true")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Optional pytest args after --, for example: --pytest-args -- -q tests/test_agent_harness.py",
    )
    parser.add_argument(
        "--write-immutable-manifest",
        action="store_true",
        help="Refresh the immutable raw-artifact hash manifest, then exit.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_policy(repo_root, Path(args.policy))

    if args.write_immutable_manifest:
        manifest = write_immutable_manifest(repo_root, policy)
        print(f"Wrote {manifest}")
        return 0

    pytest_args = None
    if args.pytest_args:
        pytest_args = [item for item in args.pytest_args if item != "--"]

    results = run_agent_checks(
        repo_root,
        include_pytest=not args.skip_pytest,
        include_diff_check=not args.skip_diff_check,
        pytest_args=pytest_args,
        policy=policy,
    )
    _print_results(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
