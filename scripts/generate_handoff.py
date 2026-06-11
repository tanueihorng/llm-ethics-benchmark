"""Generate docs/HANDOFF.md from live repo state."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.harness import build_agent_status, load_policy, render_handoff, run_agent_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate repo handoff from live state.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output", default="docs/HANDOFF.md")
    parser.add_argument("--include-checks", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_policy(repo_root)
    status = build_agent_status(repo_root, policy)
    results = None
    if args.include_checks:
        results = run_agent_checks(repo_root, include_pytest=not args.skip_pytest, policy=policy)

    output = repo_root / args.output
    output.write_text(render_handoff(status, results), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
