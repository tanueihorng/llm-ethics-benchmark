"""Generate a TC1-safe checklist for future agent sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ethical_benchmark.harness import build_agent_status, load_policy, render_tc1_checklist  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/TC1_AGENT_CHECKLIST.md.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output", default="docs/TC1_AGENT_CHECKLIST.md")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    status = build_agent_status(repo_root, load_policy(repo_root))
    output = repo_root / args.output
    output.write_text(render_tc1_checklist(status), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
