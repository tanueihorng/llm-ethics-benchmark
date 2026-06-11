"""Run a focused harness self-evaluation suite.

This wrapper gives agents a named command for checking the harness tests without
having to remember the test module path.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "tests/test_agent_harness.py"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
