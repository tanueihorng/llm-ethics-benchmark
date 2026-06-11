#!/usr/bin/env python3
"""Run a lightweight harness check when an agent turn stops."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "not in a git repo")
    return Path(result.stdout.strip())


def _print_tail(label: str, text: str, limit: int = 40) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    print(f"[fyp-hook] {label}", file=sys.stderr)
    for line in lines[-limit:]:
        print(f"[fyp-hook] {line}", file=sys.stderr)


def main() -> int:
    try:
        root = _repo_root()
    except RuntimeError as exc:
        print(f"[fyp-hook] {exc}", file=sys.stderr)
        return 0

    status = subprocess.run(
        [sys.executable, "fyp_cli.py", "agent-status"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    _print_tail("agent-status", status.stdout + status.stderr, limit=24)

    check = subprocess.run(
        [sys.executable, "scripts/agent_check.py", "--skip-pytest"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    _print_tail("lightweight agent-check", check.stdout + check.stderr, limit=40)
    if check.returncode != 0:
        print("[fyp-hook] Full finish gate remains: make agent-check", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
