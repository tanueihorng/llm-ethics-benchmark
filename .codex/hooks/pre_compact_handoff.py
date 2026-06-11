#!/usr/bin/env python3
"""Refresh compact handoff artifacts before context compaction."""

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


def _run(root: Path, args: list[str]) -> None:
    result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(f"[fyp-hook] {' '.join(args)} failed:", file=sys.stderr)
        print((result.stdout + result.stderr).strip(), file=sys.stderr)


def _has_local_changes(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip())


def main() -> int:
    try:
        root = _repo_root()
    except RuntimeError as exc:
        print(f"[fyp-hook] {exc}", file=sys.stderr)
        return 0

    if not _has_local_changes(root):
        print("[fyp-hook] Working tree clean; skipping handoff/dashboard refresh.", file=sys.stderr)
        return 0

    _run(root, [sys.executable, "scripts/generate_handoff.py"])
    _run(root, [sys.executable, "scripts/generate_agent_dashboard.py"])
    print("[fyp-hook] Refreshed docs/HANDOFF.md and docs/AGENT_DASHBOARD.md.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
