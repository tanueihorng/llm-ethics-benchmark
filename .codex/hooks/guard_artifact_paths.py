#!/usr/bin/env python3
"""Warn before tool calls that appear to target immutable raw artifacts.

Codex hook payloads can vary by surface. This hook intentionally treats stdin
as opaque text and searches for the repo's immutable result paths. It exits 0
so it is a reminder layer; `make agent-check` remains the enforcing gate.
"""

from __future__ import annotations

import re
import sys


IMMUTABLE_PATH = re.compile(r"results/[^\s'\"`]+/[^\s'\"`]+/(?:raw\.jsonl|summary\.json)")


def main() -> int:
    payload = sys.stdin.read()
    hits = sorted(set(IMMUTABLE_PATH.findall(payload)))
    if hits:
        print("[fyp-hook] Immutable artifact path detected.", file=sys.stderr)
        print("[fyp-hook] Do not mutate TC1-original raw.jsonl or summary.json.", file=sys.stderr)
        for hit in hits[:10]:
            print(f"[fyp-hook] - {hit}", file=sys.stderr)
        print("[fyp-hook] Use derived sidecars and run make agent-check before finishing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
