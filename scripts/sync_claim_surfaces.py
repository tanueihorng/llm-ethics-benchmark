#!/usr/bin/env python3
"""Inject registry-derived data blocks into current static presentation surfaces."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from claim_registry import ROOT, build_registry


DECK_THEMES = {
    "docs/fyp_showcase.html": {
        "qwen_2b": "#2563eb", "qwen_4b": "#2563eb", "llama_3_2_3b": "#0891b2",
        "mistral_7b": "#7c3aed", "phi4_mini": "#d97706",
    },
    "docs/fyp_showcase_3d.html": {
        "qwen_2b": "#5b8def", "qwen_4b": "#3f6fd8", "llama_3_2_3b": "#37d5e8",
        "mistral_7b": "#a78bfa", "phi4_mini": "#f0a13e",
    },
    "docs/fyp_showcase_earth.html": {
        "qwen_2b": "#7fa6ff", "qwen_4b": "#5b8def", "llama_3_2_3b": "#4dd6e8",
        "mistral_7b": "#b78bfa", "phi4_mini": "#ffb45c",
    },
    "docs/fyp_showcase_bluemarble.html": {
        "qwen_2b": "#7fa6ff", "qwen_4b": "#5b8def", "llama_3_2_3b": "#4dd6e8",
        "mistral_7b": "#b78bfa", "phi4_mini": "#ffb45c",
    },
    "docs/fyp_showcase_world.html": {
        "qwen_2b": "#7fa6ff", "qwen_4b": "#5b8def", "llama_3_2_3b": "#4dd6e8",
        "mistral_7b": "#b78bfa", "phi4_mini": "#ffb45c",
    },
    "docs/fyp_showcase_world3d.html": {
        "qwen_2b": "#7fa6ff", "qwen_4b": "#5b8def", "llama_3_2_3b": "#4dd6e8",
        "mistral_7b": "#b78bfa", "phi4_mini": "#ffb45c",
    },
}

PAIR_BLOCK = re.compile(
    r"(?:/\* CLAIM_REGISTRY:PAIRS [0-9a-f]+ \*/\n.*?\n/\* END_CLAIM_REGISTRY:PAIRS \*/|const PAIRS = \[.*?\n\];)",
    re.DOTALL,
)
DEFENSE_BLOCK = re.compile(
    r"(?:<!-- CLAIM_REGISTRY:DEFENSE_ASR [0-9a-f]+ -->\n.*?\n<!-- END_CLAIM_REGISTRY:DEFENSE_ASR -->|"
    r"    <div class=\"asr-table\"[^\n]+)",
    re.DOTALL,
)


def _format_deck_pairs(registry: dict, colors: dict[str, str]) -> str:
    pairs = []
    for source in registry["render"]["deck_pairs"]:
        row = dict(source)
        row["color"] = colors[row["id"]]
        if row["id"] == "phi4_mini":
            row["lpos"] = "below"
        pairs.append(row)
    payload = json.dumps(pairs, ensure_ascii=False, indent=2, separators=(",", ": "))
    fp = registry["registry_fingerprint"]
    return (
        f"/* CLAIM_REGISTRY:PAIRS {fp} */\n"
        f"const PAIRS = {payload};\n"
        "/* END_CLAIM_REGISTRY:PAIRS */"
    )


def _fmt_short(value: float) -> str:
    if value == 1.0:
        return "1.000"
    return f"{value:.3f}".lstrip("0")


def _fmt_pp(value: float) -> tuple[str, str]:
    if abs(value) < 0.0005:
        return "0.0 pp", "flat"
    if value < 0:
        return f"−{abs(value):.1f} pp", "safe"
    return f"+{value:.1f} pp", "warn"


def _format_defense_rows(registry: dict) -> str:
    rows = []
    for row in registry["render"]["defense_asr_rows"]:
        delta, cls = _fmt_pp(row["delta_pp"])
        rows.append(
            '<div class="asr-row"><span class="model">'
            f'{row["display_name"]}</span><span class="n {cls}">{delta}</span>'
            f'<span class="n">{_fmt_short(row["p_value"])}</span>'
            f'<span class="n">{_fmt_short(row["bh_q_value"])}</span></div>'
        )
    fp = registry["registry_fingerprint"]
    table = (
        '    <div class="asr-table" data-reveal style="--i:2">'
        '<div class="asr-row head"><span>Pair</span><span>Δ ASR, judge</span>'
        '<span>McNemar p</span><span>BH q</span></div>'
        + "".join(rows)
        + "</div>"
    )
    return (
        f"<!-- CLAIM_REGISTRY:DEFENSE_ASR {fp} -->\n"
        f"{table}\n"
        "<!-- END_CLAIM_REGISTRY:DEFENSE_ASR -->"
    )


def expected_surface_text(path: Path, registry: dict) -> str:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    if rel in DECK_THEMES:
        replacement = _format_deck_pairs(registry, DECK_THEMES[rel])
        updated, count = PAIR_BLOCK.subn(replacement, text, count=1)
        if count != 1:
            raise ValueError(f"could not locate PAIRS block in {rel}")
        return updated
    if rel == "docs/fyp-report-defense-deck-2026-07.html":
        updated, count = DEFENSE_BLOCK.subn(_format_defense_rows(registry), text, count=1)
        if count != 1:
            raise ValueError(f"could not locate ASR table in {rel}")
        return updated
    raise ValueError(f"no claim-surface renderer for {rel}")


def sync_surfaces(*, write: bool, root: Path = ROOT) -> list[str]:
    if root != ROOT:
        raise ValueError("sync_surfaces currently operates on the repository root")
    registry = build_registry(root)
    changed: list[str] = []
    paths = [root / rel for rel in (*DECK_THEMES, "docs/fyp-report-defense-deck-2026-07.html")]
    for path in paths:
        current = path.read_text(encoding="utf-8")
        expected = expected_surface_text(path, registry)
        if current != expected:
            changed.append(path.relative_to(root).as_posix())
            if write:
                path.write_text(expected, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = sync_surfaces(write=args.write)
    if changed:
        action = "updated" if args.write else "stale"
        for path in changed:
            print(f"{action}  {path}")
        return 0 if args.write else 1
    print("ok  generated claim-surface blocks are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
