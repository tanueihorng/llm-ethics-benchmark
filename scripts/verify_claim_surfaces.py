#!/usr/bin/env python3
"""Verify claim-registry freshness, surface coverage, and per-file semantics."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import yaml

from claim_registry import ROOT, build_registry, registry_is_fresh
from sync_claim_surfaces import sync_surfaces


MANIFEST = ROOT / "configs/claim_surfaces.yaml"
PAIR_MARKER = re.compile(
    r"/\* CLAIM_REGISTRY:PAIRS ([0-9a-f]+) \*/\nconst PAIRS = (\[.*?\]);\n"
    r"/\* END_CLAIM_REGISTRY:PAIRS \*/",
    re.DOTALL,
)
DEFENSE_MARKER = re.compile(
    r"<!-- CLAIM_REGISTRY:DEFENSE_ASR ([0-9a-f]+) -->\n(.*?)\n"
    r"<!-- END_CLAIM_REGISTRY:DEFENSE_ASR -->",
    re.DOTALL,
)
VOLATILE_PATTERNS = (
    re.compile(r"\b\d+\s+(?:automated\s+)?tests?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:claim\s+(?:lock|gate)|verify-claims|machine\s+checks?)[^\n]{0,40}\d+/\d+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+/\d+\s+(?:claim\s+)?checks?\b", re.IGNORECASE),
    re.compile(
        r"data-count=[\"']\d+[\"'][^\n]{0,160}(?:automated\s+tests?|claim\s+checks?|machine\s+checks?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:ahead|behind)\s+(?:of\s+)?(?:origin/)?main\s+by\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:main|origin/main|HEAD)\s+(?:is\s+)?(?:at|on)\s+[0-9a-f]{7,40}\b", re.IGNORECASE),
)


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def discover_paths(manifest: dict[str, Any], root: Path = ROOT) -> set[str]:
    found: set[str] = set()
    for pattern in manifest["discovery"]:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    return found


def unregistered_surfaces(manifest: dict[str, Any], root: Path = ROOT) -> set[str]:
    registered = {row["path"] for row in manifest["surfaces"]}
    return discover_paths(manifest, root) - registered


def find_volatile_claims(text: str) -> list[str]:
    hits: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        # Dated revision logs and audit transcripts are immutable history, not
        # live status claims. Current-facing prose must omit the date escape.
        if re.match(r'^[\s>|*\[\"()\-]*20\d{2}-\d{2}-\d{2}\b', line):
            continue
        probe = re.sub(r"[*`_]", "", line)
        for pattern in VOLATILE_PATTERNS:
            if pattern.search(probe):
                hits.append(f"line {line_number}: {line.strip()[:140]}")
                break
    return hits


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        paragraphs.append("".join(paragraph.itertext()))
    return "\n".join(paragraphs)


def _surface_text(path: Path) -> str:
    if path.suffix == ".docx":
        return _docx_text(path)
    return path.read_text(encoding="utf-8")


def validate_deck_pairs(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    match = PAIR_MARKER.search(text)
    if not match:
        return False, "missing generated PAIRS marker"
    if match.group(1) != registry["registry_fingerprint"]:
        return False, "PAIRS marker fingerprint is stale"
    actual = json.loads(match.group(2))
    expected = registry["render"]["deck_pairs"]
    if len(actual) != len(expected):
        return False, f"expected {len(expected)} pairs, found {len(actual)}"
    for got, want in zip(actual, expected, strict=True):
        for key, value in want.items():
            if got.get(key) != value:
                return False, f"{want['id']}.{key}: {got.get(key)!r} != {value!r}"
    return True, f"{len(actual)} registry-derived pair records"


def validate_defense(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    match = DEFENSE_MARKER.search(text)
    if not match:
        return False, "missing generated defense-ASR marker"
    if match.group(1) != registry["registry_fingerprint"]:
        return False, "defense-ASR marker fingerprint is stale"
    block = match.group(2)
    for row in registry["render"]["defense_asr_rows"]:
        if row["display_name"] not in block:
            return False, f"missing defense row for {row['pair_id']}"
        for value in (row["p_value"], row["bh_q_value"]):
            rendered = "1.000" if value == 1.0 else f"{value:.3f}".lstrip("0")
            if rendered not in block:
                return False, f"missing {row['pair_id']} value {rendered}"
    return True, "all defense rows match registry p/q values"


def _validate_markdown(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    comparable = text.replace("*", "").replace("`", "")
    rows = {
        row[0]: row for row in registry["render"]["report_table_6_1"]
        if row[1] == "HarmBench ASR (judge)"
    }
    for pair_id, row in rows.items():
        lines = [line for line in comparable.splitlines() if pair_id in line]
        if not lines:
            return False, f"missing row for {pair_id}"
        if not any(all(token in line for token in (row[2], row[3], row[4])) for line in lines):
            return False, f"{pair_id} row is not registry-derived ({row[2]}->{row[3]} {row[4]})"
    return True, "five per-pair ASR rows match registry"


def _validate_primary_text(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    comparable = (
        text.replace("−", "-")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace(r"\,", "")
    )
    required = [
        "-0.090",
        "-0.032",
        "-0.048",
        "512",
    ]
    missing = [token for token in required if token not in comparable]
    if missing:
        return False, f"missing primary-study token {missing[0]!r}"
    if registry["claims"]["multiplicity"]["asr_survivor_count"] != 0:
        return False, "registry itself reports an ASR BH survivor"
    return True, "primary values and budget present"


def _validate_zip(surface: dict[str, Any], root: Path) -> tuple[bool, str]:
    archive_path = root / surface["path"]
    source = (root / surface["source"]).read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        try:
            bundled = archive.read(surface["zip_member"])
        except KeyError:
            return False, f"missing zip member {surface['zip_member']}"
    if bundled != source:
        return False, f"{surface['zip_member']} differs from {surface['source']}"
    return True, f"{surface['zip_member']} byte-matches source"


def validate_surface(
    surface: dict[str, Any], registry: dict[str, Any], root: Path = ROOT
) -> list[tuple[str, bool, str]]:
    path = root / surface["path"]
    results: list[tuple[str, bool, str]] = []
    local_root_absent = surface.get("availability") == "local_optional" and not (root / "fyp_submission").exists()
    if not path.exists():
        if local_root_absent:
            return [("availability", True, "local submission tree absent; optional on clone")]
        return [("availability", False, "required surface is missing")]

    text: str | None = None
    for profile in surface.get("profiles", []):
        if profile == "pdf":
            ok = path.read_bytes().startswith(b"%PDF-")
            results.append((profile, ok, "valid PDF signature" if ok else "invalid PDF signature"))
            continue
        if profile == "zip_matches_tex":
            ok, detail = _validate_zip(surface, root)
            results.append((profile, ok, detail))
            continue
        if profile == "byte_matches_source":
            source = root / surface["source"]
            ok = path.read_bytes() == source.read_bytes()
            results.append((profile, ok, "byte-matches generated source" if ok else f"differs from {surface['source']}"))
            continue
        if profile == "fresh_from_source":
            source = root / surface["source"]
            ok = path.stat().st_mtime_ns >= source.stat().st_mtime_ns
            results.append((profile, ok, "not older than source" if ok else f"older than {surface['source']}"))
            continue
        if text is None:
            text = _surface_text(path)
        if profile == "registry_consumer":
            tokens = ("loadClaimRegistry", "CLAIMS.render")
            missing = [token for token in tokens if token not in text]
            results.append((profile, not missing, "imports and consumes registry" if not missing else f"missing {missing[0]}"))
        elif profile == "volatile_free":
            live_text = text
            if path.suffix == ".docx" and "Appendix G: Document Revision History" in live_text:
                live_text = live_text.split("Appendix G: Document Revision History", 1)[0]
            hits = find_volatile_claims(live_text)
            results.append((profile, not hits, "no volatile counts/Git state" if not hits else hits[0]))
        elif profile == "markdown_primary":
            ok, detail = _validate_markdown(text, registry)
            results.append((profile, ok, detail))
        elif profile in {"docx_primary", "latex_primary"}:
            ok, detail = _validate_primary_text(text, registry)
            results.append((profile, ok, detail))
        elif profile == "generated_deck_pairs":
            ok, detail = validate_deck_pairs(text, registry)
            results.append((profile, ok, detail))
        elif profile == "generated_defense_asr":
            ok, detail = validate_defense(text, registry)
            results.append((profile, ok, detail))
        elif profile == "snapshot_banner":
            ok = "DATA SNAPSHOT" in text
            results.append((profile, ok, "snapshot visibly bannered" if ok else "missing DATA SNAPSHOT banner"))
        elif profile == "project_log_current":
            current = text.split("### Historical 2026-07-02 snapshot", 1)[0]
            hits = find_volatile_claims(current)
            results.append((profile, not hits, "current-state prose has no volatile counts/Git state" if not hits else hits[0]))
        else:
            results.append((profile, False, "unknown validation profile"))
    if not surface.get("profiles"):
        results.append(("registered", True, f"classified as {surface['lifecycle']}"))
    return results


def run_checks(root: Path = ROOT, manifest_path: Path = MANIFEST) -> list[tuple[str, str, str]]:
    manifest = load_manifest(manifest_path)
    registry = build_registry(root)
    checks: list[tuple[str, str, str]] = []
    fresh, detail = registry_is_fresh(root)
    checks.append(("PASS" if fresh else "FAIL", "registry:fresh", detail))

    unregistered = sorted(unregistered_surfaces(manifest, root))
    checks.append((
        "PASS" if not unregistered else "FAIL",
        "surfaces:coverage",
        f"{len(manifest['surfaces'])} registered" if not unregistered else f"unregistered: {unregistered[0]}",
    ))

    stale_blocks = sync_surfaces(write=False, root=root)
    checks.append((
        "PASS" if not stale_blocks else "FAIL",
        "surfaces:generated-blocks",
        "all generated blocks current" if not stale_blocks else f"stale: {stale_blocks[0]}",
    ))

    for surface in manifest["surfaces"]:
        for profile, ok, result_detail in validate_surface(surface, registry, root):
            checks.append(("PASS" if ok else "FAIL", f"{surface['id']}:{profile}", result_detail))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    if not args.quiet:
        width = max(len(name) for _, name, _ in checks)
        for status, name, detail in checks:
            marker = "ok  " if status == "PASS" else "FAIL"
            print(f"{marker}  {name:<{width}}  {detail}")
        failures = sum(status == "FAIL" for status, _, _ in checks)
        print("-" * 60)
        print(f"{len(checks)} surface checks: {len(checks) - failures} pass, {failures} fail")
    return 1 if any(status == "FAIL" for status, _, _ in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
