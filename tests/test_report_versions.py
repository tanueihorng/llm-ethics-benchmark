"""Report-version registry gate (D74, 2026-08-25).

configs/report_versions.yaml claims to be the complete map of every FYP
report / thesis / interim document version. A registry nobody checks is worse
than none: it goes stale silently and then misleads the next reader. This
module keeps four properties true:

1. COVERAGE — every rendered .docx under docs/ is registered (role-tagged
   `alias` reuses, e.g. a canonical base doubling as a humanized pair member,
   are allowed on top of single ownership).
2. EXISTENCE + SANITY — every registered artifact exists on disk and yields a
   non-trivial structure extraction (>= 1 Heading 1, >= MIN_BLOCKS blocks).
   Skipped on a fresh clone where the gitignored docx are absent (D60).
3. SYNC — docs/report_versions/inventory.json, the committed per-edge
   .summary.md records, and docs/REPORT_VERSIONS.md match what the tool
   recomputes from the current artifacts.
4. PRIVACY — committed summaries never quote more than QUOTE_CAP characters
   of prose (the .full.* extracts stay gitignored).

Clearing a failure here means regenerating (`make report-versions`) or fixing
the registry — never deleting the check.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = str(REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from report_tools.report_version_diff import (  # noqa: E402
    MIN_BLOCKS,
    QUOTE_CAP,
    TRANSIENT_RE,
    diff_blocks,
    edge_slug,
    edges_of,
    extract_blocks,
    load_registry,
    rel,
)
from report_tools.workflows import required_outputs

OUT_DIR = REPO_ROOT / "docs" / "report_versions"
INDEX_PATH = REPO_ROOT / "docs" / "REPORT_VERSIONS.md"


def _docx_on_disk() -> list[Path]:
    docs = REPO_ROOT / "docs"
    if not docs.exists():
        return []
    return [p for p in sorted(docs.rglob("*.docx")) if not TRANSIENT_RE.match(p.name)]


def test_registry_parses_and_families_are_unique():
    registry = load_registry()
    ids = [f["id"] for f in registry["families"]]
    assert len(ids) == len(set(ids)), f"duplicate family ids: {ids}"
    assert ids, "registry defines no families"


def test_every_docx_on_disk_is_registered():
    disk = _docx_on_disk()
    if not disk:
        pytest.skip("no docx under docs/ (fresh clone); nothing to cover")
    registry = load_registry()
    registered = {v["path"] for fam in registry["families"]
                  for v in fam.get("versions", [])}
    registered |= {v["path"] for v in registry.get("standalone_documents", [])}
    unregistered = [str(p.relative_to(REPO_ROOT)) for p in disk
                    if str(p.relative_to(REPO_ROOT)) not in registered]
    assert not unregistered, (
        f"docx on disk missing from configs/report_versions.yaml: {unregistered}. "
        "Register them (a family or standalone_documents) so version tracking "
        "cannot silently lose a document.")


def test_every_registered_artifact_exists_and_extracts():
    registry = load_registry()
    missing: list[str] = []
    thin: list[tuple[str, int, int]] = []
    have_any = bool(_docx_on_disk())
    # CI has an explicit clean-clone coverage contract. Locally the original
    # complete-inventory requirement remains in force, including history.
    profile = os.environ.get("FYP_REPORT_PROFILE")
    required = set(required_outputs(registry, profile)) if profile else None
    for fam in registry["families"]:
        for v in fam.get("versions", []):
            path = rel(v["path"])
            if not path.exists():
                if (required is not None and v["path"] in required) or (required is None and have_any):
                    missing.append(v["path"])
                continue
            blocks = extract_blocks(path)
            h1 = [b for b in blocks if b.kind == "heading" and b.level == 1]
            if len(h1) < 1 or len(blocks) < MIN_BLOCKS:
                thin.append((v["path"], len(h1), len(blocks)))
    assert not missing, f"registered artifacts absent while others exist locally: {missing}"
    assert not thin, f"extraction implausibly thin (H1, blocks): {thin}"


def test_inventory_and_summaries_exist_and_match_edges():
    registry = load_registry()
    edges = [e for e in edges_of(registry)
             if rel(e["from_path"]).exists() and rel(e["to_path"]).exists()]
    if not edges:
        pytest.skip("edge artifacts absent (fresh clone)")
    inventory_path = OUT_DIR / "inventory.json"
    assert inventory_path.exists(), "inventory.json missing — run `make report-versions`"
    inv_edges = {e["slug"]: e
                 for e in json.loads(inventory_path.read_text(encoding="utf-8"))["edges"]}
    index_text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
    problems: list[str] = []
    for edge in edges:
        slug = edge_slug(edge["family_id"], edge["from_label"], edge["to_label"])
        rec = inv_edges.get(slug)
        if rec is None:
            problems.append(f"{slug}: no inventory entry — run `make report-versions`")
            continue
        summary = OUT_DIR / "changes" / f"{slug}.summary.md"
        if not summary.exists():
            problems.append(f"{slug}: {summary.name} missing — run `make report-versions`")
        if edge["family_id"] not in index_text:
            problems.append(f"index does not mention family {edge['family_id']}")
        if slug not in index_text:
            problems.append(f"index does not link change record {slug}")
    assert not problems, "; ".join(problems)


def test_summary_quote_cap_is_enforced_in_committed_records():
    changes = OUT_DIR / "changes"
    summaries = sorted(changes.glob("*.summary.md")) if changes.exists() else []
    if not summaries:
        pytest.skip("no generated change records")
    pattern = re.compile(r"“([^”]{%d,})”" % (QUOTE_CAP + 2))
    offenders = []
    for f in summaries:
        text = f.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            offenders.append((f.name, len(m.group(1))))
    assert not offenders, (
        f"committed summaries quote more than {QUOTE_CAP} chars of prose: {offenders[:5]}. "
        "Full prose belongs only in the gitignored .full.* extracts (D60 rationale).")


def test_full_prose_extracts_are_gitignored():
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in ("docs/report_versions/changes/*.full.md",
                 "docs/report_versions/changes/*.full.json"):
        assert name in gi, f".gitignore must exclude {name}"
    fulls = sorted((OUT_DIR / "changes").glob("*.full.*")) if (OUT_DIR / "changes").exists() else []
    if not fulls:
        pytest.skip("no .full.* extracts on disk")
    out = subprocess.run(["git", "check-ignore", "--", *[str(p) for p in fulls]],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    ignored = set(out.stdout.split())
    not_ignored = [str(p) for p in fulls if str(p) not in ignored]
    assert not not_ignored, f"full-prose extracts would be committed: {not_ignored}"


def test_recomputed_diff_counts_match_inventory():
    """Strongest sync check: recount one edge per family live and compare.

    A stale inventory after any artifact rebuild is caught without re-running
    every extraction; the CLI's own `all` run covers all edges when invoked.
    """
    inv_path = OUT_DIR / "inventory.json"
    if not inv_path.exists():
        pytest.skip("inventory.json missing — run `make report-versions`")
    inv = {e["slug"]: e
           for e in json.loads(inv_path.read_text(encoding="utf-8"))["edges"]}
    checked = 0
    failures: list[str] = []
    for fam in load_registry()["families"]:
        if fam.get("diff") != "consecutive":
            continue
        versions = fam.get("versions", [])
        for a, b in zip(versions, versions[1:]):
            if not (rel(a["path"]).exists() and rel(b["path"]).exists()):
                continue
            ops = diff_blocks(extract_blocks(rel(a["path"])),
                              extract_blocks(rel(b["path"])))
            counts: dict[str, int] = {}
            for o in ops:
                counts[o.op] = counts.get(o.op, 0) + 1
            slug = edge_slug(fam["id"], a.get("label", a["path"]),
                             b.get("label", b["path"]))
            rec = inv.get(slug)
            if rec is None:
                failures.append(f"{slug}: absent from inventory")
                continue
            for key in ("added", "removed", "edited", "renumber_only"):
                want, got = rec["counts"][key], counts.get(key, 0)
                if want != got:
                    failures.append(
                        f"{slug}: inventory says {key}={want}, live recount gives {got}")
            checked += 1
            break  # one edge per family suffices for staleness detection
    if checked == 0:
        pytest.skip("no fully-present consecutive pair found (fresh clone?)")
    assert not failures, ("stale change records — rerun `make report-versions`: "
                          + "; ".join(failures))
