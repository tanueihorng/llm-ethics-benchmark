"""The local backup must cover what is unrecoverable, and must refuse to write inside the repo.

This is the only protection for the parts of the project that are deliberately
not in git: ~97 MB of raw model outputs that need TC1 GPU hours and gated
HuggingFace licences to regenerate, two 200-item gold sets labelled by hand, and
the supervisor correspondence that `.gitignore` marks "never commit".

Two properties are pinned. The coverage list must keep naming those groups, so a
future edit cannot silently drop the irreplaceable ones; and the destination
guard must reject a path inside the repository, because a backup living on the
disk it is protecting against is not a backup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "backup_local.py"


def load_backup():
    spec = importlib.util.spec_from_file_location("backup_local", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backup_local"] = module
    spec.loader.exec_module(module)
    return module


def test_the_unrecoverable_groups_are_all_covered() -> None:
    """Losing any of these cannot be undone by re-running anything."""
    backup = load_backup()
    patterns = {pattern for _label, pattern, _sensitive in backup.GROUPS}

    assert "docs/email_*.md" in patterns, "supervisor correspondence dropped from the backup"
    assert "human_labels/**/*" in patterns, "the hand-labelled gold sets dropped from the backup"
    assert "fyp_submission/**/*" in patterns, "the submission tree dropped from the backup"
    raw = {p for p in patterns if p.endswith("raw.jsonl")}
    assert len(raw) >= 3, f"raw model outputs must cover all three result trees, got {raw}"


def test_raw_outputs_and_gold_sets_are_marked_sensitive() -> None:
    """They hold unredacted harmful text; --no-sensitive must be able to drop them."""
    backup = load_backup()
    for label, pattern, sensitive in backup.GROUPS:
        if pattern.endswith("raw.jsonl") or pattern.startswith("human_labels"):
            assert sensitive, f"{label} holds harmful text but is not marked sensitive"
        if pattern.startswith("docs/email_"):
            assert not sensitive, "correspondence is private, not harmful-content sensitive"


def test_no_sensitive_excludes_exactly_the_sensitive_groups() -> None:
    backup = load_backup()
    _picked_all, summary_all = backup.collect(include_sensitive=True)
    _picked_safe, summary_safe = backup.collect(include_sensitive=False)

    for label, info in summary_safe.items():
        if summary_all[label]["sensitive"]:
            assert info["skipped"], f"{label} is sensitive but was not skipped"
        else:
            assert not info["skipped"], f"{label} is not sensitive but was skipped"


@pytest.mark.parametrize("dest", [".", "docs", "docs/viva"])
def test_destination_inside_the_repo_is_refused(dest: str, monkeypatch, capsys) -> None:
    """A backup on the disk it protects against is not a backup."""
    backup = load_backup()
    monkeypatch.setattr(sys, "argv", ["backup_local.py", "--dest", str(REPO_ROOT / dest)])
    assert backup.main() == 2
    assert "inside the repository" in capsys.readouterr().err
