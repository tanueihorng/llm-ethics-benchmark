"""Archive everything that exists ONLY on this machine.

Some of this project is deliberately not in git, and those decisions are correct:
`.gitignore` says supervisor correspondence is "keep local, never commit", D60
untracked the generated documents, and the raw model outputs contain unredacted
harmful text that must not be committed anywhere. The consequence is that a
laptop failure loses them, and git is not the tool that fixes it -- a backup is.

What is genuinely unrecoverable, measured rather than assumed:

    results*/**/raw.jsonl   ~97 MB over 60 files. The actual experimental record.
                            Regenerating it needs TC1 GPU hours AND accepted
                            HuggingFace licences for gated weights. In practice
                            it cannot be reproduced.
    human_labels/           ~1.4 MB. Two 200-item gold sets the student labelled
                            BY HAND. No amount of compute brings these back.
    docs/email_*.md         Supervisor correspondence, written once.
    fyp_submission/         LaTeX sources, Overleaf bundles, submission PDFs.
    D94 local-only files    Report and deck builders, rendered reports, decks and
                            showcase pages. Read from the D94 block of .gitignore,
                            so the backup follows the ignore rules automatically.

Everything else in the repo is either tracked or rebuildable from a tracked
builder, so it is not included and does not need to be.

    python scripts/backup_local.py --dest ~/Dropbox/fyp_backup
    make backup DEST=~/Dropbox/fyp_backup

The archive CONTAINS UNREDACTED HARMFUL PROMPTS AND MODEL RESPONSES unless you
pass --no-sensitive. That is normal for this kind of study, but it means the
destination must be somewhere you would be comfortable storing that: an
encrypted disk or a private drive, not a shared team folder.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def local_only_rules(gitignore: Path = ROOT / ".gitignore") -> tuple[list[str], list[str]]:
    """Returns (globs, excluded dirs) for the D94 local-only report and deck files.

    The globs come from the .gitignore block headed `# D94` rather than being
    restated here, so a path added to the local-only set is backed up without a
    second edit. Directory rules elsewhere in .gitignore that sit inside a D94
    directory (world3d's caches and build output) were already local and
    rebuildable before D94, so they stay out of the archive.
    """
    if not gitignore.exists():
        return [], []
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    start = next((i for i, line in enumerate(lines) if line.startswith("# D94")), None)
    if start is None:
        return [], []
    block: list[str] = []
    end = len(lines)
    for index in range(start, len(lines)):
        if not lines[index]:
            end = index
            break
        if not lines[index].startswith("#"):
            block.append(lines[index])
    globs = [rule + "**/*" if rule.endswith("/") else rule for rule in block]
    d94_dirs = [rule for rule in block if rule.endswith("/")]
    excluded = [
        rule for rule in lines[:start] + lines[end:]
        if rule.endswith("/") and not rule.startswith("#")
        and any(rule != d94 and rule.startswith(d94) for d94 in d94_dirs)
    ]
    return globs, excluded


LOCAL_ONLY_LABEL = "local-only reports and presentations (D94)"
LOCAL_ONLY_GLOBS, LOCAL_ONLY_EXCLUDED_DIRS = local_only_rules()

# (label, glob relative to the repo root, sensitive?)
GROUPS: list[tuple[str, str, bool]] = [
    ("supervisor correspondence", "docs/email_*.md", False),
    ("submission staging tree", "fyp_submission/**/*", False),
    ("generated documents", "docs/*.docx", False),
    ("archived documents", "docs/archive/*.docx", False),
    ("hand-labelled gold sets", "human_labels/**/*", True),
    ("raw model outputs (128)", "results/**/raw.jsonl", True),
    ("raw model outputs (512)", "results_512/**/raw.jsonl", True),
    ("raw model outputs (multi-seed)", "results_sensitivity_512/**/raw.jsonl", True),
    *((LOCAL_ONLY_LABEL, pattern, False) for pattern in LOCAL_ONLY_GLOBS),
]

SKIP_PARTS = {".git", "node_modules", "__pycache__", ".venv", ".DS_Store"}


def collect(include_sensitive: bool) -> tuple[list[tuple[Path, str, bool]], dict]:
    picked: list[tuple[Path, str, bool]] = []
    seen: set[Path] = set()
    summary: dict[str, dict] = {}

    for label, pattern, sensitive in GROUPS:
        if sensitive and not include_sensitive:
            summary[label] = {"files": 0, "bytes": 0, "sensitive": True, "skipped": True}
            continue
        # Several globs can share a label (the D94 group), so accumulate.
        entry = summary.setdefault(label, {"files": 0, "bytes": 0, "sensitive": sensitive,
                                           "skipped": False})
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            if SKIP_PARTS & set(path.parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if any(rel.startswith(prefix) for prefix in LOCAL_ONLY_EXCLUDED_DIRS):
                continue
            if path in seen:
                continue
            seen.add(path)
            picked.append((path, label, sensitive))
            entry["files"] += 1
            entry["bytes"] += path.stat().st_size
    return picked, summary


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n/1:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt(n: int) -> str:
    step = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if step < 1024 or unit == "GB":
            return f"{step:.1f} {unit}"
        step /= 1024
    return f"{step:.1f} GB"


def build_manifest(summary: dict, include_sensitive: bool, stamp: str) -> str:
    lines = [
        "fyp_quant local backup",
        f"created: {stamp}",
        f"source:  {ROOT}",
        "",
        "This archive holds the parts of the project that exist ONLY on the author's",
        "machine. Everything else is in git or is rebuilt by a tracked builder.",
        "",
    ]
    for label, info in summary.items():
        if info["skipped"]:
            lines.append(f"  [skipped]  {label}  (--no-sensitive was passed)")
        else:
            mark = "  SENSITIVE" if info["sensitive"] else ""
            lines.append(f"  {info['files']:>4} files  {_fmt(info['bytes']):>9}  {label}{mark}")
    lines += [
        "",
        "SENSITIVE entries contain unredacted harmful prompts and model responses.",
        "Store this archive accordingly: encrypted or private, never a shared folder.",
        "",
        "To restore, extract into a fresh clone of the repository; every path is",
        "relative to the repository root.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True,
                        help="directory to write the archive into (must be OUTSIDE the repo)")
    parser.add_argument("--no-sensitive", action="store_true",
                        help="exclude raw model outputs and the human gold sets")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    dest = args.dest.expanduser().resolve()
    # A backup that lives inside the thing it is backing up is not a backup, and
    # would also sit next to a .gitignore that is the reason these files are loose.
    if dest == ROOT or ROOT in dest.parents:
        print(f"refusing: --dest {dest} is inside the repository", file=sys.stderr)
        return 2

    include_sensitive = not args.no_sensitive
    picked, summary = collect(include_sensitive)
    if not picked:
        print("nothing to back up (is this the right repository?)", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    manifest = build_manifest(summary, include_sensitive, stamp)
    print(manifest)

    total = sum(info["bytes"] for info in summary.values() if not info["skipped"])
    print(f"  TOTAL  {len(picked)} files  {_fmt(total)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"fyp_quant_backup_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path, _label, _sensitive in picked:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        info = tarfile.TarInfo("MANIFEST.txt")
        data = manifest.encode()
        info.size = len(data)
        import io
        tar.addfile(info, io.BytesIO(data))

    # Verify by reading it back rather than trusting the write.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getnames()
    expected = len(picked) + 1
    if len(members) != expected:
        print(f"\nARCHIVE VERIFY FAILED: {len(members)} members, expected {expected}",
              file=sys.stderr)
        return 1

    print(f"\nwrote {archive}  ({_fmt(archive.stat().st_size)}, {len(members)} members, verified)")
    if include_sensitive:
        print("this archive contains unredacted harmful text -- store it privately")
    return 0


if __name__ == "__main__":
    sys.exit(main())
