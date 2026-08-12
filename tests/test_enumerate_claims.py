"""The T51 enumeration instrument.

The audit that produced this tool found a bug in the tool itself, mid-audit:
`<w:t[^>]*>` also matches `<w:tab>`, so table-of-contents field markup was being
captured as body text. That is the same defect shape T51 exists to catch -- a
pattern loose enough to match the wrong thing -- so it is pinned here rather than
left to be rediscovered.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "scripts" / "report_tools" / "enumerate_claims.py"
DOC = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"


def _run(tmp_path: Path) -> list[dict]:
    import json

    out = tmp_path / "sentences.json"
    subprocess.run(
        [sys.executable, str(TOOL), str(DOC), str(out)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    return json.loads(out.read_text())


@pytest.mark.skipif(not DOC.exists(), reason="submission candidate not built (gitignored, D60)")
def test_extraction_carries_no_wordprocessing_markup(tmp_path: Path) -> None:
    """The <w:tab> regression. Body text must never contain raw OOXML."""
    rows = _run(tmp_path)
    leaked = [r for r in rows if "<w:" in r["sentence"] or "w:val=" in r["sentence"]]
    assert not leaked, (
        f"{len(leaked)} extracted sentences contain OOXML markup, e.g. "
        f"{leaked[0]['sentence'][:120]!r}. The tag pattern has loosened again: "
        "`<w:t[^>]*>` matches `<w:tab>`, so field markup is captured as text."
    )


@pytest.mark.skipif(not DOC.exists(), reason="submission candidate not built (gitignored, D60)")
def test_extraction_covers_every_chapter(tmp_path: Path) -> None:
    """Coverage, not just cleanliness: silently dropping a chapter would make an
    enumeration audit report a clean result over a partial document."""
    rows = _run(tmp_path)
    chapters = {r["chapter"] for r in rows}
    for n in range(1, 8):
        assert any(c.startswith(f"Chapter {n}:") for c in chapters), (
            f"Chapter {n} produced no sentences; the heading walk is broken and any "
            "audit run on this output would be scoped to less than the whole document"
        )
    assert sum(1 for r in rows if r["table"]) > 100, "table cells stopped being extracted"
