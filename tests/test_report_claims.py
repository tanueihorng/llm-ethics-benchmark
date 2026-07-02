"""Claim lock for the canonical FYP report (D43).

Runs the deterministic claim checker (scripts/verify_report_claims.py): every
load-bearing number in scripts/build_fyp_report_v5.js is asserted both in the
report text and against the committed analysis artifacts, so the report can
never silently drift from its evidence. Local-only checks (gitignored raws)
SKIP on a fresh clone; a FAIL here means a report claim and an artifact
disagree — fix the report (or the artifact provenance), never this test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_report_claims as vrc  # noqa: E402


def test_report_claims_match_artifacts() -> None:
    checker = vrc.run_checks()
    fails = [(n, d) for s, n, d in checker.results if s == "FAIL"]
    assert not fails, "report/artifact claim mismatches: " + "; ".join(
        f"{n} ({d})" for n, d in fails
    )
    # the suite must actually have run a meaningful number of checks
    assert len(checker.results) >= 40
