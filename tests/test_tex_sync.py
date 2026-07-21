"""T44 FS-16: tex↔JS source-consistency gate.

The hand-maintained LaTeX mirrors sit outside every generated-surface gate: the
Overleaf-zip check byte-matches the zips against the SAME tex files
(self-consistency), so a tex mirror can drift arbitrarily far from its JS
source without any gate firing. Four independent tex defects were found by the
2026-07-20 full sweep in exactly this hole (FS-9 Jin misattribution, FS-12
stale κ labels + 0.28≠0.29, FS-13 stale front matter, FS-10 venue drops).

This gate encodes the drift classes that actually occurred:
 1. STALE-TEXT patterns that must never (re)appear in a mirror.
 2. SYNC tokens: verdict-bearing table values/labels that must appear in BOTH
    the tex mirror and its JS source (each side's own typography).
Mirrors are gitignored → every check SKIPS gracefully on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

THESIS_TEX = ROOT / "fyp_submission/report_latex/final_thesis/mythesis.tex"
INTERIM_TEX = ROOT / "fyp_submission/report_latex/interim_report/mythesis.tex"
THESIS_JS = ROOT / "scripts/build_fyp_thesis_v4.js"
INTERIM_JS = ROOT / "scripts/build_fyp_interim.js"

# Patterns that must not appear in ANY mirror (each caught drifting once).
STALE = [
    "poor to moderate",                        # pre-T43 κ label (FS-12)
    "& worst;",                                # pre-T43 κ label (FS-12)
    "0.25--0.28",                              # numeric drift vs JS 0.25–0.29 (FS-12)
    "no consistent trend across bit-widths",   # Jin misattribution (FS-9)
]
# Interim-only: front matter presenting completed analyses as future (FS-13).
STALE_INTERIM = [
    "(architecturally independent; planned)",
    "will re-score the same generations",
    "Significance will be reported",
    "which the final report will complete.",
]
# Tokens that must exist on BOTH sides of each mirror pair: (tex form, js form).
SYNC = {
    "thesis": [
        ("0.36--0.59", "0.36 – 0.59"),
        ("0.25--0.29", "0.25 – 0.29"),
        ("fair to moderate", "fair to moderate"),
        ("fair, the lowest", "fair, the lowest"),
        ("340 immutable raw files", "340 immutable raw files"),
        ("\\bibitem{refConnor}", "Connor"),
    ],
    "interim": [
        ("0.36--0.59", "0.36 – 0.59"),
        ("0.25--0.29", "0.25 – 0.29"),
        ("fair to moderate", "fair to moderate"),
        ("fair, the lowest", "fair, the lowest"),
        ("\\bibitem{refConnor}", "Connor"),
    ],
}


def check_stale(tex_text: str, interim: bool) -> list[str]:
    """Return the stale patterns present in a mirror text (empty = clean)."""
    pats = STALE + (STALE_INTERIM if interim else [])
    return [p for p in pats if p in tex_text]


def check_sync(tex_text: str, js_text: str, key: str) -> list[str]:
    """Return sync tokens missing from either side (empty = in sync)."""
    problems = []
    for tex_form, js_form in SYNC[key]:
        if tex_form not in tex_text:
            problems.append(f"tex missing {tex_form!r}")
        if js_form not in js_text:
            problems.append(f"js missing {js_form!r}")
    return problems


def _require(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout (gitignored mirror)")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("tex,interim", [(THESIS_TEX, False), (INTERIM_TEX, True)])
def test_mirror_carries_no_stale_text(tex: Path, interim: bool) -> None:
    hits = check_stale(_require(tex), interim)
    assert not hits, f"{tex}: stale drift patterns present: {hits}"


@pytest.mark.parametrize("tex,js,key", [(THESIS_TEX, THESIS_JS, "thesis"), (INTERIM_TEX, INTERIM_JS, "interim")])
def test_mirror_in_sync_with_js_source(tex: Path, js: Path, key: str) -> None:
    problems = check_sync(_require(tex), js.read_text(encoding="utf-8"), key)
    assert not problems, f"{tex} vs {js}: {problems}"


# ---- must-fire self-tests (D42: never trust a gate that has not fired) ----

def test_gate_fires_on_stale_label() -> None:
    assert check_stale("Qwen & 0.36--0.59 & poor to moderate \\\\", interim=False)


def test_gate_fires_on_planned_front_matter() -> None:
    assert check_stale("Second judge & GPT-4o (architecturally independent; planned)", interim=True)


def test_gate_fires_on_desynced_table() -> None:
    good_js = THESIS_JS.read_text(encoding="utf-8")
    assert check_sync("tex without the kappa rows at all", good_js, "thesis")


def test_gate_quiet_on_clean_pair() -> None:
    if not THESIS_TEX.exists():
        pytest.skip("mirror absent")
    assert not check_stale(THESIS_TEX.read_text(encoding="utf-8"), interim=False)
