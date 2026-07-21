"""Usage→presence citation gate (D54): every NAMED statistical method must have
a bibliography entry in every document builder that names it.

Why this exists: the ieee_citations lock in verify_report_claims.py checks the
OPPOSITE direction (every reference entry has an in-text bracket, in strict
first-use order). A method named in prose with no bracket at all — the exact
defect of the uncited Benjamini-Hochberg procedure found on 2026-07-20, first
flagged by the cut "25th verifier" P3 on 2026-07-11 and then lost — is
invisible to that gate and to every enumerate-existing-citations audit. This
test enumerates from the METHODS NAMED IN THE PROSE inward: if a lexicon term
appears anywhere in a builder (or LaTeX mirror), the file's reference block
must contain the matching source. See PROJECT_LOG D54 (audit symmetry).

A FAIL means a named method has no reference: add the reference (and its
in-text citation, which ieee_citations then enforces) — never delete the
method name and never weaken the lexicon without a logged decision.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# method-name pattern (searched in the document body) -> required substring of
# the reference block. Patterns are chosen to survive the humanizer layer
# ("Benjamini, Hochberg") and LaTeX spellings ("Cohen's $\kappa$").
LEXICON = {
    r"Benjamini[-–, ]+\s*Hochberg": "Benjamini",
    r"McNemar": "McNemar",
    r"Wilcoxon": "Wilcoxon",
    r"Cohen[’']?s?\s+(?:\$\\kappa\$|κ|kappa)": "A coefficient of agreement",
    r"Cohen[’']?s?\s+d_?z?\b": "Statistical Power Analysis",
    r"[Bb]ootstrap": "Efron",
    r"Bonferroni": "Bonferroni",
    r"Clopper[-–]Pearson": "Clopper",
    r"Mann[-–]Whitney": "Mann",
    r"Fisher[’']?s? exact": "Fisher",
    r"Krippendorff": "Krippendorff",
    r"Landis\s*(?:and|&)\s*Koch": "Landis",
    # T44 FS-11: the MDE/power method must carry its source (Connor 1987). Keyed
    # on the procedure names, not "Cohen's d" (the blind spot that let the
    # thesis quote MDE values with no power reference in its bibliography).
    r"power analysis|minimum[- ]detectable": "Connor",
}

BUILDERS = [
    "scripts/build_fyp_report_v5.js",
    "scripts/build_fyp_report_humanized.js",
    "scripts/build_fyp_thesis_v4.js",
    "scripts/build_fyp_thesis_humanized.js",
    "scripts/build_fyp_interim.js",
    "scripts/build_fyp_interim_humanized.js",
]
TEX_MIRRORS = [
    "fyp_submission/report_latex/final_thesis/mythesis.tex",
    "fyp_submission/report_latex/interim_report/mythesis.tex",
]


def split_refs(text: str, path: str) -> tuple[str, str]:
    """Return (body, reference_block) for a builder or LaTeX mirror."""
    if path.endswith(".tex"):
        i0 = text.find(r"\begin{thebibliography}")
        i1 = text.find(r"\end{thebibliography}", i0)
        assert i0 != -1 and i1 != -1, f"{path}: no thebibliography block"
        return text[:i0] + text[i1:], text[i0:i1]
    i0 = text.find("const refs = [")
    i1 = text.find("];", i0)
    assert i0 != -1 and i1 != -1, f"{path}: no `const refs = [` block"
    return text[:i0] + text[i1:], text[i0:i1]


def uncited_methods(text: str, path: str) -> list[str]:
    body, refs = split_refs(text, path)
    missing = []
    for pattern, ref_needle in LEXICON.items():
        if re.search(pattern, body) and ref_needle not in refs:
            missing.append(f"{pattern!r} named but reference {ref_needle!r} absent")
    return missing


@pytest.mark.parametrize("rel", BUILDERS + TEX_MIRRORS)
def test_named_methods_are_cited(rel: str) -> None:
    path = REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present in this checkout")
    missing = uncited_methods(path.read_text(encoding="utf-8"), rel)
    assert not missing, f"{rel}: uncited named methods: " + "; ".join(missing)


# ---------------------------------------------------------------------------
# Must-fire fixtures: prove the gate actually detects the defect it claims to
# (harness rule: never trust a gate you have not watched fail).
# ---------------------------------------------------------------------------

def test_gate_fires_on_uncited_method() -> None:
    """The exact 2026-07-20 defect: BH named in prose, absent from refs."""
    fake = (
        'PJ("significance under a Benjamini-Hochberg correction"),\n'
        'const refs = [\n'
        '    "Q. McNemar, Psychometrika, 1947.",\n'
        '];\n'
    )
    missing = uncited_methods(fake, "fake_builder.js")
    assert any("Benjamini" in m for m in missing), (
        "gate failed to fire on an uncited Benjamini-Hochberg mention"
    )


def test_gate_fires_on_humanized_spelling() -> None:
    """The humanizer rewrites the en-dash: 'Benjamini, Hochberg' must still fire."""
    fake = (
        'PJ("under a Benjamini, Hochberg control over the family"),\n'
        'const refs = [\n'
        '];\n'
    )
    missing = uncited_methods(fake, "fake_builder.js")
    assert any("Benjamini" in m for m in missing)


def test_gate_fires_on_tex_mirror() -> None:
    fake = (
        "Cohen's $\\kappa$ is reported.\n"
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{ref1} Q. McNemar, 1947.\n"
        "\\end{thebibliography}\n"
    )
    missing = uncited_methods(fake, "fake.tex")
    assert any("coefficient of agreement" in m for m in missing)


def test_gate_quiet_when_reference_present() -> None:
    fake = (
        'PJ("under a Benjamini-Hochberg correction [23]"),\n'
        'const refs = [\n'
        '    "Y. Benjamini and Y. Hochberg, JRSS-B, 1995.",\n'
        '];\n'
    )
    assert uncited_methods(fake, "fake_builder.js") == []
