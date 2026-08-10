"""Claim-to-source fitness gates (T48 / D56).

Context. On 2026-07-26 an independent claim-to-source audit found 26 defects in
the report — every reference real, correctly formatted and genuinely about its
topic, but 26 sentences that stretched past what the cited paper shows or cited
nothing at all. Nineteen were inherited from v5, the report of record, which was
under all 93 numeric claim locks and had passed six prior audits.

The locks could not see any of it. A lock asserts "this number appears in this
file"; it can never assert "this source entails this sentence". Neither can the
citation-completeness gate, which enumerates from the reference list outward.

Fitness itself is not machine-checkable. Two things about it are, and both are
gated here:

  1. Evidence grade. A bibliography entry with no archival venue is a preprint,
     and a reader must be able to tell. Codex's audit flagged one such entry;
     a reverse pass over the whole bibliography found twelve.

  2. The LANGUAGE of over-claiming. The `epistemic_hedge` scan in
     configs/artifact_policy.yaml triggers review on the phrasings that produced
     the audit's findings. It is a review trigger, not a vocabulary ban: a hit is
     cleared by rewording or by an allowlist entry that says why the line is
     legitimate. These tests prove the scan actually fires, because a gate no one
     has watched fail is decoration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ethical_benchmark.harness.agent import _scan_policy_group, load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_BUILDERS = [
    "scripts/build_fyp_report_v6.js",
]

# An entry naming any of these is archivally published. Anything else in the
# bibliography is a preprint, a technical report, or a book.
VENUE_MARKERS = re.compile(
    r"Proc\.|Proceedings|Transactions|Journal|IEEE|ACM|Findings of|vol\.\s|"
    r"NeurIPS|ICML|ICLR|ACL|EMNLP|NAACL|MLSys|Conference on",
    re.IGNORECASE,
)
# Corporate model cards and technical reports are the correct citation for a
# model release — no venue exists for them — and they identify themselves by
# organisation. They are not research preprints and need no preprint label.
TECH_REPORT_MARKERS = re.compile(
    r"technical report|Alibaba Group|Meta AI|Microsoft|Google DeepMind|"
    r"Mistral AI|Gemma Team",
    re.IGNORECASE,
)
# Books carry a publisher instead of a venue.
BOOK_MARKERS = re.compile(r"Chapman & Hall|Erlbaum|Springer|Wiley|MIT Press", re.IGNORECASE)


def reference_entries(text: str) -> list[str]:
    i0 = text.find("const refs = [")
    i1 = text.find("];", i0)
    assert i0 != -1 and i1 != -1, "no `const refs = [` block"
    return re.findall(r'\n    "((?:[^"\\]|\\.)+)",', text[i0:i1])


def unlabelled_preprints(entries: list[str]) -> list[str]:
    """Entries with no venue, no tech-report identity, not a book, and no label."""
    bad = []
    for n, entry in enumerate(entries, start=1):
        if VENUE_MARKERS.search(entry) or BOOK_MARKERS.search(entry):
            continue
        if TECH_REPORT_MARKERS.search(entry):
            continue
        if "preprint" in entry.lower():
            continue
        bad.append(f"[{n}] {entry[:90]}")
    return bad


@pytest.mark.parametrize("rel", CANONICAL_BUILDERS)
def test_preprints_are_labelled(rel: str) -> None:
    """A non-peer-reviewed source must announce itself as one.

    Fix a failure by adding '(preprint)' to the entry — never by deleting the
    reference, and never by inventing a venue it does not have.
    """
    path = REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present in this checkout")
    bad = unlabelled_preprints(reference_entries(path.read_text(encoding="utf-8")))
    assert not bad, (
        f"{rel}: bibliography entries with no archival venue and no preprint label:\n  "
        + "\n  ".join(bad)
    )


def test_preprint_gate_fires_on_unlabelled_entry() -> None:
    """Must-fire: strip the label from a real preprint and the gate must catch it."""
    entries = reference_entries(
        (REPO_ROOT / "scripts/build_fyp_report_v6.js").read_text(encoding="utf-8")
    )
    mutated = [e.replace(" (preprint)", "") for e in entries]
    assert unlabelled_preprints(mutated), (
        "removing every preprint label produced no finding — the gate is not "
        "actually testing evidence grade"
    )


def test_preprint_gate_quiet_on_peer_reviewed_entry() -> None:
    """Control: a properly published entry must not be flagged as a preprint."""
    published = [
        "D. J. Schuirmann, “A comparison of the two one-sided tests procedure,” "
        "Journal of Pharmacokinetics and Biopharmaceutics, vol. 15, no. 6, 1987.",
    ]
    assert unlabelled_preprints(published) == []


# ---------------------------------------------------------------------------
# epistemic_hedge scan
# ---------------------------------------------------------------------------

def test_hedge_scan_is_configured_over_the_canonical_builder() -> None:
    policy = load_policy(REPO_ROOT)
    group = policy.get("epistemic_hedge")
    assert group, "epistemic_hedge group missing from configs/artifact_policy.yaml"
    assert "scripts/build_fyp_report_v6.js" in group["scan_paths"], (
        "the canonical report builder must be hedge-scanned; when a builder is "
        "promoted, swap it into epistemic_hedge.scan_paths in the same edit"
    )
    assert "scripts/build_rewritten_fyp_report_v1.py" in group["scan_paths"], (
        "the live corpus-informed rewrite must remain inside the claim-fitness scan"
    )


def test_hedge_scan_is_clean_on_the_real_repo() -> None:
    policy = load_policy(REPO_ROOT)
    result = _scan_policy_group(REPO_ROOT, policy["epistemic_hedge"], "epistemic-hedge")
    assert result.status == "pass", (
        "epistemic-hedge scan is dirty: " + "; ".join(result.details or [])
    )


@pytest.mark.parametrize(
    "line,finding",
    [
        ("Within the study's resolution, INT8 is therefore capability-lossless.", "C14"),
        ("re-scored by a second, architecturally unrelated frontier judge.", "C07"),
        ("a fully independent open-weight guard is the natural cross-check.", "C08"),
        ("the matched-pair deltas are unaffected by the template choice.", "C16"),
        ("The one prior study that isolates quantization at four bits.", "C21"),
        ("it provides the strongest practical isolation of the variable.", "C23"),
        ("This is what the mechanistic literature predicts.", "C09"),
        ("That headroom is the mechanistic reason the pair is intact.", "C10"),
        ("four-bit has become the de facto compression standard.", "C11"),
        ("it does have enough power to resolve the capability losses.", "C15"),
        ("there is no community-standard instrument for measuring it.", "C25"),
        ("the null result rules out any safety regression.", "H01"),
    ],
)
def test_hedge_scan_fires_on_each_audited_defect(tmp_path: Path, line: str, finding: str) -> None:
    """Every pattern traces to a real 2026-07-26 audit finding and must fire on it."""
    policy = load_policy(REPO_ROOT)
    (tmp_path / "README.md").write_text(line + "\n", encoding="utf-8")
    group = {"scan_paths": ["README.md"], "patterns": policy["epistemic_hedge"]["patterns"]}
    result = _scan_policy_group(tmp_path, group, "epistemic-hedge")
    assert result.status == "fail", f"{finding}: hedge scan did not fire on {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        # The corrected forms must NOT fire, or the gate would block its own fix.
        "INT8 capability is statistically equivalent to full precision within ±0.05.",
        "a second frontier judge from a different provider (gpt-4o).",
        "an open-weight third judge applying a different safety-classification construct.",
        "the matched-pair deltas are internally controlled for it.",
        "The closest prior comparison in trustworthiness scope is Hong et al.",
        "it provides controlled isolation of the quantization step.",
        "This is what an extrapolation from the mechanistic literature would suggest.",
    ],
)
def test_hedge_scan_accepts_the_corrected_wording(tmp_path: Path, line: str) -> None:
    policy = load_policy(REPO_ROOT)
    (tmp_path / "README.md").write_text(line + "\n", encoding="utf-8")
    group = {"scan_paths": ["README.md"], "patterns": policy["epistemic_hedge"]["patterns"]}
    result = _scan_policy_group(tmp_path, group, "epistemic-hedge")
    assert result.status == "pass", (
        f"hedge scan fires on the CORRECTED wording {line!r}: "
        + "; ".join(result.details or [])
    )


def test_allowlist_must_name_the_offending_span(tmp_path: Path) -> None:
    """An allow entry exempts the span it matches, not the line it sits on.

    This test previously asserted the opposite — that a nearby context marker
    exempted its whole line — which is exactly the behaviour that let an
    unearned superlative survive on report line 521 while the gate read clean.
    An exemption must now quote the phrase being exempted.
    """
    patterns = [r"\bfully independent\b"]
    # A nearby marker does NOT exempt anything.
    group = {"scan_paths": ["README.md"], "patterns": patterns,
             "allow": [r"legitimate context marker"]}
    (tmp_path / "README.md").write_text(
        "a fully independent judge, legitimate context marker\n", encoding="utf-8"
    )
    assert _scan_policy_group(tmp_path, group, "hedge").status == "fail", (
        "a context marker exempted a violation it does not overlap"
    )
    # Naming the offending span DOES exempt it.
    group["allow"] = [r"a fully independent judge, legitimate context marker"]
    assert _scan_policy_group(tmp_path, group, "hedge").status == "pass"
    # ...and only it.
    (tmp_path / "README.md").write_text("a fully independent judge\n", encoding="utf-8")
    assert _scan_policy_group(tmp_path, group, "hedge").status == "fail"


# ---------------------------------------------------------------------------
# Rendered-output gate
# ---------------------------------------------------------------------------
# Every citation gate in this repo parses the BUILDER. None of them looked at
# the built document. On 2026-07-26 a helper script rebuilt the refs array as
# bare strings instead of docx-js objects, dropping H1("References") and the
# numberedList wrapper. docx-js ignores non-paragraph entries without error, so
# the report built cleanly, passed the IEEE first-use lock, passed
# test_citation_completeness, and shipped with NO BIBLIOGRAPHY AT ALL — in every
# v6 build for a full session. Source-level correctness is not rendered
# correctness; this gate reads the artifact the examiner opens.

DOCX_FOR_BUILDER = {
    "scripts/build_fyp_report_v5.js": "docs/FYP_Report_2026-07-01_v5.docx",
    "scripts/build_fyp_report_v6.js": "docs/FYP_Report_2026-07-25_v6.docx",
}


def docx_text(path: Path) -> str:
    import zipfile
    from xml.etree import ElementTree

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n".join("".join(p.itertext()) for p in root.findall(".//w:p", ns))


@pytest.mark.parametrize("builder,docx", sorted(DOCX_FOR_BUILDER.items()))
def test_every_reference_renders_into_the_document(builder: str, docx: str) -> None:
    builder_path, docx_path = REPO_ROOT / builder, REPO_ROOT / docx
    if not builder_path.exists() or not docx_path.exists():
        pytest.skip(f"{builder} or {docx} not present in this checkout")

    entries = reference_entries(builder_path.read_text(encoding="utf-8"))
    rendered = docx_text(docx_path)

    assert "References" in rendered, (
        f"{docx}: no References heading in the RENDERED document — the refs array "
        f"is probably a list of bare strings instead of docx-js objects "
        f"(H1(\"References\") + ...numberedList([...])), which docx-js drops silently"
    )
    # Match on a distinctive slice of each entry: the author surname plus the
    # opening of the title, which survives Word's run-splitting.
    missing = []
    for n, entry in enumerate(entries, start=1):
        needle = entry.split(",")[0].strip()
        if len(needle) >= 4 and needle not in rendered:
            missing.append(f"[{n}] {needle}")
    assert not missing, (
        f"{docx}: {len(missing)} of {len(entries)} bibliography entries do not "
        f"appear in the rendered document: " + ", ".join(missing[:6])
    )


def test_allow_suppresses_only_the_matched_span_not_the_line(tmp_path: Path) -> None:
    """Regression: line-level allow hid a real defect.

    Report line 521 carried BOTH an allowlisted phrase ("the strongest
    internal-validity property of the study") and an unearned superlative ("the
    strongest practical isolation available"). Because `allow` suppressed the
    whole line, epistemic-hedge reported clean while the superlative survived —
    found only by an independent re-audit. Allow now blanks the matched span and
    scans the remainder, so a second violation on the same line still fires.
    """
    group = {
        "scan_paths": ["README.md"],
        "patterns": [r"\bstrongest practical\b", r"\bthe strongest\b"],
        "allow": [r"the strongest internal-validity property of the study"],
    }
    # Allowed phrase alone -> clean.
    (tmp_path / "README.md").write_text(
        "This is the strongest internal-validity property of the study.\n", encoding="utf-8"
    )
    assert _scan_policy_group(tmp_path, group, "hedge").status == "pass"

    # Allowed phrase AND a separate violation on the SAME line -> must fire.
    (tmp_path / "README.md").write_text(
        "This is the strongest internal-validity property of the study. "
        "It is also the strongest practical isolation available.\n",
        encoding="utf-8",
    )
    result = _scan_policy_group(tmp_path, group, "hedge")
    assert result.status == "fail", (
        "an allowlisted phrase suppressed a different violation sharing its line"
    )


@pytest.mark.parametrize(
    "line,finding",
    [
        ("such models are rarely deployed in full precision.", "RA1 prevalence"),
        ("they are almost never used in full precision.", "RA2 prevalence"),
        ("drive routine use of four-bit quantization.", "RA2 prevalence"),
        ("The deployment reality of these models is stark.", "RA2 prevalence"),
        ("the strongest practical isolation available under the configuration.", "RA3 superlative"),
        ("the most widely adopted harmful-behaviour collection in safety research.", "RA4 superlative"),
        ("Eight-bit is effectively a free precision point on capability.", "RA5 INT8"),
        ("no capability delta appears at the INT8 precision point.", "RA6 INT8"),
        ("HarmBench was evaluated in its standard configuration, which presents them directly.", "RA7 protocol"),
        ("a difference on the safety axis at equivalent capability.", "RA8 stale claim"),
        ("different calibrations, attributable to differences in RLHF recipe.", "RA9 causal"),
    ],
)
def test_hedge_scan_fires_on_each_reaudit_defect(tmp_path: Path, line: str, finding: str) -> None:
    """The second audit found nine defects the first pattern set missed. Each
    widened pattern must fire on the wording that got past it."""
    policy = load_policy(REPO_ROOT)
    (tmp_path / "README.md").write_text(line + "\n", encoding="utf-8")
    group = {"scan_paths": ["README.md"], "patterns": policy["epistemic_hedge"]["patterns"]}
    assert _scan_policy_group(tmp_path, group, "hedge").status == "fail", (
        f"{finding}: widened hedge scan did not fire on {line!r}"
    )
