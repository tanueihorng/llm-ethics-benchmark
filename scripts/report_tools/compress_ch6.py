#!/usr/bin/env python3
"""Chapter 6 compression (T49) with a declared-removals gate.

The audit-fix applier's G3 rule was "no number may disappear from the document".
That is exactly right for a correctness pass and exactly wrong for a compression
pass, where the whole point is to remove numeric narration that a table already
carries — plus a small amount of maintenance detail (SLURM job ids, run times)
that should never have been in the narrative.

So G3 becomes: every removed number must be either
  (a) still present elsewhere in the document (moved into a table, or restated),
      which the checker verifies mechanically; or
  (b) named in ALLOWED_REMOVALS with a written reason.
Anything else aborts the run. That keeps the no-loss invariant real while
letting a deliberate cut through.

Usage: import EDITS/ALLOWED_REMOVALS from a pass module and call run().
"""
import collections
import importlib.util
import re
import sys

_spec = importlib.util.spec_from_file_location("_base", "apply_audit_fixes.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

TARGET = _base.TARGET
body = _base.body
mask = _base.mask
numbers = _base.numbers
models = _base.models
model_identities = _base.model_identities
refs_of = _base.refs_of
citation_order = _base.citation_order
VALID_MODELS = _base.VALID_MODELS


DOCX = "/Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-25_v6.docx"


def _rendered_numbers():
    """Numbers in the BUILT document, tables included.

    The builder-source check alone is blind to every registry-driven table:
    Table 6.1's values live in CLAIMS.render at runtime, not as literals in the
    .js, so moving a number from prose into that table looks to a source-level
    gate exactly like deleting it. (It flagged Llama's NF4 MMLU 0.573 as lost
    when the table had just started carrying it.) Same lesson as the missing
    bibliography: check the artifact the reader opens.
    """
    import subprocess
    import zipfile
    from xml.etree import ElementTree

    subprocess.run(
        ["node", "/Users/tanueihorng/fyp_quant/scripts/build_fyp_report_v6.js"],
        capture_output=True, check=True,
        cwd="/Users/tanueihorng/fyp_quant",
    )
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(DOCX) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    text = "\n".join("".join(p.itertext()) for p in root.iter())
    return collections.Counter(re.findall(r"\d+(?:\.\d+)?", mask(text)))


def words_in_chapter6(text: str) -> int:
    """Rough prose word count for Chapter 6, from the builder source."""
    lines = text.split("\n")
    try:
        a = next(i for i, l in enumerate(lines) if 'H1("Chapter 6' in l)
        b = next(i for i, l in enumerate(lines) if 'H1("Chapter 7' in l)
    except StopIteration:
        return -1
    chunk = "\n".join(lines[a:b])
    return len(" ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)).split())


def run(edits, allowed_removals, label="compression pass", allow_reorder=False):
    before = open(TARGET, encoding="utf-8").read()
    after = before
    problems = []

    for fid, want, old, new in edits:
        got = after.count(old)
        if got != want:
            problems.append(f"{fid}: expected {want} occurrence(s), found {got}")
            continue
        after = after.replace(old, new)
    if problems:
        print("ABORT (G1 anchor mismatch), nothing written:")
        for p in problems:
            print("   " + p)
        sys.exit(1)

    fam_b, fam_a = models(before), models(after)
    ident_b, ident_a = model_identities(before), model_identities(after)
    dropped_fam = fam_b - fam_a
    invented = (ident_a - ident_b) - VALID_MODELS
    if dropped_fam or invented:
        print("ABORT (G2 model drift), nothing written:")
        if dropped_fam:
            print(f"   families no longer named: {sorted(dropped_fam)}")
        if invented:
            print(f"   invented model identities: {sorted(invented)}")
        sys.exit(1)

    # G3 is evaluated on the RENDERED document, so a number moved from prose into
    # a registry-driven table counts as kept, not lost. The builder is written,
    # built, measured, and restored if the gate fails — nothing survives an abort.
    rendered_before = _rendered_numbers()
    open(TARGET, "w", encoding="utf-8").write(after)
    try:
        rendered_after = _rendered_numbers()
    except Exception:
        open(TARGET, "w", encoding="utf-8").write(before)
        _rendered_numbers()
        raise
    vanished = {n for n in (rendered_before - rendered_after) if rendered_after[n] == 0}
    reduced = {n for n in (rendered_before - rendered_after) if rendered_after[n] > 0}
    undeclared = sorted(vanished - set(allowed_removals))
    if undeclared:
        open(TARGET, "w", encoding="utf-8").write(before)
        _rendered_numbers()
        print("ABORT (G3 undeclared number loss), builder and docx restored:")
        for n in undeclared:
            print(f"   {n!r} no longer appears anywhere in the RENDERED document")
        print("   Add it to ALLOWED_REMOVALS with a reason, or keep it in a table.")
        sys.exit(1)

    # BUG FIXED 2026-07-26: G4/G5 ran AFTER the write and printed "nothing
    # written", which was false — a G5 abort on the first Appendix-I move left
    # the relocation applied with a broken citation order. Aborts here must
    # restore the builder and rebuild, exactly as the G3 branch does.
    def _abort(msg):
        open(TARGET, "w", encoding="utf-8").write(before)
        _rendered_numbers()
        print(msg + ", builder and docx restored")
        sys.exit(1)

    n_refs = len(refs_of(after))
    cited = set(citation_order(after))
    orphans = [n for n in range(1, n_refs + 1) if n not in cited]
    if orphans:
        _abort(f"ABORT (G4 orphaned references {orphans})")

    order = citation_order(after)
    if order != sorted(order) and not allow_reorder:
        bad = [(i, order[i]) for i in range(1, len(order)) if order[i] < order[i - 1]]
        _abort(f"ABORT (G5 first-use order broken at {bad[:5]})")

    w_before, w_after = words_in_chapter6(before), words_in_chapter6(after)
    print(f"{label}: applied {len(edits)} edits, all gates passed")
    print(f"  Chapter 6 prose: {w_before} -> {w_after} words ({w_after - w_before:+d})")
    print(f"  G3 numbers dropped entirely: {len(vanished)} (all declared)")
    for n in sorted(vanished):
        print(f"       {n:<10} {allowed_removals[n]}")
    print(f"  G3 numbers with fewer mentions but still present: {len(reduced)}")
    print(f"  G4 references: {n_refs}, all cited;  G5 first-use order intact")
