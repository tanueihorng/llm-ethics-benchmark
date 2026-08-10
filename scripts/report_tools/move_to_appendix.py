#!/usr/bin/env python3
"""Move whole blocks out of Chapter 6 into a new supporting-detail appendix.

Paraphrasing Chapter 6 recovers ~30 words a paragraph; the chapter is 9 000
words over target, so the lever is relocation, not rewording. Every moved block
keeps its exact wording — nothing is re-derived, so no new claim can be
introduced by the move — and Chapter 6 keeps a pointer in its place.

Safety: the compression gates run on the RENDERED document, so a number that
moves from Chapter 6 into the appendix is still present and passes G3. A number
that is genuinely lost still aborts the run and restores the builder.
"""
import re
import sys

import compress_ch6 as C

TARGET = C.TARGET
APPENDIX_CONST = "appendixI"
APPENDIX_TITLE = "Appendix I: Supporting Detail for the Results Chapter"


def _find_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    raise SystemExit(f"ABORT: anchor not found: {needle[:80]!r}")


def move_blocks(blocks):
    """blocks: list of (start_needle, end_needle, appendix_heading, pointer_text).

    The block is lines[start .. end) — end_needle marks the FIRST line that
    stays behind. Blocks must be listed in document order.
    """
    src = open(TARGET, encoding="utf-8").read()
    lines = src.split("\n")

    moved_sections = []
    # Work bottom-up so earlier indices stay valid as we splice.
    resolved = []
    for start_needle, end_needle, heading, pointer in blocks:
        a = _find_line(lines, start_needle)
        b = _find_line(lines, end_needle, a + 1)
        resolved.append((a, b, heading, pointer))
    resolved.sort(key=lambda r: r[0])
    for i in range(1, len(resolved)):
        if resolved[i][0] < resolved[i - 1][1]:
            raise SystemExit("ABORT: blocks overlap")

    for a, b, heading, pointer in reversed(resolved):
        block = lines[a:b]
        moved_sections.insert(0, (heading, block))
        lines[a:b] = [f'  PJ({pointer}),'] if pointer else []

    # Build the appendix
    out = [f"const {APPENDIX_CONST} = [", f'  H1("{APPENDIX_TITLE}"),',
           '  PJ("This appendix holds the supporting detail for Chapter 6: the '
           'full scorer-validation trail, the robustness arms, and the '
           'per-category and per-subject breakdowns. It is placed here so the '
           'results chapter carries the findings and the evidence for them stays '
           'available and unabridged. Nothing in this appendix is re-derived; '
           'each block is the original text, relocated."),']
    for heading, block in moved_sections:
        out.append(f'  H2("{heading}"),')
        out.extend(block)
    out.append("];")
    out.append("")

    text = "\n".join(lines)
    if f"const {APPENDIX_CONST}" in text:
        # Appendix I already exists: append these sections to it instead of
        # creating a second one. Later passes extend the same appendix.
        end = text.index("];", text.index(f"const {APPENDIX_CONST} = ["))
        addition = "\n".join(out[2:-2])  # drop the const/H1/intro and the closing ];
        text = text[:end] + addition + "\n" + text[end:]
        return src, text
    marker = "const appendixH = ["
    i = text.index(marker)
    text = text[:i] + "\n".join(out) + "\n" + text[i:]
    text = text.replace("  ...appendixH,", "  ...appendixH,\n  ...appendixI,", 1)
    return src, text


def run(blocks, allowed_removals, label):
    before, after = move_blocks(blocks)
    # Relocating a block legitimately changes which reference is cited first, so
    # G5 is deferred: apply, renumber by first use, then verify the order really
    # is monotone. Blocking the move instead would make the appendix impossible.
    C.run([("block-move", 1, before, after)], allowed_removals, label=label,
          allow_reorder=True)
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "renumber_refs.py"], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[0] if r.stdout else r.stderr[:200])
    text = open(TARGET, encoding="utf-8").read()
    order = C.citation_order(text)
    assert order == sorted(order), f"first-use order still broken after renumber: {order[:60]}"
    n = len(C.refs_of(text))
    assert set(order) == set(range(1, n + 1)), "reference set/citation set mismatch"
    print(f"  citations renumbered: {n} references, strict first-use order verified")
