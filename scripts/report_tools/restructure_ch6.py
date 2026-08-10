#!/usr/bin/env python3
"""Restructure Chapter 6: move the RQ synthesis to the terminal position, fold the
135-word stub into the scale section, group the per-pair observations together, and
renumber every section and every inbound cross-reference in one atomic pass.

Aborts without writing if any cross-reference would fail to resolve afterwards.
"""
import re, sys

TARGET = "/Users/tanueihorng/fyp_quant/scripts/build_fyp_report_v6.js"

# old H2 number -> (new number, new heading level)  ("H3" demotes it under its parent)
REMAP = {
    "6.1": ("6.1", "H2"),
    "6.2": ("6.2", "H2"),
    "6.7": ("6.2.1", "H3"),   # 135-word stub folded under the scale section
    "6.3": ("6.3", "H2"),
    "6.4": ("6.4", "H2"),
    "6.5": ("6.5", "H2"),
    "6.6": ("6.6", "H2"),
    "6.8": ("6.7", "H2"),     # Qwen3-4B observations
    "6.10": ("6.8", "H2"),    # Llama observations, now adjacent to the other two
    "6.11": ("6.9", "H2"),    # cross-family comparison
    "6.12": ("6.10", "H2"),   # judge validation
    "6.13": ("6.11", "H2"),   # cross-family extension
    "6.14": ("6.12", "H2"),   # mechanism probe
    "6.15": ("6.13", "H2"),   # INT8 precision
    "6.16": ("6.14", "H2"),   # generation length
    "6.9": ("6.15", "H2"),    # RQ synthesis -> TERMINAL
}
# the order sections must appear in after the move
ORDER = ["6.1", "6.2", "6.7", "6.3", "6.4", "6.5", "6.6", "6.8", "6.10",
         "6.11", "6.12", "6.13", "6.14", "6.15", "6.16", "6.9"]

def main():
    lines = open(TARGET, encoding="utf-8").read().split("\n")
    a = next(i for i, l in enumerate(lines) if 'H1("Chapter 6: Results and Analysis"' in l)
    b = next(i for i in range(a, len(lines)) if lines[i].strip() == "];")

    # split the chapter into a preamble plus one block per H2
    heads = [(i, re.match(r'^\s*H2\("(\d+\.\d+)', lines[i]).group(1))
             for i in range(a, b) if re.match(r'^\s*H2\("\d+\.\d+', lines[i])]
    assert heads, "no H2 sections found"
    preamble = lines[a:heads[0][0]]
    blocks = {}
    for j, (i, num) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else b
        blocks[num] = lines[i:end]

    missing = [n for n in ORDER if n not in blocks]
    extra = [n for n in blocks if n not in ORDER]
    if missing or extra:
        print("ABORT: section set mismatch. missing=%s extra=%s" % (missing, extra)); sys.exit(1)

    # rebuild in the new order, renumbering headings (and their H3 children)
    out = list(preamble)
    for old in ORDER:
        new, level = REMAP[old]
        blk = list(blocks[old])
        # the H2 heading line itself
        m = re.match(r'^(\s*)H2\("(%s)\s+(.*)$' % re.escape(old), blk[0])
        assert m, "heading parse failed for %s: %s" % (old, blk[0][:80])
        indent, rest = m.group(1), m.group(3)
        blk[0] = '%s%s("%s %s' % (indent, level, new, rest)
        # child H3 headings: 6.11.2 -> 6.9.2
        for k in range(1, len(blk)):
            blk[k] = re.sub(r'(H3\(")%s\.(\d+)' % re.escape(old),
                            lambda mm, nn=new: mm.group(1) + nn + "." + mm.group(2), blk[k])
        out.extend(blk)

    lines[a:b] = out
    s = "\n".join(lines)

    # --- update every inbound cross-reference, longest-first to avoid prefix collisions ---
    submap = {}
    for old, (new, _) in REMAP.items():
        submap[old] = new
        for child in range(1, 6):
            submap["%s.%d" % (old, child)] = "%s.%d" % (new, child)
    keys = sorted(submap, key=len, reverse=True)
    pat = re.compile(r'(Section|Sections)\s+(' + "|".join(re.escape(k) for k in keys) + r')\b')
    n_refs = len(pat.findall(s))
    s = pat.sub(lambda m: "%s %s" % (m.group(1), submap[m.group(2)]), s)

    # --- verify: every "Section 6.x" now resolves to a heading that exists ---
    headings = set(re.findall(r'H[23]\("(\d+\.\d+(?:\.\d+)?)', s))
    refs = set(re.findall(r'Sections?\s+(6\.\d+(?:\.\d+)?)', s))
    dangling = sorted(refs - headings)
    if dangling:
        print("ABORT: these cross-references would not resolve: %s" % dangling)
        print("       (nothing written)"); sys.exit(1)

    open(TARGET, "w", encoding="utf-8").write(s)
    print("Chapter 6 restructured.")
    print("  sections reordered: %d" % len(ORDER))
    print("  cross-references rewritten: %d" % n_refs)
    print("  every 'Section 6.x' reference resolves to an existing heading: YES")
    print("\n  key moves:")
    print("    RQ synthesis  6.9  -> 6.15  (now the LAST section, nothing follows it)")
    print("    135-word stub 6.7  -> 6.2.1 (folded under the scale section it belonged to)")
    print("    Llama obs     6.10 -> 6.8   (now adjacent to the other two per-pair sections)")

if __name__ == "__main__":
    main()
