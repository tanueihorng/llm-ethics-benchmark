#!/usr/bin/env python3
"""Resolve [[key]] citation placeholders into IEEE numbers and reorder the refs array
so numbering is strictly by first use, which is what the ieee_citations lock requires.

Existing numeric [n] citations are first converted to [[__old_n]] keys so that old and
new references are ordered together by their true first appearance in the document.
"""
import re, sys, json

TARGET = "/Users/tanueihorng/fyp_quant/scripts/build_fyp_report_v6.js"
W = "/private/tmp/claude-501/-Users-tanueihorng-fyp-quant/5b90599c-4ee9-48ca-9a73-9afd0d327912/scratchpad/fyp_study"

def main():
    s = open(TARGET, encoding="utf-8").read()
    i0 = s.find("const refs = [")
    i1 = s.find("];", i0)
    assert i0 > 0 and i1 > i0, "refs array not found"
    block = s[i0:i1]
    entries = re.findall(r'\n    "((?:[^"\\]|\\.)+)",', block)
    n_old = len(entries)

    # new refs keyed for placeholder use
    newrefs = {r["key"]: r["ieee"] for r in json.load(open(f"{W}/meta/new_refs.json"))}

    body_before, body_after = s[:i0], s[i1:]

    # 1. numeric [n] -> [[__old_n]] in the body (outside the refs array)
    def tokenise(txt):
        return re.sub(r'\[(\d{1,3})\]', lambda m: "[[__old_%s]]" % m.group(1), txt)
    body_before, body_after = tokenise(body_before), tokenise(body_after)

    # 2. walk the document in order, assign numbers by first use
    doc = body_before + " REFS " + body_after
    order, seen = [], set()
    for m in re.finditer(r'\[\[([A-Za-z0-9_]+)\]\]', doc):
        k = m.group(1)
        if k not in seen:
            seen.add(k); order.append(k)

    # 3. build the new reference list in that order
    out_entries, unknown = [], []
    for k in order:
        if k.startswith("__old_"):
            idx = int(k[6:])
            if not (1 <= idx <= n_old):
                unknown.append(k); continue
            out_entries.append(entries[idx - 1])
        elif k in newrefs:
            out_entries.append(newrefs[k])
        else:
            unknown.append(k)
    if unknown:
        print("ABORT: unknown citation keys: %s" % sorted(set(unknown))); sys.exit(1)

    # every old reference must still be cited somewhere
    cited_old = {int(k[6:]) for k in order if k.startswith("__old_")}
    dropped = [i for i in range(1, n_old + 1) if i not in cited_old]
    if dropped:
        print("ABORT: these existing references are no longer cited: %s" % dropped); sys.exit(1)

    num = {k: i + 1 for i, k in enumerate(order)}
    doc = re.sub(r'\[\[([A-Za-z0-9_]+)\]\]', lambda m: "[%d]" % num[m.group(1)], doc)

    # CRITICAL: the refs array is not a list of strings. It is a list of docx-js
    # OBJECTS — H1("References"), the intro P(...), and the entries wrapped in
    # ...numberedList([...]). An earlier version of this script rebuilt the block
    # as bare strings, which docx-js silently ignores: the document then builds
    # cleanly, passes every source-level citation gate (they parse this array,
    # not the rendered file), and ships WITH NO BIBLIOGRAPHY AT ALL. That shipped
    # in every v6 build until 2026-07-26. Preserve the wrapper, and assert it.
    new_block = (
        'const refs = [\n'
        '  H1("References"),\n'
        '  P("References are cited in IEEE numbered style; the bracketed numbers '
        'below correspond to the in-text citations.", { after: 200 }),\n'
        '  ...numberedList([\n'
        + "".join('    "%s",\n' % e for e in out_entries)
        + '  ])\n'
    )
    for token in ('H1("References")', "...numberedList([", "])"):
        assert token in new_block, f"refs block lost {token!r}"
    before, after = doc.split(" REFS ")
    # NOTE: `before` is s[:i0] and therefore does NOT contain "const refs = [".
    # An earlier version called before.find(...) here, got -1, and silently chopped the
    # final newline, merging a comment line into the array declaration. Concatenate directly.
    open(TARGET, "w", encoding="utf-8").write(before + new_block + after)

    print("references: %d -> %d  (%d newly added)" % (n_old, len(out_entries), len(out_entries) - n_old))
    print("numbering: strictly by first use, verified over %d distinct keys" % len(order))
    added = [k for k in order if not k.startswith("__old_")]
    if added:
        print("newly cited, in first-use order:")
        for k in added:
            print("   [%d] %s" % (num[k], k))

if __name__ == "__main__":
    main()
