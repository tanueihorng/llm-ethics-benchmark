"""Extract every sentence of a rendered .docx with its chapter/section context.

The enumeration instrument behind the T51 audit (docs/audits/2026-08-13_t51_enumeration/).
Usage: python scripts/report_tools/enumerate_claims.py <docx> [out.json]

Enumeration basis: the RENDERED docx (D56 -- the builder is not the artifact),
body paragraphs AND table cells AND captions. Nothing is sampled.
"""
import zipfile, re, json, sys
from pathlib import Path

DOC = sys.argv[1] if len(sys.argv) > 1 else 'docs/rewritten_fyp_report_v1.docx'
x = zipfile.ZipFile(DOC).read('word/document.xml').decode('utf8')

def textof(blk):
    t = ''.join(re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>', blk, re.S))
    for a, b in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&apos;', "'")):
        t = t.replace(a, b)
    return t

# Walk paragraphs in document order, tracking heading context and whether we are
# inside a table (w:tbl) -- a table cell's paragraphs are still w:p.
units = []
chapter = section = '(front matter)'
pos = 0
tbl_spans = [(m.start(), m.end()) for m in re.finditer(r'<w:tbl[ >].*?</w:tbl>', x, re.S)]
def in_table(i):
    return any(s <= i < e for s, e in tbl_spans)

for m in re.finditer(r'<w:p[ >].*?</w:p>', x, re.S):
    blk, i = m.group(0), m.start()
    txt = textof(blk).strip()
    if not txt:
        continue
    h = re.search(r'w:val="(Heading\d)"', blk)
    if h and not in_table(i):
        if h.group(1) == 'Heading1':
            chapter, section = txt, txt
        else:
            section = txt
        continue
    units.append({'chapter': chapter, 'section': section, 'table': in_table(i), 'text': txt})

# Sentence split that does not break on decimals, abbreviations, or bracket cites.
ABBR = r'(?<!\bFig)(?<!\bTab)(?<!\bEq)(?<!\be\.g)(?<!\bi\.e)(?<!\bcf)(?<!\bvs)(?<!\bDr)(?<!\bal)(?<!\bNo)(?<!\bp)'
SPLIT = re.compile(ABBR + r'(?<![A-Z])(?<!\d)\.\s+(?=[A-Z(“])')
rows = []
for u in units:
    parts = [u['text']] if u['table'] else SPLIT.split(u['text'])
    for p in parts:
        p = p.strip()
        if len(p) < 3:
            continue
        rows.append({**u, 'sentence': p})

Path(sys.argv[2] if len(sys.argv) > 2 else 'sentences.json').write_text(json.dumps(rows, indent=1))
print(f"{DOC}: {len(units)} units -> {len(rows)} sentences ({sum(1 for r in rows if r['table'])} in tables)")
