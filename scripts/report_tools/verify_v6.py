#!/usr/bin/env python3
"""Verify the rewritten v6 docx against v5: content preserved, voice metrics improved."""
import zipfile, re, sys, json, statistics as st, collections

V5 = "/Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-01_v5.docx"
V6 = "/Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-25_v6.docx"
W = "/private/tmp/claude-501/-Users-tanueihorng-fyp-quant/5b90599c-4ee9-48ca-9a73-9afd0d327912/scratchpad/fyp_study"

def paras(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    out = []
    for pr in re.findall(r'<w:p[ >].*?</w:p>', x, re.S):
        style = re.search(r'w:pStyle w:val="([^"]+)"', pr)
        t = "".join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', pr, re.S))
        t = (t.replace("&amp;", "&").replace("&quot;", '"')
              .replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">"))
        if t.strip():
            out.append((style.group(1) if style else "", t.strip()))
    return out

def body_prose(ps):
    """Body chapters only, excluding headings, references, appendices."""
    chap, keep = "front", []
    for s, t in ps:
        if s == "Heading1":
            chap = t
        if s.startswith("Heading"):
            continue
        if not chap.startswith("Chapter"):
            continue
        keep.append(t)
    return keep

def sents(txt):
    t = re.sub(r'\s+', ' ', txt)
    t = re.sub(r'(?i)\b(e\.g|i\.e|et al|vs|Dr|Prof|Fig|Eq|approx|cf|No)\.', r'\1', t)
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z(“"])', t)
            if len(p.strip()) > 25 and len(p.split()) >= 5]

CONN = r'(?i)^(However|Therefore|Moreover|Furthermore|In addition|Although|Because|By contrast|Consequently|Nevertheless|Thus|Hence|First|Second|Third|Finally|In practice|For this reason|Instead|Conversely|Similarly|Accordingly|That is|In contrast)\b'

def metrics(path, label):
    ps = paras(path)
    bp = body_prose(ps)
    txt = "\n".join(bp)
    w = len(txt.split())
    S = sents(txt)
    lens = [len(s.split()) for s in S]
    plens = [len(p.split()) for p in bp]
    m = dict(
        label=label, body_words=w, paragraphs=len(bp),
        para_mean=round(st.mean(plens), 1),
        para_p90=round(st.quantiles(plens, n=10)[8], 1) if len(plens) > 10 else 0,
        para_over200=sum(1 for x in plens if x > 200),
        sent_mean=round(st.mean(lens), 1) if lens else 0,
        semi_k=round(1000 * txt.count(';') / w, 2),
        emdash_total=txt.count('—'),
        emdash_k=round(1000 * txt.count('—') / w, 2),
        paren_k=round(1000 * len(re.findall(r'\([^)]{8,}\)', txt)) / w, 2),
        notY=len(re.findall(r'(?<=\w),\s+not\s+(?!significant|statistically|significantly)\w', txt)),
        para_over200_pct=round(100 * sum(1 for x in plens if x > 200) / len(plens), 1),
        para_max=max(plens),
        section_sym=txt.count('§'),
        connective_k=round(1000 * sum(1 for s in S if re.match(CONN, s)) / w, 2),
        connective_pct=round(100 * sum(1 for s in S if re.match(CONN, s)) / max(len(S), 1), 1),
        however_k=round(1000 * len(re.findall(r'(?i)\bhowever\b', txt)) / w, 2),
        robust_k=round(1000 * len(re.findall(r'(?i)\brobust(ly)?\b', txt)) / w, 2),
        maint=sum(len(re.findall(p, txt, re.I)) for p in
                  [r'folded in', r'has now been', r'now been run', r'superseded', r'\bretired\b']),
    )
    return m, ps

# Reuse the applier's model-name masking so that hyphens inside identifiers
# (Qwen3-1.7B, Llama-3.2-3B) are not read as minus signs, and so that a size moved
# into a canonical model name is not reported as a deleted number.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_ar", f"{W}/apply_rewrite.py")
_ar = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ar)

def numbers(ps):
    """Every numeric magnitude across the whole document, model names masked."""
    t = " ".join(x[1] for x in ps)
    return collections.Counter(re.findall(r'\d+(?:\.\d+)?', _ar.mask_models(t)))

def signed_numbers(ps):
    """Explicitly signed numbers (sign preceded by space or bracket), model names masked."""
    t = _ar.mask_models(" ".join(x[1] for x in ps))
    return collections.Counter(m.group(1) for m in re.finditer(
        r'(?:(?<=\s)|(?<=\()|(?<=\[))([-+−]\d+(?:\.\d+)?)', t))

# Every threshold below is the MEASURED corpus ceiling/floor, not an invented number.
TARGETS = dict(semi_k=(3.79, 'le'), emdash_total=(5, 'le'), paren_k=(13.84, 'le'),
               notY=(3, 'le'), section_sym=(0, 'le'), connective_k=(5.06, 'ge'),
               however_k=(0.7, 'ge'), para_p90=(166.2, 'le'), para_over200_pct=(4.4, 'le'),
               para_max=(354, 'le'), maint=(0, 'le'))
CORPUS = dict(semi_k='median 0.36, MAX 3.79', emdash_total='14/15 papers = 0',
              paren_k='median 1.97, MAX 13.84', notY='median 1, MAX 3',
              connective_k='median 5.06', however_k='14/15 papers higher',
              para_p90='median 114.6, MAX 166.2', para_over200_pct='median 0.6, MAX 4.4',
              para_max='median 222, MAX 354', maint='0/15 papers')

def main():
    m5, ps5 = metrics(V5, "v5 (original)")
    m6, ps6 = metrics(V6, "v6 (rewritten)")

    print("=" * 92)
    print("CONTENT PRESERVATION")
    print("=" * 92)
    n5, n6 = numbers(ps5), numbers(ps6)
    miss, extra = n5 - n6, n6 - n5
    print(f"  distinct numeric tokens: v5 {len(n5)}  v6 {len(n6)}")
    if miss or extra:
        print(f"  ✗ NUMBER DRIFT  missing from v6: {dict(list(miss.items())[:25])}")
        print(f"                  added in v6   : {dict(list(extra.items())[:25])}")
    else:
        print("  ✓ every numeric magnitude in v5 is present in v6 with the same multiplicity")
    s5, s6 = signed_numbers(ps5), signed_numbers(ps6)
    if (s6 - s5) or (s5 - s6):
        print(f"  ! signed-form drift  lost={dict(s5-s6)} gained={dict(s6-s5)}  "
              f"(check these are abstract cuts or sign-neutral rewordings)")
    else:
        print("  ✓ every signed value (+x / -x) preserved exactly")
    c5 = collections.Counter(re.findall(r'\[\d{1,3}\]', " ".join(x[1] for x in ps5)))
    c6 = collections.Counter(re.findall(r'\[\d{1,3}\]', " ".join(x[1] for x in ps6)))
    print(f"  citation brackets: v5 {sum(c5.values())}  v6 {sum(c6.values())}  "
          f"{'✓ all distinct refs preserved' if set(c5)==set(c6) else '✗ set differs: ' + str(set(c5)^set(c6))}")
    h5 = [t for s, t in ps5 if s.startswith("Heading")]
    h6 = [t for s, t in ps6 if s.startswith("Heading")]
    print(f"  headings: v5 {len(h5)}  v6 {len(h6)}  "
          f"{'✓ structure identical' if h5==h6 else '✗ heading list differs'}")
    if h5 != h6:
        for a in set(h5) ^ set(h6): print("      Δ", a[:80])

    print()
    print("=" * 92)
    print("VOICE METRICS  (body chapters only)")
    print("=" * 92)
    print(f"{'metric':<18}{'v5':>10}{'v6':>10}{'target':>12}{'':>4}{'corpus':<26}")
    order = ['semi_k', 'emdash_total', 'emdash_k', 'paren_k', 'notY', 'section_sym',
             'connective_k', 'connective_pct', 'however_k', 'robust_k', 'maint',
             'para_mean', 'para_p90', 'para_max', 'para_over200', 'para_over200_pct',
             'sent_mean', 'body_words', 'paragraphs']
    npass = nfail = 0
    for k in order:
        tgt = TARGETS.get(k)
        flag = ''
        if tgt:
            lim, direction = tgt
            ok = (m6[k] <= lim) if direction == 'le' else (m6[k] >= lim)
            flag = '✓' if ok else '✗'
            npass += ok; nfail += (not ok)
            tgtstr = f"{'≤' if direction=='le' else '≥'}{lim}"
        else:
            tgtstr = '—'
        print(f"{k:<18}{m5[k]:>10}{m6[k]:>10}{tgtstr:>12}  {flag:<2}{CORPUS.get(k,''):<26}")
    print(f"\n  acceptance targets: {npass} pass, {nfail} fail")
    json.dump(dict(v5=m5, v6=m6), open(f"{W}/meta/v6_metrics.json", "w"), indent=1)
    return 0 if (not miss and not extra and h5 == h6 and set(c5) == set(c6)) else 1

if __name__ == '__main__':
    sys.exit(main())
