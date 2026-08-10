"""Showcase build of the oral-defence deck (T55) as a single self-contained HTML file.

This does NOT restate the talk. It imports ``scripts/build_viva_deck.py`` and
records the slides that builder produces, so the web deck and the .pptx cannot
disagree: one source of content, two renderers. That is the same construction
``build_rewritten_fyp_interim_v1.py`` uses for the interim document, and for the
same reason (D56) — two copies drift, and a correction applied to one is silently
absent from the other.

Recording rather than refactoring is deliberate. The .pptx is the graded artefact
and is pinned by tests; swapping in a recorder leaves its builder byte-untouched.

The rendered page embeds every figure as a data URI and loads no external font,
script or stylesheet, so it opens from a file:// URL on a machine with no network
— which is the point of a showcase copy you can email or archive.

    python scripts/build_viva_deck_html.py
"""

from __future__ import annotations

import base64
import html
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPTX_BUILDER = ROOT / "scripts" / "build_viva_deck.py"
FIGS = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "viva" / "FYP_Viva_Deck.html"


class RecordingDeck:
    """Stands in for build_viva_deck.Deck and captures slides instead of drawing them.

    The method signatures must track the real Deck exactly; a mismatch surfaces
    immediately as a TypeError when build() runs, which is the intended failure.
    """

    records: list[dict] = []

    def __init__(self) -> None:
        RecordingDeck.records = []

    def title_slide(self) -> None:
        self.records.append({"kind": "title", "budget": "0:00 - 0:30"})

    def section(self, num, kicker, headline, budget, script) -> None:
        self.records.append({"kind": "section", "num": num, "kicker": kicker,
                             "headline": headline, "budget": budget, "script": script})

    def content(self, title, *, kicker=None, bullets=None, body=None, image=None,
                stat=None, budget="", script="", image_caption=None):
        self.records.append({"kind": "content", "title": title, "kicker": kicker,
                             "bullets": bullets, "body": body, "image": image,
                             "stat": stat, "budget": budget, "script": script,
                             "image_caption": image_caption})

    def cards(self, title, rows, *, kicker=None, budget="", script="", note=None):
        self.records.append({"kind": "cards", "title": title, "rows": rows,
                             "kicker": kicker, "budget": budget, "script": script,
                             "note": note})

    def closing(self, budget, script) -> None:
        self.records.append({"kind": "closing", "budget": budget, "script": script})

    def save(self) -> Path:
        return OUT


def record_slides() -> tuple[list[dict], dict]:
    spec = importlib.util.spec_from_file_location("build_viva_deck", PPTX_BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    meta = {"student": module.STUDENT, "matric": module.MATRIC, "title": module.TITLE,
            "project": module.PROJECT, "supervisor": module.SUPERVISOR,
            "school": module.SCHOOL, "year": module.YEAR}
    module.Deck = RecordingDeck          # the swap; build() constructs Deck() itself
    module.build()                        # runs the ASR-survivor guard on the way through
    return list(RecordingDeck.records), meta


def data_uri(name: str) -> str:
    raw = (FIGS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


E = html.escape


def render_slide(rec: dict, index: int, total: int) -> str:
    kind = rec["kind"]
    page = f'<span class="page">{index}<i>/{total}</i></span>'

    if kind == "title":
        return "".join([
            '<section class="slide dark title">',
            '<div class="mark"></div>',
            f'<p class="eyebrow">Final Year Project &middot; {E(M["project"])}</p>',
            f'<h1>{E(M["title"])}</h1>',
            f'<p class="who">{E(M["student"])} &middot; {E(M["matric"])}</p>',
            f'<p class="meta">Supervisor: {E(M["supervisor"])}<br>{E(M["school"])}<br>{E(M["year"])}</p>',
            page, '</section>'])

    if kind == "section":
        return "".join([
            '<section class="slide dark divider">',
            f'<div class="num">{E(rec["num"])}</div>',
            f'<div><p class="eyebrow">{E(rec["kicker"])}</p>',
            f'<h2>{E(rec["headline"]).replace(chr(10), "<br>")}</h2></div>',
            page, '</section>'])

    if kind == "closing":
        return "".join([
            '<section class="slide dark closing">',
            '<h1>Thank you</h1><p class="who">Questions</p>',
            f'<p class="meta">{E(M["student"])} &middot; {E(M["matric"])} &middot; {E(M["project"])}</p>',
            page, '</section>'])

    parts = ['<section class="slide">']
    if rec.get("kicker"):
        parts.append(f'<p class="eyebrow">{E(rec["kicker"])}</p>')
    parts.append(f'<h2>{E(rec["title"])}</h2>')

    if kind == "cards":
        parts.append('<div class="cards">')
        for head, sub in rec["rows"]:
            parts.append(f'<article><h3>{E(head)}</h3><p>{E(sub)}</p></article>')
        parts.append('</div>')
        if rec.get("note"):
            parts.append(f'<p class="note">{E(rec["note"])}</p>')
    else:
        has_img = bool(rec.get("image"))
        parts.append(f'<div class="body{" split" if has_img else ""}">')
        parts.append('<div class="col">')
        if rec.get("stat"):
            value, label = rec["stat"]
            parts.append(f'<p class="stat">{E(value)}</p><p class="statlabel">{E(label)}</p>')
        if rec.get("body"):
            parts.append(f'<p class="lede">{E(rec["body"])}</p>')
        if rec.get("bullets"):
            parts.append('<ul>')
            for b in rec["bullets"]:
                parts.append(f'<li>{E(b)}</li>')
            parts.append('</ul>')
        parts.append('</div>')
        if has_img:
            parts.append('<figure><img src="%s" alt="%s">' % (
                data_uri(rec["image"]), E(rec.get("image_caption") or rec["title"])))
            if rec.get("image_caption"):
                parts.append(f'<figcaption>{E(rec["image_caption"])}</figcaption>')
            parts.append('</figure>')
        parts.append('</div>')

    parts.append(page)
    parts.append('</section>')
    return "".join(parts)


CSS = """
/* The stage is a FIXED 1280x720 canvas that is scaled as a unit to fit the
   viewport, rather than a fluid layout that reflows. Two reasons. It makes the
   web deck geometrically identical to the .pptx (13.333in x 7.5in at 96dpi is
   exactly 1280x720, and every size below is the pptx point size x 4/3), and it
   makes overflow impossible at any window size -- the first fluid version, built
   with clamp() sizes, overflowed thirteen of seventeen slides once the pane got
   narrow, because type has a minimum size and a 16:9 box does not. */
*{box-sizing:border-box;margin:0;padding:0}
:root{--indigo:#21295C;--deep:#161B3D;--gold:#C99B2B;--gold-l:#E8C15F;
--ink:#232433;--muted:#6B7280;--tint:#F2F3F8;--paper:#fff;--s:1;
--serif:Cambria,Georgia,"Times New Roman",serif;
--sans:Calibri,"Helvetica Neue",Arial,sans-serif}
html,body{height:100%;background:#0d1030;font-family:var(--sans);color:var(--ink);
overflow:hidden}
.deck{position:fixed;inset:0;display:grid;place-items:center}
.stage{position:relative;width:1280px;height:720px;flex:none;
transform:scale(var(--s));transform-origin:center center}
.slide{display:none;position:absolute;inset:0;
background:var(--paper);padding:56px 72px;
border-radius:6px;box-shadow:0 30px 90px rgba(0,0,0,.5);overflow:hidden}
/* Flex column so the content region can claim the leftover height and centre in
   it. A first pass used display:block and every short slide sat in the top half
   with a third of the frame empty underneath. */
.slide.on{display:flex;flex-direction:column}
.slide.dark{background:var(--indigo);color:#fff}
.slide.title,.slide.closing{background:var(--deep);display:none;
flex-direction:column;justify-content:center}
.slide.title.on,.slide.closing.on{display:flex}
.slide.divider{display:none;align-items:center;gap:44px}
.slide.divider.on{display:flex;flex-direction:row}
h1{font-family:var(--serif);font-weight:700;font-size:53px;
line-height:1.14;letter-spacing:-.01em;max-width:19ch;margin-bottom:.5em}
h2{font-family:var(--serif);font-weight:700;color:var(--indigo);
font-size:39px;line-height:1.16;margin-bottom:.5em}
.dark h2{color:#fff;margin:0}
h3{font-family:var(--serif);font-weight:700;color:var(--indigo);
font-size:23px;margin-bottom:.45em}
.eyebrow{font-size:15px;letter-spacing:.14em;text-transform:uppercase;
font-weight:700;color:var(--gold);margin-bottom:.9em}
.dark .eyebrow{color:var(--gold-l)}
.title .mark{width:14px;height:14px;border-radius:50%;background:var(--gold);margin-bottom:22px}
.who{font-size:27px;font-weight:700;color:var(--gold-l);margin:.5em 0}
.meta{font-size:19px;line-height:1.75;color:#B9BED8}
.divider .num{font-family:var(--serif);font-weight:700;color:var(--deep);
background:var(--gold);width:83px;aspect-ratio:1;border-radius:50%;
display:grid;place-items:center;font-size:43px;flex:none}
.body{display:flex;flex-direction:column;justify-content:center;flex:1}
.body.split{display:grid;grid-template-columns:1fr 1.02fr;gap:44px;
align-items:center}
.stat{font-family:var(--serif);font-weight:700;color:var(--indigo);
font-size:83px;line-height:1}
.statlabel{color:var(--muted);font-size:20px;margin:.5em 0 1.1em;
line-height:1.5;max-width:52ch}
.lede{font-size:21px;line-height:1.55;margin-bottom:1em;max-width:60ch}
ul{list-style:none;display:flex;flex-direction:column;gap:15px}
li{position:relative;padding-left:1.25em;font-size:20px;
line-height:1.5;max-width:62ch}
li::before{content:"";position:absolute;left:0;top:.52em;width:7px;height:7px;
border-radius:50%;background:var(--gold)}
figure{margin:0}
/* Cap BOTH dimensions. Capping width alone overflowed the "What I built" slide by
   115px, because system_architecture.png is taller than it is wide (2000x2172)
   and a 100%-width column made it ~650px tall inside a 720px frame. */
figure img{max-width:100%;max-height:452px;width:auto;height:auto;
display:block;margin:0 auto}
figcaption{margin-top:.6em;text-align:center;color:var(--muted);font-style:italic;
font-size:15px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:24px;
flex:1;align-content:center}
.cards article{background:var(--tint);border-radius:10px;padding:26px}
.cards p{font-size:17px;line-height:1.55}
.note{margin-top:1.1em;color:var(--muted);font-style:italic;font-size:16px}
.page{position:absolute;right:32px;bottom:26px;
color:var(--muted);font-size:15px;font-variant-numeric:tabular-nums}
.dark .page{color:#7d84ad}
.page i{opacity:.55;font-style:normal}
.bar{position:fixed;left:0;top:0;height:3px;background:var(--gold);transition:width .25s}
.notes{position:fixed;left:0;right:0;bottom:0;max-height:42vh;overflow:auto;
background:rgba(13,16,48,.97);color:#dfe3f5;padding:18px 26px;
font-size:13.5px;line-height:1.65;display:none;border-top:2px solid var(--gold)}
.notes.on{display:block}
.notes b{color:var(--gold-l);font-weight:700}
.notes p{white-space:pre-wrap;max-width:88ch}
.hint{position:fixed;left:18px;bottom:14px;color:#6f77a6;font-size:11px}
@media print{.bar,.notes,.hint{display:none}html,body{overflow:visible}
.deck{position:static}.stage{transform:none;page-break-after:always}
.slide{display:block!important;position:relative;page-break-after:always;box-shadow:none}}
"""

JS = """
const S=[...document.querySelectorAll('.slide')],N=S.length,notes=NOTES;
let i=0,show=false;
const bar=document.querySelector('.bar'),nx=document.querySelector('.notes');
/* Scale the fixed 1280x720 stage to the viewport. The margin leaves room for the
   drop shadow, and the notes drawer takes the bottom 42vh when it is open. */
function fit(){const pad=48,hAvail=innerHeight*(show?0.56:1)-pad;
/* Floor at a positive value: a viewport reported as 0x0 (an offscreen or
   not-yet-laid-out frame) otherwise yields a NEGATIVE scale, which flips the
   stage through the origin instead of merely shrinking it. */
document.documentElement.style.setProperty('--s',
  Math.max(0.05, Math.min((innerWidth-pad)/1280, hAvail/720)));}
addEventListener('resize',fit);
function draw(){S.forEach((s,k)=>s.classList.toggle('on',k===i));
bar.style.width=((i+1)/N*100)+'%';
nx.innerHTML='<p><b>'+notes[i].b+'</b>\\n\\n'+notes[i].s.replace(/</g,'&lt;')+'</p>';
nx.classList.toggle('on',show);fit();location.hash=i+1}
function go(d){i=Math.min(N-1,Math.max(0,i+d));draw()}
addEventListener('keydown',e=>{
if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(1)}
else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}
else if(e.key==='Home'){i=0;draw()} else if(e.key==='End'){i=N-1;draw()}
else if(e.key.toLowerCase()==='n'){show=!show;draw()}});
addEventListener('click',e=>{if(!e.target.closest('.notes'))go(1)});
i=Math.max(0,Math.min(N-1,(parseInt(location.hash.slice(1))||1)-1));draw();
"""


def build() -> Path:
    global M
    slides, M = record_slides()
    total = len(slides)
    body = "".join(render_slide(r, n, total) for n, r in enumerate(slides, start=1))
    notes = json.dumps([{"b": s.get("budget", ""), "s": s.get("script", "")} for s in slides])

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{E(M["title"])} — oral defence</title>'
        f'<style>{CSS}</style></head><body>'
        '<div class="bar"></div>'
        f'<div class="deck"><div class="stage">{body}</div></div>'
        '<div class="notes"></div>'
        '<p class="hint">&larr; &rarr; navigate &middot; N speaker notes &middot; '
        'Cmd/Ctrl-P prints all slides</p>'
        f'<script>const NOTES={notes};{JS}</script>'
        '</body></html>'
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"{out}  ({out.stat().st_size/1_000_000:.2f} MB, {len(RecordingDeck.records)} slides)")
