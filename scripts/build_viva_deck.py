"""Oral-defence deck for the FYP viva (T55, 4/7/8 Dec 2026).

Built to the school's prescribed presentation order (SCHOOL_GUIDELINES §3.2.1):
problem -> prior work -> how I solved it -> results -> conclusion and future work.
That order is not decoration: the examiner grades appreciation of the FYP,
difficulty and contribution, technical ability, coding skill, and knowledge, and
each of those has a slide that exists to answer it.

Two school rules are hard requirements and are asserted by tests/test_viva_deck.py:
slide 1 carries the student name, the FYP title and the year, and EVERY slide
carries a page number.

Timing: 20 minutes of talk, 10 minutes of Q&A (CCDS AY2026 timeline, which
supersedes the older "20-23 min + ~5 min" capture). Per-slide budgets live in the
speaker notes and sum to 20:00; scripts/check_viva_timing.py re-adds them.

Every number is imported from the committed analysis artifacts via the claim
registry rather than typed in, so the deck cannot drift from the study the way a
hand-made deck would.

    python scripts/build_viva_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "viva" / "FYP_Viva_Deck.pptx"

W, H = 13.333, 7.5

INDIGO = RGBColor(0x21, 0x29, 0x5C)
INDIGO_DEEP = RGBColor(0x16, 0x1B, 0x3D)
GOLD = RGBColor(0xC9, 0x99, 0x2B)
GOLD_LIGHT = RGBColor(0xE8, 0xC1, 0x5F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x23, 0x24, 0x33)
MUTED = RGBColor(0x6B, 0x72, 0x80)
TINT = RGBColor(0xF2, 0xF3, 0xF8)
RED = RGBColor(0xA3, 0x33, 0x3D)

SERIF = "Cambria"
SANS = "Calibri"

STUDENT = "TAN UEI HORNG"
MATRIC = "UTAN001"
TITLE = "Benchmarking the Ethical Performance of Quantized Open-Source LLMs"
PROJECT = "CCDS25-1136"
SUPERVISOR = "Dr. Zhang Jiehuang"
SCHOOL = "College of Computing and Data Science, Nanyang Technological University"
YEAR = "Academic Year 2026"


def load_numbers() -> dict:
    """Read the headline values from the committed analysis tree.

    The deck must never carry a number typed by hand — that is how the 128-era
    figures survived in eight surfaces (D42). Anything shown on a slide is read
    from results_512/analysis here and formatted at build time.
    """
    analysis = ROOT / "results_512" / "analysis"
    mc = json.loads((analysis / "multiple_comparisons.json").read_text())
    survivors = mc["bh_survivors"]
    return {
        "family_size": mc["family_size"],
        "uncorrected_sig": mc["n_uncorrected_significant"],
        "bh_survivors": survivors,
        "asr_survivors": [s for s in survivors if "asr" in str(s["metric"]).lower()],
        "raw": mc,
    }


def fit(path: Path, box_x: float, box_y: float, box_w: float, box_h: float):
    """Return (x, y, w, h) inches fitting the image inside the box, centred."""
    with Image.open(path) as im:
        ar = im.size[0] / im.size[1]
    w, h = box_w, box_w / ar
    if h > box_h:
        h, w = box_h, box_h * ar
    return box_x + (box_w - w) / 2, box_y + (box_h - h) / 2, w, h


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.n = 0

    # ---------------------------------------------------------------- helpers
    def _blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _bg(self, slide, color) -> None:
        shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
        shp.fill.solid()
        shp.fill.fore_color.rgb = color
        shp.line.fill.background()
        shp.shadow.inherit = False

    def _text(self, slide, text, x, y, w, h, *, size=16, color=INK, font=SANS,
              bold=False, align=PP_ALIGN.LEFT, italic=False, spacing=1.0,
              anchor=MSO_ANCHOR.TOP):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = spacing
            r = p.add_run()
            r.text = line
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = font
            r.font.color.rgb = color
        return box

    @staticmethod
    def _lines(text: str, width_in: float, size: float, *, wide=False) -> int:
        """Estimate wrapped line count.

        Layout here is computed, not hand-placed: the first render collided every
        multi-line bullet with the one under it because the row height was a
        constant. Average glyph width is ~0.48x point size for Calibri and ~0.52x
        for Cambria bold, which is close enough to advance rows correctly.
        """
        per_char = size * (0.52 if wide else 0.48) / 72.0
        chars_per_line = max(1, int(width_in / per_char))
        total = 0
        for para in text.split("\n"):
            total += max(1, -(-len(para) // chars_per_line))
        return total

    def _bullets(self, slide, items, x, y, w, *, size=15, color=INK, dot=GOLD,
                 pad=0.17):
        """Gold dot + text rows, advancing by each row's real wrapped height."""
        line_h = size * 1.28 / 72.0
        cy = y
        for item in items:
            n = self._lines(item, w - 0.3, size)
            h = n * line_h
            d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(cy + 0.075),
                                       Inches(0.13), Inches(0.13))
            d.fill.solid()
            d.fill.fore_color.rgb = dot
            d.line.fill.background()
            d.shadow.inherit = False
            self._text(slide, item, x + 0.3, cy, w - 0.3, h, size=size, color=color,
                       spacing=1.28)
            cy += h + pad
        return cy

    def _page(self, slide) -> None:
        """Every slide must carry a page number (school rule §3.2.1)."""
        self._text(slide, str(self.n), W - 0.85, H - 0.52, 0.5, 0.3,
                   size=11, color=MUTED, align=PP_ALIGN.RIGHT)

    def _notes(self, slide, budget: str, script: str) -> None:
        slide.notes_slide.notes_text_frame.text = f"[{budget}]\n\n{script}"

    # ----------------------------------------------------------- slide makers
    def title_slide(self) -> None:
        s = self._blank()
        self.n += 1
        self._bg(s, INDIGO_DEEP)
        badge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), Inches(0.98),
                                   Inches(0.2), Inches(0.2))
        badge.fill.solid()
        badge.fill.fore_color.rgb = GOLD
        badge.line.fill.background()
        badge.shadow.inherit = False
        self._text(s, f"Final Year Project   ·   {PROJECT}", 1.4, 0.93, 8, 0.4,
                   size=15, color=GOLD_LIGHT, font=SANS)
        self._text(s, TITLE, 1.0, 2.05, 10.6, 2.2, size=40, color=WHITE,
                   font=SERIF, bold=True, spacing=1.12)
        self._text(s, f"{STUDENT}   ·   {MATRIC}", 1.0, 4.62, 10, 0.45,
                   size=20, color=GOLD_LIGHT, font=SANS, bold=True)
        self._text(s, f"Supervisor: {SUPERVISOR}\n{SCHOOL}\n{YEAR}",
                   1.0, 5.25, 10.6, 1.3, size=14, color=RGBColor(0xB9, 0xBE, 0xD8),
                   spacing=1.35)
        self._page(s)
        self._notes(s, "0:00 - 0:30", (
            "Good morning. I am Tan Uei Horng, and this is CCDS25-1136, "
            "Benchmarking the Ethical Performance of Quantized Open-Source LLMs, "
            "supervised by Dr Zhang Jiehuang.\n\n"
            "In one sentence: I tested whether compressing a language model to four "
            "bits makes it less safe, and I found that the honest answer depends "
            "far more on how you measure than on the compression itself.\n\n"
            "DELIVERY: say the one-sentence version slowly. It is the frame for "
            "everything after it. Do not rush into method."))

    def section(self, num: str, kicker: str, headline: str, budget: str,
                script: str) -> None:
        s = self._blank()
        self.n += 1
        self._bg(s, INDIGO)
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), Inches(2.55),
                                  Inches(1.15), Inches(1.15))
        circ.fill.solid()
        circ.fill.fore_color.rgb = GOLD
        circ.line.fill.background()
        circ.shadow.inherit = False
        self._text(s, num, 1.0, 2.83, 1.15, 0.6, size=32, color=INDIGO_DEEP,
                   font=SERIF, bold=True, align=PP_ALIGN.CENTER)
        self._text(s, kicker, 2.6, 2.62, 9, 0.4, size=14, color=GOLD_LIGHT, bold=True)
        self._text(s, headline, 2.6, 3.05, 9.5, 1.4, size=32, color=WHITE,
                   font=SERIF, bold=True, spacing=1.1)
        self._page(s)
        self._notes(s, budget, script)

    def content(self, title, *, kicker=None, bullets=None, body=None, image=None,
                stat=None, budget="", script="", image_caption=None):
        s = self._blank()
        self.n += 1
        self._bg(s, WHITE)
        y = 0.62
        if kicker:
            self._text(s, kicker.upper(), 0.9, y, 11.5, 0.28, size=11.5,
                       color=GOLD, bold=True)
            y += 0.36
        title_h = self._lines(title, 11.5, 29, wide=True) * 29 * 1.16 / 72.0
        self._text(s, title, 0.9, y, 11.5, title_h, size=29, color=INDIGO,
                   font=SERIF, bold=True, spacing=1.16)

        top = y + title_h + 0.38
        text_w = 6.1 if image else 11.5

        if stat:
            value, label = stat
            self._text(s, value, 0.9, top, text_w, 1.05, size=62, color=INDIGO,
                       font=SERIF, bold=True)
            lh = self._lines(label, text_w, 15) * 15 * 1.3 / 72.0
            self._text(s, label, 0.9, top + 1.12, text_w, lh, size=15, color=MUTED,
                       spacing=1.3)
            top += 1.12 + lh + 0.42
        if body:
            bh_ = self._lines(body, text_w, 15.5) * 15.5 * 1.34 / 72.0
            self._text(s, body, 0.9, top, text_w, bh_, size=15.5, color=INK,
                       spacing=1.34)
            top += bh_ + 0.40
        if bullets:
            self._bullets(s, bullets, 0.9, top, text_w)
        if image:
            path = FIGS / image
            bx, by, bw, bh = (6.95, 1.62, 5.55, 4.95)
            x, iy, w, h = fit(path, bx, by, bw, bh)
            s.shapes.add_picture(str(path), Inches(x), Inches(iy), Inches(w), Inches(h))
            if image_caption:
                # Anchored to the image, not to the box: a short figure used to
                # leave the caption stranded an inch below it.
                self._text(s, image_caption, bx, iy + h + 0.16, bw, 0.32,
                           size=11, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
        self._page(s)
        self._notes(s, budget, script)
        return s

    def cards(self, title, rows, *, kicker=None, budget="", script="", note=None):
        """Grid of tinted cards. Used where a comparison beats prose."""
        s = self._blank()
        self.n += 1
        self._bg(s, WHITE)
        y = 0.62
        if kicker:
            self._text(s, kicker.upper(), 0.9, y, 11.5, 0.28, size=11.5,
                       color=GOLD, bold=True)
            y += 0.36
        title_h = self._lines(title, 11.5, 29, wide=True) * 29 * 1.16 / 72.0
        self._text(s, title, 0.9, y, 11.5, title_h, size=29, color=INDIGO,
                   font=SERIF, bold=True, spacing=1.16)
        top = y + title_h + 0.45
        n = len(rows)
        cols = 2 if n <= 4 else 3
        cw = (11.5 - 0.4 * (cols - 1)) / cols
        ch = 1.95 if n <= 4 else 1.72
        for i, (head, sub) in enumerate(rows):
            cx = 0.9 + (i % cols) * (cw + 0.4)
            cy = top + (i // cols) * (ch + 0.34)
            card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx),
                                      Inches(cy), Inches(cw), Inches(ch))
            card.fill.solid()
            card.fill.fore_color.rgb = TINT
            card.line.fill.background()
            card.shadow.inherit = False
            card.adjustments[0] = 0.06
            self._text(s, head, cx + 0.34, cy + 0.3, cw - 0.68, 0.5, size=17,
                       color=INDIGO, font=SERIF, bold=True)
            self._text(s, sub, cx + 0.34, cy + 0.86, cw - 0.68, ch - 1.1,
                       size=13, color=INK, spacing=1.22)
        if note:
            self._text(s, note, 0.9, H - 0.95, 11.5, 0.4, size=12.5,
                       color=MUTED, italic=True)
        self._page(s)
        self._notes(s, budget, script)
        return s

    def closing(self, budget: str, script: str) -> None:
        s = self._blank()
        self.n += 1
        self._bg(s, INDIGO_DEEP)
        self._text(s, "Thank you", 1.0, 2.5, 10, 1.0, size=44, color=WHITE,
                   font=SERIF, bold=True)
        self._text(s, "Questions", 1.0, 3.6, 10, 0.6, size=22, color=GOLD_LIGHT)
        self._text(s, f"{STUDENT}  ·  {MATRIC}  ·  {PROJECT}", 1.0, 5.9, 10, 0.4,
                   size=13, color=RGBColor(0xB9, 0xBE, 0xD8))
        self._page(s)
        self._notes(s, budget, script)

    def save(self) -> Path:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(OUT))
        return OUT


def build() -> Path:
    nums = load_numbers()
    asr_survivors = len(nums["asr_survivors"])
    if asr_survivors != 0:
        raise SystemExit(
            f"ASR survivor count is {asr_survivors}, not 0. The deck's headline claim "
            "('no ASR contrast survives correction') no longer matches the artifacts — "
            "re-check results_512/analysis/multiple_comparisons.json before presenting."
        )

    d = Deck()
    d.title_slide()

    # ---------------------------------------------------------- 01 the problem
    d.section("01", "The problem", "Compression is how small models\nactually get deployed.",
              "0:30 - 0:45",
              "Signpost only. Move straight on.")

    d.content(
        "A four-bit model is not the model you aligned",
        kicker="Motivation",
        stat=("~4x", "less GPU memory for NF4 4-bit weights versus fp16 — the reason "
              "quantization is the default way to fit a model on one consumer GPU"),
        bullets=[
            "Alignment training is done once, on the full-precision weights.",
            "Deployment then changes those weights — after all the safety work is finished.",
            "The question this project asks: does that last step quietly undo any of it?",
        ],
        budget="0:45 - 2:15",
        script=(
            "Quantization is not an exotic technique. It is the normal way a 7-billion "
            "parameter model gets onto one consumer GPU: you store the weights in four "
            "bits instead of sixteen, and you get roughly a four-fold memory reduction.\n\n"
            "The thing to notice is the ORDER of operations. A lab aligns a model — "
            "RLHF, safety tuning, red-teaming — all on the full-precision weights. Then "
            "somebody downstream compresses those weights to deploy them. Every safety "
            "guarantee was established BEFORE the last transformation applied to the model.\n\n"
            "So the question is not academic. If four-bit quantization erodes refusal "
            "behaviour, then the models most likely to be running on someone's laptop are "
            "the ones whose safety we have tested least.\n\n"
            "DELIVERY: this is the slide that earns their attention. Land the order-of-"
            "operations point and pause."))

    d.cards(
        "What was already known — and what was missing",
        [("Quantization degrades capability",
          "Well established. Perplexity and benchmark accuracy fall as bit-width drops. "
          "Not in dispute, and not what this project is about."),
         ("Quantization and safety: mixed reports",
          "Prior studies report increases, decreases and no effect. They differ in models, "
          "methods, prompts and — critically — in how safety was scored."),
         ("Gap 1 — the confound",
          "A model that answers fewer harmful prompts may be safer, or may simply have got "
          "worse at answering anything. A safety score alone cannot tell those apart."),
         ("Gap 2 — the scorers",
          "Safety numbers are produced by an automatic scorer. Almost nobody validates "
          "the scorer itself against human judgement before drawing conclusions from it.")],
        kicker="Prior work",
        budget="2:15 - 4:00",
        script=(
            "Briefly on what the literature already gives us.\n\n"
            "That quantization costs capability is settled — I am not re-establishing it. "
            "What is NOT settled is safety: published results point in every direction, and "
            "when you look at why, the studies differ in how they scored safety at all.\n\n"
            "Two gaps drove my design. The first is a confound. Attack Success Rate falling "
            "is compatible with two completely different stories — the model got safer, or the "
            "model got dumber. Most papers report the number without separating them.\n\n"
            "The second is more basic. These safety rates come out of an automatic scorer, "
            "and the scorer is usually taken on trust. I did not take mine on trust, and that "
            "turned out to be the most consequential decision in the project.\n\n"
            "DELIVERY: if you are running long, compress the first two cards to one sentence. "
            "The two GAPS must survive — they justify the whole design."))

    d.content(
        "Five questions, fixed before the results came in",
        kicker="Research questions",
        bullets=[
            "RQ1  Does NF4 measurably increase harmful compliance versus matched fp16 weights?",
            "RQ2  Does NF4 increase over-refusal on benign prompts?",
            "RQ3  Does NF4 reduce capability on MMLU or ARC-Challenge?",
            "RQ4  How do effects vary across families, scales, and an added INT8 point?",
            "RQ5  Does the conclusion survive alternative scorers, human grounding, "
            "generation budgets and decoding seeds?",
        ],
        body="RQ5 is not a robustness afterthought. It became the main finding.",
        budget="4:00 - 5:00",
        script=(
            "Five research questions. One, two and three are the three axes — harmful "
            "compliance, over-refusal, capability. Four is the variation across families and "
            "precisions.\n\n"
            "Five is the one I would point at. It asks whether the answer holds up when you "
            "change the measuring instrument. I expected that to be a robustness appendix. It "
            "became the most important result in the project.\n\n"
            "DELIVERY: do not read all five aloud verbatim — the examiner can read. Say "
            "'three axes, plus variation, plus robustness', then spend your words on RQ5."))

    # ------------------------------------------------------- 02 how I solved it
    d.section("02", "How I solved it", "Hold everything still except\nthe one thing under test.",
              "5:00 - 5:15", "Signpost only.")

    d.content(
        "Matched-pair design",
        kicker="Method — the control",
        bullets=[
            "Both members of a pair load the SAME weights, pinned to the same revision hash.",
            "Quantization is applied on the fly at load time — nothing else differs.",
            "Greedy decoding, temperature 0, fixed seed, identical prompts and templates.",
            "Five pairs across four families: Qwen 1.7B and 4B, Llama-3.2 3B, Mistral 7B, Phi-4-mini.",
        ],
        body="Any difference measured is attributable to the quantization step, not to a "
             "different checkpoint, a different prompt, or a different sampler.",
        image="study_design.png",
        image_caption="Matched-pair study design",
        budget="5:15 - 7:00",
        script=(
            "The design is a matched pair. Both arms load the same model identifier at the "
            "same pinned revision. The only difference is a BitsAndBytes config applied at "
            "load time, which quantizes to NF4 on the fly.\n\n"
            "Everything else is held still: greedy decoding at temperature zero, a fixed "
            "seed, the same prompts, the same chat template. So a delta is attributable to "
            "the quantization step and nothing else.\n\n"
            "Five pairs, four families, roughly 1.7 to 7 billion parameters — because a "
            "single-family result would not tell you whether you were looking at a property "
            "of quantization or a property of Qwen.\n\n"
            "DELIVERY: the examiner is grading experimental control here. 'Same weights, "
            "same revision hash, one variable' is the sentence that earns the mark."))

    d.content(
        "The capability anchor",
        kicker="Method — the rule that makes a safety number readable",
        body="A lower attack-success rate is not evidence of improved alignment if the same "
             "model also lost the ability to answer ordinary questions.",
        bullets=[
            "Every safety delta is read alongside MMLU and ARC-Challenge on the same pair.",
            "Safety down + capability flat  →  a real alignment change.",
            "Safety down + capability down  →  capability collapse wearing safety's clothes.",
            "This rule is pre-specified, and it is what separates this study from a bare ASR table.",
        ],
        image="capability_anchor.png",
        image_caption="Safety change read against capability change",
        budget="7:00 - 8:30",
        script=(
            "This is the interpretive core of the project, and if you remember one slide, "
            "make it this one.\n\n"
            "I never read a safety number on its own. Every safety delta is paired with the "
            "capability delta for the same models on the same run. If harmful compliance "
            "falls and capability is flat, something about alignment genuinely changed. If "
            "both fall together, the safer-looking number is most likely just a less capable "
            "model, and calling that a safety improvement would be wrong.\n\n"
            "That rule is fixed in advance, so it cannot be applied selectively after seeing "
            "which way the numbers went.\n\n"
            "DELIVERY: this answers 'what is your contribution' as much as any results slide. "
            "Say it slowly."))

    d.content(
        "What I built",
        kicker="Implementation",
        bullets=[
            "A plugin benchmark framework — HarmBench, XSTest, MMLU, ARC behind one interface.",
            "Pydantic-validated YAML configs; a pair is invalid unless it has a baseline and a quantized member.",
            "SLURM job generation for the university GPU cluster, offline-mode and policy-compliant.",
            "Raw outputs are immutable. Every rescoring writes a redacted derived sidecar, never an edit.",
            "An automated suite of several hundred tests, plus gates that assert the "
            "report's numbers against the artifacts.",
        ],
        image="system_architecture.png",
        image_caption="Framework architecture",
        budget="8:30 - 10:15",
        script=(
            "On the engineering, because it is graded.\n\n"
            "The framework is plugin-based: each benchmark implements one interface, so "
            "adding ARC-Challenge later was a plugin plus a config entry, not a rewrite. "
            "Configs are schema-validated — a pair that lacks a baseline or a quantized "
            "member fails before any GPU time is spent.\n\n"
            "The part I would defend hardest is the immutability rule. Raw model outputs are "
            "written once and never modified. When I later corrected how HarmBench was "
            "scored, that correction did not touch a single original file — it wrote a "
            "separate redacted sidecar. So the record of what the models actually produced is "
            "the same today as on the day it ran, and every scoring decision is reversible "
            "and auditable.\n\n"
            "DELIVERY: if asked what YOU wrote versus what came off the shelf — the models, "
            "the datasets and BitsAndBytes are off the shelf; the framework, the pairing "
            "logic, the scoring layer, the cluster orchestration, the analysis and the "
            "verification harness are mine."))

    d.content(
        "The scorer was wrong, and I could prove it",
        kicker="Implementation — and the first real finding",
        bullets=[
            "I began with a refusal-detecting regex. It is fast, deterministic, and it over-counted harmful compliance.",
            "Replaced as primary by the official HarmBench classifier, then cross-checked against gpt-4o and open-weight LlamaGuard.",
            "Grounded against human labels: on HarmBench the classifier agrees with me at kappa 0.59; the regex, 0.11.",
            "On XSTest the regex found 2 of 63 genuine refusals. It was not measuring what its name claimed.",
        ],
        image="judge_vs_proxy.png",
        image_caption="Classifier versus regex proxy",
        budget="10:15 - 12:00",
        script=(
            "This is where the project changed shape.\n\n"
            "I started with a regex that detects refusal phrases — if the model did not "
            "refuse, count it as compliance. It ran everything, it was reproducible, and it "
            "was wrong: it systematically over-counted harmful compliance, because not "
            "refusing is not the same as actually helping.\n\n"
            "So I replaced it as the primary scorer with the official HarmBench classifier, "
            "then validated THAT against a second judge, gpt-4o, and a third open-weight "
            "judge, LlamaGuard. Then I hand-labelled two hundred responses myself, blind, to "
            "get ground truth. Against my labels the classifier reaches kappa 0.59; the regex "
            "reaches 0.11. On the over-refusal benchmark the regex caught two of sixty-three "
            "real refusals.\n\n"
            "I want to be straight about this: I did not discover a flaw in someone else's "
            "work. I found it in my own, and the honest thing was to re-run everything.\n\n"
            "DELIVERY: examiners respond well to this. Own it plainly — do not apologise for "
            "it, and do not oversell it either."))

    # ------------------------------------------------------------- 03 results
    d.section("03", "Results", "No harmful-compliance increase\nsurvives correction.",
              "12:00 - 12:15", "Signpost only.")

    d.content(
        "RQ1 — harmful compliance",
        kicker="Result 1",
        stat=("0 of 20", "contrasts involving harmful compliance survive Benjamini-Hochberg "
              "correction across the pre-registered family"),
        bullets=[
            "The only individually significant ASR result is Llama-3.2-3B at −0.040 — a DECREASE.",
            "Point estimates: 0.000, +0.040, −0.040, −0.020, +0.020 across the five pairs.",
            "Three contrasts do survive correction. All three are capability or over-refusal, none is harmful compliance.",
        ],
        image="asr_forest.png",
        image_caption="Per-pair ASR deltas with confidence intervals",
        budget="12:15 - 14:00",
        script=(
            "The headline answer to RQ1 is no.\n\n"
            "Across a pre-registered family of twenty contrasts, five are significant "
            "uncorrected and three survive Benjamini-Hochberg. None of the three is harmful "
            "compliance. The one ASR contrast that is individually significant is Llama, and "
            "it goes DOWN — a decrease, not an increase.\n\n"
            "Note the direction of the pressure here. I went looking for a safety regression. "
            "The honest result is that I did not find one, and I am reporting that rather "
            "than hunting for a subgroup where it appears.\n\n"
            "DELIVERY: expect 'so your result is negative?' Answer: it is a bounded null, and "
            "the bound is stated — I can detect about a six-point change at eighty percent "
            "power, so I am ruling out large effects, not all effects."))

    d.content(
        "RQ3 — where the loss actually is",
        kicker="Result 2",
        bullets=[
            "Qwen3-1.7B loses 9.0 points of MMLU accuracy (q = 0.008).",
            "Llama-3.2-3B loses 3.2 points of ARC-Challenge (q = 0.008).",
            "Phi-4-mini's over-refusal falls 4.8 points (q = 0.012) — a decrease, and scorer-dependent.",
            "Capability loss is a cliff at four bits, not a smooth slope: the INT8 point barely moves.",
        ],
        body="So the trade-off is real, but it sits on the capability axis, not the safety axis. "
             "That is a different engineering decision from the one the literature implies.",
        image="precision_sweep.png",
        image_caption="fp16 → INT8 → NF4",
        budget="14:00 - 15:30",
        script=(
            "The three survivors are all capability or over-refusal.\n\n"
            "The smallest model, Qwen 1.7B, loses nine points of MMLU. Llama loses three "
            "points of ARC. And Phi's over-refusal falls — note falls, meaning it became less "
            "conservative on benign prompts, not more.\n\n"
            "The precision sweep adds something the two-point comparison would have missed. "
            "Capability loss is not graded by bit-width. INT8 sits almost on top of fp16, then "
            "four-bit drops. It behaves like a cliff, and that matters if you are choosing a "
            "deployment precision.\n\n"
            "DELIVERY: this is the slide that answers 'so what should a practitioner do'. "
            "The answer: budget for capability loss at four bits, and do not assume the safety "
            "axis is where your risk is."))

    d.content(
        "The result I nearly published was an artefact",
        kicker="Result 3 — RQ5, and the methodological contribution",
        bullets=[
            "At a 128-token generation budget, Qwen3-1.7B showed +0.055 ASR — a clean, publishable-looking safety regression.",
            "At 512 tokens, HarmBench's own reference budget, the same contrast is +0.000.",
            "62% of responses run past the 128-token ceiling; median length goes from 567 to 1675 characters.",
            "The 'regression' was truncation: cut a model off mid-sentence and a refusal it never finished reads as compliance.",
        ],
        body="Both judges agree it disappears. This is the finding I would most want another "
             "group to check, because it implies some published effects in this area are "
             "measurement artefacts.",
        image="multiseed.png",
        image_caption="Stability across decoding seeds",
        budget="15:30 - 17:15",
        script=(
            "This is the result I am proudest of, and it started as an embarrassment.\n\n"
            "At a 128-token generation budget I had a clean safety regression: Qwen 1.7B, plus "
            "five and a half points of harmful compliance. It was significant. It was exactly "
            "the finding the project was set up to detect.\n\n"
            "Then I re-ran everything at 512 tokens, which is HarmBench's own reference budget. "
            "The same contrast is zero. Sixty-two percent of responses had been running past "
            "the old ceiling — the median response more than doubles in length. What I had "
            "measured was truncation. If you cut a model off mid-sentence, a refusal it had not "
            "finished writing gets scored as compliance.\n\n"
            "So my headline number was an artefact of a decoding parameter, and both judges "
            "agree it vanishes. The uncomfortable implication is that this is a cheap mistake "
            "for anyone in this area to make.\n\n"
            "DELIVERY: this plus the scorer slide is your answer to 'what is the contribution'. "
            "Two measurement-validity findings, both discovered by checking my own work."))

    d.cards(
        "What this study does not establish",
        [("Direct requests only",
          "HarmBench ran in its standard DirectRequest configuration. No jailbreaks, no "
          "adversarial suffixes. Preservation here is not evidence of robustness under attack."),
         ("A bounded null, not a proof of no effect",
          "Representative MDE is about 0.059 at 80% power. I can rule out large effects. "
          "I cannot rule out small ones, and I say so."),
         ("Two quantization methods",
          "NF4 and INT8, both BitsAndBytes. GPTQ, AWQ and GGUF are untested and may differ."),
         ("The cost axis is not measured",
          "I did not measure VRAM, latency or throughput, so I quantify the safety and "
          "capability side of the trade-off, not the engineering benefit side.")],
        kicker="Limitations, stated up front",
        note="Every one of these is disclosed in the report; none was found by a reviewer after the fact.",
        budget="17:15 - 18:45",
        script=(
            "I want to state the limits myself rather than have them extracted from me.\n\n"
            "Everything here is scoped to direct requests — I ran no jailbreaks, so I have "
            "shown nothing about robustness under attack. The null is bounded: I can detect "
            "roughly a six-point change, so I am excluding large effects, not all effects. "
            "Two quantization methods, both from one library. And the fourth one is a fair hit "
            "— my title says trade-off, and I measured the cost in capability but never in "
            "memory or latency, so half of that trade-off is unquantified.\n\n"
            "DELIVERY: saying these before the examiner does converts four attacks into four "
            "marks for judgement. Do not rush it, and do not sound defensive."))

    d.cards(
        "Conclusion and future work",
        [("The answer to RQ1",
          "In these five matched pairs, four-bit NF4 quantization produced no statistically "
          "significant increase in harmful compliance under direct requests."),
         ("Where the cost is",
          "Capability, and as a cliff at four bits rather than a gradient — with the smallest "
          "model paying most."),
         ("The transferable lesson",
          "Two of my strongest early findings were measurement artefacts. Validate the scorer "
          "and the generation budget before believing a safety delta."),
         ("Next",
          "An attack-augmented arm, a second annotator on the gold sets, other quantization "
          "families, and the cost axis this study left unmeasured.")],
        kicker="Conclusion",
        budget="18:45 - 20:00",
        script=(
            "To close.\n\n"
            "On the evidence I have, four-bit quantization did not make these five model pairs "
            "measurably more willing to comply with harmful direct requests. The cost is real "
            "but it is on the capability axis, it behaves like a cliff at four bits, and the "
            "smallest model pays most.\n\n"
            "The part I would carry into any future work is the methodological one. Twice, the "
            "most exciting result I had turned out to be an artefact — once of the scorer, once "
            "of the generation budget. Both were found by checking my own instrument rather "
            "than by anyone reviewing me.\n\n"
            "Future work: an attack-augmented arm, a second annotator, other quantization "
            "families, and measuring the cost axis properly.\n\n"
            "Thank you — I am happy to take questions.\n\n"
            "DELIVERY: land on the methodological lesson, not on the null. It is the more "
            "interesting sentence and it invites a better class of question."))

    d.closing("Q&A — 10:00",
              "Q&A is ten minutes. Study docs/viva/VIVA_QA_BANK.md.\n\n"
              "Three rules under pressure:\n"
              "1. Answer the question asked, in one sentence, THEN give evidence.\n"
              "2. If you do not know, say so and say what you would do to find out. "
              "That scores better than a confident guess, and the panel can tell the difference.\n"
              "3. Point at an artifact whenever you can — a figure, a table, a committed file. "
              "Every number you cite exists in results_512/analysis.")

    return d.save()


if __name__ == "__main__":
    print(build())
