// Agentic-AI Workflow Report builder — Word document via docx-js
// A SEPARATE deliverable: how the student used agentic AI tools (under human
// direction + verification) to accelerate the project. Does NOT touch the FYP
// report or thesis. Mirrors the house style of scripts/build_fyp_report.js.
//
// Figures are generated reproducibly at build time: light-theme SVGs (literal
// colours + system fonts) are rasterised to PNG by @resvg/resvg-js into
// docs/figures/agentic/, then embedded.
//
// Build:  NODE_PATH=$(npm root -g) node scripts/build_agentic_report.js
// (docx is installed globally; @resvg/resvg-js is a local dependency.)

const fs = require("fs");
const path = require("path");
const { Resvg } = require("@resvg/resvg-js");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TabStopType,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun,
} = require("docx");

// ------------------------------------------------------------
// Document constants (US Letter, 1in margins) — house style
// ------------------------------------------------------------
const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360
const FONT = "Calibri", MONO = "Consolas";
const ACCENT = "2563EB";  // signal blue
const COL_BORDER = { style: BorderStyle.SINGLE, size: 4, color: "C9D0DE" };
const TABLE_BORDERS = {
  top: COL_BORDER, bottom: COL_BORDER, left: COL_BORDER, right: COL_BORDER,
  insideHorizontal: COL_BORDER, insideVertical: COL_BORDER,
};
const CELL_MARGINS = { top: 70, bottom: 70, left: 120, right: 120 };
const HEADER_SHADE = { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

const FIGDIR = path.join(__dirname, "..", "docs", "figures", "agentic");
fs.mkdirSync(FIGDIR, { recursive: true });

// ============================================================
// FIGURE GENERATION — light-theme SVG (literal colours, system
// fonts) → PNG via resvg. Keeps the Word doc fully self-contained.
// ============================================================
const C = {
  ink: "#0f1521", text: "#161d29", soft: "#39434f", muted: "#73809a", faint: "#9aa3b2",
  line: "#e6e9ef", card: "#ffffff", card2: "#f4f6fa", paper: "#fbfcfe",
  signal: "#2563eb", cyan: "#0891b2", cap: "#d97706", safe: "#0a9d6e", violet: "#7c3aed", harm: "#dc2626",
};
const SANS = "Helvetica, Arial, sans-serif";
const SERIF = "Georgia, 'Times New Roman', serif";

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function tx(x, y, s, { size = 16, fill = C.text, anchor = "start", weight = "normal", font = SANS, italic = "normal", ls } = {}) {
  return `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" fill="${fill}" text-anchor="${anchor}" font-weight="${weight}" font-style="${italic}"${ls ? ` letter-spacing="${ls}"` : ""}>${esc(s)}</text>`;
}
function box(x, y, w, h, { r = 14, fill = C.card, stroke = C.line, sw = 1.5 } = {}) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}
function arrowR(x1, y, x2, color = C.faint) { // horizontal, points right, tip at x2
  return `<line x1="${x1}" y1="${y}" x2="${x2 - 9}" y2="${y}" stroke="${color}" stroke-width="2.4"/>` +
    `<polygon points="${x2},${y} ${x2 - 11},${y - 7} ${x2 - 11},${y + 7}" fill="${color}"/>`;
}
function svgWrap(W, H, inner) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<rect width="${W}" height="${H}" fill="${C.paper}"/>` + inner + `</svg>`;
}

// --- Figure 1: the three-rung ladder ---
function figLadder() {
  const W = 1100, H = 340; let s = "";
  // rung 1
  s += box(20, 80, 300, 160, { stroke: C.faint });
  s += tx(44, 112, "RUNG 1", { size: 14, fill: C.muted, font: SERIF, weight: "bold", ls: 1.5 });
  s += tx(44, 152, "Language model", { size: 23, fill: C.ink, weight: "bold" });
  s += tx(44, 184, "Predicts the next words.", { size: 16, fill: C.soft });
  s += tx(44, 208, "You type; it types back.", { size: 16, fill: C.soft });
  s += tx(44, 230, "e.g. a chat window", { size: 14, fill: C.muted });
  s += arrowR(326, 160, 372);
  // rung 2
  s += box(380, 64, 300, 192, { stroke: C.signal, sw: 1.8 });
  s += tx(404, 96, "RUNG 2  ·  + TOOLS", { size: 14, fill: C.signal, weight: "bold", ls: 1.2 });
  s += tx(404, 138, "AI agent", { size: 23, fill: C.ink, weight: "bold" });
  s += tx(404, 170, "Can act, not just talk:", { size: 16, fill: C.soft });
  s += tx(404, 192, "read & write files, run code,", { size: 16, fill: C.soft });
  s += tx(404, 214, "run tests, search the repo.", { size: 16, fill: C.soft });
  s += tx(404, 240, "e.g. Claude Code, Codex", { size: 14, fill: C.muted });
  s += arrowR(686, 160, 732);
  // rung 3
  s += box(740, 44, 340, 232, { stroke: C.safe, sw: 2 });
  s += tx(764, 76, "RUNG 3  ·  + RULES, MEMORY, CHECKS", { size: 13, fill: C.safe, weight: "bold", ls: 1 });
  s += tx(764, 118, "Agent harness", { size: 23, fill: C.ink, weight: "bold" });
  s += tx(764, 150, "The agent, but fenced in by", { size: 16, fill: C.soft });
  s += tx(764, 172, "house rules, a shared memory,", { size: 16, fill: C.soft });
  s += tx(764, 194, "guardrails and a verification", { size: 16, fill: C.soft });
  s += tx(764, 216, "gate — safe on real work.", { size: 16, fill: C.soft });
  s += tx(764, 248, "the subject of this report", { size: 14, fill: C.muted });
  return svgWrap(W, H, s);
}

// --- Figure 2: the working loop (horizontal flow + return) ---
function figLoop() {
  const W = 1100, H = 330; let s = "";
  const boxes = [
    { x: 20, c: C.signal, n: "1", t: "You decide", d: "question · method · the call", you: true },
    { x: 290, c: C.cyan, n: "2", t: "Agent drafts", d: "code · tests · docs · analysis" },
    { x: 560, c: C.cap, n: "3", t: "Verify", d: "tests · judge · skeptic" },
    { x: 830, c: C.safe, n: "4", t: "Record", d: "logbook · locked evidence", you: true },
  ];
  const bw = 250, by = 64, bh = 96;
  boxes.forEach((b, i) => {
    s += box(b.x, by, bw, bh, { stroke: b.c, sw: 1.8, fill: b.you ? "#f3f7ff" : C.card });
    s += `<circle cx="${b.x + 30}" cy="${by + 34}" r="15" fill="${b.c}" fill-opacity="0.16"/>`;
    s += tx(b.x + 30, by + 39, b.n, { size: 16, fill: b.c, weight: "bold", anchor: "middle", font: SERIF });
    s += tx(b.x + 56, by + 34, b.t, { size: 17, fill: C.text, weight: "bold" });
    s += tx(b.x + 56, by + 60, b.d, { size: 13, fill: C.soft });
    if (b.you) s += tx(b.x + 56, by + 80, "← you", { size: 12.5, fill: b.c, weight: "bold" });
    if (i < boxes.length - 1) s += arrowR(b.x + bw, by + bh / 2, b.x + bw + 20);
  });
  // return arrow under all four, from box4 bottom back to box1 bottom
  const y0 = by + bh, yb = y0 + 56;
  s += `<path d="M955 ${y0} L955 ${yb} L145 ${yb} L145 ${y0 + 11}" fill="none" stroke="${C.faint}" stroke-width="2.4"/>`;
  s += `<polygon points="145,${y0} 138,${y0 + 13} 152,${y0 + 13}" fill="${C.faint}"/>`;
  s += tx(550, yb - 8, "You open and close the loop — it never skips verification.", { size: 14, fill: C.muted, anchor: "middle", italic: "italic", font: SERIF });
  return svgWrap(W, H, s);
}

// --- Figure 3: the six-layer control plane + protected band ---
function figHarness() {
  const W = 1120, H = 560; let s = "";
  s += tx(40, 30, "A FRESH AGENT ENTERS HERE", { size: 13, fill: C.signal, weight: "bold", ls: 1.2 });
  s += `<line x1="150" y1="38" x2="150" y2="62" stroke="${C.signal}" stroke-width="2.4"/><polygon points="150,66 143,53 157,53" fill="${C.signal}"/>`;
  const layers = [
    ["1", "Law — the house rules", "AGENTS.md + CLAUDE.md — read the log first; never edit raw evidence."],
    ["2", "State — the single source of truth", "PROJECT_LOG.md: the one living logbook of what's done and decided."],
    ["3", "Task context — small, on demand", "Skills + task packets, loaded only when relevant — not one giant prompt."],
    ["4", "Isolation — specialist reviewers", "Subagents audit one narrow concern, then hand findings back."],
    ["5", "Hooks — automatic tripwires", "Warn before touching raw data; refresh notes; sanity-check on stop."],
    ["6", "Finish gate — final verification", "make agent-check: 8 machine checks must pass before any change is “done.”"],
  ];
  let y = 72; const lh = 70, lw = 580;
  layers.forEach((L, i) => {
    const last = i === 5;
    s += box(40, y, lw, lh - 8, { r: 11, stroke: last ? C.safe : C.line, sw: last ? 1.8 : 1.4, fill: last ? "#eefaf4" : C.card });
    s += `<circle cx="74" cy="${y + 31}" r="15" fill="${last ? C.safe : C.signal}" fill-opacity="0.15"/>`;
    s += tx(74, y + 36, L[0], { size: 15, fill: last ? C.safe : C.signal, weight: "bold", anchor: "middle", font: SERIF });
    s += tx(102, y + 28, L[1], { size: 16, fill: C.text, weight: "bold" });
    s += tx(102, y + 50, L[2], { size: 12.5, fill: C.soft });
    y += lh;
  });
  // protected band
  const bx = 648, bw = 432, bt = 72, bbh = y - 8 - bt;
  s += box(bx, bt, bw, bbh, { r: 13, fill: "#fdf1f1", stroke: "#e9c6c6", sw: 1.4 });
  s += tx(bx + bw / 2, bt + 30, "WHAT THE HARNESS PROTECTS", { size: 13, fill: C.harm, weight: "bold", anchor: "middle", ls: 1 });
  s += `<line x1="${bx + 22}" y1="${bt + 44}" x2="${bx + bw - 22}" y2="${bt + 44}" stroke="#e9c6c6"/>`;
  const items = [
    [C.harm, "Immutable raw evidence", "raw.jsonl · summary.json — hash-locked"],
    [C.violet, "Derived sidecars", "scores.v2.* · scores.judge.* — corrections"],
    [C.signal, "Analysis & report", "re-derived, never hand-typed"],
    [C.cap, "Redaction boundary", "no raw harmful text leaves — IDs + labels only"],
  ];
  let iy = bt + 78;
  items.forEach(it => {
    s += `<rect x="${bx + 24}" y="${iy - 12}" width="12" height="12" rx="3" fill="${it[0]}"/>`;
    s += tx(bx + 46, iy, it[1], { size: 14.5, fill: C.text, weight: "bold" });
    s += tx(bx + 46, iy + 20, it[2], { size: 12, fill: C.soft });
    iy += 60;
  });
  s += `<line x1="620" y1="${bt + bbh / 2}" x2="646" y2="${bt + bbh / 2}" stroke="${C.safe}" stroke-width="2.4"/>`;
  return svgWrap(W, H, s);
}

// --- Figure 4: memory layers ---
function figMemory() {
  const W = 1100, H = 320; let s = "";
  // centre
  s += box(390, 84, 320, 150, { r: 16, fill: "#eef3ff", stroke: C.signal, sw: 2 });
  s += tx(550, 122, "PROJECT_LOG.md", { size: 21, fill: C.ink, weight: "bold", anchor: "middle" });
  s += tx(550, 148, "SINGLE SOURCE OF TRUTH", { size: 13, fill: C.signal, weight: "bold", anchor: "middle", ls: 1 });
  s += tx(550, 178, "status · open actions (T#)", { size: 13.5, fill: C.soft, anchor: "middle" });
  s += tx(550, 200, "decisions (D#) · changelog", { size: 13.5, fill: C.soft, anchor: "middle" });
  // left two
  s += box(40, 78, 286, 74, { r: 12, stroke: C.line });
  s += tx(64, 110, "HANDOFF.md", { size: 16, fill: C.text, weight: "bold" });
  s += tx(64, 132, "next-session bridge · “verify first”", { size: 12.5, fill: C.soft });
  s += box(40, 166, 286, 74, { r: 12, stroke: C.line });
  s += tx(64, 198, "todo.md", { size: 16, fill: C.text, weight: "bold" });
  s += tx(64, 220, "tactical resumable backlog", { size: 12.5, fill: C.soft });
  s += arrowR(328, 115, 388);
  s += arrowR(328, 203, 388, C.faint);
  s += tx(358, 150, "point back to", { size: 11.5, fill: C.muted, anchor: "middle" });
  // right
  s += box(774, 84, 286, 150, { r: 12, fill: "#eefaf4", stroke: "#bfe6d4", sw: 1.4 });
  s += tx(796, 116, "Every change writes:", { size: 15, fill: C.text, weight: "bold" });
  s += tx(796, 144, "•  one changelog row", { size: 13, fill: C.soft });
  s += tx(796, 166, "•  a decision, if one was made", { size: 13, fill: C.soft });
  s += tx(796, 188, "•  an open-action update", { size: 13, fill: C.soft });
  s += tx(796, 214, "enforced by the finish gate", { size: 12, fill: C.muted });
  s += `<line x1="772" y1="159" x2="714" y2="159" stroke="${C.safe}" stroke-width="2.4"/><polygon points="710,159 723,152 723,166" fill="${C.safe}"/>`;
  return svgWrap(W, H, s);
}

// --- Figure 5: test-suite growth ---
function figTests() {
  const TESTS = [
    [186, "v2 scorer + CIs"], [189, "McNemar"], [192, "multi-seed infra"], [215, "latent hardening"],
    [218, "2nd judge"], [223, "ARC benchmark"], [246, "cross-family"], [261, "mechanism probe"],
    [282, "INT8 groundwork"], [289, "INT8 run"], [295, "scorer audit"], [306, "full audit + guards"],
  ];
  const W = 1100, H = 420, m = { t: 34, r: 34, b: 110, l: 76 };
  const lo = 170, hi = 315, n = TESTS.length;
  const X = i => m.l + i * ((W - m.l - m.r) / (n - 1));
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  let s = "";
  for (let g = 180; g <= 300; g += 40) {
    s += `<line x1="${m.l}" y1="${Y(g)}" x2="${W - m.r}" y2="${Y(g)}" stroke="${C.line}"/>`;
    s += tx(m.l - 12, Y(g) + 5, String(g), { size: 15, fill: C.muted, anchor: "end" });
  }
  let d = `M${X(0)} ${Y(TESTS[0][0])}`;
  TESTS.forEach((p, i) => { if (i) d += ` L${X(i)} ${Y(p[0])}`; });
  s += `<path d="${d} L${X(n - 1)} ${H - m.b} L${X(0)} ${H - m.b} Z" fill="${C.signal}" fill-opacity="0.09"/>`;
  s += `<path d="${d}" fill="none" stroke="${C.signal}" stroke-width="3" stroke-linejoin="round"/>`;
  TESTS.forEach((p, i) => {
    const cx = X(i), cy = Y(p[0]);
    s += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="${C.signal}"/>`;
    if (i === 0 || i === n - 1) s += tx(cx, cy - 16, String(p[0]), { size: 19, fill: C.ink, weight: "bold", anchor: i === 0 ? "start" : "end", font: SERIF });
    s += `<g transform="rotate(-40 ${cx} ${H - m.b + 22})">` + tx(cx, H - m.b + 22, p[1], { size: 13.5, fill: C.muted, anchor: "end" }) + `</g>`;
  });
  s += tx(m.l, 24, "Automated tests (all passing)", { size: 14, fill: C.soft, weight: "bold" });
  return svgWrap(W, H, s);
}

// --- Figure 6: changelog authorship ---
function figAuthors() {
  const W = 1000, H = 250, m = { t: 30, r: 70, b: 46, l: 210 };
  const AUTH = [["Claude Code", 83, C.signal], ["Codex", 16, C.cyan]];
  const max = 92, plotW = W - m.l - m.r, X = v => m.l + (v / max) * plotW;
  const rowH = (H - m.t - m.b) / AUTH.length;
  let s = "";
  AUTH.forEach((a, i) => {
    const y = m.t + i * rowH + rowH / 2;
    s += tx(m.l - 16, y + 6, a[0], { size: 18, fill: C.text, weight: "bold", anchor: "end" });
    s += `<rect x="${m.l}" y="${y - 17}" width="${Math.max(3, X(a[1]) - m.l)}" height="34" rx="7" fill="${a[2]}" fill-opacity="0.9"/>`;
    s += tx(X(a[1]) + 12, y + 6, String(a[1]), { size: 18, fill: a[2], weight: "bold", font: SERIF });
  });
  s += tx(m.l, H - 14, "changelog rows authored via each agent (execution volume, not scientific credit)", { size: 13, fill: C.muted });
  return svgWrap(W, H, s);
}

function rasterize(name, svg, width) {
  const r = new Resvg(svg, { fitTo: { mode: "width", value: width }, font: { loadSystemFonts: true }, background: "white" });
  const png = r.render().asPng();
  fs.writeFileSync(path.join(FIGDIR, name), png);
}
console.log("Rendering figures…");
rasterize("fig1_ladder.png", figLadder(), 1700);
rasterize("fig2_loop.png", figLoop(), 1700);
rasterize("fig3_harness.png", figHarness(), 1700);
rasterize("fig4_memory.png", figMemory(), 1700);
rasterize("fig5_tests.png", figTests(), 1700);
rasterize("fig6_authors.png", figAuthors(), 1600);

// ============================================================
// DOCX HELPERS (house style)
// ============================================================
function run(text, o = {}) {
  return new TextRun({ text, font: o.mono ? MONO : FONT, size: o.size || 22, bold: o.bold, italics: o.italics, color: o.color });
}
function P(children, o = {}) {
  return new Paragraph({
    children: Array.isArray(children) ? children : [run(children, o)],
    spacing: { after: o.after !== undefined ? o.after : 130, line: o.line || 300 },
    alignment: o.alignment, pageBreakBefore: o.pageBreakBefore || false,
  });
}
function PJ(text, o = {}) {
  return new Paragraph({
    children: Array.isArray(text) ? text : [run(text, o)],
    spacing: { after: o.after || 150, line: 300 }, alignment: AlignmentType.JUSTIFIED,
  });
}
function H1(text, brk = true) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run(text, { size: 32, bold: true })], spacing: { before: 360, after: 200 }, pageBreakBefore: brk });
}
function H2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(text, { size: 26, bold: true })], spacing: { before: 280, after: 140 } }); }
function H3(text) { return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [run(text, { size: 23, bold: true })], spacing: { before: 220, after: 110 } }); }
function Bullet(children, level = 0) {
  return new Paragraph({ numbering: { reference: "bullets", level }, children: Array.isArray(children) ? children : [run(children)], spacing: { after: 70, line: 280 } });
}
let __num = 0;
function numberedList(items) {
  __num += 1; const ref = `nl${__num}`;
  return items.map(it => new Paragraph({ numbering: { reference: ref, level: 0 }, children: Array.isArray(it) ? it : [run(it)], spacing: { after: 80, line: 282 } }));
}
let __fig = 0;
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function Figure(file, caption, dispW = 540) {
  __fig += 1;
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const { w, h } = pngSize(buf);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: dispW, height: Math.round(dispW * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
      children: [new TextRun({ text: `Figure ${__fig}.  `, font: FONT, size: 18, italics: true, bold: true, color: "555555" }),
                 new TextRun({ text: caption, font: FONT, size: 18, italics: true, color: "555555" })] }),
  ];
}
function shadedCell(children, { width, fill, borders, margins } = {}) {
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    borders, margins: margins || CELL_MARGINS, children,
  });
}
// Callout: full-width 1-cell table, thick coloured left border + light fill.
function callout(label, bodyRuns, accent, fill) {
  const borders = { top: NONE, bottom: NONE, right: NONE, left: { style: BorderStyle.SINGLE, size: 22, color: accent } };
  const children = [
    new Paragraph({ spacing: { after: 50 }, children: [new TextRun({ text: label.toUpperCase(), font: MONO, size: 16, bold: true, color: accent })] }),
    new Paragraph({ spacing: { after: 0, line: 290 }, children: Array.isArray(bodyRuns) ? bodyRuns : [run(bodyRuns, { size: 21 })] }),
  ];
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ children: [shadedCell(children, { width: CONTENT_W, fill: fill || "F4F7FC", borders, margins: { top: 120, bottom: 120, left: 200, right: 160 } })] })],
  });
}
function plain(text) { return [callout("In plain terms", text, "0891B2", "ECF7F9"), P("", { after: 60 })]; }

function buildTable(headers, rows, widths) {
  if (!widths) { const w = Math.floor(CONTENT_W / headers.length); widths = headers.map(() => w); widths[widths.length - 1] += CONTENT_W - w * headers.length; }
  const headerRow = new TableRow({ tableHeader: true, children: headers.map((hd, i) =>
    shadedCell([new Paragraph({ children: [run(hd, { bold: true, size: 20 })] })], { width: widths[i], fill: "D9E2F3" })) });
  const bodyRows = rows.map(cells => new TableRow({ children: cells.map((cell, i) =>
    shadedCell([Array.isArray(cell) ? new Paragraph({ children: cell, spacing: { line: 276 } }) : new Paragraph({ children: [run(cell, { size: 20 })], spacing: { line: 276 } })], { width: widths[i] })) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, borders: TABLE_BORDERS, rows: [headerRow, ...bodyRows] });
}
// Stat band: borderless 4-col single row, big number + label.
function statBand(items) {
  const w = Math.floor(CONTENT_W / items.length); const widths = items.map(() => w); widths[widths.length - 1] += CONTENT_W - w * items.length;
  const cells = items.map((it, i) => shadedCell([
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: it[0], font: "Georgia", size: 52, bold: true, color: "0F1521" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0, line: 250 }, children: [run(it[1], { size: 17, color: "39434F" })] }),
  ], { width: widths[i], fill: "F4F6FA", borders: { top: NONE, bottom: NONE, left: { style: BorderStyle.SINGLE, size: 6, color: "FFFFFF" }, right: { style: BorderStyle.SINGLE, size: 6, color: "FFFFFF" } }, margins: { top: 160, bottom: 160, left: 80, right: 80 } }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE }, rows: [new TableRow({ children: cells })] });
}
const B = (t) => run(t, { bold: true });          // bold run
const T = (t) => run(t);                            // plain run
const code = (t) => run(t, { mono: true, size: 20 });
const sub = (t) => new TextRun({ text: t, break: 1, italics: true, size: 18, color: "73809A", font: FONT }); // 2nd line in a cell
// Multi-line cell: array of runs separated by hard line breaks (docx ignores "\n" in a run).
function lines(items, o = {}) { return items.map((t, i) => new TextRun({ text: t, font: o.mono ? MONO : FONT, size: o.size || 20, bold: o.bold, ...(i ? { break: 1 } : {}) })); }

// ============================================================
// CONTENT
// ============================================================
const body = [];
const push = (...x) => x.forEach(e => Array.isArray(e) ? body.push(...e) : body.push(e));

// ---- Cover ----
push(
  new Paragraph({ children: [new TextRun(" ")], spacing: { after: 700 } }),
  P([run("NANYANG TECHNOLOGICAL UNIVERSITY", { size: 28, bold: true })], { alignment: AlignmentType.CENTER, after: 80 }),
  P([run("College of Computing and Data Science", { size: 24 })], { alignment: AlignmentType.CENTER, after: 560 }),
  P([run("FINAL YEAR PROJECT", { size: 22, bold: true })], { alignment: AlignmentType.CENTER, after: 70 }),
  P([run("Methods & Tooling Report", { size: 22, italics: true })], { alignment: AlignmentType.CENTER, after: 440 }),
  P([run("Engineering an AI-Accelerated Research Workflow", { size: 36, bold: true, color: "0F1521" })], { alignment: AlignmentType.CENTER, after: 110 }),
  P([run("How agentic AI tools were used — under human direction and continuous verification — to accelerate a quantized-LLM safety study", { size: 24, italics: true, color: "39434F" })], { alignment: AlignmentType.CENTER, after: 700 }),
  P([run("Project Code:  CCDS25-1136", { size: 24, bold: true })], { alignment: AlignmentType.CENTER, after: 200 }),
  P([run("Student:  TAN UEI HORNG  (UTAN001)", { size: 26, bold: true })], { alignment: AlignmentType.CENTER, after: 110 }),
  P([run("Email:  UTAN001@e.ntu.edu.sg", { size: 22 })], { alignment: AlignmentType.CENTER, after: 110 }),
  P([run("Supervisor:  Dr. Zhang Jiehuang  (jiehuang.zhang@ntu.edu.sg)", { size: 22 })], { alignment: AlignmentType.CENTER, after: 520 }),
  P([run("Document date:  26 June 2026", { size: 24, bold: true })], { alignment: AlignmentType.CENTER, after: 90 }),
  P([run("A companion to the main FYP report and thesis. It documents the method by which the work was produced; it does not restate or alter any scientific result.", { size: 20, italics: true, color: "555555" })], { alignment: AlignmentType.CENTER }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---- TOC ----
push(H1("Table of Contents", false), new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));

// ---- 0. Executive summary ----
push(H1("0.  Executive Summary"));
push(PJ("This is a separate, standalone report. It does not change or restate the scientific findings of the project — those live in the main FYP report and thesis. Its single purpose is to explain, in accessible terms, how the work was produced: the agentic AI tooling that was built and operated to do a rigorous research project faster, and the safeguards that keep the results trustworthy."));
push(callout("The one thing to take away", [
  T("This project did not ask an AI to “do the research.” AI coding agents were used the way an engineer uses power tools: to execute decisions quickly. A "),
  B("verification harness"), T(" was then built around those tools — a set of automatic checks, immutable evidence, and independent review — that repeatedly "),
  B("caught the AI’s mistakes"), T(". The most important scientific correction in this project (Section 9) happened "),
  run("because", { italics: true }), T(" of that harness, not in spite of the AI. The trustworthiness of the results rests on human judgement plus machine-checkable rigor — not on trusting a model."),
], "D97706", "FCF5EA"));
push(P("", { after: 80 }));
push(statBand([["306", "automated tests guarding the framework (all passing)"], ["120", "raw evidence files locked by cryptographic hash — never edited"], ["8", "checks in the one command that must pass before a change is “done”"], ["37", "dated decisions logged with rationale, append-only and tamper-evident"]]));
push(P([run("Every figure in this report is sourced from the repository’s committed evidence: ", { size: 19, color: "555555" }), code("docs/PROJECT_LOG.md"), run(" (the project’s single source of truth), ", { size: 19, color: "555555" }), code("configs/artifact_policy.yaml"), run(", the harness code, and the test suite. Where a number could drift, the live value is one command away (Section 12).", { size: 19, color: "555555" })], { after: 60 }));

// ---- 1. What agentic AI is ----
push(H1("1.  What “Agentic AI” Actually Is"));
push(PJ("If you have used a chat assistant such as ChatGPT, you have used a language model: you type, it types back. An agentic AI tool is the next step — the same kind of model, but wired up so it can take real actions on a real project. There are three rungs (Figure 1)."));
push(Figure("fig1_ladder.png", "The three rungs. A chat model becomes an agent when you give it tools; an agent becomes a harness when you give it rules, memory and checks. This report is about rung 3.", 560));
push(plain("A language model alone is a clever autocomplete. An agent can press the buttons. A harness is the workshop built around the agent — with safety rails, a logbook, and a final inspection — so that a fast but occasionally-wrong assistant can work on a real scientific project without breaking it or quietly corrupting the data."));
push(PJ("The catch — and the reason the rest of this report matters — is that these tools are fast but fallible. They can write a hundred lines of correct code in seconds and then state something subtly wrong with complete confidence. An agentic workflow is only as trustworthy as the checks wrapped around it. The methodological contribution described here is building those checks."));

// ---- 2. The tools ----
push(H1("2.  The Tools, and Why"));
push(PJ("Two commercial coding agents were used, plus an in-repository harness built specifically for this project. Two different agents from two different vendors were run deliberately, so that neither becomes a single point of failure — and so each can review the other’s work."));
push(buildTable(
  ["Tool", "What it is", "How it was used here"],
  [
    [[B("Claude Code"), sub("Anthropic · Opus model")], "A command-line AI coding agent that reads and edits the repository, runs commands, and runs tests.", "Primary author of the framework code, analysis scripts, tests and docs. Logged 83 of the ~99 agent-authored changelog entries."],
    [[B("Codex"), sub("OpenAI · CLI agent")], "A second, independent coding agent with its own subagent and hook system.", "Second author and cross-checker (16 entries). The repository’s specialist reviewers and lifecycle hooks (Section 6) are built on Codex primitives."],
    [[B("The in-repo harness"), sub("built by the student")], "Not a product — a set of files that turn the project’s rules into machine-checkable contracts.", "Orients each agent, scopes its task, checks its work, and protects the evidence. Everything from Section 4 onward describes this."],
  ],
  [2100, 3400, 3860]));
push(P("", { after: 40 }));
push(callout("Why two agents", [T("It is the same instinct as a "), B("second opinion"), T(" in medicine. Two independent agents, reading the same house rules, are far less likely to share the same blind spot. A built-in check fails the project if their shared rulebook ever drifts apart (Section 4), so they genuinely stay in step.")], "2563EB", "EFF3FB"));
push(P("", { after: 80 }));
push(PJ("“Logged 83 entries” is a count of changelog rows, not a measure of scientific contribution. It indicates how much execution ran through the agent — typing, refactoring, test-writing, document-syncing — the mechanical volume the tooling absorbed so that human time could be spent on the science. The judgement calls behind those entries were the student’s."));

// ---- 3. Working loop ----
push(H1("3.  How the Work Actually Happens"));
push(PJ("Every change moves through the same four-beat loop (Figure 2). The human opens and closes it; the agent only ever does the middle. Nothing an agent produces is believed until it has been verified and recorded."));
push(Figure("fig2_loop.png", "The working loop. The agent only occupies step 2. The human opens the loop (decide) and closes it (verify and record) — and the loop never skips verification.", 560));
push(H3("Who owns what"));
push(PJ("The honest division of labour. The left column is irreducibly human — it is the research. The right column is what the tooling absorbed."));
push(buildTable(
  ["The student owns (the science)", "The agent accelerates (the execution)"],
  [
    [lines([
       "•  The research question and hypotheses",
       "•  The matched-pair methodology and controls",
       "•  Which models, benchmarks and metrics",
       "•  Interpreting results and what counts as significant",
       "•  Judgement calls (e.g. promoting the judge to primary)",
       "•  The scientific narrative and every claim in the report",
       "•  Communication with the supervisor",
     ]),
     lines([
       "•  Writing boilerplate: loaders, plugins, SLURM scripts",
       "•  Refactoring and keeping code consistent",
       "•  Generating and maintaining the test suite",
       "•  Searching the codebase and drafting documentation",
       "•  Catching inconsistencies across files",
       "•  Running adversarial self-audits on request",
       "•  Keeping the logbook and handoff notes in sync",
     ])],
  ],
  [4680, 4680]));
push(P([run("The dividing line:  ", { bold: true }), T("the agent proposes; the human disposes. No result, label, or conclusion enters the project until it has cleared verification and been recorded — which is exactly what Sections 4–9 enforce.")], { after: 120 }));

// ---- 4. Harness ----
push(H1("4.  The Agent Harness: an Operating System for AI Assistants"));
push(PJ("A coding agent starts every session with no memory of the last one. Left alone it will re-learn the project from scratch, contradict earlier decisions, and occasionally touch something it should not. The harness is six thin layers — all of them simply files in the repository — that orient a fresh agent, scope it, check it, and protect the evidence (Figure 3)."));
push(Figure("fig3_harness.png", "The harness control plane. An agent passes top-to-bottom; the protected-evidence band on the right is what these layers exist to defend. Adapted from docs/architecture/fyp_quant_agent_harness_architecture.svg.", 560));
push(plain("Think of onboarding a brilliant but forgetful new assistant every single morning. The harness is the printed induction pack on their desk: the rules, the project logbook, the one-page brief for today’s task, a colleague who double-checks their work, a few safety interlocks, and a sign-off checklist they cannot skip. It is the difference between a helpful assistant and a hazard."));
push(PJ("Crucially, every layer is a plain file under version control — diffable, reviewable, and auditable. There is no hidden automation: to see exactly what an agent is told, one simply opens the file."));

// ---- 5. Single source of truth ----
push(H1("5.  One Logbook the Whole Project Trusts"));
push(PJ("The biggest risk with a forgetful, fast assistant is drift — yesterday’s decision quietly contradicted today. The cure is a single, authoritative logbook that every change must update. In this project that is docs/PROJECT_LOG.md, and the rule is absolute: nothing is committed without a matching log entry (Figure 4)."));
push(Figure("fig4_memory.png", "Three memory layers, one authority. HANDOFF (a session bridge) and todo.md (a tactical buffer) both point back to PROJECT_LOG — so convenience notes never become competing records.", 520));
push(H3("What makes the logbook trustworthy"));
push(Bullet([B("Decisions are never edited. "), T("When a decision is reversed, a new dated entry references the old one — so the reasoning trail is tamper-evident, not rewritten. There are 37 such decisions (D1–D37).")]));
push(Bullet([B("The log is part of the change. "), T("One row per change records when, which files, what changed, why, whether the report was rebuilt, and who. The git history shows nearly every feature commit paired with a log row.")]));
push(Bullet([B("Tasks and decisions are numbered. "), T("Open items are T<N>, decisions are D<N>, never renumbered — so any conversation or commit can reference one unambiguously.")]));
push(Bullet([B("It cannot quietly fall behind. "), T("The finish gate fails if the working tree changed but the log did not, and a scanner flags out-of-date “current-state” text in the bridge notes.")]));
push(plain("This is a laboratory notebook with the discipline turned up. Because two different AI agents and the student all read and write the same logbook — and a check refuses any change that skips it — there is exactly one place to learn “where the project stands,” and it is always current."));

// ---- 6. Automation inventory ----
push(H1("6.  Skills, Specialist Reviewers, and Tripwires"));
push(PJ("Three kinds of small, reusable automation were configured. None of them run the science; they make the routine parts fast and the risky parts safe. Each is a file that can be shown to a reviewer."));
push(H3("Skills — focused briefings, loaded on demand"));
push(PJ("Instead of one enormous prompt, the agent pulls in only the short briefing relevant to the current job."));
push(buildTable(["Skill", "What it routes"], [
  ["fyp-report-audit", "Report work: verify every claim against the analysis files, rebuild the report when results change, never leak raw harmful text."],
  ["fyp-second-judge", "Adding a second, independent safety judge — writing new redacted sidecars without overwriting existing evidence."],
  ["fyp-harness-maintenance", "Changes to the harness itself: prefer on-demand context and tests over long instructions; never weaken a privacy or immutability check."],
  ["fyp-meetup-brief", "Keeping any explanation of the workflow grounded in real files and commands — no overclaiming, no exposed raw data."],
], [2600, 6760]));
push(H3("Subagents — five specialist reviewers, each in a clean context"));
push(PJ("A reviewer that examines only one narrow risk catches more than a generalist skimming everything. Each hands findings back; the human decides the edits."));
push(buildTable(["Subagent (read-only reviewer)", "Focus"], [
  ["artifact-guardian", "Any mutation of raw evidence, missing hashes, or a path that would copy raw harmful text into a doc or chat."],
  ["report-auditor", "Stale claims, report-versus-log drift, and judge-versus-proxy mistakes across the documents."],
  ["judge-reviewer", "That the official safety classifier stays primary and that any second judge writes fresh, redacted sidecars only."],
  ["tc1-ops", "That cluster work is routed safely (batch jobs only, offline mode preserved) and never violates HPC head-node policy."],
  ["meetup-story", "Turning repository evidence into a grounded human explanation — without overclaiming automation."],
], [2600, 6760]));
push(H3("Hooks — three automatic tripwires"));
push(buildTable(["Fires…", "And does"], [
  ["Before a tool runs", "If a command is about to touch raw.jsonl or summary.json, it warns to use a derived sidecar instead."],
  ["Before memory is compacted", "Regenerates the handoff and dashboard notes so nothing is lost when the agent’s context is summarised."],
  ["When the agent stops", "Runs a quick status and sanity check, and reminds that the full finish gate still has to pass."],
], [2900, 6460]));
push(plain("Skills are job sheets, subagents are specialist inspectors, hooks are smoke detectors. Together they mean the boring work is automated and the dangerous work trips an alarm before it happens — without the human having to remember every rule every time."));

// ---- 7. Guardrails ----
push(H1("7.  Guardrails That Keep AI-Accelerated Research Honest"));
push(PJ("Speed is worthless if it quietly corrupts the evidence. Three guardrails — immutable evidence, a redaction boundary, and end-to-end reproducibility — mean an AI can move fast through this project without being able to fake, mutate, or lose a single result."));
push(buildTable(["Guardrail", "What it guarantees"], [
  [[B("1.  Immutable evidence")], "The 120 raw result files from the cluster are locked by SHA-256 hash and never edited. Every metric is re-derived from them by script; any silent change is caught by the finish gate."],
  [[B("2.  Redaction boundary")], "Committed safety-judge files contain only prompt IDs and yes/no labels — never the raw harmful prompts or model responses. The study reproduces from the public repository without redistributing harmful text."],
  [[B("3.  Reproducibility")], "Fixed seeds (42), greedy decoding, pinned dependencies, offline cluster runs, and a 306-test suite. Re-running the analysis re-creates the published numbers byte-for-byte."],
], [2600, 6760]));
push(H3("Corrections without touching the originals"));
push(PJ("When a scoring method had to change (and it did — see Section 9), the original cluster outputs were not overwritten. The corrected scores were written into separate, auditable “sidecar” files. So the project carries both the original record and every correction, side by side, with a full trail of which is which. This is the single most important reason the headline result can be reproduced exactly from immutable files today."));
push(callout("An honest limit, stated plainly", [T("No guardrail is perfect, and the harness says so itself. The redaction scanner catches explicit leak patterns and data-shaped snippets, but it cannot "), run("prove", { italics: true }), T(" that no paraphrase of a sensitive response exists anywhere — so it is described in the project’s own self-evaluation as a strong filter, not a proof. Statistical significance is reported as nominal and uncorrected, with the multiple-comparisons caveat disclosed. Stating limits like these, rather than hiding them, is part of what makes the rest credible.")], "D97706", "FCF5EA"));

// ---- 8. Finish gate ----
push(H1("8.  The Finish Gate: One Command That Must Pass"));
push(PJ("Before any change is considered “done,” a single command — make agent-check — runs eight mechanical checks. If any one fails, the change is not finished. This is the moment where the project’s prose rules become executable: a checklist an agent cannot talk its way around."));
const checks = [
  ["Docs in sync", "the two rulebooks (AGENTS.md, CLAUDE.md) must stay byte-identical; the build fails if they ever drift."],
  ["Logbook updated", "if the working tree changed but PROJECT_LOG.md did not, fail. The log is part of the change."],
  ["Raw evidence intact", "re-hash all 120 raw files against the manifest; fail on any mutation or unrecorded new file."],
  ["No data leaks", "scan docs for raw harmful prompt/response text or sensitive data keys; fail if any slipped in."],
  ["No stale claims", "scan for superseded “current-state” phrases (e.g. “results pending”) that should have been updated."],
  ["Report fresh", "if a result-bearing file changed without the report being rebuilt, fail. Editing the report source alone will not satisfy it."],
  ["Clean diff", "fail on stray whitespace or leftover merge-conflict markers."],
  ["All tests pass", "run the full 306-test suite; fail if any test fails."],
];
push(buildTable(["✓", "Check"], checks.map(c => [[run("✓", { bold: true, color: "0A9D6E", size: 22 })], [B(c[0] + " — "), T(c[1])]]), [620, 8740]));
push(plain("This is a pre-flight checklist. A pilot does not take off on instinct; they run the list and every item must read green. make agent-check is that list for this project — and a second copy (make harness-eval) deliberately breaks the repository in known ways to prove the checklist actually catches problems."));

// ---- 9. Self-correction ----
push(H1("9.  When the Harness Caught the AI — and Changed the Science"));
push(PJ("This is the most important section for judging trust. If an AI-accelerated project never disagreed with the AI, one should be suspicious. This one disagreed repeatedly — and the disagreements were caught, verified, and used to correct the findings."));
push(callout("Decision D16 — the correction that reset the headline result", [run("“Did the model get safer — or did my scorer just mislabel it?”", { italics: true, size: 24 })], "2563EB", "EFF3FB"));
push(P("", { after: 60 }));
push(PJ([B("The headline example (D16).  "), T("The first, fast safety scorer was a pattern-matcher. It worked, but it was not trusted. The official 13-billion-parameter HarmBench classifier was run as an independent judge over the same outputs. It "), B("materially disagreed"), T(" with the fast scorer — the pattern-matcher was over-counting harmful answers. The disagreement was not cosmetic: it "), B("moved the study’s only statistically significant safety result from one model to another"), T(", and flipped the direction of the smallest model. The judge was promoted to the primary scorer and the analysis was rewritten. Because the raw evidence was immutable, the correction was made safely in sidecar files — no original was touched.")]));
push(PJ([T("That is the whole thesis of this report in one episode: "), run("the value was not the AI’s first answer — it was the verification environment that proved the first answer wrong and allowed it to be fixed without losing the original record.", { italics: true }), T(" The same pattern recurred:")]));
push(...numberedList([
  [B("A confident “story” did not survive scrutiny (D33). "), T("An appealing mechanism explanation — “4-bit quantization erodes the model’s refusals” — was tested by a dedicated adversarial critic pass. It failed: the effect was symmetric (it moved both ways) and traced to general capability softening, not a targeted attack on safety. The overclaim was caught and not written; the report states the honest, weaker finding.")],
  [B("An external audit found the scorer’s blind spot (D11). "), T("Before D16, an audit found the first pattern-matcher missed common modern refusal phrasings, inflating the harm rate. It was confirmed by hand, a corrected scorer was written, and the outputs were re-scored — which flipped two of three model labels. This is where the project’s “verify before you believe” discipline began.")],
  [B("The single best number was deliberately tempered (D23). "), T("The headline effect came from one decoding run. It was re-run across five random seeds; the average was about half the headline, and not always the same sign. Rather than keep the flattering number, the effect was reported as “the top of a range, not a fixed value.”")],
  [B("A second judge and a second benchmark, as self-checks (D26, D29). "), T("An independent second safety judge was added (it agreed on direction but flagged the result as borderline) and a second capability benchmark (which showed an earlier gap had been overstated). Both could have weakened the story; both were reported in full.")],
  [B("Two full adversarial audits, each finding re-checked by a skeptic (D36, D37). "), T("Twice, the whole repository was audited where every flagged issue had to be independently re-verified by a separate “skeptic” agent trying to disprove it. One audit raised 44 issues → 43 confirmed, 0 that invalidated any result (the skeptic correctly threw out the one false alarm). The load-bearing numbers reproduced byte-for-byte from the locked evidence.")],
]));
push(plain("A good research process is defined by how it behaves when the convenient answer is wrong. Here, the convenient answer was wrong several times — and each time it was caught by a check, a judge, or a skeptic, then corrected in the open with the original evidence preserved. That is the opposite of “the AI did it.”"));

// ---- 10. Payoff ----
push(H1("10.  The Payoff, Measured Honestly"));
push(PJ("The acceleration did not show up as “the AI wrote the project.” It showed up as depth that could be afforded: a study that grew from a basic three-pair experiment into a five-family, three-precision investigation with two independent judges, two capability benchmarks, a mechanism probe, and two full audits — each guarded by tests (Figure 5)."));
push(Figure("fig5_tests.png", "The test suite grew alongside the scope. Each rise is a robustness layer that the harness made safe to add without breaking what already worked.", 560));
push(Figure("fig6_authors.png", "Logged changes by agent — execution volume, not scientific credit. Every entry was a human-directed change.", 480));
push(PJ("The point is not “more is better.” It is that each robustness layer in Section 9 — the second judge, the multi-seed run, the second benchmark, the audits — is exactly the kind of careful, repetitive work that usually gets skipped under time pressure. The tooling made that work cheap enough to actually do, and the harness made it safe to do without destabilising the result. The acceleration was spent on rigor, not on cutting corners."));

// ---- 11. Limits ----
push(H1("11.  Limits, and How This Stayed Responsible"));
push(PJ("Using AI tooling responsibly means being clear-eyed about what it cannot do, and leaving an audit trail so a skeptical reader can check the work rather than take anyone’s word for it."));
push(buildTable(["Limit", "Mitigation"], [
  [[B("Models can be confidently wrong")], "Every load-bearing claim is verified against committed evidence, an independent judge, or a test — never accepted on the model’s say-so. Section 9 is the receipts."],
  [[B("An agent forgets between sessions")], "Mitigated by the single-source-of-truth logbook and generated handoff notes — but those notes are explicitly marked “verify before acting.”"],
  [[B("Automated checks are not proofs")], "The redaction scanner is a strong filter, not a guarantee; significance is nominal/uncorrected. Both are disclosed in the project’s own write-ups."],
  [[B("Acceleration is not authorship")], "The AI accelerated execution. The research question, design, interpretation and conclusions are the student’s, and are declared as such in the thesis’s AI-usage statement."],
], [3000, 6360]));
push(PJ("Two practices keep this honest and checkable. First, the project is fully auditable — every decision, with its rationale and date, is in the append-only log, and every result re-derives from hash-locked evidence. Second, AI usage is declared, not hidden: the main thesis carries an explicit AI-usage statement, and this report exists precisely to make the method transparent."));
push(callout("The standard held throughout", [T("A reader who distrusts AI entirely should still be able to trust this project — by ignoring the tooling and checking the evidence directly. The immutable raw files, the independent judge results, the test suite, and the dated decision log are all there to be inspected. The tooling made the work faster; the "), run("evidence", { italics: true }), T(" is what makes it true.")], "0A9D6E", "ECF7F0"));

// ---- 12. Glossary + verify ----
push(H1("12.  Glossary, and How to Check This Yourself"));
push(H3("Glossary"));
push(buildTable(["Term", "Plain meaning"], [
  ["Language model / LLM", "An AI trained to predict text. The engine inside ChatGPT-like tools."],
  ["Agent", "A language model wired up with tools so it can read files, run code, and run tests — not just chat."],
  ["Agent harness", "The rules, memory, guardrails and checks wrapped around an agent so it can work safely on a real project. The subject of this report."],
  ["Skill / task packet", "A short, focused briefing loaded only when relevant, instead of one giant prompt."],
  ["Subagent", "A specialist reviewer that audits one narrow concern in a clean context and reports back."],
  ["Hook", "An automatic tripwire that fires at a set moment (before a tool runs, before memory is summarised, when the agent stops)."],
  ["Immutable artifact", "A raw evidence file that is never edited; corrections go into separate sidecar files instead."],
  ["Redaction", "Storing only IDs and yes/no labels for safety tests — never the raw harmful text — so the study can be shared without redistributing harm."],
  ["Judge model", "A separate, stronger model used to independently score outputs, as a check on a faster first-pass scorer."],
  ["Finish gate", "One command (make agent-check) running eight checks that must all pass before a change is “done.”"],
], [2700, 6660]));
push(H3("Verify it yourself — the live repository"));
push(PJ("Nothing in this report asks for blind trust. From the project root, these read-only commands reproduce its claims:"));
push(buildTable(["Command", "What it shows"], [
  [[code("python fyp_cli.py agent-status")], "The live project truth: git state, what is pending, when the log was last updated, and the suggested next action."],
  [[code("make agent-check")], "Runs all eight checks from Section 8, including the full 306-test suite and the 120-file evidence-hash check."],
  [[code("make harness-eval")], "Deliberately breaks the repository in known ways to confirm the harness actually catches each kind of problem."],
  [[code("docs/PROJECT_LOG.md")], "The single source of truth: status, the 37 dated decisions with rationale, open actions, and the full changelog."],
], [3500, 5860]));
push(P("", { after: 80 }));
push(P([run("Companion documents.  ", { bold: true, size: 20 }), run("The scientific findings live in the main FYP report (docs/FYP_Report_2026-06-14.docx) and the standalone thesis (docs/FYP_Thesis_2026-06-18.docx). The full engineering diagrams (editable draw.io sources) are under docs/architecture/. This report documents the method by which the work was produced; it does not restate or alter any scientific result.", { size: 20, color: "555555", italics: true })]));
push(new Paragraph({ spacing: { before: 260 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: "C9D0DE" } }, children: [] }));
push(P([run("Engineering an AI-Accelerated Research Workflow", { bold: true, size: 20 }), run("  ·  Tan Uei Horng (UTAN001)  ·  CCDS25-1136  ·  Supervisor Dr. Zhang Jiehuang  ·  NTU College of Computing and Data Science  ·  Companion methods report, generated 26 June 2026.", { size: 18, color: "73809A" })], { alignment: AlignmentType.CENTER, after: 40 }));

// ============================================================
// ASSEMBLE
// ============================================================
const doc = new Document({
  creator: "TAN UEI HORNG",
  title: "Engineering an AI-Accelerated Research Workflow — CCDS25-1136",
  description: "Methods & tooling report: how agentic AI tools were used (under human direction and verification) to accelerate the CCDS25-1136 quantized-LLM safety study.",
  features: { updateFields: true },   // Word auto-populates the Table of Contents on open
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true, color: "0F1521" }, paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: "2563EB" }, paragraph: { spacing: { before: 220, after: 110 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1200, hanging: 300 } } } },
      ] },
      ...Array.from({ length: 12 }, (_, i) => ({ reference: `nl${i + 1}`, levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } },
      ] })),
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
      children: [
        new TextRun({ text: "CCDS25-1136 — TAN UEI HORNG (UTAN001)", font: FONT, size: 18, italics: true, color: "888888" }),
        new TextRun({ text: "\t" }),
        new TextRun({ text: "Agentic-AI Workflow — Methods Report", font: FONT, size: 18, italics: true, color: "888888" }),
      ] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Page ", font: FONT, size: 18 }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 }),
      new TextRun({ text: " of ", font: FONT, size: 18 }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18 }),
    ] })] }) },
    children: body,
  }],
});

const OUTPUT = path.join(__dirname, "..", "docs", "Agentic_AI_Workflow_Report_2026-06-26.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("WROTE: " + OUTPUT);
  console.log("Size: " + buf.length + " bytes");
});
