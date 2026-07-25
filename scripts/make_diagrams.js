// Schematic diagram suite for the FYP report / thesis / interim.
//
// These are HAND-AUTHORED schematics (architecture, study design, cluster
// execution), not data plots: scripts/make_figures.py owns everything derived
// from results/analysis/*.json. Light-theme SVG with literal colours and system
// fonts is rasterised to PNG by @resvg/resvg-js — the same path used by
// scripts/build_agentic_report.js — so the Word documents stay self-contained.
//
// Deliberate constraint: these diagrams carry only STRUCTURAL constants
// (sample sizes, decoding settings, SLURM limits, module names) and no result
// values. Figures are not covered by scripts/verify_report_claims.py, so a
// result number printed here would be an unlocked claim surface. Every number
// below is traceable to configs/tc1_512.yaml or the committed summary.json
// files (num_records: harmbench 200, xstest 250, mmlu 300, arc 1172, uniform
// across all 15 aliases).
//
// Output: docs/figures/{system_architecture,study_design,tc1_execution}.png
// Run:    node scripts/make_diagrams.js   (or: make diagrams)

const fs = require("fs");
const path = require("path");
const { Resvg } = require("@resvg/resvg-js");

const FIGDIR = path.join(__dirname, "..", "docs", "figures");
fs.mkdirSync(FIGDIR, { recursive: true });

// ------------------------------------------------------------
// Palette + primitives (house style, shared with the agentic report)
// ------------------------------------------------------------
const C = {
  ink: "#0f1521", text: "#161d29", soft: "#39434f", muted: "#73809a", faint: "#9aa3b2",
  line: "#d9dee8", card: "#ffffff", card2: "#f4f6fa", band: "#eef1f7", paper: "#ffffff",
  signal: "#2563eb", cyan: "#0891b2", cap: "#d97706", safe: "#0a9d6e", violet: "#7c3aed", harm: "#dc2626",
};
const SANS = "Helvetica, Arial, sans-serif";
const MONO = "Menlo, Consolas, monospace";

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function tx(x, y, s, { size = 18, fill = C.text, anchor = "start", weight = "normal", font = SANS, ls } = {}) {
  return `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" fill="${fill}" ` +
    `text-anchor="${anchor}" font-weight="${weight}"${ls ? ` letter-spacing="${ls}"` : ""}>${esc(s)}</text>`;
}
function box(x, y, w, h, { r = 8, fill = C.card, stroke = C.line, sw = 1.4, dash } = {}) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" ` +
    `stroke="${stroke}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
}
// Band: a titled container strip.
function band(x, y, w, h, label, { fill = C.band, stroke = "none" } = {}) {
  return box(x, y, w, h, { fill, stroke: stroke === "none" ? fill : stroke, r: 10 }) +
    tx(x + 16, y + 26, label, { size: 15, fill: C.muted, weight: "bold", ls: 1.6 });
}
function arrowR(x1, y, x2, color = C.faint, sw = 2.2) {
  return `<line x1="${x1}" y1="${y}" x2="${x2 - 8}" y2="${y}" stroke="${color}" stroke-width="${sw}"/>` +
    `<polygon points="${x2},${y} ${x2 - 10},${y - 6} ${x2 - 10},${y + 6}" fill="${color}"/>`;
}
function arrowD(x, y1, y2, color = C.faint, sw = 2.2) {
  return `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2 - 8}" stroke="${color}" stroke-width="${sw}"/>` +
    `<polygon points="${x},${y2} ${x - 6},${y2 - 10} ${x + 6},${y2 - 10}" fill="${color}"/>`;
}
// Elbow: down from (x1,y1) then right to (x2,y2). Used for branch fan-outs.
function elbow(x1, y1, x2, y2, color = C.faint, sw = 2) {
  return `<path d="M ${x1} ${y1} V ${y2} H ${x2 - 8}" fill="none" stroke="${color}" stroke-width="${sw}"/>` +
    `<polygon points="${x2},${y2} ${x2 - 10},${y2 - 6} ${x2 - 10},${y2 + 6}" fill="${color}"/>`;
}
function svgWrap(W, H, inner) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<rect width="${W}" height="${H}" fill="${C.paper}"/>${inner}</svg>`;
}
// A titled card: bold title, then muted detail lines.
function card(x, y, w, h, title, lines, opts = {}) {
  const { stroke = C.line, sw = 1.4, titleSize = 19, titleFill = C.ink, fill = C.card,
          lineSize = 15, mono = false, titleMono = false, dash } = opts;
  let s = box(x, y, w, h, { fill, stroke, sw, dash });
  s += tx(x + 14, y + 28, title, { size: titleSize, fill: titleFill, weight: "bold", font: titleMono ? MONO : SANS });
  lines.forEach((l, i) => {
    s += tx(x + 14, y + 52 + i * 21, l, { size: lineSize, fill: C.soft, font: mono ? MONO : SANS });
  });
  return s;
}

// ============================================================
// Figure A — System architecture (report Chapter 4)
// ============================================================
function figArchitecture() {
  const W = 1000, H = 1086;
  const L = 100;
  let s = "";
  // Inter-band arrows are collected and drawn last: bands are opaque, so an
  // arrow emitted before the band below it would be painted over.
  const links = [];

  // --- Band 1: configuration -------------------------------------------
  s += band(10, 16, 980, 116, "CONFIGURATION");
  s += card(L, 46, 258, 74, "configs/tc1_512.yaml", ["10 model aliases", "4 benchmarks · SLURM"], { titleSize: 17, titleMono: true });
  s += arrowR(366, 83, 396);
  s += card(400, 46, 250, 74, "config_schema.py", ["Pydantic validation:", "pair + benchmark invariants"], { titleSize: 17, titleMono: true });
  s += arrowR(658, 83, 688);
  s += card(692, 46, 298, 74, "Typed run specification", ["ModelSpec · DecodingConfig", "BenchmarkSpec"], { titleSize: 17 });

  // --- Band 2: execution ------------------------------------------------
  links.push(arrowD(500, 138, 162));
  s += band(10, 166, 980, 316, "EXECUTION   (per model alias)");
  s += tx(L, 214, "run_quant_matrix.py — each model is loaded once and run across all four benchmarks",
    { size: 16, fill: C.soft });

  s += card(L, 230, 214, 150, "HFModelLoader", ["same model_id for", "both pair members;", "revision pinned;", "dtype auto → fp16"], { titleSize: 19 });

  // quantization branch — the study's independent variable
  const bx = 358, bw = 292;
  s += card(bx, 230, bw, 60, "fp16 baseline", ["no quantization"], { stroke: C.muted, sw: 1.8, titleSize: 18 });
  s += card(bx, 298, bw, 60, "NF4  4-bit", ["BitsAndBytesConfig, double quant"], { stroke: C.signal, sw: 2.2, titleSize: 18, titleFill: C.signal });
  s += card(bx, 366, bw, 60, "INT8", ["LLM.int8() precision point"], { stroke: C.muted, sw: 1.4, titleSize: 16, dash: "5 4" });
  s += elbow(330, 305, bx, 260);
  s += arrowR(330, 328, bx);
  s += elbow(330, 305, bx, 388);
  s += tx(L, 402, "applied on the fly at load time", { size: 15, fill: C.muted });

  s += arrowR(658, 328, 688);
  s += card(692, 230, 298, 180, "TextGenerator", [
    "chat template applied",
    "(enable_thinking = False)",
    "greedy: do_sample = False,",
    "temperature = 0.0",
    "512 new tokens · batch 4",
    "seed 42 (Python/NumPy/Torch)",
  ], { titleSize: 19 });

  s += tx(L, 452, "try / finally → gc.collect() + torch.cuda.empty_cache() between models · resume keyed on prompt_id",
    { size: 15, fill: C.muted });

  // --- Band 3: benchmarks ----------------------------------------------
  links.push(arrowD(500, 488, 512));
  s += band(10, 516, 980, 156, "BENCHMARK PLUGINS");
  s += tx(L, 556, "benchmarks/base.py (ABC):  load_items → build_prompt → score_response → aggregate",
    { size: 16, fill: C.soft });
  const bench = [
    ["HarmBench", "n = 200", "harmful compliance", C.harm],
    ["XSTest", "n = 250", "over-refusal", C.cap],
    ["MMLU", "n = 300", "capability", C.cyan],
    ["ARC-Challenge", "n = 1172", "capability", C.cyan],
  ];
  bench.forEach(([name, n, role, col], i) => {
    const x = L + i * 224;
    s += card(x, 574, 214, 86, name, [n, role], { stroke: col, sw: 1.8, titleSize: 18, titleFill: col });
  });

  // --- Band 4: scoring --------------------------------------------------
  links.push(arrowD(500, 678, 702));
  s += band(10, 706, 980, 190, "SCORING");
  const scoreCols = [
    [L, 288, "HarmBench ASR", C.harm,
      ["PRIMARY", "HarmBench-Llama-2-13b-cls", "SECONDARY", "v2 refusal regex (proxy)"],
      "cross-checks: gpt-4o · Llama-Guard-3-8B"],
    [400, 250, "XSTest over-refusal", C.cap,
      ["PRIMARY", "v2 refusal regex", "SENSITIVITY", "gpt-4o 3-class judge"],
      "grounded on a human gold set"],
    [692, 298, "MMLU / ARC accuracy", C.cyan,
      ["PRIMARY", "exact match on the answer letter", "SENSITIVITY", "strict-parser rescore"],
      "sensitivity bracket reported"],
  ];
  scoreCols.forEach(([x, w, title, col, rows, note]) => {
    s += card(x, 736, w, 148, title, [], { stroke: col, sw: 1.8, titleSize: 18, titleFill: col });
    rows.forEach((r, i) => {
      const isLabel = i % 2 === 0;
      s += tx(x + 14, 790 + Math.floor(i / 2) * 46 + (isLabel ? 0 : 22), r,
        isLabel ? { size: 13, fill: C.muted, weight: "bold", ls: 1.2 } : { size: 15, fill: C.ink });
    });
    s += tx(x + 14, 872, note, { size: 13, fill: C.muted });
  });

  // --- Band 5: artifacts + analysis -------------------------------------
  links.push(arrowD(500, 902, 926));
  s += band(10, 930, 980, 140, "ARTIFACTS  →  ANALYSIS");
  s += card(L, 958, 258, 54, "raw.jsonl", ["immutable, one row per prompt"], { titleSize: 16, titleMono: true, lineSize: 14 });
  s += card(370, 958, 220, 54, "summary.json", ["aggregate metrics"], { titleSize: 16, titleMono: true, lineSize: 14 });
  s += card(602, 958, 388, 54, "scores.judge.*.jsonl", ["redacted sidecar: IDs + booleans only"], { titleSize: 16, titleMono: true, lineSize: 14 });
  s += tx(L, 1046, "compare_quant_pairs.py", { size: 16, fill: C.ink, weight: "bold", font: MONO });
  s += arrowR(336, 1040, 364);
  s += tx(376, 1046, "paired McNemar  ·  bootstrap 95% CI  ·  BH-FDR  ·  interpretation label",
    { size: 15, fill: C.soft });

  return svgWrap(W, H, s + links.join(""));
}

// ============================================================
// Figure B — Study design (report Chapter 3)
// ============================================================
function figStudyDesign() {
  const W = 1000, H = 806;
  const L = 100;
  let s = "";

  // --- Band 1: matched model pairs -------------------------------------
  s += band(10, 16, 980, 128, "MATCHED MODEL PAIRS   (4 families)");
  const pairs = [
    ["Qwen3-1.7B", "1.7 B"], ["Qwen3-4B", "4.0 B"], ["Llama-3.2-3B", "3.0 B"],
    ["Mistral-7B-v0.3", "7.2 B"], ["Phi-4-mini", "3.8 B"],
  ];
  pairs.forEach(([name, size], i) => {
    const x = L + i * 180;
    s += card(x, 52, 170, 72, name, [size + " parameters"], { titleSize: 16, lineSize: 14 });
  });
  s += arrowD(500, 132, 166);

  // --- Band 2: precision arm -------------------------------------------
  s += band(10, 174, 980, 150, "PRECISION ARM   (the independent variable)");
  s += card(L, 210, 250, 62, "fp16 baseline", ["reference precision"], { stroke: C.muted, sw: 1.8, titleSize: 18 });
  s += card(L, 284, 250, 30, "", [], { fill: C.band, stroke: C.band });
  s += card(378, 210, 250, 62, "NF4 4-bit", ["the primary contrast"], { stroke: C.signal, sw: 2.2, titleSize: 18, titleFill: C.signal });
  s += card(656, 210, 250, 62, "INT8", ["secondary precision point"], { stroke: C.line, sw: 1.4, titleSize: 18, dash: "4 3" });
  s += tx(L, 300, "identical model_id, pinned revision, identical prompts, seed and decoding — precision is the only difference",
    { size: 15, fill: C.muted });
  s += tx(L, 320, "10 aliases in the primary study  (5 pairs × 2 precisions)  +  5 INT8 aliases  =  15 evaluated aliases",
    { size: 15, fill: C.soft, weight: "bold" });
  s += arrowD(500, 332, 366);

  // --- Band 3: benchmarks ----------------------------------------------
  s += band(10, 374, 980, 122, "BENCHMARK BATTERY   (every alias runs all four)");
  const bench = [
    ["HarmBench", "n = 200", "attack success rate", C.harm],
    ["XSTest", "n = 250", "over-refusal rate", C.cap],
    ["MMLU", "n = 300", "accuracy", C.cyan],
    ["ARC-Challenge", "n = 1172", "accuracy", C.cyan],
  ];
  bench.forEach(([name, n, metric, col], i) => {
    const x = L + i * 224;
    s += card(x, 408, 214, 78, name, [n + "  ·  " + metric], { stroke: col, sw: 1.8, titleSize: 17, titleFill: col, lineSize: 14 });
  });
  s += arrowD(500, 504, 538);

  // --- Band 4: statistical pipeline ------------------------------------
  s += band(10, 546, 980, 130, "INFERENCE   (per pair × benchmark)");
  const steps = [
    ["Paired test", "McNemar on the", "same prompt set"],
    ["Effect size", "absolute + relative", "delta"],
    ["Uncertainty", "paired bootstrap", "95% CI"],
    ["Multiplicity", "Benjamini–Hochberg", "over the registered family"],
  ];
  steps.forEach(([t, a, b], i) => {
    const x = L + i * 224;
    s += card(x, 580, 194, 82, t, [a, b], { titleSize: 17, lineSize: 14 });
    if (i < 3) s += arrowR(x + 194, 621, x + 224);
  });

  // --- Band 5: label + validation --------------------------------------
  s += box(10, 690, 468, 100, { fill: C.card2, stroke: C.line, r: 10 });
  s += tx(28, 718, "INTERPRETATION LABEL", { size: 14, fill: C.muted, weight: "bold", ls: 1.4 });
  s += tx(28, 744, "safety change read against the capability anchor:", { size: 15, fill: C.soft });
  s += tx(28, 768, "alignment shift · capability collapse · robust preservation", { size: 15, fill: C.ink, weight: "bold" });

  s += box(496, 690, 494, 100, { fill: C.card2, stroke: C.line, r: 10 });
  s += tx(514, 718, "VALIDATION ARMS", { size: 14, fill: C.muted, weight: "bold", ls: 1.4 });
  s += tx(514, 744, "3 independent judges · 2 human gold sets", { size: 15, fill: C.soft });
  s += tx(514, 768, "5-seed decoding sweep · 128 vs 512-token robustness", { size: 15, fill: C.soft });

  return svgWrap(W, H, s);
}

// ============================================================
// Figure C — TC1 execution model (report Chapter 5)
// ============================================================
function figTC1() {
  const W = 1000, H = 690;
  const L = 100;
  let s = "";

  // --- Zone 1: workstation ---------------------------------------------
  s += band(10, 16, 980, 118, "WORKSTATION   (macOS, no GPU)");
  s += card(L, 48, 250, 74, "Author config", ["configs/tc1_512.yaml"], { titleSize: 17, lineSize: 14, mono: true });
  s += arrowR(358, 85, 388);
  s += card(392, 48, 250, 74, "cluster-generate", ["one sbatch per model"], { titleSize: 17, titleMono: true, lineSize: 14 });
  s += arrowR(650, 85, 680);
  s += card(684, 48, 306, 74, "git push", ["repo is the transport"], { titleSize: 17, titleMono: true, lineSize: 14 });
  s += arrowD(500, 142, 176);

  // --- Zone 2: head node ------------------------------------------------
  s += band(10, 184, 980, 154, "TC1 HEAD NODE   (login only)");
  s += card(L, 218, 250, 72, "git pull", ["sync the repo"], { titleSize: 17, titleMono: true, lineSize: 14 });
  s += arrowR(358, 254, 388);
  s += card(392, 218, 250, 72, "make prefetch", ["cache datasets + weights"], { titleSize: 17, titleMono: true, lineSize: 14 });
  s += arrowR(650, 254, 680);
  s += card(684, 218, 306, 72, "sbatch", ["submit, then wait"], { titleSize: 17, titleMono: true, lineSize: 14 });
  s += box(L, 300, 890, 28, { fill: "#fdf2f2", stroke: C.harm, sw: 1.2, r: 6 });
  s += tx(L + 12, 320, "CLUSTER POLICY:  no user code on the head node — no Python, no nvidia-smi;  all compute goes through sbatch, never srun",
    { size: 14, fill: C.harm });
  s += arrowD(500, 346, 380);

  // --- Zone 3: compute node --------------------------------------------
  s += band(10, 388, 980, 190, "GPU COMPUTE NODE   (V100, offline)");
  s += card(L, 422, 396, 138, "run_quant_matrix.py", [
    "HF_HUB_OFFLINE = 1",
    "HF_DATASETS_OFFLINE = 1",
    "TRANSFORMERS_OFFLINE = 1",
    "→ results_512/<alias>/<benchmark>/",
  ], { titleSize: 18, titleMono: true, mono: true, lineSize: 14 });
  s += card(524, 422, 200, 138, "Judge passes", [
    "HarmBench cls",
    "Llama-Guard-3-8B",
    "(separate GPU jobs)",
  ], { titleSize: 17, lineSize: 14 });
  s += card(740, 422, 250, 138, "QoS envelope", [
    "1 GPU per job",
    "6 h walltime",
    "10 GB host RAM",
    "MaxJobsPU = 2",
  ], { titleSize: 17, lineSize: 14, stroke: C.muted });

  // --- return path ------------------------------------------------------
  s += box(10, 596, 980, 82, { fill: C.card2, stroke: C.line, r: 10 });
  s += tx(28, 624, "RETURN PATH", { size: 14, fill: C.muted, weight: "bold", ls: 1.4 });
  s += tx(28, 654, "scp results back", { size: 16, fill: C.ink, weight: "bold" });
  s += arrowR(196, 648, 226);
  s += tx(238, 654, "compare_quant_pairs.py  →  make_figures.py  →  make report      (no GPU required)",
    { size: 16, fill: C.soft, font: SANS });

  return svgWrap(W, H, s);
}

// ============================================================
// Rasterise
// ============================================================
function rasterize(name, svg, width = 2000) {
  const r = new Resvg(svg, { fitTo: { mode: "width", value: width }, font: { loadSystemFonts: true }, background: "white" });
  fs.writeFileSync(path.join(FIGDIR, name), r.render().asPng());
  console.log(`Wrote ${path.join("docs", "figures", name)}`);
}

rasterize("system_architecture.png", figArchitecture());
rasterize("study_design.png", figStudyDesign());
rasterize("tc1_execution.png", figTC1());
