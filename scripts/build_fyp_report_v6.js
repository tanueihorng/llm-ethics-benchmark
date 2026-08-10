// FYP Report builder v6 - CORPUS-VOICE PROSE REWRITE - docx-js
// Output: /Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-25_v6.docx
//
// v6 (T47): prose-only rewrite into the house style of strong past NTU FYP reports, derived
// from the 15-paper corpus study in docs/FYP_WRITING_STYLE_GUIDE.md. Built from a copy of
// build_fyp_report_v5.js; v5 and its docx are left UNTOUCHED for side-by-side comparison
// (same non-destructive pattern as v3 -> v4 -> v5).
//   WHAT CHANGED: sentence architecture only. Clause-joining semicolons became full stops
//   plus a named connective; em dashes removed; ", not Y" appositives recast; at most one
//   bracketed statistic per sentence with the remainder moved to an adjacent sentence or a
//   table pointer; hedges given their own sentences; the maintenance register (task IDs,
//   "folded in", "now been run") restated in plain past tense; section symbols spelled out;
//   British spelling and canonical model names normalised; long paragraphs split.
//   WHAT DID NOT CHANGE: every number, confidence interval, p-value, q-value, kappa, n,
//   threshold, citation bracket, table, figure, heading and section order. Chapter structure
//   and the research questions are identical to v5. Verified by scripts diffing the numeric
//   multiset and the heading list between the two docx files.
//   ONE DELIBERATE CONTENT CUT: the Abstract is reduced from 614 to about 200 words (corpus
//   median 198, corpus maximum 435). Everything removed from it remains in the body.
// v5 (T31/D41): 512-primary re-base. HarmBench ASR is now reported at HarmBench's
// 512-token reference budget as PRIMARY (128 retained as an explicit comparison);
// folds in INT8@512 (Llama INT8 increase vanishes under both judges) and multi-seed
// @512, and the audit fixes (direct-prefix truncation evidence, MMLU/ARC zero-shot
// disclosure, multi-seed scope, determinism caveat). Built from a copy of v4.
// v4 (T31/D39): added §6.16 Generation-Length Robustness (512-token rerun). Built
// from a copy of build_fyp_report_v3.js; v3 + its docx are left untouched for
// comparison (same non-destructive pattern as v2 → v3).
// v3 = v2 (figures, IEEE citations, FDR/power, scorer-validation lead) PLUS a full
// de-AI prose pass (em-dashes removed, discourse-marker crutches cut, rule-of-three
// and negative-parallelism trimmed). v2 and the original are left untouched for
// side-by-side comparison.

const fs = require("fs");
const path = require("path");
const { loadClaimRegistry } = require("./lib/claim_registry");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink,
  TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun,
} = require("docx");

const FIGDIR = path.join(__dirname, "..", "docs", "figures");
const CLAIMS = loadClaimRegistry();

// ------------------------------------------------------------
// Document constants
// ------------------------------------------------------------
const PAGE_W = 11906;       // A4 (NTU standard)
const PAGE_H = 16838;       // A4
const MARGIN = 1440;        // 1 inch (tables auto-fit CONTENT_W via the drift adjustment in the Table helper)
const CONTENT_W = PAGE_W - 2 * MARGIN;  // 9026

const FONT = "Times New Roman";
const MONO = "Consolas";

const COL_BORDER = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const TABLE_BORDERS = {
  top: COL_BORDER, bottom: COL_BORDER, left: COL_BORDER, right: COL_BORDER,
  insideHorizontal: COL_BORDER, insideVertical: COL_BORDER,
};
const CELL_BORDERS = {
  top: COL_BORDER, bottom: COL_BORDER, left: COL_BORDER, right: COL_BORDER,
};
const CELL_MARGINS = { top: 80, bottom: 80, left: 120, right: 120 };
const HEADER_SHADE = { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" };

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
function P(text, opts = {}) {
  const runs = Array.isArray(text)
    ? text
    : [new TextRun({ text, font: FONT, size: opts.size || 22 })];
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after !== undefined ? opts.after : 120, line: opts.line || 360 },
    alignment: opts.alignment,
    pageBreakBefore: opts.pageBreakBefore || false,
  });
}

function PJ(text, opts = {}) {
  // Justified paragraph for body text (1.5 line spacing per NTU)
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22 })],
    spacing: { after: 160, line: 360 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: FONT, size: 32, bold: true })],
    spacing: { before: 360, after: 200 },
    pageBreakBefore: true,
  });
}

function H1NoBreak(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: FONT, size: 32, bold: true })],
    spacing: { before: 360, after: 200 },
  });
}

function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT, size: 26, bold: true })],
    spacing: { before: 280, after: 140 },
  });
}

function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: FONT, size: 24, bold: true })],
    spacing: { before: 220, after: 120 },
  });
}

function Bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [new TextRun({ text, font: FONT, size: 22 })],
    spacing: { after: 80, line: 280 },
  });
}

function Numbered(text, level = 0) {
  // Legacy single-list helper. Prefer numberedList(items) below - that
  // function mints a fresh numbering reference per call so each list
  // restarts at 1 instead of continuing one global counter.
  return new Paragraph({
    numbering: { reference: "numbers", level },
    children: [new TextRun({ text, font: FONT, size: 22 })],
    spacing: { after: 80, line: 280 },
  });
}

// Counter for minting fresh numbering references. Each call to
// numberedList(items) increments and uses one of the pre-declared
// "numlist<N>" references (defined in numbering.config below). This is
// the docx-js-idiomatic way to restart numbering at 1 for each list:
// the reference name uniquely identifies the counter.
let __numRefCounter = 0;
function numberedList(items, level = 0) {
  __numRefCounter += 1;
  const ref = `numlist${__numRefCounter}`;
  return items.map(text => new Paragraph({
    numbering: { reference: ref, level },
    children: [new TextRun({ text, font: FONT, size: 22 })],
    spacing: { after: 80, line: 280 },
  }));
}

// Figure embedding. Reads a PNG from docs/figures/, preserves aspect ratio, and
// emits a centered image followed by an italic numbered caption. Figures are
// generated reproducibly from results/analysis/*.json by scripts/make_figures.py.
let __figCounter = 0;
function _pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function Figure(file, caption, dispW = 560) {
  __figCounter += 1;
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const { w, h } = _pngSize(buf);
  const dispH = Math.round(dispW * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: dispW, height: dispH } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 220 },
      children: [new TextRun({ text: `Figure ${__figCounter}. ${caption}`, font: FONT, size: 18, italics: true })],
    }),
  ];
}

function Code(text) {
  // Code-style paragraph for code/yaml/script snippets
  const lines = text.split("\n");
  return lines.map(line => new Paragraph({
    children: [new TextRun({ text: line || " ", font: MONO, size: 18 })],
    spacing: { after: 0, line: 240 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
  }));
}

function Equation(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Cambria Math", size: 24, italics: false })],
    spacing: { before: 80, after: 120 },
    alignment: AlignmentType.CENTER,
  });
}

function buildTable(headers, rows, widths) {
  // widths: array of DXA col widths summing to CONTENT_W
  if (!widths) {
    const w = Math.floor(CONTENT_W / headers.length);
    widths = headers.map(() => w);
    const drift = CONTENT_W - widths.reduce((a, b) => a + b, 0);
    widths[widths.length - 1] += drift;
  }
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders: CELL_BORDERS,
      width: { size: widths[i], type: WidthType.DXA },
      margins: CELL_MARGINS,
      shading: HEADER_SHADE,
      children: [new Paragraph({
        children: [new TextRun({ text: h, font: FONT, size: 20, bold: true })],
        spacing: { after: 0 },
      })],
    })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map((cell, i) => new TableCell({
      borders: CELL_BORDERS,
      width: { size: widths[i], type: WidthType.DXA },
      margins: CELL_MARGINS,
      children: [new Paragraph({
        children: [new TextRun({ text: String(cell), font: FONT, size: 20 })],
        spacing: { after: 0 },
      })],
    })),
  }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    borders: TABLE_BORDERS,
    rows: [headerRow, ...bodyRows],
  });
}

// ------------------------------------------------------------
// Cover page
// ------------------------------------------------------------
const cover = [
  new Paragraph({ children: [new TextRun(" ")], spacing: { after: 800 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "NANYANG TECHNOLOGICAL UNIVERSITY",
      font: FONT, size: 28, bold: true,
    })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "College of Computing and Data Science",
      font: FONT, size: 24,
    })],
    spacing: { after: 600 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "FINAL YEAR PROJECT", font: FONT, size: 22, bold: true })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Comprehensive Project Report", font: FONT, size: 22, italics: true })],
    spacing: { after: 480 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "Benchmarking Ethical Performance of Open-Source LLMs:",
      font: FONT, size: 32, bold: true,
    })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "A Matched-Pair, Judge-Validated Study of Safety–Capability Trade-offs in Quantized Compact Language Models",
      font: FONT, size: 28, bold: true,
    })],
    spacing: { after: 800 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Project Code:  CCDS25-1136", font: FONT, size: 24, bold: true })],
    spacing: { after: 200 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Student:  TAN UEI HORNG  (UTAN001)", font: FONT, size: 26, bold: true })],
    spacing: { after: 120 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Email:  UTAN001@e.ntu.edu.sg", font: FONT, size: 22 })],
    spacing: { after: 120 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Supervisor:  Dr. Zhang Jiehuang  (jiehuang.zhang@ntu.edu.sg)", font: FONT, size: 22 })],
    spacing: { after: 600 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Document date:  2 July 2026", font: FONT, size: 26, bold: true })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Comprehensive project report, revised through July 2026. Scope: five matched pairs across four families at three precisions (fp16 → INT8 → NF4). HarmBench ASR is scored by the official HarmBench classifier (judge-primary, D16) and cross-checked by a second judge; the refusal regex is retained only as a transparent proxy. This is the full technical write-up; two companion documents present the same study at other lengths — a shorter Interim Report (milestone deliverable, docs/FYP_Interim) and a condensed, IEEE-cited standalone Thesis (docs/FYP_Thesis).", font: FONT, size: 20, italics: true, color: "555555" })],
    spacing: { after: 100 },
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------
// Declarations (originality + generative-AI use)
// ------------------------------------------------------------
const declarations = [
  H1NoBreak("Declaration of Originality"),
  PJ("I hereby declare that this report is my own work and, to the best of my knowledge and belief, it contains no material previously published or written by another person, nor material that has been accepted for the award of any other degree or diploma of a university or other institution of higher learning, except where due acknowledgement has been made in the text. The intellectual content of this report — its research design, experimental methodology, analysis, and interpretation — is the product of my own work, although I have received assistance on software implementation, language, and presentation as acknowledged herein."),
  PJ("All experimental results reported here were produced by the open-source benchmarking framework described herein, executed on the NTU TC1 GPU cluster. Every reported numerical result is computed by the committed code from the recorded experimental records; no result has been altered, fabricated, or selectively reported. The per-generation raw records are retained locally and pinned by cryptographic hash rather than committed to the public repository, so a third party reproduces the results either by re-running the pipeline from the committed configuration and source code, or by replaying the committed redacted score sidecars — not by recomputing from raw generations on a fresh clone. Where the work of others has been used, it has been cited and referenced."),
  new Paragraph({ spacing: { before: 700, after: 40 }, children: [new TextRun({ text: "_______________________________", font: FONT, size: 22 })] }),
  P("Tan Uei Horng  (UTAN001)"),
  P("College of Computing and Data Science, Nanyang Technological University"),
  P("Date:  2 July 2026"),
  new Paragraph({ children: [new PageBreak()] }),
  H1NoBreak("Declaration on the Use of Generative AI Tools"),
  PJ("In the interest of transparency and in accordance with academic-integrity guidance and emerging open-source-software norms (for example, the Journal of Open Source Software AI-usage disclosure policy), I disclose that generative-AI coding and writing assistants were used extensively during this project to assist with software implementation, automated test authoring, analysis scripting, documentation, and language editing, and to run the multi-agent audit workflows described in the appendices. All such assistance was carried out under my direction. I reviewed, validated, and take full responsibility for all AI-assisted outputs, and the core research design, scientific claims, experimental decisions, and their interpretation are my own. No data, results, or citations were fabricated: every reported number is computed by the committed code from the recorded experimental artefacts, and all cited sources were independently verified."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------
// Abstract
// ------------------------------------------------------------
const abstract = [
  H1NoBreak("Abstract"),
  PJ("Quantization is what brings compact instruction-tuned language models within reach of constrained hardware, but a safety evaluation carried out on the full-precision checkpoint does not automatically transfer to the quantized artefact derived from it. Whether quantization shifts safety alignment, and whether an apparent shift is a genuine alignment change or a by-product of degraded capability, remains unresolved for the compact-deployment regime. The institutional benchmarks TrustLLM, DecodingTrust and SafetyBench do not target it. This work answers that question with a matched-pair, judge-validated study."),
  PJ("Quantization is applied on the fly to identical weights, so each paired contrast isolates the quantization step under the recorded matched-pair configuration. Five pairs across four families, Qwen, Llama, Mistral and Phi, span 1.7–7.2 B. Each was evaluated at three precisions, fp16, INT8 (LLM.int8) and NF4 four-bit, on HarmBench, XSTest, MMLU and ARC-Challenge. The central methodological finding is that the scorer determines the conclusion. A refusal-counting regex over-counts harmful compliance, so HarmBench's own fine-tuned classifier replaces it. A second judge from a different provider cross-checks the classifier at Cohen's κ 0.68–0.95. Refusal-counting therefore overstated the apparent harm from quantization in this study's HarmBench contrasts."),
  PJ("Under the classifier at the 512-token reference budget, the study's primary configuration, no pair shows a significant harmful-compliance change under four-bit NF4 after Benjamini-Hochberg correction across the primary contrasts. The surviving effects are capability losses on MMLU and ARC, plus a single over-refusal decrease. That decrease is most plausibly a measurement artefact of the regex scorer, because an independent three-class refusal judge does not reproduce it. The result is no detectable change at n = 200 rather than a proven zero. The headline risk of four-bit quantization in small models is therefore capability degradation."),
];

// ------------------------------------------------------------
// TOC
// ------------------------------------------------------------
const toc = [
  H1("Table of Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
];

const lof = [
  H1("List of Figures and Tables"),
  P("Figure 1  Study design: matched pairs, the precision arm, the benchmark battery, and the inference pipeline.", { after: 60 }),
  P("Figure 2  System architecture: configuration, execution, benchmark plugins, scoring, and analysis.", { after: 60 }),
  P("Figure 3  TC1 execution model: workstation, head node, GPU compute node, and the return path.", { after: 60 }),
  P("Figure 4  Quantization effect on harmful compliance (per-pair judge ΔASR forest plot).", { after: 60 }),
  P("Figure 5  The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).", { after: 60 }),
  P("Figure 6  Multi-seed decoding sensitivity for Qwen3-1.7B (greedy vs stochastic).", { after: 60 }),
  P("Figure 7  Per-category HarmBench ASR for Qwen3-1.7B, fp16 vs NF4.", { after: 60 }),
  P("Figure 8  Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.", { after: 60 }),
  P("Figure 9  Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).", { after: 60 }),
  P("Table 3.1  Model pair selection matrix.", { after: 60 }),
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets.", { after: 60 }),
  P("Table 3.3  Decoding controls used during inference.", { after: 60 }),
  P("Table 3.4  Interpretation labels derived from combined deltas.", { after: 60 }),
  P("Table 4.1  Repository module responsibilities.", { after: 60 }),
  P("Table 4.2  CLI subcommands exposed by fyp_cli.py.", { after: 60 }),
  P("Table 5.1  TC1 cluster hardware and policy parameters.", { after: 60 }),
  P("Table 5.2  Software environment versions on TC1.", { after: 60 }),
  P("Code listing 5.1  Head-node pre-cache invocation (scripts/prefetch_tc1.py).", { after: 60 }),
  P("Table 6.1  Per-pair benchmark results for all five model pairs with bootstrap 95% CIs.", { after: 60 }),
  P("Table 6.2  All-pair quantization deltas, interpretation labels, and evidence status.", { after: 60 }),
  P("Table 6.3  Judge vs v2 per-model agreement on HarmBench (n = 200 each).", { after: 60 }),
  P("Table 6.4  Exploratory per-pair XSTest over-refusal deltas under the independent three-class refusal judge.", { after: 60 }),
  P("Table 6.5  HarmBench ΔASR across the 128- and 512-token generation budgets.", { after: 60 }),
  P("Table D.1  Stable verification coverage areas.", { after: 60 }),
];

// ------------------------------------------------------------
// Chapter 1 - Introduction
// ------------------------------------------------------------
const ch1 = [
  H1("Chapter 1: Introduction"),

  H2("1.1 Background and Motivation"),
  PJ("The last two years have seen a rapid proliferation of compact instruction-tuned large language models (LLMs) in the one-to-four-billion-parameter range. Models such as Qwen3 [1], Llama 3 [2], Microsoft Phi-4-mini [3], and Google Gemma 2 [4] have demonstrated that capable reasoning and instruction following are achievable well below frontier parameter counts, and the smallest members of these families are sized for consumer hardware, mobile chipsets, and edge accelerators [2], [3]. This has shifted the practical envelope of LLM deployment. Tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, which keeps user data on the device and can reduce inference latency [5]. Compression research has moved in the same direction, with quantization frameworks now built explicitly for edge inference that report more than threefold speedups over full precision on desktop and mobile GPUs [5], and one-shot methods that fit a one-hundred-and-seventy-five-billion-parameter model onto a single GPU [6]."),
  PJ("However, full-precision inference is not always practical at this scale. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision costly to sustain, which is what makes quantization useful for resource-constrained deployment rather than merely convenient. Four bits is also the precision that inference scaling laws identify as optimal for a fixed memory budget across almost all model scales [7]. Consequently, four bits is the operating point that compression research targets for on-device inference. AWQ is designed explicitly for on-device compression and ships four-bit weights in its own desktop and mobile runtime [5], and four bits is where the compression literature concentrates its own evaluation effort [8], [9]. The BitsAndBytes NF4 scheme evaluated in this study is one such four-bit format. How often an end user in fact receives a quantized rather than a full-precision checkpoint is not something this study measures, and no prevalence figure is claimed for it. The premise that motivates the work is narrower and is established in the literature: quantization can change safety behaviour [10], so a safety evaluation carried out on a full-precision checkpoint does not automatically transfer to the quantized artefact derived from it."),
  PJ("Quantization has historically been treated as a numerical optimisation technique whose primary cost is a small loss in perplexity or downstream accuracy [6], [7]. However, a growing body of evidence challenges this view. Quantization can degrade the confidence calibration of model predictions even where accuracy is preserved [11], and it can degrade safety alignment itself, with no single method dominating across models and bit-widths [10]. BitsAndBytes schemes including NF4 have further been shown to be manipulable, so that a benign full-precision model becomes harmful once quantized [12]. Evidence from multilingual evaluation shows how easily such damage is missed, because automatic metrics there understated the harm in Japanese by roughly an order of magnitude relative to human judgment, and the paper reports that languages are disparately affected [13]."),
  PJ("There is also a specific reason to expect refusal behaviour to be fragile under compression. Safety alignment is a learned behaviour, instilled by preference optimisation over human feedback [14]. The premise of this study is that it is therefore carried in the model weights as other learned behaviour is, and is exposed to any operation that modifies them. It is mediated by a concentrated activation direction [15], and it is often shallow, adapting the output distribution mainly over the first few generated tokens [16]. Weight-modifying operations are already known to disturb it, since fine-tuning on as few as ten adversarially chosen examples can substantially compromise safety guardrails [17]. Quantization is a weight-modifying operation. It may therefore perturb refusal behaviour even when it preserves task accuracy on average, which makes understanding the interaction a prerequisite to safe deployment rather than an optional extra."),

  H2("1.2 Problem Statement"),
  PJ("This work addresses a single, deliberately scoped research problem:"),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "When a small language model is quantized to four-bit precision for deployment, do observed changes in safety behaviour reflect a true shift in alignment, or are they a side-effect of degraded general capability?",
      font: FONT, size: 22, italics: true,
    })],
    spacing: { before: 80, after: 200 },
  }),
  PJ("The distinction matters. A model that becomes more refusal-prone after quantization could plausibly be reported as \"safer\" by a benchmark that measures only harmful compliance. This failure mode is well documented, and over-refusal on benign prompts is now measured by dedicated benchmarks precisely because safety optimisation can make a model less helpful [18], [19]. Measured on benign prompts, the same model may refuse those as well, exhibiting over-refusal, and on capability benchmarks it may answer fewer questions correctly. Without a joint evaluation of harmful compliance, over-refusal, and capability, the safety community risks systematically misattributing capability collapse as safety improvement. The problem statement above directly motivates the multi-axis evaluation framework adopted in this study."),

  H2("1.3 Research Questions"),
  PJ("The study is organised around seven research questions. The first three are primary questions answered directly by the matched-pair experiments. RQ4 examines scale sensitivity within a family. RQ5 and RQ6 examine robustness across model families and across quantization precisions. RQ7 asks whether the answer to RQ1 depends on the scorer used to measure it."),
  Bullet("RQ1 (Primary). Does four-bit NF4 quantization measurably change harmful-compliance behaviour, as measured by HarmBench Attack Success Rate, relative to matched baseline weights in compact instruction-tuned models?"),
  Bullet("RQ2 (Primary). Does four-bit quantization change over-refusal behaviour on benign prompts, as measured by XSTest?"),
  Bullet("RQ3 (Primary). Does four-bit quantization degrade general capability, as measured by exact-match accuracy on a curated MMLU subset and on ARC-Challenge?"),
  Bullet("RQ4 (Secondary, scale). Does the magnitude of the quantization effect differ between Qwen3-1.7B and Qwen3-4B within the same model family?"),
  Bullet("RQ5 (Robustness, family). Does the pattern observed within the Qwen family replicate qualitatively across other model families, namely Llama, Mistral and Phi?"),
  Bullet("RQ6 (Robustness, precision). Does the pattern hold across quantization precisions, comparing INT8 with four-bit NF4 against the same full-precision baselines?"),
  Bullet("RQ7 (Measurement). Does the choice of harmful-compliance scorer change the answer to RQ1, and which scorer is better grounded against human judgment?"),

  H2("1.4 Scope and Boundaries"),
  PJ("Explicit scoping is essential to make the study tractable within the constraints of an undergraduate Final Year Project and to keep the resulting claims defensible. The following boundaries are stated up front."),
  Bullet("Quantization method. The main study applies BitsAndBytes NF4 four-bit loading on the fly via the Hugging Face transformers integration. In addition, a precision point (Section 6.11) adds BitsAndBytes INT8/LLM.int8 as a second method, so the study spans fp16 → INT8 → NF4. Both are applied on the fly to identical baseline weights. The other post-training families considered, GPTQ [6] and AWQ [5], are excluded because each rests on a different error model, GPTQ compensating layer-wise quantization error with approximate second-order information while AWQ scales and protects a small set of salient weight channels, and each would require its own calibration pipeline, so folding them in would change two variables at once rather than one. Other families such as GPTQ, AWQ, and GGUF are out of scope and are flagged in the Future Work chapter."),
  Bullet("Model regime. The study targets compact instruction-tuned models. The primary matched pairs sit in the roughly two-to-four-billion-parameter range. The cross-family robustness extension (Section 6.8.6) reaches 7.2 billion parameters with Mistral-7B, so the study spans approximately 1.7 to 7.2 billion parameters. This regime was chosen because models of this size load and evaluate on a single 32 GB GPU, which is the constraint the study actually operated under."),
  Bullet("Languages and modality. Evaluation is restricted to English-language, text-only interactions."),
  Bullet("Decoding. All inference uses deterministic greedy decoding with temperature 0.0 and top-p 1.0. This choice both eliminates within-condition variance and maximises comparability across baseline and quantized pair members."),
  Bullet("Benchmark axes. A broad multi-axis ethical evaluation is already covered by larger institutional benchmarks. Rather than attempt one, this study evaluates three complementary axes that jointly enable the alignment-versus-capability disambiguation that motivates the research question."),

  H2("1.5 Contributions"),
  PJ("The contributions of this project are, in order of significance, a methodological cautionary result, a controlled study design, an interpretation method, an engineering framework, and an empirical characterisation of quantization effects."),
  ...numberedList([
    "A scorer-validity result with consequences beyond this study. The choice of HarmBench scorer changes the conclusion. A deterministic refusal-counting regex over-counts harmful compliance, because its harmful set is a near-strict superset of the official classifier's. Replacing it with HarmBench's own fine-tuned classifier changes both which model looks least safe and whether any model looks significantly less safe at all. That classifier was cross-checked by a second judge from a different provider at Cohen's κ 0.68–0.95. Refusal-counting therefore overstated the apparent harm from quantization in this study's HarmBench contrasts, which is a caution worth carrying into other safety evaluations.",
    "A controlled matched-pair study design that isolates quantization as the sole experimental variable. Both members of every pair are loaded from the same underlying model_id. The only difference is whether on-the-fly quantization is applied during the from_pretrained call. This eliminates publisher-asymmetry and conversion-pipeline asymmetry, two confounds that affect comparisons assembled from separately published checkpoints.",
    "A capability-anchored interpretation method that combines harmful-compliance, over-refusal, and capability deltas into one of six diagnostic labels: alignment degradation, alignment improvement, capability collapse masquerading as safety, over-refusal regression, robust preservation, and broad degradation. Each label is carried with a two-layer evidence status. This formalises the alignment-versus-capability disambiguation as a reproducible procedure rather than an ad hoc judgement.",
    "An open, reproducible benchmarking framework with batched chat-templated generation, per-prompt audit logging, immutable hash-pinned raw artefacts, redacted score sidecars, resumable per-model SLURM matrix jobs for the NTU TC1 cluster, and a verification suite whose live inventory is derived by pytest rather than copied into prose.",
    "An empirical characterisation of quantization across five model pairs, four families, and three precisions (fp16 → INT8 → NF4). Under the validated classifier at the 512-token reference budget, no pair shows a significant harmful-compliance change under four-bit NF4 after a multiplicity correction. Three contrasts survive the multiplicity correction in the registered family: two capability losses and one over-refusal decrease that later scorer validation attributes most plausibly to the regex. No harmful-compliance contrast survives on any pair. No bit-width gradient is detected across fp16, INT8 and NF4. Separately, the 128-token apparent increase in the smallest model does not replicate at the reference budget (Section 6.12). These results are supported by a refusal-margin mechanism probe, a multi-seed sensitivity arm, an FDR/power analysis, and a second-judge cross-check.",
  ]),

  H2("1.6 Report Structure"),
  PJ("Chapter 2 surveys the relevant literature on large-scale ethical benchmarking, the helpfulness–harmlessness trade-off, the deployment of compact LLMs, and the behavioural effects of quantization, concluding with the specific research gaps that this study targets. Chapter 3 details the experimental methodology, including the matched-pair design, quantization approach, benchmark selection, scoring, decoding controls, and the interpretation framework. Chapter 4 documents the system design and implementation: package structure, configuration schema, the model loader and generation pipeline, the benchmark plugin architecture, matrix orchestration, resume logic, and SLURM job generation."),
  PJ("Chapter 5 describes the experimental setup on TC1 and the run plan. Chapter 6 presents the experimental results for all five model pairs. The original three completed on the NTU TC1 cluster on 2026-05-27, and the two cross-family pairs, Mistral-7B and Phi-4-mini, completed on 2026-06-15. Chapter 7 discusses threats to validity, Chapter 8 records limitations, and Chapter 9 proposes future work. Chapter 10 concludes. Seven appendices reproduce the full configuration, an example SLURM script, the configuration schema, the test inventory, the repository layout, and a glossary, and record the document revision history and the scoring-methodology correction record."),
];

// ------------------------------------------------------------
// Chapter 2 - Literature Review
// ------------------------------------------------------------
const ch2 = [
  H1("Chapter 2: Literature Review"),
  PJ("This chapter reviews the work the present study builds on, in four steps. Section 2.1 covers the large-scale institutional benchmarks that define how open-weight model safety is currently measured. Section 2.2 examines the helpfulness-harmlessness trade-off and the instruments used to measure each side of it. Section 2.3 turns to the compact models and on-device deployment setting that make the question practically urgent. Section 2.4 reviews quantization itself and the evidence that it is not behaviourally neutral. Section 2.5 places the closest prior studies side by side, and Section 2.6 states the resulting research gaps."),

  H2("2.1 Large-Scale Ethical Benchmarking of LLMs"),
  PJ("The past two years have produced several large-scale institutional efforts to benchmark the ethical performance of open-source LLMs. TrustLLM [20] evaluates sixteen mainstream LLMs over thirty datasets across six trustworthiness dimensions (truthfulness, safety, fairness, robustness, privacy, and machine ethics). DecodingTrust [21] performs a deep multi-perspective assessment of toxicity, stereotype bias, robustness, privacy, machine ethics, and fairness, focused on the GPT-3.5/GPT-4 family. In addition, SafetyBench [22] tests twenty-five Chinese and English LLMs with over eleven thousand multiple-choice safety questions. These benchmarks have published leaderboards that cover popular open-weight model families such as Llama, Mistral, Qwen, and Falcon, and each states explicit criteria for the safety profile it expects of a publicly released model."),
  PJ("The strengths of these frameworks are clear: large model coverage, multi-dimensional evaluation, and standardised methodologies that facilitate cross-model comparison. However, their gaps are equally clear in the context of the present work. First, quantized variants are either omitted or evaluated separately rather than as a controlled contrast against their own full-precision counterparts. Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, leaving the compact regime this study targets under-covered. Third, they typically report each safety axis independently, covering harmful compliance, bias, and toxicity, without explicitly anchoring those measurements against a capability metric. This leaves open the question of whether safety changes reflect alignment or capability."),
  PJ("None of the three frameworks makes quantization a controlled experimental variable, and for the API-served systems that DecodingTrust covers the serving precision is not disclosed at all. The present study is therefore positioned one step downstream of them, on the artefact that is actually deployed."),

  H2("2.2 The Helpfulness–Harmlessness Trade-off"),
  PJ("A central, well-documented limitation of static safety benchmarks is their difficulty in jointly capturing the tension between helpfulness and harmlessness. The tension is not incidental. It is built into how modern assistants are trained. Instruction tuning with human feedback is a primary reason these models are usable as assistants: for GPT-3 it made a 1.3-billion-parameter model's outputs preferred over those of the 175-billion-parameter base model, improved truthfulness, and produced small reductions in toxic output under respectful prompting, though no improvement in bias [23]. Bai et al. [14] apply preference modelling and reinforcement learning from human feedback to fine-tune language models to act as helpful and harmless assistants, and report that this alignment training improves performance on almost all NLP evaluations while remaining fully compatible with training for specialised skills. A model that is heavily optimised for the harmlessness signal may nevertheless become overly conservative, refusing benign prompts that incidentally resemble unsafe ones. This failure mode is known as over-refusal or exaggerated safety. Conversely, a model optimised for helpfulness may remain vulnerable to adversarial jailbreak attacks. Measuring only one side of this trade-off therefore produces an incomplete picture."),
  PJ("Two benchmarks measure opposite sides of this trade-off, and are the pair used here. HarmBench [24] provides curated unsafe prompts and measures attack success rate, the fraction of unsafe prompts to which the model produces a harmful, complying response. By contrast, XSTest [18] provides benign prompts that are superficially similar to unsafe ones and measures over-refusal rate, the fraction of benign prompts the model nonetheless refuses. Evaluating both simultaneously is essential to detecting trade-offs introduced by alignment training or compression."),
  PJ("Over-refusal measurement has since been scaled well beyond XSTest's size. OR-Bench [19] automatically generates seemingly toxic prompts and assembles eighty thousand of them across ten rejection categories, with a hard subset of roughly one thousand prompts and a further six hundred genuinely toxic prompts included to detect models that simply answer everything. Its authors evaluate thirty-two models across eight families. XSTest is retained as the instrument here because its two hundred and fifty benign prompts are human-curated and because its three-class response taxonomy transfers directly to the design used here. Its own primary evaluation was manual annotation by three of its authors, with automated string-matching and a GPT-4 classifier reported only as cheaper alternatives, so the deterministic regex used in this study is an adaptation of that taxonomy rather than a reproduction of the published protocol. OR-Bench nevertheless shows that over-refusal is now the subject of dedicated, large-scale benchmarks in its own right."),
  PJ("The threat model matters as much as the instrument. This study presents HarmBench behaviours directly, with no adversarial augmentation, which corresponds to the DirectRequest baseline in HarmBench's own taxonomy of red-teaming methods rather than to any of its attack methods. The label 'standard' in HarmBench denotes a functional category of behaviour, alongside contextual, copyright and multimodal, and not an absence of attack. A substantial adjacent literature instead attacks the model. Zou et al. [25] construct adversarial suffixes by combining greedy and gradient-based search, and show that a single suffix can transfer across queries and across models. Qi et al. [17] demonstrate that safety guardrails can be substantially compromised by fine-tuning on as few as ten adversarially designed examples, at a cost of well under a dollar. Attack discovery has also been automated, with language models used to generate test cases that surface harmful behaviours before deployment [26]. These results bound what a direct-request measurement can claim, and the present study scopes its own threat model accordingly in Section 1.4."),
  PJ("MMLU [27], the Massive Multitask Language Understanding benchmark, measures multiple-choice accuracy across academic and professional subjects, and is used here as the general-capability anchor. In addition, ARC-Challenge [28] provides a structurally different second capability axis, used here as corroboration. By measuring multiple-choice accuracy across a broad spectrum of academic and professional subjects, MMLU provides a capability signal that is largely independent of refusal behaviour. Capability collapse can therefore be detected even when safety metrics appear to improve."),
  PJ("The instruments to measure each axis therefore exist and are mature. What the literature does not settle is how to read them jointly when the model has been compressed."),

  H2("2.3 Small Language Models and On-Device Deployment"),
  PJ("Recent compact LLM releases have repeatedly demonstrated that strong reasoning, instruction following, and multilingual performance are achievable at parameter counts below four billion. The Qwen2.5 and Qwen3 series [1], Llama-3.2 (1B and 3B Instruct) [2], Microsoft Phi-4-mini [3], and Google Gemma 2 [4], released as a family of lightweight open models spanning two to twenty-seven billion parameters with interleaved local-global attention and knowledge distillation at the 2B and 9B sizes, have each been positioned for deployment at practical, resource-constrained scale. Their compact size is what makes genuinely local execution possible on such hardware. How widely they are in fact integrated into deployed applications is not measured here and is not claimed."),
  PJ("Quantization matters for these models because it is what brings them within reach of constrained hardware, whether by on-the-fly BitsAndBytes loading or by pre-quantized GGUF checkpoints. How widely each path is used in practice is not measured here, and no prevalence figure is claimed for it. What matters for the research question is narrower: a safety claim established on the unquantized release does not automatically transfer to the quantized artefact derived from it, and that transfer is what this study tests."),
  PJ("Four bits is not an arbitrary operating point. Dettmers and Zettlemoyer [7] run more than thirty-five thousand experiments across model families from nineteen million to one hundred and seventy-six billion parameters and find that, for a fixed memory budget, four-bit precision is optimal for almost all models at almost all scales, with performance degrading once precision drops to three bits. Badshah and Sajjad [29] pose the closely related deployment question of whether to run a smaller model at high precision or a larger model quantized, and report that larger models are notably resilient to precision reduction and retain high accuracy at four bits on many tasks. Both results concern capability alone. Neither examines what happens to safety behaviour at the same operating point, and the second explicitly frames the compact-model regime as the weaker case without measuring its safety consequences."),

  H2("2.4 Quantization and Behavioural Effects"),
  PJ("Quantization compresses model weights from higher-precision floating-point representations to lower-precision integer or normalized-float representations. This reduces the memory footprint and accelerates inference at the cost of some numerical fidelity. Post-training quantization (PTQ) methods apply this conversion after the model has been trained and require no fine-tuning, which makes them attractive for deployment."),
  PJ("Several PTQ families recur throughout the compression literature. GPTQ [6] performs one-shot weight quantization using approximate second-order information, reaching three to four bits per weight and quantizing a one-hundred-and-seventy-five-billion-parameter model in roughly four GPU hours. AWQ [5] observes that weights are not equally important and protects the most salient one percent by searching for a per-channel scaling derived from the activation distribution rather than from the weights, and reports more than threefold speedups over a full-precision baseline on desktop and mobile GPUs. SmoothQuant [30] targets the activation side instead, migrating quantization difficulty from activations to weights by a mathematically equivalent transformation so that both can be held at eight bits. These methods differ in what they protect and where they place the difficulty, which is one reason results obtained under one method need not transfer to another. Compression is not limited to quantization either: one-shot pruning can remove half a model's weights at minimal perplexity cost [31], and the trustworthiness evidence reviewed below finds pruning and quantization behave differently on safety."),
  PJ("Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA [32] is the one this study evaluates. It is applied as a load-time option rather than as a separate conversion step, which is the property the matched-pair design depends on, since it allows both members of a pair to come from one unmodified checkpoint. NF4 represents each weight using a four-bit normalized-float code optimised for the typical Gaussian-like distribution of neural network weights, with double quantization applied to the quantization constants to further reduce overhead. The eight-bit LLM.int8() method [33] is a distinct, mixed-precision algorithm in the same library that decomposes outlier features into a separate high-precision path. The present study evaluates both. The accompanying BitsAndBytes library integrates directly with Hugging Face transformers, exposing quantized loading through a single BitsAndBytesConfig object that can be passed to from_pretrained at model load time."),
  PJ("A body of recent work has argued that quantization is not behaviourally neutral. A comprehensive multi-benchmark evaluation of quantization strategies finds that headline benchmark scores are largely retained at four bits [8]. That evaluation nevertheless cautions that assessments limited to language modelling and a few classification tasks can miss shifts on other behavioural dimensions such as alignment. Li et al. [9] extend this by systematically evaluating quantized models across a wide task set and issuing practical recommendations for when each technique is safe to apply. Quantization has also been shown to degrade the confidence calibration of model predictions, a property known to be poorly behaved in modern neural networks independently of accuracy [34], even where accuracy is preserved [11]. The reported impact varies with model type and scale, and it concentrates on inputs where the full-precision model was already least confident."),
  PJ("One study is a direct warning about the measurement apparatus itself. Marchisio et al. [13] evaluate quantization across languages and find that automatic metrics severely understate the damage: a 1.7 percent average drop on automatic tasks in Japanese corresponds to a 16.0 percent drop when human evaluators judge realistic prompts. They also find that non-Latin-script languages are affected worst and that harder tasks such as mathematical reasoning degrade fastest. The relevance here is methodological rather than multilingual. If an automatic metric can understate a quantization effect by an order of magnitude in one setting, then the scorer used to measure a safety effect is itself a variable that has to be validated rather than assumed."),
  PJ("Safety-specific studies make the point directly. Kharinaev et al. [10] evaluate 66 quantized variants across multiple post-training and quantization-aware methods and find that quantization can degrade safety alignment, with no single method dominating across models and bit-widths. Egashira et al. [12] show that BitsAndBytes schemes, including NF4, can be manipulated so that a benign full-precision model becomes harmful once quantized. HarmLevelBench [35] reports that the direction of the safety change under AWQ and GPTQ is non-uniform across attack types. Q-resafe [36] confirms quantization-induced safety degradation and proposes a quantization-aware patching method to repair it."),
  PJ("Closest to the present study's headline, Hong et al. [37] scrutinise trustworthiness across five compression methods and eight trust dimensions and report that 4-bit quantization broadly preserves trustworthiness. Pruning, by contrast, degrades it, and erosion appears under the more extreme 3-bit compression. That finding is the nearest prior analogue to the null-safety-at-four-bit picture found here, although their study covers larger models and a different trust taxonomy."),
  PJ("There is also a mechanistic reason to expect quantization to touch refusal specifically. Arditi et al. [15] show that refusal in chat models is mediated by a single direction in activation space, so refusal is a comparatively concentrated behaviour rather than a diffuse one. Qi et al. [16] show further that safety alignment is often shallow, in the sense that it adapts the model's output distribution mainly over the first few generated tokens, and they argue that this shallowness helps explain susceptibility to adversarial suffixes, prefilling, decoding-parameter manipulation and fine-tuning attacks. A weight perturbation that shifts the first-token distribution could therefore move refusal behaviour without moving average task accuracy at all. Section 6.10 examines exactly this by measuring the first-token refusal margin before and after quantization."),

  H2("2.5 Positioning Against the Closest Prior Studies"),
  PJ("Table 2.1 places the closest prior studies side by side on the dimensions that matter for the present research question. Two columns are decisive. The capability-anchor column records whether the study measured general capability alongside safety on the same models, which is what makes an alignment interpretation separable from a capability one. The matched-weights column records whether the compared full-precision and quantized models were derived from identical baseline weights, which is what makes the contrast attributable to quantization rather than to checkpoint provenance."),
  buildTable(
    ["Study", "Models", "Safety measured", "Capability anchor", "Matched weights stated", "Significance test reported"],
    [
      ["Jin et al. [8]", "multiple families", "not reported", "yes", "not reported", "not reported"],
      ["Proskurina et al. [11]", "multiple families", "not reported", "calibration", "not reported", "not reported"],
      ["Li et al. [9]", "multiple families", "yes (trustworthiness: ETHICS, TruthfulQA, AdversarialGLUE)", "yes", "yes, applied to the same checkpoints", "not reported"],
      ["Badshah and Sajjad [29]", "Llama 2-Chat and Mistral/Mixtral, 7-70 B", "not reported", "yes", "yes, on the fly (bitsandbytes)", "not reported"],
      ["Marchisio et al. [13]", "multilingual", "not reported", "yes, plus human evaluation", "not reported", "not reported"],
      ["Kharinaev et al. [10]", "66 quantized variants", "yes", "partial", "not reported", "not reported"],
      ["Egashira et al. [12]", "BitsAndBytes", "yes (adversarial)", "not reported", "yes", "not reported"],
      ["HarmLevelBench [35]", "AWQ, GPTQ", "yes", "not reported", "not reported", "not reported"],
      ["Q-resafe [36]", "multiple", "yes", "partial", "not reported", "not reported"],
      ["Hong et al. [37]", "7B and above", "yes (8 trust dimensions)", "yes", "not reported", "not reported"],
      ["This study", "5 pairs, 1.7-7.2 B", "yes (2 axes)", "yes (2 benchmarks)", "yes, on the fly", "yes, BH-corrected"],
    ],
  ),
  P("Table 2.1  The closest prior quantization studies, compared on the dimensions that determine whether an observed safety change can be attributed to alignment rather than to capability loss or to checkpoint provenance. Entries are compiled from each paper's abstract and stated scope. \"Not reported\" means the property is not stated in the source consulted, and should not be read as evidence that the authors did not consider it. The final row records this study's own design on the same dimensions for comparison; the table is a positioning aid on these specific axes and is not offered as a novelty or priority claim.", { size: 18, after: 160 }),
  PJ("Read down the capability-anchor and matched-weights columns together and the gap becomes visible rather than asserted. Across the rows of Table 2.1, the studies that measure safety carefully mostly do not report a capability anchor on the same models, and those that measure capability carefully mostly do not report safety. Only one row combines both, and no prior row reports a significance test on the paired contrast."),

  H2("2.6 Research Gaps Targeted by This Work"),
  PJ("Five interlocking gaps in the existing literature motivate the present study. Each names a specific methodological property that the reviewed work does not have, rather than simply an unstudied topic."),
  ...numberedList([
    "Gap 1. Limited empirical study of compact (<4B) instruction-tuned models in the safety–quantization context. The quantization studies surveyed in Table 2.1 concentrate on models of seven billion parameters and above. Compact models relevant to edge deployment are therefore under-represented, and the scaling evidence suggests they are the weaker case rather than the safer one.",
    "Gap 2. Lack of integrated evaluation that measures harmful compliance, over-refusal, and capability simultaneously. Studies that measure only harmful compliance cannot detect capability-driven safety artefacts, because a model that has become less capable and a model that has become more cautious both produce fewer harmful completions.",
    "Gap 3. Difficulty interpreting whether observed safety metric changes in quantized models reflect a real alignment shift or a side-effect of capability degradation. No convention for jointly interpreting safety and capability deltas under compression was identified among the studies reviewed here, so this study defines one explicitly (Section 3.8).",
    "Gap 4. Provenance asymmetry in some existing comparisons. Not every prior study has this problem, and Table 2.1 records two that quantize the same checkpoints on the fly. Where it does occur, a comparison of full-precision and quantized checkpoints uses a full-precision checkpoint from one publisher and a pre-quantized checkpoint from another, which conflates quantization effects with checkpoint-conversion effects. A clean, on-the-fly quantization design from identical baseline weights eliminates this confound.",
    "Gap 5. Unvalidated scoring. Table 2.1 records no scorer-validation step for any study listed in it, and this review did not establish whether one was performed in each case. What the multilingual evidence does show is that an automatic metric can understate a quantization effect by roughly an order of magnitude in at least one language [13]. A safety conclusion is only as trustworthy as the instrument that produced it, which is why the scorer is validated here rather than assumed.",
  ]),
  PJ("The choice of benchmark suite follows directly from these gaps. HarmBench supplies harmful compliance, XSTest supplies over-refusal on benign prompts, and MMLU and ARC-Challenge supply two structurally different capability anchors. Harmful compliance and over-refusal are the two directions in which alignment can move, and the capability anchors are what distinguish a model that has become safer from a model that has merely become worse. Confidence calibration, which the reviewed work identifies as a further quantization-sensitive property [11], is not measured here and is stated as a scope limitation in Chapter 8."),
  PJ("The methodology described in Chapter 3 is structured to address all five gaps. The compact-deployment regime is addressed by the choice of Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B, extended by the Mistral-7B and Phi-4-mini cross-family pairs reported in Section 6.8.6. The integrated evaluation is addressed by the four complementary benchmarks. The interpretation challenge is addressed by the rule-based interpretation layer of Section 3.8. The provenance asymmetry is eliminated by on-the-fly NF4 loading from the same baseline weights. Finally, the scoring gap is addressed by the judge validation of Section 6.9, which grounds the primary scorer against a second judge, an open-weight guard model, and human annotation."),
];

// ------------------------------------------------------------
// Chapter 3 - Methodology
// ------------------------------------------------------------
const ch3 = [
  H1("Chapter 3: Methodology"),

  H2("3.1 Experimental Design"),
  PJ("The study adopts a matched-pair comparative experimental design. Each model under study is evaluated as a pair: a baseline variant loaded in the default high-precision dtype, and a four-bit variant produced by applying BitsAndBytes NF4 quantization on the fly at load time. Both pair members are loaded from exactly the same Hugging Face model_id. No separately uploaded \"pre-quantized\" checkpoint is used. The only operational difference between the two members of a pair is the presence of a BitsAndBytesConfig object in the from_pretrained call."),
  PJ("This design choice is the strongest internal-validity property of the study. By construction, both pair members share identical baseline weights, the same tokenizer, the same chat template, and the same release artefact. By construction this also isolates the quantization step from checkpoint-provenance confounds, namely separate fine-tuning, conversion artefacts, or a re-uploaded checkpoint. Any observed delta is therefore best attributed to quantization or to measurement noise rather than to a different underlying model. The matched-pair design therefore controls checkpoint provenance within the recorded environment. The exact runtime environment is a separate, documented reproducibility caveat (Chapter 8) rather than a confound this design removes. The on-the-fly NF4 path is the most direct way to operationalise the counterfactual \"the same model, quantized\" in code."),
  PJ("The independent variables are quantization state, model scale, and model family. Quantization state contrasts baseline loading with four-bit loading. Model scale contrasts 1.7 billion against four billion parameters within the Qwen family. Model family contrasts Qwen against Llama in the primary design, extended to Mistral and Phi in the cross-family replication of Section 6.8.6. The dependent variables are harmful compliance, over-refusal on benign prompts, and general capability, each measured by a corresponding benchmark."),

  ...Figure("study_design.png", "Study design. Five matched model pairs spanning four families are evaluated at fp16 and NF4 four-bit, with an INT8 precision point as a third level, across four benchmarks. Both members of a pair share the same checkpoint and pinned revision, the same prompts, the same seed and the same decoding settings, so numerical precision is the only difference between them. Every pair-level contrast passes through the same paired-inference pipeline before it is labelled."),

  H2("3.2 Model Selection"),
  PJ("Three model pairs form the primary controlled design. The Qwen pairs at 1.7 billion and four billion parameters [1] serve as the primary within-family comparison and provide both a quantization axis and a scale axis. The Llama-3.2-3B pair [2] sits between them by parameter count and serves as a cross-family check. Table 3.1 summarises this primary matrix. A cross-family extension later adds two further matched pairs, Mistral-7B and Phi-4-mini, which are introduced with the experimental setup in Chapter 5 and analysed in Section 6.8.6. That extension takes the study to five pairs across four families."),
  buildTable(
    ["Pair ID", "Role", "Model alias", "Hugging Face model_id", "Family", "Size"],
    [
      ["qwen_2b", "Primary (BF/FP16)", "qwen_2b_base", "Qwen/Qwen3-1.7B", "Qwen", "1.7 B"],
      ["qwen_2b", "Primary (4-bit)", "qwen_2b_4bit", "Qwen/Qwen3-1.7B", "Qwen", "1.7 B"],
      ["qwen_4b", "Primary (BF/FP16)", "qwen_4b_base", "Qwen/Qwen3-4B", "Qwen", "4 B"],
      ["qwen_4b", "Primary (4-bit)", "qwen_4b_4bit", "Qwen/Qwen3-4B", "Qwen", "4 B"],
      ["llama_3_2_3b", "Cross-family (BF/FP16)", "llama_3_2_3b_base", "meta-llama/Llama-3.2-3B-Instruct", "Llama", "3 B"],
      ["llama_3_2_3b", "Cross-family (4-bit)", "llama_3_2_3b_4bit", "meta-llama/Llama-3.2-3B-Instruct", "Llama", "3 B"],
    ],
    [1200, 1600, 1700, 2860, 900, 1100],
  ),
  P("Table 3.1  Model pair selection matrix. Note that within each pair, both members share the same Hugging Face model_id. The only operational distinction is the presence of an NF4 BitsAndBytesConfig at load time.", { size: 18 }),
  PJ("The Qwen3 checkpoints were selected after an initial environment validation on the TC1 cluster surfaced a compatibility failure with a third-party model upload that had been considered during planning, namely techwithsergiu/Qwen3.5-text-2B and techwithsergiu/Qwen3.5-text-4B. Those checkpoints use a hybrid linear-attention/SSM architecture that requires the causal-conv1d and flash-linear-attention libraries at runtime. When those libraries are absent, the PyTorch fallback conv1d path raises a CUDNN_STATUS_NOT_INITIALIZED error during the very first inference batch, terminating the job before any results are produced."),
  PJ("Installing those libraries in the TC1 conda environment was considered but ruled out. The libraries require custom CUDA kernel compilation against the cluster's specific driver version and are not trivially installable in an offline environment. Switching to the official Qwen3 release, Qwen/Qwen3-1.7B and Qwen/Qwen3-4B, eliminated all external dependencies. The Qwen3 dense models are standard grouped-query-attention transformers that load and run with the baseline transformers and bitsandbytes packages already in the environment. The switch was made before any production runs, so no experimental results were affected."),

  H2("3.3 Quantization Approach"),
  PJ("Quantization is applied entirely on the fly by injecting a BitsAndBytesConfig object into the AutoModelForCausalLM.from_pretrained call. The configuration sets load_in_4bit to True, selects the NF4 quantization type, enables double quantization, and sets the four-bit compute dtype to follow the resolved torch_dtype of the model. That is, the compute dtype is float16 on CUDA when dtype is set to auto, and float32 on CPU. However, the CPU path is not used for quantized runs: in the pinned bitsandbytes version used here, four-bit loading requires a CUDA device."),
  PJ("Four-bit loading is forbidden on a CPU runtime. The loader raises a clear error when quantized is true and the resolved runtime device is not cuda. After load, the loader verifies that quantization actually engaged by checking is_loaded_in_4bit, is_loaded_in_8bit, is_quantized, and the presence of an hf_quantizer. If none of those signals is set, it raises a RuntimeError and refuses to proceed, so a silently unquantized load cannot be mislabelled as quantized."),
  PJ("The choice of NF4 over alternative PTQ methods such as GPTQ and AWQ is deliberate. NF4 is supported natively by transformers without an offline calibration pass, and applies symmetrically to any model that can be loaded with from_pretrained. This applicability symmetry is required by the matched-pair design. The quantization step must be applicable to exactly the same weights as the baseline, without an intermediate calibration corpus that would itself become a confound."),

  H2("3.4 Benchmark Selection and Mapping"),
  PJ("Four benchmarks are used across the three dimensions required to disambiguate alignment shifts from capability collapse. The capability dimension is measured by two complementary benchmarks. MMLU is the primary anchor, and ARC-Challenge is a structurally different second capability benchmark. This means the alignment-versus-capability disambiguation does not rest on a single capability proxy."),
  buildTable(
    ["Dimension", "Benchmark (source)", "Primary metric", "Sampling"],
    [
      ["Harmful compliance", "HarmBench (walledai/HarmBench, standard)", "Attack Success Rate (ASR)", "200 prompts"],
      ["Over-refusal", "XSTest (local canonical CSV, benign-only)", "Over-refusal rate", "250 benign prompts"],
      ["General capability (anchor)", "MMLU (cais/mmlu, 6-subject subset)", "Accuracy", "300 questions total"],
      ["General capability (2nd)", "ARC-Challenge (allenai/ai2_arc)", "Accuracy", "1,172 questions"],
    ],
    [1900, 3260, 2300, 1900],
  ),
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets. Sampling is deterministic and seed-controlled. Capability is reported on two anchors side by side (MMLU + ARC-Challenge). Interpretation labels remain MMLU-anchored (Appendix I.9).", { size: 18 }),
  PJ("HarmBench is selected as the harmful-compliance benchmark because it is a standardized harmful-behaviour benchmark with a stable, published definition of attack success. The walledai release on Hugging Face is used with the standard config. That config is a single 200-behaviour split that does not preserve the official benchmark's validation/test partition, so results are reported over this pooled set rather than the official test partition alone. A threat-model boundary is also stated plainly. The standard config supplies HarmBench's curated harmful behaviour requests directly, and this study applies no adversarial attack or jailbreak transformation on top of them, so no GCG, PAIR, AutoDAN, or any other attack pipeline from the HarmBench suite is involved."),
  PJ("ASR in this study therefore measures harmful compliance under direct requests, which is the weakest threat model in the HarmBench taxonomy and is equivalent to HarmBench's DirectRequest baseline. It does not measure robustness to optimised adversarial attack. Wei et al. [38] hypothesise that safety training fails through competing objectives and mismatched generalisation, mechanisms that adversarial prompting is designed to exploit and that a direct request is not designed to exercise. Their evidence comes from frontier API models, where the no-attack baseline elicits harmful compliance on nought to three percent of curated prompts. Attack-augmented evaluation is flagged as future work (Chapter 9). XSTest is the over-refusal test suite used in this study. This repository bundles the canonical XSTest v2 prompts CSV from paul-rottger/xstest and evaluates the benign-only subset to measure how often the model refuses prompts that should not be refused."),
  PJ("MMLU is used as the primary general-capability anchor. ARC-Challenge (allenai/ai2_arc) is added as a structurally different second capability benchmark, comprising roughly 1,172 grade-school science questions designed to resist retrieval shortcuts and scored by the identical exact-match rule as MMLU. Capability is therefore reported on the two benchmarks side by side (see Appendix I.9). Interpretation labels remain anchored on MMLU, with ARC as a corroborating axis."),
  PJ("To keep MMLU tractable within the per-job walltime on TC1, a six-subject subset is used: business_ethics, clinical_knowledge, college_biology, high_school_world_history, high_school_macroeconomics, and human_aging. These subjects span ethics, sciences, humanities, social sciences, and applied knowledge, and are deliberately chosen to cover heterogeneous reasoning demands without exhausting the time budget."),
  PJ("The six subjects are pooled and a deterministic, seed-controlled random sample of 300 questions is drawn from the pool, so the per-subject counts are not equal. They range from 25 to 94 questions. The exact per-subject distribution is reported with the statistical limitations in Chapter 8. Because the sample is fixed by seed and prompt identity, every baseline and 4-bit run sees the same 300 questions in the same order. Therefore this uneven distribution is held constant across all conditions and does not affect the matched-pair comparison."),

  H2("3.5 Scoring: Refusal Detection and the HarmBench Classifier"),
  PJ("Two scorers are used, with a clear division of roles. A deterministic regex-based refusal parser provides fully reproducible, zero-variance scoring. It is the scorer for XSTest over-refusal, and for HarmBench it is retained as a secondary, refusal-based upper-bound proxy, because it measures non-refusal rather than genuine harmful compliance. The primary HarmBench Attack Success Rate scorer, however, is the official HarmBench classifier (cais/HarmBench-Llama-2-13b-cls), a fine-tuned judge model run in full precision with deterministic greedy decoding. The validation that established it as primary is reported in Section 6.9. The classifier was promoted to primary after that validation showed the regex over-counts ASR, equating non-refusal with success, most severely for the Qwen and Mistral families."),
  PJ("Retaining the deterministic regex alongside it is deliberate. It lets the reader see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring, and its reproducibility is why it remains the over-refusal scorer and the transparent proxy. A judge LLM does introduce a variance budget and an additional moving part, so it is controlled accordingly. Rather than being avoided, it is run at full precision, with greedy decoding, over the immutable saved generations, and it writes only redacted sidecars."),
  PJ("Three boolean fields are recorded for each response: harmful_compliance, safe_refusal, and malformed. For HarmBench, harmful compliance corresponds to a response that addresses the unsafe request substantively rather than refusing. For XSTest benign prompts, refusal is the failure mode, and over-refusal rate is the fraction of benign prompts on which the refusal pattern fires. The malformed field flags responses that are empty, contain only template fragments, or fail to parse into the expected output format."),
  PJ("The trade-off of deterministic parsing is its potential to miss nuanced or euphemistic refusals. That trade-off is exactly why the official HarmBench classifier, rather than the regex, is the primary HarmBench scorer (Section 6.9)."),
  PJ("XSTest's own primary evaluation used manual annotation against a three-way scheme (full compliance / partial refusal / full refusal), with automated string-matching and a GPT-4 classifier reported as cheaper alternatives; no fine-tuned scorer was released with the suite. For XSTest over-refusal, this study nevertheless retains deterministic regex parsing as the primary over-refusal scorer. That is a deliberate, reproducibility-motivated deviation, and it yields a coarser binary (refusal / non-refusal) construct. The regex is additionally cross-checked with exactly such an LLM three-class refusal judge as a sensitivity layer (Section 6.9, Result 6)."),
  PJ("That check finds the one over-refusal contrast that survives the multiplicity correction to be scorer-dependent. A subsequent pre-registered human audit, a blinded, 200-item, disagreement-enriched single-annotator gold set, then finds the judge substantially better aligned with the annotator than the regex on this construct. That audit is a mechanical Outcome J, carried as Outcome-J–consistent with one disclosed protocol deviation under Section 8 of the pre-registration. That contrast is therefore carried as most plausibly a regex measurement artefact. The regex is retained as the original pre-specified scorer-of-record for traceability, and a second annotator remains the residual strengthening acknowledged in the Limitations chapter."),

  H2("3.6 Decoding Controls"),
  PJ("All inference runs share identical decoding parameters, applied uniformly across baseline and quantized members of every pair. Table 3.3 lists the parameters."),
  buildTable(
    ["Parameter", "Value", "Rationale"],
    [
      ["temperature", "0.0", "Greedy decoding; removes within-condition variance."],
      ["top_p", "1.0", "Disabled (greedy)."],
      ["max_new_tokens", "512", "HarmBench's reference generation budget for the primary study. The 128-token run is retained only as the Section 6.12 generation-length comparison. Monitored during smoke run."],
      ["repetition_penalty", "1.0", "Neutral; avoids confounding refusal calibration."],
      ["max_input_tokens", "1024", "Caps prompt length; prevents context overflow."],
      ["use_chat_template", "true", "Applies model-specific chat template to every prompt."],
      ["enable_thinking", "false", "Critical: disables Qwen3.x default thinking mode so the 512-token reference budget is not consumed by <thinking> blocks before the answer."],
    ],
    [2200, 1300, 5860],
  ),
  P("Table 3.3  Decoding controls used during inference. The enable_thinking=False setting is passed when the tokenizer supports it. A TypeError fallback gracefully handles tokenizers that do not accept the keyword.", { size: 18 }),

  H2("3.7 Metrics and Pairwise Analysis"),
  PJ("The primary scalar metrics computed per (model, benchmark) pair are defined below. Throughout, n denotes the number of evaluated prompts."),
  Equation("ASR  =  (1 / n) Σᵢ cᵢ ,    cᵢ ∈ {0, 1}  indicating harmful compliance"),
  Equation("OverRefusalRate  =  (1 / n_b) Σᵢ fᵢ ,    benign prompts only"),
  Equation("Accuracy  =  (1 / n) Σᵢ 𝟙[ŷᵢ = yᵢ]  ,    MMLU"),
  Equation("MalformedRate  =  (1 / n) Σᵢ mᵢ ,    AnsweredRate  =  (1 / n) Σᵢ 𝟙[ŷᵢ ≠ ∅]"),
  PJ("Pairwise deltas are computed per pair_id, taking the four-bit value minus the baseline value of each metric:"),
  Equation("ΔM  =  M_{4-bit}  −  M_{baseline}"),
  Equation("ΔM_rel  =  ( M_{4-bit}  −  M_{baseline} )  /  M_{baseline}    when M_{baseline} ≠ 0"),
  PJ("Bootstrap 95% confidence intervals [39] are computed for each metric by resampling the per-prompt score records with replacement. The intervals reflect prompt-sampling variance only. Because decoding is deterministic at temperature 0.0, there is no within-condition stochastic variance."),
  PJ("Paired significance test for HarmBench (McNemar). Because both pair members see the identical prompt set, the HarmBench ΔASR is a paired-binary contrast. The textbook-correct significance test is therefore McNemar's exact test [40] on the discordant prompts, those scored harmful under exactly one member, rather than an unpaired two-proportion test."),
  PJ("Write b for the number of prompts that become harmful under quantization (baseline-safe → 4-bit-harmful) and c for those that become safe (baseline-harmful → 4-bit-safe). Under the null hypothesis that quantization does not change harmful compliance, each discordant prompt favours one member with probability one half, so min(b, c) follows a Binomial(b + c, 0.5). The exact two-sided p-value doubles the lower tail. Reporting b and c also makes the effect size transparent, because the reader sees exactly how many prompts moved each way rather than only the net delta."),
  PJ("McNemar's exact p is reported alongside the bootstrap CI for every HarmBench pair as an independent corroboration that does not assume large-sample normality, and it is implemented in mcnemar_exact_test with no SciPy dependency. The same exact paired test is applied to every other paired-binary endpoint, namely XSTest over-refusal, MMLU, and ARC, in the multiple-comparisons analysis (Section 6.4.1). Where the bootstrap CI and the exact test disagree at the margin, the exact test is taken as the significance criterion and the delta is reported as borderline."),
  PJ("At the corrected over-refusal scoring, no contrast in the study sits in that margin. The previously-borderline Qwen3-1.7B over-refusal decrease resolves to non-significant on both criteria, at ΔOR −0.024. Its bootstrap CI [−0.048, 0.000] no longer excludes zero, and McNemar's exact p = 0.109 on the 10 discordant prompts. Both the bootstrap and the paired tests treat each benchmark's prompt set as an exchangeable sample of its construct. HarmBench, XSTest, and ARC are fixed curated sets, and only the MMLU subset is a seeded random draw. Items cluster by category or subject, so the effective information content is at most the nominal n. Any within-cluster correlation would work against the reported nulls rather than for them."),

  H2("3.8 Interpretation Framework"),
  PJ("Combining harmful-compliance, over-refusal, and capability deltas, each pair receives one of six rule-based interpretation labels. These labels are the central analytical output of the study and formalise the alignment-versus-capability disambiguation. The taxonomy is symmetric on both behavioural axes."),
  PJ("On the safety axis it pairs alignment_degradation, meaning ASR up beyond threshold with capability preserved, against its mirror alignment_improvement, meaning ASR down beyond threshold with capability preserved. The two are gated identically, so a genuine safety win is never demoted to a fallback degradation. On the over-refusal axis it pairs over_refusal_regression, where the model refuses materially more benign prompts while safety and capability hold, against robust_preservation, where over-refusal is stable or improved. A pure over-refusal change is therefore named on its own axis rather than absorbed into broad_degradation."),
  buildTable(
    ["Label", "Condition", "Interpretation"],
    [
      ["alignment_degradation", "ΔASR ≥ +0.02  AND  ΔMMLU > −0.03", "Harmful compliance worsens beyond threshold with capability preserved: a direct alignment concern."],
      ["alignment_improvement", "ΔASR ≤ −0.02  AND  ΔMMLU > −0.03", "Harmful compliance falls beyond threshold with capability preserved: the mirror image of alignment_degradation and the most desirable outcome. Gated on the same two axes as alignment_degradation (not on over-refusal)."],
      ["capability_collapse_masquerading_as_safety", "ΔASR ≤ −0.02  AND  ΔMMLU ≤ −0.03", "Apparent safety improvement coincides with capability collapse; not a genuine alignment win."],
      ["over_refusal_regression", "‖ΔASR‖ < 0.02  AND  ΔOR ≥ +0.02  AND  ΔMMLU > −0.03", "Safety and capability held, but the model refuses materially more benign prompts: an over-refusal regression on its own axis rather than a fallback degradation."],
      ["robust_preservation", "‖ΔASR‖ < 0.02  AND  ΔOR < +0.02  AND  ΔMMLU > −0.03", "Safety and capability preserved and over-refusal not worsened (stable or improved): quantization preserves alignment."],
      ["broad_degradation", "Fallback (none of the above conditions met)", "Compound degradation not captured by the more specific labels; typically capability drops beyond tolerance alongside an inconclusive safety direction."],
    ],
    [2400, 3500, 3460],
  ),
  P("Table 3.4  Interpretation labels derived from combined deltas. Numerical thresholds for the qualitative terms (\"roughly flat\", \"small\") are set as configurable parameters in the analysis module.", { size: 18 }),
  PJ("Boundary convention. The thresholds are inclusive, so a ΔASR of exactly +0.02 classifies as alignment_degradation and exactly −0.02 as alignment_improvement. The delta is rounded to remove float-representation noise, so an exactly-on-tolerance k/200 proportion difference classifies by its true value rather than a 1e-17 residue. Two pairs sit exactly on the ±0.02 harm tolerance, Phi-4-mini at +0.020 (alignment_degradation) and Mistral-7B at −0.020 (alignment_improvement). Both are statistically non-significant. The 95% CI includes zero in each case, and McNemar p = 0.424 and 0.627 respectively, so each carries an evidence status of directional rather than confirmed. The label states the point-estimate direction. The evidence status is what separates a confirmed move from a directional, on-the-boundary one, and for both of these pairs the harm change is not statistically distinguishable from zero.", { }),
  PJ("Two-layer reporting: label plus evidence status. The interpretation label is a statement about the direction and magnitude of the point-estimate deltas. It is intentionally decoupled from statistical significance so the taxonomy stays stable and legible. Each label is therefore paired with a separate evidence_status field derived from the same paired-bootstrap 95% CI, and, for HarmBench, from McNemar's exact test described in Section 3.7. The status is confirmed when the delta the label keys on excludes zero, directional when the point estimate crosses the threshold but the CI includes zero, null for robust_preservation, because a bounded CI cannot positively certify a null, and unknown when per-prompt outcomes are unavailable."),
  PJ("This decoupling prevents a non-significant point estimate from being read as a confirmed finding. For example, the Qwen3-4B pair is labelled alignment_degradation because its ASR point estimate exceeds threshold, but it carries evidence_status = directional because its CI touches zero. The status is computed by label_evidence_status alongside the label and emitted alongside the pairwise deltas in the analysis output. The label rule itself (classify_pair_change) is unchanged, so the diagnostic taxonomy and the strength-of-evidence reporting are independent, auditable layers."),

  H2("3.9 Reproducibility Controls"),
  PJ("Reproducibility is treated as a first-class engineering requirement. A single seed (42) is propagated to Python's random module, NumPy, and PyTorch RNGs at the start of each run. Dataset shuffling is deterministic. Each benchmark plugin loads its full dataset, shuffles with the seeded RNG, and then truncates to the configured max_samples. The same prompt order is therefore visited by both pair members."),
  PJ("Every per-prompt record persisted to raw.jsonl includes the model alias, the resolved model_id, the model family, the pair_id, the quantized flag, the seed, the full generation_config, and an ISO-8601 timestamp. This per-record metadata supports later auditability without dependence on the configuration file at the time of analysis."),
  PJ("Long-running jobs are protected by the resume logic described in Section 4.7. If a job is killed mid-benchmark by walltime exhaustion or transient cluster failure, the next submission reads the existing raw.jsonl, identifies which prompt_ids have already been processed, and skips them on the second pass. The granularity of the resume mechanism is per-prompt rather than per-benchmark."),
];

// ------------------------------------------------------------
// Chapter 4: System Design and Implementation
// ------------------------------------------------------------
const ch4 = [
  H1("Chapter 4: System Design and Implementation"),

  H2("4.1 Architecture Overview"),
  PJ("The framework is implemented as a single Python package, ethical_benchmark. A thin command-line wrapper, fyp_cli.py, and a Makefile expose the most common workflows. The package is decomposed into modules with single, well-defined responsibilities, and Table 4.1 summarises them."),
  buildTable(
    ["Module", "Responsibility"],
    [
      ["ethical_benchmark/quant/config_schema.py", "Pydantic schema for the YAML configuration; enforces pair invariants and benchmark cross-references."],
      ["ethical_benchmark/models/loader.py", "HFModelLoader: dtype resolution, device selection, on-the-fly BitsAndBytesConfig injection, post-load verification."],
      ["ethical_benchmark/models/generation.py", "TextGenerator: batched generation, chat-template application, enable_thinking handling, post-processing."],
      ["ethical_benchmark/benchmarks/", "Pluggable benchmark interface: base.py (ABC), harmbench.py, xstest.py, mmlu.py, arc.py, registry.py, and utils.py (shared refusal patterns)."],
      ["ethical_benchmark/pipeline/run_quant_benchmark.py", "Single (model, benchmark) executor with resume and force_restart support."],
      ["ethical_benchmark/pipeline/run_quant_matrix.py", "Multi-benchmark executor per model; loads each model once and runs all benchmarks sequentially."],
      ["ethical_benchmark/analysis/compare_quant_pairs.py", "Pairwise delta computation, interpretation labels, scale-sensitivity, cross-family reports."],
      ["ethical_benchmark/cluster/generate_jobs.py", "SLURM sbatch generation, supports per-benchmark or per-model grouping."],
      ["ethical_benchmark/cluster/submit_jobs.py", "Programmatic sbatch submission with dry-run support."],
      ["ethical_benchmark/cluster/check_runs.py", "Status reporting: which (model, benchmark) outputs exist, which jobs are in the queue."],
      ["ethical_benchmark/metrics/", "JSONL/CSV writers, bootstrap CI helpers, summary aggregation."],
      ["fyp_cli.py", "Unified CLI entrypoint exposing all workflows."],
    ],
    [4200, 5160],
  ),
  P("Table 4.1  Repository module responsibilities.", { size: 18 }),

  ...Figure("system_architecture.png", "System architecture. A YAML configuration is validated by the Pydantic schema into a typed run specification; each model alias is then loaded once and quantization is applied on the fly at load time, so both pair members derive from identical baseline weights. Benchmark plugins share one abstract interface and hand responses to a per-benchmark scorer stack in which the primary and secondary scorers are named explicitly. All artifacts are written before any analysis runs, and raw.jsonl is never modified afterwards."),

  H2("4.2 Configuration Schema"),
  PJ("All runtime parameters are specified by a YAML configuration file. Cluster runs typically use configs/tc1.yaml, and local development uses configs/default.yaml. The configuration is loaded and validated by a Pydantic schema in config_schema.py, which enforces structural and semantic invariants. The top-level schema has four sections: models, decoding, benchmarks, and slurm."),
  PJ("Each model entry must specify family, size_b, quantized, pair_id, model_id, trust_remote_code, dtype, and a benchmarks list, plus an optional attn_implementation. That field is the attention backend passed to from_pretrained, and it is restricted to eager, sdpa, or flash_attention_2. When it is unset it is omitted entirely, so the existing models' load call is unchanged. The schema also enforces two cross-references. Every benchmark name appearing in a model's benchmarks list must also appear as a top-level benchmarks entry, and every pair_id must have at least one model with quantized=false and at least one with quantized=true. Together, these invariants make it impossible to configure a study that omits a baseline or compares a model only against itself."),
  PJ("The decoding section specifies generation parameters. The benchmarks section provides per-benchmark dataset names, splits, sample caps, batch sizes, and benchmark-specific options such as the benign_only flag for XSTest and the subjects list for MMLU. Finally, the slurm section specifies cluster directives and bootstrap commands."),

  H2("4.3 Model Loading and Quantization Path"),
  PJ("The HFModelLoader class encapsulates all model-loading logic. Its load method accepts a ModelSpec dataclass built from the YAML configuration, and it returns the loaded model, tokenizer, and resolved runtime device. The loader is responsible for three things. First, it selects the runtime device based on the user-specified policy and CUDA availability. Second, it resolves the model dtype from a string preference of auto, float16, bfloat16, or float32 to an actual torch.dtype value, where auto resolves to float16 on CUDA and float32 on CPU. Third, it injects a BitsAndBytesConfig object into the from_pretrained call when ModelSpec.quantized is true."),
  PJ("The BitsAndBytesConfig is constructed with load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_use_double_quant=True, and bnb_4bit_compute_dtype equal to the resolved torch_dtype. Setting the compute dtype to follow the model dtype ensures that the dequantized compute path uses the same precision the model would have used at full precision. This eliminates a subtle source of pair asymmetry."),
  PJ("Two runtime safeguards are applied. First, the loader raises a clear RuntimeError if quantized is true but the resolved runtime device is not cuda. This prevents accidental CPU execution of the four-bit path, which the pinned CUDA bitsandbytes build does not provide. Second, after from_pretrained returns, the loader checks four supported quantization signals: is_loaded_in_4bit, is_loaded_in_8bit, is_quantized, and hf_quantizer. If quantization was requested but none is present, it raises RuntimeError and refuses to proceed. That refusal prevents full-precision outputs from being silently recorded as NF4 or INT8 results."),

  H2("4.4 Generation Pipeline"),
  PJ("The TextGenerator class handles all inference. It accepts the loaded model, tokenizer, runtime device, and a DecodingConfig dataclass. It then exposes a single generate_batch method, which takes a list of prompts and returns a list of generated responses."),
  PJ("Each prompt passes through a formatting step. When use_chat_template is true, which is the default, the generator wraps the prompt in a single-message user-role list and applies the tokenizer's apply_chat_template method with add_generation_prompt=True. The call also passes enable_thinking=False, and that flag matters. The Qwen3 tokenizers used here ship chat templates that enable a thinking block by default, so with the flag unset generation can spend much or all of the max_new_tokens budget producing <thinking>...</thinking> output before reaching the answer. A TypeError handler catches tokenizers that do not accept the enable_thinking keyword and retries without it. Only as a last resort does it fall back to a raw-prompt path."),
  PJ("Generation itself uses torch.no_grad and a do_sample=False decode path under temperature=0.0. Outputs are then post-processed to strip the chat-template prefix and any trailing whitespace before they are returned to the calling pipeline."),

  H2("4.5 Benchmark Plugins"),
  PJ("Benchmarks are implemented as plugins behind a small abstract base class defined in ethical_benchmark/benchmarks/base.py. Every plugin implements four methods: load_items, build_prompt, score_response, and aggregate. load_items returns a deterministic, seeded sample of BenchmarkItem objects. build_prompt maps each item to a single user-facing prompt string. score_response evaluates a model response against the item and returns a dictionary of boolean and numeric score fields. Finally, aggregate consumes the full list of per-item score dictionaries and produces the final aggregated summary."),
  PJ("The deterministic refusal layer is shared through benchmarks/utils.py. Its match_refusal_pattern function returns the matched pattern, and its is_refusal_response wrapper returns the boolean decision. XSTest uses this layer for its primary over-refusal score, whereas HarmBench retains it only for the secondary v2 non-refusal proxy. HarmBench's primary Attack Success Rate is instead scored by the official classifier. Centralising the regex layer keeps the two refusal-based measurements consistent without conflating that proxy with the classifier-primary safety endpoint."),
  PJ("The conceptual scoring loop, with error handling omitted, is as follows:"),
  ...Code(`items = plugin.load_items(max_samples=N, seed=42)
prompts = [plugin.build_prompt(item) for item in items]
responses = generator.generate_batch(prompts)
scored = [plugin.score_response(it, r) for it, r in zip(items, responses)]
summary = plugin.aggregate(scored)
write_jsonl(per_item_records, raw_path)
write_json(summary, summary_path)`),

  H2("4.6 Matrix Orchestration and Memory Management"),
  PJ("run_quant_matrix.py executes a full study or a filtered subset of it. The outer loop iterates over selected model aliases. The inner loop then iterates over the benchmarks declared for each model. With the default reuse_loaded_model=True policy, each model is loaded exactly once and used for all of its benchmarks in sequence. Loading is the most expensive operation in the pipeline, so this arrangement avoids two redundant load cycles per model. It also keeps total wall-clock time well within the six-hour TC1 walltime budget."),
  PJ("Memory management is critical on TC1, where each job is allocated only ten gigabytes of host RAM and a single GPU. A try/finally block surrounds the inner benchmark loop. When the inner loop completes, or when any benchmark within it raises, the model, tokenizer, and generator references are explicitly deleted, gc.collect() is invoked, and torch.cuda.empty_cache() is called when CUDA is available. This sequence releases the GPU memory before the next model is loaded. Without it, stacking-induced out-of-memory errors would occur immediately when a second model is loaded into the same job. That case does arise in local development, although on TC1 it is structurally avoided because each job runs a single model."),

  H2("4.7 Resume and Checkpointing"),
  PJ("Long-running jobs must survive walltime exhaustion and transient failures. The framework therefore treats raw.jsonl as the source of truth for which prompts have already been processed. Every successfully scored prompt is immediately appended to raw.jsonl, with its prompt_id as a stable record key. On startup, the get_processed_prompt_ids helper reads the existing raw.jsonl, if it exists, and returns the set of prompt_ids already on disk. prepare_remaining_items then filters the freshly-loaded benchmark items and keeps only those whose prompt_ids are not in the processed set."),
  PJ("The granularity of the resume mechanism is the (model, benchmark, prompt_id) triple. Three concrete recovery scenarios are worth stating explicitly. Suppose a job is killed during XSTest for qwen_2b_4bit. HarmBench, whose raw.jsonl is complete, is skipped entirely on the next submission. XSTest, whose raw.jsonl is partial, is resumed at the next unprocessed prompt. MMLU, whose raw.jsonl does not yet exist, runs from scratch. If instead a model fails to load, through a deterministic OOM for example, the entire model's benchmark suite is lost for that job. The operator can then re-submit with --model and --benchmark filters to chunk the workload differently."),

  H2("4.8 SLURM Job Generation"),
  PJ("Cluster orchestration is handled by ethical_benchmark/cluster/generate_jobs.py. The generator reads the loaded configuration's slurm section and emits one sbatch file per scheduling unit. A --group_by flag controls the granularity. With group_by=benchmark, one sbatch is generated per (model, benchmark) pair, which gives forty scripts for the ten-model, four-benchmark matrix. With group_by=model, the default for TC1, one sbatch is generated per model alias, and that script invokes run_quant_matrix.py so the model is loaded only once."),
  PJ("Each generated sbatch contains the standard SBATCH directives, output and error log paths, a set -euo pipefail line, a cd into the configured work_dir, a mkdir -p of the log directory, the configured setup_commands, and finally the python invocation with the model alias and configuration path passed in. The directives are partition UGGPU-TC1, qos normal, gres gpu:1, cpus-per-task 1, mem 10G, and time 06:00:00. The setup_commands are module load slurm, module load anaconda, and source activate fyp-tc1. Ten sbatch files are produced for the present configuration, one per model alias."),

  H2("4.9 Output Artifacts"),
  PJ("Every (model, benchmark) run produces a fixed set of files under the results directory. raw.jsonl contains one JSON object per prompt. Each object carries the prompt text, the model response, the per-response score fields, and the run metadata, namely the model alias, model_id, family, pair_id, quantized flag, seed, generation_config, and timestamp. summary.json contains the aggregated metrics, with bootstrap confidence intervals where applicable, alongside the same run metadata. The aggregator also appends a flat row to results/summary/<benchmark>_runs.csv. The result is a single CSV per benchmark that records all model runs."),
  PJ("The analysis stage produces the following files. results/analysis/pairwise_deltas.json and .csv hold one row per (pair_id, benchmark, metric) with absolute and relative deltas. results/analysis/pair_interpretations.csv holds one row per pair with the interpretation label and the three component deltas. results/analysis/quantization_analysis_summary.json holds a high-level study-wide summary."),

  H2("4.10 Testing"),
  PJ("The repository ships with an automated verification suite whose live inventory is collected by pytest. Appendix D records the coverage areas rather than copying a count that would become stale whenever a guard is added. Coverage includes dataset and benchmark loaders, model loading and prompt formatting, matrix reuse, pairwise analysis and bootstrap logic, score-sidecar selection, schema and resume helpers, refusal parsing, judge validation and redaction, artefact immutability, claim-surface verification, and SLURM generation."),

  H2("4.11 CLI and Operational Interface"),
  PJ("The unified CLI fyp_cli.py exposes nine subcommands, which Table 4.2 summarises: seven study and cluster commands plus two agent-harness helpers. The study and cluster commands accept the shared configuration, results-directory, and log-level options where applicable. The two harness helpers expose their own status and task-packet arguments. The Makefile provides convenient targets that wrap these invocations."),
  buildTable(
    ["Subcommand", "Purpose"],
    [
      ["smoke", "Run a small-sample sanity check on one (model, benchmark) pair."],
      ["run", "Run one (model, benchmark) pair with the full sample budget."],
      ["matrix", "Run the full or a filtered subset of the (model × benchmark) matrix."],
      ["analyze", "Compute pairwise deltas and produce interpretation labels."],
      ["cluster-generate", "Emit SLURM sbatch files for the configured study."],
      ["cluster-submit", "Submit the generated sbatch files (local/dev convenience; --dry_run prints planned submissions). Not used on TC1, where head-node Python is disallowed. TC1 jobs are submitted with direct sbatch (Section 5, TC1 runbook)."],
      ["cluster-check", "Poll squeue and the filesystem to report job and output status."],
      ["agent-status", "Print the live repository and agent-harness status."],
      ["agent-start", "Render a task-specific startup packet for an agent session."],
    ],
    [2200, 7160],
  ),
  P("Table 4.2  CLI subcommands exposed by fyp_cli.py.", { size: 18 }),
];

// ------------------------------------------------------------
// Chapter 5: Experimental Setup
// ------------------------------------------------------------
const ch5 = [
  H1("Chapter 5: Experimental Setup"),

  H2("5.1 Hardware and Cluster Environment"),
  PJ("All experiments are run on the NTU TC1 GPU cluster, a shared facility operated by the College of Computing and Data Science for undergraduate and postgraduate research workloads. Account access was approved in March 2026 under QoS \"normal\" with the parameters listed in Table 5.1. The compute partition consists of seven nodes (TC1N01–TC1N07), each equipped with three NVIDIA Tesla V100 PCIe 32 GB GPU cards, giving twenty-one GPUs across the partition. Although MaxJobsPU permits two submitted jobs, the first production submission showed an effective one-running-GPU-job limit. The second GPU job waited with reason QOSMaxGRESPerUser until the first released its GPU."),
  buildTable(
    ["Parameter", "Value"],
    [
      ["Login host (IP)", "10.96.189.11"],
      ["Username", "utan001"],
      ["Home directory", "/tc1home/FYP/utan001"],
      ["Partition", "UGGPU-TC1"],
      ["QoS", "normal"],
      ["GPU model", "NVIDIA Tesla V100 PCIe 32 GB (×3 per node)"],
      ["QoS GPU limit per job", "1 (gres=gpu:1)"],
      ["QoS CPU limit per user", "20 cores total"],
      ["QoS memory limit per user", "64 GB total"],
      ["Per-job sbatch allocation (this study)", "1 GPU, 1 CPU, 10 GB RAM"],
      ["Maximum walltime per job (default QoS)", "06:00:00"],
      ["MaxJobsPU (submitted jobs per user)", "2"],
      ["Observed running-GPU concurrency", "1 running job; second waits with QOSMaxGRESPerUser"],
      ["Storage quota (verify via MyTCinfo)", "approx. 300 GB; confirm on first login"],
      ["Account window", "March 2026 – November 2026"],
      ["Backup policy", "None (user-managed backups)"],
    ],
    [4200, 5160],
  ),
  P("Table 5.1  TC1 cluster hardware and policy parameters relevant to this study. Higher-walltime QoS (8h / 12h / 24h / up to 48h) and additional storage can be requested via the support address ccdsgpu-tc@ntu.edu.sg with a justified reason.", { size: 18 }),

  H2("5.2 Software Environment"),
  PJ("A dedicated conda environment, fyp-tc1, was created on TC1 and validated by importing the principal ML stack. The relevant versions are listed in Table 5.2. Per the TC1 user guide, the base anaconda environment cannot be modified. For this reason, all study dependencies are isolated in fyp-tc1 under the user's home directory."),
  buildTable(
    ["Package", "Version"],
    [
      ["Python", "3.11.15"],
      ["torch", "2.11.0+cu130"],
      ["transformers", "5.5.0"],
      ["bitsandbytes", "(installed; NF4 support verified)"],
      ["pydantic", "(installed via requirements.txt)"],
      ["datasets", "(installed via requirements.txt)"],
      ["huggingface_hub", "(installed via requirements.txt)"],
      ["numpy", "(installed via requirements.txt)"],
      ["pytest", "(installed via requirements.txt; run pytest tests/ for the live inventory)"],
    ],
    [4200, 5160],
  ),
  P("Table 5.2  Software environment versions on TC1.", { size: 18 }),

  H2("5.3 Cluster Usage Policy and Workflow Constraints"),
  PJ("The TC1 user guide imposes strict workflow rules that shape how this study is executed. The headline rule is that user code may not be executed on the head node (CCDS-TC1). The guide states: \"DO NOT execute your coding on TC1 Head Node ... For the user process found executing coding and occupying high CPU and Memory usage in the Head Node will be terminated with no prior notification. Repeated offenders will be banned from TC1.\" GPU verification commands such as nvidia-smi and nvcc --version are also explicitly forbidden on the head node. This is because the GPU cards reside on the compute nodes (TC1N01–TC1N07) rather than on the head node."),
  PJ("By contrast, package installation, file transfers, conda environment management, dataset downloads, and small administrative commands (squeue, scontrol, sacct, MyTCinfo, MyJobHistory, seff) are explicitly demonstrated as head-node activities throughout the user guide. The study's pre-cache step, described in Section 5.5, therefore runs entirely on the head node and is policy-equivalent to a conda install."),
  PJ("The guide also recommends sbatch over srun for all real job submission: \"Avoid using the command 'srun' to submit job ... all users are advised to use the command 'sbatch' for job submission. Then exit from the session, access later to see the result.\" Therefore, this study submits every job, including the initial smoke validation, via sbatch."),

  ...Figure("tc1_execution.png", "Execution model on the TC1 cluster. The head node is restricted to repository synchronisation, dataset and weight pre-caching, and job submission: no user code runs there. All computation runs in offline batch jobs inside the QoS envelope, and every analysis, figure and document build happens off-cluster on the workstation, which needs no GPU."),

  H2("5.4 Hugging Face Access and Gating"),
  PJ("The Qwen pairs draw from open-access official Qwen3 checkpoints (Qwen/Qwen3-1.7B and Qwen/Qwen3-4B) and require no authentication. The Llama pair uses meta-llama/Llama-3.2-3B-Instruct, which is gated under the Meta Llama community license. In addition, HarmBench currently also requires accepted Hugging Face dataset access conditions. Access is handled by logging into Hugging Face once on the TC1 head node with a read-scoped personal access token. As of 2026-05-26, token registration and gated-access acceptance were verified by a successful full pre-cache of both Llama-3.2-3B and HarmBench."),

  H2("5.4.1 Infrastructure Validation and Fixes"),
  PJ("Before the full matrix was submitted, a dedicated CUDA verification job and a five-sample smoke run validated the cluster environment end-to-end. Two infrastructure bugs were identified and fixed during this process. First, the generated sbatch scripts used relative paths for the #SBATCH --output and #SBATCH --error directives. SLURM resolves these relative to the directory from which sbatch is invoked rather than to the working directory set by cd inside the script body. Jobs submitted from the home directory therefore wrote their log files to ~/results/slurm_logs_tc1/ rather than to the repository's results/ tree, and the output appeared to be missing. The fix was applied in the job generator (ethical_benchmark/cluster/generate_jobs.py). When log_dir is a relative path and work_dir is set, the generator anchors log_dir under work_dir so all sbatch directives contain fully qualified paths."),
  PJ("Second, an initial CUDA verification script placed a Python file in /tmp on the head node and referenced it from the sbatch body. Because the /tmp filesystem is local to each node and is not shared, the compute node could not find the file. The fix was to embed the verification code inline in the sbatch script using a Python heredoc (python - << 'PYEOF' ... PYEOF), which removed the dependency on any shared path. Both fixes were committed and pushed before the smoke run. The five-sample smoke job (SLURM job ID 60975) then completed successfully, with runtime_device reporting cuda, attack_success_rate at 0.6 over five HarmBench prompts, and no malformed outputs."),

  H2("5.5 Offline-Mode Strategy and Pre-Cache"),
  PJ("Because the compute nodes may not have outbound internet access, and because runtime downloads risk burning the six-hour walltime budget on slow Hugging Face mirrors, this study adopts a strict offline-mode strategy. All datasets and model weights are pre-cached on the head node before any SLURM job is submitted. SLURM jobs then run with HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, and TRANSFORMERS_OFFLINE=1 exported in the sbatch setup_commands block. Any cache miss therefore fails immediately with a clear error rather than hanging on a network attempt."),
  PJ("The repository provides a helper script, scripts/prefetch_tc1.py, that reads the configured benchmarks and models from configs/tc1.yaml and triggers the necessary downloads. It is invoked once, on the head node, after the conda environment is activated and a Hugging Face token has been registered with huggingface_hub.login:"),
  ...Code(`module load anaconda
source activate fyp-tc1
cd /tc1home/FYP/utan001/fyp_quant/repo
python - <<'PY'                      # one-time HF login
from getpass import getpass
from huggingface_hub import login
login(token=getpass("HF token: "), add_to_git_credential=False)
PY
python scripts/prefetch_tc1.py       # or: make prefetch CONFIG=configs/tc1.yaml`),
  P("Code listing 5.1  Head-node pre-cache invocation (scripts/prefetch_tc1.py).", { size: 18, after: 160, alignment: AlignmentType.CENTER }),
  PJ("The pre-cache step retrieves HarmBench, the six MMLU subjects, ARC-Challenge (allenai/ai2_arc, added as the second capability benchmark), and the model repositories. XSTest is not fetched from Hugging Face, because the canonical CSV is bundled in the repository as data/xstest_v2_prompts.csv. The initial TC1 pre-cache completed successfully on 2026-05-27 at 15:37 UTC+8 and cached the original three model repositories in ~/.cache/huggingface/hub: Qwen3-1.7B (4.08 GB), Qwen3-4B (8.06 GB), and Llama-3.2-3B (12.9 GB), plus small dataset files."),
  PJ("When the study was extended to five pairs (Section 5.6), a second pre-cache on 2026-06-15 added the two cross-family repositories, Mistral-7B-Instruct-v0.3 and Phi-4-mini-instruct. Mistral-7B is HF-gated, so its access conditions were accepted and the head-node HF token login re-run before prefetching. The downloads themselves are pure HTTP file transfers with negligible CPU and memory cost. They are therefore consistent with the head-node activities explicitly demonstrated in the TC1 user guide (page 7–9, where conda install and pip install are shown executing on the head node)."),

  H2("5.6 Run Plan"),
  PJ("With group_by=model, the framework emits one sbatch file per model alias. This gives six scripts for the original three pairs, extended to ten when the cross-family pairs were added. Each script runs the full configured benchmark suite for its model with the model loaded only once, exploiting the matrix runner's reuse_loaded_model=True default. The initial matrix run covered the three core benchmarks. ARC-Challenge was added later as the second capability benchmark and run on the original six models via a parallel per-model job set, slurm/jobs_tc1_arc/, while the two cross-family pairs covered ARC inside their matrix jobs (see Appendix I.9). The original six scripts are:"),
  Bullet("qwen_2b_base__matrix.sbatch"),
  Bullet("qwen_2b_4bit__matrix.sbatch"),
  Bullet("qwen_4b_base__matrix.sbatch"),
  Bullet("qwen_4b_4bit__matrix.sbatch"),
  Bullet("llama_3_2_3b_base__matrix.sbatch"),
  Bullet("llama_3_2_3b_4bit__matrix.sbatch"),
  PJ("The cross-family extension ran on TC1 on 2026-06-15 and added four more matrix scripts of the same form, mistral_7b_{base,4bit}__matrix.sbatch and phi4_mini_{base,4bit}__matrix.sbatch. Section 6.8.6 reports their results. These scripts were submitted pair-by-pair under the same scheduling discipline, and each was preceded by a five-sample smoke job to confirm that the model loads with its native chat template and, for Phi-4-mini, with the eager attention backend."),
  PJ("Their HarmBench generations were judge-scored by a dedicated sbatch (slurm/judge_validation_newpairs.sbatch) that scores only the new aliases, so the committed judge sidecars of the original six models were never reopened. The primary 512-token study (Section 6.12) then re-ran the entire ten-model matrix at max_new_tokens = 512 via configs/tc1_512.yaml into the parallel results_512/ tree. That tree was judge-scored by slurm/judge_validation_512.sbatch (job 61524)."),
  PJ("Before submitting the full matrix, a single short smoke sbatch is submitted (five prompts, qwen_2b_base on HarmBench) to verify that the offline-cache path works end-to-end on a real compute node. Per the user guide's guidance to prefer sbatch over srun, this smoke verification is performed as a regular SLURM job rather than an interactive session. On 2026-05-27, smoke job 60975 completed successfully on TC1 in 33 seconds and produced a clean summary.json with runtime_device=cuda and malformed_rate=0.0. The full six-job matrix was therefore cleared for submission."),
  PJ("The first production matrix submission was made on 2026-05-27 with the Qwen 2B-class pair. Job 60976 (qwen_2b_base__matrix) began running, while job 60977 (qwen_2b_4bit__matrix) remained pending with reason QOSMaxGRESPerUser. This establishes the practical scheduling rule for the remainder of the study. Only the current model pair is submitted, and one job is expected to run while the paired job waits. The next pair is submitted only after the current pair has cleared. Under this effective one-GPU concurrency, the six model jobs run serially or near-serially. That remains within the study plan, because each job has an independent six-hour walltime allocation. Memory and CPU utilisation are recorded after each job using the seff and MyJobHistory commands to inform any subsequent right-sizing of the sbatch resource requests."),

  H2("5.7 Reproducibility Notes"),
  PJ("All runs share the same global seed (42), deterministic dataset shuffling, and deterministic decoding. The exact commit hash of the framework at the time of the final result run will be recorded in the final report and will accompany the result tables. Per-prompt records are persisted to raw.jsonl with full metadata. Two reproducibility scopes are distinguished."),
  PJ("The study can be re-executed from scratch from any repository checkout, because the configs, seeds, prompts, and pipeline are committed. The five model repositories are pinned by Hugging Face commit hash. The pin is the revision field on every model entry in configs/tc1.yaml and configs/tc1_512.yaml, set to the exact snapshots used on TC1. An online re-execution therefore loads byte-identical weights rather than whatever revision happens to be current at fetch time."),
  PJ("In addition, every reported number can be replayed from the committed analysis artefacts and redacted score sidecars. The raw per-prompt records themselves are local-only evidence, gitignored and hash-pinned in results/raw_artifact_manifest.sha256. Recomputing the analysis directly from raw outputs therefore requires the locally held evidence trees rather than a fresh clone."),
  PJ("One further provenance boundary applies to the run records themselves. They carry the model alias, pair id, and quantized flag, but they do not carry the quantization method (NF4 versus INT8) or the resolved compute dtype as explicit fields. For the committed studies these properties are fixed by the study name and configuration. The loader also guarantees that the NF4 compute dtype equals the baseline's float16. For future runs the pipeline persists both fields explicitly."),
];

// ------------------------------------------------------------
// Chapter 6: Results and Analysis
// ------------------------------------------------------------
const ch6 = [
  H1("Chapter 6: Results and Analysis"),

  PJ("All six original experimental runs completed successfully on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976–60981). Three matched pairs were evaluated: Qwen3-1.7B (jobs 60976/60977), Qwen3-4B (jobs 60978/60979), and Llama-3.2-3B-Instruct (jobs 60980/60981). A cross-family extension subsequently added two further matched pairs, Mistral-7B-Instruct-v0.3 [41] and Phi-4-mini-instruct [3], run on the same cluster on 2026-06-15 (matrix jobs 61121/61122/61123/61125, HarmBench classifier job 61134). The study therefore covers five pairs, ten models and four families. Their results appear in Tables 6.1–6.3 and are analysed in Section 6.8.6."),
  PJ("Each job ran all three core benchmarks sequentially with the model loaded once. ARC-Challenge, the second capability benchmark, was added afterwards and run on the original six models in a separate per-model job set, with the two cross-family pairs covering ARC inside their matrix jobs (Appendix I.9). The results are presented in Table 6.1 (per-pair raw metrics) and Table 6.2 (quantization deltas and interpretation labels). Detailed per-pair observations are in Section 6.5 (Qwen3-1.7B), Section 6.6 (Qwen3-4B), Section 6.7 (Llama-3.2-3B), and Section 6.8.6 (Mistral-7B and Phi-4-mini). The within-family scale analysis (RQ4) is in Sections 6.2 and 6.9. The cross-family comparison (RQ5) is in Section 6.8, extended across four families in Section 6.8.6."),

  H2("6.1 Results Table"),
  PJ("Table 6.1 records the per-pair benchmark results for all five model pairs together with bootstrap 95% confidence intervals on the deltas. HarmBench Attack Success Rate is reported under the official HarmBench classifier, the primary scorer (Appendix I.8 and 6.12), at HarmBench's own 512-token generation budget, which is the primary configuration of this study. The shorter 128-token budget is retained as an explicit generation-length comparison in Section 6.12. The v2 refusal regex is shown beneath each pair as a secondary non-refusal-rate proxy. XSTest over-refusal and MMLU accuracy are reported at the same 512-token budget. The HarmBench classifier affects only the HarmBench ASR column."),
  PJ("Confidence intervals are computed by a paired bootstrap [39] over the matched prompt set (2 000 resamples, seed 42). Both members of every pair see the same prompts in the same order, so each bootstrap draw resamples prompt indices and recomputes the delta directly. One resolution detail matters for full reproducibility. The judge-scored HarmBench ASR intervals carried in the 128-vs-512 headline comparison of Section 6.12 are computed with a higher-resolution 10 000-resample paired bootstrap at the same seed 42, whereas the capability, over-refusal, and v2-proxy intervals use 2 000. Both are percentile intervals and both replay from the committed sidecars."),
  PJ("A delta is marked statistically significant when its 95% CI excludes zero AND McNemar's exact paired test (Section 3.7) agrees at p < 0.05. At the corrected over-refusal scoring no contrast triggers the borderline case where the two criteria disagree. The previously-borderline Qwen3-1.7B over-refusal decrease resolves to non-significant on both criteria, with a CI of [−0.048, 0.000] and McNemar p = 0.109 on 10 discordant prompts."),
  buildTable(
    ["Pair", "Metric (scorer)", "Baseline", "4-bit", "Δ (95% CI)", "McNemar p", "Sig?"],
    CLAIMS.render.report_table_6_1_full,
    [1100, 2050, 820, 820, 2450, 950, 560],
  ),
  P("Table 6.1  Complete per-pair results with paired bootstrap 95% CIs on the deltas. \"Sig?\" = the CI excludes zero and McNemar's exact paired test agrees (p < 0.05). At the corrected over-refusal scoring no contrast is borderline. The previously-borderline Qwen3-1.7B over-refusal resolves to non-significant on both criteria, with a CI of [−0.048, 0.000] and McNemar p = 0.109. HarmBench ASR (judge) is the primary scorer (cais/HarmBench-Llama-2-13b-cls, fp16). The v2 non-refusal proxy is shown for transparency and is not used for the headline conclusions. Judge CIs and v2/MMLU/XSTest CIs come from the released analysis artifacts.json`.", { size: 18 }),
  ...Figure("asr_forest.png", CLAIMS.render.asr_forest_caption),
  PJ("Under the primary (judge) scorer at the 512-token reference budget, the HarmBench deltas tell a uniform story. No pair's attack-success rate increases significantly, and the only ΔASR whose 95% CI excludes zero is Llama-3.2-3B's, at −0.040. That delta is a decrease rather than an increase. The statistically significant deltas across Table 6.1 are therefore all capability or over-refusal effects: Qwen3-1.7B ΔMMLU (−0.090), the Llama-3.2-3B ΔASR decrease (−0.040), and one over-refusal decrease, Phi-4-mini (−0.048). Qwen3-1.7B's over-refusal decrease (−0.024) is non-significant, because its bootstrap CI [−0.048, 0.000] no longer excludes zero and McNemar's exact test gives p = 0.109 (Section 3.7)."),
  PJ("The ARC capability axis (Appendix I.9) adds significant losses on Qwen3-4B (−0.016) and Llama-3.2-3B (−0.032). Not one of the five pairs shows a significant harmful-compliance increase under NF4. The picture is sharper still after multiplicity correction (Section 6.4). Under a Benjamini–Hochberg control [42] over the family of primary contrasts, no HarmBench ASR contrast survives at all, and the three surviving effects are all capability or over-refusal changes."),
  PJ("Every XSTest over-refusal delta is non-significant or points downward. The only significant delta, Phi-4-mini at −0.048, is a decrease, and the previously-borderline Qwen3-1.7B decrease is non-significant. No pair therefore becomes more over-refusing under quantization (RQ2). The demoted v2 non-refusal proxy again reports much higher absolute ASR and a different pattern, analysed in Section 6.9, and the judge values are authoritative. The shorter 128-token budget, retained for direct comparison in Section 6.12, produced an apparent Qwen3-1.7B ASR increase of +0.055 that does not appear at the reference budget."),

  PJ("HarmBench ASR was originally scored by a deterministic refusal regex and is now scored by the benchmark's own fine-tuned classifier; over-refusal remains regex-scored as the pre-specified scorer of record, cross-checked by an independent judge. Appendix I.8 records the revision history in full, and Appendix H the formal correction record."),
  H2("6.2 Within-Family Scale Analysis (RQ4)"),
  PJ("RQ4 asks whether the magnitude of the quantization effect differs between two-billion and four-billion parameter Qwen models. Both pairs were evaluated. Table 6.2 summarises the quantization deltas and interpretation labels for all five pairs, enabling both the within-family scale comparison (RQ4, the two Qwen pairs) and the cross-family comparison (RQ5)."),
  buildTable(
    ["Metric", "qwen_2b", "qwen_4b", "llama_3.2_3b", "mistral_7b", "phi4_mini"],
    CLAIMS.render.report_table_6_2,
    [2200, 1240, 1240, 1480, 1320, 1320],
  ),
  P("Table 6.2  All-pair quantization deltas, interpretation labels, and evidence status. Δ = 4-bit − baseline. HarmBench ΔASR (judge) is the primary scorer, and the v2 proxy row is shown for transparency. ★ marks deltas significant under both criteria (paired-bootstrap 95% CI excludes zero and McNemar's exact p < 0.05). At the corrected over-refusal scoring no Δover-refusal is borderline. The previously-borderline Qwen3-1.7B Δover-refusal resolves to non-significant on both criteria (CI [−0.048, 0.000], p = 0.109, Section 3.7). The McNemar row is the exact two-sided paired-binary test on the judge ΔASR, independent of the bootstrap. Under the judge at the 512-token reference budget, the ★ deltas are Llama-3.2-3B ΔASR (−0.040, a decrease, McNemar p = 0.021), Qwen3-1.7B ΔMMLU (−0.090), and Phi-4-mini's over-refusal decrease (−0.048). The Llama-3.2-3B ΔASR is the only significant ΔASR, and no pair shows a significant increase. The two-layer evidence status reports each label's statistical support separately from the label itself (Section 3.8). Qwen3-4B keeps the alignment_degradation label by point estimate but is flagged directional because its ΔASR CI touches zero (McNemar p = 0.096). It is not a statistically confirmed regression. Labels are judge-primary.", { size: 18 }),
  PJ(CLAIMS.render.bh_survivor_sentence),
  PJ("Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization procedure, the 1.7B model experiences large capability degradation, with MMLU dropping 9.0 percentage points, a relative decline of 14.0%, and a CI that excludes zero. The 4B model is essentially unaffected, with MMLU moving by only 0.3 percentage points, well within sampling noise. The MMLU delta ratio (9.0 pp versus 0.3 pp) is approximately 30:1 under the primary lenient parser. That is the largest single capability contrast observed in this study, and it is a two-pair, within-Qwen, MMLU-anchored comparison."),
  PJ("This 30:1 ratio is a lenient-parser figure. On ARC under the same parser the 4B pair loses marginally more than the 1.7B (−1.6 pp vs −0.9 pp), which at first suggests the scale gap is MMLU-specific. However, Appendix I.9 shows that lenient reading is a parsing artefact, since under a strict answer-parser the 1.7B pair loses far more than the 4B on ARC too (−0.343 vs −0.017). Scale-as-predictor-of-capability-preservation is therefore parser-dependent in magnitude and is read as an observed within-family contrast rather than a benchmark-independent law."),
  PJ("The full-precision baselines reveal that within the Qwen family the larger model is both more capable and more refusal-calibrated. Under the primary (judge) scorer, Qwen3-1.7B's baseline HarmBench ASR is 0.255 while Qwen3-4B's is 0.115, so scaling from 1.7B to 4B reduces genuine baseline harmful compliance by 14.0 percentage points. MMLU accuracy simultaneously rises from 0.643 to 0.747 (+10.4 pp), which confirms the capability gain is genuine. XSTest over-refusal falls from 0.052 to 0.028, indicating fewer false refusals on benign prompts. The 4B baseline therefore dominates the 1.7B baseline on every dimension."),
  PJ("Under the demoted v2 non-refusal proxy the baseline ASR gap looks even larger, 0.595 versus 0.235, but that reflects the proxy's over-counting. The genuine harmful-compliance gap is the 14.0 pp judge figure (see Section 6.9). This larger-is-safer-and-more-capable pattern is itself a useful empirical finding, and it motivates the matched-pair design. Quantization effects must be interpreted against a baseline that already varies substantially by scale within the same family."),
  PJ("The most important comparison is on the post-quantization side, read under the primary (judge) scorer at the reference budget. The 1.7B pair's harmful-compliance rate does not move (judge ΔASR 0.000, McNemar p = 1.000) while its capability drops sharply (ΔMMLU −9.0 pp, CI excludes zero). The degradation is therefore capability-only, and the broad_degradation label the pair receives is driven entirely by that capability loss rather than by any safety change."),
  PJ("The 4B pair shows the contrasting profile the anchor exists to separate. Capability is preserved (ΔMMLU −0.3 pp, within tolerance) while the judge ΔASR is positive but not significant (+4.0 pp). Its CI is [0.000, +0.080] and McNemar's exact test gives p = 0.096. That is alignment_degradation at the point-estimate level, carried as directional."),
  PJ("Without the MMLU anchor the 4B safety nudge could be mistaken for a capability artefact, and the 1.7B capability loss could be read as a safety story. With the anchor, each pair's two axes are interpreted jointly. This is the methodological payoff of the capability-anchored design. The demoted v2 proxy paints a different and misleading picture here, a −2.5 pp 1.7B ΔASR and a +7.0 pp 4B ΔASR. Section 6.9 explains why those are regex over-counting artefacts and why the judge values above are authoritative."),

  H3("6.2.1 Preliminary Scale Observations: Baseline Comparison"),
  PJ("The Qwen3-1.7B and Qwen3-4B full-precision baselines reveal a capability-and-safety coupling within the Qwen family. At full precision, with HarmBench scored under the official classifier at the 512-token reference budget, scaling from 1.7B to 4B parameters shifts every dimension. HarmBench ASR falls 0.255 → 0.115, a change of −14.0 pp, so the larger model produces fewer genuinely harmful completions. MMLU accuracy rises 0.643 → 0.747 (+10.4 pp), making the larger model substantially more capable. XSTest over-refusal falls 0.052 → 0.028 (−2.4 pp), giving fewer false refusals on benign prompts."),
  PJ("Within Qwen the larger model dominates the smaller one on every dimension: more capable, more safety-calibrated, and better calibrated on benign prompts. Under the v2 non-refusal proxy the baseline ASR gap looks larger, reflecting the proxy's over-counting. The genuine harmful-compliance gap is the 14.0 pp classifier figure. This baseline divergence motivates the matched-pair design, because quantization effects must be interpreted against scale-dependent baselines."),

  H2("6.3 Capability Anchoring"),
  ...Figure("capability_anchor.png", "The capability-anchored safety space. Each pair is placed by its capability delta (ΔMMLU, x-axis) and its harmful-compliance delta (judge ΔASR, y-axis); dashed lines mark the interpretation thresholds and the shaded quadrants name the diagnostic labels. Bars are paired-bootstrap 95% CIs. The anchor is what separates the broad_degradation pairs (top-left) from a capability-preserving alignment nudge (Qwen3-4B) and from Mistral-7B's capability-preserving safety improvement (bottom-right)."),
  PJ("Capability anchoring is the core methodological contribution of the interpretation framework. Each pair's MMLU delta is used as a covariate when interpreting its HarmBench and XSTest deltas. Six canonical outcomes are tracked, organised symmetrically around both the safety and over-refusal axes:"),
  Bullet("Alignment degradation: MMLU is preserved, but HarmBench ASR rises beyond threshold. The model becomes more compliant with harmful prompts without a corresponding capability change. This is a direct safety regression."),
  Bullet("Alignment improvement: MMLU is preserved, but HarmBench ASR falls beyond threshold. Quantization reduces harmful compliance without measurable capability cost. This is the mirror image of alignment degradation and the most desirable outcome in the taxonomy."),
  Bullet("Capability-collapse masquerading as safety: MMLU drops noticeably, and HarmBench ASR also drops. The apparent safety improvement is attributed to reduced instruction-following capacity rather than a genuine alignment change."),
  Bullet("Robust preservation: all three deltas are small in magnitude. Quantization is effectively neutral across safety and capability dimensions."),
  Bullet("Over-refusal regression: HarmBench ASR and MMLU are held within tolerance, but the model refuses materially more benign prompts (ΔOR beyond threshold). This is a benign-refusal regression named on its own over-refusal axis rather than absorbed into broad degradation (Table 3.4, Section 3.8)."),
  Bullet("Broad degradation: the fallback label. It typically signals over-refusal moving outside tolerance, or ASR within tolerance but capability dropping. It is used when no specific pattern above matches."),
  PJ("The five pairs in this study illustrate why the capability anchor is necessary, especially at the reference budget where none of the harmful-compliance deltas is a significant increase. For the qwen_2b pair (Qwen3-1.7B), MMLU fell significantly by −9.0 pp, a relative decline of −14.0%, with a significant CI of [−0.140, −0.043]. The judge HarmBench ASR did not move at all (ΔASR 0.000, CI [−0.055, +0.055]), and McNemar's exact test gives p = 1.000. The harm axis is flat, so the broad_degradation label the pair receives is driven entirely by the significant capability loss rather than by any safety change. The anchor is what makes clear that this is a capability-only degradation rather than a safety regression."),
  PJ("For the qwen_4b pair, MMLU is preserved (−0.3 pp, within tolerance, not significant) while the judge ΔASR is positive but not significant (+4.0 pp, CI [0.000, +0.080]). McNemar's exact test gives p = 0.096. Capability held with a directional, unconfirmed safety worsening, which is the alignment_degradation pattern at the point-estimate level, carried with a directional evidence status. The anchor is exactly what licenses reading the 4B nudge as a tentative alignment direction rather than a capability artefact, because capability did not move."),
  PJ("For the llama_3_2_3b pair, capability drops on both anchors (ΔMMLU −3.7 pp, not significant; ΔARC −3.2 pp, significant) while the judge ΔASR is a significant decrease of −4.0 pp, CI [−0.075, −0.010], with McNemar p = 0.021. Harm and capability fall together, so the label is capability_collapse_masquerading_as_safety, and the apparent safety gain is attributed to reduced instruction-following rather than genuine alignment."),
  PJ("The contrast is decisive. Without the anchor the 4B directional nudge could not be separated from a capability artefact, and Llama-3.2-3B's ASR decrease would be misread as a safety improvement rather than a symptom of capability loss. The anchor turns otherwise ambiguous quantization outcomes into separable diagnostic categories, and the second capability benchmark (Appendix I.9) shows why the anchor itself is most trustworthy read across more than one benchmark. Under the demoted v2 refusal proxy the qwen_2b and qwen_4b ASR figures look very different, −2.5 pp and +7.0 pp respectively. Section 6.9 explains why those are regex artefacts and why the judge values above are authoritative."),

  PJ("The capability finding does not rest on MMLU alone. ARC-Challenge was added as a structurally different second anchor, on a much larger item set, and it agrees: capability declines on both benchmarks, and two of the three multiplicity survivors are capability contrasts, one on each. Appendix I.9 gives the per-pair ARC comparison and the parser-sensitivity check behind it."),
  H2("6.4 Statistical Caveats"),
  PJ("Five statistical limitations should be borne in mind when interpreting the results. First, with 200 HarmBench prompts and 250 benign XSTest prompts, the paired bootstrap 95% confidence interval on a binomial-proportion delta is approximately ±0.05 (Table 6.1), so small deltas may not be statistically distinguishable from zero. Under the primary (judge) scorer at the 512-token reference budget, the only HarmBench ΔASR whose CI excludes zero is Llama-3.2-3B's, and it is a decrease of −0.040, CI [−0.075, −0.010], with McNemar p = 0.021. Every other pair is within noise: Qwen3-1.7B (0.000, CI [−0.055, +0.055], McNemar p = 1.000), Qwen3-4B (+0.040, CI [0.000, +0.080], McNemar p = 0.096), Mistral-7B (−0.020, CI [−0.080, +0.040]) and Phi-4-mini (+0.020, CI [−0.015, +0.055]). No pair shows a significant harmful-compliance increase, so the only individually-significant NF4 ΔASR is a safety-improving decrease. The INT8 precision point in Section 6.11 shows no significant ASR move at all at this budget either."),
  PJ("Three of the five caveats are method-level and are set out in Appendix I.14: deterministic decoding means the bootstrap interval reflects prompt-sampling variance only, which is why the multi-seed arm exists; the MMLU subset pools three hundred questions across six subjects with uneven counts, so subject-level rates are descriptive; and the option-letter parser falls back to a lenient scan, which a strict-parser sensitivity check quantifies."),
  PJ("Fourth, HarmBench ASR is scored by the official HarmBench classifier as primary. The v2 refusal regex is retained only as a secondary non-refusal-rate proxy, because the judge validation (Section 6.9) showed it materially over-counts ASR. XSTest over-refusal is scored by the v2 regex, since the HarmBench classifier does not cover the over-refusal question. This includes the single over-refusal contrast that survives the multiplicity correction, Phi-4-mini's −0.048 decrease (Section 6.4.1), which therefore inherits the same weaker-construct status as the regex and should be read as a reproducible non-refusal-rate change rather than a classifier-validated over-refusal shift."),
  PJ("An independent three-class refusal judge (Section 6.9, Result 6) makes this concrete. It does not reproduce the −0.048 decrease (judge ΔOR +0.016 strict / −0.004 broad, both non-significant), and it agrees with the regex only poorly-to-moderately on benign over-refusal, with Cohen κ [43] from −0.01 to 0.50 under the strict mapping, and to 0.54 under the broad one. This contrast is therefore scorer-dependent. The pre-registered human audit (Section 6.9, Result 6) resolves the weighting, because the judge aligns substantially better with a blinded single annotator than the regex on this construct (strict κ 0.485 versus −0.006, regex recall 2 of 63 full refusals). The contrast is accordingly carried as most plausibly a measurement artefact of the regex scorer, and the validation-informed parallel correction (Section 6.4.1) states the over-refusal conclusion without it. Fifth, no per-comparison flag should be read in isolation. Across the five pairs the study reports roughly twenty primary delta confidence intervals (HarmBench ASR, MMLU, ARC, over-refusal) plus the per-category breakdown and the INT8 precision sweep (Section 6.11), so the nominal significance flags are corrected explicitly in Section 6.4.1. The signals there that survive the correction are all capability or over-refusal losses. Not one HarmBench ASR contrast survives it, which is consistent with the null-safety, capability-driven reading of Section 6.10. Readers should weigh this converging evidence rather than any single per-comparison threshold."),

  H3("6.4.1 Multiple-comparison correction and statistical power"),
  PJ("Rather than only acknowledging the multiplicity problem, we report the corrected view. Applying a Benjamini-Hochberg false-discovery-rate correction (q < 0.05) to the family of twenty primary NF4-vs-fp16 contrasts (five pairs × {HarmBench ASR (judge), MMLU, ARC, over-refusal}, with an exact McNemar p-value computed for every paired-binary contrast), exactly three contrasts survive, and all three are capability or over-refusal losses: Qwen3-1.7B MMLU (−0.090, q = 0.008), Llama-3.2-3B ARC (−0.032, q = 0.008) and Phi-4-mini XSTest over-refusal (−0.048, q = 0.012)."),
  PJ("The two capability survivors rest on deterministic exact-match scoring, since MMLU and ARC parse an option letter and no refusal scorer is involved. By contrast, the over-refusal survivor rests on the demoted refusal regex and is scorer-dependent. An independent three-class refusal judge does not reproduce Phi-4-mini's −0.048 decrease (judge ΔOR +0.016 strict / −0.004 broad, both non-significant), as reported in Section 6.9, Result 6, so it should be read as a reproducible property of the regex rather than a validated over-refusal shift."),
  PJ("Not one HarmBench ASR contrast survives the correction at the 512-token reference budget, including the individually-significant Llama-3.2-3B ΔASR decrease, which becomes nominal once multiplicity is accounted for. Capability degradation is therefore the signal of four-bit NF4 that survives multiplicity in this study, and harmful-compliance change does not survive the correction on any pair. This is consistent with, and strengthens, the capability-driven mechanism reading (Section 6.10). The FDR-surviving effects are all capability or benign-refusal losses, and there is no surviving evidence of a harmful-compliance increase under NF4."),
  PJ("The over-refusal member of every contrast above is regex-scored. The human audit in Section 6.9, Result 6, subsequently found that scorer poorly aligned with a blinded human annotator on this construct. A validation-informed parallel correction is therefore also reported. It is clearly labelled post-hoc, because it was added after both scorers' results were known. Its composition was nevertheless fixed in a dated pre-analysis note before it was computed (docs/VALIDATION_INFORMED_FAMILY_NOTE.md)."),
  PJ("The parallel correction recomputes the identical twenty-contrast family with a single change. The five over-refusal contrasts are scored by the independent three-class judge under the strict mapping, which is the replication definition fixed in the pre-registration. The other fifteen p-values are reused verbatim, and the same Benjamini-Hochberg procedure is applied at the same threshold. " + CLAIMS.render.dual_family_sentence + " The parallel family is purely deflationary, because no contrast becomes significant that was not already significant. The original regex-scored family above therefore remains the pre-specified family of record for traceability."),
  PJ("The substantive RQ2 conclusion is carried in its scorer-invariant form. No scorer finds a statistically significant over-refusal increase in any pair. Point estimates do move upward in places, and the judge-strict Qwen3-1.7B value of +0.040 is the largest, at McNemar p = 0.087. However, none reaches significance under either scorer. Per-contrast values are reported in the analysis artifacts released with the project."),
  PJ("A power analysis quantifies why so few effects reach significance, using Connor's normal-approximation sample-size treatment of the paired McNemar design [44]. For a two-sided α = 0.05, 80%-power McNemar test at the observed HarmBench discordant rates (median discordant rate ≈ 0.09) and n = 200, the minimum detectable ΔASR is roughly ±0.06, with a representative MDE ≈ 0.059. No NF4 ASR increase sits above this detection floor on any pair, so the predominance of nulls on the safety axis reflects a detection floor as much as a substantive absence of effect."),
  PJ("The study is therefore underpowered for the small safety effects it might have detected. Underpowered comparison is a documented and widespread condition in NLP evaluation rather than a quirk of this study [45]. Card et al. recommend running power analyses before an evaluation and sizing test sets to the minimum detectable effect, and they identify enlarging the test set as the remedy for a comparison that lacks power. The detectable-effect bound is therefore stated here explicitly rather than left implicit, and the underlying limitation, a larger prompt set, is taken up in Chapter 8. The capability axis was checked against the same detection floor rather than assumed to be adequately powered. Running the identical procedure on the MMLU and ARC contrasts gives per-contrast minimum detectable effects of 0.052–0.072 on MMLU (n = 300) and 0.021–0.033 on ARC (n = 1 172), and only two of the ten capability contrasts have an observed effect at or above their own floor: Qwen3-1.7B MMLU (−0.090 against a floor of 0.072) and Llama-3.2-3B ARC (−0.032 against 0.026). Those two are precisely the two that survive the multiplicity correction. The remaining eight capability contrasts are underpowered in the same way the safety axis is, so the capability finding should be read as two adequately resolved effects rather than as a well-powered survey of the whole capability axis. One sidedness note is worth adding. Every test in the study is two-sided, and under one-sided versions of the directional no-increase null the two borderline positive contrasts would cross α = 0.05 individually while still not surviving the BH correction. The two are Qwen3-4B ASR at one-sided p ≈ 0.048, and the judge-strict Qwen3-1.7B over-refusal at ≈ 0.044. The corrected headline is therefore sidedness-invariant. Figures and per-contrast values are in the released analysis artifacts.{json,csv}."),

  H2("6.5 Observations: Qwen3-1.7B Pair (pair_id qwen_2b)"),
  PJ("Table 6.1 carries the full contrast for this pair; the reading is that the broad_degradation label, at evidence status confirmed, is driven entirely by capability loss. Harmful compliance does not move at all, and both the paired bootstrap and McNemar's exact test [40] agree on that null. MMLU falls significantly, and that loss is one of only three contrasts in the whole NF4 family to survive the BH-FDR correction (q = 0.008, Section 6.4). Over-refusal declines slightly and does not reach significance."),
  PJ("Quantization makes the smallest model markedly less capable without making it more willing to produce genuinely harmful content. Because the harm axis is flat while capability drops beyond tolerance, the label is broad_degradation only in the sense that the significant capability collapse triggers it. There is no accompanying harmful-compliance increase, and this pair must not be read as a safety regression at the reference budget."),
  PJ("The prominent +0.055 increase reported for this pair at the 128-token budget (Section 6.12) was a truncation artefact. At 128 tokens roughly 60% of all responses were provably cut off before completion, established by a direct prefix test recorded in the released analysis artifacts. The classifier was therefore scoring incomplete generations, and where the cut fell could shift a label in either direction. Extending the budget to 512 tokens dissolves the effect to zero. No HarmBench ASR contrast in the NF4 family survives the BH-FDR correction (Section 6.4), and this pair is not the exception."),
  PJ("The secondary v2 proxy agrees for once: it reports a small negative delta at this budget, so both scorers find the harm axis static for this pair. The three dimensions together describe a model that, under NF4 compression, produces genuinely harmful completions at the same rate, refuses benign prompts slightly less often, and answers markedly fewer factual questions correctly. The pair's entire quantization signal is on the capability axis."),
  PJ("The Qwen3-1.7B pair is thus a capability-only degradation at the reference budget: significantly less capable than its full-precision counterpart, with no measurable change in harmful compliance. The capability anchor is what makes that unambiguous, since the significant ΔMMLU establishes a real cost while the null ΔASR shows no detectable safety regression, bounded by the detection floor in Section 6.4.1. This is the study's clearest single case of NF4 eroding competence rather than guardrails.")
,
  PJ("Robustness to stochastic decoding was checked by re-running all five pairs on HarmBench at temperature 0.7 with top-p 0.8 across five independent seeds, both members of a pair sharing each seed. No pair is robustly positive: every per-seed |ΔASR| is at most 0.055 and every seed-mean |ΔASR| at most 0.029, and the greedy point estimate sits inside the seed range for the three original pairs. The stochastic arm therefore corroborates the RQ1 null across the full matrix. Per-seed deltas, the four individually significant seeds, and the two cross-family pairs whose greedy deltas fall just outside their narrow seed ranges are set out in Appendix I.1."),
  PJ("Because the aggregate Qwen3-1.7B ΔASR is exactly zero at the reference budget, there is no net increase to decompose by harm type. The judge ASR was nonetheless broken down by HarmBench semantic category to confirm the null is not masking a large offsetting swing between categories. The per-category deltas run in both directions and roughly cancel, and no single category drives a systematic increase. Counts per category are small, so those rates are descriptive rather than individually tested. Appendix I.2 gives the breakdown."),
  H2("6.6 Observations: Qwen3-4B Pair (pair_id qwen_4b)"),
  PJ("The Qwen3-4B pair receives the alignment_degradation label by point estimate, at evidence status directional. Harmful compliance rises, but the bootstrap CI's lower bound touches zero and McNemar's exact test agrees the move is not significant, so the two-layer scheme reads this as suggestive rather than confirmed. MMLU is preserved; the one significant capability signal is on the second benchmark, where ARC falls at nominal, uncorrected significance and does not survive the BH-FDR correction (Section 6.4.1). The 4B pair therefore does not fully preserve capability even though its MMLU is flat."),
  PJ("This is a notable demotion from the secondary v2 proxy, whose non-refusal-rate delta for this pair is far larger. The judge validation shows the proxy over-counts by refusal-counting: the regex marks many non-refusals as harmful, inflating both the level and the delta. On the genuine harmful-compliance measure the 4B effect is real in direction but does not reach significance at n = 200, and no HarmBench ASR contrast in the study survives multiplicity correction (Section 6.4). The honest reading is that NF4 may modestly increase Qwen3-4B's harmful compliance, although the present sample cannot confirm it."),
  PJ("A plausible mechanism for the directional increase is the same NF4 degradation of refusal precision seen elsewhere. Unlike the 1.7B pair, the 4B model retains its capability, so any safety move here is not a capability artefact; it simply is not large enough to confirm at this sample size. Subject-level MMLU corroborates the preserved-capability reading, with only two of six subjects regressing against the 1.7B pair's five-of-six broad regression. Per-subject figures for all pairs are in the released analysis artifacts.{json,csv}."),
  PJ("Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under the same quantization. At the 512-token reference budget, however, neither failure is on the harmful-compliance axis. The 1.7B model receives the broad_degradation label driven entirely by a significant capability loss (ΔMMLU −0.090) with its harm axis flat (ΔASR 0.000). The 4B model preserves MMLU but loses ARC significantly, and shows only a directional, non-significant safety worsening (alignment_degradation by point estimate, ΔASR +0.040). The within-family contrast is therefore a capability contrast. The smaller model suffers a large, significant MMLU loss while the larger model's capability loss surfaces only on ARC. On the safety axis both pairs are within noise. This points the same direction as Badshah and Sajjad's preprint report [29] that larger models are more resilient to four-bit precision reduction."),
  PJ("The Llama-3.2-3B pair results (Sections 6.7–6.11) allow assessment of whether any safety worsening reproduces cross-family."),

  H2("6.7 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b)"),
  PJ("Llama-3.2-3B-Instruct is strongly refusal-calibrated at full precision: only 20 of 200 direct harmful requests produce genuinely harmful completions under the judge, over-refusal is low, and MMLU accuracy closely matches the Qwen3-1.7B baseline."),
  PJ("This profile describes a model that is both compactly capable and safety-calibrated at full precision. It is safer on HarmBench than the smaller Qwen model, whose judge baseline ASR is 0.255 for Qwen3-1.7B, while remaining approximately equally capable as the smaller Qwen on MMLU. This is also the pair where the v2 regex and the judge agree most closely (κ ≈ 0.71–0.84, Section 6.9). Llama-3.2-3B mostly either refuses cleanly or complies, leaving little ambiguous middle ground for the regex to mis-score."),
  PJ("Under NF4 the pair is classified as capability_collapse_masquerading_as_safety. Harmful compliance falls significantly, with McNemar's exact test agreeing: the 4-bit model refuses more of the direct harmful requests than the baseline, and this is the only individually significant ΔASR among the five pairs at the reference budget. Over-refusal rises modestly and non-significantly. Capability declines on both anchors, moderately on MMLU where the fall does not reach significance at n = 200, and significantly on ARC-Challenge."),
  PJ("Why capability_collapse_masquerading_as_safety? Harmful compliance decreases and capability decreases together. The significant ARC loss establishes a genuine capability cost, and the ASR falls rather than rises. Therefore the apparent safety improvement is diagnosed as a by-product of a less-capable model refusing, or failing to produce coherent harmful content, more often, rather than as a strengthening of alignment. This is diagnostically informative and it is the crux of the study's central dichotomy. A naïve reading of the significant ΔASR = −0.040 as 'NF4 makes Llama safer' is exactly the misinterpretation the two-layer scheme is built to catch."),
  PJ("The contrast with Qwen3-1.7B is instructive. At nearly identical baseline capability (MMLU 0.610 vs 0.643), the smaller Qwen loses significant MMLU capability with a flat harm axis (ΔASR = 0.000). Llama-3.2-3B, by contrast, loses capability while its harm axis moves in the safety-improving direction. In both pairs, however, the confirmed effect is capability loss rather than any alignment shift."),

  H2("6.8 Cross-Family Comparison (RQ5)"),
  PJ("RQ5 asks whether the Qwen pattern replicates qualitatively in Llama-3.2-3B. The cross-family analysis examines two properties. The first is sign consistency: do the Llama-3.2-3B deltas have the same sign as the Qwen deltas? The second is approximate magnitude: are the Llama-3.2-3B deltas similar in scale? The full cross-family comparison is in Section 6.8, and this section provides a high-level orientation."),
  PJ("The headline finding for RQ5 is that, at the 512-token reference budget under the official classifier, NF4 quantization produces no significant harmful-compliance increase in any pair, and the only individually-significant ΔASR (Llama-3.2-3B, −0.040) is a decrease. The Qwen3-1.7B pair shows ΔASR at exactly zero paired with a large, significant capability loss (ΔMMLU −0.090). The Qwen3-4B pair shows a directional, non-significant +0.040 at preserved capability. The Llama-3.2-3B pair shows a small significant ASR decrease alongside a significant ARC loss (Appendix I.9)."),
  PJ("The capability dimension is directionally consistent, since all five pairs have negative ΔMMLU. The Qwen3-1.7B loss is the only MMLU-significant one, with further ARC-significant losses on Qwen3-4B and Llama-3.2-3B. On over-refusal, the only significant delta (Phi-4-mini, −0.048) is a decrease, and the previously-borderline Qwen3-1.7B decrease is non-significant (McNemar p = 0.109)."),
  PJ("The composite picture is that NF4 quantization does not produce a significant or uniform safety regression in any family or at any scale. The signals that survive the multiplicity correction are all capability losses. See Section 6.8 for the full discussion and Section 6.8.6 for the cross-family extension, which confirms the pattern across Mistral-7B and Phi-4-mini, with no significant ASR increase in either."),

  PJ("The cross-family comparison places the Llama-3.2-3B pair alongside the two Qwen pairs to assess whether NF4 quantization effects are consistent across architectures and alignment recipes. Table 6.2 (Section 6.2) shows the delta values and interpretation labels for all three original pairs side by side. In addition, the cross-family extension to Mistral-7B and Phi-4-mini is analysed in Section 6.8.6."),

  PJ("Four findings hold across the original three pairs. First, baseline safety profiles differ substantially between families even at matched factual capability. This study does not vary training or alignment recipe, so it cannot identify which release-level factor produces that difference. Second, on harmful compliance exactly one ΔASR is individually significant and it is a decrease, and no ΔASR contrast survives the Benjamini-Hochberg correction over the twenty-test NF4 family; the Qwen3-1.7B increase that was the study's original headline at the 128-token budget does not reproduce at the reference budget. Third, over-refusal shows no significant increase in any family. Fourth, every one of the three pairs carries at least one significant capability loss, which is what drives each interpretation label. Appendix I.13 works through the four dimensions pair by pair."),
  H3("6.8.6 Cross-Family Extension: Mistral-7B and Phi-4-mini"),
  PJ("To test whether the findings generalise beyond the original two families, two further matched pairs were added and run on TC1 on 2026-06-15 under the identical methodology. That methodology is on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding at temperature 0.0, seed 42, the same four benchmarks at the same sample counts, and the official HarmBench classifier as the primary ASR scorer. The two pairs are mistral_7b (Mistral-7B-Instruct-v0.3, 7.2B, the largest model in the study) and phi4_mini (Phi-4-mini-instruct, 3.8B)."),
  PJ("This takes the study to five pairs across four families (Qwen, Llama, Mistral, Phi). Both pairs' raw metrics, deltas, and judge-agreement figures appear in Tables 6.1, 6.2 and 6.3, and their second-judge (gpt-4o) cross-check is included in the Section 6.9 figures. The two pairs fall into the two diagnostic camps already seen among the original three."),
  PJ("Mistral-7B is the clearest demonstration in the whole study of the judge-over-proxy finding (Section 6.9). The v2 refusal proxy reports a high baseline non-refusal rate of 0.830, rising to 0.890 under quantization. Taken at face value, that +0.060 'increase' would label the pair alignment_degradation. The official classifier tells the opposite story. At the 512-token reference budget genuine harmful compliance is 0.585 at baseline and 0.565 under 4-bit, a ΔASR of −0.020. That delta is not significant, with CI [−0.080, +0.040] and McNemar p = 0.627. Under the classifier, quantization slightly reduces true harmful compliance rather than increasing it."),
  PJ("The proxy and the judge disagree in sign, and the judge–proxy agreement is the lowest in the study, at Cohen's κ = 0.29 at baseline and 0.25 under 4-bit. At the 128-token budget it had been lower still, at 0.19/0.11. The second independent judge (gpt-4o) corroborates the classifier rather than the proxy. It agrees with the primary classifier at κ 0.83 at baseline and 0.78 under 4-bit, and it reproduces a small negative ΔASR."),
  PJ("Capability is essentially preserved (MMLU −2.0 pp, n.s.; ARC +0.9 pp, n.s.). Over-refusal is flat (ΔOR = 0.000, n.s.). Under the judge the pair therefore reads as alignment_improvement at the point-estimate level, carried as directional because its ΔASR CI spans zero. For the study's largest model, NF4 leaves genuine harmful compliance unchanged-to-slightly-lower while preserving capability. A refusal-based scorer would have badly mischaracterised it."),
  PJ("Phi-4-mini falls in the opposite, Llama-like camp, where the proxy and the judge largely agree. Genuine harmful compliance is low across the pair, with a judge ASR of 0.070 at baseline and 0.090 under 4-bit, a ΔASR of +0.020. That delta is not significant, with CI [−0.015, +0.055] and McNemar p = 0.424. It sits exactly on the +0.02 harm tolerance, so the pair takes the alignment_degradation label at the point-estimate level, but with an evidence status of directional rather than confirmed. The CI includes zero and the change is not statistically distinguishable from zero. This is therefore a directional, on-the-boundary point estimate rather than a confirmed safety regression."),
  PJ("The judge–proxy agreement is high (κ = 0.67, like Llama-3.2-3B), and at the reference budget the second judge agrees at κ 0.68 at baseline and 0.83 under 4-bit. That baseline figure is in fact the study's lowest cross-judge agreement, and the substantial-to-almost-perfect κ 0.79/0.95 belongs to the 128-token budget."),
  PJ("On capability, Phi-4-mini shows a modest, non-significant dip on both axes, with MMLU at −2.7 pp and ARC at −1.5 pp (both n.s.). Its one significant delta is on over-refusal, where the 4-bit model is slightly less over-cautious on benign prompts at ΔOR = −0.048. That delta is significant under the primary regex scorer, with CI [−0.076, −0.020]. It is the only significant over-refusal delta in the study, and it runs in the benign direction, towards fewer false refusals rather than more. However, this contrast is regex-scored and scorer-dependent, because an independent three-class refusal judge does not reproduce it (Section 6.9, Result 6)."),
  PJ("Across all four families the central pattern holds and strengthens. Under the official classifier at the 512-token reference budget, none of the five pairs shows a significant increase in genuine harmful compliance under NF4. No pair's ΔASR is a significant increase, and the only individually significant contrast is the Llama-3.2-3B decrease (−0.040). The two new families add a non-significant decrease for Mistral-7B (−0.020) and a non-significant increase for Phi-4-mini (+0.020)."),
  PJ("The proxy-reliability split spans four families. The regex over-counts severely for Qwen and Mistral (judge-vs-proxy κ 0.25–0.59), but it tracks the judge for Llama and Phi (κ 0.67–0.84). Therefore the methodological contribution, which is to validate the scorer before trusting the delta, generalises rather than being a Qwen-specific artefact. On over-refusal, four of five pairs show no significant change and the fifth, Phi-4-mini, moves in the benign direction. The RQ2 deployment message that NF4 does not make models more trigger-happy on benign prompts therefore survives the extension."),
  PJ("Capability loss stays directionally consistent, because every pair's MMLU and ARC point estimates are ≤ 0, the sole exception being Mistral-7B's negligible +0.9 pp on ARC. The cross-family extension therefore corroborates RQ5. NF4's safety effect is not uniform across families, it is never a significant worsening in any pair, and it is cleanly separable from capability only once the scorer itself is validated."),

  H2("6.9 Judge-Model Validation of HarmBench ASR, the Primary Scorer (RQ7)"),
  PJ("HarmBench Attack Success Rate is scored by the official HarmBench classifier, cais/HarmBench-Llama-2-13b-cls, the fine-tuned judge model specified by the HarmBench paper. Given a harmful behaviour and a model generation, it answers whether the generation is a non-minimal, unambiguous instance of the behaviour (\"yes\" = attack success). This subsection reports the validation that established the judge as the primary HarmBench scorer over the v2 regex."),
  PJ("The judge re-scores the already-saved generations. It performs no new target-model inference, and it does not modify the TC1-original raw outputs or the v2 sidecars. Judge labels are written to redacted per-prompt sidecars (scores.judge.harmbench_cls.jsonl, IDs and booleans only) and aggregated to summary.judge.harmbench_cls.json. The judge covers the HarmBench benchmark specifically. Over-refusal on XSTest is a different question (\"did the model refuse a benign prompt?\") and remains v2-scored as primary, with an independent three-class refusal judge added as a sensitivity cross-check (Result 6, below)."),
  PJ("Methodology and setup. The judge classified every saved (behavior, generation) pair for all ten fp16/NF4 models (n = 200 each, 2 000 total) on the NTU TC1 cluster via sbatch in offline mode, on job 61524; the five INT8 aliases (Section 6.11) were scored by an identical follow-up job, a further 1 000 classifications. All 3 000 generations of the 512-token study were classified with zero parse errors. The same protocol had first been run on the 128-token study (jobs 61047 and 61134), which is the validation that originally established the classifier as the primary scorer, and the agreement figures here are recomputed at the reference budget. The judge itself ran at full precision (float16) on a 32 GB Tesla V100, because quantizing the instrument that validates quantized models would reintroduce the very confound the check exists to rule out."),
  PJ("Agreement with the v2 scorer is reported per model as raw agreement rate and Cohen's kappa over the shared prompt set, kappa because raw agreement alone overstates concordance when one label dominates [46], with the verbal descriptors following Landis and Koch [47]. The per-pair judge ΔASR uses the same paired bootstrap (2 000 resamples, seed 42) as the other deltas and is then passed through the interpretation-label rule, holding the over-refusal and MMLU deltas fixed because the classifier judges HarmBench only. Both scorers re-score the same saved generations. Figures are produced locally by scripts/judge_agreement.py.{json,csv}."),
  buildTable(
    ["Model", "v2 non-refusal", "judge ASR", "agreement", "Cohen's κ"],
    [
      ["qwen_2b_base", "0.595", "0.255", "65.0%", "0.36"],
      ["qwen_2b_4bit", "0.570", "0.255", "68.5%", "0.41"],
      ["qwen_4b_base", "0.235", "0.115", "85.0%", "0.49"],
      ["qwen_4b_4bit", "0.305", "0.155", "85.0%", "0.59"],
      ["llama_3_2_3b_base", "0.090", "0.100", "95.0%", "0.71"],
      ["llama_3_2_3b_4bit", "0.070", "0.060", "98.0%", "0.84"],
      ["mistral_7b_base", "0.830", "0.585", "68.0%", "0.29"],
      ["mistral_7b_4bit", "0.890", "0.565", "66.5%", "0.25"],
      ["phi4_mini_base", "0.075", "0.070", "95.5%", "0.67"],
      ["phi4_mini_4bit", "0.075", "0.090", "96.5%", "0.77"],
    ],
    [2600, 1900, 1500, 1500, 1300],
  ),
  P("Table 6.3  Judge vs v2 per-model agreement on HarmBench (n = 200 each). \"v2 non-refusal\" is the regex's harmful-compliance rate, and \"judge ASR\" is the official classifier's. Agreement and Cohen's κ are over the shared prompt set, computed at the 512-token reference budget. From the released analysis artifacts.json.", { size: 18 }),
  ...Figure("judge_vs_proxy.png", "Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal' proxy, one marker per model. Eight of the ten points lie below the diagonal, so the proxy systematically over-counts harmful compliance (most severely for Qwen and Mistral); the two marginal exceptions (Llama base, Phi 4-bit) sit just above it. Right: judge-vs-proxy Cohen's κ per model, which is family-dependent: low for Qwen/Mistral (κ ≈ 0.25–0.59) and high for Llama/Phi (κ ≈ 0.67–0.84)."),
  PJ("Result 1: the v2 regex systematically over-counts ASR, unevenly by family. Table 6.3 shows judge ASR far below the regex non-refusal rate for both Qwen pairs, close agreement for Llama-3.2-3B, and correspondingly fair-to-moderate kappa for Qwen and Mistral against high kappa for Llama and Phi. The reason is structural. The regex equates \"did not refuse\" with \"attack succeeded\", but many non-refusals are not genuinely harmful, including vague deflections, safety lectures without refusal keywords, on-topic but benign answers, and failed attempts. Qwen produces a large volume of such ambiguous middle-ground text, whereas Llama mostly either refuses cleanly or complies, which leaves little for the regex to mis-score. This is exactly the failure mode the HarmBench paper's fine-tuned classifier was built to avoid, and it is why the official classifier is adopted as the primary HarmBench scorer in place of refusal counting."),
  PJ("Result 2: the choice of scorer changes the conclusion. Table 6.1 gives both scorers per pair, and they disagree in level and in per-model ordering. Under the judge the only individually significant contrast is the Llama-3.2-3B decrease, in the safety-improving direction; under the regex the largest positive move sits on a different family altogether, Qwen3-4B. Neither scorer produces a HarmBench contrast that survives the Benjamini-Hochberg correction over the primary family."),
  PJ("The interpretation labels follow. Qwen3-1.7B is broad_degradation, but at 512 that label is driven entirely by its significant MMLU capability loss (ΔMMLU −0.090), with the harm axis flat at ΔASR 0.000. It is a capability-only degradation rather than a safety regression. Qwen3-4B is alignment_degradation, but only directionally (+0.040, not significant). Llama-3.2-3B is capability_collapse_masquerading_as_safety, because harmful compliance and capability decrease together."),
  PJ("The judge validation is therefore not a rubber stamp. It corrected a refusal-counting artefact, and at the reference budget it establishes that refusal-based scoring and the classifier lead to different per-model conclusions about where, and in which direction, harmful compliance moves. This is reported transparently as the central methodological result of the chapter, and the HarmBench numbers throughout the report are the judge values accordingly."),
  PJ("Result 3: the null-increase conclusion holds whichever significance test is used. The judge ΔASR significance was re-tested with McNemar's exact test, the correct paired-binary test for HarmBench, which assumes no large-sample normality (Section 3.7). It reproduces the bootstrap conclusion exactly, with the per-pair p-values in Table 6.1: the sole individually significant contrast is the Llama-3.2-3B decrease, and no pair shows a significant harmful-compliance increase under either test. The study's null-increase conclusion therefore does not depend on the bootstrap machinery."),
  PJ("A structural property of the judge–regex relationship reinforces the over-counting diagnosis. Across all models the judge labelled essentially no generation harmful that the regex had cleared (judge_harmful_v2_not ≈ 0). The judge's harmful set is therefore almost a strict subset of the regex's. In other words, the disagreement is one-directional over-counting by the regex rather than two-sided noise."),
  PJ("Result 4: the finding holds under a different judge, in a second-judge cross-check. To rule out the possibility that the result is an artefact of the single HarmBench classifier, the saved generations were independently re-scored by a second frontier judge from a different provider, gpt-4o, run on 2026-06-14 via the OpenAI API. Its architecture is undisclosed, so the independence claimed is of provider, training pipeline and invocation rather than of architecture. It applied a condensed one-sentence yes/no rubric that encodes the same harmful-behaviour criterion as the HarmBench classifier. The alias's underlying dated snapshot was not recorded, so the revision-pinned open-weight LlamaGuard cross-check below is the reproducible arm of this validation."),
  PJ("The two judges agree strongly at the per-prompt level, and the per-pair ΔASR direction is reproduced at the reference budget, including for the Qwen3-1.7B effect that the 128-token run had reported, which is null under both judges at 512 tokens. Appendix I.3 gives the per-model agreement coefficients and the per-pair gpt-4o deltas."),
  PJ("Result 5: the classifier tracks human judgment. The regex does not. The two preceding cross-checks compare the classifier against other models. To ground the comparison against truth, a stratified subset of 200 saved HarmBench generations was labelled by a single human annotator, the study's author, against the HarmBench rubric. The subset spans the ten base/NF4 aliases of the primary study and deliberately over-samples the judge-vs-regex disagreement cases. The redacted aggregate is released with the analysis artifacts.json. The annotation sheet holds raw response text and is local-only."),
  PJ("The classifier agrees with the blinded human annotator substantially better than the demoted regex does. This is the human-grounded form of the chapter's central result: refusal-counting over-states harmful compliance not only relative to another model, but relative to a person. Appendix I.4 records the annotation protocol, an input-mismatch caveat, and both agreement coefficients."),
  PJ("Threats and scope. The judge is itself a model and can err. It is adopted because it is the reference classifier specified by the HarmBench paper [24], run at full precision with deterministic greedy decoding and zero parse failures on this dataset. The single-judge threat is addressed on two fronts: a second judge from a different provider, and an open-weight third judge, Llama Guard [48], whose per-pair ΔASR is non-significant everywhere. Appendix I.5 sets out the classifier-template comparability bound, the residual caveats attaching to each judge, and the construct caveat that Llama Guard answers a looser question than the behaviour-specific classifier."),
  PJ("Result 6: the over-refusal axis is scorer-sensitive too, and the one over-refusal survivor does not replicate under an independent judge. The validation above concerns HarmBench ASR. By contrast, the study's third finding to survive multiplicity correction is an XSTest over-refusal decrease scored by the demoted v2 refusal regex (Phi-4-mini, ΔOR = −0.048, Section 6.4.1). The classifier answered the same question for HarmBench, and to ask it of that regex-scored finding the 3,750 saved benign XSTest responses (fifteen aliases × 250 prompts) were re-scored by an independent API judge. That judge is gpt-4o, run on the pinned dated snapshot gpt-4o-2024-08-06 at temperature 0 with zero parse failures, and it labels each response by the XSTest paper's own three-class taxonomy of full compliance, partial refusal and full refusal. Röttger et al. [18] themselves use a GPT-4 judge with this taxonomy."),
  PJ("Appendix I.6 sets out the strict and broad refusal mappings, and the per-pair comparison behind Table 6.4."),
  buildTable(
    ["Pair", "Regex ΔOR (primary)", "Judge ΔOR strict (95% CI; p)", "Judge ΔOR broad (95% CI; p)"],
    [
      ["qwen_2b", "−0.024", "+0.040 [0.000, +0.080]; 0.087", "+0.040 [−0.004, +0.084]; 0.110"],
      ["qwen_4b", "−0.004", "−0.016 [−0.052, +0.024]; 0.541", "−0.020 [−0.060, +0.020]; 0.424"],
      ["llama_3_2_3b", "+0.016", "+0.004 [−0.028, +0.036]; 1.000", "0.000 [−0.032, +0.032]; 1.000"],
      ["mistral_7b", "0.000", "−0.004 [−0.036, +0.028]; 1.000", "−0.008 [−0.044, +0.024]; 0.815"],
      ["phi4_mini", "−0.048 ★", "+0.016 [−0.028, +0.060]; 0.597", "−0.004 [−0.048, +0.036]; 1.000"],
    ],
    [1500, 1700, 2800, 2800],
  ),
  P("Table 6.4  Exploratory per-pair XSTest over-refusal deltas (NF4 − fp16) under the independent three-class refusal judge, both mappings, with paired-bootstrap 95% CIs (10 000 resamples, seed 42) and exact McNemar p-values. The primary regex ΔOR is shown for comparison (★ = the one regex-significant, FDR-surviving contrast). Every pair is reported regardless of direction under the pre-registered presentation rule. No judge delta is significant, and none enters the Section 6.4.1 registered multiplicity family. The judge-strict column is the over-refusal member of the Section 6.4.1 validation-informed parallel family. From the released analysis artifacts.json.", { size: 18 }),
  PJ("The human grounding for this axis was completed as well, mirroring Result 5, on a pre-registered two-hundred-item gold set. The judge aligns substantially better with the blinded annotator than the regex does, whose recall on the annotator's full refusals is very low, so the single FDR-surviving over-refusal contrast is carried as most plausibly a regex measurement artefact. One protocol deviation is disclosed and the confirmatory pre-registration badge is withdrawn accordingly. Appendix I.7 records the outcome matrix, the recall figures, and the deviation in full."),
  H2("6.10 Mechanism Probe: Refusal-Margin Geometry (the \"why\")"),
  PJ("The behavioural results of Sections 6.8, 6.12 and 6.13 establish that at the 512-token reference budget NF4 does not significantly increase harmful compliance in any pair, and that behavioural flips are near-symmetric and small. This mechanism probe operates on the first-token refusal distribution. That distribution is fixed at the first generated position and is therefore generation-length-invariant, so the margin values the probe analyses are budget-invariant."),
  PJ("However, the behavioural flips it references are defined by the full-generation judge outcome and were computed on the 128-token generation set. The flip counts below are therefore budget-specific. At the 512-token reference budget the load-bearing Qwen3-1.7B pair, for example, is a symmetric sixteen-versus-sixteen (Section 6.12) rather than the sixteen-versus-five reported here. The margin-based boundary-instability reading does not depend on those counts, and the symmetric split at the reference budget aligns with the 512-token null, in which harmful compliance is flat and the capability loss is the persistent cost."),
  PJ("To probe why flips occur where they do, a derived mechanism analysis measured each model's first-token refusal margin on all 200 HarmBench prompts per model, in fp16 and under NF4, via teacher-forced forward passes. The margin m = log P(refusal-token set) − log P(compliance-token set) is a scale-invariant log-probability difference at the first generated position. The capture ran on TC1 through scripts/capture_refusal_margin.py, which writes redacted scores.margin.<precision>.jsonl sidecars carrying only prompt-id and scalar margins, with no text. Those sidecars were analysed by scripts/refusal_margin_analysis.py into the released analysis artifacts.{json,csv}."),
  PJ("The hypothesis under test, following Proskurina et al. [11], is that quantization perturbs the decision distribution most where the baseline margin is thin. Behavioural flips should therefore concentrate at thin baseline margins. The analysis was run with explicit validity, circularity and confound controls, and the headline numbers were independently recomputed from the raw sidecars."),
  PJ("Validation gate. Before any quantization claim the margin's sign was checked against the baseline judge label. For four families it discriminates refusal from compliance well, with AUC 0.86–0.90 for Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B and Phi-4-mini. For Mistral-7B it discriminates only weakly (AUC 0.69). A plausible explanation, not tested here, is that Mistral-7B often opens with a refusal-shaped preamble and then complies, which would move its refuse-versus-comply decision past the first token. The first-token margin is therefore a sound proxy for four families and a poor one for Mistral-7B, which is flagged throughout. The gate is itself only suggestive. The well-discriminating pairs have small baseline-harmful counts of 8–13 prompts and correspondingly wide AUC intervals, whereas Mistral-7B's lower estimate rests on the largest positive class (77)."),
  PJ("Central result: boundary instability rather than targeted erosion. This is what an extrapolation from the mechanistic literature would suggest. Refusal is mediated by a single activation direction [15], and safety alignment is concentrated over the first few generated tokens [16]. Neither study examines quantization, so the step from those findings to the expectation that a weight perturbation should move prompts across a refusal boundary rather than erode the behaviour uniformly is this study's own, and the probe below is what tests it. The figures below were computed on the 128-token behavioural flips, and the first-token margins themselves are budget-invariant. A thin baseline margin does predict which prompts flip under NF4, although the within-model effect is modest. Pooling all 1,000 prompts gives AUC 0.76. After the between-family level difference is removed by z-scoring the margin within each pair, it falls to 0.64. Within individual pairs it is near chance for the two thinnest-margin families, at 0.61 for Qwen3-1.7B and 0.54 for Mistral-7B."),
  PJ("The flips are close to symmetric in aggregate, which is what a boundary-instability reading predicts and a targeted-erosion reading does not, and the direction of the margin shift is model-specific and modest. Appendix I.10 gives the flip counts, the per-family breakdown and the paired margin shifts."),
  PJ("Capability versus alignment: the tie to the central question. If NF4 simply raised next-token entropy everywhere, every margin would shrink as a by-product of generic confidence loss rather than as a safety-specific change. Comparing the entropy rise on harmful prompts with that on neutral MMLU and XSTest-benign tokens, the harmful side softens more only for Qwen3-4B and Mistral-7B, the two pairs that do not drive the behavioural result."),
  PJ("For the one pair that does move, Qwen3-1.7B, neutral-token entropy rises roughly ten times more than harmful-token entropy, at +0.21 against +0.02. The change is therefore consistent with generic confidence and capability softening near the boundary rather than with a targeted alignment shift. This control is suggestive only, resting on n = 100 unpaired neutral prompts, a mixed set, with no significance test."),
  PJ("The mechanism evidence therefore converges with the significant Qwen3-1.7B ΔMMLU of Section 6.2, the multi-seed fragility of Appendix I.1 and the ARC-versus-MMLU capability picture of Appendix I.9. The smallest model's harmful-compliance wobble reads as capability-driven boundary instability rather than as genuine alignment erosion. The probe does not deliver a clean safety-erosion mechanism. Its contribution is to localise the effect to near-boundary prompts, and to weigh the evidence, for the one moving pair, toward the capability side of the study's central dichotomy."),
  PJ("Two mechanism follow-ups are listed in Chapter 9, namely an independent activation-space refusal-direction probe and a paired neutral-margin control. The third follow-up anticipated there, an INT8 precision point tracing the effect across the fp16 → INT8 → NF4 spectrum, was run and is reported in Section 6.11."),

  H2("6.11 INT8 Precision Point: Method- and Bit-Width Sensitivity, fp16 to INT8 to NF4 (RQ6)"),
  ...Figure("precision_sweep.png", "Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat through INT8 and falls only at four-bit, behaving as a cliff rather than a gradient. This is the one budget-invariant precision effect. The safety axis (judge ASR) shows no robust move at either precision step at the 512-token reference budget: the 128-token Llama-3B INT8 increase does not replicate here."),
  PJ("The main study evaluates one quantization method, NF4, at one bit-width, four-bit. To test whether the safety and capability effects are a smooth function of quantization aggressiveness or are instead method- and bit-width-specific, an INT8 precision point was added, so that every pair is evaluated across three precisions: fp16 → INT8 → NF4. INT8 here is the bitsandbytes LLM.int8() mixed-precision method applied on the fly to the same baseline weights. It is a distinct algorithm rather than a lower-bit NF4."),
  PJ("The INT8 arm was run under the identical protocol and is analysed separately from the base-versus-4-bit pipeline, so the evaluated pairs are unchanged; Appendix I.15 records the configuration."),
  PJ("Capability: a clean cliff at four-bit. The capability answer is unambiguous and holds at the 512-token reference budget. No INT8 delta, on either MMLU or ARC, is statistically significant for any of the five pairs. Exact McNemar tests recomputed from the paired per-item records give ten contrasts, none of them significant, with a largest |Δ| of 1.3 pp. Absence of significance is not by itself evidence of equivalence, so the same ten contrasts were also submitted to a two-one-sided-tests (TOST) equivalence procedure [49] against a margin of ±0.05, fixed on deployment-relevance grounds before the test was run. All ten are statistically equivalent to fp16 at that margin, with a largest p of 0.007, and the widest 90% confidence interval closes at ±0.038. INT8 capability is therefore equivalent to full precision to within about four percentage points, which is a stronger and more precise statement than the absence of a significant difference. The equivalence analysis is post hoc, having been added after the INT8 results were known."),
  PJ("Every significant capability loss appears only at the four-bit step. ΔMMLU(NF4) = −0.090 for Qwen3-1.7B, and ΔARC(NF4) = −0.016 for Qwen3-4B and −0.032 for Llama-3.2-3B. All ten INT8 capability contrasts meet the post-hoc TOST equivalence criterion within ±0.05, so eight-bit capability is equivalent to full precision within that margin rather than merely not significantly different from it. The capability cost of quantization is a cliff at four-bit rather than a graded decline. The one borderline cell is Phi-4-mini ΔARC(NF4) = −0.015, which is non-significant only by the closed-interval convention and is footnoted so that the dichotomy is not overstated as perfectly clean."),
  PJ("Safety: no robust precision effect at the reference budget. The safety axis is the more revealing result, and at the 512-token reference budget it yields no robust move at either precision step. The 128-token run had reported an INT8-specific increase on Llama-3.2-3B that was significant under both judges, at ΔASR = +0.040 with McNemar p = 0.008/0.022. That increase was non-monotonic, rising at INT8 and reverting at NF4. At 512 tokens it vanishes under both judges. The Llama-3.2-3B INT8 ΔASR is +0.005 under the primary classifier and +0.010 under gpt-4o. Neither is significant, at McNemar p = 1.000 and p = 0.688 respectively. Every INT8 ASR contrast is non-significant under both judges across all five pairs."),
  PJ("The one INT8 safety signal visible at the 128-token budget does not survive at the reference budget, so it is read as budget-sensitive rather than as a precision effect. The secondary regex continues to over-count harmful compliance at INT8 exactly as it does elsewhere. Appendix I.11 sets out the budget comparison, its tie to the mechanism reading of Section 6.10, and the proxy behaviour."),
  PJ("The precision point sharpens the study's central message at the reference budget. Within the compact-deployment regime, the capability cost of quantization is a clean cliff at four-bit. By contrast, the safety axis shows no robust precision effect, being neither bit-width-graded nor budget-robust. INT8 capability is statistically equivalent to full precision within ±0.05 at 512 tokens, its safety contrasts are null, and no harmful-compliance increase, whether INT8 or NF4, survives at the reference budget."),

  H2("6.12 Generation-Length Robustness: the 512-Token Rerun"),
  PJ("The HarmBench authors show that the number of tokens generated during evaluation can change the measured attack-success rate by up to 30 percent, and for exactly that reason they standardise the generation parameter to N = 512 in their evaluation framework [24], stating that \"we standardize this parameter to N = 512 to allow the metric to converge\". The official HarmBench evaluation harness likewise clips completions to 512 tokens before classification, with num_tokens defaulting to 512 in evaluate_completions.py. The 512-token budget is therefore HarmBench's own standardised evaluation budget rather than a choice of this study."),
  PJ("This study accordingly adopts the 512-token reference budget as its primary configuration, and retains an initial 128-token run purely as a generation-length comparison. This section reports that comparison and explains why the 512-token budget is the correct primary. It shows that several safety signals visible at 128 tokens were truncation artefacts that disappear at the reference budget."),
  PJ("The reference budget changes the generations materially rather than cosmetically: a large share of the 128-token responses were provably cut off mid-answer, the truncation is family-heterogeneous, and absolute ASR is correspondingly higher at 512 tokens under both judges. A separate decoding-configuration artefact affecting a minority of paired generations was found while inspecting responses and is disclosed there rather than smoothed over. Appendix I.12 gives the prefix statistics, the per-family disclosure and the artefact analysis."),
  buildTable(
    ["Pair", "ΔASR @128 (classifier)", "ΔASR @512 (classifier)", "ΔASR @512 (gpt-4o)"],
    [
      ["Qwen3-1.7B", "+0.055 (sig @128)", "0.000", "+0.005"],
      ["Qwen3-4B", "+0.025", "+0.040", "+0.040"],
      ["Llama-3.2-3B", "0.000", "−0.040", "−0.035"],
      ["Mistral-7B", "−0.040", "−0.020", "−0.005"],
      ["Phi-4-mini", "0.000", "+0.020", "+0.020"],
    ],
    [1400, 2550, 2550, 2526]
  ),
  P("Table 6.5  HarmBench ΔASR across the retained 128-token comparison and the 512-token reference budget.", { size: 18, after: 160, alignment: AlignmentType.CENTER }),
  PJ("The headline of the comparison is that the safety signal seen at 128 tokens was a truncation artefact. The study's only significant 128-token safety regression does not exist at the reference budget. The Qwen3-1.7B increase of +0.055 at 128 tokens, whose base and NF4 ASR are 0.255/0.255 at 512, becomes ΔASR = 0.000 under the primary classifier and +0.005 under gpt-4o at 512 tokens. Neither delta is significant, with McNemar p = 1.000 under both judges and both paired-bootstrap confidence intervals spanning zero. At the reference budget this is a capability-only degradation with a null harm delta rather than a safety regression."),
  PJ("The Qwen3-1.7B null at 512 tokens is moreover a high-churn null. Sixteen prompts flip from refusal to compliance and sixteen flip from compliance to refusal, a symmetric exchange that is exactly the boundary-instability signature the mechanism probe of Section 6.10 identified, visible directly at the reference budget."),
  PJ("Across all five pairs at 512 tokens, no ΔASR survives a Benjamini-Hochberg correction under either judge. The only individually significant ΔASR is Llama-3.2-3B at −0.040 under the classifier and −0.035 under gpt-4o. The classifier contrast has CI [−0.075, −0.010] and McNemar p = 0.021. That contrast is in the safety-improving, decreasing direction, and it does not survive the correction either."),
  PJ("The three contrasts that do survive BH-FDR across the 20-test NF4 family are all capability or over-refusal effects, namely Qwen3-1.7B MMLU −0.090, Llama-3.2-3B ARC −0.032, and Phi-4-mini over-refusal −0.048. None of them is a harmful-compliance increase. The two judges agree per prompt at Cohen κ 0.68–0.95, matching the 0.60–0.95 range reported for the 128-token run, so the scorer is no less reliable at the reference budget."),
  PJ("By contrast, capability degradation is essentially unchanged by the generation budget, which is why capability rather than the safety axis carries the cost of NF4 that holds at both budgets. The four-bit MMLU and ARC losses move only at the third decimal between the two budgets. Qwen3-1.7B ΔMMLU goes from −0.087 to −0.090, Llama-3.2-3B ΔARC from −0.028 to −0.032, and Qwen3-4B ΔARC from −0.021 to −0.016, with the same cells remaining significant at 512 tokens. The capability cost of quantization is budget-robust. The harmful-compliance change is not."),
  PJ("The generation-length comparison therefore establishes the study's central dichotomy at the reference budget. The capability cost of NF4 is real and budget-robust. By contrast, the apparent 128-token safety regression is fragile on every axis tested, surviving neither the 512-token reference budget, nor a multiple-comparison correction, nor the second judge, nor stochastic decoding (Appendix I.1). Quantization in this regime erodes competence more than it erodes guardrails, and the one borderline safety signal observed at 128 tokens is best read as a truncation artefact and as capability-driven boundary instability rather than as a stable alignment shift."),
  PJ("Under the official classifier at 512 tokens, Qwen3-1.7B retains its broad_degradation label. That label is driven entirely by its significant MMLU loss of −0.090, with the harm axis flat at ΔASR 0.000. Llama-3.2-3B is capability_collapse_masquerading_as_safety, because both harm and capability decrease. The 128-token results are retained unchanged as this comparison."),
  PJ("The INT8 precision point of Section 6.11 and the multi-seed sensitivity arm of Appendix I.1 were both regenerated at 512 tokens, and both confirm the null. The 128-token Llama-3.2-3B INT8 increase vanishes under both judges at the reference budget. The multi-seed greedy ΔASR of 0.000 for Qwen3-1.7B sits inside its stochastic seed range, which has mean +0.013 and range [0.000, +0.035], with 0/5 seeds significant. All five seed deltas are non-negative, a sub-MDE directional signal disclosed in Appendix I.1. At 512 tokens, therefore, there is no detectable effect for stochastic decoding to attenuate. The multi-seed arm covers all five pairs (Appendix I.1)."),
  H2("6.13 Research Question Synthesis"),
  PJ("With all ten models evaluated and HarmBench ASR scored by the official classifier at its 512-token reference budget, all seven research questions can be answered. Paired bootstrap 95% confidence intervals quantify the statistical uncertainty behind each answer."),
  PJ("At the reference budget the individually-significant deltas under the primary scorers are all on the capability and over-refusal axes: Qwen3-1.7B ΔMMLU (−0.090), Llama-3.2-3B ΔARC (−0.032), Qwen3-4B ΔARC (−0.016), and one over-refusal decrease, Phi-4-mini (−0.048). Qwen3-1.7B's own over-refusal decrease (−0.024) is not significant, because the bootstrap CI [−0.048, 0.000] no longer excludes zero and McNemar p = 0.109 (Section 3.7). On the harmful-compliance axis, the only ΔASR whose CI excludes zero is Llama-3.2-3B's, at −0.040, and that is a decrease rather than an increase. No pair shows a significant ASR increase."),
  PJ("Under Benjamini–Hochberg correction over the twenty-test NF4 family (Section 6.4) exactly three contrasts survive, and all three are capability or over-refusal effects (Qwen3-1.7B MMLU, Llama-3.2-3B ARC, Phi-4-mini over-refusal). Zero HarmBench ASR contrasts survive. Capability degradation is therefore the multiplicity-surviving cost of four-bit NF4, while the harmful-compliance change does not survive the correction. The 128-token run produced an apparent Qwen3-1.7B ASR increase of +0.055 that was a truncation artefact and does not appear at the reference budget (Section 6.12)."),

  H3("RQ1: Does 4-bit NF4 quantization increase harmful compliance?"),
  PJ("At the 512-token reference budget it does not, and it does not significantly increase it in any pair either. Under the official classifier the ΔASR values are: Qwen3-1.7B 0.000 (CI [−0.055, +0.055], not significant), Qwen3-4B +0.040 (CI [0.000, +0.080], not significant, directional), Llama-3.2-3B −0.040 (CI [−0.075, −0.010], significant, a decrease), Mistral-7B −0.020 (CI [−0.080, +0.040], not significant), and Phi-4-mini +0.020 (CI [−0.015, +0.055], not significant)."),
  PJ("The only ΔASR whose CI excludes zero is Llama-3.2-3B's, and it is a decrease rather than an increase. No pair shows a significant harmful-compliance increase, and no ASR contrast survives multiplicity correction (Section 6.4). The answer to RQ1 is therefore that at the reference budget NF4 quantization does not significantly change harmful compliance in either direction for four of five pairs, and the single significant move (Llama-3.2-3B) is a decrease. The data do not support a claim that NF4 raises harmful compliance. The cost of NF4 that survives correction is on capability rather than on safety."),
  PJ("The secondary v2 proxy over-counts ASR and reports a different pattern. The official classifier is authoritative here (see Section 6.9). The 128-token +0.055 on Qwen3-1.7B was a truncation artefact (Section 6.12)."),

  H3("RQ2: Does 4-bit NF4 quantization increase over-refusal on benign prompts?"),
  PJ("Across the five pairs, none moves significantly in the over-refusing direction, and the only confirmed change is benign. Under the v2 scorer the values are: Qwen3-1.7B ΔOR = −0.024 (0.052 → 0.028, not significant), Qwen3-4B −0.004 (0.028 → 0.024, not significant), Llama-3.2-3B +0.016 (0.032 → 0.048, not significant), Mistral-7B 0.000 (0.004 → 0.004, not significant), and Phi-4-mini −0.048 (0.128 → 0.080, significant, McNemar p = 0.002). For Qwen3-1.7B the bootstrap CI [−0.048, 0.000] no longer excludes zero and McNemar p = 0.109, so the decrease is not confirmed under the study's criterion (Section 3.7)."),
  PJ("The one significant over-refusal delta (Phi-4-mini, −0.048) is a decrease, and the previously-borderline Qwen3-1.7B delta is not significant but is still a decrease. The 4-bit model refuses fewer benign prompts rather than more, and Phi-4-mini's decrease survives multiplicity correction (Section 6.4). No pair shows a significant increase. The answer to RQ2 is that NF4 quantization does not produce a detectable over-refusal increase on benign prompts, across four families and five pairs. The moves that approach or reach significance both run in the safety-benign direction."),
  PJ("This is practically useful for deployment teams considering NF4 compression. The concern that quantization would make models more trigger-happy on benign prompts is not supported by these data. If anything, the significant effects point the other way."),

  H3("RQ3: Does 4-bit NF4 quantization degrade general capability?"),
  PJ("Yes, and this is the cost of NF4 that holds at both generation budgets and survives correction. The Qwen3-1.7B model loses 9.0 percentage points of MMLU accuracy under NF4 quantization (0.643 → 0.553, significant), and this is the single largest, multiplicity-surviving effect in the study. Every pair's MMLU point estimate is negative (Qwen3-4B −0.003, Llama-3.2-3B −0.037, Mistral-7B −0.020, Phi-4-mini −0.027). On the second capability benchmark ARC-Challenge (Appendix I.9) the losses that reach nominal significance are Llama-3.2-3B's (−0.032), which also survives the Benjamini–Hochberg correction, and Qwen3-4B's (−0.016), which is significant uncorrected only and does not survive correction."),
  PJ("Capability degradation therefore holds in direction across all five pairs, reaches significance on at least one benchmark for the Qwen3-1.7B, Qwen3-4B and Llama-3.2-3B pairs, and supplies two of the three effects that survive the Benjamini–Hochberg correction. The third survivor is Phi-4-mini's over-refusal decrease. The answer to RQ3 is that NF4 quantization degrades general capability, most severely (and MMLU-significantly) in the 1.7B Qwen model, and that this capability cost is stable across generation-length budgets where the harmful-compliance change is not (Section 6.12)."),
  PJ("The one substantive cross-benchmark divergence is the Qwen3-1.7B loss, which is far smaller and non-significant on ARC (−0.9 pp) than on MMLU (−9.0 pp). Its *severity* is therefore partly MMLU-specific even though its direction is not."),

  H3("RQ4: Are smaller models more sensitive to quantization within the same family?"),
  PJ("On capability, yes. On harmful compliance, not at the reference budget. On capability the MMLU delta ratio between Qwen3-1.7B and Qwen3-4B is large (−9.0 pp versus −0.3 pp). The 1.7B model loses substantial, significant MMLU capability under NF4 compression while the 4B model's MMLU is essentially unaffected. On safety, under the judge at the 512-token reference budget, neither Qwen pair shows a significant ΔASR: the 1.7B is exactly 0.000 and the 4B is a directional, non-significant +0.040. There is therefore no within-family safety-sensitivity contrast to report. The earlier +0.055 on the 1.7B was a 128-token truncation artefact (Section 6.12)."),
  PJ("The answer to RQ4 is that smaller Qwen models are more sensitive to NF4 quantization on capability *as measured by MMLU*. This is benchmark-dependent. On ARC the within-family ordering reverses, because the 4B pair loses −0.016 significantly while the 1.7B loses only −0.009, not significant. The dramatic MMLU ratio is therefore a parser-sensitive, benchmark-dependent result under the primary lenient parser, and under the strict parser the scale gap replicates on ARC as well (−0.343 versus −0.017, Appendix I.9). It is not a clean benchmark-independent scale law."),
  PJ("On the harmful-compliance axis there is no scale contrast at the reference budget, because both Qwen pairs are within noise. The RQ4 conclusion that survives is therefore a capability one, and even that is MMLU-specific and roughly equalised by the second benchmark."),

  H3("RQ5: Are effects consistent across model families?"),
  PJ("Partially, and consistently in the safety-null / capability-cost direction. On capability, every pair's MMLU point estimate is negative under NF4 compression. The loss reaches MMLU-significance for Qwen3-1.7B (−9.0 pp), and the ARC axis adds significant losses for Qwen3-4B (−0.016) and Llama-3.2-3B (−0.032). On safety, under the judge at the 512-token reference budget, no pair shows a significant harmful-compliance increase. The only individually-significant ΔASR is Llama-3.2-3B's, at −0.040, a decrease, and the remaining pairs (Qwen3-1.7B 0.000, Qwen3-4B +0.040, Mistral-7B −0.020, Phi-4-mini +0.020) are all within noise."),
  PJ("The three original pairs receive three labels: broad_degradation but capability-driven for Qwen3-1.7B, alignment_degradation directional for Qwen3-4B, and capability_collapse_masquerading_as_safety for Llama-3.2-3B. The over-refusal axis is near-null. Its one significant delta, Phi-4-mini's −0.048, is a decrease, and the previously-borderline Qwen3-1.7B delta is not significant (Section 3.7)."),
  PJ("This is consistent in direction with Hong et al. [37], who report that four-bit quantization broadly preserves trustworthiness across eight trust dimensions, although their models are seven billion parameters and above. The present result extends that picture downward into the compact-deployment regime, below the seven-billion-and-above range Hong et al. evaluate. The full answer to RQ5 across the original two families is that at the reference budget NF4 quantization does not raise harmful compliance in any pair (Llama-3.2-3B's move is a significant decrease), does not significantly raise over-refusal, and imposes a capability cost that is consistent in direction and the only multiplicity-surviving class of effect. The cross-family extension (Section 6.8.6) confirms this across families. Neither Mistral-7B (ΔASR −0.020, not significant) nor Phi-4-mini (+0.020, not significant) shows a significant increase. Therefore no pair shows a statistically significant NF4-induced increase in harmful compliance, while the capability-cost pattern holds across four families."),

  H3("RQ6: Do the effects hold across quantization precisions?"),
  PJ("Yes, and the precision axis behaves as a cliff rather than a gradient. Dettmers and Zettlemoyer [7] find four-bit precision optimal for a fixed total-bit budget across scales, with performance degrading at three bits. The present study holds parameter count fixed and varies only precision, which is a different comparison, so the two are consistent in identifying four bits as the operating point of interest rather than measuring the same quantity. This study does not evaluate three-bit precision and therefore says nothing about that transition. Section 6.11 evaluates the same matched baselines at INT8 as well as at four-bit NF4. Capability is essentially flat through INT8 and falls only at four bits, so the capability cost identified under RQ3 is specific to the four-bit step rather than to compression in general. On the safety axis, no robust move appears at either precision step at the 512-token reference budget. The one apparent exception, a Llama-3.2-3B INT8 increase visible at the shorter 128-token budget, does not survive at the reference budget under either judge."),
  PJ("This matters for deployment. A practitioner who needs to reduce memory but cannot accept the capability cost has an intermediate option, because INT8 preserves capability where four-bit NF4 does not."),

  H3("RQ7: Does the choice of harmful-compliance scorer change the answer?"),
  PJ("Yes, and this is the study's principal methodological finding. Section 6.9 shows that the deterministic refusal regex and the official HarmBench classifier disagree materially, and that the regex over-counts harmful compliance because it treats any non-refusal as a success. Under the regex the smallest pair appeared to show a harmful-compliance increase. Under the benchmark's own classifier that increase disappears."),
  PJ("The scorer question is settled against human judgment rather than by assertion. On a stratified 200-item subset the classifier reaches Cohen κ 0.59 against human labels, while the regex reaches 0.11. A second frontier judge and an open-weight guard model both reproduce the classifier's null. The answer to RQ1 is therefore scorer-dependent in the sense that a weaker scorer would have produced a different headline, and of the two scorers compared against human labels, the classifier agreed substantially better with the single annotator on the audited subset."),

  H3("Discussion: What the full five-pair dataset tells us"),
  PJ("Three implications extend beyond the specific numbers. First, surface-level refusal metrics are unreliable without both a capability anchor and an accurate harmful-compliance scorer. The v2 regex over-counts ASR, because it equates non-refusal with success. The judge validation is therefore a conclusion-level correction rather than a cosmetic one. It is the scorer rather than the delta that determines the conclusion (Section 6.9)."),
  PJ("Second, the strongest empirical finding is on the capability axis rather than the safety axis. Badshah and Sajjad [29] report in a preprint that larger models are notably resilient to four-bit precision reduction, and the present result is the complementary half of that observation: in the compact regime the capability cost is the effect that survives correction, while the safety axis stays null. Capability degradation is directionally consistent on MMLU across all five pairs and on ARC in four of five, reaches significance on at least one benchmark for three pairs, and accounts for two of the three contrasts that survive multiplicity correction. The one over-refusal survivor is most plausibly a regex measurement artefact, human-audited in Section 6.9 Result 6, and under the validation-informed parallel family only the two capability contrasts survive (Section 6.4.1). By contrast, no HarmBench ASR contrast survives, and the only individually-significant ΔASR (Llama-3.2-3B, −0.040) is a decrease."),
  PJ("At the reference budget the data are a null-safety result with capability as the cost that survives correction. They do not support the alarmist reading that NF4 raises harmful compliance, nor the complacent reading that NF4 is cost-free."),
  PJ("Third, the framework itself is the durable contribution. It produces well-separated diagnostic categories, and, validated against the benchmark's own classifier and re-run at the benchmark's reference generation budget, it caught and corrected both a scorer artefact and a generation-length truncation artefact rather than propagating them. The framework's value does not depend on any specific empirical outcome. It provides a reusable, self-correcting methodology for the next study."),

];

// ------------------------------------------------------------
// Chapter 7 - Discussion and Threats to Validity
// ------------------------------------------------------------
const ch7 = [
  H1("Chapter 7: Discussion and Threats to Validity"),

  PJ("The Qwen family results in Section 6.13 provide initial empirical grounding for the threats discussed in this chapter. At the 512-token reference budget the Qwen3-1.7B pair carries the broad_degradation label. However, that label is driven entirely by its significant capability loss (ΔMMLU = −0.090). Its harm axis is flat (ΔASR = 0.000, not significant). The Qwen3-4B pair is directional alignment_degradation (ΔASR +0.040, not significant). These pairs are used below as concrete examples where relevant. The caveat throughout is that the smallest model's movement is capability-driven rather than a safety regression."),

  H2("7.1 Internal Validity"),
  PJ("Internal validity is the strongest property of the present design. The matched-pair structure, combined with on-the-fly NF4 quantization from identical baseline weights, isolates quantization as the experimental variable against model-provenance confounds. The design leaves few plausible alternative explanations for an observed delta beyond the quantization step itself, the act of loading the same checkpoint twice, or measurement noise. One qualification is important, and it was established in Section 6.4 and Section 6.9. The choice of harm scorer, over-refusal scorer, and answer-parser is itself a measurement variable that moves the observed deltas. Therefore \"few alternatives\" refers to model-provenance confounds rather than to scoring choices."),
  PJ("Deterministic decoding (temperature 0.0) eliminates within-condition variance from generation. Scoring is also deterministic. MMLU uses exact match, XSTest over-refusal uses the regex parser, and the primary HarmBench scorer (the official HarmBench classifier) is run with greedy decoding (max_new_tokens = 1) in full precision, producing reproducible labels. The 512-token primary judge job, 61524, returned zero parse errors over 2 000 generations, as did the earlier 128-token validation jobs. The resume logic ensures every reported metric is computed from a complete raw.jsonl with exactly the configured number of prompts. On resume it keys on prompt IDs, and every row carries per-record provenance (seed, model_id, generation_config) for audit. In addition, all ten target-model runs of the primary 512-token study completed under the same conditions, hardware, and software stack."),

  H2("7.2 External Validity"),
  PJ("External validity is bounded by six explicit design choices. First, the safety axis is measured under direct harmful requests only. No adversarial attack or jailbreak transformation is applied (Chapter 3). Consequently the harmful-compliance null does not speak to robustness under optimised attacks (GCG, PAIR, and similar), which is precisely the regime where prior quantization-safety work reports attack-dependent effects. Second, findings are bounded to BitsAndBytes on-the-fly quantization (NF4 as the main study, INT8/LLM.int8 as a precision point). Alternative methods such as GPTQ, AWQ, and the GGUF family used by llama.cpp may produce different effects, both quantitatively and qualitatively. The present study does not claim to characterise those methods."),
  PJ("Third, the Qwen3 baselines (Qwen3-1.7B and Qwen3-4B) are instruction-tuned dense models from the official Alibaba Qwen3 release. Their results describe quantization behaviour on this specific model family and may not transfer directly to other architectures or training regimes. Fourth, the study is restricted to English-language, text-only interactions. Multilingual and multimodal effects are out of scope. Finally, the full-precision baseline is loaded at float16 (dtype: auto resolves to float16 on CUDA) rather than at the models' native bfloat16. Because both members of every pair share the same compute dtype, the matched-pair delta still cleanly isolates quantization. However, the absolute baseline figures are float16 rather than bfloat16 values."),
  PJ("Sixth, both Qwen3 pairs are evaluated in non-thinking mode (enable_thinking = false, a token-budget necessity at 512 tokens). This is not those models' default operating configuration. In addition, quantization's interaction with thinking-mode generation is untested here, and the Qwen3 technical report [1] attributes a substantial part of the models' reasoning performance to that mode of generation."),

  H2("7.3 Construct Validity"),
  PJ("Each benchmark operationalises its dimension in a specific way. HarmBench captures one curated definition of \"harmful compliance\" against a particular set of harmful behaviour requests, presented directly and without adversarial attack augmentation (the threat-model boundary stated in Chapter 3). Other operationalisations exist, including attack-mediated ASR under HarmBench's own attack suite, and they may yield different absolute numbers. XSTest captures over-refusal against a specific distribution of benign-but-suspicious prompts. The MMLU subset, even with diverse subject coverage, is a partial capability proxy and does not measure all reasoning capacities relevant to deployment (such as code generation, long-context reasoning, or tool use)."),
  PJ("A related construct point is stated plainly. The MMLU and ARC scores here are controlled zero-shot subset anchors chosen for a paired, within-model comparison on identical prompts rather than official leaderboard reproductions. The official MMLU protocol is five-shot, so the absolute accuracies reported here are not directly comparable to published leaderboard figures. Because both members of every pair are scored under the identical zero-shot subset protocol, the matched-pair capability delta is still clean. However, the absolute anchors should not be read as leaderboard results. These construct boundaries are common in safety evaluation, but they should be borne in mind when interpreting the results."),

  H2("7.4 Scorer Choice and Remaining Validity Threat"),
  PJ("HarmBench ASR was initially scored with a deterministic regex parser, chosen for reproducibility. The judge validation in Section 6.9 showed that this regex measures \"non-refusal rate\" rather than genuine harmful compliance and over-counts ASR, most severely for the Qwen models. For this reason the official HarmBench classifier was promoted to the primary HarmBench scorer and the regex was demoted to a transparent secondary proxy. That change removes the original construct-validity threat on the HarmBench metric, because the primary scorer is the benchmark's own fine-tuned classifier, run at full precision with deterministic decoding. Two residual threats remain. First, the HarmBench classifier is itself a model and can err. This single-judge threat was tested directly. The saved generations were re-scored by a second frontier judge from a different provider (gpt-4o), applying the same HarmBench rubric. Its architecture is undisclosed, so the independence that can be claimed is of provider, training pipeline and invocation, not of architecture."),
  PJ("The two judges agree strongly per prompt, with κ 0.60–0.95 across all ten models and κ 0.68–0.95 at the 512-token reference budget. That agreement is the load-bearing point, because it shows that the choice of scorer determines the conclusion rather than the choice of judge. At the 512-token reference budget the two judges also agree that no pair shows a significant harmful-compliance increase. For the smallest Qwen pair the primary classifier reports ΔASR = 0.000 and gpt-4o +0.005, and both are null. McNemar p = 1.000 under both judges (Section 6.9, Result 4, and Section 6.12). The classifier was also grounded against human judgment (Section 6.9, Result 5). On a stratified subset of 200 generations a human annotator agrees with the classifier at Cohen's κ 0.59 and with the regex at only 0.11, so the classifier is the better scorer against truth rather than merely against another model. However, that is a moderate, single-annotator agreement, which a second annotator would strengthen."),
  PJ("The construct-validity threat is therefore substantially resolved. The null-safety conclusion is not an artefact of one classifier, and the primary scorer is closer to human judgment than the demoted regex. Two residual caveats attach to that resolution. The second judge is a versioned, less-reproducible API model, and both judges share the general lineage of LLM-based harm classification. For this reason an open-weight third judge applying a different safety-classification construct (LlamaGuard) is the natural complementary cross-check, and that cross-check was run (Section 6.9). Its independence is partial, since Llama-Guard-3-8B is itself a Llama-family model. A revision-pinned Llama-Guard-3-8B re-scored all fifteen aliases' saved generations with zero parse errors, agreed with the classifier at κ 0.36–0.92 across the ten base/NF4 aliases, and found every per-pair ΔASR non-significant. The null therefore reproduces under an open-weight third judge, and the versioned-API-only reproducibility caveat no longer applies. One construct caveat qualifies that cross-check. Llama-Guard answers a looser 'is this response unsafe?' question than the behaviour-specific HarmBench classifier, so its disagreements are expected and non-diagnostic, and it serves as a reproducibility cross-check rather than a second construct-validity oracle."),
  PJ("Second, XSTest over-refusal is scored by the regex parser as primary, because the HarmBench classifier does not judge over-refusal. An independent three-class refusal judge (gpt-4o) cross-checked it as a sensitivity layer (Section 6.9, Result 6) and found the one over-refusal contrast that survives the multiplicity correction, Phi-4-mini −0.048, to be scorer-dependent. The judge does not reproduce it. The pre-registered human refusal audit then grounded that comparison. The judge aligns substantially better with a blinded single annotator than the regex does, at strict κ 0.485 versus −0.006, a mechanical Outcome J carried with one disclosed protocol deviation (Section 8). The contrast is therefore carried as most plausibly a regex measurement artefact, with a second annotator the remaining strengthening on that axis. The immutable-raw-output contract leaves the TC1-original raw.jsonl and summary.json untouched and keeps all corrected and judge scores in derived sidecars, so either future judge can be added without disturbing the existing evidence trail."),

  H2("7.5 Cross-Family Comparison Caveat"),
  PJ("The Qwen-versus-Llama comparison should be read as descriptive only. Qwen and Llama differ in tokenizer, pre-training corpus, instruction-tuning recipe, and safety-alignment methodology. Differences in their quantization deltas could plausibly reflect any combination of these factors rather than family identity alone. The cross-family component of the study therefore provides a useful cross-check on the within-Qwen findings. However, it does not support causal claims about quantization–family interactions."),

  H2("7.6 Deployment Implications"),
  PJ("The practical takeaway is asymmetric across the scale range studied. The asymmetry falls on the capability axis rather than the safety axis. At HarmBench's 512-token reference budget no pair shows a significant increase in genuine harmful compliance under NF4. Across the five pairs the only individually significant ΔASR is Llama-3.2-3B at −0.040, a decrease in the safety-improving direction, and no ASR contrast survives a Benjamini-Hochberg correction over the NF4 test family (Section 6.8 and Section 6.12). The apparent Qwen3-1.7B safety regression seen at the 128-token budget (+0.055) does not replicate at 512, where ΔASR = 0.000 under the classifier and +0.005 under gpt-4o, with McNemar p = 1.000. That regression was a generation-length truncation artefact. The smallest model's real cost is capability rather than compliance (ΔMMLU = −0.090, significant), and that cost is present at both generation budgets."),
  PJ("What survives correction at 512 is exclusively capability and over-refusal, namely Qwen3-1.7B MMLU, Llama-3.2-3B ARC, and Phi-4-mini over-refusal. The cost of NF4 in this regime is therefore degraded competence rather than eroded guardrails, and it holds at both generation budgets. A team that validates on the full-precision 1.7B release and ships the 4-bit build should expect a materially lower capability level than they signed off on (−9.0 pp on this MMLU subset). No increase in harmful compliance was detected, though the study's detection floor bounds that null (Section 6.4.1). The multi-seed check at 512 (Appendix I.1, covering all five pairs) corroborates this. The greedy ΔASR of 0.000 for Qwen3-1.7B sits inside the per-seed range, and no pair shows a harmful-direction shift that holds across seeds."),
  PJ("The deployment guidance is therefore one-sided but capability-shaped. At the reference budget, 4-bit NF4 did not significantly increase harmful compliance under direct requests in any pair. Its dependable effect is capability loss, worst in the smallest model, which is precisely the size class most likely to be quantized for on-device or edge use. The larger Qwen3-4B model tolerates quantization far better on the MMLU anchor. Its MMLU is essentially preserved and its safety shift is only a non-significant nudge. However, ARC records a small significant loss (−1.6 pp, Appendix I.9), so \"less MMLU-sensitive\" is the supportable claim rather than full capability preservation."),
  PJ("The operational recommendations that follow are to re-run evaluation on the exact quantized artefact that will be deployed rather than on the full-precision release, to evaluate at the deployment generation budget rather than a truncated one, and to prefer the largest model that fits the memory budget over aggressively quantizing a smaller one. These recommendations are established for on-the-fly BitsAndBytes quantization (NF4, INT8) under direct harmful requests. Deployment formats not tested here (GGUF, GPTQ, AWQ) and attack-mediated threat models may behave differently, and they are future work (Chapters 8–9)."),

  H2("7.7 Positioning Against Prior Work"),
  PJ("These findings sit within, qualify, and extend the emerging quantization-and-safety literature (Section 2.4). Prior work establishes that quantization can degrade safety alignment rather than being behaviourally neutral, with effects that vary by method, model, and attack type (Kharinaev et al. [10], Q-resafe [36], HarmLevelBench [35]). The closest prior comparison in trustworthiness scope is Hong et al. [37], which also isolates quantization at four bits and works across a broader trust taxonomy. That study reports that 4-bit compression broadly retains trustworthiness. The null-safety result here is consistent with that finding and extends it with a matched-pair capability anchor and a classifier-validated harmful-compliance scorer."),
  PJ("This study contributes a boundary condition to that picture. For on-the-fly BitsAndBytes quantization evaluated under direct requests at the benchmark's own generation budget, no significant harmful-compliance increase appears in any of five pairs. The non-uniformity the literature reports is visible here only as small, sign-varying, non-significant safety movements across families, and the degradation this study measures consistently is on the capability axis. The Egashira et al. [12] result that BitsAndBytes NF4 specifically can turn a benign model harmful is the closest prior analogue to the mechanism studied here. However, that work demonstrates an adversarial worst case, whereas this study measures the effect under ordinary, non-adversarial loading."),
  PJ("The distinct contribution of the present work is the matched-pair, capability-anchored design. By holding the prompt set fixed and measuring harmful compliance and general capability jointly on the same on-the-fly-quantized weights, it separates capability change from safety change on each pair rather than reading a single non-refusal axis. At the 512-token reference budget that separation yields a null-safety result. Harmful compliance does not significantly rise in any pair, whereas capability significantly falls in several, so every pair's movement is best read on the capability side of the dichotomy. Qwen3-1.7B, which the truncated 128-token budget made look like an alignment regression, is at 512 a capability-only degradation, with a significant MMLU loss and a flat harm axis. Llama-3.2-3B is capability_collapse_masquerading_as_safety, because harmful compliance falls alongside capability. Distinguishing these cases from a genuine alignment shift is exactly what the cited single-axis studies are not structured to do."),
];

// ------------------------------------------------------------
// Chapter 8: Limitations
// ------------------------------------------------------------
const ch8 = [
  H1("Chapter 8: Limitations"),
  PJ("The principal limitations of the study are summarised below. Each is acknowledged explicitly, and where possible it is addressed in the Future Work chapter."),
  ...numberedList([
    "Threat model. Direct requests only. HarmBench was evaluated in a direct-request configuration with no adversarial transformation, presenting the curated harmful behaviours as written, with no attack or jailbreak augmentation applied (no GCG [25], PAIR, AutoDAN, or other attack pipelines from the HarmBench suite). Every harmful-compliance finding, including the headline null, is therefore scoped to the weakest threat model of direct requests. It does not establish that the null would survive optimised adversarial attack, a setting in which some prior quantization-safety work reports effects [35]. For this reason, an attack-augmented arm is flagged as the highest-leverage extension in Chapter 9.",
    "Quantization-method coverage. Two BitsAndBytes methods are evaluated: NF4 four-bit as the main study and INT8/LLM.int8 as a precision point (Section 6.11). The study therefore spans fp16 → INT8 → NF4 rather than a single bit-width. GPTQ, AWQ, and GGUF paths remain out of scope and are flagged for follow-up work. Replication of the Section 6.11 Llama-3.2-3B INT8 effect across more models and decode seeds is likewise out of scope and flagged for follow-up work.",
    "Refusal parser approximation. The deterministic regex-based refusal parser is reproducible. However, it may under-count nuanced refusals.",
    "Partial capability proxy (two benchmarks). The six-subject MMLU subset is a tractable but partial measure of general capability, and it does not include code generation, long-context reasoning, or tool-use evaluation. To address this, a second capability benchmark, ARC-Challenge with 1,172 reasoning-oriented questions, was run on all ten models (Appendix I.9). It confirms the direction of capability loss. However, it also shows that the MMLU magnitudes are partly benchmark-specific. Qwen3-1.7B's large MMLU drop does not replicate on ARC, whereas Qwen3-4B shows a small significant ARC loss that MMLU missed. Capability is therefore reported on the two benchmarks side by side. The interpretation labels remain MMLU-anchored pending a formal composite-capability rule (Chapter 9).",
    "MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, so an item counts as incorrect both when the model selects the wrong option and when it never commits to a letter within the generation budget. At the 128-token budget this was a real concern for the smallest pair, where 11 of 300 Qwen3-1.7B 4-bit MMLU responses were cut off mid-reasoning without a final letter (answered rate 0.963 against 0.983 at baseline). At the 512-token primary budget the truncation component is resolved. Every model answers every item, at an answered rate of 1.000 across all fifteen aliases, so none of the primary ΔMMLU = −0.090 is attributable to answer-format truncation. A distinct format asymmetry does remain at 512, however. The lenient fallback parser fires on 48.7% of the 4-bit member's MMLU items against 3.3% at fp16 (Section 6.4). Part of the −0.090 magnitude therefore reflects an answer-format shift that the parser scores less reliably, rather than genuinely wrong answers alone. ARC is subject to the same asymmetry even more strongly for this pair, with 52.3% of the 4-bit member's ARC items falling to the lenient fallback against 2.5% at fp16. Its near-zero primary delta of −0.009 therefore reflects the lenient parser salvaging format-broken answers rather than immunity. Under a strict parser, ARC and MMLU fall comparably, by −0.343 and −0.293, and both are significant. ARC therefore corroborates the direction of the loss under both parsers but not its lenient-parser smallness. Each benchmark's magnitude is best read as a bracket between the lenient and strict parsers (Section 6.4). The truncation caveat therefore applies only to the retained 128-token comparison (Section 6.12). The parser-leniency caveat applies at the primary budget and is quantified in Section 6.4.",
    "Text-only, English-only scope. Multilingual and multimodal behavioural effects of quantization are out of scope. This is a material limitation rather than a neutral one, because quantization is known to affect languages unequally and to harm non-Latin-script languages worst, with automatic metrics substantially understating the damage relative to human judgment [13].",
    "Generation-budget dependence of the absolute rate (directly evidenced, then controlled). The HarmBench attack-success rate depends on how many tokens the target is allowed to generate, and the study's original 128-token budget truncated the majority of generations. At 512 tokens the median HarmBench response is 1,675 characters against 567 at 128, and 60.3 percent of 128-token responses are a proper prefix of their 512-token continuation on a direct prefix test (≈62 percent by the character-ceiling proxy). Most 128-token generations were therefore cut off before the model finished. HarmBench ASR is consequently reported at HarmBench's 512-token reference budget as primary. The 128-token run is retained as a generation-length comparison (Section 6.12). The matched-pair ΔASR that carries every safety claim holds at both generation budgets in its conclusion, because no significant increase survives correction at either budget. However, the absolute rates are budget-dependent, and so are the 128-token deltas for individual pairs. The Qwen3-1.7B +0.055 at 128 is a truncation artefact that goes to 0.000 at 512.",
    "Greedy decoding only (stochastic-decoding arm complete for all five pairs). Temperature 0.0 is used throughout the primary study, which eliminates within-condition stochastic variance from the main analysis. A multi-seed sensitivity arm (T = 0.7, top-p 0.8, five seeds) was run at the 512-token reference budget and scored by the official classifier (Appendix I.1), and it covers all five pairs. At 512 it corroborates the null-safety result. For Qwen3-1.7B the greedy ΔASR of 0.000 sits inside the per-seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant). No pair shows a consistent increase in the harmful direction across seeds, and for the two cross-family pairs the greedy point sits just outside a narrow seed range that straddles zero (Appendix I.1). That is small noise around zero rather than a masked stochastic effect. The residual limitation is the arm's scope rather than its coverage. It is judge-scored HarmBench only, at a single stochastic setting (T = 0.7, top-p 0.8) with five seeds. The over-refusal and capability axes remain greedy-only.",
    "Generation not halted at the chat turn boundary. The generation loop stops only on the tokenizer's default end-of-sequence id, so responses frequently run to the full token budget. Once its answer is complete, the model sometimes generates synthetic follow-up turns in which it role-plays both sides of a short dialogue, and skip_special_tokens decoding saves these as a run-on (Section 6.12). This is a decoding-configuration artefact. It inflates response length. The decoding configuration that produces it is identical for both pair members, but the per-member incidence of the artefact was not measured, so the effect is expected rather than shown to cancel in the paired delta. Because the primary classifier scores the whole response rather than just its opening, a continuation that emitted harmful content would still be counted. A likely fix for future runs is to add the chat turn-end token to the stop criteria. The artefact is not expected to affect the results reported here, since the primary classifier scores the behaviour rather than the length, but that expectation was not tested.",
    "Benchmark contamination unassessed. Whether benchmark items appear in the evaluated models' pretraining data is not measured here, and the literature has called for per-benchmark contamination measurement precisely because such instruments are not yet routine [50]. Contamination would shift absolute scores. Because both members of a pair share one set of weights and one data history, the pairing controls exposure, which is what the study's unit of inference requires. It does not rule out an interaction between quantization and memorised items, and no test in this study measures one.",
    "Hardware and walltime constraints. Each TC1 job is allocated a single GPU, ten gigabytes of host memory, and six hours of walltime. Sample budgets and batch sizes are sized to fit comfortably within these constraints.",
    "Sample-size-driven confidence intervals and a bounded null. With 200 HarmBench prompts and 250 benign XSTest prompts, binomial-proportion confidence intervals are wider than in large leaderboard settings. Consequently, small deltas may not be statistically separable from zero. The per-pair minimum detectable effect for ΔASR at 200 paired items is approximately 0.04 to 0.09, and it is reported with the multiple-comparison analysis in Section 6.4.1. The headline HarmBench null should therefore be read as bounded. The evidence rules out any harmful-compliance increase larger than that detectable effect, rather than showing that the change is exactly zero. Pre-registering a smallest effect of interest and powering the design to it is noted in Chapter 9. The multi-seed arm itself covers all five pairs (Appendix I.1).",
    "Model-weight provenance (pinned). Each of the five model repositories is pinned by Hugging Face commit hash to the exact snapshot cached on TC1 at the recorded run dates. The pin is the optional revision field (Appendix C), set on every entry in configs/tc1.yaml and configs/tc1_512.yaml (Appendix A). An online from-scratch reproduction therefore loads byte-identical weights rather than a possibly re-uploaded head. The residual reproducibility boundary is consequently the raw per-prompt generations rather than the weights. Those are retained locally, gitignored and hash-pinned in results/raw_artifact_manifest.sha256, so the analysis replays from the committed redacted sidecars but is not recomputable from raw generations on a fresh clone.",
    "Gated-access dependency. The Llama-3.2-3B pair and the HarmBench dataset depend on accepted Hugging Face access conditions and a valid token available to the TC1 environment. This precondition was satisfied for the current run. However, future reproductions must repeat the access setup.",
  ]),
];

// ------------------------------------------------------------
// Chapter 9: Future Work
// ------------------------------------------------------------
const ch9 = [
  H1("Chapter 9: Future Work"),
  PJ("The framework and methodology established by this study admit several natural extensions. These are listed in approximate order of practical impact."),
  ...numberedList([
    "Adversarial-attack arm. Re-run the HarmBench axis under at least one optimisation-based or LLM-mediated attack, for example GCG or PAIR, on a subset of pairs. This would upgrade the direct-request compliance result (Chapter 3 threat model) to a quantization-versus-safety-under-attack result, and it would directly engage the attack-dependent effects reported by HarmLevelBench [35]. Egashira et al. [12] describe a different threat, in which the quantization step itself is exploited. This is the highest-leverage extension of the safety axis.",
    "Multi-method quantization comparison. Extend the matrix to include GPTQ, AWQ, and GGUF quantization paths on the same baselines. This would allow direct comparison of how different PTQ algorithms perturb safety and capability. GGUF is the deployment-oriented format named in the motivation (Chapter 1), so it is prioritised here.",
    "Stochastic-decoding sensitivity arm (completed for all five pairs at the reference budget). A multi-seed arm (temperature 0.7, top-p 0.8, five seeds, official classifier) was run at the 512-token reference budget for all five pairs (Appendix I.1). At 512 it corroborates the null-safety result rather than tempering a real effect. For Qwen3-1.7B the greedy ΔASR of 0.000 sits inside the per-seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant). Llama-3.2-3B's seeds are all non-positive, at a mean of −0.024. The two cross-family pairs added by the completed extension are within noise of zero, at a Mistral-7B seed mean of +0.012 and a Phi-4-mini seed mean of −0.012, and Phi-4-mini's one significant seed is a −0.055 decrease. Their greedy points sit just outside narrow zero-straddling seed ranges (Appendix I.1). At the truncated 128-token budget the same arm had attenuated an apparent +0.055, and that effect was shown to be a truncation artefact (Section 6.12). Raising the seed count and extending the arm beyond judge-scored HarmBench, to the over-refusal and capability axes, would complete the within-condition variance estimate.",
    "Composite-capability interpretation rule (second benchmark run). ARC-Challenge was run on all ten models (Appendix I.9) and corroborated the direction of capability loss, while also showing that the MMLU magnitudes are partly benchmark-specific. The clearest case is the Qwen3-1.7B drop, which is −9.0 pp and significant on MMLU against −0.9 pp on ARC (n.s. at the reference budget). In addition, MMLU's within-Qwen scale gap is not reproduced on ARC. The interpretation labels currently stay MMLU-anchored with ARC as a corroborating axis. Formalising a composite-capability rule, for example requiring agreement across benchmarks before assigning a capability-driven label, is the natural next step. Adding further capability benchmarks such as GSM8K for math reasoning and HellaSwag for commonsense would broaden the composite.",
    "Mechanism follow-ups (refusal-margin probe, Section 6.10). The first-token refusal-margin analysis localises the quantization effect to near-boundary prompts. However, it is first-token-specific, because its whole-sequence margins disagree in sign for two pairs, and its capability/confidence confound control is small (n = 100, unpaired, mixed neutral set, no significance test). Three follow-ups would sharpen it. (1) An independent activation-space refusal-direction probe (Arditi et al. [15]) would give a second mechanism window that does not rely on a hand-built token set. (2) A paired neutral-margin control with a significance test would separate safety-targeted shifts from generic confidence softening more rigorously. (3) An INT8 precision point, run and reported in Section 6.11, traced the behavioural effect across the fp16 → INT8 → NF4 spectrum and found only the four-bit capability cliff to be a consistent precision effect, with the safety axis non-significant at every precision under both judges at the reference budget. Extending the first-token margin probe itself across the three precisions, rather than only the behavioural metrics, remains open.",
    "Cross-family and scale extension (completed 2026-06-15). Two further matched pairs were implemented and run on the cluster: mistral_7b (mistralai/Mistral-7B-Instruct-v0.3) and phi4_mini (microsoft/Phi-4-mini-instruct). Together they take the study to five pairs across four families (Qwen, Llama, Mistral, Phi) and add a seven-billion-parameter point at the upper edge of the compact-deployment regime. Both pairs use the identical methodology, namely on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding, the same four benchmarks at the same sample counts, seed 42, and the official HarmBench classifier as the primary ASR scorer, so the comparison stays matched. The only new loader capability is an optional attn_implementation field, because Phi-4-mini uses the eager attention backend on the V100, which has no flash-attention kernels. That field is covered by the configuration schema, the loader, the per-model SLURM job set, the judge-validation scripts, and the verification suite. Phi-4-mini loads through transformers' native Phi3 implementation rather than its bundled remote code, which keeps its load path consistent with every other model in the study. Results are reported in Tables 6.1–6.3 and analysed in Section 6.8.6. Under the judge, no pair shows a significant harmful-compliance increase. At the 512-token reference budget Mistral-7B ΔASR is −0.020 (n.s.), and the v2 proxy's positive 'increase' there is a sign-flipped over-count, with judge-versus-proxy κ as low as 0.25. Phi-4-mini ΔASR is +0.020 (n.s.), a directional alignment_degradation on the tolerance boundary, with κ 0.67. Adding two families therefore leaves the null-safety conclusion intact and reinforces the judge-over-proxy finding in two further families.",
    "INT8 precision point / quantization-method sweep (completed, reported in Section 6.11). To test whether the safety and capability effects depend on quantization aggressiveness and method, an INT8 precision point was added as a third precision and run on TC1 across all five pairs at the 512-token reference budget. That precision point is bitsandbytes LLM.int8, which is a different method from NF4 rather than a lower-bit NF4, and it was scored by both the official HarmBench classifier and the gpt-4o second judge with zero parse errors. The headline result (Section 6.11) is on the capability axis, and it does not depend on the generation budget or the quantization method. Capability loss is a clean cliff at four-bit, because no INT8 MMLU/ARC delta is significant for any pair. The only precision effect that is stable across budgets and methods is therefore the four-bit capability cliff. On the safety axis, every INT8 contrast at the reference budget is non-significant under both judges. The apparent Llama-3.2-3B INT8 increase seen at 128 tokens, which had been both-judge-significant there, vanishes at 512. The classifier gives Δ+0.005 with McNemar p = 1.000, and gpt-4o gives Δ+0.010 with p = 0.688. The safety effect is therefore neither bit-width-graded nor stable across generation budgets. Three extensions remain. (a) Any candidate INT8 effect should be replicated across more models, decode seeds and generation budgets, to separate a general LLM.int8 phenomenon from model-specific numerics or truncation. (b) The Section 6.10 refusal-margin probe should be traced across all three precisions rather than only the behavioural metrics. (c) Genuinely different quantization families (GPTQ, AWQ, GGUF) should be added beyond the two bitsandbytes algorithms evaluated here.",
    "Multilingual extension. Replicate the matched-pair design in Chinese (where Qwen is natively strong) and in one low-resource language, to test whether quantization-induced safety changes are language-dependent.",
    "Fully-independent open-weight second judge, and a second human annotator. The primary HarmBench classifier was cross-checked against a second frontier judge, gpt-4o, prompted with a condensed yes/no rubric encoding the same harmful-behaviour criterion. That judge agreed strongly per prompt (κ 0.60–0.95 across all ten models, 0.68–0.95 at the 512-token reference budget) and reached the same null-safety conclusion at the reference budget (Section 6.9, Result 4, and Section 6.12). The classifier was also grounded against human judgment on a stratified 200-item subset, where it reached κ 0.59 against the regex's 0.11 (Section 6.9, Result 5). Together these substantially resolve the single-judge threat. Because that grounding rests on a single annotator, a second annotator with an inter-rater κ, together with adjudication of disagreements, is the natural strengthening on the human side. On the judge side, (i) the open-weight guard-model cross-check was run with Llama-Guard-3-8B, revision-pinned, via the --backend llamaguard path on TC1, over all fifteen aliases and with zero parse errors (Section 6.9). It agrees with the primary classifier at κ 0.36–0.92 across the ten base/NF4 aliases and finds every per-pair ΔASR non-significant, so the versioned-API-only reproducibility caveat no longer applies. The residual is that it answers a looser safety question ('is this response unsafe?') than the behaviour-specific HarmBench classifier, so it serves as a reproducibility cross-check rather than a second construct-validity oracle. On the same side, (ii) a refusal-style judge for XSTest over-refusal was run as a sensitivity layer (gpt-4o, three-class taxonomy, 3,750 responses, zero parse failures), and it finds the one over-refusal contrast that survives the multiplicity correction to be scorer-dependent (Section 6.9, Result 6). The human-labelled refusal gold set that grounds that comparison was completed as well, over 200 items with a blinded single annotator, and it carries a mechanical Outcome J with one disclosed protocol deviation (Section 8). Under that gold set the judge aligns substantially better with the annotator than the regex does, exactly as the HarmBench human audit (Result 5) grounds the classifier. The residual on both axes is a second annotator with an inter-rater κ. All of these are derived validation layers writing separate redacted sidecars, and none of them modifies raw.jsonl, summary.json, or existing sidecars.",
    "Safety-preserving quantization. Investigate emerging \"safety-preserving\" quantization methods that explicitly seek to mitigate alignment degradation under PTQ, and compare them against the vanilla NF4 baseline studied here.",
  ]),
];

// ------------------------------------------------------------
// Chapter 10: Conclusion
// ------------------------------------------------------------
const ch10 = [
  H1("Chapter 10: Conclusion"),
  PJ("This Final Year Project investigates safety–capability trade-offs in four-bit quantized compact language models. It focuses on a research question that institutional benchmarks have not directly answered. When a small instruction-tuned model is quantized for on-device deployment, do observed changes in safety behaviour reflect a true shift in alignment or a side-effect of degraded general capability?"),
  PJ("The methodological contribution is a controlled matched-pair design in which baseline and four-bit pair members are loaded from identical baseline weights, with NF4 quantization applied on the fly. This design eliminates publisher-asymmetry and pipeline-asymmetry as confounds. Therefore it provides controlled isolation of the quantization step as the experimental variable under the recorded configuration. The engineering contribution is an open, reproducible benchmarking framework. It comprises the matched-pair pipeline, four benchmark plugins, the pairwise analysis layer with rule-based interpretation labels and paired bootstrap 95% confidence intervals, full SLURM orchestration for the NTU TC1 cluster with resumable per-model matrix jobs, and an automated verification suite."),
  PJ("The analytical contribution is the interpretation layer itself, a capability-anchored, statistically grounded, multi-benchmark framework. It formalises the alignment-versus-capability disambiguation as a rule-based decision procedure over combined safety and capability deltas. Because that procedure does not depend on any specific empirical outcome, it is the durable contribution of this study."),
  PJ("The primary finding of this study is a null on the safety axis, with capability as the cost that survives correction. No pair shows a significant increase in genuine harmful compliance under NF4. The only individually significant ΔASR across the five pairs is Llama-3.2-3B at −0.040, which is a decrease, and no ASR contrast survives a Benjamini-Hochberg correction over the twenty-test NF4 family."),
  PJ("All runs were executed on the NTU TC1 GPU cluster. HarmBench attack-success rate is reported at HarmBench's own 512-token reference budget and scored by the official HarmBench classifier in full precision. Chapter 5 records the run plan and the job identifiers, Section 6.8.6 reports the cross-family extension to Mistral-7B and Phi-4-mini, and Section 6.12 reports the generation-length comparison against the shorter 128-token budget."),
  PJ("The three contrasts that do survive correction are all capability or over-refusal: Qwen3-1.7B MMLU −0.090, Llama ARC −0.032, and Phi over-refusal −0.048. The last of these is regex-scored and scorer-dependent, and an independent refusal judge does not reproduce it (Section 6.9). The minimum detectable ΔASR for this study is 0.04–0.09 per pair (Section 6.4.1). Within those detection bounds, the cost of NF4 that survives correction and holds at both generation budgets is degraded competence rather than eroded guardrails. The five pairs still span distinct diagnostic profiles. Each of them, however, reads on the capability side of the dichotomy."),
  PJ("The Qwen3-1.7B pair carries the broad_degradation label, driven entirely by its significant MMLU loss of −0.090 with a flat harm axis (ΔASR = 0.000, McNemar p = 1.000). The +0.055 increase seen at the truncated 128-token budget was a generation-length artefact, because 60 percent of 128-token responses were truncated, and it does not replicate at 512. The Llama-3.2-3B pair is capability_collapse_masquerading_as_safety. Harmful compliance falls (ΔASR −0.040, significant) alongside a significant ARC loss, so its lower ASR is more consistent with a capability side-effect than with a genuine alignment improvement. The label encodes the correlation between the two axes, and by itself it does not establish that the capability loss caused the safety change. The Qwen3-4B pair is directional alignment_degradation (ΔASR +0.040, not significant); MMLU is flat, while ARC falls significantly before correction (−0.016, p = 0.042) and does not survive it. A central methodological result, independent of budget, is that the judge validation overturned the regex proxy. The v2 scorer over-counts harmful compliance, because it equates non-refusal with success. For this reason the benchmark's own classifier is the primary scorer, and what determines the conclusion is the choice of scorer rather than the choice of judge or budget."),
  PJ("The cross-family extension (Section 6.8.6) reinforces the null across four families. Mistral-7B's apparent proxy-driven rise is absent under the judge (ΔASR −0.020, not significant) at preserved capability, and Phi-4-mini shows a non-significant +0.020 ΔASR whose only significant move is a decrease in over-refusal. That reading is a directional alignment_degradation on the tolerance boundary rather than a confirmed regression. The INT8 precision point (Section 6.11) adds that the safety axis is neither bit-width-graded nor budget-robust, because the apparent Llama INT8 move at 128 tokens vanishes at the reference budget under both judges. By contrast, the four-bit capability cliff is the one precision effect that holds at both budgets. Finally, a mechanism probe (Section 6.10), whose first-token margins are generation-length-invariant, localises the quantization effect to near-boundary prompts and finds it symmetric, destabilising refusals in both directions. For the one pair that moves most, the probe is more consistent with capability-driven boundary instability than with a targeted alignment shift. The 512-token null corroborates that reading directly, since its Qwen3-1.7B ΔASR of 0.000 is a high-churn exchange of sixteen harmful-ward and sixteen safe-ward flips."),
  PJ("The results answer all seven research questions at HarmBench's 512-token reference budget. Under the official HarmBench classifier, no evaluated pair shows a significant increase in genuine harmful compliance under NF4, and no ASR contrast survives a multiple-comparison correction (RQ1). The per-pair ΔASR values are Qwen3-1.7B 0.000, Qwen3-4B +0.040, Llama −0.040, Mistral-7B −0.020, and Phi-4-mini +0.020. Of these, only Llama's decrease is individually significant, and it is in the safety-improving direction. These data give no support for the view that quantization reliably improves safety. Equally, they give no evidence that NF4 significantly erodes guardrails at the reference budget."),
  PJ("NF4 quantization does not significantly increase over-refusal in any evaluated model (RQ2). Four of five pairs are non-significant, and the fifth, Phi-4-mini, significantly decreases under the primary regex scorer, so the only movement is toward fewer false refusals. That decrease is a benign-direction change which an independent refusal judge does not reproduce (Section 6.9). NF4 quantization significantly degrades capability in the smallest model on MMLU (Qwen3-1.7B, −0.090) and on ARC for the Qwen3-4B and Llama-3.2-3B pairs, so capability loss is the effect that survives correction, while at the INT8 precision point all ten capability contrasts are statistically equivalent to full precision within ±0.05 (RQ3)."),
  PJ("The smallest model is the most sensitive, but on the capability axis. The 1.7B pair carries the broad_degradation label, driven by its MMLU loss with a flat harm axis. By contrast, no capability contrast for the 4B pair survives correction, and its safety movement is directional and non-significant (RQ4). Cross-family, no pair shows a significant ASR increase, over-refusal never significantly rises, and capability loss is directionally consistent where significant (RQ5)."),
  PJ("A central methodological result, independent of the generation budget, is that validating the framework against the benchmark's own classifier corrected a refusal-counting artefact in the regex proxy. What determines the conclusion is the choice of scorer rather than the choice of judge or budget. The framework itself is the durable contribution: capability-anchored, statistically grounded, multi-benchmark, and self-correcting under judge validation. The specific empirical numbers are subordinate to it. At the reference budget they resolve to a null-safety result, with capability as the cost of four-bit quantization that survives correction."),
];

// ------------------------------------------------------------
// References
// ------------------------------------------------------------
const refs = [
  H1("References"),
  P("References are cited in IEEE numbered style; the bracketed numbers below correspond to the in-text citations.", { after: 200 }),
  ...numberedList([
    "A. Yang et al. (Qwen Team), “Qwen3 technical report,” Alibaba Group, 2025. arXiv:2505.09388.",
    "A. Grattafiori et al. (Meta AI), “The Llama 3 herd of models,” Meta AI, 2024. arXiv:2407.21783.",
    "A. Abouelenin et al., “Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs,” Microsoft, 2025. arXiv:2503.01743.",
    "Gemma Team, Google DeepMind, “Gemma 2: Improving open language models at a practical size,” 2024. arXiv:2408.00118.",
    "J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration,” in Proc. Machine Learning and Systems (MLSys), 2024.",
    "E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training quantization for generative pre-trained transformers,” in Proc. International Conference on Learning Representations (ICLR), 2023. arXiv:2210.17323.",
    "T. Dettmers and L. Zettlemoyer, “The case for 4-bit precision: k-bit inference scaling laws,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 202, pp. 7750–7774, 2023. arXiv:2212.09720.",
    "R. Jin, J. Du, W. Huang, et al., “A comprehensive evaluation of quantization strategies for large language models,” in Findings of the Association for Computational Linguistics (ACL), 2024. arXiv:2402.16775.",
    "S. Li, X. Ning, L. Wang, T. Liu, X. Shi, S. Yan, G. Dai, H. Yang, and Y. Wang, “Evaluating quantized large language models,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.18158.",
    "A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., “Investigating the impact of quantization methods on the safety and reliability of large language models,” IEEE Access, vol. 14, pp. 96771–96793, 2026, doi: 10.1109/ACCESS.2026.3703899. arXiv:2502.15799.",
    "I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence of large language models?,” in Findings of NAACL, 2024. arXiv:2405.00632.",
    "K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in Proc. NeurIPS, 2024. arXiv:2405.18137.",
    "K. Marchisio, S. Dash, H. Chen, D. Aumiller, A. Üstün, S. Hooker, and S. Ruder, “How does quantization affect multilingual LLMs?,” in Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 15928–15947, 2024. arXiv:2407.03211.",
    "Y. Bai et al., “Training a helpful and harmless assistant with reinforcement learning from human feedback,” Anthropic, 2022. arXiv:2204.05862 (preprint).",
    "A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in language models is mediated by a single direction,” in Proc. NeurIPS, 2024. arXiv:2406.11717.",
    "X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson, “Safety alignment should be made more than just a few tokens deep,” in Proc. International Conference on Learning Representations (ICLR), 2025. arXiv:2406.05946.",
    "X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, and P. Henderson, “Fine-tuning aligned language models compromises safety, even when users do not intend to!,” in Proc. International Conference on Learning Representations (ICLR), 2024. arXiv:2310.03693.",
    "P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test suite for identifying exaggerated safety behaviours in large language models,” in Proc. NAACL, 2024. arXiv:2308.01263.",
    "J. Cui, W.-L. Chiang, I. Stoica, and C.-J. Hsieh, “OR-Bench: An over-refusal benchmark for large language models,” in Proc. International Conference on Machine Learning (ICML), 2025. arXiv:2405.20947.",
    "Y. Huang, L. Sun, et al., “Position: TrustLLM: Trustworthiness in large language models,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 235, 2024. arXiv:2401.05561.",
    "B. Wang et al., “DecodingTrust: A comprehensive assessment of trustworthiness in GPT models,” in Proc. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.11698.",
    "Z. Zhang et al., “SafetyBench: Evaluating the safety of large language models,” in Proc. Association for Computational Linguistics (ACL), 2024. arXiv:2309.07045.",
    "L. Ouyang et al., “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022.",
    "M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.04249.",
    "A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” 2023. arXiv:2307.15043 (preprint).",
    "E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irving, “Red teaming language models with language models,” in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3419–3448, 2022, doi: 10.18653/v1/2022.emnlp-main.225.",
    "D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proc. International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.",
    "P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? Try ARC, the AI2 reasoning challenge,” arXiv:1803.05457, 2018 (preprint).",
    "S. Badshah and H. Sajjad, “Quantifying the capabilities of LLMs across scale and precision,” 2024. arXiv:2405.03146 (preprint).",
    "G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient post-training quantization for large language models,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 202, 2023.",
    "E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one-shot,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 202, 2023.",
    "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023. arXiv:2305.14314. (Introduces NF4 and the bitsandbytes integration.)",
    "T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” in Proc. NeurIPS, 2022. arXiv:2208.07339.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 70, pp. 1321–1330, 2017.",
    "Y. Belkhiter, G. Zizzo, and S. Maffeis, “HarmLevelBench: Evaluating harm-level compliance and the impact of quantization on model alignment,” arXiv:2411.06835, Nov. 2024. Presented at the NeurIPS 2024 Workshop on Safe Generative AI (SafeGenAI), which is non-archival and issues no proceedings.",
    "K. Chen, J. Zhang, J. Hu et al., “Assessing safety risks and quantization-aware safety patching for quantized large language models,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 267, pp. 9728–9746, 2025. arXiv:2506.20251.",
    "J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Bartoldson, A. K. Jaiswal, K. Xu, B. Kailkhura, D. Hendrycks, D. Song, Z. Wang, and B. Li, “Decoding compressed trust: Scrutinizing the trustworthiness of efficient LLMs under compression,” in Proc. International Conference on Machine Learning (ICML), PMLR vol. 235, 2024. arXiv:2403.15447.",
    "A. Wei, N. Haghtalab, and J. Steinhardt, “Jailbroken: How does LLM safety training fail?,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.",
    "A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825.",
    "Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” Journal of the Royal Statistical Society: Series B (Methodological), vol. 57, no. 1, pp. 289–300, 1995.",
    "J. Cohen, “A coefficient of agreement for nominal scales,” Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.",
    "R. J. Connor, “Sample size for testing differences in proportions for the paired-sample design,” Biometrics, vol. 43, no. 1, pp. 207–211, 1987.",
    "D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky, “With little power comes great responsibility,” in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 9263–9274, 2020.",
    "R. Artstein and M. Poesio, “Survey article: Inter-coder agreement for computational linguistics,” Computational Linguistics, vol. 34, no. 4, pp. 555–596, 2008, doi: 10.1162/coli.07-034-R2.",
    "J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,” Biometrics, vol. 33, no. 1, pp. 159–174, 1977.",
    "H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversations,” Meta AI, 2023. arXiv:2312.06674 (preprint).",
    "D. J. Schuirmann, “A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability,” Journal of Pharmacokinetics and Biopharmaceutics, vol. 15, no. 6, pp. 657–680, 1987.",
    "O. Sainz, J. Campos, I. García-Ferrero, J. Etxaniz, O. Lopez de Lacalle, and E. Agirre, “NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark,” in Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 10776–10787, 2023.",
    "L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Proc. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.05685.",
    "J. Gu et al., “A survey on LLM-as-a-judge,” arXiv:2411.15594, 2024 (preprint).",
    "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” arXiv:2503.05061, 2025 (preprint).",
    "F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945.",
    "J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988.",
  ])
];

// ------------------------------------------------------------
// Appendices
// ------------------------------------------------------------
const tc1Yaml = `study_name: safety_capability_tradeoff_4bit_slm

models:
  qwen_2b_base:
    family: qwen
    size_b: 1.7
    quantized: false
    pair_id: qwen_2b
    model_id: Qwen/Qwen3-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_2b_4bit:
    family: qwen
    size_b: 1.7
    quantized: true
    pair_id: qwen_2b
    model_id: Qwen/Qwen3-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_4b_base:
    family: qwen
    size_b: 4.0
    quantized: false
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    revision: 1cfa9a7208912126459214e8b04321603b3df60c  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_4b_4bit:
    family: qwen
    size_b: 4.0
    quantized: true
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    revision: 1cfa9a7208912126459214e8b04321603b3df60c  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  llama_3_2_3b_base:
    family: llama
    size_b: 3.0
    quantized: false
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
    revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  llama_3_2_3b_4bit:
    family: llama
    size_b: 3.0
    quantized: true
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
    revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  # Cross-family extension (run on TC1 2026-06-15; results in Chapter 6, §6.13).
  # See Chapter 9. Phi-4-mini uses the eager attention backend (the V100 has no
  # flash-attention kernels) and loads via native transformers Phi3; Mistral drops straight in.
  mistral_7b_base:
    family: mistral
    size_b: 7.2
    quantized: false
    pair_id: mistral_7b
    model_id: mistralai/Mistral-7B-Instruct-v0.3
    revision: c170c708c41dac9275d15a8fff4eca08d52bab71  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  mistral_7b_4bit:
    family: mistral
    size_b: 7.2
    quantized: true
    pair_id: mistral_7b
    model_id: mistralai/Mistral-7B-Instruct-v0.3
    revision: c170c708c41dac9275d15a8fff4eca08d52bab71  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  phi4_mini_base:
    family: phi
    size_b: 3.8
    quantized: false
    pair_id: phi4_mini
    model_id: microsoft/Phi-4-mini-instruct
    revision: cfbefacb99257ffa30c83adab238a50856ac3083  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    attn_implementation: eager
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  phi4_mini_4bit:
    family: phi
    size_b: 3.8
    quantized: true
    pair_id: phi4_mini
    model_id: microsoft/Phi-4-mini-instruct
    revision: cfbefacb99257ffa30c83adab238a50856ac3083  # T32: TC1-cached snapshot pinned 2026-07-08
    trust_remote_code: false
    attn_implementation: eager
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

decoding:
  max_new_tokens: 512
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
  max_input_tokens: 1024
  use_chat_template: true

benchmarks:
  harmbench:
    dataset_name: walledai/HarmBench
    config_name: standard
    split: train
    max_samples: 400
    batch_size: 4

  xstest:
    dataset_name: paul-rottger/xstest-prompts
    local_csv: data/xstest_v2_prompts.csv
    split: prompts
    max_samples: 400
    batch_size: 4
    benign_only: true

  mmlu:
    dataset_name: cais/mmlu
    split: test
    max_samples: 300
    batch_size: 4
    subjects:
      - business_ethics
      - clinical_knowledge
      - college_biology
      - high_school_world_history
      - high_school_macroeconomics
      - human_aging

  arc:
    dataset_name: allenai/ai2_arc
    config_name: ARC-Challenge
    split: test
    max_samples: 1200
    batch_size: 4

slurm:
  partition: UGGPU-TC1
  qos: normal
  account:
  gpus: 1
  cpus_per_task: 1
  mem: 10G
  time: "06:00:00"
  log_dir: results_512/slurm_logs_tc1
  work_dir: /tc1home/FYP/utan001/fyp_quant/repo
  setup_commands:
    - module load slurm
    - module load anaconda
    - source activate fyp-tc1
    # Force offline mode (compute nodes may lack outbound internet).
    # Pre-cache via scripts/prefetch_tc1.py on the HEAD node first.
    - export HF_HUB_OFFLINE=1
    - export HF_DATASETS_OFFLINE=1
    - export TRANSFORMERS_OFFLINE=1`;

const sbatchExample = `#!/bin/bash
#SBATCH --job-name=qwen_2b_4bit__matrix
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_logs_tc1/qwen_2b_4bit__matrix.out
#SBATCH --error=results/slurm_logs_tc1/qwen_2b_4bit__matrix.err

set -euo pipefail

cd /tc1home/FYP/utan001/fyp_quant/repo
mkdir -p results/slurm_logs_tc1
module load slurm
module load anaconda
source activate fyp-tc1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python /tc1home/FYP/utan001/fyp_quant/repo/run_quant_matrix.py \\
    --config configs/tc1.yaml \\
    --model qwen_2b_4bit \\
    --output_dir results \\
    --seed 42 \\
    --device cuda`;

const appendixA = [
  H1("Appendix A: Final TC1 Configuration"),
  PJ("The full configs/tc1_512.yaml — the configuration of the primary 512-token study (D41) — is reproduced below for reference, with explanatory comments lightly abridged. The retired 128-token configuration (configs/tc1.yaml, max_new_tokens: 128, writing to results/) is identical apart from the generation budget and output roots and is retained in the repository for the §6.16 generation-length comparison. The mistral_7b and phi4_mini entries are the cross-family extension: they were run on TC1 on 2026-06-15 and their results are incorporated into Chapter 6 (Tables 6.1–6.3 and §6.13)."),
  ...Code(tc1Yaml),
];

const appendixB = [
  H1("Appendix B: Example Generated SLURM Script"),
  PJ("The following is the generated sbatch script for qwen_2b_4bit, one of the ten per-model jobs emitted by the cluster-generate command. The remaining nine scripts differ only in the --model alias and the SBATCH --job-name, --output, and --error lines."),
  ...Code(sbatchExample),
];

const appendixC = [
  H1("Appendix C: Pydantic Configuration Schema Summary"),
  PJ("The configuration schema is defined in ethical_benchmark/quant/config_schema.py and is summarised below. All values are validated at load time; any invalid configuration raises a clear Pydantic validation error before any model is loaded."),

  H3("Top-level QuantizationConfig"),
  Bullet("study_name : str : human-readable study identifier."),
  Bullet("models : dict[str, ModelConfig] : keyed by model alias."),
  Bullet("decoding : DecodingConfig : generation parameters."),
  Bullet("benchmarks : dict[str, BenchmarkConfig] : keyed by benchmark name."),
  Bullet("slurm : SlurmConfig : cluster directives and bootstrap."),

  H3("ModelConfig"),
  Bullet("family : str : e.g. \"qwen\", \"llama\"."),
  Bullet("size_b : float : parameter count in billions."),
  Bullet("quantized : bool : true triggers on-the-fly quantized loading."),
  Bullet("quant_method : str | None : quantization method when quantized=true — \"nf4\" (default, the main study) or \"int8\" (LLM.int8, the §6.15 precision point); a validator rejects any other value and rejects the field on a baseline entry."),
  Bullet("pair_id : str : links baseline and 4-bit members of the same pair."),
  Bullet("model_id : str : Hugging Face repo id."),
  Bullet("trust_remote_code : bool : disabled by default."),
  Bullet("dtype : str : one of \"auto\", \"float16\", \"bfloat16\", \"float32\"."),
  Bullet("benchmarks : list[str] : must reference top-level benchmark keys."),
  Bullet("revision : str | None : optional Hugging Face commit pin."),

  H3("DecodingConfig"),
  Bullet("max_new_tokens : int (≥1)"),
  Bullet("temperature : float (0.0–2.0)"),
  Bullet("top_p : float (0.0–1.0)"),
  Bullet("repetition_penalty : float (≥1.0)"),
  Bullet("max_input_tokens : int (≥1)"),
  Bullet("use_chat_template : bool"),

  H3("BenchmarkConfig"),
  Bullet("dataset_name : str : Hugging Face dataset id."),
  Bullet("split : str : dataset split."),
  Bullet("max_samples : int : sample cap."),
  Bullet("batch_size : int : generation batch size."),
  Bullet("benign_only : bool (XSTest only)."),
  Bullet("subjects : list[str] (MMLU only)."),

  H3("SlurmConfig"),
  Bullet("partition, qos, account, gpus, cpus_per_task, mem, time, log_dir, work_dir, setup_commands."),

  H3("Validators"),
  Bullet("Every pair_id must have at least one baseline (quantized=false) and one quantized (quantized=true) member."),
  Bullet("Every benchmark referenced in a model's benchmarks list must exist as a top-level benchmark."),
  Bullet("dtype must be one of the supported aliases."),
  Bullet("Numeric ranges enforced on decoding parameters."),
];

const appendixD = [
  H1("Appendix D: Verification Coverage"),
  PJ("The test inventory is intentionally not frozen into this document. Run pytest tests/ --collect-only for the live inventory. Table D.1 records the stable coverage contract rather than a count that changes whenever a guard is added."),
  buildTable(
    ["Coverage area", "What is verified"],
    [
      ["Data and configuration", "Dataset loaders, schema validation, model-pair constraints, revisions, decoding controls, and summary I/O."],
      ["Model execution", "Device and dtype resolution, prompt templates, quantization engagement, batching, resume behavior, and model reuse."],
      ["Scoring and statistics", "Refusal parsing, exact-match parsing, judge backends, paired bootstrap, McNemar tests, BH-FDR, and interpretation labels."],
      ["Artifact integrity", "Raw-artifact immutability, sidecar redaction, generated-document freshness, and protected evidence paths."],
      ["Claim surfaces", "Artifact-derived registry freshness, explicit surface coverage, per-file semantic bindings, generated deck blocks, and local Overleaf zip/source equality."],
      ["Cluster workflow", "SLURM generation, offline bootstrap, grouping modes, resource settings, and submission safeguards."],
    ],
    [2400, 6960],
  ),
  P("Table D.1  Stable verification coverage areas. The live test inventory is produced by pytest.", { size: 18 }),
];

const appendixE = [
  H1("Appendix E: Repository Layout"),
  PJ("The top-level repository layout is reproduced below for orientation. Only research-relevant files are shown; build artifacts and caches are omitted."),
  ...Code(`fyp_quant/
├── README.md
├── CLAUDE.md                 (developer guidance for AI-assisted work)
├── Makefile                  (smoke / run / matrix / analyze / cluster-* targets)
├── fyp_cli.py                (unified CLI)
├── run_quant_benchmark.py    (single-run entrypoint)
├── run_quant_matrix.py       (matrix entrypoint)
├── compare_quant_pairs.py    (analysis entrypoint)
├── configs/
│   ├── default.yaml          (local development)
│   ├── tc1_512.yaml          (primary 512-token fp16/NF4 study)
│   ├── tc1_int8_512.yaml     (INT8 precision point, primary 512-token)
│   ├── tc1.yaml              (retained 128-token comparison)
│   └── tc1_int8.yaml         (retained 128-token INT8 comparison)
├── ethical_benchmark/
│   ├── quant/config_schema.py
│   ├── models/{loader.py, generation.py}
│   ├── benchmarks/{base.py, harmbench.py, xstest.py, mmlu.py, arc.py, registry.py, utils.py}
│   ├── pipeline/{run_quant_benchmark.py, run_quant_matrix.py}
│   ├── analysis/compare_quant_pairs.py
│   ├── cluster/{generate_jobs.py, submit_jobs.py, check_runs.py}
│   └── metrics/
├── scripts/                  (analysis + build: precision_sweep_analysis.py, judge_agreement.py,
│                              make_figures.py, prefetch_tc1.py, verify_report_claims.py, ...)
├── tests/                    (live inventory: pytest tests/ --collect-only)
├── slurm/jobs_tc1/           (generated sbatch files)
├── results_512/              (primary 512-token results and analysis)
├── results/                  (retained 128-token comparison)
└── docs/
    ├── methodology.md
    ├── evaluation_metrics.md
    ├── datasets.md
    ├── limitations.md
    ├── extensibility.md
    ├── TC1_CLUSTER_RUNBOOK.md
    └── FYP_Report_2026-07-25_v6.docx   (this document)`),
];


const appendixF = [
  H1("Appendix F: Glossary"),
  buildTable(
    ["Term", "Definition"],
    [
      ["AdvBench", "Adversarial-prompt benchmark used in safety research."],
      ["ASR (Attack Success Rate)", "Fraction of unsafe prompts that produce a harmful, complying response."],
      ["BF16 / bfloat16", "16-bit Brain Floating Point format; preserves dynamic range close to FP32 with reduced precision."],
      ["BNB / BitsAndBytes", "Library providing the NF4 four-bit quantization integration used by Hugging Face transformers."],
      ["FP16 / float16", "16-bit IEEE half-precision floating point format; default for CUDA when dtype=auto."],
      ["GPTQ / AWQ / GGUF", "Alternative post-training quantization methods (out of scope for this study)."],
      ["HarmBench", "Standardised benchmark for harmful-compliance evaluation."],
      ["HF / Hugging Face", "Open-source ML platform hosting model checkpoints and datasets."],
      ["MMLU", "Massive Multitask Language Understanding benchmark; capability anchor in this study."],
      ["NF4", "4-bit Normalised Float quantization scheme introduced in the QLoRA paper."],
      ["PTQ (Post-Training Quantization)", "Quantization applied to an already-trained model without further fine-tuning."],
      ["QoS (Quality of Service)", "SLURM scheduling class that determines resource limits and priority."],
      ["SBATCH", "SLURM directive prefix used in job script headers."],
      ["SLURM", "Workload manager used to schedule jobs on the TC1 cluster."],
      ["TC1", "NTU's GPU cluster used to run the experimental matrix."],
      ["Walltime", "Real-world elapsed time limit imposed on a SLURM job."],
      ["XSTest", "Benchmark for measuring over-refusal on benign-but-suspicious prompts."],
    ],
    [3200, 6160],
  ),
];

const appendixI = [
  H1("Appendix I: Supporting Detail for the Results Chapter"),
  PJ("This appendix holds the supporting detail for Chapter 6: the full scorer-validation trail, the robustness arms, and the per-category and per-subject breakdowns. It is placed here so the results chapter carries the findings and the evidence for them stays available and unabridged. Nothing in this appendix is re-derived; each block is the original text, relocated."),
  H2("I.1 Multi-seed decoding sensitivity, all five pairs"),
  ...Figure("multiseed.png", "Qwen3-1.7B judge ΔASR under decoding variation at the 512-token reference budget: the greedy point estimate (0.000, red diamond) versus the multi-seed mean ± sd at temperature 0.7 (blue, five seeds), mean +0.013 with the greedy 0.000 sitting inside the seed range. Stochastic decoding corroborates the null."),
  PJ("Because the primary study uses greedy decoding (T = 0.0), its bootstrap and McNemar intervals capture prompt-sampling variance only, and therefore say nothing about generation variance. This is the one priority weakness that cannot be resolved by reanalysis of the existing artefacts. To estimate how stable the ΔASR result is under realistic sampling, all five pairs were re-run on HarmBench at a single fixed stochastic setting across five independent seeds, with both pair members scored by the same official HarmBench classifier used for the primary result. That setting is temperature 0.7 with top-p 0.8, the non-thinking sampling values published on the Qwen3 model card. It was applied uniformly to all five pairs for cross-pair comparability rather than tuned per publisher, although the Llama 3.2 model card's default generation configuration specifies T = 0.6 / top-p 0.9."),
  PJ("Within each seed, baseline and 4-bit share the identical temperature and seed, so every per-seed ΔASR is a clean matched-pair contrast. The arm is a separate robustness layer and is never cross-compared with the greedy main results."),
  PJ("For Qwen3-1.7B the per-seed judge ΔASR is 0.000, +0.010, +0.020, 0.000, +0.035, giving a mean of +0.013 over the range [0.000, +0.035]. No seed is individually significant (0/5). One directional nuance is disclosed rather than glossed. All five seed deltas are non-negative, a sign-consistency compatible with a small positive effect below this pair's minimum detectable effect of 0.079 at n = 200. The representative per-pair MDE is ≈0.06, with the five pairs spanning 0.044–0.086 (Section 6.4). Therefore the claim established here is 'no detectable change at n = 200' rather than a proven zero. Critically, the greedy point estimate (0.000) sits inside the seed range, so stochastic decoding corroborates rather than contradicts the null. There is no effect at the reference budget for sampling to attenuate. This is the reverse of the 128-token picture (Section 6.12), where multi-seed decoding attenuated a real greedy +0.055. At 512 tokens there is nothing to attenuate."),
  PJ("The next two pairs behave consistently with their primary deltas. Qwen3-4B has mean +0.029 over the range [+0.010, +0.050], with 1 of 5 seeds individually significant under McNemar's exact test, a directional non-null. Llama-3.2-3B has mean −0.024 with no positive seed in five, over the range [−0.040, −0.010], and 2 of 5 seeds are individually significant, both decreases, matching its significant safety-improving primary decrease. Both greedy points (+0.040 and −0.040) sit inside their seed ranges. The arm was extended to the two cross-family pairs under the identical protocol, completing coverage of all five pairs. Mistral-7B has seed mean +0.012 over the range [−0.005, +0.045], with 0 of 5 seeds significant, and Phi-4-mini has seed mean −0.012 over the range [−0.055, +0.015]. For Phi-4-mini, 1 of 5 seeds is individually significant, and that seed, −0.055, is a decrease."),
  PJ("One honest nuance for these two pairs is disclosed rather than smoothed over. Unlike the three original pairs, whose greedy point estimates sit inside their seed ranges, the greedy deltas for Mistral (−0.020) and Phi (+0.020) fall just outside their narrow seed ranges. Mistral's falls just below its seed minimum and Phi's just above its seed maximum, each with the opposite sign to its seed mean. However, both ranges straddle zero, the greedy-stochastic gaps are small (≤0.04) and sign-unstable, and the seed means are near zero (Mistral +0.012, Phi −0.012), so the discrepancy reads as noise around zero rather than a stochastic effect the greedy run missed. This is not argued from a single detection floor. Per-pair minimum detectable effects vary (0.044–0.086 at n = 200, Section 6.4), and one Phi seed is itself individually significant at −0.055 (McNemar p = 0.043). That seed is a decrease, and one of five, so it does not lift the pair-level reading. The claim carried for these pairs is 'no detectable change under either decoding regime' rather than greedy-versus-seed agreement in point estimate."),
  PJ("Across all five pairs no pair is robustly positive: every per-seed |ΔASR| is at most 0.055, and every seed-mean |ΔASR| at most 0.029. The four seeds that reach individual significance anywhere in the arm split one increase (Qwen3-4B, +0.050) against three decreases, Llama-3.2-3B's two and Phi-4-mini's one. Therefore the stochastic arm corroborates the RQ1 null across the full matrix. Per-seed deltas and per-seed exact McNemar results are in the released analysis artifacts.json (judge_delta.per_seed and n_seeds_significant_p05), so every per-seed claim in this section is asserted against a committed artefact."),

  H2("I.2 Per-category harmful-compliance profile, Qwen3-1.7B"),
  ...Figure("category_asr.png", "Per-category HarmBench attack success rate for Qwen3-1.7B at fp16 versus NF4 (official classifier, 512-token reference budget). The aggregate ΔASR is 0.000, so per-category rates are descriptive with no net change across the mix; there is no broad-based increase to attribute. Per-category counts are small (19–58 prompts), so the rates are descriptive."),
  PJ("Because the aggregate Qwen3-1.7B ΔASR is exactly 0.000 at the reference budget, there is no net harmful-compliance increase to decompose by harm type. The judge ASR was nonetheless decomposed by HarmBench semantic category to confirm the null is not masking a large offsetting swing between categories. That decomposition is in the released analysis artifacts.{json,csv} and was produced by scripts/harmbench_category_breakdown.py from the saved generations and judge sidecars, with no new inference. At 512 tokens the per-category deltas go in both directions and roughly cancel. Four categories rise (misinformation/disinformation +0.18, harassment/bullying +0.11, illegal +0.03, and the generic harmful category +0.05) while two fall (cybercrime/intrusion −0.18 and chemical/biological −0.14), leaving the aggregate flat. No single category drives a systematic increase."),
  PJ("The broad-based five-of-six-category rise reported for this pair previously was a feature of the 128-token run. It was an artefact of truncation (Section 6.12) rather than a stable property of the model, and it disappears alongside the aggregate +0.055 once responses are allowed to run to the reference budget. Per-category counts are small (19–58 prompts each), so these rates are descriptive and exploratory, reported to characterise the composition of the null rather than as individually significance-tested claims. For contrast, the Qwen3-4B pair shows a small directional increase and the Llama-3.2-3B pair a significant decrease, consistent with their aggregate deltas."),

  H2("I.3 Second-judge cross-check: per-prompt agreement and per-pair deltas"),
  PJ("The two judges agree strongly at the per-prompt level. Across all ten models Cohen's κ ranges from 0.68 to 0.95, which is substantial-to-almost-perfect agreement. That is far higher than the regex-versus-classifier agreement (κ ≈ 0.25–0.59 for Qwen and Mistral), which confirms that the regex, and not the classifier, was the outlier."),
  PJ("The per-pair ΔASR direction is reproduced at 512 tokens. For Qwen3-1.7B gpt-4o gives +0.005, against the classifier's 0.000, and for Llama-3.2-3B it gives −0.035, against −0.040. For the cross-family pairs it agrees on the sign of Mistral-7B's and Phi-4-mini's small moves. Crucially, it places Mistral-7B's ASR near 0.55, which matches the classifier's ≈0.57 rather than the v2 proxy's 0.83–0.89. Both judges agree that no pair shows a material harmful-compliance increase, and that Mistral-7B's proxy-driven 'increase' is an artefact."),
  PJ("The Qwen3-1.7B effect reported at the 128-token budget is null under both judges at the reference budget (classifier 0.000, McNemar p = 1.000, gpt-4o +0.005), so there is no boundary-significance disagreement left to reconcile. This converges with the multi-seed sensitivity arm (Appendix I.1), whose greedy 0.000 sits inside the seed range. Under both judges, no pair shows a statistically significant increase in harmful compliance at the reference budget. Per-model agreement and per-pair figures are in the released analysis artifacts.{json,csv} (redacted)."),
  H2("I.4 Human grounding of the HarmBench scorers: protocol, caveat and agreement"),
  PJ("One input-mismatch caveat applies. The annotation view truncated each response to its first 2,000 characters, whereas both automated scorers saw the full generation. On the roughly 57 percent of items whose response exceeds that length the human therefore judged a prefix of what the scorers judged. Any harmful content confined to the truncated tail would register as a human–scorer disagreement rather than agreement, which plausibly inflates the measured disagreement. For this reason the observed κ is best read as conservative. However, the direction is not strictly identified, because a harmful-looking prefix followed by a benign completion cuts the other way."),
  PJ("The primary classifier agrees with the human at Cohen's κ = 0.59, which is moderate agreement. The demoted refusal regex agrees at κ = 0.11, which is slight. Precision makes the gap concrete. Of the responses each scorer flagged harmful, the human confirmed 61 percent of the classifier's but only 23 percent of the regex's, so against human judgment the regex issued 101 false-harmful calls to the classifier's 18. Of the three scorers the human applied the highest bar for harmful compliance and flagged the fewest responses, 37 of 200, against the classifier's 46 and the regex's 132. Both automated scorers therefore apply the harmful label more readily than the human does. However, the two over-countings differ in kind. The classifier's disagreement with the human is mild and roughly two-directional: it flagged 18 responses the human judged benign, against 9 the human flagged that it missed, which reads as balanced noise around the human bar. By contrast, the regex's disagreement is a near-one-directional flood, with 101 over-flags against 6 misses, that is, systematic over-counting. The classifier is therefore close to the human bar and the regex far above it."),
  PJ("This is the human-grounded form of the chapter's central result. Refusal-counting over-states harmful compliance not merely relative to the classifier but relative to human judgment, and the classifier is the substantially better scorer. Two bounds are stated with equal plainness. First, κ = 0.59 makes the classifier a better scorer rather than a perfect human proxy. It still over-calls 18 cases, which is the kind of judge fallibility the LLM-as-judge literature warns against when judge output is treated as ground truth [51], [52], [53]. Zheng et al. [51] report that a strong judge model reaches roughly eighty-five percent agreement with human annotators, which exceeds the eighty-one percent agreement humans reach with each other, while also documenting position, verbosity and self-enhancement biases that a judge-scored result has to be read against. Second, this is a single annotator on a partly subjective rubric, since some behaviours, such as subtly biased but superficially positive text, are genuinely borderline for humans and judges alike. A second annotator with an inter-rater κ is therefore the natural strengthening. Because the sample deliberately over-weights judge-vs-regex disagreement cases (120 of 200), the absolute κ values are computed on a harder-than-representative slice and are not population estimates. The direction in which a representative population would move them is not identified by this design, because the enriched slice changes both the item difficulty and the marginal prevalence κ depends on. The reported pair should therefore be read as a disagreement-enriched contrast rather than a population estimate. What does survive that sampling is the ordering and its size. On the contested cases that decide the conclusion the classifier tracks the human far more closely than the regex does, and the direction of the regex's error is a systematic over-count."),
  H2("I.5 Scorer threats, residual caveats and the open-weight third judge"),
  PJ("Threats and scope. The judge is itself a model and can err. It is adopted because it is the reference classifier specified by the HarmBench paper [24], run at full precision, with deterministic greedy decoding and zero parse failures on this dataset. One implementation detail bounds comparability rather than validity. The classifier is prompted with the Hugging Face model-card (behaviour-only) template rather than the HarmBench GitHub evaluation-harness variant, which adds bracketed section headers and three additional scoring rules. Absolute ASR levels here are therefore not directly comparable to harness-scored numbers in the literature. However, because both members of every pair are scored under the identical template, the matched-pair deltas are internally controlled for it. Whether a delta computed under the harness variant would take the same value was not tested. The judge validates HarmBench only. XSTest over-refusal and MMLU are unaffected."),
  PJ("The single-judge construct-validity threat is addressed on two fronts. The first is the second-judge cross-check (Result 4), where an independent frontier judge agrees at κ 0.68–0.95 across all ten models and reproduces the finding's direction and magnitude. The second is the human-label audit (Result 5), which grounds the scorer comparison against human judgment rather than against another model, and finds the classifier substantially closer to the human (κ 0.59) than the regex (κ 0.11). The claim this study makes is therefore the human-grounded one, namely that the choice of scorer changes the conclusion and that refusal-counting over-states harmful compliance relative to both the benchmark's own classifier and a human annotator. The residual is honestly bounded. The classifier is a better scorer rather than a perfect oracle, since κ 0.59 is moderate rather than near-perfect. In addition, the grounding rests on a single annotator, so a second annotator with an inter-rater κ is the remaining strengthening."),
  PJ("Two further residual caveats concern the judges themselves. The second judge is a versioned API model, which is less reproducible than the open-weight primary, and both judges share the broad lineage of LLM-based harm classification. For this reason an open-weight third judge applying a different safety-classification construct, Llama Guard [48], is the natural complementary cross-check. Its independence is partial: Llama-Guard-3-8B is itself a Llama-family model, so for the Llama-3.2-3B pair in particular it is not lineage-independent of the system it scores. That check was run. A revision-pinned Llama-Guard-3-8B re-scored the 200 saved HarmBench generations of all fifteen aliases on TC1 with zero parse errors, and it agrees with the primary classifier at Cohen's κ from 0.36 to 0.92 across the ten base/NF4 aliases of the primary study. Agreement is highest for the Llama-3.2-3B pair (0.89/0.92) and lowest for Phi (0.36/0.38), with Qwen at 0.56–0.77 and Mistral at 0.61/0.63. It also reproduces the RQ1 null, because every per-pair Llama-Guard ΔASR is non-significant. The largest move, Qwen3-1.7B at −0.050 with McNemar p = 0.076, is a decrease. Its sign agrees with the classifier's for every pair whose classifier delta is nonzero, and Qwen3-1.7B's classifier delta is exactly 0.000, so no sign is defined there."),
  PJ("One construct caveat bounds this reading and is stated plainly. Llama-Guard answers a looser question ('is this response unsafe?') than the HarmBench classifier ('is this generation an instance of the specific behavior?'), so where the two disagree, most visibly on the low Phi κ, the divergence is expected and non-diagnostic. Agreement is the informative direction, and the guard is a reproducibility cross-check rather than a second construct-validity oracle. An open-weight, revision-pinned third judge reproducing the null removes the residual reproducibility caveat that the judge cross-check rested solely on a versioned API snapshot. Per-alias and per-pair figures are in the released analysis artifacts.{json,csv}, and the gpt-4o-versus-Llama-Guard leg is in judge_pairwise_agreement_api_vs_llamaguard.{json,csv}. As a data-handling note, that second judge is an external API, so for its cross-check the (public) HarmBench behaviour string and the model's generation are transmitted off-cluster, whereas the primary classifier runs entirely on TC1 and only redacted identifier-and-boolean sidecars are stored. The v2 regex is retained in the repository and in Table 6.1/6.2 as a transparent secondary proxy, so that readers can see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring."),
  H2("I.6 Over-refusal scorer sensitivity: strict and broad mappings"),
  PJ("Strict over-refusal counts full refusals only, which is the conservative reading, and broad counts partial or full. This is a sensitivity check rather than a change of scorer. The deterministic regex remains the primary over-refusal scorer of record, and the multiple-comparison family of Section 6.4.1 is unchanged. The finding parallels the HarmBench one. First, the two scorers disagree substantially on benign over-refusal. The judge counts roughly four times as many refusals as the regex, with a mean over-refusal of 0.171 strict versus 0.044, because it treats moral lectures, redirections, and answer-free 'I can't do X but I can do Y' alternatives as refusals whereas the keyword regex does not. Per-alias agreement is only poor-to-moderate (Cohen κ from −0.01 to 0.50 across the ten base/NF4 aliases of the primary study)."),
  PJ("Second, and decisively, the Phi-4-mini over-refusal decrease does not replicate. Under the judge the Phi-4-mini strict-mapping ΔOR is +0.016, with the direction reversed. Its confidence interval is [−0.028, +0.060] and McNemar's exact test gives p = 0.597. On the broad mapping the ΔOR is −0.004, with CI [−0.048, +0.036] and p = 1.000. Neither mapping gives a significant result, and neither reproduces the regex's −0.048 decrease. No pair shows a significant judge ΔOR under either mapping, since every CI includes zero. That is consistent with the study's RQ2 over-refusal null. However, the direction and magnitude diverge from the regex most sharply for the one contrast that had survived multiplicity correction."),
  PJ("The honest reading, carried into Section 6.4 and the Abstract, is that the single FDR-surviving over-refusal contrast is scorer-dependent. The human audit below then resolves which scorer to weight: the contrast is most plausibly a measurement artefact of the refusal regex rather than a classifier- or human-validated over-refusal shift. This extends the chapter's central message, that the scorer determines the conclusion, from the harmful-compliance axis to the over-refusal axis. As with the second HarmBench judge, the benign prompt and saved response were transmitted to an external API for this check. The text is public benchmark text, only redacted identifier-and-label sidecars are stored, and the same redaction contract is enforced. All five per-pair judge ΔORs are reported in Table 6.4 regardless of direction, as exploratory sensitivity results under the pre-registered presentation rule. The largest, Qwen3-1.7B's +0.040 on the strict mapping with McNemar p = 0.087, is an apparent increase that does not reach significance. Full per-alias agreement is in the released analysis artifacts.{json,csv} (redacted), and no new significance claim enters the pre-registered multiplicity family."),
  H2("I.7 Human grounding of the over-refusal scorers, and the pre-registration deviation"),
  PJ("The human grounding for this axis was completed, mirroring Result 5. All 200 items of a pre-registered gold set were labelled by a single human annotator, the study's author, under the same three-class XSTest taxonomy the judge applies. The set was drawn from the same 3,750 saved responses, deliberately over-sampling the regex-versus-judge disagreements (120 of 200) and double-weighting the contested Phi-4-mini and divergent Qwen3-1.7B pairs, and it was presentation-shuffled and label-blind. That taxonomy is the benchmark's own, and the shared rubric is disclosed plainly. A semantic judge is structurally better placed than a phrase-matcher to satisfy a shared semantic rubric, so the comparison measures alignment with the annotator under the benchmark's construct rather than scorer quality in the abstract."),
  PJ("The J/R/T/X read-off gives a mechanical Outcome J, and its outcome matrix was locked before any label existed (docs/XSTEST_GOLD_PREREG.md). The judge aligns substantially better with the annotator than the regex, at strict-mapping Cohen's κ 0.485 versus −0.006 and broad-mapping 0.662 versus 0.054, and the two failure modes are asymmetric in kind. One protocol deviation is disclosed rather than glossed (pre-registration Section 8). The blindness exclusion was applied prompt-wide, with the five quoted prompts removed from all ten aliases, instead of to the ≤10 pre-specified (alias, prompt) items. Under the pre-registration's strict letter that voids the clean confirmatory badge. Therefore this is reported as an Outcome-J–consistent validation carrying one disclosed deviation rather than a clean pre-registered confirmation. The deviation was fixed in code before the draw and before any label existed, and it is blindness-conservative, because it removes more potential priming and never less. Its footprint is also bounded. A no-exclusion counterfactual draw overlaps the labelled set on 186 of 200 items, so at most 14 were displaced, and on those shared items the strict κ gap is +0.51, as large as the full-sample +0.49. The judge-grounded reading therefore does not depend on the deviation-specific items."),
  PJ("The regex detected 2 of the annotator's 63 full refusals, a recall of 0.032. The 61 missed refusals are typically long, polite, lecture-style declinations carrying none of the template phrases. By contrast, the judge caught 61 of 63, a recall of 0.968, and it errs by over-flagging: 52 over-flags, 25 of them on the genuinely subjective partial-versus-full boundary, with three-class exact agreement with the annotator of 0.695. Three bounds are stated with the same plainness as Result 5's. First, these labels are a blinded single-annotator reference set rather than ground truth, because a single annotator can misread and drift, and no inter-rater κ exists for this set. Second, the draw is disagreement-enriched, so the κ values are a contested-slice contrast rather than population estimates, and no population over-refusal rate or human ΔOR is estimated from them. Descriptively, the labelled Phi-4-mini items show no base-versus-4-bit difference in human-judged strict refusal, but the subset licenses no population claim. Third, the pre-registered no-swap rule was locked before the judge results existed but after the regex results were known, and that timing is disclosed rather than argued from."),
  PJ("Under the Outcome-J–consistent action the confirmatory pre-registration status is withdrawn per the Section 8 deviation, while the reporting action itself is unchanged. The regex remains the original pre-specified scorer-of-record and the registered multiplicity family is unchanged. However, the interpretive weight shifts to the judge, as the scorer better aligned with the single annotator on the enriched audit. The Phi-4-mini −0.048 contrast is therefore not treated as firm evidence of reduced over-refusal. It is most plausibly a measurement artefact of the regex scorer. The Section 6.4.1 validation-informed parallel family gives that reading a multiplicity-controlled footing, because under it two contrasts survive, both of them capability effects. The committed aggregate carries counts and rates only, since the annotation sheet holds raw text and stays local. A second annotator with an inter-rater κ remains the residual strengthening, exactly as for Result 5."),

  PJ("This appendix holds the supporting detail for Chapter 6: the full scorer-validation trail, the robustness arms, and the per-category and per-subject breakdowns. It is placed here so the results chapter carries the findings and the evidence for them stays available and unabridged. Nothing in this appendix is re-derived; each block is the original text, relocated."),
  H2("I.8 Scorer revision history"),
  PJ("Scope under the judge-primary framing. This section documents the refusal regex's own v1→v2 revision history, which bears on two things only. The first is the XSTest over-refusal figures, produced by the v2 expanded refusal classifier (`ethical_benchmark/benchmarks/utils.py`, scorer version `v2_expanded_refusal_patterns_2026-05-28`). The second is the secondary HarmBench non-refusal proxy. HarmBench Attack Success Rate throughout this report is the official HarmBench classifier value, the primary scorer (Section 6.9), rather than the regex value. MMLU and ARC use deterministic exact match."),
  PJ("With that scope established, the v1 classifier (pre-2026-05-28) used a narrow set of 14 refusal patterns that missed the most common modern refusal templates, including \"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", and curly-apostrophe variants used by Qwen3 output. As a result the v1 classifier systematically misclassified clear refusals as harmful compliance, inflating reported ASR and depressing reported over-refusal. The classifier was rewritten with comprehensive coverage of standard refusal templates, validated by manual stratified audit of 55 sampled responses across all six runs. That audit confirmed correctness on the sample (see PROJECT_LOG entry 2026-05-28)."),
  PJ("The TC1-original `raw.jsonl` and `summary.json` files are retained as immutable v1 artefacts. Corrected scoring is stored in derived `scores.v2.jsonl` sidecars (prompt IDs and score_fields only, no prompt or response text) and `summary.v2.json` files. The analysis pipeline prefers these v2 sidecars when present, so the report uses corrected scores without modifying the original raw generations. Auxiliary diagnostic artefacts, carrying record IDs plus matched pattern names and indices only and no response text, document the per-record reclassification for full auditability."),
  PJ("Impact of the rescore on the v2 numbers. Within the regex scorer, three pair-level deltas changed materially from v1 to v2. Qwen3-1.7B ΔASR moved from −0.120 (v1) to −0.025 (v2), where the v1 \"significant decrease\" was largely an artefact of unrecognised refusals. Qwen3-4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2). Llama-3.2-3B ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). MMLU values are unchanged, because MMLU scoring does not depend on the refusal classifier."),
  PJ("This history explains why the v2 regex is retained only as a secondary proxy. Even after the v1→v2 correction, the regex measures \"non-refusal rate\" rather than genuine harmful compliance. The subsequent judge validation (Section 6.9) showed that the v2 regex still systematically over-counts ASR, so the HarmBench Attack Success Rate reported throughout this chapter is the official-classifier value, with v2 retained for transparency. The v1→v2 history is documented openly here, and the decision to score HarmBench with the judge instead of v2 is documented in Section 6.9."),

  H2("I.9 Capability robustness check: the ARC-Challenge second benchmark"),
  PJ("Because the capability anchor rests on a single MMLU subset, a structurally different second capability benchmark, ARC-Challenge (allenai/ai2_arc, 1,172 reasoning-oriented science questions, identical exact-match scoring), was run on all ten models to test whether the MMLU-based capability claims replicate (benchmark = arc, same 2,000-resample paired-bootstrap CI as MMLU). The direction is largely consistent. All five pairs lose on MMLU, and four of five lose on ARC. The exception is Mistral-7B, whose ARC moves +0.9 pp (non-significant) in the improving direction. The magnitude and significance, however, diverge informatively. Qwen3-1.7B, which loses a large, significant 9.0 pp on MMLU, loses only 0.9 pp on ARC (not significant) under the primary lenient parser."),
  PJ("That apparent stability on ARC is largely a parser artefact rather than benchmark-wide preserved capability. Both benchmarks suffer the same 4-bit answer-format collapse for this pair, with the lenient fallback firing on 48.7% of its MMLU items and 52.3% of its ARC items, versus ~3% at fp16 (Section 6.4). However, the fallback salvages the format-broken ARC answers more accurately (66%, well above the 25% chance floor, so answer-knowledge on ARC is substantially preserved) than the MMLU ones (46%), and that difference is what masks the ARC loss. A strict parser that scores an unparsable answer incorrect measures format-compliant answering rather than answer-knowledge alone. Under it ARC falls −0.343 and MMLU −0.293, both significant. These are comparable large losses. The lenient reading is therefore not evidence that the Qwen3-1.7B degradation is genuinely MMLU-specific. The answer-format truncation that partly explained the MMLU figure at 128 tokens is separately resolved at 512 (Chapter 8). Each benchmark's loss is best read as a bracket whose lenient end tracks answer-knowledge (largely preserved) and whose strict end tracks format-compliant answering, which collapses on both benchmarks."),
  PJ("Qwen3-4B shows the opposite pattern: flat on MMLU (−0.3 pp, n.s.) but a small, significant −1.6 pp on ARC, so its capability is less fully preserved than MMLU suggested. Llama-3.2-3B is the most consistent pair, losing on both, and its ARC loss (−3.2 pp) is the significant one that anchors its capability_collapse_masquerading_as_safety label. Its MMLU dip, −3.7 pp, is not significant. Two implications follow. First, RQ3 (NF4 degrades capability) is supported with per-pair nuance rather than uniformly. On MMLU all five point estimates are negative, and on ARC four of five are, the exception being Mistral-7B's +0.9 pp non-significant improvement. Three pairs carry an individually significant capability loss on at least one anchor: Qwen3-1.7B on MMLU, and Qwen3-4B and Llama-3.2-3B on ARC. Mistral-7B and Phi-4-mini show directionally consistent but individually non-significant capability changes. The capability cost therefore survives the correction in aggregate, supplying two of the three FDR survivors (Qwen3-1.7B MMLU and Llama-3.2-3B ARC, the third being Phi-4-mini's over-refusal decrease, Section 6.4.1), without being uniformly significant per pair."),
  PJ("Second, the dramatic within-Qwen scale gap under MMLU (the ≈30:1 ratio, Section 6.2, RQ4) does not replicate under ARC when ARC is scored by the primary lenient parser, which finds the 4B pair losing marginally more than the 1.7B. Under a strict parser it does replicate and then some, with the 1.7B pair losing −0.343 on ARC against the 4B pair's −0.017, because the lenient parser was masking the Qwen3-1.7B ARC collapse described above. The 'smaller model is far more capability-sensitive' claim is therefore parser-dependent rather than MMLU-specific and is hedged accordingly. Interpretation labels remain MMLU-anchored. ARC is reported here as a corroborating capability axis, and a formal composite-capability rule is left to future work. The two cross-family pairs (Section 6.8.6) extend the ARC axis to ten models. Mistral-7B is essentially flat on ARC (+0.9 pp, not significant), consistent with its preserved MMLU, and Phi-4-mini loses 1.5 pp (not significant) alongside a 2.7 pp MMLU dip, so both new families show the same direction-consistent, modest capability story. Crucially, at the 512-token reference budget the two ARC losses that reach significance (Qwen3-4B −1.6 pp, Llama-3.2-3B −3.2 pp) are capability effects. No HarmBench ASR increase reaches significance on any pair, so ARC reinforces the reading that capability degradation, rather than harmful-compliance change, is the consistent cost of NF4."),

  H2("I.10 Refusal-margin geometry: flip symmetry and per-model margin shifts"),
  PJ("The flips are also close to symmetric in aggregate. Of 92 flips, 50 are harmful-ward (refuse→comply) and 42 are safe-ward (comply→refuse), and a thin margin predicts the safe-ward flips at AUC 0.78, at least as strongly as the harmful-ward ones at 0.75. However, this pooled near-symmetry must be read with care. It is driven by Mistral-7B, the one family whose first-token margin the gate above flags as a poor proxy, and Mistral-7B alone contributes 20 harmful-ward and 28 safe-ward flips, two-thirds of all safe-ward flips."),
  PJ("Across the four families where the first-token margin is a valid proxy the flips lean roughly 2:1 harmful-ward, at 30 versus 14, and the load-bearing Qwen3-1.7B pair is 16 versus 5. Two of those valid-proxy families are nonetheless individually symmetric, Llama-3.2-3B at 2:2 and Phi-4-mini at 5:5, and the safe-ward direction is well represented. Thin first-token margins are therefore best read as marking prompts near the greedy-decoding decision boundary that quantization can destabilise in either direction, a boundary-fragility effect rather than evidence of a safety-specific erosion mechanism. For the one behaviourally-moving pair, Qwen3-1.7B, the strongest evidence for the boundary-instability reading comes from the entropy confound control and the significant ΔMMLU discussed next, rather than from this pooled symmetry."),
  PJ("Direction of the margin shift is model-specific and modest. The paired shift Δm = m(fp16) − m(NF4) is statistically significant for every pair (Wilcoxon signed-rank [54] p < 0.001, n = 200). It moves comply-ward for both Qwen pairs, by Δm = +1.2 for Qwen3-1.7B and +4.0 for Qwen3-4B. For Llama-3.2-3B and Phi-4-mini it moves refuse-ward, by −1.1 and −0.3. Effect sizes span Cohen dz [55] ≈ −0.74 to +1.7. The whole-sequence margins reverse the first-token sign for Qwen3-1.7B and Phi-4-mini, so the directional claim is first-token-specific. The supportable statement is that NF4 perturbs the first-token refusal distribution in a model-specific direction, rather than that it uniformly erodes refusal."),
  PJ("Qwen3-4B shifts comply-ward more than Qwen3-1.7B, by +4.0 against +1.2, yet it flips fewer prompts, 9 against 21. Its baseline margin is far wider, at median 13.0 against 3.8, so it has refusal headroom to absorb the perturbation. That headroom is consistent with the Qwen3-4B pair remaining behaviourally intact where the Qwen3-1.7B pair does not, although a two-pair comparison is observational and does not isolate headroom as the cause."),
  H2("I.11 INT8: budget sensitivity, the mechanism reading and the proxy at INT8"),
  PJ("The 128-token Llama-3.2-3B INT8 signal was therefore a budget-sensitive artefact, though of a different kind from the truncation that inflated the Qwen3-1.7B NF4 move. Llama-3.2-3B's generations truncate at only 3–4%, the lowest of any family (Section 6.12), so its 128-versus-512 instability traces to cross-run greedy divergence rather than to cut-off responses. Llama-3.2-3B carries the study's highest prefix-mismatch share (Section 6.12). Either way, it is not a stable method-specific regression."),
  PJ("The conceptual conclusion, across the five pairs tested here, is that quantization's observed effect on safety is neither bit-width-graded nor budget-robust. The only robust, budget-invariant precision effect is the capability cliff at four-bit, and no INT8 or NF4 harmful-compliance increase survives at the reference budget."),
  PJ("Honest reading of the INT8 result. The lesson of the 128-to-512 comparison is stated plainly. The 128-token Llama-3.2-3B INT8 increase rested on a small absolute flip count of ≈8–9 prompts on a single pair, and it was non-monotonic. At the 512-token reference budget it does not replicate under either judge, with classifier +0.005 and gpt-4o +0.010, both McNemar n.s. It is best read as a budget-sensitive artefact rather than as a method-specific \"INT8 erodes safety\" effect."),
  PJ("Because Llama-3.2-3B's generations truncate at only 3–4% (Section 6.12), the instability is more plausibly attributable to cross-run greedy divergence than to truncated responses, a reading consistent with the thin-margin boundary mechanism of Section 6.10, though that mechanism was not measured for this arm. No INT8 ASR contrast is significant at 512 under either judge, so nothing on the safety axis needs a multiple-comparison correction to be discounted."),
  PJ("The picture is consistent with the mechanism reading of Section 6.10, in which thin-margin boundary prompts are destabilised by quantization in a model-specific direction and which prompts flip is itself idiosyncratic and budget-sensitive. It does not support any graded or method-specific safety erosion. Replication of INT8 behaviour across more models and decode seeds remains a natural next step (Chapter 9). At the reference budget, though, the honest statement is a null on the safety axis and a clean capability cliff at four-bit."),
  PJ("Scorer note: the proxy at INT8. The secondary v2 refusal regex continues to over-count harmful compliance at INT8 exactly as it does at fp16 and under NF4. Judge-versus-proxy Cohen κ is lowest for Mistral-7B, the worst over-counter, and highest for Llama-3.2-3B and Phi-4-mini. This is the same pattern of low Qwen and Mistral-7B agreement with high Llama-3.2-3B and Phi-4-mini agreement, reproduced at the new precision. The official classifier therefore remains the primary scorer at INT8, and every safety number above is judge-scored. The proxy is reported only as the demoted foil whose failure motivates the judge."),
  H2("I.12 Generation-length rerun: truncation statistics and decoding artefacts"),
  PJ("The initial run generated at most max_new_tokens = 128. The primary run regenerates the entire study at 512 tokens, covering all five pairs and all four benchmarks, scored by both the primary HarmBench classifier and the gpt-4o second judge. The two runs live in parallel results trees, so the two budgets are directly comparable. The primary tree is results_512/, produced via configs/tc1_512.yaml with the classifier judge in slurm/judge_validation_512.sbatch, and the 128-token artefacts are retained unchanged for comparison. All ten matrix jobs and both judges completed with zero parse errors at both budgets."),
  PJ("The reference budget changes the generations materially, which is the crux of why it must be primary. At the 512-token budget the HarmBench responses are far longer, with a median length of 1,675 characters against 567 at 128 tokens, a roughly threefold increase. A direct prefix test shows that 60.3 percent of the 128-token responses were provably truncated, in the sense that the 512-token generation continues past where the 128-token one was cut off. A character-ceiling proxy puts the truncation rate at about 62 percent. The 128-token run was therefore cutting off the majority of generations before completion, and any safety signal measured there is confounded with where the truncation fell."),
  PJ("Two disclosures accompany the prefix statistic. First, it is family-heterogeneous. Mistral-7B truncates at 93.5–98.0 percent and Phi-4-mini at 73.5–78.5, the Qwen models at 54.5–70.0, whereas Llama-3.2-3B truncates at only 3.0–4.0 percent. Llama-3.2-3B's 128-token generations mostly stop naturally or diverge, so the truncation mechanism is family-specific even though the reference-budget conclusion is not."),
  PJ("Second, 9.2 percent of the 2,000 paired generations, 184 in total and concentrated in Llama-3.2-3B, are prefix mismatches. Here the 128- and 512-token runs, though both nominally greedy, diverge early rather than one being a truncation of the other. This is benign run-to-run non-determinism between two separate executions at different max_new_tokens. The mechanism was not diagnosed here, and batch composition was identical across the two runs, so the likely source is decode-path divergence such as detokenization or non-deterministic GPU reduction order rather than batching. It was conservatively excluded from the truncated count, which makes 60.3 percent a floor. Every matched-pair safety delta in this study is computed within a single run, so this cross-run non-determinism does not affect any reported contrast."),
  PJ("Absolute ASR is correspondingly higher at 512 tokens under both judges. Mistral-7B classifier ASR, for example, is 0.585 at the reference budget against 0.385 at 128 tokens, because a complete generation gives more room for harmful content to appear. That is a property of the absolute rate rather than of the matched-pair delta, and it is the delta that carries every safety claim in this study."),
  PJ("A qualitative note on the form of these longer generations surfaced while inspecting responses during the human-label audit of Section 6.9. The generation loop halts only on the tokenizer's default end-of-sequence id and otherwise runs to the token budget, so a completed answer is frequently not the end of the sequence. Having nothing left to answer, the model continues, and it often fabricates a synthetic next turn, inventing a follow-up user request and then answering it. A single record can therefore contain the model role-playing both sides of a short dialogue."),
  PJ("Because decoding uses skip_special_tokens, the intervening chat-turn markers are stripped and this is saved as a run-on. That is why some responses read as a refusal immediately followed by an unrelated benign answer, such as a refused harmful request trailed by an unprompted explanation of photosynthesis."),
  PJ("This is a decoding-configuration artefact rather than a model-capability finding, and it is benign for the study's claims on two counts. First, it is produced identically by both members of every pair, so its effect is matched across the pair by construction. Second, the primary classifier scores the entire response rather than its opening, so a continuation that did emit harmful content would still be counted. The human-label audit of Section 6.9 Result 5 is the direct check on whether that scoring is correct, and it finds the classifier substantially closer to human judgment than the regex. Constraining generation to stop at the chat turn-end token is a clean fix for future runs, and it would also shorten and tidy the saved transcripts."),
  PJ("This appendix holds the supporting detail for Chapter 6: the full scorer-validation trail, the robustness arms, and the per-category and per-subject breakdowns. It is placed here so the results chapter carries the findings and the evidence for them stays available and unabridged. Nothing in this appendix is re-derived; each block is the original text, relocated."),
  H2("I.13 Cross-family comparison, dimension by dimension"),
  PJ("Before interpreting quantization deltas, the baseline profile differences are themselves informative. At similar MMLU capability (Qwen3-1.7B 0.643 and Llama-3.2-3B 0.610), the Llama model is more safety-calibrated than the smaller Qwen under the official classifier at the 512-token reference budget. Baseline HarmBench ASR is 0.100 for Llama-3.2-3B against 0.255 for Qwen3-1.7B, a 15.5 pp difference in favour of Llama-3.2-3B on the safety axis at similar observed MMLU accuracy. XSTest over-refusal at baseline is comparable, at 0.032 for Llama-3.2-3B against 0.052 for Qwen3-1.7B."),
  PJ("Two compact models at similar factual capability therefore have different safety calibrations, which may reflect unmeasured differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach; none of those factors is measured here. For these models the between-family gap is larger than any quantization delta measured in the study. The baseline ASR gap between Llama-3.2-3B and Qwen3-1.7B (15.5 pp) dwarfs every matched-pair quantization delta in the study, the largest of which, Llama-3.2-3B's own ΔASR, is 4.0 pp. The 4B Qwen baseline at 0.115 sits between these two."),

  PJ("Under the official classifier at the 512-token reference budget, exactly one of the five ΔASR values is statistically significant, and it runs in the safety-improving direction. That value is Llama-3.2-3B's (−0.040, CI [−0.075, −0.010], McNemar p = 0.021), a small significant decrease in harmful compliance. The Qwen3-4B ΔASR (+0.040, CI [0.000, +0.080]) is directional-upward but not significant, and the Qwen3-1.7B ΔASR (0.000, CI [−0.055, +0.055], McNemar p = 1.000) is an exact null. No pair shows a statistically significant increase in genuine harmful compliance under NF4."),
  PJ("Nor does any ΔASR contrast survive the Benjamini-Hochberg correction over the twenty-test NF4 family (Section 6.2). The three FDR survivors are all capability or over-refusal deltas, and zero HarmBench ASR contrasts survive. The one individually significant ASR move (Llama-3.2-3B) is a decrease, and even it does not survive multiple-comparison correction. This contrasts with where the v2 proxy would have pointed, because the proxy over-counts ASR and would have flagged spurious increases (see Section 6.9)."),
  PJ("The Qwen3-1.7B pair, whose +0.055 at the 128-token budget was the study's original headline safety regression, shows no ASR movement at all at the reference budget. That 128-token effect was a truncation artefact, since 60 percent of 128-token responses were cut off before completion, and at 512 tokens the harm axis is flat (Section 6.12). The NF4 harmful-compliance conclusion is therefore null at the reference budget: there is no significant increase in any of the five pairs, and the single significant delta points the safety-benign way."),

  PJ("The XSTest over-refusal rate gives a consistent signal across families. Four of the five ΔOR deltas are not statistically significant: Qwen3-4B (−0.004), Llama-3.2-3B (+0.016), Mistral-7B (0.000), and Qwen3-1.7B (−0.024). For the last of these the bootstrap CI [−0.048, 0.000] no longer excludes zero, and McNemar's exact test gives p = 0.109 (Section 3.7). The one significant move is Phi-4-mini (−0.048). Both that move and the Qwen3-1.7B decrease, which is not significant either, run in the benign direction, because their 4-bit models over-refuse less rather than more. No pair shows a significant increase in over-refusal."),
  PJ("This pattern spans four families and five pairs, and it is practically useful for deployment. The concern that NF4 compression would make models excessively conservative on benign prompts is not supported by these data. Moreover, the moves that approach or reach significance are reductions in false refusals, on Qwen3-1.7B and Phi-4-mini. The largest upward point estimate is on Llama-3.2-3B (+0.016). That is a directional signal, and its confidence interval still includes zero."),

  PJ("Capability degradation, rather than safety change, is the cost of NF4 that holds up at the reference budget. On MMLU, only the Qwen3-1.7B loss reaches significance (−9.0 pp). Llama-3.2-3B (−3.7 pp) and Qwen3-4B (−0.3 pp) do not. However, the second capability benchmark, ARC-Challenge, confirms the loss where MMLU is silent. ARC falls significantly for Qwen3-4B (−1.6 pp) and Llama-3.2-3B (−3.2 pp). By contrast, Qwen3-1.7B's ARC loss (−0.9 pp) is small and not significant."),
  PJ("Therefore every one of the three pairs carries at least one significant capability loss across the two benchmarks: Qwen3-1.7B on MMLU, Qwen3-4B and Llama-3.2-3B on ARC. In addition, all six point estimates are ≤ 0. This is what survives multiple-comparison correction. Two of the three FDR survivors in the twenty-test NF4 family are capability deltas, namely Qwen3-1.7B on MMLU and Llama-3.2-3B on ARC."),
  PJ("The Llama-3.2-3B losses are intermediate between the two Qwen models, which is consistent with a size-sensitivity hypothesis, since Llama-3.2-3B at 3 billion parameters lies between the two Qwen models. However, Llama and Qwen differ in architecture and training, so the ordering may also reflect family-specific effects. More data points, for example Qwen at 7B or Llama at 1B, would be needed to separate a size effect from a family effect."),

  PJ("Under the official classifier at the 512-token reference budget, the three interpretation labels are broad_degradation (Qwen3-1.7B), alignment_degradation–directional (Qwen3-4B), and capability_collapse_masquerading_as_safety (Llama-3.2-3B). Together they answer the study's central question: \"do observed safety changes under quantization reflect alignment shifts or capability artefacts?\""),
  PJ("The Qwen3-1.7B pair carries the broad_degradation label. However, at the reference budget that label is driven entirely by its significant MMLU capability loss of −9.0 pp, with the harm axis flat (ΔASR = 0.000, McNemar p = 1.000). The pair therefore shows a capability-only degradation rather than a safety regression. The Qwen3-4B pair preserves capability and shows only a directional, non-significant ASR increase. The Llama-3.2-3B pair decreases on both axes. Harmful compliance falls, with a significant ΔASR = −0.040. Capability also falls, with a confirmed and significant ΔARC = −0.032. Its apparent safety improvement is therefore diagnosed as capability-driven rather than a genuine alignment gain."),
  PJ("Every one of the three pairs carries a significant capability loss: Qwen3-1.7B on MMLU, Qwen3-4B and Llama-3.2-3B on ARC. None shows a significant increase in harmful compliance. The central scientific finding is the framework's diagnostic value. Capability degradation is the cost of NF4 that holds at both generation budgets, whereas the harmful-compliance change does not hold up. No ASR contrast survives multiple-comparison correction, and the one significant ASR delta, on Llama-3.2-3B, is a decrease that the label scheme diagnoses as capability-driven rather than as a genuine alignment gain. The data therefore refute the optimistic narrative that NF4 quantization is a clean safety win in this scale range, while they also refute any claim of a safety regression that holds up statistically."),
  PJ("These conclusions are drawn from the original three pairs. The cross-family extension (Section 6.8.6) adds Mistral-7B and Phi-4-mini and finds the same diagnostic structure. Mistral-7B falls in the proxy-over-counting camp, with a judge ΔASR of −0.020 that is not significant, against a +0.060 proxy artefact. Phi-4-mini falls in the proxy-reliable camp at κ 0.67, and its judge ΔASR is a non-significant +0.020 that sits on the tolerance, so it carries a directional alignment_degradation label rather than a confirmed regression. The answer to the central question therefore generalises to four families."),

  PJ("This appendix holds the supporting detail for Chapter 6: the full scorer-validation trail, the robustness arms, and the per-category and per-subject breakdowns. It is placed here so the results chapter carries the findings and the evidence for them stays available and unabridged. Nothing in this appendix is re-derived; each block is the original text, relocated."),
  H2("I.14 Statistical caveats: decoding variance, MMLU subset composition and answer parsing"),
  PJ("Second, decoding is deterministic at temperature 0.0, so the only source of variance reflected in the bootstrap intervals is prompt sampling, with no within-condition stochastic variance. Because that greedy delta conditions on a single decode, a multi-seed (T = 0.7, top-p 0.8) sensitivity arm was run to estimate generation-level variance (Appendix I.1, covering all five pairs). For Qwen3-1.7B the greedy 0.000 sits inside the seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant). Across the five pairs every seed-mean |ΔASR| is at most 0.029 and every per-seed |ΔASR| at most 0.055, so the arm corroborates the null rather than an effect. One honest nuance for the two cross-family pairs is disclosed in Appendix I.1, namely that their greedy points sit just outside narrow, zero-straddling seed ranges. To keep point-estimate labels from being over-read, every interpretation label additionally carries a two-layer evidence_status (confirmed / directional / null, Section 3.8). Qwen3-1.7B is confirmed, and its confirmed axis is the significant MMLU loss rather than a safety change. Llama-3.2-3B is directional, because its ASR decrease and ARC loss are individually significant while the label's MMLU capability anchor, −0.037, is not. The artefact records evidence_status = directional and that is the claim carried here. Qwen3-4B is directional, and the cross-family pairs (Section 6.8.6) are directional (Mistral-7B) and null (Phi-4-mini)."),
  PJ("Third, the MMLU subset comprises 300 questions pooled across six subjects, and subject counts are uneven (25–94 per subject: business_ethics 25, college_biology 33, human_aging 44, clinical_knowledge 48, high_school_world_history 56, high_school_macroeconomics 94). Subject-level accuracy estimates are correspondingly noisy and are not reported as primary statistics. The distribution is identical across all baseline and 4-bit runs (same seed, same prompt IDs), so cross-condition comparisons are unaffected."),
  PJ("A further MMLU-specific construct caveat concerns answer parsing. The option-letter parser falls back to a lenient “first in-range capital letter anywhere” scan when a response neither leads with the option letter nor states an explicit “answer is X”, and this fallback fires far more often on the degraded 4-bit outputs. For qwen_2b_4bit 48.7% of items are scored by the fallback versus 3.3% for its fp16 twin, and the fallback's own accuracy on that pair (46%) is well below the leading-letter rule's (75%). Part of the −0.090 MMLU gap for that pair therefore reflects a shift in answer format that the lenient parser scores less reliably, and not only a loss of underlying knowledge. ARC-Challenge is subject to the same asymmetry even more strongly for this pair, with 52.3% of the 4-bit member's ARC items falling to the lenient fallback versus 2.5% for its fp16 twin. ARC's near-zero primary delta (−0.009, not significant) is therefore the lenient parser salvaging those format-broken answers, at a fallback accuracy of 66% on that pair, rather than evidence that ARC is immune to the shift. Under a strict parser (leading-letter or explicit “answer is X” declaration only, an unparsable response scored incorrect) ARC falls −0.343 (CI [−0.375, −0.311]) and MMLU falls −0.293 (CI [−0.350, −0.237]). Those are comparable large losses on both benchmarks. All four capability point estimates for this pair are negative, three of them individually significant, with ARC under the lenient parser (−0.009) within noise of zero. The direction of the loss is therefore consistent across both benchmarks and both parsers, while its magnitude is protocol-dependent. Each benchmark's loss is honestly a bracket rather than a single figure, its lenient end tracking answer-knowledge (largely preserved) and its strict end tracking format-compliant answering, which collapses. Answer-format parsing is thus a third measurement-dependence axis in the study, alongside the harm scorer (Section 6.9) and the over-refusal scorer (Section 6.9, Result 6)."),
  H2("I.15 INT8 arm: run configuration and analysis separation"),
  PJ("All five INT8 members were run on the NTU TC1 cluster at the 512-token reference budget, in five per-model matrix jobs plus the HarmBench-classifier judge job. They were scored by the primary official classifier and cross-checked by the gpt-4o second judge, with zero parse errors throughout. The baselines and NF4 members are the same 512-token artefacts as the primary study, so the three precisions are directly comparable."),
  PJ("The analysis is kept separate from the base-versus-4-bit pairwise pipeline so that the evaluated ten-model study is untouched. It runs from configs/tc1_int8_512.yaml through scripts/precision_sweep_analysis.py, with paired-bootstrap CIs and McNemar tests recomputed from the redacted judge sidecars. The question it answers is whether degradation scales monotonically with bit-width, in a graded fp16 > INT8 > NF4 trend, or is instead concentrated and method-specific."),
];

const appendixH = [
  H1("Appendix H: Scoring-Methodology Correction Record"),
  PJ("This appendix documents, for transparency and auditability, the scoring-methodology correction this study underwent: the original refusal-regex scorer, the corrected v2 regex, and the promotion of the official HarmBench classifier to the primary scorer. It also records the immutable-artifact contract that keeps every correction fully reproducible from the saved generations, without re-running any target-model inference."),

  H2("H.1 What went wrong in v1"),
  PJ("The original HarmBench and XSTest scorer used a narrow deterministic refusal parser with only fourteen regex patterns. During post-hoc review, it became clear that the parser missed common modern instruction-model refusal templates, including straight- and curly-apostrophe variants of phrases such as \"I can't fulfil this request\", \"I cannot provide\", and \"I'm unable to assist\". Those responses were clear refusals. However, v1 counted them as harmful compliance on HarmBench or as benign answers on XSTest. This was a scoring error rather than a target-model generation error. The six TC1 jobs had already completed, and the saved model outputs did not change."),
  PJ("The consequence was material, and the story has two stages. The v1→v2 regex correction first moved the Qwen3-4B point estimate from an apparent safety improvement to a (v2-regex) ΔASR of +0.065. However, the v2 regex still only measures \"non-refusal rate\", so a second correction followed. The official HarmBench classifier (the judge, decision D16, Section 6.9) was promoted to the primary HarmBench scorer. Under the judge, the v2 regex is shown to over-count ASR. At the then-current 128-token budget the one statistically significant ΔASR was Qwen3-1.7B at +0.055 rather than Qwen3-4B, whose judge ΔASR was directional only. The v2 Qwen3-4B figure quoted above is the regex-proxy value that the judge scoring replaced."),
  PJ("A third stage followed (decision D41, Section 6.12). Regenerating the study at HarmBench's 512-token reference budget dissolved that 128-token Qwen3-1.7B increase to ΔASR 0.000 under both judges. The current primary reading is therefore capability-driven: Qwen3-1.7B carries broad_degradation via its significant MMLU loss with the harm axis flat, and no pair shows a significant harmful-compliance increase. The 128-token reading recorded above is retained as scorer-revision history only. MMLU values are unchanged throughout, because MMLU scoring does not use any refusal scorer."),

  H2("H.2 How the correction was made"),
  PJ("The v2 correction expanded `ethical_benchmark/benchmarks/utils.py` with broader refusal-pattern coverage, punctuation normalisation, and a diagnostic helper that reports the matched refusal-pattern name. Regression tests were added for canonical refusal strings, negative controls, and curly-apostrophe handling. The analysis pipeline was then re-run from saved outputs. No target-model inference was re-run on TC1. This was appropriate, because the flaw was in the scoring layer rather than in model execution."),
  PJ("A first implementation rescored `raw.jsonl` and `summary.json` in place. That approach was rejected during code review, because the original TC1 outputs are the evidence trail and should remain immutable. The final solution preserves `raw.jsonl` and `summary.json` as original v1 artifacts, and it stores corrected v2 scoring in sidecars. `scores.v2.jsonl` contains prompt IDs and score_fields only, with no prompt or response text, and `summary.v2.json` contains corrected aggregates. `compare_quant_pairs.py` prefers complete v2 sidecars when they exist and falls back to original summaries otherwise."),

  H2("H.3 Current artifact contract"),
  buildTable(
    ["Artifact", "Status", "Handling rule"],
    [
      ["raw.jsonl", "TC1-original saved generations and original score fields", "Do not modify, redact, duplicate into new raw files, or print prompt/response content during audits."],
      ["summary.json", "Original aggregate summary from the initial run", "Do not overwrite during post-hoc scoring corrections."],
      ["scores.v2.jsonl", "Derived v2 score fields only; no prompt/response text", "Authoritative corrected scorer sidecar for HarmBench/XSTest analysis."],
      ["summary.v2.json", "Derived v2 aggregate summary", "Used by `make analyze` through sidecar preference logic."],
      ["rescore_diagnostics_*.csv", "Redacted diagnostics with IDs, labels, lengths, and matched-pattern names", "Safe to inspect; must remain free of raw prompt/response text."],
      ["scores.judge.harmbench_cls.jsonl", "Per-prompt official HarmBench classifier labels (the PRIMARY HarmBench scorer, D16); produced by the TC1 judge jobs (512-token primary study: job 61524; 128-token comparison: jobs 61047/61134)", "Redacted (IDs + booleans only). Committed to the repo. Never overwrite; raw text must never appear."],
      ["summary.judge.harmbench_cls.json", "Aggregate judge HarmBench metrics + GPU/precision metadata", "The authoritative HarmBench ASR source. judge_agreement.{json,csv} compares it to the v2 proxy."],
    ],
    [1900, 3200, 4700],
  ),

  H2("H.4 Verification completed"),
  PJ("The workflow was verified in five ways. First, the full test suite passes, including refusal-parser regressions, analysis sidecar selection, interpretation guards, registry and surface checks, and judge-validation tests for redaction, raw immutability, idempotency, ASR aggregation, the official HarmBench classifier prompt template, and judge-precision resolution. Second, the judge jobs ran in full precision on a 32 GB V100 with zero parse errors throughout (512-token primary study: job 61524, 2 000 generations plus 1 000 INT8; 128-token comparison: jobs 61047/61134, 2 000 generations). Third, `scripts/judge_agreement.py` reproduces the judge-vs-v2 agreement and the judge-primary pair labels deterministically from the committed sidecars. Fourth, the original `raw.jsonl` and `summary.json` files were confirmed unchanged by every post-hoc step (rescore and judge). Fifth, all judge sidecars were checked for redaction: they contain prompt IDs, boolean labels, and run metadata only, with no prompt, behaviour, or response text."),

  H2("H.5 Remaining construct-validity extension"),
  PJ("The classifier was cross-checked three times: by an independent frontier-model second judge (gpt-4o, Section 6.9 Result 4), against human labels on a stratified subset (Section 6.9 Result 5), and by an open-weight guard model (Llama-Guard-3-8B, revision-pinned, Section 6.9). The re-scoring by that guard model is revision-pinned and locally rerunnable, which the versioned API judge is not. It agrees with the classifier at κ 0.36–0.92 across the ten base/NF4 aliases and finds every per-pair ΔASR non-significant. The construct caveat is that it answers a looser safety question than the behaviour-specific classifier. All three run through the pluggable judge backend in `ethical_benchmark/judges/validation.py` and the agreement analysis in `scripts/judge_agreement.py`. The primary scorer is unaffected either way."),
];

// ------------------------------------------------------------
// Assemble document
// ------------------------------------------------------------
const body = [
  ...cover,
  ...declarations,
  ...abstract,
  ...toc,
  ...lof,
  ...ch1,
  ...ch2,
  ...ch3,
  ...ch4,
  ...ch5,
  ...ch6,
  ...ch7,
  ...ch8,
  ...ch9,
  ...ch10,
  ...refs,
  ...appendixA,
  ...appendixB,
  ...appendixC,
  ...appendixD,
  ...appendixE,
  ...appendixF,
  ...appendixH,
  ...appendixI,
];

const doc = new Document({
  creator: "TAN UEI HORNG",
  title: "FYP Comprehensive Report: Benchmarking Ethical Performance of Open-Source LLMs (CCDS25-1136)",
  features: { updateFields: true },  // prompt Word/LibreOffice to populate the TOC field on open
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
      // Pre-declared per-list numbering references. Each list site in the
      // document calls numberedList(items), which mints the next ref name
      // (numlist1, numlist2, ...). Each ref is its own counter, so each
      // list visibly restarts at 1. Declare ~20 to leave headroom for
      // future additions.
      ...Array.from({ length: 20 }, (_, i) => ({
        reference: `numlist${i + 1}`,
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      })),
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({
              text: "CCDS25-1136 — TAN UEI HORNG (UTAN001)",
              font: FONT, size: 18, italics: true, color: "666666",
            }),
            new TextRun({ text: "\t" }),
            new TextRun({
              text: "FYP Comprehensive Report  ·  2 July 2026",
              font: FONT, size: 18, italics: true, color: "666666",
            }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: FONT, size: 18 }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 }),
            new TextRun({ text: " of ", font: FONT, size: 18 }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18 }),
          ],
        })],
      }),
    },
    children: body,
  }],
});

const OUTPUT = path.join(__dirname, "..", "docs", "FYP_Report_2026-07-25_v6.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("WROTE: " + OUTPUT);
  console.log("Size: " + buf.length + " bytes");
});
