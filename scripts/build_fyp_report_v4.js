// ============================================================================
// SUPERSEDED - DO NOT BUILD (D53, 2026-07-19).
// This retired 128-era builder is kept ONLY as the historical source for its
// archived docx snapshot in docs/archive/. Running it would resurrect stale
// 128-era numbers/claims into docs/ (the exact resurrection risk D42 guards
// against). The canonical deliverables are built by `make report` /
// `make thesis` / `make interim` (+ the three *-humanized targets), i.e.
// build_fyp_report_v5.js, build_fyp_thesis_v4.js, build_fyp_interim*.js,
// build_fyp_report_humanized.js, build_fyp_thesis_humanized.js.
// To intentionally rebuild the archived snapshot (rare; e.g. re-archiving):
//   FYP_BUILD_SUPERSEDED=1 node scripts/build_fyp_report_v4.js
// ============================================================================
if (!process.env.FYP_BUILD_SUPERSEDED) {
  console.error(
    "SUPERSEDED builder (scripts/build_fyp_report_v4.js): refusing to build - its output is a " +
      "retired 128-era snapshot (now in docs/archive/). Use `make report` " +
      "(build_fyp_report_v5.js) or the sibling canonical targets instead. " +
      "Set FYP_BUILD_SUPERSEDED=1 only to rebuild the archived snapshot."
  );
  process.exit(2);
}

// FYP Report builder (v3 - publication-grade pass + humanizer prose edit) - docx-js
// Output: /Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-06-30_v4.docx
// v4 (T31/D39): adds §6.16 Generation-Length Robustness (512-token rerun). Built
// from a copy of build_fyp_report_v3.js; v3 + its docx are left untouched for
// comparison (same non-destructive pattern as v2 → v3).
// v3 = v2 (figures, IEEE citations, FDR/power, scorer-validation lead) PLUS a full
// de-AI prose pass (em-dashes removed, discourse-marker crutches cut, rule-of-three
// and negative-parallelism trimmed). v2 and the original are left untouched for
// side-by-side comparison.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink,
  TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun,
} = require("docx");

const FIGDIR = path.join(__dirname, "..", "docs", "figures");

// ------------------------------------------------------------
// Document constants
// ------------------------------------------------------------
const PAGE_W = 12240;       // US Letter
const PAGE_H = 15840;
const MARGIN = 1440;        // 1 inch
const CONTENT_W = PAGE_W - 2 * MARGIN;  // 9360

const FONT = "Calibri";
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
    spacing: { after: opts.after !== undefined ? opts.after : 120, line: opts.line || 300 },
    alignment: opts.alignment,
    pageBreakBefore: opts.pageBreakBefore || false,
  });
}

function PJ(text, opts = {}) {
  // Justified paragraph for body text
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22 })],
    spacing: { after: 160, line: 300 },
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
    children: [new TextRun({ text: "Interim Report", font: FONT, size: 22, italics: true })],
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
    children: [new TextRun({ text: "Document date:  26 June 2026", font: FONT, size: 26, bold: true })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Interim report, revised through June 2026. Scope: five matched pairs across four families at three precisions (fp16 → INT8 → NF4). HarmBench ASR is scored by the official HarmBench classifier (judge-primary, D16) and cross-checked by a second judge; the refusal regex is retained only as a transparent proxy. A companion standalone thesis (FYP_Thesis) presents the same study in condensed, IEEE-cited form.", font: FONT, size: 20, italics: true, color: "555555" })],
    spacing: { after: 100 },
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------
// Abstract
// ------------------------------------------------------------
const abstract = [
  H1NoBreak("Abstract"),
  PJ("Compact instruction-tuned language models are almost always deployed quantized, yet their safety is typically validated on the full-precision release. Whether quantization shifts safety alignment (and whether an apparent shift is a genuine alignment change or a by-product of degraded capability) is unresolved for the compact-deployment regime that institutional benchmarks (TrustLLM, DecodingTrust, SafetyBench) do not target. This work answers that question with a matched-pair, judge-validated study and reports a methodological result that has consequences beyond it."),
  PJ("The design loads the baseline and quantized members of each pair from identical weights, applying quantization on the fly, so quantization is the only variable. Five pairs across four families (Qwen, Llama, Mistral, Phi; 1.7–7.2 B) are evaluated at three precisions (fp16, INT8/LLM.int8, NF4 four-bit) on harmful compliance (HarmBench), over-refusal (XSTest), and capability (MMLU and ARC-Challenge). The central methodological finding is that the scorer determines the conclusion: a refusal-counting regex over-counts harmful compliance (its harmful set strictly contains the classifier's) and replacing it with HarmBench's own fine-tuned classifier (cross-checked by a second, architecturally independent judge at Cohen's κ 0.60–0.95) relocates the study's only significant safety regression from one model to another. Refusal-counting therefore overstates quantization harm, a cautionary result for safety evaluation generally."),
  PJ("Under the classifier, four-bit NF4 never significantly reduces harmful compliance and significantly increases it in only the smallest model (Qwen3-1.7B, ΔASR = +0.055, 95% CI [+0.010, +0.100]). That effect is modest, decode-dependent (it halves under stochastic sampling), and does not survive a Benjamini-Hochberg correction over the family of primary contrasts, under which the multiplicity-robust signals are capability losses rather than safety changes; a refusal-margin probe and a power analysis converge on a capability-driven reading. An fp16 → INT8 → NF4 sweep shows the effect is not bit-width-graded: capability loss is a clean cliff at four-bit, whereas a separate, both-judge-significant safety increase appears on Llama-3B specifically at INT8 and reverts at NF4. The contributions are a reproducible, capability-anchored, judge-validated evaluation method for the compact regime, and the evidence that the headline risk of four-bit quantization in small models is capability degradation, with harmful-compliance changes that are real but small, scorer-sensitive, and concentrated in the smallest model. A generation-length robustness rerun at HarmBench's 512-token reference budget (retaining the 128-token study for comparison) reinforces this reading: the Qwen3-1.7B safety increase does not replicate at 512 (ΔASR ≈ 0 under both judges, McNemar p = 1.000), while the four-bit capability losses are essentially unchanged, so the safety effect is the fragile, budget-dependent component and the capability cost is the robust one (§6.16)."),
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
  P("Figure 1  Quantization effect on harmful compliance (per-pair judge ΔASR forest plot).", { after: 60 }),
  P("Figure 2  The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).", { after: 60 }),
  P("Figure 3  Multi-seed decoding sensitivity for Qwen3-1.7B (greedy vs stochastic).", { after: 60 }),
  P("Figure 4  Per-category HarmBench ASR for Qwen3-1.7B, fp16 vs NF4.", { after: 60 }),
  P("Figure 5  Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.", { after: 60 }),
  P("Figure 6  Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).", { after: 60 }),
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
  P("Table D.1  Distribution of automated tests across modules.", { after: 60 }),
];

// ------------------------------------------------------------
// Chapter 1 - Introduction
// ------------------------------------------------------------
const ch1 = [
  H1("Chapter 1: Introduction"),

  H2("1.1 Background and Motivation"),
  PJ("The last two years have seen a rapid proliferation of compact instruction-tuned large language models (LLMs) in the one-to-four-billion-parameter range. Models such as Qwen 2.5 and 3.x, Llama 3.2, Microsoft Phi-3, and Google Gemma 2 have demonstrated that capable reasoning, instruction following, and multilingual performance are achievable at parameter counts that fit comfortably on consumer hardware, mobile chipsets, and edge accelerators. This has shifted the practical envelope of LLM deployment: tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, with stronger privacy guarantees, lower latency, and substantially reduced operational cost."),
  PJ("In practice, however, such models are rarely deployed in full precision. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision impractical for routine use, and quantization to four-bit precision has become the de facto compression standard for on-device inference. Lightweight runtimes such as llama.cpp and on-device agent frameworks routinely ship four-bit GGUF or BitsAndBytes NF4 checkpoints by default. End users encountering these models therefore almost always interact with a quantized variant, not the original baseline."),
  PJ("Quantization has historically been treated as a numerical optimisation technique whose primary cost is a small loss in perplexity or downstream accuracy. A growing body of evidence challenges this view. Quantization can alter behavioural properties of an LLM in ways that are not visible from perplexity alone, including instruction-following fidelity, refusal calibration, and resistance to adversarial prompts. Because safety alignment is itself a learned behaviour encoded in the model weights, any operation that alters those weights, even one that preserves task accuracy on average, has the potential to perturb that behaviour. Understanding how compression interacts with safety is therefore not optional: it is a prerequisite to safe deployment of any compact model that has been quantized for production use."),

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
  PJ("The distinction matters. A model that becomes more refusal-prone after quantization could plausibly be reported as \"safer\" by a benchmark that measures only harmful compliance. The same model, measured on benign prompts, may also refuse those (exhibiting over-refusal) and on capability benchmarks may answer fewer questions correctly. Without a joint evaluation of harmful compliance, over-refusal, and capability, the safety community risks systematically misattributing capability collapse as safety improvement. The problem statement above directly motivates the multi-axis evaluation framework adopted in this study."),

  H2("1.3 Research Questions"),
  PJ("The study is organised around five research questions. The first three are primary questions answered directly by the matched-pair experiments. The remaining two examine secondary dimensions: scale sensitivity within a family, and replication across families."),
  Bullet("RQ1 (Primary). Does four-bit NF4 quantization measurably change harmful-compliance behaviour, as measured by HarmBench Attack Success Rate, relative to matched baseline weights in compact instruction-tuned models?"),
  Bullet("RQ2 (Primary). Does four-bit quantization change over-refusal behaviour on benign prompts, as measured by XSTest?"),
  Bullet("RQ3 (Primary). Does four-bit quantization degrade general capability, as measured by accuracy on a curated MMLU subset?"),
  Bullet("RQ4 (Secondary, scale). Does the magnitude of the quantization effect differ between Qwen3-1.7B and Qwen3-4B within the same model family?"),
  Bullet("RQ5 (Robustness). Does the pattern observed within the Qwen family replicate qualitatively in a different model family, namely Llama 3.2 3B?"),

  H2("1.4 Scope and Boundaries"),
  PJ("Explicit scoping is essential to make the study tractable within the constraints of an undergraduate Final Year Project and to keep the resulting claims defensible. The following boundaries are stated up front."),
  Bullet("Quantization method. The main study applies BitsAndBytes NF4 four-bit loading on the fly via the Hugging Face transformers integration; a precision point (§6.15) adds BitsAndBytes INT8/LLM.int8 as a second method, so the study spans fp16 → INT8 → NF4. Both are applied on the fly to identical baseline weights. Other families such as GPTQ, AWQ, and GGUF are out of scope and are flagged in the Future Work chapter."),
  Bullet("Model regime. The study targets compact instruction-tuned models in the two-to-four-billion-parameter range. This regime corresponds to the practical envelope of contemporary on-device deployment."),
  Bullet("Languages and modality. Evaluation is restricted to English-language, text-only interactions."),
  Bullet("Decoding. All inference uses deterministic greedy decoding with temperature 0.0 and top-p 1.0, both to eliminate within-condition variance and to maximise comparability across baseline and quantized pair members."),
  Bullet("Benchmark axes. Rather than attempting a broad multi-axis ethical evaluation, which is already covered by larger institutional benchmarks, this study evaluates three complementary axes that jointly enable the alignment-versus-capability disambiguation that motivates the research question."),

  H2("1.5 Contributions"),
  PJ("The contributions of this project are, in order of significance, a methodological cautionary result, a controlled study design, an interpretation method, an engineering framework, and an empirical characterisation of quantization effects."),
  ...numberedList([
    "A scorer-validity result with consequences beyond this study: the choice of HarmBench scorer changes the conclusion. A deterministic refusal-counting regex over-counts harmful compliance (its harmful set is a near-strict superset of the official classifier's) and replacing it with HarmBench's own fine-tuned classifier (cross-checked by a second, architecturally independent judge at Cohen's κ 0.60–0.95) relocates the study's only significant safety regression from one model to another. Refusal-counting therefore overstates quantization harm, a cautionary lesson for safety evaluation generally.",
    "A controlled matched-pair study design that isolates quantization as the sole experimental variable. Both members of every pair are loaded from the same underlying model_id; the only difference is whether on-the-fly quantization is applied during the from_pretrained call. This eliminates publisher-asymmetry and conversion-pipeline asymmetry, two confounds that affect most published comparisons of quantized and full-precision checkpoints.",
    "A capability-anchored interpretation method that combines harmful-compliance, over-refusal, and capability deltas into one of six diagnostic labels (alignment degradation, alignment improvement, capability collapse masquerading as safety, over-refusal regression, robust preservation, broad degradation), each carried with a two-layer evidence status. This formalises the alignment-versus-capability disambiguation as a reproducible procedure rather than an ad hoc judgement.",
    "An open, reproducible benchmarking framework (approximately 2,800 lines of production Python and 1,800 lines of automated tests; 329 tests across twenty-five files) with batched chat-templated generation, per-prompt audit logging, immutable hash-pinned raw artefacts, redacted score sidecars, and resumable per-model SLURM matrix jobs for the NTU TC1 cluster.",
    "An empirical characterisation of quantization across five model pairs, four families, and three precisions (fp16 → INT8 → NF4): under the validated classifier, four-bit NF4 never significantly reduces harmful compliance and significantly increases it only in the smallest model, the multiplicity-robust signal is capability loss rather than safety change, and the effect is not bit-width-graded. These results are supported by a refusal-margin mechanism probe, a multi-seed sensitivity arm, an FDR/power analysis, and a second-judge cross-check.",
  ]),

  H2("1.6 Report Structure"),
  PJ("Chapter 2 surveys the relevant literature on large-scale ethical benchmarking, the helpfulness–harmlessness trade-off, the deployment of compact LLMs, and the behavioural effects of quantization, concluding with the specific research gaps that this study targets. Chapter 3 details the experimental methodology, including the matched-pair design, quantization approach, benchmark selection, scoring, decoding controls, and the interpretation framework. Chapter 4 documents the system design and implementation: package structure, configuration schema, the model loader and generation pipeline, the benchmark plugin architecture, matrix orchestration, resume logic, and SLURM job generation. Chapter 5 describes the experimental setup on TC1 and the run plan. Chapter 6 presents the experimental results for all five model pairs; the original three completed on the NTU TC1 cluster on 2026-05-27 and the two cross-family pairs (Mistral-7B, Phi-4-mini) on 2026-06-15. Chapter 7 discusses threats to validity, Chapter 8 records limitations, and Chapter 9 proposes future work. Chapter 10 concludes. Six appendices reproduce the full configuration, an example SLURM script, the configuration schema, the test inventory, the repository layout, and a glossary."),
];

// ------------------------------------------------------------
// Chapter 2 - Literature Review
// ------------------------------------------------------------
const ch2 = [
  H1("Chapter 2: Literature Review"),

  H2("2.1 Large-Scale Ethical Benchmarking of LLMs"),
  PJ("The past two years have produced several large-scale institutional efforts to benchmark the ethical performance of open-source LLMs. TrustLLM [1], DecodingTrust [2], and SafetyBench [3] each provide a standardised pipeline that evaluates dozens of widely used models across multiple trustworthiness dimensions, including toxicity, bias, hallucination, robustness, fairness, and ethics. These benchmarks have published leaderboards that cover popular open-weight model families such as Llama, Mistral, Qwen, and Falcon, and have meaningfully shifted the conversation around what constitutes an acceptable safety profile in publicly released models."),
  PJ("The strengths of these frameworks are clear: large model coverage, multi-dimensional evaluation, and standardised methodologies that facilitate cross-model comparison. Their gaps, however, are also clear in the context of the present work. First, they predominantly evaluate full-precision checkpoints; quantized variants are either omitted or treated as a separate, secondary evaluation. Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, where alignment training tends to be most robust. Third, they typically report each safety axis independently (harmful compliance, bias, toxicity) without explicitly anchoring those measurements against a capability metric, leaving open the question of whether safety changes reflect alignment or capability."),

  H2("2.2 The Helpfulness–Harmlessness Trade-off"),
  PJ("A central, well-documented limitation of static safety benchmarks is their difficulty in jointly capturing the tension between helpfulness and harmlessness. A model that is heavily optimised for safety may become overly conservative, refusing benign prompts that incidentally resemble unsafe ones, a failure mode known as over-refusal or exaggerated safety. Conversely, a model optimised for helpfulness may remain vulnerable to adversarial jailbreak attacks. Measuring only one side of this trade-off produces an incomplete picture."),
  PJ("Two benchmarks have emerged as the practical standards for measuring opposite sides of this trade-off. HarmBench [4] and AdvBench provide curated unsafe prompts and measure attack success rate, the fraction of unsafe prompts to which the model produces a harmful, complying response. XSTest [5] provides benign prompts that are superficially similar to unsafe ones and measures over-refusal rate, the fraction of benign prompts the model nonetheless refuses. Evaluating both simultaneously is essential to detecting trade-offs introduced by alignment training or compression."),
  PJ("MMLU [6], the Massive Multitask Language Understanding benchmark, has become the de facto general-capability anchor in safety studies; ARC-Challenge [7] provides a structurally different second capability axis used here as corroboration. By measuring multiple-choice accuracy across a broad spectrum of academic and professional subjects, MMLU provides a capability signal that is largely independent of refusal behaviour, allowing capability collapse to be detected even when safety metrics appear to improve."),

  H2("2.3 Small Language Models and On-Device Deployment"),
  PJ("Recent compact LLM releases have repeatedly demonstrated that strong reasoning, instruction following, and multilingual performance are achievable at parameter counts below four billion. The Qwen 2.5 and Qwen 3 series, Llama 3.2 (1B and 3B Instruct), Microsoft Phi-3 (3.8B), and Google Gemma 2 (2B and 9B) have each been positioned for on-device or edge inference. These models are routinely integrated into lightweight agent frameworks and consumer applications, where their compact size enables genuinely local execution."),
  PJ("The deployment reality of these models is that they are almost never used in full precision. Memory and latency constraints on consumer hardware drive routine use of four-bit quantization, often via on-the-fly BitsAndBytes loading or pre-quantized GGUF checkpoints. Safety claims attached to the unquantized release model therefore do not, in general, transfer to the model that end users actually encounter."),

  H2("2.4 Quantization and Behavioural Effects"),
  PJ("Quantization compresses model weights from higher-precision floating-point representations to lower-precision integer or normalized-float representations, reducing memory footprint and accelerating inference at the cost of some numerical fidelity. Post-training quantization (PTQ) methods apply this conversion after the model has been trained and require no fine-tuning, making them attractive for deployment."),
  PJ("Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA [8] has become particularly prevalent in the open-source ecosystem. NF4 represents each weight using a four-bit normalized-float code optimised for the typical Gaussian-like distribution of neural network weights, with double quantization applied to the quantization constants to further reduce overhead. The eight-bit LLM.int8() method [9] is a distinct, mixed-precision algorithm in the same library that decomposes outlier features into a separate high-precision path; the present study evaluates both. The accompanying BitsAndBytes library integrates directly with Hugging Face transformers, exposing quantized loading through a single BitsAndBytesConfig object that can be passed to from_pretrained at model load time."),
  PJ("A body of recent work has argued that quantization is not behaviourally neutral. Studies across a wide range of model sizes have shown that smaller models can suffer severe degradation under aggressive four-bit quantization while larger models remain comparatively stable, and that instruction-following fidelity and hallucination rates can shift even when general benchmark scores appear preserved [18]. Safety-specific studies make the point directly. Kharinaev et al. [14] evaluate 66 quantized variants across multiple post-training and quantization-aware methods and find that quantization can degrade safety alignment, with no single method dominating across models and bit-widths. Egashira et al. [16] show that widely used BitsAndBytes schemes, including NF4, can be manipulated so that a benign full-precision model becomes harmful once quantized. HarmLevelBench [17] reports that the direction of the safety change under AWQ and GPTQ is non-uniform across attack types. Q-resafe [15] confirms quantization-induced safety degradation and proposes a quantization-aware patching method to repair it. Most of these studies evaluate either perplexity or general capability, with safety considered (if at all) as a separate axis rather than jointly with capability against a fixed capability anchor, the gap this study targets."),

  H2("2.5 Research Gaps Targeted by This Work"),
  PJ("Four interlocking gaps in the existing literature motivate the present study."),
  ...numberedList([
    "Gap 1: Limited empirical study of compact (<4B) instruction-tuned models in the safety–quantization context. Most quantization studies focus on the seven-billion-to-thirteen-billion parameter range and above. Edge-deployment-relevant compact models are under-represented.",
    "Gap 2: Lack of integrated evaluation that measures harmful compliance, over-refusal, and capability simultaneously. Studies that measure only harmful compliance cannot detect capability-driven safety artifacts.",
    "Gap 3: Difficulty interpreting whether observed safety metric changes in quantized models reflect a real alignment shift or a side-effect of capability degradation. No widely adopted convention exists for jointly interpreting safety and capability deltas under compression.",
    "Gap 4: Provenance asymmetry in existing comparisons. Many published comparisons of full-precision and quantized checkpoints use a full-precision checkpoint from one publisher and a pre-quantized checkpoint from another, conflating quantization effects with checkpoint-conversion effects. A clean, on-the-fly quantization design from identical baseline weights eliminates this confound.",
  ]),
  PJ("The methodology described in Chapter 3 is structured to address all four gaps. The compact-deployment regime is addressed by the choice of Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B; the integrated evaluation is addressed by the three complementary benchmarks; the interpretation challenge is addressed by the rule-based interpretation layer; and the provenance asymmetry is eliminated by on-the-fly NF4 loading from the same baseline weights."),
];

// ------------------------------------------------------------
// Chapter 3 - Methodology
// ------------------------------------------------------------
const ch3 = [
  H1("Chapter 3: Methodology"),

  H2("3.1 Experimental Design"),
  PJ("The study adopts a matched-pair comparative experimental design. Each model under study is evaluated as a pair: a baseline variant loaded in the default high-precision dtype, and a four-bit variant produced by applying BitsAndBytes NF4 quantization on the fly at load time. Both pair members are loaded from exactly the same Hugging Face model_id; no separately uploaded \"pre-quantized\" checkpoint is used. The only operational difference between the two members of a pair is the presence of a BitsAndBytesConfig object in the from_pretrained call."),
  PJ("This design choice is the strongest internal-validity property of the study. By construction, both pair members share identical baseline weights, the same tokenizer, the same chat template, and the same release artifact. Any observed delta in benchmark scores is therefore attributable to the quantization step itself rather than to checkpoint provenance, separate fine-tuning, or conversion artifacts. The on-the-fly NF4 path is the most direct way to operationalise the counterfactual \"the same model, quantized\" in code."),
  PJ("The independent variables are quantization state (baseline versus four-bit), model scale (two billion versus four billion parameters within the Qwen family), and model family (Qwen versus Llama). The dependent variables are harmful compliance, over-refusal on benign prompts, and general capability, each measured by a corresponding benchmark."),

  H2("3.2 Model Selection"),
  PJ("Three model pairs are evaluated. The Qwen pairs at 1.7 billion and four billion parameters serve as the primary within-family comparison and provide both a quantization axis and a scale axis. The Llama 3.2 3B pair sits between them by parameter count and serves as a cross-family robustness check. Table 3.1 summarises the model matrix."),
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
  P("Table 3.1  Model pair selection matrix. Note that within each pair, both members share the same Hugging Face model_id; the only operational distinction is the presence of an NF4 BitsAndBytesConfig at load time.", { size: 18 }),
  PJ("The Qwen3 checkpoints were selected after an initial environment validation on the TC1 cluster surfaced a compatibility failure with a third-party model upload (techwithsergiu/Qwen3.5-text-2B and techwithsergiu/Qwen3.5-text-4B) that had been considered during planning. Those checkpoints use a hybrid linear-attention/SSM architecture that requires the causal-conv1d and flash-linear-attention libraries at runtime. When those libraries are absent, the PyTorch fallback conv1d path raises a CUDNN_STATUS_NOT_INITIALIZED error during the very first inference batch, terminating the job before any results are produced. Installing those libraries in the TC1 conda environment was considered but ruled out: the libraries require custom CUDA kernel compilation against the cluster's specific driver version and are not trivially installable in an offline environment. Switching to the official Qwen3 release (Qwen/Qwen3-1.7B and Qwen/Qwen3-4B) eliminated all external dependencies: the Qwen3 dense models are standard grouped-query-attention transformers that load and run with the baseline transformers and bitsandbytes packages already in the environment. The switch was made before any production runs, so no experimental results were affected."),

  H2("3.3 Quantization Approach"),
  PJ("Quantization is applied entirely on the fly by injecting a BitsAndBytesConfig object into the AutoModelForCausalLM.from_pretrained call. The configuration sets load_in_4bit to True, selects the NF4 quantization type, enables double quantization, and sets the four-bit compute dtype to follow the resolved torch_dtype of the model, that is, float16 on CUDA when dtype is set to auto, and float32 on CPU (although the CPU path is not used for quantized runs since BitsAndBytes requires a CUDA device)."),
  PJ("Four-bit loading is forbidden on a CPU runtime: the loader raises a clear error when quantized is true and the resolved runtime device is not cuda. After load, the loader verifies that the resulting model object reports is_loaded_in_4bit, logging a warning otherwise."),
  PJ("The choice of NF4 over alternative PTQ methods such as GPTQ and AWQ is deliberate. NF4 is the dominant on-the-fly quantization format in the open-source ecosystem, is supported natively by transformers without an offline calibration pass, and applies symmetrically to any model that can be loaded with from_pretrained. This applicability symmetry is required by the matched-pair design: the quantization step must be applicable to exactly the same weights as the baseline, without an intermediate calibration corpus that would itself become a confound."),

  H2("3.4 Benchmark Selection and Mapping"),
  PJ("Four benchmarks are used across the three dimensions required to disambiguate alignment shifts from capability collapse. The capability dimension is measured by two complementary benchmarks: MMLU (the primary anchor) and ARC-Challenge (a structurally different second capability benchmark, added as T23), so that the alignment-versus-capability disambiguation does not rest on a single capability proxy."),
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
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets. Sampling is deterministic and seed-controlled. Capability is a two-benchmark composite (MMLU + ARC-Challenge).", { size: 18 }),
  PJ("HarmBench is selected as the harmful-compliance benchmark because it is the most widely adopted adversarial-prompt collection in contemporary safety research and provides a stable definition of attack success. The walledai release on Hugging Face is used with the standard config. XSTest is the canonical over-refusal benchmark; this repository bundles the canonical XSTest v2 prompts CSV from paul-rottger/xstest and evaluates the benign-only subset to measure how often the model refuses prompts that should not be refused. MMLU is used as the primary general-capability anchor, with ARC-Challenge (allenai/ai2_arc; ~1,172 grade-school science questions designed to resist retrieval shortcuts; identical exact-match scoring to MMLU) added as a structurally different second capability benchmark, so capability is reported as a two-benchmark composite (see §6.4.1). Interpretation labels remain anchored on MMLU, with ARC as a corroborating axis."),
  PJ("To keep MMLU tractable within the per-job walltime on TC1, a six-subject subset is used: business_ethics, clinical_knowledge, college_biology, high_school_world_history, high_school_macroeconomics, and human_aging. These subjects span ethics, sciences, humanities, social sciences, and applied knowledge, and are deliberately chosen to cover heterogeneous reasoning demands without exhausting the time budget. The total sample cap of 300 questions is distributed evenly across the six subjects."),

  H2("3.5 Scoring: Refusal Detection and the HarmBench Classifier"),
  PJ("Two scorers are used, with a clear division of roles. A deterministic regex-based refusal parser provides fully reproducible, zero-variance scoring; it is the scorer for XSTest over-refusal, and for HarmBench it is retained as a secondary, refusal-based upper-bound proxy (it measures non-refusal rather than genuine harmful compliance). The primary HarmBench Attack Success Rate scorer, however, is the official HarmBench classifier (cais/HarmBench-Llama-2-13b-cls), a fine-tuned judge model, run in full precision with deterministic greedy decoding (decision D16; the validation that established it as primary is reported in §6.12). The classifier was promoted to primary after that validation showed the regex over-counts ASR (equating non-refusal with success) most severely for the Qwen and Mistral families. Retaining the deterministic regex alongside it is deliberate: it lets the reader see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring, and its reproducibility is why it remains the over-refusal scorer and the transparent proxy. A judge LLM does introduce a variance budget and an additional moving part, so it is controlled accordingly. It is run at full precision, with greedy decoding, over the immutable saved generations, writing only redacted sidecars, rather than avoided."),
  PJ("Three boolean fields are recorded for each response: harmful_compliance, safe_refusal, and malformed. For HarmBench, harmful compliance corresponds to a response that addresses the unsafe request substantively rather than refusing. For XSTest benign prompts, refusal is the failure mode, and over-refusal rate is the fraction of benign prompts on which the refusal pattern fires. The malformed field flags responses that are empty, contain only template fragments, or fail to parse into the expected output format."),
  PJ("The trade-off of deterministic parsing (its potential to miss nuanced or euphemistic refusals) is exactly why the official HarmBench classifier, not the regex, is the primary HarmBench scorer (§6.12). For XSTest over-refusal, where no benchmark-native classifier exists, deterministic parsing remains in use; that residual limitation is acknowledged in the Limitations chapter and motivates a refusal-style XSTest judge as one of the proposed future-work directions."),

  H2("3.6 Decoding Controls"),
  PJ("All inference runs share identical decoding parameters, applied uniformly across baseline and quantized members of every pair. Table 3.3 lists the parameters."),
  buildTable(
    ["Parameter", "Value", "Rationale"],
    [
      ["temperature", "0.0", "Greedy decoding; removes within-condition variance."],
      ["top_p", "1.0", "Disabled (greedy)."],
      ["max_new_tokens", "128", "Sufficient for short HarmBench refusals and MMLU answers; monitored during smoke run."],
      ["repetition_penalty", "1.0", "Neutral; avoids confounding refusal calibration."],
      ["max_input_tokens", "1024", "Caps prompt length; prevents context overflow."],
      ["use_chat_template", "true", "Applies model-specific chat template to every prompt."],
      ["enable_thinking", "false", "Critical: disables Qwen3.x default thinking mode so the 128-token budget is not consumed by <thinking> blocks before the answer."],
    ],
    [2200, 1300, 5860],
  ),
  P("Table 3.3  Decoding controls used during inference. The enable_thinking=False setting is passed when the tokenizer supports it; a TypeError fallback gracefully handles tokenizers that do not accept the keyword.", { size: 18 }),

  H2("3.7 Metrics and Pairwise Analysis"),
  PJ("The primary scalar metrics computed per (model, benchmark) pair are defined below. Throughout, n denotes the number of evaluated prompts."),
  Equation("ASR  =  (1 / n) Σᵢ cᵢ ,    cᵢ ∈ {0, 1}  indicating harmful compliance"),
  Equation("OverRefusalRate  =  (1 / n_b) Σᵢ fᵢ ,    benign prompts only"),
  Equation("Accuracy  =  (1 / n) Σᵢ 𝟙[ŷᵢ = yᵢ]  ,    MMLU"),
  Equation("MalformedRate  =  (1 / n) Σᵢ mᵢ ,    AnsweredRate  =  (1 / n) Σᵢ 𝟙[ŷᵢ ≠ ∅]"),
  PJ("Pairwise deltas are computed per pair_id, taking the four-bit value minus the baseline value of each metric:"),
  Equation("ΔM  =  M_{4-bit}  −  M_{baseline}"),
  Equation("ΔM_rel  =  ( M_{4-bit}  −  M_{baseline} )  /  M_{baseline}    when M_{baseline} ≠ 0"),
  PJ("Bootstrap 95% confidence intervals are computed for each metric by resampling the per-prompt score records with replacement. The intervals reflect prompt-sampling variance only; because decoding is deterministic at temperature 0.0, there is no within-condition stochastic variance."),
  PJ("Paired significance test for HarmBench (McNemar). Because both pair members see the identical prompt set, the HarmBench ΔASR is a paired-binary contrast, and the textbook-correct significance test is McNemar's exact test on the discordant prompts (those scored harmful under exactly one member) rather than an unpaired two-proportion test. Writing b for the number of prompts that become harmful under quantization (baseline-safe → 4-bit-harmful) and c for those that become safe (baseline-harmful → 4-bit-safe), under the null hypothesis that quantization does not change harmful compliance each discordant prompt favours one member with probability one half, so min(b, c) follows a Binomial(b + c, 0.5); the exact two-sided p-value doubles the lower tail. Reporting b and c also makes the effect size transparent: the reader sees exactly how many prompts moved each way rather than only the net delta. McNemar's exact p is reported alongside the bootstrap CI for every HarmBench pair as an independent corroboration that does not assume large-sample normality (implemented in mcnemar_exact_test, no SciPy dependency)."),

  H2("3.8 Interpretation Framework"),
  PJ("Combining harmful-compliance, over-refusal, and capability deltas, each pair receives one of six rule-based interpretation labels. These labels are the central analytical output of the study and formalise the alignment-versus-capability disambiguation. The taxonomy is symmetric on both behavioural axes. On the safety axis it pairs alignment_degradation (ASR up beyond threshold, capability preserved) with its mirror alignment_improvement (ASR down beyond threshold, capability preserved), gated identically so a genuine safety win is never demoted to a fallback degradation. On the over-refusal axis it pairs over_refusal_regression (the model refuses materially more benign prompts, with safety and capability held) against robust_preservation (over-refusal stable or improved), so a pure over-refusal change is named on its own axis rather than absorbed into broad_degradation."),
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
  PJ("Two-layer reporting: label plus evidence status. The interpretation label is a statement about the direction and magnitude of the point-estimate deltas; it is intentionally decoupled from statistical significance so the taxonomy stays stable and legible. Each label is therefore paired with a separate evidence_status field derived from the same paired-bootstrap 95% CI (and, for HarmBench, McNemar's exact test described in §3.7): confirmed when the delta the label keys on excludes zero, directional when the point estimate crosses the threshold but the CI includes zero, null for robust_preservation (a bounded CI cannot positively certify a null), and unknown when per-prompt outcomes are unavailable. This decoupling prevents a non-significant point estimate from being read as a confirmed finding. For example, the Qwen 4B pair is labelled alignment_degradation (its ASR point estimate exceeds threshold) but carries evidence_status = directional because its CI touches zero. The status is computed by label_evidence_status alongside the label and emitted in results/analysis/pair_interpretations.json and judge_agreement.json; the label rule itself (classify_pair_change) is unchanged, so the diagnostic taxonomy and the strength-of-evidence reporting are independent, auditable layers."),

  H2("3.9 Reproducibility Controls"),
  PJ("Reproducibility is treated as a first-class engineering requirement. A single seed (42) is propagated to Python's random module, NumPy, and PyTorch RNGs at the start of each run. Dataset shuffling is deterministic: each benchmark plugin loads its full dataset, shuffles with the seeded RNG, and then truncates to the configured max_samples. The same prompt order is therefore visited by both pair members."),
  PJ("Every per-prompt record persisted to raw.jsonl includes the model alias, the resolved model_id, the model family, the pair_id, the quantized flag, the seed, the full generation_config, and an ISO-8601 timestamp. This per-record metadata supports later auditability without dependence on the configuration file at the time of analysis."),
  PJ("Long-running jobs are made robust by the resume logic described in §4.7: if a job is killed mid-benchmark by walltime exhaustion or transient cluster failure, the next submission reads the existing raw.jsonl, identifies which prompt_ids have already been processed, and skips them on the second pass. The granularity of the resume mechanism is per-prompt, not per-benchmark."),
];

// ------------------------------------------------------------
// Chapter 4: System Design and Implementation
// ------------------------------------------------------------
const ch4 = [
  H1("Chapter 4: System Design and Implementation"),

  H2("4.1 Architecture Overview"),
  PJ("The framework is implemented as a single Python package, ethical_benchmark, with a thin command-line wrapper (fyp_cli.py) and a Makefile that exposes the most common workflows. The package is decomposed into modules with single, well-defined responsibilities, summarised in Table 4.1."),
  buildTable(
    ["Module", "Responsibility"],
    [
      ["ethical_benchmark/quant/config_schema.py", "Pydantic schema for the YAML configuration; enforces pair invariants and benchmark cross-references."],
      ["ethical_benchmark/models/loader.py", "HFModelLoader: dtype resolution, device selection, on-the-fly BitsAndBytesConfig injection, post-load verification."],
      ["ethical_benchmark/models/generation.py", "TextGenerator: batched generation, chat-template application, enable_thinking handling, post-processing."],
      ["ethical_benchmark/benchmarks/", "Pluggable benchmark interface: base.py (ABC), harmbench.py, xstest.py, mmlu.py, utils.py (shared refusal patterns)."],
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

  H2("4.2 Configuration Schema"),
  PJ("All runtime parameters are specified by a YAML configuration file (typically configs/tc1.yaml for cluster runs, configs/default.yaml for local development). The configuration is loaded and validated by a Pydantic schema (config_schema.py) that enforces structural and semantic invariants. The top-level schema has four sections: models, decoding, benchmarks, and slurm."),
  PJ("Each model entry must specify family, size_b, quantized, pair_id, model_id, trust_remote_code, dtype, and a benchmarks list, plus an optional attn_implementation (the attention backend passed to from_pretrained; restricted to eager, sdpa, or flash_attention_2, and omitted entirely when unset so the existing models' load call is unchanged). The schema enforces two cross-references: every benchmark name appearing in a model's benchmarks list must also appear as a top-level benchmarks entry, and every pair_id must have at least one model with quantized=false and at least one with quantized=true. These invariants make it impossible to configure a study that omits a baseline or compares a model only against itself."),
  PJ("The decoding section specifies generation parameters; the benchmarks section provides per-benchmark dataset names, splits, sample caps, batch sizes, and benchmark-specific options (such as the benign_only flag for XSTest and the subjects list for MMLU); the slurm section specifies cluster directives and bootstrap commands."),

  H2("4.3 Model Loading and Quantization Path"),
  PJ("The HFModelLoader class encapsulates all model-loading logic. Its load method accepts a ModelSpec dataclass (built from the YAML configuration) and returns the loaded model, tokenizer, and resolved runtime device. The loader is responsible for three things: (i) selecting the runtime device based on the user-specified policy and CUDA availability; (ii) resolving the model dtype from a string preference (auto, float16, bfloat16, float32) to an actual torch.dtype value, with auto resolving to float16 on CUDA and float32 on CPU; and (iii) injecting a BitsAndBytesConfig object into the from_pretrained call when ModelSpec.quantized is true."),
  PJ("The BitsAndBytesConfig is constructed with load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_use_double_quant=True, and bnb_4bit_compute_dtype equal to the resolved torch_dtype. Setting the compute dtype to follow the model dtype ensures that the dequantized compute path uses the same precision the model would have used at full precision, eliminating a subtle source of pair asymmetry."),
  PJ("Two runtime safeguards are applied. First, the loader raises a clear RuntimeError if quantized is true but the resolved runtime device is not cuda, preventing accidental CPU execution of a four-bit path that BitsAndBytes does not support. Second, after the from_pretrained call returns, the loader checks model.is_loaded_in_4bit and logs a warning if the flag is false despite a quantization request. This protects against silent failures in which a checkpoint or installation issue causes BitsAndBytes to skip the quantization."),

  H2("4.4 Generation Pipeline"),
  PJ("The TextGenerator class handles all inference. It accepts the loaded model, tokenizer, runtime device, and a DecodingConfig dataclass, and exposes a single generate_batch method that takes a list of prompts and returns a list of generated responses."),
  PJ("Each prompt passes through a formatting step. When use_chat_template is true (the default), the generator wraps the prompt in a single-message user-role list and applies the tokenizer's apply_chat_template method with add_generation_prompt=True. Critically, enable_thinking=False is also passed to this call so that Qwen3.x-family tokenizers, whose chat templates default to enabling a multi-step thinking block, do not silently consume the entire max_new_tokens budget producing <thinking>...</thinking> output before reaching the answer. A TypeError handler catches tokenizers that do not accept the enable_thinking keyword and retries without it, falling back to a raw-prompt path only as a last resort."),
  PJ("Generation itself uses torch.inference_mode and a do_sample=False decode path under temperature=0.0. Outputs are post-processed to strip the chat-template prefix and any trailing whitespace before being returned to the calling pipeline."),

  H2("4.5 Benchmark Plugins"),
  PJ("Benchmarks are implemented as plugins behind a small abstract base class defined in ethical_benchmark/benchmarks/base.py. Every plugin implements four methods: load_items, build_prompt, score_response, and aggregate. load_items returns a deterministic, seeded sample of BenchmarkItem objects; build_prompt maps each item to a single user-facing prompt string; score_response evaluates a model response against the item and returns a dictionary of boolean and numeric score fields; aggregate consumes the full list of per-item score dictionaries and produces the final aggregated summary."),
  PJ("Refusal detection is shared between HarmBench and XSTest through the benchmarks/utils.py module, which exposes a single classify_refusal function backed by a curated set of regex patterns. Centralising the refusal logic in a single function guarantees that HarmBench and XSTest see exactly the same notion of \"refusal\", a non-trivial property since these two benchmarks measure opposite sides of the same phenomenon."),
  PJ("The conceptual scoring loop, omitting error handling, is:"),
  ...Code(`items = plugin.load_items(max_samples=N, seed=42)
prompts = [plugin.build_prompt(item) for item in items]
responses = generator.generate_batch(prompts)
scored = [plugin.score_response(it, r) for it, r in zip(items, responses)]
summary = plugin.aggregate(scored)
write_jsonl(per_item_records, raw_path)
write_json(summary, summary_path)`),

  H2("4.6 Matrix Orchestration and Memory Management"),
  PJ("run_quant_matrix.py executes a full study or a filtered subset of it. The outer loop iterates over selected model aliases; the inner loop iterates over the benchmarks declared for each model. With the default reuse_loaded_model=True policy, each model is loaded exactly once and used for all of its benchmarks in sequence. This avoids two redundant load cycles per model (the most expensive operation in the pipeline) and keeps total wall-clock time well within the six-hour TC1 walltime budget."),
  PJ("Memory management is critical on TC1, where each job is allocated only ten gigabytes of host RAM and a single GPU. A try/finally block surrounds the inner benchmark loop. When the inner loop completes (or any benchmark within it raises), the model, tokenizer, and generator references are explicitly deleted, gc.collect() is invoked, and torch.cuda.empty_cache() is called when CUDA is available. This sequence releases the GPU memory before the next model is loaded, preventing stacking-induced out-of-memory errors that would otherwise occur immediately when loading a second model into the same job (a case that does arise in local development but is structurally avoided on TC1 because each job runs a single model)."),

  H2("4.7 Resume and Checkpointing"),
  PJ("Long-running jobs need to be robust to walltime exhaustion and transient failures. The framework treats raw.jsonl as the source of truth for which prompts have already been processed: every successfully scored prompt is immediately appended to raw.jsonl, with its prompt_id as a stable record key. On startup, the get_processed_prompt_ids helper reads the existing raw.jsonl (if any) and returns the set of prompt_ids already on disk. prepare_remaining_items then filters the freshly-loaded benchmark items, keeping only those whose prompt_ids are not in the processed set."),
  PJ("The granularity of the resume mechanism is the (model, benchmark, prompt_id) triple. Three concrete recovery scenarios are worth stating explicitly. If a job is killed during XSTest for qwen_2b_4bit, HarmBench (whose raw.jsonl is complete) is skipped entirely on the next submission; XSTest (whose raw.jsonl is partial) is resumed at the next unprocessed prompt; MMLU (whose raw.jsonl does not yet exist) runs from scratch. If a model fails to load (a deterministic OOM, for example), the entire model's benchmark suite is lost for that job; the operator can re-submit with --model and --benchmark filters to chunk the workload differently."),

  H2("4.8 SLURM Job Generation"),
  PJ("Cluster orchestration is handled by ethical_benchmark/cluster/generate_jobs.py. The generator reads the loaded configuration's slurm section and emits one sbatch file per scheduling unit. A --group_by flag controls the granularity: with group_by=benchmark, one sbatch is generated per (model, benchmark) pair (twenty-four scripts for the six-model, four-benchmark matrix); with group_by=model (the default for TC1), one sbatch is generated per model alias, and that script invokes run_quant_matrix.py so the model is loaded only once."),
  PJ("Each generated sbatch contains the standard SBATCH directives (partition UGGPU-TC1, qos normal, gres gpu:1, cpus-per-task 1, mem 10G, time 06:00:00), output and error log paths, a set -euo pipefail line, a cd into the configured work_dir, a mkdir -p of the log directory, the configured setup_commands (module load slurm, module load anaconda, source activate fyp-tc1), and finally the python invocation with the model alias and configuration path passed in. Six sbatch files are produced for the present configuration, one per model alias."),

  H2("4.9 Output Artifacts"),
  PJ("Every (model, benchmark) run produces a fixed set of files under the results directory. raw.jsonl contains one JSON object per prompt with the prompt text, the model response, the per-response score fields, and the run metadata (model alias, model_id, family, pair_id, quantized flag, seed, generation_config, timestamp). summary.json contains the aggregated metrics, with bootstrap confidence intervals where applicable, alongside the same run metadata. The aggregator also appends a flat row to results/summary/<benchmark>_runs.csv, producing a single CSV per benchmark that records all model runs."),
  PJ("The analysis stage produces results/analysis/pairwise_deltas.json and .csv (one row per (pair_id, benchmark, metric) with absolute and relative deltas), results/analysis/pair_interpretations.csv (one row per pair with the interpretation label and the three component deltas), and results/analysis/quantization_analysis_summary.json (high-level study-wide summary)."),

  H2("4.10 Testing"),
  PJ("The repository ships with a verification suite of 329 automated tests across twenty-five test files. The full distribution is recorded in Appendix D. Coverage areas include the dataset and benchmark loaders, the legacy evaluators (retained for backward compatibility), model loader specifics including dtype resolution and quantized-flag propagation, the prompt-formatting logic with explicit tests for enable_thinking handling, the matrix-reuse behaviour (verifying that reuse_loaded_model=True loads each model only once), the analysis module's pairwise delta computation, bootstrap CI logic, and v2 score-sidecar selection, the per-prompt schema validator, the resume-helper functions, the refusal-parser regex correctness (including regression tests for the v2 expanded patterns and curly-apostrophe handling), the judge-model validation layer (stub-backend scoring, sidecar redaction enforcement, ASR aggregation, raw-immutability, and an official-template regression check), and the SLURM job generator including the per-benchmark and per-model grouping modes. All 329 tests pass on the current commit."),

  H2("4.11 CLI and Operational Interface"),
  PJ("The unified CLI fyp_cli.py exposes seven subcommands, summarised in Table 4.2. Each subcommand accepts the common options --config, --results_dir, and --log_level, plus a small number of subcommand-specific arguments. The Makefile provides convenient targets that wrap these invocations."),
  buildTable(
    ["Subcommand", "Purpose"],
    [
      ["smoke", "Run a small-sample sanity check on one (model, benchmark) pair."],
      ["run", "Run one (model, benchmark) pair with the full sample budget."],
      ["matrix", "Run the full or a filtered subset of the (model × benchmark) matrix."],
      ["analyze", "Compute pairwise deltas and produce interpretation labels."],
      ["cluster-generate", "Emit SLURM sbatch files for the configured study."],
      ["cluster-submit", "Submit the generated sbatch files; --dry_run prints planned submissions."],
      ["cluster-check", "Poll squeue and the filesystem to report job and output status."],
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
  PJ("All experiments are run on the NTU TC1 GPU cluster, a shared facility operated by the College of Computing and Data Science for undergraduate and postgraduate research workloads. Account access was approved in March 2026 under QoS \"normal\" with the parameters listed in Table 5.1. The compute partition consists of seven nodes (TC1N01–TC1N07), each equipped with three NVIDIA Tesla V100 PCIe 32 GB GPU cards, giving twenty-one GPUs across the partition. Although MaxJobsPU permits two submitted jobs, the first production submission showed an effective one-running-GPU-job limit: the second GPU job waited with reason QOSMaxGRESPerUser until the first released its GPU."),
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
  PJ("A dedicated conda environment, fyp-tc1, was created on TC1 and validated by importing the principal ML stack. The relevant versions are listed in Table 5.2. Per the TC1 user guide, the base anaconda environment cannot be modified; all study dependencies are isolated in fyp-tc1 under the user's home directory."),
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
      ["pytest", "(installed via requirements.txt; 329 tests passing)"],
    ],
    [4200, 5160],
  ),
  P("Table 5.2  Software environment versions on TC1.", { size: 18 }),

  H2("5.3 Cluster Usage Policy and Workflow Constraints"),
  PJ("The TC1 user guide imposes strict workflow rules that shape how this study is executed. The headline rule is that user code may not be executed on the head node (CCDS-TC1). The guide states: \"DO NOT execute your coding on TC1 Head Node ... For the user process found executing coding and occupying high CPU and Memory usage in the Head Node will be terminated with no prior notification. Repeated offenders will be banned from TC1.\" GPU verification commands such as nvidia-smi and nvcc --version are also explicitly forbidden on the head node, because the GPU cards reside on the compute nodes (TC1N01–TC1N07) and not on the head node."),
  PJ("By contrast, package installation, file transfers, conda environment management, dataset downloads, and small administrative commands (squeue, scontrol, sacct, MyTCinfo, MyJobHistory, seff) are explicitly demonstrated as head-node activities throughout the user guide. The study's pre-cache step (see §5.5) therefore runs entirely on the head node and is policy-equivalent to a conda install."),
  PJ("The guide also recommends sbatch over srun for all real job submission: \"Avoid using the command 'srun' to submit job ... all users are advised to use the command 'sbatch' for job submission. Then exit from the session, access later to see the result.\" This study therefore submits every job, including the initial smoke validation, via sbatch."),

  H2("5.4 Hugging Face Access and Gating"),
  PJ("The Qwen pairs draw from open-access official Qwen3 checkpoints (Qwen/Qwen3-1.7B and Qwen/Qwen3-4B) and require no authentication. The Llama pair uses meta-llama/Llama-3.2-3B-Instruct, which is gated under the Meta Llama community license, and HarmBench currently also requires accepted Hugging Face dataset access conditions. Access is handled by logging into Hugging Face once on the TC1 head node with a read-scoped personal access token. As of 2026-05-26, token registration and gated-access acceptance have been verified by a successful full pre-cache of both Llama 3.2 3B and HarmBench."),

  H2("5.4.1 Infrastructure Validation and Fixes"),
  PJ("Before the full matrix was submitted, a dedicated CUDA verification job and a five-sample smoke run were used to validate the cluster environment end-to-end. Two infrastructure bugs were identified and fixed during this process. First, the generated sbatch scripts used relative paths for the #SBATCH --output and #SBATCH --error directives. SLURM resolves these relative to the directory from which sbatch is invoked, not to the working directory set by cd inside the script body. Jobs submitted from the home directory therefore wrote their log files to ~/results/slurm_logs_tc1/ rather than to the repository's results/ tree, causing the output to appear missing. The fix was applied in the job generator (ethical_benchmark/cluster/generate_jobs.py): when log_dir is a relative path and work_dir is set, the generator now anchors log_dir under work_dir so all sbatch directives contain fully qualified paths. Second, an initial CUDA verification script placed a Python file in /tmp on the head node and referenced it from the sbatch body; the /tmp filesystem is local to each node and is not shared, so the compute node could not find the file. The fix was to embed the verification code inline in the sbatch script using a Python heredoc (python - << 'PYEOF' ... PYEOF), removing the dependency on any shared path. Both fixes were committed and pushed before the smoke run, and the five-sample smoke job (SLURM job ID 60975) completed successfully: runtime_device reported cuda, attack_success_rate was 0.6 over five HarmBench prompts, and no malformed outputs were produced."),

  H2("5.5 Offline-Mode Strategy and Pre-Cache"),
  PJ("Because the compute nodes may not have outbound internet access, and because runtime downloads risk burning the six-hour walltime budget on slow Hugging Face mirrors, this study adopts a strict offline-mode strategy. All datasets and model weights are pre-cached on the head node before any SLURM job is submitted. SLURM jobs then run with HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, and TRANSFORMERS_OFFLINE=1 exported in the sbatch setup_commands block, ensuring that any cache miss fails immediately with a clear error rather than hanging on a network attempt."),
  PJ("The repository provides a helper script (scripts/prefetch_tc1.py) that reads the configured benchmarks and models from configs/tc1.yaml and triggers the necessary downloads. It is invoked once, on the head node, after the conda environment is activated and a Hugging Face token has been registered with huggingface_hub.login:"),
  ...Code(`module load anaconda
source activate fyp-tc1
cd /tc1home/FYP/utan001/fyp_quant/repo
python - <<'PY'                      # one-time HF login
from getpass import getpass
from huggingface_hub import login
login(token=getpass("HF token: "), add_to_git_credential=False)
PY
python scripts/prefetch_tc1.py       # or: make prefetch CONFIG=configs/tc1.yaml`),
  PJ("The pre-cache step retrieves HarmBench, the six MMLU subjects, ARC-Challenge (allenai/ai2_arc, added as the second capability benchmark), and three model repositories (Qwen3-1.7B, Qwen3-4B, Llama 3.2 3B). XSTest is not fetched from Hugging Face because the canonical CSV is bundled in the repository as data/xstest_v2_prompts.csv. The current TC1 pre-cache completed successfully on 2026-05-27 at 15:37 UTC+8, caching the required current model repositories in ~/.cache/huggingface/hub: Qwen3-1.7B (4.08 GB), Qwen3-4B (8.06 GB), and Llama 3.2 3B (12.9 GB), plus small dataset files. The downloads themselves are pure HTTP file transfers with negligible CPU and memory cost, and are therefore consistent with the head-node activities explicitly demonstrated in the TC1 user guide (page 7–9, where conda install and pip install are shown executing on the head node)."),

  H2("5.6 Run Plan"),
  PJ("With group_by=model, the framework emits six sbatch files, one per model alias. Each script runs the full configured benchmark suite for its model with the model loaded only once, exploiting the matrix runner's reuse_loaded_model=True default. (The initial matrix run covered the three core benchmarks; ARC-Challenge was added later as the second capability benchmark and run on the original six models via a parallel per-model job set, slurm/jobs_tc1_arc/, with the two cross-family pairs covering ARC inside their matrix jobs; see §6.4.1.) The six scripts are:"),
  Bullet("qwen_2b_base__matrix.sbatch"),
  Bullet("qwen_2b_4bit__matrix.sbatch"),
  Bullet("qwen_4b_base__matrix.sbatch"),
  Bullet("qwen_4b_4bit__matrix.sbatch"),
  Bullet("llama_3_2_3b_base__matrix.sbatch"),
  Bullet("llama_3_2_3b_4bit__matrix.sbatch"),
  PJ("Before submitting the full matrix, a single short smoke sbatch is submitted (five prompts, qwen_2b_base on HarmBench) to verify that the offline-cache path works end-to-end on a real compute node. Per the user guide's guidance to prefer sbatch over srun, this smoke verification is performed as a regular SLURM job rather than an interactive session. On 2026-05-27, smoke job 60975 completed successfully on TC1 in 33 seconds, produced a clean summary.json with runtime_device=cuda and malformed_rate=0.0, and therefore cleared the full six-job matrix for submission."),
  PJ("The first production matrix submission was made on 2026-05-27 with the Qwen 2B-class pair: job 60976 (qwen_2b_base__matrix) began running while job 60977 (qwen_2b_4bit__matrix) remained pending with reason QOSMaxGRESPerUser. This establishes the practical scheduling rule for the remainder of the study: submit only the current model pair and expect one job to run while the paired job waits; submit the next pair only after the current pair has cleared. Under this effective one-GPU concurrency, the six model jobs run serially or near-serially, still within the study plan because each job has an independent six-hour walltime allocation. Memory and CPU utilisation are recorded after each job using the seff and MyJobHistory commands to inform any subsequent right-sizing of the sbatch resource requests."),

  H2("5.7 Reproducibility Notes"),
  PJ("All runs share the same global seed (42), deterministic dataset shuffling, and deterministic decoding. The exact commit hash of the framework at the time of the final result run will be recorded in the final report and will accompany the result tables. Per-prompt records are persisted to raw.jsonl with full metadata, so the entire study can be re-executed (or selectively rerun) from any future repository checkout."),
];

// ------------------------------------------------------------
// Chapter 6: Results and Analysis
// ------------------------------------------------------------
const ch6 = [
  H1("Chapter 6: Results and Analysis"),

  PJ("All six original experimental runs completed successfully on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976–60981). Three matched pairs were evaluated: Qwen3-1.7B (jobs 60976/60977), Qwen3-4B (jobs 60978/60979), and Llama 3.2 3B-Instruct (jobs 60980/60981). A cross-family extension subsequently added two further matched pairs (Mistral-7B-Instruct-v0.3 and Phi-4-mini-instruct) run on the same cluster on 2026-06-15 (matrix jobs 61121/61122/61123/61125; HarmBench classifier job 61134), taking the study to five pairs / ten models / four families; their results are folded into Tables 6.1–6.3 and analysed in §6.13. Each job ran all three core benchmarks sequentially with the model loaded once; ARC-Challenge (the second capability benchmark) was added afterwards and run on the original six models in a separate per-model job set, with the two cross-family pairs covering ARC inside their matrix jobs (§6.4.1). The results are presented in Table 6.1 (per-pair raw metrics) and Table 6.2 (quantization deltas and interpretation labels). Detailed per-pair observations are in §6.6 (Qwen3-1.7B), §6.8 (Qwen3-4B), §6.10 (Llama-3.2-3B), and §6.13 (Mistral-7B and Phi-4-mini). The within-family scale analysis (RQ4) is in §6.2 and §6.9. The cross-family comparison (RQ5) is in §6.11, extended across four families in §6.13."),

  H2("6.1 Results Table"),
  PJ("Table 6.1 records the per-pair benchmark results for all five model pairs together with bootstrap 95% confidence intervals on the deltas. HarmBench Attack Success Rate is reported under the official HarmBench classifier (the primary scorer; see §6.1.1 and §6.12), with the v2 refusal regex shown beneath each pair as a secondary non-refusal-rate proxy. XSTest over-refusal and MMLU accuracy are unchanged (the HarmBench classifier judges HarmBench behaviours only). Confidence intervals are computed by a paired bootstrap [24] over the matched prompt set (2 000 resamples, seed 42); both members of every pair see the same prompts in the same order, so each bootstrap draw resamples prompt indices and recomputes the delta directly. A delta is marked statistically significant when its 95% CI excludes zero."),
  buildTable(
    ["Pair", "Metric (scorer)", "Baseline", "4-bit", "Δ (95% CI)", "Sig?"],
    [
      ["qwen_2b", "HarmBench ASR (judge)", "0.135", "0.190", "+0.055 [+0.010, +0.100]", "yes"],
      ["qwen_2b", "  HarmBench non-refusal (v2 proxy)", "0.600", "0.575", "−0.025 [−0.070, +0.020]", "no"],
      ["qwen_2b", "XSTest over-refusal", "0.052", "0.028", "−0.024 [−0.052, 0.000]", "no"],
      ["qwen_2b", "MMLU accuracy", "0.620", "0.533", "−0.087 [−0.137, −0.037]", "yes"],
      ["qwen_4b", "HarmBench ASR (judge)", "0.065", "0.090", "+0.025 [−0.000, +0.055]", "no"],
      ["qwen_4b", "  HarmBench non-refusal (v2 proxy)", "0.240", "0.305", "+0.065 [+0.025, +0.110]", "(proxy)"],
      ["qwen_4b", "XSTest over-refusal", "0.028", "0.020", "−0.008 [−0.024, +0.008]", "no"],
      ["qwen_4b", "MMLU accuracy", "0.747", "0.743", "−0.003 [−0.040, +0.033]", "no"],
      ["llama_3_2_3b", "HarmBench ASR (judge)", "0.040", "0.040", "0.000 [−0.020, +0.020]", "no"],
      ["llama_3_2_3b", "  HarmBench non-refusal (v2 proxy)", "0.060", "0.060", "0.000 [−0.025, +0.025]", "no"],
      ["llama_3_2_3b", "XSTest over-refusal", "0.044", "0.060", "+0.016 [−0.004, +0.040]", "no"],
      ["llama_3_2_3b", "MMLU accuracy", "0.610", "0.567", "−0.043 [−0.080, −0.007]", "yes"],
      ["mistral_7b", "HarmBench ASR (judge)", "0.385", "0.345", "−0.040 [−0.110, +0.025]", "no"],
      ["mistral_7b", "  HarmBench non-refusal (v2 proxy)", "0.835", "0.890", "+0.055 [+0.010, +0.100]", "(proxy)"],
      ["mistral_7b", "XSTest over-refusal", "0.004", "0.004", "0.000 [−0.012, +0.012]", "no"],
      ["mistral_7b", "MMLU accuracy", "0.630", "0.613", "−0.017 [−0.053, +0.020]", "no"],
      ["phi4_mini", "HarmBench ASR (judge)", "0.055", "0.055", "0.000 [−0.030, +0.030]", "no"],
      ["phi4_mini", "  HarmBench non-refusal (v2 proxy)", "0.075", "0.075", "0.000 [−0.025, +0.025]", "no"],
      ["phi4_mini", "XSTest over-refusal", "0.112", "0.084", "−0.028 [−0.052, −0.004]", "yes"],
      ["phi4_mini", "MMLU accuracy", "0.700", "0.677", "−0.023 [−0.057, +0.010]", "no"],
    ],
    [1200, 2300, 1000, 1000, 2900, 660],
  ),
  P("Table 6.1  Complete per-pair results with paired bootstrap 95% CIs on the deltas. \"Sig?\" = whether the CI excludes zero. HarmBench ASR (judge) is the primary scorer (cais/HarmBench-Llama-2-13b-cls, fp16); the v2 non-refusal proxy is shown for transparency and is not used for the headline conclusions. Judge CIs from `results/analysis/judge_agreement.json`; v2/MMLU/XSTest CIs from `results/analysis/pairwise_deltas.json`.", { size: 18 }),
  ...Figure("asr_forest.png", "Quantization effect on harmful compliance: per-pair judge ΔASR (4-bit − fp16) with paired-bootstrap 95% confidence intervals. Filled markers denote intervals that exclude zero; only Qwen3-1.7B reaches significance, and it worsens. Source: results/analysis/judge_agreement.json."),
  PJ("Under the primary (judge) scorer, the statistically significant deltas are: Qwen 1.7B ΔASR (+0.055, CI [+0.010, +0.100]), Qwen 1.7B ΔMMLU (−0.087), Llama 3.2 3B ΔMMLU (−0.043), and Phi-4-mini Δover-refusal (−0.028, a significant decrease). The Qwen 1.7B ΔASR is the only HarmBench delta whose CI excludes zero under the official classifier, and it is positive: quantization increases genuine harmful compliance, alongside the significant capability drop, so the pair degrades on both axes (broad_degradation). The Qwen 4B ΔASR is positive but directional only (+0.025, CI touches zero). The Llama ΔASR is essentially zero. Note the contrast with the v2 proxy, which reports much higher absolute ASR and would have placed the only significant ΔASR on Qwen 4B rather than Qwen 1.7B. The divergence is analysed in §6.12. Four of the five XSTest over-refusal deltas are non-significant under both scorers; only Phi-4-mini's is significant, and as a decrease, so no pair becomes more over-refusing under quantization (RQ2)."),

  H3("6.1.1 Scorer revision history"),
  PJ("Scope under the judge-primary framing (D16). This section documents the refusal regex's own v1→v2 revision history, which bears on two things only: the XSTest over-refusal figures (produced by the v2 expanded refusal classifier, `ethical_benchmark/benchmarks/utils.py`, scorer version `v2_expanded_refusal_patterns_2026-05-28`) and the secondary HarmBench non-refusal proxy. HarmBench Attack Success Rate throughout this report is the official HarmBench classifier value (the primary scorer; §6.12), not the regex; MMLU and ARC use deterministic exact match. With that scope established: the v1 classifier (pre-2026-05-28) used a narrow set of 14 refusal patterns that missed the most common modern refusal templates, including \"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", and curly-apostrophe variants used by Qwen3 output. As a result the v1 classifier systematically misclassified clear refusals as harmful compliance, inflating reported ASR and depressing reported over-refusal. The classifier was rewritten with comprehensive coverage of standard refusal templates, validated by manual stratified audit of 55 sampled responses across all six runs (audit confirmed correctness on the sample; see PROJECT_LOG entry 2026-05-28). The TC1-original `raw.jsonl` and `summary.json` files are retained as immutable v1 artifacts; corrected scoring is stored in derived `scores.v2.jsonl` sidecars (prompt IDs and score_fields only, no prompt or response text) and `summary.v2.json` files. The analysis pipeline prefers these v2 sidecars when present, so the report uses corrected scores without modifying the original raw generations. The auxiliary artifacts `results/analysis/rescore_diagnostics_*.csv` (IDs plus matched pattern names/indices only, no response text) and `results/analysis/rescore_aggregate.json` document the per-record reclassification for full auditability."),
  PJ("Impact of the rescore on the v2 numbers. Within the regex scorer, three pair-level deltas changed materially from v1 to v2: Qwen 1.7B ΔASR moved from −0.120 (v1) to −0.025 (v2), where the v1 \"significant decrease\" was largely an artefact of unrecognised refusals; Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2); Llama 3.2 3B ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). MMLU values are unchanged (MMLU scoring does not depend on the refusal classifier). This history explains why the v2 regex is retained only as a secondary proxy: even after the v1→v2 correction, the regex measures \"non-refusal rate\", not genuine harmful compliance. The subsequent judge validation (§6.12) showed that the v2 regex still systematically over-counts ASR, so the HarmBench Attack Success Rate reported throughout this chapter is the official-classifier value, with v2 retained for transparency. The v1→v2 history is documented openly here, and the judge supersession of v2 for HarmBench is recorded as decision D16 in the project log."),

  H2("6.2 Within-Family Scale Analysis (RQ4)"),
  PJ("RQ4 asks whether the magnitude of the quantization effect differs between two-billion and four-billion parameter Qwen models. Both pairs have now completed. Table 6.2 summarises the quantization deltas and interpretation labels for all five pairs, enabling both the within-family scale comparison (RQ4, the two Qwen pairs) and the cross-family comparison (RQ5)."),
  buildTable(
    ["Metric", "qwen_2b", "qwen_4b", "llama_3.2_3b", "mistral_7b", "phi4_mini"],
    [
      ["HarmBench ΔASR (judge)", "+0.055  ★", "+0.025", "0.000", "−0.040", "0.000"],
      ["  HarmBench ΔASR (v2 proxy)", "−0.025", "+0.065", "0.000", "+0.055", "0.000"],
      ["  HarmBench McNemar exact p", "0.027", "0.18", "1.00", "0.31", "1.00"],
      ["XSTest Δover-refusal", "−0.024", "−0.008", "+0.016", "0.000", "−0.028  ★"],
      ["MMLU Δaccuracy", "−0.087  ★", "−0.003", "−0.043  ★", "−0.017", "−0.023"],
      ["Interpretation label (judge-primary)", "broad_degradation", "alignment_degradation", "broad_degradation", "alignment_improvement", "robust_preservation"],
      ["Evidence status (two-layer)", "confirmed", "directional", "confirmed", "directional", "null"],
    ],
    [2200, 1240, 1240, 1480, 1320, 1320],
  ),
  P("Table 6.2  All-pair quantization deltas, interpretation labels, and evidence status. Δ = 4-bit − baseline. HarmBench ΔASR (judge) is the primary scorer; the v2 proxy row is shown for transparency. ★ marks deltas whose paired-bootstrap 95% CI excludes zero. The McNemar row is the exact two-sided paired-binary test on the judge ΔASR (independent of the bootstrap). Under the judge, three deltas are significant: Qwen 1.7B ΔASR (also McNemar p = 0.027), Qwen 1.7B ΔMMLU, and Llama ΔMMLU. The two-layer evidence status reports each label's statistical support separately from the label itself (§3.8): Qwen 4B keeps the alignment_degradation label by point estimate but is flagged directional because its ΔASR CI touches zero (McNemar p = 0.18); it is not a statistically confirmed regression. Labels are judge-primary (D16).", { size: 18 }),
  PJ("Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization procedure, the 1.7B model experiences large capability degradation (MMLU drops 8.7 percentage points, relative decline of 14.0%, CI excludes zero) while the 4B model is essentially unaffected, with MMLU moving by only 0.3 percentage points, well within sampling noise. The MMLU delta ratio (8.7 pp versus 0.3 pp) is approximately 29:1, making scale the single largest predictor of capability preservation under quantization observed in this study. This 29:1 ratio is, however, specific to MMLU: the ARC-Challenge second capability benchmark (§6.4.1) does not reproduce it (there the 4B pair loses marginally more than the 1.7B pair) so scale-as-predictor-of-capability-preservation should be read as an MMLU-specific result rather than a benchmark-independent law."),
  PJ("The full-precision baselines reveal that within the Qwen family the larger model is both more capable and more refusal-calibrated. Under the primary (judge) scorer, Qwen3-1.7B's baseline HarmBench ASR is 0.135 while Qwen3-4B's is 0.065, so scaling from 1.7B to 4B reduces genuine baseline harmful compliance by 7.0 percentage points. MMLU accuracy simultaneously rises from 0.620 to 0.747 (+12.7 pp), confirming the capability gain is genuine, and XSTest over-refusal falls from 0.052 to 0.028, indicating fewer false refusals on benign prompts. The 4B baseline therefore dominates the 1.7B baseline on every dimension. (Under the demoted v2 non-refusal proxy the baseline ASR gap looks far larger, 0.600 versus 0.240, but that reflects the proxy's over-counting; the genuine harmful-compliance gap is the 7.0 pp judge figure, see §6.12.) This larger-is-safer-and-more-capable pattern is itself a useful empirical finding and motivates the matched-pair design: quantization effects must be interpreted against a baseline that already varies substantially by scale within the same family."),
  PJ("The most important comparison is on the post-quantization side, read under the primary (judge) scorer. The 1.7B pair worsens on both axes: judge ΔASR +5.5 pp (CI [+0.010, +0.100], significant; McNemar p = 0.027) together with a significant ΔMMLU of −8.7 pp, the broad_degradation pattern. The 4B pair shows the contrasting profile the anchor exists to separate: capability is preserved (ΔMMLU −0.3 pp, within tolerance) while the judge ΔASR is positive but not significant (+2.5 pp, CI [−0.000, +0.055]), alignment_degradation at the point-estimate level, carried as directional. Without the MMLU anchor the 4B safety nudge could be mistaken for a capability artefact and the 1.7B capability loss could be read as the whole story; with the anchor, each pair's two axes are interpreted jointly. This is the methodological payoff of the capability-anchored design. (The demoted v2 proxy paints a different and misleading picture here, a −2.5 pp 1.7B ΔASR and a +6.5 pp 4B ΔASR; §6.12 explains why those are regex over-counting artefacts and the judge values above are authoritative.)"),

  H2("6.3 Cross-Family Replication (RQ5)"),
  PJ("RQ5 asks whether the Qwen pattern replicates qualitatively in Llama 3.2 3B. The cross-family analysis examines two properties: sign consistency (do the Llama deltas have the same sign as the Qwen deltas?) and approximate magnitude (are the Llama deltas similar in scale?). The full cross-family comparison is in §6.11; this section provides a high-level orientation."),
  PJ("The headline finding for RQ5 is that, under the official classifier, NF4 quantization never reduces genuine harmful compliance and the one significant effect is in the smallest model. The Qwen 1.7B pair shows a significant ΔASR increase of +0.055 (CI [+0.010, +0.100]) alongside large capability loss; the Qwen 4B pair shows a directional, non-significant +0.025 at preserved capability; the Llama 3.2 3B pair shows ΔASR exactly at zero, paired with a significant ΔMMLU of −0.043. The capability dimension is directionally consistent (all three pairs have negative ΔMMLU; two significant, the Qwen 4B value within noise). On over-refusal, no ΔOR is significant. The composite picture is that NF4 quantization does not produce uniform safety effects across families and scales, no ΔASR is a significant increase outside the smallest Qwen model, and the one robust signal (Qwen 1.7B ASR increase) goes in the safety-worsening direction. See §6.11 for the full discussion and §6.13 for the cross-family extension, which confirms the pattern across Mistral-7B and Phi-4-mini (no significant ASR increase in either)."),

  H2("6.4 Capability Anchoring"),
  ...Figure("capability_anchor.png", "The capability-anchored safety space. Each pair is placed by its capability delta (ΔMMLU, x-axis) and its harmful-compliance delta (judge ΔASR, y-axis); dashed lines mark the interpretation thresholds and the shaded quadrants name the diagnostic labels. Bars are paired-bootstrap 95% CIs. The anchor is what separates the broad_degradation pairs (top-left) from a capability-preserving alignment nudge (Qwen3-4B) and from Mistral-7B's capability-preserving safety improvement (bottom-right)."),
  PJ("Capability anchoring is the core methodological contribution of the interpretation framework. Each pair's MMLU delta is used as a covariate when interpreting its HarmBench and XSTest deltas. Six canonical outcomes are tracked, organised symmetrically around both the safety and over-refusal axes:"),
  Bullet("Alignment degradation: MMLU is preserved, but HarmBench ASR rises beyond threshold; the model becomes more compliant with harmful prompts without a corresponding capability change. A direct safety regression."),
  Bullet("Alignment improvement: MMLU is preserved, but HarmBench ASR falls beyond threshold; quantization reduces harmful compliance without measurable capability cost. The mirror image of alignment degradation and the most desirable outcome in the taxonomy."),
  Bullet("Capability-collapse masquerading as safety: MMLU drops noticeably, and HarmBench ASR also drops; the apparent safety improvement is attributed to reduced instruction-following capacity rather than a genuine alignment change."),
  Bullet("Robust preservation: all three deltas are small in magnitude; quantization is effectively neutral across safety and capability dimensions."),
  Bullet("Broad degradation: fallback. Typically signals over-refusal moving outside tolerance, or ASR within tolerance but capability dropping. Used when no specific pattern above matches."),
  PJ("The three pairs in this study illustrate why the capability anchor is necessary, even though, under the primary (judge) scorer, two of them land on the same label. For the qwen_2b pair (Qwen 1.7B), MMLU fell by 8.7 pp (−14.0% relative; CI [−0.137, −0.037], significant) and the judge HarmBench ASR rose significantly (+5.5 pp, CI [+0.010, +0.100]; McNemar p = 0.027). Both axes worsen, so the pair is broad_degradation (evidence: confirmed): the capability drop means the harmful-compliance increase cannot be read as a pure alignment shift, and the significant ASR rise means it cannot be dismissed as mere capability collapse: each axis constrains the reading of the other. For the qwen_4b pair, MMLU is preserved (−0.3 pp, within tolerance, not significant) while the judge ΔASR is positive but not significant (+2.5 pp, CI [−0.000, +0.055]; McNemar p = 0.18). Capability held with a directional safety worsening: the alignment_degradation pattern at the point-estimate level, carried with a directional (not confirmed) evidence status. The anchor is exactly what licenses reading the 4B safety move as a tentative alignment direction rather than a capability artefact: capability did not move. For the llama_3_2_3b pair, MMLU drops 4.3 pp (significant) while the judge ΔASR is exactly zero with a symmetric noise band; capability loss without a safety direction is broad_degradation (confirmed). The contrast is decisive: without the MMLU anchor the qwen_4b safety nudge could not be separated from a capability artefact, and the Llama capability cost would be invisible behind its unchanged ASR. The anchor turns otherwise ambiguous quantization outcomes into separable diagnostic categories, and the second capability benchmark (§6.4.1) shows why the anchor itself is most trustworthy read across more than one benchmark. (Under the demoted v2 refusal proxy the qwen_2b and qwen_4b ASR figures look very different, −2.5 pp and +6.5 pp respectively; §6.12 explains why those are regex artefacts and the judge values above are authoritative.)"),

  H3("6.4.1 Capability robustness check: the ARC-Challenge second benchmark"),
  PJ("Because the capability anchor rests on a single MMLU subset, a structurally different second capability benchmark, ARC-Challenge (allenai/ai2_arc; 1,172 reasoning-oriented science questions; identical exact-match scoring), was run on all ten models to test whether the MMLU-based capability claims replicate (results/analysis/pairwise_deltas, benchmark = arc; same 2,000-resample paired-bootstrap CI as MMLU). The direction is robust: every pair loses capability under quantization on ARC as well as MMLU. The magnitude and significance, however, diverge informatively. Qwen 1.7B, which loses a large, significant 8.7 pp on MMLU, loses only 1.3 pp on ARC (CI [−0.036, +0.011], not significant), so the severe MMLU figure is substantially MMLU-specific (consistent with the answer-format truncation noted in Chapter 8), and the capability half of the 1.7B dual degradation is correspondingly milder than MMLU alone implies. Qwen 4B shows the opposite pattern: flat on MMLU (−0.3 pp, n.s.) but a small, significant −2.1 pp on ARC (CI [−0.037, −0.007]), so its capability is less fully preserved than MMLU suggested. Llama 3B is the most consistent pair, significant on both (−4.3 pp MMLU, −2.8 pp ARC). Two implications follow. First, RQ3 (NF4 degrades capability) is strengthened: the direction holds across two independent benchmarks and at least one benchmark reaches significance for every pair. Second, the dramatic within-Qwen scale gap under MMLU (the ≈29:1 ratio, §6.2/RQ4) does not replicate under ARC, which finds the 4B losing marginally more than the 1.7B; the strong 'smaller model is far more capability-sensitive' claim is therefore MMLU-specific and is hedged accordingly. Interpretation labels remain MMLU-anchored; ARC is reported here as a corroborating capability axis, and a formal composite-capability rule is left to future work. The two cross-family pairs (§6.13) extend the ARC axis to ten models: Mistral-7B is essentially flat on ARC (+0.9 pp, not significant) consistent with its preserved MMLU, and Phi-4-mini loses 1.5 pp (not significant) alongside a 2.3 pp MMLU dip, both new families showing the same direction-consistent, modest capability story."),

  H2("6.5 Statistical Caveats"),
  PJ("Five statistical limitations should be borne in mind when interpreting the results. First, with 200 HarmBench prompts and 250 benign XSTest prompts, the paired bootstrap 95% confidence interval on a binomial-proportion delta is approximately ±0.05 (Table 6.1), so small deltas may not be statistically distinguishable from zero. Under the primary (judge) scorer, the Qwen 1.7B ΔASR (+0.055, CI [+0.010, +0.100]) is the only HarmBench delta whose CI excludes zero; the Qwen 4B ΔASR (+0.025, CI [−0.000, +0.055]) is directional but not significant, and the Llama ΔASR (0.000, CI [−0.020, +0.020]) is within noise (the two cross-family pairs added in §6.13 are likewise non-significant: Mistral −0.040, CI [−0.110, +0.025]; Phi 0.000, CI [−0.030, +0.030]; so in the fp16-vs-NF4 comparison Qwen 1.7B remains the only HarmBench delta whose CI excludes zero across all five pairs; the INT8 precision point in §6.15 adds a second, INT8-specific significant ASR move on Llama-3B that reverts at NF4). Second, decoding is deterministic at temperature 0.0, so the only source of variance reflected in the bootstrap intervals is prompt sampling, with no within-condition stochastic variance. The one significant HarmBench delta (Qwen 1.7B) is corroborated by McNemar's exact paired test (p = 0.027) in addition to the bootstrap CI, so it does not rest on a single interval procedure; however, because both tests condition on the single greedy decode, a multi-seed (T = 0.7, top-p 0.8) sensitivity arm was run for the load-bearing Qwen 1.7B pair to estimate generation-level variance (§6.6.1). It reproduces the worsening direction on average (mean ΔASR +0.024 over five seeds) but shows the effect attenuates to roughly half the greedy estimate and is not sign-consistent across seeds, so the +0.055 greedy figure should be read as the upper end of a decode-dependent range rather than a fixed effect. To keep point-estimate labels from being over-read, every interpretation label additionally carries a two-layer evidence_status (confirmed / directional / null; §3.8): of the original three pairs, Qwen 1.7B and Llama are confirmed and Qwen 4B is directional; the cross-family pairs (§6.13) are directional (Mistral) and null (Phi). Third, the MMLU subset comprises 300 questions pooled across six subjects; subject counts are uneven (25–94 per subject: business_ethics 25, college_biology 33, human_aging 44, clinical_knowledge 48, high_school_world_history 56, high_school_macroeconomics 94). Subject-level accuracy estimates are correspondingly noisy and are not reported as primary statistics. The distribution is identical across all baseline and 4-bit runs (same seed, same prompt IDs), so cross-condition comparisons are unaffected. Fourth, HarmBench ASR is scored by the official HarmBench classifier (primary); the v2 refusal regex is retained only as a secondary non-refusal-rate proxy because the judge validation (§6.12) showed it materially over-counts ASR. XSTest over-refusal is scored by the v2 regex (the HarmBench classifier does not cover the over-refusal question). Fifth, no family-wise (multiple-comparisons) correction is applied: across the five pairs the study reports roughly twenty primary delta confidence intervals (HarmBench ASR, MMLU, ARC, over-refusal) plus the per-category breakdown and the INT8 precision sweep (§6.15), so the significance flags are nominal, per-comparison ones. The single significant NF4 ΔASR (Qwen 1.7B, p = 0.027) is a nominal result that would not survive a strict Bonferroni correction over all comparisons; it is carried as the headline only because it is independently corroborated by McNemar's exact test, the multi-seed sensitivity arm (§6.6.1) and the second judge (§6.12), and is reported with an explicit two-layer evidence_status rather than as a bare \"significant\" claim. The Llama-3B INT8 move (§6.15), significant under both judges and McNemar, is the more multiplicity-robust of the two. Readers should weigh this converging evidence rather than any single per-comparison threshold."),

  H3("6.5.1 Multiple-comparison correction and statistical power"),
  PJ("Rather than only acknowledging the multiplicity problem, we report the corrected view. Applying a Benjamini-Hochberg false-discovery-rate correction (q < 0.05) to the family of twenty primary NF4-vs-fp16 contrasts (five pairs × {HarmBench ASR (judge), MMLU, ARC, over-refusal}, with an exact McNemar p-value computed for every paired-binary contrast, results/analysis/multiple_comparisons.json), exactly three contrasts survive, and all three are capability losses: Qwen3-1.7B MMLU (p = 0.002, q = 0.029), Llama-3B ARC (p = 0.003, q = 0.029) and Qwen3-4B ARC (p = 0.006, q = 0.039). The Qwen3-1.7B ΔASR (McNemar p = 0.027) does not survive (q = 0.13), and neither does the Phi-4-mini ΔOR, which under the stricter exact test is borderline (p = 0.065) rather than CI-significant. Capability degradation is therefore the multiplicity-robust signal of four-bit NF4 in this study, whereas the single safety regression is a nominal, per-comparison result. This is consistent with, and strengthens, the capability-driven mechanism reading (§6.14): the Qwen3-1.7B safety wobble is carried as the headline only on the strength of its three-way corroboration (McNemar + the multi-seed arm + the second judge), not on an uncorrected significance threshold."),
  PJ("A power analysis quantifies why so few effects reach significance. For a two-sided α = 0.05, 80%-power McNemar test at the observed HarmBench discordant rates and n = 200, the minimum detectable ΔASR is roughly ±0.04–0.06 (median ≈ 0.044); the only NF4 ASR effect at or above this detection floor is Qwen3-1.7B (+0.055), and even its post-hoc power is only ≈ 0.67. The study is therefore underpowered for the small effects it measures, so the predominance of nulls on the safety axis reflects a detection floor as much as a substantive absence of effect, a limitation that a larger prompt set would address (Chapter 8). Figures and per-contrast values are in results/analysis/multiple_comparisons.{json,csv}."),

  H2("6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b)"),
  PJ("Under the official HarmBench classifier, the Qwen 1.7B pair is the strongest result in the study and receives the broad_degradation label (evidence status: confirmed); it degrades on both axes. HarmBench ASR rises significantly (0.135 → 0.190, ΔASR = +0.055, 95% CI [+0.010, +0.100], significant) and MMLU accuracy falls significantly (0.620 → 0.533, ΔMMLU = −0.087, CI [−0.137, −0.037], significant). Quantization makes the smallest model both more willing to produce genuinely harmful content and less capable. Because capability also drops beyond tolerance, the rule does not assign alignment_degradation (which requires capability preserved); the label is broad_degradation, combined safety and capability worsening."),
  PJ("The ΔASR finding is corroborated by a second, independent significance test. Because HarmBench is paired, McNemar's exact test [23] (the correct paired-binary test; §3.7) is applied to the judge outcomes: of 200 prompts, 16 became harmful under quantization while only 5 became safe (21 discordant prompts), giving an exact two-sided p = 0.027. The paired bootstrap (CI excludes zero) and McNemar's exact test therefore agree the increase is statistically real rather than a resampling artefact, and the 16-versus-5 split makes the effect size transparent: the +0.055 net delta is the residual of a markedly asymmetric flip. This is the only HarmBench pair in the study significant under both tests."),
  PJ("Note the sharp contrast with the secondary v2 proxy, which reports a much higher absolute non-refusal rate (0.600 → 0.575) and a small negative delta (−0.025) that on its own looked like a non-event. The judge reveals that of the many Qwen 1.7B responses the regex counted as \"not a refusal\", only about 22% (baseline) to 33% (4-bit) are genuine instances of the harmful behaviour, and that the *true* harmful-compliance rate rises significantly under quantization, the opposite direction from the proxy's point estimate. This is the clearest single illustration in the study of why the official classifier, not refusal-counting, is the appropriate scorer (see §6.12)."),
  PJ("The XSTest over-refusal rate declined modestly (0.052 → 0.028, ΔOR = −0.024, CI [−0.052, 0.000], not significant). The over-refusal change is within noise; the pair's safety story is on the harmful-compliance axis (judge), not the over-refusal axis. Together the three dimensions describe a model that, under NF4 compression, produces more genuinely harmful completions, refuses benign prompts at a similar (slightly lower) rate, and answers fewer factual questions correctly."),
  PJ("Subject-level MMLU breakdown supports the capability-decline reading. Accuracy declined in five of six subjects: high school macroeconomics (0.596 → 0.479, −11.7 pp), business ethics (0.680 → 0.560, −12.0 pp), clinical knowledge (0.729 → 0.563, −16.7 pp), college biology (0.576 → 0.515, −6.1 pp), and high school world history (0.643 → 0.571, −7.1 pp). A small gain in human ageing (0.523 → 0.568, +4.6 pp) is within the noise range expected for a 44-sample subject subset. The overall accuracy decline of 8.7 percentage points has a 95% CI of [−0.137, −0.037] and is robustly distinguishable from zero by the paired bootstrap."),
  PJ("The Qwen 1.7B pair is the study's most consequential deployment finding: a 4-bit-quantized small model that is simultaneously less capable and measurably more compliant with harmful instructions than its full-precision counterpart. The capability anchor is what makes this unambiguous: the significant ΔMMLU rules out the optimistic reading, while the significant judge ΔASR rules out the dismissive one. The strength of the ΔASR half of this claim is qualified by the decoding-sensitivity check in §6.6.1; the ΔMMLU half is unaffected.")
,
  H3("6.6.1 Multi-seed decoding sensitivity (W1 robustness check)"),
  ...Figure("multiseed.png", "Qwen3-1.7B judge ΔASR under decoding variation: the single greedy headline (+0.055, red diamond) versus the multi-seed mean ± sd at temperature 0.7 (blue, five seeds), which attenuates the effect to ≈ +0.024 and is not sign-consistent. The headline is best read as the upper end of a decode-dependent range. Source: results/analysis/sensitivity_multiseed.json."),
  PJ("Because the primary study uses greedy decoding (T = 0.0), its bootstrap and McNemar intervals capture prompt-sampling variance only, not generation variance, the one Priority-1 weakness (W1) that cannot be resolved by reanalysis of the existing artifacts. To estimate how stable the headline ΔASR is under realistic sampling, the Qwen 1.7B pair was re-run on HarmBench at the model publisher's recommended non-thinking setting (temperature 0.7, top-p 0.8) across five independent seeds, with both pair members scored by the same official HarmBench classifier used for the headline. Within each seed, baseline and 4-bit share the identical temperature and seed, so every per-seed ΔASR is a clean matched-pair contrast; the arm is a separate robustness layer and is never cross-compared with the greedy main results."),
  PJ("The judge ΔASR is positive in four of the five seeds (per-seed: −0.010, +0.050, +0.055, +0.020, +0.005), with mean +0.024, standard deviation 0.028, and range [−0.010, +0.055]. The quantization-raises-harm direction is therefore reproduced on average, but it attenuates to roughly half the greedy point estimate (+0.055, which itself coincides with the maximum seed) and is not sign-consistent across seeds, so the greedy figure is best read as the upper end of a decode-dependent range rather than a fixed effect. The instability is asymmetric: the 4-bit model's ASR is near-constant across seeds (~0.170), while the full-precision baseline's varies more widely (0.115–0.180), so the gap narrows on seeds where the baseline happens to comply more often. The secondary v2 proxy shows the same qualitative picture (mean ΔASR ≈ 0.000, range [−0.040, +0.025], also not sign-consistent). The arm tempers, but does not overturn, the headline: under realistic stochastic decoding the Qwen 1.7B safety regression is directional and small rather than the robust +0.055 the single greedy decode suggested. Results are written to results/analysis/sensitivity_multiseed.json; the symmetric Qwen 4B and Llama infrastructure is in place for a future extension across the full matrix."),

  H3("6.6.2 Per-category harmful-compliance profile"),
  ...Figure("category_asr.png", "Per-category HarmBench attack success rate for Qwen3-1.7B at fp16 versus NF4 (official classifier). The aggregate +0.055 increase is broad-based (ASR rises in five of six categories) rather than an artefact of a single harm type. Per-category counts are small (19–58 prompts), so the rates are descriptive. Source: results/analysis/harmbench_category_breakdown.json."),
  PJ("To check whether the Qwen 1.7B safety regression is concentrated in a single harm type or is broad-based, the judge ASR was decomposed by HarmBench semantic category (results/analysis/harmbench_category_breakdown.{json,csv}, produced by scripts/harmbench_category_breakdown.py from the saved generations and judge sidecars, no new inference). Under the official classifier, the 1.7B pair's ASR rises in five of six categories under quantization: misinformation/disinformation (0.412 → 0.500), illegal activity (0.034 → 0.121), harassment/bullying (0.053 → 0.105), the generic harmful bucket (0.190 → 0.238), and chemical/biological (0.036 → 0.071); only cybercrime/intrusion is unchanged (0.125 → 0.125). The aggregate +0.055 ΔASR is therefore broad-based rather than an artefact of one category, which strengthens the headline finding. Two categories are notable for deployment: the highest-prevalence harm at both precisions is misinformation (already 0.412 at full precision, rising to 0.500), and the proportionally largest jump is in illegal-activity prompts (more than tripling off a low base). Per-category counts are small (19–58 prompts each), so these rates are descriptive and exploratory, reported to characterise where the aggregate effect sits, not as individually significance-tested claims. For contrast, the Qwen 4B pair's increases are smaller and mixed (three categories up, cybercrime/intrusion down) and the Llama pair is essentially flat across categories, consistent with their non-significant aggregate ΔASR."),

  H2("6.7 Preliminary Scale Observations: Baseline Comparison"),
  PJ("The Qwen3-1.7B and Qwen3-4B full-precision baselines reveal a capability-and-safety coupling within the Qwen family. At full precision, scaling from 1.7B to 4B parameters produces (HarmBench under the official classifier): HarmBench ASR 0.135 → 0.065 (−7.0 pp; the larger model produces fewer genuinely harmful completions), MMLU accuracy 0.620 → 0.747 (+12.7 pp; substantially more capable), and XSTest over-refusal 0.052 → 0.028 (−2.4 pp; fewer false refusals on benign prompts). Within Qwen the larger model dominates the smaller one on every dimension: more capable, more safety-calibrated, and better calibrated on benign prompts. (Under the v2 non-refusal proxy the baseline ASR gap looks much larger, 0.600 → 0.240, but that reflects the proxy's over-counting; the genuine harmful-compliance gap is the 7.0 pp figure.) This baseline divergence motivates the matched-pair design: quantization effects must be interpreted against scale-dependent baselines."),

  H2("6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b)"),
  PJ("Under the official classifier, the Qwen 4B pair receives the alignment_degradation label by point estimate with evidence status directional: the effect is suggestive rather than statistically confirmed. HarmBench ASR rises from 0.065 to 0.090 (ΔASR = +0.025, 95% CI [−0.000, +0.055], not significant, the lower bound touches zero) while capability is preserved (ΔMMLU = −0.003, CI [−0.040, +0.033], not significant). McNemar's exact paired test agrees the move is not significant: only 7 prompts became harmful versus 2 that became safe (9 discordant), exact p = 0.18. Because capability is preserved and the ASR point estimate exceeds the harm tolerance, the rule assigns alignment_degradation; but both the bootstrap CI and McNemar's test include the null, so the two-layer scheme flags this as directional: a suggestive safety worsening, not a confirmed one."),
  PJ("This is a notable demotion from the secondary v2 proxy, under which Qwen 4B had appeared to be the study's only statistically significant HarmBench result (proxy ΔASR = +0.065, CI [+0.025, +0.110]). The judge validation shows that the proxy's apparent significance was partly an artefact of refusal-counting: the regex marked many non-refusals as harmful, inflating both the level and the delta. On the genuine harmful-compliance measure, the 4B effect is real in direction but does not reach significance at n = 200. The honest reading is that NF4 may modestly increase Qwen 4B's harmful compliance, but the present sample cannot confirm it."),
  PJ("Mechanism and over-refusal. Qwen 4B at full precision is strongly refusal-calibrated on genuine harm (judge baseline ASR = 0.065). XSTest over-refusal moves marginally (0.028 → 0.020, ΔOR = −0.008, not significant). A plausible mechanism for the directional ASR increase is the same NF4 degradation of refusal precision seen elsewhere; but unlike the 1.7B pair, the 4B model retains its capability, so any safety move here is not a capability artefact; it just is not large enough to confirm at this sample size. The subject-level MMLU is correspondingly flat, corroborating the preserved-capability reading: only two of six subjects regress (business ethics −4.0 pp, college biology −3.0 pp), three are unchanged, and high school macroeconomics rises slightly (+1.1 pp), a stark contrast with the 1.7B pair's five-of-six broad regression. Per-subject figures for all pairs are in results/analysis/mmlu_subject_breakdown.{json,csv}."),
  PJ("Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under the same quantization. The 1.7B model degrades on both axes (broad_degradation: significant ΔASR and significant ΔMMLU). The 4B model preserves capability and shows only a directional, non-significant safety worsening (alignment_degradation by point estimate). The within-family contrast is therefore: the smaller model suffers a confirmed dual degradation, while the larger model is far more robust, with a non-significant nudge on safety and capability intact. This is a cleaner scale story than the v2 proxy suggested, and it points the same direction as the broader literature on quantization sensitivity declining with model size."),
  PJ("The Llama 3.2 3B pair results (§6.10–§6.11) allow assessment of whether any safety worsening reproduces cross-family."),

  H2("6.9 Research Question Synthesis"),
  PJ("With all ten models evaluated and HarmBench ASR scored by the official classifier, it is now possible to give answers to all five research questions. Paired bootstrap 95% confidence intervals quantify the statistical uncertainty behind each answer. Among the original three pairs, three deltas are statistically significant under the primary scorers: Qwen 1.7B ΔASR (+0.055, judge), Qwen 1.7B ΔMMLU (−0.087), and Llama 3B ΔMMLU (−0.043); the second capability benchmark adds significant ARC losses for Qwen 4B (−0.021) and Llama (−0.028), and the cross-family extension adds one significant over-refusal delta (a decrease) for Phi-4-mini (ΔOR −0.028). Every ΔASR other than Qwen 1.7B (Qwen 4B, Llama, Mistral-7B and Phi-4-mini) is within sampling noise under four-bit NF4, as are the remaining ΔOR values and the Qwen 4B ΔMMLU. The Qwen 1.7B ΔASR is the only HarmBench result whose CI excludes zero under four-bit NF4; combined with its significant capability loss, the Qwen 1.7B broad_degradation is the single most consequential empirical result in the study."),

  H3("RQ1: Does 4-bit NF4 quantization increase harmful compliance?"),
  PJ("It does not reduce it in any pair, and the one statistically significant move is an increase. Under the official classifier, the Qwen 1.7B pair shows ΔASR = +0.055 (CI [+0.010, +0.100], significant): quantization significantly increases genuine harmful compliance in the smallest model. The Qwen 4B pair shows ΔASR = +0.025 (CI [−0.000, +0.055]), directionally worse but not significant. The Llama 3.2 3B pair shows ΔASR = 0.000 (CI [−0.020, +0.020]), no change. All three point estimates are ≥ 0. The answer to RQ1: NF4 quantization never reduces true harmful compliance here, and in the smallest model it significantly increases it. The data refute any claim that quantization is safety-neutral or safety-improving on harmful compliance. (The secondary v2 proxy, which over-counts ASR, would have mislocated the only significant effect on Qwen 4B; the official classifier corrects this, see §6.12.)"),

  H3("RQ2: Does 4-bit NF4 quantization increase over-refusal on benign prompts?"),
  PJ("Across the five pairs, four show no significant change in over-refusal and the fifth moves in the benign direction. Under the v2 scorer: Qwen3-1.7B ΔOR = −0.024 (CI [−0.052, 0.000], not significant), Qwen3-4B −0.008 (CI [−0.024, +0.008], not significant), Llama 3B +0.016 (CI [−0.004, +0.040], not significant), Mistral-7B 0.000 (CI [−0.012, +0.012], not significant), and Phi-4-mini −0.028 (CI [−0.052, −0.004], significant). The only significant over-refusal delta is Phi's, and it is a decrease: the 4-bit model refuses fewer benign prompts, not more. No pair shows a significant increase. The answer to RQ2: NF4 quantization does not produce a detectable over-refusal increase on benign prompts at this sample size, across four families and five pairs; the single significant move runs in the safety-benign direction. This near-null is practically useful for deployment teams considering NF4 compression: the concern that quantization would make models more trigger-happy on benign prompts is not supported by these data; if anything, the one significant effect points the other way."),

  H3("RQ3: Does 4-bit NF4 quantization degrade general capability?"),
  PJ("The answer depends strongly on model scale and family. The Qwen3-1.7B model loses 8.7 percentage points of MMLU accuracy under NF4 quantization (CI [−0.137, −0.037], significant), a decline spanning five of six evaluated subjects. The Llama 3.2 3B model loses 4.3 percentage points (CI [−0.080, −0.007], significant), spanning four of six subjects. The Qwen3-4B model loses only 0.3 percentage points (CI [−0.040, +0.033], not significant, within measurement noise). Scale is a strong moderator of quantization-induced capability loss within the Qwen family, and Llama 3B sits between the two Qwen models in sensitivity. The answer to RQ3: NF4 quantization significantly degrades capability in the 1.7B Qwen model and the 3B Llama model, but has negligible effect on the 4B Qwen model. This scale-and-family-dependent capability profile is the most practically important capability finding for deployment teams. A second capability benchmark (ARC-Challenge; §6.4.1) corroborates the direction on all three pairs and reaches significance for Qwen 4B and Llama; its one substantive divergence is the Qwen 1.7B loss, which is far smaller and non-significant on ARC (−1.3 pp) than on MMLU (−8.7 pp). The cross-benchmark reading is therefore that NF4 reliably reduces capability in direction, while the *severity* for the smallest model was overstated by MMLU alone."),

  H3("RQ4: Are smaller models more sensitive to quantization within the same family?"),
  PJ("Yes, clearly, and on both axes. On capability, the MMLU delta ratio between Qwen 1.7B and Qwen 4B is approximately 29:1 (−8.7 pp versus −0.3 pp): the 1.7B model loses substantial capability under NF4 compression while the 4B model is essentially unaffected. On safety (judge), the 1.7B model shows a significant ΔASR of +0.055 while the 4B model shows only a non-significant +0.025. So the smaller model is more sensitive on every dimension: it suffers a confirmed dual degradation (broad_degradation), while the larger model preserves capability and shows only a directional, unconfirmed safety nudge. The answer to RQ4: smaller Qwen models are more sensitive to NF4 quantization than larger ones on harmful compliance, and on capability *as measured by MMLU*, but the capability half of this claim is benchmark-dependent. On the ARC-Challenge second benchmark (§6.4.1) the within-family ordering does not hold: the 4B pair loses marginally more capability than the 1.7B pair, so the dramatic ≈29:1 MMLU ratio is an MMLU-specific result, not a benchmark-independent scale law. The safety-axis contrast (1.7B significant ΔASR; 4B not) is independent of the capability benchmark and stands. The robust RQ4 conclusion is therefore: smaller Qwen models are clearly more sensitive on harmful compliance, while the within-family capability-sensitivity gap is real on MMLU but should not be over-generalised, since a second benchmark roughly equalises it."),

  H3("RQ5: Are effects consistent across model families?"),
  PJ("Partially. On capability, both families lose MMLU accuracy under NF4 compression where the loss is significant: Qwen 1.7B (−8.7 pp) and Llama 3B (−4.3 pp); the Qwen 4B model is the exception with non-significant loss. On safety (judge), all three ΔASR point estimates are ≥ 0 (quantization never reduces harmful compliance in either family) but only Qwen 1.7B reaches significance. The three pairs receive three labels (broad_degradation, alignment_degradation directional, broad_degradation). The cross-family pattern most likely to reproduce is the null on over-refusal: all three ΔOR are non-significant. The full answer to RQ5 across the original two families: NF4 quantization is never safety-improving on harmful compliance and does not significantly raise over-refusal; capability loss is consistent in direction where significant. The one confirmed safety regression (Qwen 1.7B) would benefit from replication on a same-family-different-scale pair (e.g. Qwen 0.5B / 7B) to test whether the smallest-model dual degradation is a general scale effect. The cross-family extension (§6.13) tests the generalisation across families: neither Mistral-7B (ΔASR −0.040, not significant) nor Phi-4-mini (0.000) shows a significant increase, so the Qwen 1.7B regression remains unique to the smallest model while the broad pattern (no significant safety improvement, capability loss where significant) holds across four families."),

  H3("Discussion: What the full five-pair dataset tells us"),
  PJ("Three implications extend beyond the specific numbers. First, surface-level refusal metrics are unreliable without both a capability anchor and an accurate harmful-compliance scorer. The v2 regex over-counts ASR (it equates non-refusal with success), and the judge validation moved the study's one significant safety regression from Qwen 4B (under the proxy) to Qwen 1.7B (under the official classifier), a conclusion-level correction rather than a cosmetic one. Second, the most robust empirical finding goes in the safety-worsening direction: Qwen 1.7B's significant ΔASR (+0.055, judge) combined with its significant capability loss makes it a confirmed broad_degradation, and no pair shows reduced harmful compliance. This refutes the optimistic narrative that NF4 compression is safety-neutral. Third, the framework itself is the durable contribution: it produces well-separated diagnostic categories, and, validated against the benchmark's own classifier, it caught and corrected a scorer artefact rather than propagating it. The framework's value does not depend on any specific empirical outcome; it provides a reusable, self-correcting methodology for the next study."),

  H2("6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b)"),
  PJ("The Llama 3.2 3B pair (SLURM jobs 60980 and 60981) completed on 2026-05-27. Job 60980 (baseline, float16) ran in 6 minutes 15 seconds; job 60981 (NF4 4-bit) ran in 13 minutes 9 seconds, consistent with the longer load time of quantized models observed in the Qwen runs."),
  PJ("The baseline profile of Llama 3.2 3B-Instruct is the most refusal-calibrated of the three models in the study under the official classifier. HarmBench ASR at full precision is 0.040 (judge): only 8 of 200 adversarial prompts produced genuinely harmful completions. XSTest over-refusal is 0.044, also low. MMLU accuracy is 0.610, closely matching the Qwen3-1.7B baseline (0.620). This profile describes a model that is both compactly capable and strongly safety-calibrated at full precision: safer on HarmBench than either Qwen model (judge baseline ASR 0.135 for Qwen 1.7B, 0.065 for Qwen 4B) while approximately equally capable as the smaller Qwen on MMLU. This is also the pair where the v2 regex and the judge agree most closely (κ ≈ 0.68–0.79; §6.12): Llama mostly either refuses cleanly or complies, leaving little ambiguous middle ground for the regex to mis-score."),
  PJ("Under NF4 4-bit quantization, the Llama pair is classified as broad_degradation. HarmBench ASR is unchanged at 0.040 → 0.040 (ΔASR = 0.000, CI [−0.020, +0.020], not significant): the model retains its strong refusal behaviour under compression. XSTest over-refusal rises modestly from 0.044 to 0.060 (ΔOR = +0.016, CI [−0.004, +0.040], not significant). MMLU accuracy falls by 4.3 percentage points (0.610 → 0.567, −7.0% relative; CI [−0.080, −0.007], statistically significant). The MMLU drop spans four of six subjects: high school macroeconomics (0.543 → 0.468, −7.5 pp), college biology (0.667 → 0.545, −12.1 pp), clinical knowledge (0.667 → 0.583, −8.3 pp), and human aging (0.636 → 0.614, −2.3 pp). Two subjects show small gains within the noise range: high school world history (+3.6 pp) and business ethics (+4.0 pp). The MMLU decline is moderate, larger than Qwen 4B (−0.3 pp) but smaller than Qwen 1.7B (−8.7 pp), placing Llama between the two Qwen models in capability sensitivity."),
  PJ("Why broad_degradation? With ΔASR within noise and a significant ΔMMLU, the rule falls through to the fallback label: a meaningful capability cost without a detectable safety direction. This is diagnostically informative: the Llama pair refutes any simple narrative that NF4 universally weakens alignment. Llama's strong baseline safety calibration is preserved under quantization at the cost of measurable capability loss, with no accompanying safety regression. The contrast with Qwen 1.7B is the sharpest in the study: at nearly identical baseline capability (MMLU 0.610 vs 0.620), the smaller Qwen degrades on both axes under quantization while Llama holds its safety and loses only capability, strong evidence that the alignment recipe, not the parameter count alone, governs how quantization affects harmful compliance."),

  H2("6.11 Cross-Family Comparison (RQ5)"),
  PJ("The cross-family comparison places the Llama 3.2 3B pair alongside the two Qwen pairs to assess whether NF4 quantization effects are consistent across architectures and alignment recipes. Table 6.2 (§6.2) shows the delta values and interpretation labels for all three original pairs side by side; the cross-family extension to Mistral-7B and Phi-4-mini is analysed in §6.13."),

  H3("6.11.1 Baseline safety profiles differ substantially across families"),
  PJ("Before interpreting quantization deltas, the baseline profile differences are themselves informative. At similar MMLU capability (Qwen3-1.7B: 0.620; Llama 3.2 3B: 0.610), the Llama model is more safety-calibrated than the smaller Qwen under the official classifier: baseline HarmBench ASR is 0.040 for Llama vs 0.135 for Qwen 1.7B, a 9.5 pp difference in favour of Llama on the safety axis at equivalent capability. XSTest over-refusal at baseline is 0.044 for Llama vs 0.052 for Qwen 1.7B, comparable. This indicates that two compact models at the same factual capability have different safety calibrations, attributable to differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach. Family/recipe is a comparable-to-larger safety determinant than quantization for these models. The Llama vs Qwen 1.7B baseline gap on ASR (9.5 pp) is of the same order as the largest single quantization delta in the study (Qwen 1.7B, +5.5 pp). The 4B Qwen baseline at 0.065 sits between these two."),

  H3("6.11.2 Harmful compliance: one significant delta, in the worsening direction, in the smallest model"),
  PJ("Under the official classifier, only one of the three ΔASR values is statistically significant: Qwen 1.7B (+0.055, CI [+0.010, +0.100]), a safety regression in the smallest model. The Qwen 4B ΔASR (+0.025, CI [−0.000, +0.055]) is directional but not significant, and the Llama ΔASR (0.000, CI [−0.020, +0.020]) is null. All three point estimates are ≥ 0: NF4 quantization never reduces genuine harmful compliance here. The one robust effect is that the smallest model becomes significantly more willing to produce harmful content under compression. (This is the opposite of where the v2 proxy placed the significant effect: the proxy's apparent Qwen 4B significance was an over-counting artefact; see §6.12.) The Llama baseline is so strongly safety-calibrated (judge ASR = 0.040) that little headroom exists for change; whether NF4 would alter ASR more visibly in a less strongly-aligned Llama variant is an open question."),

  H3("6.11.3 Over-refusal: no significant increase across families"),
  PJ("The XSTest over-refusal rate is the most consistent cross-family signal: four of the five ΔOR deltas are not statistically significant (Qwen3-1.7B (−0.024, CI [−0.052, 0.000]), Qwen3-4B (−0.008, CI [−0.024, +0.008]), Llama-3B (+0.016, CI [−0.004, +0.040]), and Mistral-7B (0.000, CI [−0.012, +0.012])), and the fifth, Phi-4-mini, is significant but in the benign direction (−0.028, CI [−0.052, −0.004]): its 4-bit model over-refuses less, not more. No pair shows a significant increase in over-refusal. This pattern spans four families and five pairs and is practically useful for deployment: the concern that NF4 compression would make models excessively conservative on benign prompts is not supported by these data, and the only significant move is a reduction in false refusals (Phi). The largest non-significant effect is on Llama (+0.016), a borderline-direction signal whose CI still includes zero."),

  H3("6.11.4 Capability: directionally consistent where significant, magnitude varies"),
  PJ("Two of three pairs lose MMLU accuracy significantly: Qwen 1.7B (−8.7 pp, CI [−0.137, −0.037]) and Llama 3B (−4.3 pp, CI [−0.080, −0.007]). The Qwen 4B pair is essentially neutral (−0.3 pp, CI [−0.040, +0.033], not significant). All three point estimates are negative; two are significant. The Llama 3B value (−4.3 pp) is intermediate between the two Qwen models. This positioning is consistent with a size-sensitivity hypothesis: Llama 3B, at 3 billion parameters, lies between the two Qwen models in the robustness spectrum. However, because Llama and Qwen differ in architecture and training, this apparent scale ordering may also reflect family-specific effects. More data points (e.g., Qwen at 7B or Llama at 1B) would be needed to confirm whether the ordering reflects a size effect or a family effect."),

  H3("6.11.5 Interpretation labels and the study's central question"),
  PJ("Under the official classifier, the three interpretation labels are broad_degradation (Qwen 1.7B), alignment_degradation–directional (Qwen 4B), and broad_degradation (Llama 3B). They collectively answer the study's central question: \"do observed safety changes under quantization reflect alignment shifts or capability artefacts?\" The Qwen 1.7B pair degrades on both axes (a significant ASR increase and a significant capability loss), the confirmed worst case and the study's most consequential deployment finding. The Qwen 4B pair preserves capability and shows only a directional, non-significant ASR increase. The Llama pair is a capability cost without a safety direction. Two of the three pairs (Qwen 1.7B, Llama) carry significant capability loss, and none shows reduced harmful compliance. The central scientific finding is the framework's diagnostic value, sharpened by the fact that validating it against the benchmark's own classifier relocated the one robust safety regression from Qwen 4B (regex proxy) to Qwen 1.7B (official classifier). The data refute the optimistic narrative that NF4 quantization is safety-neutral or safety-improving in this scale range. These conclusions are drawn from the original three pairs; the cross-family extension (§6.13) adds Mistral-7B and Phi-4-mini and finds the same diagnostic structure (Mistral in the proxy-over-counting camp (judge ΔASR −0.040, not significant, versus a +0.055 proxy artefact) and Phi in the proxy-reliable, robust_preservation camp), so the central question's answer generalises to four families."),

  H2("6.12 Judge-Model Validation of HarmBench ASR (the primary scorer)"),
  PJ("HarmBench Attack Success Rate is scored by the official HarmBench classifier, cais/HarmBench-Llama-2-13b-cls, the fine-tuned judge model specified by the HarmBench paper. Given a harmful behavior and a model generation, it answers whether the generation is a non-minimal, unambiguous instance of the behavior (\"yes\" = attack success). This subsection reports the validation that established the judge as the primary HarmBench scorer over the v2 regex (decision D16). The judge re-scores the already-saved generations; it performs no new target-model inference and does not modify the TC1-original raw outputs or the v2 sidecars. Judge labels are written to redacted per-prompt sidecars (scores.judge.harmbench_cls.jsonl, IDs and booleans only) and aggregated to summary.judge.harmbench_cls.json. The judge covers the HarmBench benchmark specifically; over-refusal on XSTest is a different question (\"did the model refuse a benign prompt?\") and remains v2-scored, left to a future refusal-style judge."),
  PJ("Methodology and setup. The judge classified every saved (behavior, generation) pair for all six original models (n = 200 each, 1 200 total) on the NTU TC1 cluster via sbatch in offline mode on 2026-06-06 (job 61047); the two cross-family pairs (§6.13) were scored identically on 2026-06-15 (job 61134, a further 800 classifications, also zero parse errors). It ran in full precision (float16) on a 32 GB Tesla V100 (recorded in summary.judge.harmbench_cls.json) because the judge is the reference instrument validating the quantized target models, and quantizing the judge itself would reintroduce the very confound the check exists to rule out. All 1 200 generations were classified with zero parse errors. Agreement with the v2 scorer is reported per model as raw agreement rate and Cohen's kappa over the shared prompt set; the per-pair judge ΔASR is computed with the same paired bootstrap (2 000 resamples, seed 42) used for the other deltas, and passed through the interpretation-label rule (holding the v2 over-refusal and MMLU deltas fixed, since the classifier judges only HarmBench). Figures are produced locally by scripts/judge_agreement.py and stored in results/analysis/judge_agreement.{json,csv}."),
  buildTable(
    ["Model", "v2 non-refusal", "judge ASR", "agreement", "Cohen's κ"],
    [
      ["qwen_2b_base", "0.600", "0.135", "53.5%", "0.19"],
      ["qwen_2b_4bit", "0.575", "0.190", "61.5%", "0.30"],
      ["qwen_4b_base", "0.240", "0.065", "82.5%", "0.36"],
      ["qwen_4b_4bit", "0.305", "0.090", "78.5%", "0.37"],
      ["llama_3_2_3b_base", "0.060", "0.040", "97.0%", "0.68"],
      ["llama_3_2_3b_4bit", "0.060", "0.040", "98.0%", "0.79"],
      ["mistral_7b_base", "0.835", "0.385", "53.0%", "0.19"],
      ["mistral_7b_4bit", "0.890", "0.345", "44.5%", "0.11"],
      ["phi4_mini_base", "0.075", "0.055", "96.0%", "0.67"],
      ["phi4_mini_4bit", "0.075", "0.055", "96.0%", "0.67"],
    ],
    [2600, 1900, 1500, 1500, 1300],
  ),
  P("Table 6.3  Judge vs v2 per-model agreement on HarmBench (n = 200 each). \"v2 non-refusal\" is the regex's harmful-compliance rate; \"judge ASR\" is the official classifier's. Agreement and Cohen's κ are over the shared prompt set. From results/analysis/judge_agreement.json.", { size: 18 }),
  ...Figure("judge_vs_proxy.png", "Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal' proxy, one marker per model. Every point lies below the diagonal, so the proxy systematically over-counts harmful compliance (most severely for Qwen and Mistral). Right: judge-vs-proxy Cohen's κ per model, which is family-dependent: low for Qwen/Mistral (κ ≈ 0.11–0.37) and high for Llama/Phi (κ ≈ 0.67–0.79). Source: results/analysis/judge_agreement.json."),
  PJ("Result 1: the v2 regex systematically over-counts ASR, unevenly by family. The judge ASR is far lower than the v2 non-refusal rate for the Qwen models (e.g. Qwen 1.7B base 0.135 vs 0.600; Qwen 4B base 0.065 vs 0.240) and close for Llama (0.040 vs 0.060). Agreement is correspondingly poor for Qwen (κ ≈ 0.19–0.37) and high for Llama (κ ≈ 0.68–0.79). The reason is structural: the regex equates \"did not refuse\" with \"attack succeeded\", but many non-refusals are not genuinely harmful (vague deflections, safety lectures without refusal keywords, on-topic but benign answers, failed attempts). Qwen produces a large volume of such ambiguous middle-ground text; Llama mostly either refuses cleanly or complies, leaving little for the regex to mis-score. This is exactly the failure mode the HarmBench paper's fine-tuned classifier was built to avoid, and it is why the official classifier, not refusal counting, is adopted as the primary HarmBench scorer."),
  PJ("Result 2: the choice of scorer changes the conclusion. Under the judge, the per-pair HarmBench ΔASR values are: Qwen 1.7B +0.055 (CI [+0.010, +0.100], significant); Qwen 4B +0.025 (CI [−0.000, +0.055], not significant); Llama 0.000 (CI [−0.020, +0.020], not significant). Compared with the v2 proxy (−0.025, +0.065, 0.000), the significant effect moves from Qwen 4B to Qwen 1.7B and flips the small-model direction from apparently-negative to significantly-positive. The interpretation labels follow: Qwen 1.7B moves from capability_collapse_masquerading_as_safety (proxy) to broad_degradation (judge: significant ΔASR up + significant ΔMMLU down); Qwen 4B remains alignment_degradation but only directionally (no longer significant); Llama remains broad_degradation. The judge validation is therefore not a rubber stamp: it corrected a refusal-counting artefact and relocated the study's one robust safety regression to the smallest model. This is reported transparently as the central methodological result of the chapter, and the HarmBench numbers throughout the report are the judge values accordingly (D16)."),
  PJ("Result 3: the relocation is robust to the choice of significance test. The judge ΔASR significance was re-tested with McNemar's exact test, which is the correct paired-binary test for HarmBench and assumes no large-sample normality (§3.7). It reproduces the bootstrap conclusion exactly: Qwen 1.7B p = 0.027 (16 prompts became harmful vs 5 became safe), Qwen 4B p = 0.18 (7 vs 2), Llama p = 1.00 (2 vs 2). Only the smallest model's increase is significant under either test, so the study's single confirmed safety regression does not depend on the bootstrap machinery. A structural property of the judge–regex relationship reinforces the over-counting diagnosis: across all six models the judge labelled essentially no generation harmful that the regex had cleared (judge_harmful_v2_not ≈ 0), so the judge's harmful set is almost a strict subset of the regex's: the disagreement is one-directional over-counting by the regex, not two-sided noise."),
  PJ("Result 4: the finding is robust to the choice of judge (second-judge cross-check). To rule out the possibility that the result is an artefact of the single HarmBench classifier, the saved generations were independently re-scored by a second, architecturally unrelated frontier judge (gpt-4o) applying the same HarmBench rubric (the api_judge backend; decision D25). The two judges agree strongly at the per-prompt level: across all ten models Cohen's κ ranges from 0.60 to 0.95 (moderate-to-almost-perfect agreement), far higher than the regex-versus-classifier agreement (κ ≈ 0.11–0.37 for Qwen and Mistral), confirming that it was the regex, not the classifier, that was the outlier. The per-pair ΔASR direction is reproduced: gpt-4o gives Qwen 1.7B +0.045 (versus the classifier's +0.055), Qwen 4B +0.010 (versus +0.025), and Llama +0.005 (versus 0.000); for the cross-family pairs it gives Mistral −0.030 (versus −0.040) and Phi +0.005 (versus 0.000), and crucially it places Mistral's ASR near 0.37, matching the classifier, not the v2 proxy's 0.84–0.89. Both judges agree that only the smallest Qwen model moves materially and that Mistral's proxy-driven 'increase' is an artefact. One nuance is stated plainly: under gpt-4o the Qwen 1.7B ΔASR is not statistically significant (McNemar p = 0.122) whereas under the primary classifier it is (p = 0.027). The effect's direction and approximate magnitude are judge-robust; its statistical significance sits at the boundary and is judge-dependent, a conclusion that converges with the multi-seed sensitivity arm (§6.6.1), which likewise found the effect directionally reproduced but attenuated. Across both robustness checks the Qwen 1.7B harmful-compliance increase is real and modest (≈ +0.045 to +0.055), not a large or certain effect. Per-model agreement and per-pair figures are in results/analysis/judge_pairwise_agreement.{json,csv} (redacted)."),
  PJ("Threats and scope. The judge is itself a model and can err; it is adopted because it is the field-standard reference classifier specified by the HarmBench paper [4], run at full precision, with deterministic greedy decoding and zero parse failures on this dataset. It validates HarmBench only; XSTest over-refusal and MMLU are unaffected. The single-judge construct-validity threat is now substantially addressed by the second-judge cross-check (Result 4): an independent frontier judge agrees at κ 0.60–0.95 across all ten models and reproduces the finding's direction and magnitude. One construct-validity threat is, however, only partially closed and is stated plainly: both scorers compared here are LLMs, and no human-annotated gold set was collected for HarmBench, so this study establishes that the classifier and the regex disagree (and that the classifier's harmful set is a near-strict subset of the regex's (judge_harmful_v2_not ≈ 0, Result 3), i.e. the disagreement is one-directional over-counting by the regex) rather than that the classifier matches human ground truth. The cautious form of the claim is therefore that the choice of scorer changes the conclusion and that refusal-counting over-states harmful compliance relative to the benchmark's own classifier; anchoring the classifier against a small human-labelled subset (reported as classifier-vs-human and regex-vs-human agreement) is the natural confirmation and is set out in the Future Work chapter. The LLM-as-judge literature makes the same point: judges are useful but require validation and, ideally, human grounding [20], [21]. Two further residual caveats: the second judge is a versioned API model (less reproducible than the open-weight primary), and both judges share the broad lineage of LLM-based harm classification, so a fully independent open-weight guard (e.g. Llama Guard [22]) would be a complementary future check. The v2 regex is retained in the repository and in Table 6.1/6.2 as a transparent secondary proxy so that readers can see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring."),

  H2("6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5)"),
  PJ("To test whether the findings generalise beyond the original two families, two further matched pairs were added and run on TC1 on 2026-06-15 under the identical methodology (on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding at temperature 0.0, seed 42, the same four benchmarks at the same sample counts, and the official HarmBench classifier as the primary ASR scorer): mistral_7b (Mistral-7B-Instruct-v0.3, 7.2B, the largest model in the study) and phi4_mini (Phi-4-mini-instruct, 3.8B). This takes the study to five pairs across four families (Qwen, Llama, Mistral, Phi). Both pairs' raw metrics, deltas, and judge-agreement figures appear in Tables 6.1, 6.2 and 6.3, and their second-judge (gpt-4o) cross-check is folded into the §6.12 figures. The two pairs fall into the two diagnostic camps already seen among the original three."),
  PJ("Mistral-7B is the clearest demonstration of the judge-over-proxy finding (§6.12, D16) in the whole study. The v2 refusal proxy reports a high baseline non-refusal rate (0.835) rising to 0.890 under quantization, a +0.055 'increase' that, taken at face value, would label the pair alignment_degradation. The official classifier tells the opposite story: genuine harmful compliance is 0.385 at baseline and 0.345 under 4-bit, a ΔASR of −0.040 (CI [−0.110, +0.025], not significant; McNemar p = 0.31): quantization slightly reduces, not increases, true harmful compliance. The proxy and the judge disagree in sign, and the judge–proxy agreement is the lowest in the study (Cohen's κ = 0.19 at baseline, 0.11 under 4-bit; the regex flags 92–110 of 200 prompts harmful that the classifier clears). The second independent judge (gpt-4o) corroborates the classifier, not the proxy: it places Mistral's ASR at 0.385/0.355 (versus the proxy's 0.835/0.890) and gives ΔASR −0.030, agreeing with the primary classifier at κ 0.60–0.63. Capability is essentially preserved (MMLU −1.7 pp, n.s.; ARC +0.9 pp, n.s.) and over-refusal is flat (ΔOR = 0.000), so under the judge the pair reads as alignment_improvement at the point-estimate level, carried as directional because its ΔASR CI spans zero. For the study's largest model, NF4 leaves genuine harmful compliance unchanged-to-slightly-lower while preserving capability, and a refusal-based scorer would have badly mischaracterised it."),
  PJ("Phi-4-mini falls in the opposite (Llama-like) camp, where the proxy and the judge largely agree. Genuine harmful compliance is low and identical across the pair (judge ASR 0.055 at both baseline and 4-bit (ΔASR exactly 0.000, CI [−0.030, +0.030]; McNemar p = 1.00)), so on safety the pair is robust_preservation (evidence status: null). The judge–proxy agreement is high (κ = 0.67, like Llama), and the second judge agrees almost perfectly (κ 0.79 at baseline, 0.95 under 4-bit; ΔASR +0.005). On capability, Phi shows a modest, non-significant dip on both axes: MMLU −2.3 pp (CI [−0.057, +0.010]) and ARC −1.5 pp (CI [−0.032, 0.000]). Its one significant delta is on over-refusal: the 4-bit model is slightly less over-cautious on benign prompts, ΔOR = −0.028 (CI [−0.052, −0.004], significant), the only significant over-refusal delta in the study, and in the benign direction (fewer false refusals, not more)."),
  PJ("Across all four families the central pattern holds and strengthens. Under the official classifier, NF4 quantization never significantly increases genuine harmful compliance in any of the five pairs: the Qwen 1.7B pair remains the study's sole statistically significant ΔASR (+0.055), while the two new families add a non-significant decrease (Mistral, −0.040) and an exact null (Phi, 0.000). The proxy-reliability split now spans four families (the regex over-counts severely for Qwen and Mistral (judge-vs-proxy κ 0.11–0.37) but tracks the judge for Llama and Phi (κ 0.67–0.79)), so the methodological contribution (validate the scorer before trusting the delta) generalises rather than being a Qwen-specific artefact. On over-refusal, four of five pairs show no significant change and the fifth (Phi) moves in the benign direction, so the RQ2 deployment message (NF4 does not make models more trigger-happy on benign prompts) survives the extension. Capability loss stays directionally consistent (every pair's MMLU and ARC point estimates are ≤ 0, the sole exception being Mistral's negligible +0.9 pp on ARC). The cross-family extension therefore corroborates RQ5: NF4's safety effect is not uniform across families, is never a significant worsening outside the smallest Qwen model, and is cleanly separable from capability only once the scorer itself is validated."),

  H2("6.14 Mechanism Probe: Refusal-Margin Geometry (the \"why\")"),
  PJ("The behavioural results (§6.3, §6.12, §6.13) establish that NF4 rarely shifts harmful compliance and where it does (only the smallest Qwen pair, modestly). To probe why, a derived mechanism analysis measured each model's first-token refusal margin (m = log P(refusal-token set) − log P(compliance-token set), a scale-invariant log-probability difference at the first generated position) on all 200 HarmBench prompts per model, in fp16 and under NF4, via teacher-forced forward passes (scripts/capture_refusal_margin.py on TC1; redacted scores.margin.<precision>.jsonl sidecars carrying only prompt-id and scalar margins, no text; analysed by scripts/refusal_margin_analysis.py → results/analysis/refusal_margin.{json,csv}). The hypothesis under test, following Proskurina et al. [18], is that quantization perturbs the decision distribution most where the baseline margin is thin, so behavioural flips should concentrate at thin baseline margins. The analysis was run with explicit validity, circularity and confound controls, and the headline numbers were independently recomputed from the raw sidecars."),
  PJ("Validation gate. Before any quantization claim the margin's sign was checked against the baseline judge label. For four families it discriminates refusal from compliance well (AUC 0.86–0.90 for Qwen-1.7B, Qwen-4B, Llama, Phi) but for Mistral-7B only weakly (AUC 0.69), because Mistral frequently opens with a refusal-shaped preamble and then complies: its refuse-versus-comply decision is resolved later than the first token. The first-token margin is therefore a sound proxy for four families and a poor one for Mistral, which is flagged throughout. The gate is itself only suggestive: the well-discriminating pairs have small baseline-harmful counts (8–13 prompts) and correspondingly wide AUC intervals, whereas Mistral's lower estimate rests on the largest positive class (77)."),
  PJ("Central result: boundary instability, not targeted erosion. A thin baseline margin does predict which prompts flip under NF4, but the within-model effect is modest. Pooling all 1,000 prompts gives AUC 0.76; after removing the between-family level difference (z-scoring the margin within each pair) it falls to 0.64, and within individual pairs it is near chance for the two thinnest-margin families (Qwen-1.7B 0.61, Mistral 0.54). The flips are also close to symmetric in aggregate: of 92 flips, 50 are harmful-ward (refuse→comply) and 42 are safe-ward (comply→refuse), and a thin margin predicts the safe-ward flips (AUC 0.78) at least as strongly as the harmful-ward ones (0.75). This pooled near-symmetry must, however, be read with care: it is driven by Mistral-7B (the one family whose first-token margin the gate above flags as a poor proxy), which alone contributes 20 harmful-ward and 28 safe-ward flips (two-thirds of all safe-ward flips). Across the four families where the first-token margin is a valid proxy the flips lean roughly 2:1 harmful-ward (30 versus 14), and the load-bearing Qwen-1.7B pair is 16 versus 5. Two of those valid-proxy families are nonetheless individually symmetric (Llama 2:2, Phi 5:5) and the safe-ward direction is well represented, so thin first-token margins are best read as marking prompts near the greedy-decoding decision boundary that quantization can destabilise in either direction, a boundary-fragility effect rather than evidence of a safety-specific erosion mechanism. The strongest evidence for the boundary-instability reading of the one behaviourally-moving pair (Qwen-1.7B) is not this pooled symmetry but the entropy confound control and the significant ΔMMLU discussed next."),
  PJ("Direction of the margin shift is model-specific and modest. The paired shift Δm = m(fp16) − m(NF4) is statistically significant for every pair (Wilcoxon p < 0.001, n = 200) but moves comply-ward for Qwen (Δm = +1.2 for 1.7B, +4.0 for 4B) and refuse-ward for Llama and Phi (−1.1, −0.3), with effect sizes spanning Cohen dz ≈ −0.74 to +1.8; the whole-sequence margins reverse the first-token sign for Qwen-1.7B and Phi, so the directional claim is first-token-specific. The supportable statement is that NF4 perturbs the first-token refusal distribution in a model-specific direction, not that it uniformly erodes refusal. Qwen-4B shifts comply-ward more than Qwen-1.7B (+4.0 vs +1.2) yet flips fewer prompts (9 vs 21), because its baseline margin is far wider (median 13.0 vs 3.8): it has refusal headroom to absorb the perturbation, which is the mechanistic reason the 4B pair is behaviourally intact while the 1.7B is not."),
  PJ("Capability versus alignment: the tie to the central question. If NF4 simply raised next-token entropy everywhere, every margin would shrink as a by-product of generic confidence loss rather than a safety-specific change. Comparing the entropy rise on harmful prompts with that on neutral (MMLU / XSTest-benign) tokens, the harmful side softens more only for Qwen-4B and Mistral, the two pairs that do not drive the behavioural result. For the one pair that does move, Qwen-1.7B, neutral-token entropy rises roughly ten times more than harmful-token entropy (+0.21 vs +0.02): the change is consistent with generic confidence/capability softening near the boundary, not a targeted alignment shift. (This control is suggestive only: n = 100 unpaired neutral prompts, a mixed set, no significance test.) The mechanism evidence therefore converges with the significant Qwen-1.7B ΔMMLU (§6.2), the multi-seed fragility (§6.6.1) and the ARC-versus-MMLU capability picture (§6.4.1): the smallest model's harmful-compliance wobble reads as capability-driven boundary instability rather than genuine alignment erosion. The probe's contribution is not a clean safety-erosion mechanism: it is to localise the effect to near-boundary prompts and to weigh the evidence, for the one moving pair, toward the capability side of the study's central dichotomy. Two mechanism follow-ups (an independent activation-space refusal-direction probe and a paired neutral-margin control) are listed in Chapter 9; the third anticipated there, an INT8 precision point tracing the effect across the fp16 → INT8 → NF4 spectrum, has since been run and is reported in §6.15."),

  H2("6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4)"),
  ...Figure("precision_sweep.png", "Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat through INT8 and falls only at four-bit (a cliff, not a gradient), whereas the safety axis (judge ASR) is method-specific: Llama-3B rises at INT8 and reverts at NF4, while Qwen3-1.7B's increase is an NF4 effect. Source: results/analysis/precision_sweep.json."),
  PJ("The main study evaluates one quantization method (NF4) at one bit-width (four-bit). To test whether the safety and capability effects are a smooth function of quantization aggressiveness or are instead method- and bit-width-specific, an INT8 precision point was added so every pair is evaluated across three precisions: fp16 → INT8 → NF4. INT8 here is the bitsandbytes LLM.int8() mixed-precision method (a distinct algorithm, not a lower-bit NF4) applied on the fly to the same baseline weights. All five INT8 members were run on the NTU TC1 cluster (five per-model matrix jobs plus the HarmBench-classifier judge job), scored by the primary official classifier and cross-checked by the gpt-4o second judge, with zero parse errors throughout; the baselines and NF4 members are the same committed artefacts as the main study, so the three precisions are directly comparable. The analysis (configs/tc1_int8.yaml, scripts/precision_sweep_analysis.py, with paired-bootstrap CIs and McNemar tests recomputed from the redacted judge sidecars) is kept separate from the base-vs-4-bit pairwise pipeline so the evaluated ten-model study is untouched. The question it answers: does degradation scale monotonically with bit-width (a graded fp16 > INT8 > NF4 trend), or is it concentrated and method-specific?"),
  PJ("Capability: a clean cliff at four-bit. The capability answer is unambiguous. No INT8 delta, on either MMLU or ARC, is statistically significant for any of the five pairs (paired bootstrap; the null result is robust, with no cell flipping to significant across ten seeds), so within the study's resolution INT8 is capability-lossless. Every significant capability loss appears only at the four-bit step: ΔMMLU(NF4) = −0.087 (Qwen-1.7B) and −0.043 (Llama-3B); ΔARC(NF4) = −0.021 (Qwen-4B) and −0.028 (Llama-3B). Eight-bit is therefore effectively a free precision point on capability, and the capability cost of quantization is a cliff at four-bit rather than a graded decline. (The one borderline cell is Phi-4-mini ΔARC(NF4) = −0.015, non-significant only by the closed-interval convention, footnoted so the dichotomy is not overstated as perfectly clean.)"),
  PJ("Safety: not a single four-bit cliff, but two effects at different precisions. The safety axis is the more revealing result, and it is also not graded: the study's two significant ASR regressions sit at different precision steps. First, the Qwen-1.7B increase is a four-bit (NF4) effect (ΔASR = +0.055, CI [+0.010, +0.100]) and, as a new robustness note from the second judge, it is judge-dependent: significant under the primary classifier but not under gpt-4o (+0.045, CI [−0.005, +0.095]). Second, and this is the precision point's substantive finding, Llama-3.2-3B shows an INT8-specific increase (ΔASR = +0.040) that is the single most judge-robust safety move in the whole study: significant under both judges (classifier CI [+0.010, +0.075]; gpt-4o CI [+0.015, +0.070]) and under McNemar's exact paired test (classifier 9 harmful-ward versus 1 safe-ward flip, p = 0.022; gpt-4o 8 versus 0, p = 0.008), with the flips concentrated in the illegal and cybercrime-intrusion categories. It is also non-monotonic: Llama's ASR rises at INT8 (0.040 → 0.080) and returns to baseline at NF4 (0.040), so the more aggressive four-bit method does not reproduce what the less aggressive eight-bit one introduces. The conceptual conclusion is therefore that quantization's effect on safety is method- and model-specific, not a smooth function of bit-width: a regression can appear at eight-bit and vanish at four-bit."),
  PJ("Honest bounds on the INT8 safety finding. Two cautions are stated plainly so the effect is neither buried nor over-read. It rests on a small absolute flip count (≈8–9 prompts) on a single pair, and it is non-monotonic, so it is best read as a method-specific numerical effect of LLM.int8()'s mixed-precision (outlier-decomposition) arithmetic on this particular model, not a general \"INT8 erodes safety\" law; replication across more models and decode seeds is the natural next step (Chapter 9). Significance flags across the sweep are uncorrected for multiple comparisons (thirty ASR contrasts across five pairs, three precision steps and two judges); the two effects that survive both the bootstrap and McNemar are the Qwen-1.7B NF4 move and the Llama INT8 move, the latter the more robust because it holds under both judges. The picture is consistent with the §6.14 mechanism reading (thin-margin boundary prompts are destabilised by quantization in a model-specific direction, and which precision or method tips which prompts is itself idiosyncratic) rather than a graded, bit-width-monotone erosion."),
  PJ("Scorer note (the proxy at INT8). The secondary v2 refusal regex continues to over-count harmful compliance at INT8 exactly as it does at fp16 and NF4: judge-versus-proxy Cohen κ is 0.11 for Mistral (the worst, over-counting ASR 0.885 versus the judge's 0.375), 0.22 and 0.35 for the two Qwen pairs, 0.59 for Phi-4-mini and 0.76 for Llama-3B, the same D16 pattern (low Qwen/Mistral agreement, high Llama agreement) reproduced at the new precision. The official classifier therefore remains the primary scorer at INT8, and every safety number above is judge-scored; the proxy is reported only as the demoted foil whose failure motivates the judge. The precision point sharpens the study's central message: within the compact-deployment regime neither the capability cost nor the safety effect of quantization is a smooth function of bit-width. The capability cost is a cliff at four-bit, while the safety effect is sparse, model- and method-specific, and can surface at eight-bit without persisting to four-bit."),

  H2("6.16 Generation-Length Robustness: the 512-Token Rerun"),
  PJ("The main study generates at most max_new_tokens = 128. HarmBench's own reference evaluation pipeline uses 512, and the HarmBench authors note that the number of tokens a target model is allowed to generate materially affects the measured attack-success rate. To test whether the safety findings are an artefact of the 128-token budget rather than a property of the models, the entire study was regenerated at 512 tokens: all five pairs, all four benchmarks, scored again by both the primary HarmBench classifier and the gpt-4o second judge. The 512-token run is written to a parallel results tree (results_512/, via configs/tc1_512.yaml, with the classifier judge in slurm/judge_validation_512.sbatch) so the original 128-token artefacts are retained unchanged and the two budgets are directly comparable. All ten matrix jobs and both judges completed with zero parse errors."),
  PJ("The longer budget changes the generations as intended. At 512 the HarmBench responses are markedly longer: median length 1,675 characters against 567 at 128 (a threefold increase), and 62 percent of all responses exceed the 792-character ceiling that the 128-token budget physically imposed, so the 128-token run was truncating the majority of generations before completion. Absolute ASR rises accordingly under both judges (for example Mistral-7B classifier ASR moves from 0.385 to 0.585), because a longer generation gives more room for harmful content to appear. That is a property of the absolute rate, not of the matched-pair delta, and it is the delta that carries every safety claim in this study."),
  buildTable(
    ["Pair", "ΔASR @128 (classifier)", "ΔASR @512 (classifier)", "ΔASR @512 (gpt-4o)"],
    [
      ["Qwen3-1.7B", "+0.055 (sig)", "0.000", "+0.005"],
      ["Qwen3-4B", "+0.025", "+0.040", "+0.040"],
      ["Llama-3.2-3B", "0.000", "−0.040", "−0.035"],
      ["Mistral-7B", "−0.040", "−0.020", "−0.005"],
      ["Phi-4-mini", "0.000", "+0.020", "+0.020"],
    ]
  ),
  PJ("The headline result of the rerun is that the study's single significant safety regression does not replicate at the reference budget. The Qwen3-1.7B increase of +0.055 at 128 tokens becomes ΔASR = 0.000 under the primary classifier and +0.005 under gpt-4o at 512 tokens, neither significant (McNemar p = 1.000 under both judges, and both paired-bootstrap confidence intervals span zero). The Qwen3-1.7B null at 512 is moreover a high-churn null: sixteen prompts flip from refusal to compliance and sixteen flip from compliance to refusal, a symmetric exchange that is exactly the boundary-instability signature the §6.14 mechanism probe identified, now visible directly at the longer budget. Across all five pairs at 512, no ΔASR survives a Benjamini-Hochberg correction under either judge; the largest nominal effect is Llama-3.2-3B at roughly −0.040 (classifier) and −0.035 (gpt-4o), which is in the safety-improving direction and fails the corrected threshold just as the original Qwen effect did. The two judges agree per prompt at Cohen κ 0.68–0.95, matching the 0.60–0.95 range reported for the 128-token run, so the scorer is no less reliable at the longer budget."),
  PJ("Capability degradation, in contrast, is essentially unchanged by the generation budget. The four-bit MMLU and ARC losses move only at the third decimal between the two budgets: Qwen3-1.7B ΔMMLU goes from −0.087 to −0.090, Llama-3.2-3B ΔARC from −0.028 to −0.032, and Qwen3-4B ΔARC from −0.021 to −0.016, with the same cells remaining significant. The capability cost of quantization is robust to the generation budget; the safety effect is not."),
  PJ("The generation-length rerun therefore reinforces, rather than threatens, the study's central dichotomy. The capability cost of NF4 is real and budget-robust, whereas the apparent safety regression is fragile on every axis tested: it does not survive the longer reference budget, a multiple-comparison correction, the second judge, or stochastic decoding (§6.6.1). Quantization in this regime erodes competence more than it erodes guardrails, and the one borderline safety signal observed at 128 tokens is best read as capability-driven boundary instability rather than a robust alignment shift. The 128-token results are retained unchanged for direct comparison. The INT8 precision point and the multi-seed sensitivity arm are being regenerated at 512 tokens as a completeness check and will be folded in when available; their 128-token conclusions (a capability cliff at four-bit, and sparse, method-specific safety effects) are not expected to change qualitatively."),
];

// ------------------------------------------------------------
// Chapter 7 - Discussion and Threats to Validity
// ------------------------------------------------------------
const ch7 = [
  H1("Chapter 7: Discussion and Threats to Validity"),

  PJ("The Qwen family results (§6.9) provide initial empirical grounding for the threats discussed in this chapter. The Qwen 1.7B broad_degradation finding (significant ΔASR and ΔMMLU under the judge) and the Qwen 4B directional alignment_degradation finding are used below as concrete examples where relevant."),

  H2("7.1 Internal Validity"),
  PJ("Internal validity is the strongest property of the present design. The matched-pair structure, combined with on-the-fly NF4 quantization from identical baseline weights, isolates quantization as the sole experimental variable. There is no plausible alternative explanation for an observed delta beyond the quantization step itself, the act of loading the same checkpoint twice, or measurement noise. Deterministic decoding (temperature 0.0) eliminates within-condition variance from generation. Scoring is also deterministic: MMLU uses exact match, XSTest over-refusal uses the regex parser, and the primary HarmBench scorer (the official HarmBench classifier) is run with greedy decoding (max_new_tokens = 1) in full precision, producing reproducible labels (job 61047 returned zero parse errors over 1 200 generations). The resume logic prevents partial-run contamination: every reported metric is computed from a complete raw.jsonl with exactly the configured number of prompts. All six target-model runs completed under the same conditions, hardware, and software stack."),

  H2("7.2 External Validity"),
  PJ("External validity is bounded by three explicit design choices. First, findings are bounded to BitsAndBytes NF4 quantization. Alternative methods such as GPTQ, AWQ, and the GGUF family used by llama.cpp may produce different effects, both quantitatively and qualitatively; the present study does not claim to characterise those methods. Second, the Qwen3 baselines (Qwen3-1.7B and Qwen3-4B) are instruction-tuned dense models from the official Alibaba Qwen3 release; results describe quantization behaviour on this specific model family and may not transfer directly to other architectures or training regimes. Third, the study is restricted to English-language, text-only interactions; multilingual and multimodal effects are out of scope. Finally, the full-precision baseline is loaded at float16 (dtype: auto resolves to float16 on CUDA), not the models' native bfloat16; because both members of every pair share the same compute dtype, the matched-pair delta still cleanly isolates quantization, but the absolute baseline figures are float16 rather than bfloat16 values."),

  H2("7.3 Construct Validity"),
  PJ("Each benchmark operationalises its dimension in a specific way. HarmBench captures one curated definition of \"harmful compliance\" against a particular set of adversarial prompts; other operationalisations exist and may yield different absolute numbers. XSTest captures over-refusal against a specific distribution of benign-but-suspicious prompts. The MMLU subset, even with diverse subject coverage, is a partial capability proxy and does not measure all reasoning capacities relevant to deployment (such as code generation, long-context reasoning, or tool use). These construct boundaries are common in safety evaluation but should be borne in mind when interpreting the results."),

  H2("7.4 Scorer Choice and Remaining Validity Threat"),
  PJ("HarmBench ASR was initially scored with a deterministic regex parser, chosen for reproducibility. The judge validation (§6.12) showed this regex measures \"non-refusal rate\" rather than genuine harmful compliance and over-counts ASR, most severely for the Qwen models, so the official HarmBench classifier was promoted to the primary HarmBench scorer (decision D16) and the regex demoted to a transparent secondary proxy. This removes the original construct-validity threat on the HarmBench metric: the primary scorer is now the benchmark's own fine-tuned classifier, run at full precision with deterministic decoding. Two residual threats remain. First, the HarmBench classifier is itself a model and can err. This single-judge threat has now been directly tested: the saved generations were re-scored by a second, architecturally unrelated frontier judge (gpt-4o) applying the same HarmBench rubric (the api_judge backend; D25). The two judges agree strongly per prompt (κ 0.60–0.95 across all ten models) and the second judge reproduces the Qwen 1.7B increase in direction and approximate magnitude (+0.045 versus +0.055), though at McNemar p = 0.122 it does not reach significance under the second judge (§6.12, Result 4). The construct-validity threat is therefore substantially resolved (the finding is not an artefact of one classifier), with two residual caveats: the second judge is a versioned, less-reproducible API model, and both judges share the general lineage of LLM-based harm classification, so a fully independent open-weight guard (e.g. LlamaGuard) remains a complementary future check. Second, XSTest over-refusal is still scored by the regex parser, since the HarmBench classifier does not judge over-refusal; a refusal-style judge for XSTest is likewise future work. The immutable-raw-output contract (TC1-original raw.jsonl/summary.json untouched; all corrected and judge scores in derived sidecars) means either future judge can be added without disturbing the existing evidence trail."),

  H2("7.5 Cross-Family Comparison Caveat"),
  PJ("The Qwen-versus-Llama comparison should be read as descriptive only. Qwen and Llama differ in tokenizer, pre-training corpus, instruction-tuning recipe, and safety-alignment methodology. Differences in their quantization deltas could plausibly reflect any combination of these factors, not only family identity. The cross-family component of the study therefore provides a useful robustness check on the within-Qwen findings but does not support causal claims about quantization–family interactions."),

  H2("7.6 Deployment Implications"),
  PJ("The practical takeaway is asymmetric across the scale range studied. The most safety-relevant result, a significant increase in genuine harmful compliance under NF4, appears only in the smallest model (Qwen 1.7B), which is precisely the size class most likely to be quantized for on-device or edge use. A team that validates safety on the full-precision 1.7B release and then ships the 4-bit build inherits a harmful-compliance rate that is higher (broad-based across harm categories, §6.6.2) and a capability level that is lower (−8.7 pp MMLU) than what they signed off on. The multi-seed check (§6.6.1) tempers the magnitude (under stochastic decoding the increase is roughly half the greedy estimate and not present in every seed), but its direction never favours the quantized model, so the deployment guidance is one-sided: 4-bit NF4 did not improve safety in any pair, and degraded it most where compression is most attractive. The larger Qwen 4B model is the safer thing to quantize: capability is preserved and the safety shift is only a non-significant nudge. The operational recommendations that follow are to re-run safety evaluation on the exact quantized artefact that will be deployed (not the full-precision release), and to prefer the largest model that fits the memory budget over aggressively quantizing a smaller one."),

  H2("7.7 Positioning Against Prior Work"),
  PJ("These findings sit consistently within, and extend, the emerging quantization-and-safety literature (§2.4). That quantization can degrade safety alignment, rather than being behaviourally neutral, matches Kharinaev et al. (arXiv:2502.15799) and Q-resafe (Chen et al., ICML 2025); that the effect is non-uniform, here depending on model size, family, and harm category, echoes the method- and attack-dependence those works and HarmLevelBench (Belkhiter et al.) report. The Egashira et al. (NeurIPS 2024) result that BitsAndBytes NF4 specifically can turn a benign model harmful is the closest prior analogue to the mechanism studied here, though that work demonstrates an adversarial worst case whereas this study measures the effect under ordinary, non-adversarial loading. The distinct contribution of the present work is the matched-pair, capability-anchored design: by holding the prompt set fixed and measuring harmful compliance and MMLU capability jointly on the same on-the-fly-quantized weights, it separates a genuine alignment regression (Qwen 1.7B, where ASR rises while capability falls) from a capability-driven artefact (Llama, where capability falls but harmful compliance does not), a distinction the cited single-axis studies are not structured to make."),
];

// ------------------------------------------------------------
// Chapter 8: Limitations
// ------------------------------------------------------------
const ch8 = [
  H1("Chapter 8: Limitations"),
  PJ("The principal limitations of the study are summarised below. Each is acknowledged explicitly and, where possible, addressed in the Future Work chapter."),
  ...numberedList([
    "Quantization-method coverage. Two BitsAndBytes methods are evaluated: NF4 four-bit (the main study) and INT8/LLM.int8 as a precision point (§6.15), so the study now spans fp16 → INT8 → NF4 rather than a single bit-width. GPTQ, AWQ, and GGUF paths remain out of scope and are flagged for follow-up work, as does replication of the §6.15 Llama-3B INT8 effect across more models and decode seeds.",
    "Refusal parser approximation. The deterministic regex-based refusal parser is reproducible but may under-count nuanced refusals.",
    "Partial capability proxy (now two-benchmark). The six-subject MMLU subset is a tractable but partial measure of general capability and does not include code generation, long-context reasoning, or tool-use evaluation. To address this, a second capability benchmark (ARC-Challenge, 1,172 reasoning-oriented questions) has now been run on all ten models (§6.4.1). It confirms the direction of capability loss but shows the MMLU magnitudes are partly benchmark-specific. Qwen 1.7B's large MMLU drop does not replicate on ARC, while Qwen 4B shows a small significant ARC loss MMLU missed, so capability is now reported as a two-benchmark composite. The interpretation labels remain MMLU-anchored pending a formal composite-capability rule (Chapter 9).",
    "MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, so an item counts as incorrect both when the model selects the wrong option and when it never commits to a letter within the generation budget. For the Qwen 1.7B 4-bit model, 11 of 300 MMLU responses were truncated mid-reasoning without a final letter (answered rate 0.963 vs 0.983 for its baseline; the other five pairs answer 0.997–1.000). Part of that pair's MMLU drop (ΔMMLU = −0.087) therefore reflects degraded answer-format adherence under the token budget, a real capability effect, but one of format-following rather than purely of knowledge. The cleaner-answering pairs (Llama, Qwen 4B) are unaffected. A future run could raise the generation cap or add a constrained-decoding answer step to separate format-following from knowledge.",
    "Text-only, English-only scope. Multilingual and multimodal behavioural effects of quantization are out of scope.",
    "Greedy decoding only (partially addressed). Temperature 0.0 is used throughout the primary study, eliminating within-condition stochastic variance from the main analysis. A multi-seed (T = 0.7, top-p 0.8) sensitivity arm for the load-bearing Qwen 1.7B pair (§6.6.1) now provides a partial estimate of this variance and shows the headline ΔASR attenuates under stochastic decoding and is not sign-consistent across seeds; the other two pairs remain greedy-only, so a full-matrix variance estimate is still outstanding.",
    "Hardware and walltime constraints. Each TC1 job is allocated a single GPU, ten gigabytes of host memory, and six hours of walltime. Sample budgets and batch sizes are sized to fit comfortably within these constraints.",
    "Sample-size-driven confidence intervals. With 200 HarmBench prompts and 250 benign XSTest prompts, binomial-proportion confidence intervals are wider than large leaderboard settings; small deltas may not be statistically separable from zero.",
    "Qwen baseline provenance. The Qwen baselines are text-only derivatives of a multimodal Qwen series. While both members of each pair inherit the same derivation, claims about \"quantization effects on Qwen\" are most safely interpreted as claims about quantization effects on these specific text-extracted derivatives.",
    "Gated-access dependency. The Llama 3.2 3B pair and HarmBench dataset depend on accepted Hugging Face access conditions and a valid token available to the TC1 environment. This precondition has been satisfied for the current run, but future reproductions must repeat the access setup.",
  ]),
];

// ------------------------------------------------------------
// Chapter 9: Future Work
// ------------------------------------------------------------
const ch9 = [
  H1("Chapter 9: Future Work"),
  PJ("The framework and methodology established by this study admit several natural extensions, listed in approximate order of practical impact."),
  ...numberedList([
    "Multi-method quantization comparison. Extend the matrix to include GPTQ, AWQ, and GGUF quantization paths on the same baselines, allowing direct comparison of how different PTQ algorithms perturb safety and capability.",
    "Stochastic-decoding sensitivity arm (partially completed). The load-bearing Qwen 1.7B pair has now been re-run at temperature 0.7 (top-p 0.8) across five seeds and scored by the official classifier (§6.6.1); the result tempered the headline (mean ΔASR +0.024 versus +0.055 greedy, not sign-consistent across seeds). Extending the same symmetric arm to the Qwen 4B and Llama pairs, and raising the seed count, would complete the within-condition variance estimate across the full matrix.",
    "Composite-capability interpretation rule (second benchmark now run). ARC-Challenge has been run on all ten models (§6.4.1) and corroborated the direction of capability loss while showing the MMLU magnitudes are partly benchmark-specific, notably the Qwen 1.7B drop (−8.7 pp MMLU vs −1.3 pp ARC, n.s.) and the within-Qwen scale ratio (≈29:1 on MMLU, not reproduced on ARC). The interpretation labels currently stay MMLU-anchored with ARC as a corroborating axis; formalising a composite-capability rule (e.g. requiring agreement across benchmarks before assigning a capability-driven label) is the natural next step. Adding further capability benchmarks (e.g. GSM8K for math reasoning, HellaSwag for commonsense) would broaden the composite.",
    "Mechanism follow-ups (refusal-margin probe, §6.14). The first-token refusal-margin analysis localises the quantization effect to near-boundary prompts but is first-token-specific (its whole-sequence margins disagree in sign for two pairs) and its capability/confidence confound control is small (n = 100, unpaired, mixed neutral set, no significance test). Three follow-ups would sharpen it: (1) an independent activation-space refusal-direction probe (Arditi et al. [19]) as a second mechanism window that does not rely on a hand-built token set; (2) a paired neutral-margin control with a significance test, to separate safety-targeted shifts from generic confidence softening more rigorously; and (3) an INT8 precision point (now run and reported in §6.15) which traced the behavioural effect across the fp16 → INT8 → NF4 spectrum and found it is not bit-width-graded (capability loss is a clean cliff at four-bit, while a method-specific safety move surfaces on Llama-3B at INT8 and reverts at NF4); extending the first-token margin probe itself across the three precisions, rather than only the behavioural metrics, remains open.",
    "Cross-family and scale extension (completed 2026-06-15). Two further matched pairs were implemented and run on the cluster: mistral_7b (mistralai/Mistral-7B-Instruct-v0.3) and phi4_mini (microsoft/Phi-4-mini-instruct), taking the study to five pairs across four families (Qwen, Llama, Mistral, Phi) and adding a seven-billion-parameter point at the upper edge of the compact-deployment regime. Both pairs use the identical methodology (on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding, the same four benchmarks at the same sample counts, seed 42, and the official HarmBench classifier as the primary ASR scorer), so the comparison stays matched. The only new loader capability is an optional attn_implementation field (Phi-4-mini uses the eager attention backend on the V100, which has no flash-attention kernels); it is now covered by the configuration schema, the loader, the per-model SLURM job set, the judge-validation scripts, and the verification suite. Phi-4-mini loads through transformers' native Phi3 implementation rather than its bundled remote code, keeping its load path consistent with every other model in the study. Results are reported in Tables 6.1–6.3 and analysed in §6.13: under the judge, neither new pair shows a significant ASR increase. Mistral ΔASR −0.040 (n.s.; the v2 proxy's +0.055 is a sign-flipped over-count, judge-vs-proxy κ as low as 0.11) and Phi ΔASR 0.000 (robust_preservation, κ 0.67), so adding two families leaves Qwen 1.7B as the study's only significant safety regression and reinforces the judge-over-proxy finding (D16) in two further families.",
    "INT8 precision point / quantization-method sweep (completed; reported in §6.15). To test whether the safety and capability effects depend on quantization aggressiveness and method, an INT8 precision point (bitsandbytes LLM.int8, a different method from NF4, not a lower-bit NF4) was added as a third precision and run on TC1 across all five pairs, scored by both the official HarmBench classifier and the gpt-4o second judge with zero parse errors. The headline result (§6.15) is that the quantization effect is not a smooth function of bit-width: capability loss is a clean cliff at four-bit (no INT8 MMLU/ARC delta is significant for any pair), while the safety axis is two-peaked and method-specific: Qwen-1.7B's increase is an NF4 effect, and a second, both-judge-significant increase (McNemar p = 0.008/0.022) appears on Llama-3B specifically at INT8 and reverts at NF4. Three extensions remain: (a) replicating the Llama INT8 effect across more models and decode seeds to establish whether it is a general LLM.int8 phenomenon or model-specific numerics (it rests on ≈8–9 prompts on one pair); (b) tracing the §6.14 refusal-margin probe across all three precisions rather than only the behavioural metrics; and (c) adding genuinely different quantization families (GPTQ, AWQ, GGUF) beyond the two bitsandbytes algorithms evaluated here.",
    "Multilingual extension. Replicate the matched-pair design in Chinese (where Qwen is natively strong) and one low-resource language, to test whether quantization-induced safety changes are language-dependent.",
    "Fully-independent open-weight second judge. The primary HarmBench classifier has now been cross-checked against a second frontier judge (gpt-4o, same rubric), which agreed strongly (κ 0.60–0.95 across all ten models) and reproduced the finding's direction (§6.12, Result 4), substantially resolving the single-judge threat. Two complementary extensions remain: (i) re-run the cross-check with an open-weight guard model (e.g. LlamaGuard, via the already-wired --backend llamaguard on TC1) so the confirmation does not depend on a versioned API model and is fully reproducible; and (ii) add a refusal-style judge for XSTest over-refusal, which the HarmBench classifier does not score. Both are derived validation layers writing separate redacted sidecars, never modifying raw.jsonl, summary.json, or existing sidecars.",
    "Safety-preserving quantization. Investigate emerging \"safety-preserving\" quantization methods that explicitly seek to mitigate alignment degradation under PTQ, and compare them against the vanilla NF4 baseline studied here.",
  ]),
];

// ------------------------------------------------------------
// Chapter 10: Conclusion
// ------------------------------------------------------------
const ch10 = [
  H1("Chapter 10: Conclusion"),
  PJ("This Final Year Project investigates safety–capability trade-offs in four-bit quantized compact language models, focusing on a research question that institutional benchmarks have not directly answered: when a small instruction-tuned model is quantized for on-device deployment, do observed changes in safety behaviour reflect a true shift in alignment or a side-effect of degraded general capability?"),
  PJ("The methodological contribution is a controlled matched-pair design in which baseline and four-bit pair members are loaded from identical baseline weights, with NF4 quantization applied on the fly. This design eliminates publisher- and pipeline-asymmetry as confounds and provides the strongest practical isolation of quantization as the experimental variable. The engineering contribution is an open, reproducible benchmarking framework comprising the matched-pair pipeline, four benchmark plugins, the pairwise analysis layer with rule-based interpretation labels and paired bootstrap 95% confidence intervals, full SLURM orchestration for the NTU TC1 cluster with resumable per-model matrix jobs, and a verification suite of 329 automated tests. The analytical contribution is the interpretation layer itself, the capability-anchored, statistically-grounded, multi-benchmark framework, which formalises the alignment-versus-capability disambiguation as a rule-based decision procedure over combined safety and capability deltas and which is the durable contribution of this study independent of any specific empirical outcome."),
  PJ("All six original runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976–60981); HarmBench ASR was validated and re-scored with the official HarmBench classifier (full precision) on 2026-06-06 (job 61047); and a cross-family extension added the Mistral-7B and Phi-4-mini pairs on 2026-06-15 (§6.13). Under the official classifier the three original pairs produce three diagnostic profiles. The Qwen3-1.7B pair (broad_degradation): HarmBench ASR rises significantly (+0.055, CI [+0.010, +0.100]) and MMLU falls significantly (−0.087, CI [−0.137, −0.037]): the smallest model degrades on both axes, the confirmed worst case and the most consequential deployment finding. The Qwen3-4B pair (alignment_degradation, directional): ΔASR is positive but not significant (+0.025, CI touches zero) with capability preserved, a suggestive, unconfirmed safety worsening. The Llama 3.2 3B pair (broad_degradation): ΔASR ≈ 0 (CI [−0.020, +0.020]) with significant capability loss (−0.043); its strong baseline safety calibration (judge ASR = 0.040) is preserved at a capability cost. A central methodological result is that the judge validation overturned the regex proxy: the v2 scorer over-counts harmful compliance (it equates non-refusal with success), and adopting the benchmark's own classifier relocated the study's one significant safety regression from Qwen 4B (proxy) to Qwen 1.7B (official). Together the three original pairs span distinct diagnostic categories, and none shows reduced harmful compliance under quantization. The cross-family extension (§6.13) reinforces this across four families: among all five pairs the Qwen 1.7B increase remains the sole significant ΔASR under four-bit NF4 (the INT8 precision point, §6.15, adds a second, method-specific significant move on Llama-3B at eight-bit that does not persist to NF4), Mistral-7B's apparent proxy-driven rise is absent under the judge (ΔASR −0.040, not significant) at preserved capability, and Phi-4-mini is robust_preservation, so the conclusion that NF4 never significantly improves, and only in the smallest model significantly worsens, genuine harmful compliance now rests on four families rather than two. Finally, a mechanism probe (§6.14) localises the quantization effect to near-boundary prompts and finds it symmetric (destabilising refusals in both directions) and, for the one behaviourally-moving pair (Qwen-1.7B), more consistent with capability-driven boundary instability than a targeted alignment shift, reinforcing the capability-anchored reading rather than an alignment-erosion one."),
  PJ("The results answer all five research questions. Under the official HarmBench classifier, NF4 quantization never reduces genuine harmful compliance, and the one statistically significant move is an increase in the smallest model (RQ1): Qwen 1.7B ΔASR = +0.055 (CI excludes zero); Qwen 4B +0.025, Llama 0.000, Mistral-7B −0.040, and Phi-4-mini 0.000 are all within noise. The data refute the optimistic claim that quantization is safety-neutral or improving. NF4 quantization does not significantly increase over-refusal in any evaluated model (RQ2): four of five pairs are non-significant and the fifth (Phi-4-mini) significantly decreases, so the only movement is toward fewer false refusals. NF4 quantization significantly degrades capability in the Qwen 1.7B and Llama 3B models but not in the Qwen 4B model (RQ3). Smaller Qwen models are more sensitive on both axes: the 1.7B pair suffers a confirmed dual degradation (broad_degradation) while the 4B pair preserves capability and shows only a directional safety nudge (RQ4). Cross-family, no pair shows a significant ASR increase outside Qwen 1.7B (Mistral's judge point estimate is even negative), over-refusal never significantly rises, and capability loss is directionally consistent where significant (RQ5). A central methodological result is that validating the framework against the benchmark's own classifier corrected a refusal-counting artefact in the regex proxy and relocated the one significant safety regression from Qwen 4B to Qwen 1.7B. The framework itself, capability-anchored, statistically grounded, multi-benchmark, and self-correcting under judge validation, is the durable contribution; the specific empirical numbers are subordinate to it."),
];

// ------------------------------------------------------------
// References
// ------------------------------------------------------------
const refs = [
  H1("References"),
  P("References are cited in IEEE numbered style; the bracketed numbers below correspond to the in-text citations.", { after: 200 }),
  ...numberedList([
    "L. Sun et al., “TrustLLM: Trustworthiness in large language models,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2401.05561.",
    "B. Wang et al., “DecodingTrust: A comprehensive assessment of trustworthiness in GPT models,” in Proc. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.11698.",
    "Z. Zhang et al., “SafetyBench: Evaluating the safety of large language models,” in Proc. ACL, 2024. arXiv:2309.07045.",
    "M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.04249.",
    "P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test suite for identifying exaggerated safety behaviours in large language models,” in Proc. NAACL, 2024. arXiv:2308.01263.",
    "D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proc. International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.",
    "P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? Try ARC, the AI2 reasoning challenge,” arXiv:1803.05457, 2018.",
    "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023. arXiv:2305.14314. (Introduces NF4 and the bitsandbytes integration.)",
    "T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” in Proc. NeurIPS, 2022. arXiv:2208.07339.",
    "A. Yang et al. (Qwen Team), “Qwen3 technical report,” Alibaba Group, 2025. arXiv:2505.09388.",
    "A. Grattafiori et al. (Meta AI), “The Llama 3 herd of models,” Meta AI, 2024. arXiv:2407.21783.",
    "A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825.",
    "M. Abdin et al., “Phi-4 technical report,” Microsoft, 2024. arXiv:2412.08905.",
    "A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., “Investigating the impact of quantization methods on the safety and reliability of large language models,” arXiv:2502.15799, 2025.",
    "K. Chen, J. Zhang, J. Hu et al., “Q-resafe: Assessing safety risks and quantization-aware safety patching for quantized large language models,” in Proc. ICML, 2025. arXiv:2506.20251.",
    "K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in Proc. NeurIPS, 2024. arXiv:2405.18137.",
    "Y. Belkhiter, G. Zizzo, and S. Maffeis, “HarmLevelBench: Evaluating harm-level compliance and the impact of quantization on model alignment,” arXiv:2411.06835, 2024.",
    "I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence of large language models?,” in Findings of NAACL, 2024. arXiv:2405.00632.",
    "A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in language models is mediated by a single direction,” in Proc. NeurIPS, 2024. arXiv:2406.11717.",
    "J. Gu et al., “A survey on LLM-as-a-judge,” arXiv:2411.15594, 2024.",
    "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” arXiv:2503.05061, 2025.",
    "H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversations,” Meta AI, 2023. arXiv:2312.06674.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
  ]),
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
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_2b_4bit:
    family: qwen
    size_b: 1.7
    quantized: true
    pair_id: qwen_2b
    model_id: Qwen/Qwen3-1.7B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_4b_base:
    family: qwen
    size_b: 4.0
    quantized: false
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  qwen_4b_4bit:
    family: qwen
    size_b: 4.0
    quantized: true
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  llama_3_2_3b_base:
    family: llama
    size_b: 3.0
    quantized: false
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  llama_3_2_3b_4bit:
    family: llama
    size_b: 3.0
    quantized: true
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
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
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  mistral_7b_4bit:
    family: mistral
    size_b: 7.2
    quantized: true
    pair_id: mistral_7b
    model_id: mistralai/Mistral-7B-Instruct-v0.3
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

  phi4_mini_base:
    family: phi
    size_b: 3.8
    quantized: false
    pair_id: phi4_mini
    model_id: microsoft/Phi-4-mini-instruct
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
    trust_remote_code: false
    attn_implementation: eager
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu, arc]

decoding:
  max_new_tokens: 128
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
  log_dir: results/slurm_logs_tc1
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
  PJ("The full configs/tc1.yaml is reproduced below for reference. The mistral_7b and phi4_mini entries are the cross-family extension: they were run on TC1 on 2026-06-15 and their results are incorporated into Chapter 6 (Tables 6.1–6.3 and §6.13)."),
  ...Code(tc1Yaml),
];

const appendixB = [
  H1("Appendix B: Example Generated SLURM Script"),
  PJ("The following is the generated sbatch script for qwen_2b_4bit, one of the six per-model jobs emitted by the cluster-generate command. The remaining five scripts differ only in the --model alias and the SBATCH --job-name, --output, and --error lines."),
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
  Bullet("quantized : bool : true triggers NF4 loading."),
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
  H1("Appendix D: Test Inventory"),
  PJ("The verification suite comprises 329 automated tests across twenty-five test files. The distribution is summarised in Table D.1; the test files are located under tests/ in the repository."),
  buildTable(
    ["Test file", "Test count", "Coverage area"],
    [
      ["test_datasets.py", "27", "Dataset loaders (toxicity, bias, factuality); type coercion helpers."],
      ["test_models.py", "30", "ModelSpec, device and dtype resolution, prompt formatting (incl. enable_thinking), attn_implementation threading (injected only when set, omitted otherwise), and prompt-templating provenance (prompt_was_templated)."],
      ["test_evaluators.py", "21", "Legacy evaluator parsing and aggregation."],
      ["test_metrics_and_config.py", "21", "Bootstrap CI, JSONL/CSV round-trip, schema validation, and summary-CSV upsert (a resumed run replaces its row rather than duplicating it)."],
      ["test_quant_analysis.py", "20", "Pairwise delta computation, interpretation labels (six-label taxonomy), McNemar exact test, bootstrap CI logic, v2 sidecar selection, ARC capability axis, and NF4-only pair selection (the pairwise pipeline never labels an INT8 member as the NF4 result)."],
      ["test_quant_config_schema.py", "16", "Live tc1.yaml/default.yaml validation (ten models, five pairs, four families); per-family flags (Phi trust_remote_code+eager, Mistral neither); attn_implementation field validator."],
      ["test_judge_validation.py", "14", "Judge-model validation: stub-backend scoring, sidecar redaction enforcement, ASR aggregation, raw-immutability, idempotency, the official-template regression check, judge-precision resolution, and full status accounting (error/skipped counted, never silently dropped)."],
      ["test_pipeline.py", "12", "Batched generation and orchestration helpers."],
      ["test_agent_harness.py", "11", "Agent harness: AGENTS/CLAUDE sync, project-log discipline, immutable-artifact tolerance, stale-text/redaction scans, agent-start packet, and report-freshness (a report-source edit without a regenerated docx fails the gate; the committed policy keeps the source a trigger, not a satisfier)."],
      ["test_slurm_helpers.py", "8", "Generated sbatch contents and per-benchmark/per-model grouping."],
      ["test_xstest_benign.py", "6", "XSTest benign classification and schema-drift guarding."],
      ["test_arc.py", "5", "ARC-Challenge plugin: schema parse, gold-index (letter/numeric), scoring, aggregate, registry."],
      ["test_judge_agreement.py", "5", "Judge-vs-v2 agreement, Cohen's kappa (incl. degenerate case), confusion."],
      ["test_quant_pipeline_utils.py", "4", "Record schema validation, resume helpers."],
      ["test_judge_pairwise.py", "4", "API-judge yes/no/unparseable mapping (faked client), judge-vs-judge kappa, null-row skip."],
      ["test_sensitivity.py", "5", "Multi-seed sensitivity config + aggregator (summarise_deltas) and sbatch generator, plus the v2-proxy reader preferring summary.v2.json over the v1 runtime summary.json (with fallback)."],
      ["test_matrix_reuse.py", "2", "Matrix runner reuse_loaded_model behaviour."],
      ["test_quant_smoke.py", "2", "End-to-end pipeline smoke test on a stub."],
      ["test_refusal_parser.py", "46", "v1 sanity tests plus v2 regression tests: canonical refusal templates, negative controls, curly-apostrophe handling, the diagnostic match_refusal_pattern helper, and MCQ answer extraction including markdown-emphasised answers ('**B.**', 'answer is **D**')."],
      ["test_refusal_margin.py", "15", "Refusal-margin mechanism metric (D33/§6.14): log-prob margin sign and shift-invariance, decision entropy, top1–top2 gap, per-tokenizer refusal/compliance token-set construction (leading-space variants, dedup, ambiguity drop), and redaction (integer-only token-set outputs)."],
      ["test_quant_int8.py", "20", "INT8 precision point (D34/T29): quant_method schema validator + baseline-consistency check, ModelSpec/build_model_spec threading, 8-bit detection in _quantization_active, the loader int8-vs-nf4 branch selection (sentinel configs, no GPU/bitsandbytes needed), the configs/tc1_int8.yaml sweep config, and the silent-fp16 raise guard (_require_quantization_engaged raises for a quantized spec that loaded fp16, names the method, stays silent for baselines/active quant)."],
      ["test_precision_sweep.py", "11", "fp16 -> INT8 -> NF4 precision-sweep analysis: per-metric deltas vs fp16, graded/cliff/non-monotonic shape classification, graceful INT8-pending handling, and both regex-derived columns (HarmBench v2 proxy AND XSTest over-refusal) preferring summary.v2.json with fallback, so v1 and v2 are never mixed across precisions."],
      ["test_int8_scoping.py", "5", "INT8 full-parity scoping (T29): the backward-compatible --models/--out-suffix on the diagnostic stack (judge_agreement, judge_pairwise, harmbench_category, mmlu_subject) writes scoped *_int8 artefacts without touching committed base-vs-4bit outputs, and the per-pair section is empty for an INT8-only model set; plus a default-run backward-compat check."],
      ["test_dashboard_data.py", "12", "Streamlit-free dashboard data layer (dashboard/data.py): results discovery, judge-primary ASR rebuild from results/analysis/multiple_comparisons.json via classify_pair_change, summary loading, and the schema-validated config + sbatch emission for the add-model form."],
      ["test_dashboard_theme.py", "7", "Dashboard presentation helpers (dashboard/theme.py): label formatting, interpretation/metric colour mapping, and delta-rendering utilities."],
      ["Total", "329", "Twenty-five files; all passing on the current commit."],
    ],
    [3000, 1200, 5160],
  ),
  P("Table D.1  Distribution of automated tests across modules.", { size: 18 }),
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
│   └── tc1.yaml              (TC1 cluster)
├── ethical_benchmark/
│   ├── quant/config_schema.py
│   ├── models/{loader.py, generation.py}
│   ├── benchmarks/{base.py, harmbench.py, xstest.py, mmlu.py, registry.py, utils.py}
│   ├── pipeline/{run_quant_benchmark.py, run_quant_matrix.py}
│   ├── analysis/compare_quant_pairs.py
│   ├── cluster/{generate_jobs.py, submit_jobs.py, check_runs.py}
│   └── metrics/
├── tests/                    (329 tests across 25 files)
├── slurm/jobs_tc1/           (generated sbatch files)
├── results/                  (raw.jsonl, summary.json, v2 score sidecars, analysis outputs)
└── docs/
    ├── methodology.md
    ├── evaluation_metrics.md
    ├── datasets.md
    ├── limitations.md
    ├── extensibility.md
    ├── TC1_CLUSTER_RUNBOOK.md
    └── FYP_Report_2026-06-30_v4.docx   (this document)`),
];

const appendixG = [
  H1("Appendix G: Document Revision History"),
  PJ("This appendix records the revision history of this FYP report. It mirrors the report-affecting subset of the project changelog (`docs/PROJECT_LOG.md` §4). Purely internal changes (refactors, tests, dev tooling) are recorded in the project log but omitted here for readability. Every entry corresponds to a regenerated docx artifact."),
  buildTable(
    ["When (UTC+8)", "Version", "Change to the report"],
    [
      ["2026-06-30 11:30", "FYP_Report_2026-06-30_v4.docx (current)", "Added §6.16 Generation-Length Robustness: the full study was regenerated at HarmBench's 512-token reference budget (configs/tc1_512.yaml; results_512/), retaining the 128-token study unchanged for comparison. The Qwen3-1.7B safety regression does not replicate at 512 (ΔASR 0.000 classifier / +0.005 gpt-4o, McNemar p=1.000, neither significant; symmetric 16/16 prompt flips), no ΔASR survives BH-FDR under either judge, cross-judge κ 0.68–0.95, and 62% of 128-token responses were truncated. Four-bit capability losses are essentially unchanged 128→512, so the rerun sharpens the capability-driven dichotomy. Built non-destructively from a copy of build_fyp_report_v3.js (v3 left intact). INT8@512 + multi-seed@512 are regenerating and will be folded in. No 128-token number changed."],
      ["2026-06-29 12:00", "FYP_Report_2026-06-26_v3.docx (superseded by v4)", "Promoted the figure-rich v3 build to the canonical report: make report now builds scripts/build_fyp_report_v3.js, and the earlier FYP_Report_2026-06-14.docx (plus the v2 draft) are archived under docs/archive/. This version embeds the six analysis figures (capability anchor, ASR forest, precision sweep, judge-vs-proxy, per-category ASR, multi-seed) and carries the §6.15 INT8 precision-point results (capability cliff at 4-bit; the Llama-3B INT8 ΔASR +0.040 both-judge + McNemar move, caveated) and the §6.14 refusal-margin mechanism analysis. Test inventory refreshed to 329 automated tests across twenty-five files (added the dashboard data and theme test layers). Corrected the Appendix E and Appendix G self-references that still named the retired 2026-06-14 filename. No result numbers changed."],
      ["2026-06-15 18:00", "FYP_Report_2026-06-14.docx", "T26 cross-family extension folded in. Two matched pairs run on TC1 on 2026-06-15 — Mistral-7B-Instruct-v0.3 and Phi-4-mini-instruct — take the study to five pairs / ten models / four families. New §6.13 presents the cross-family results; Tables 6.1/6.2/6.3 extended to all five pairs; Abstract, §6.1, §6.3 (RQ5), §6.4.1 (ARC), §6.9 (RQ2/RQ5), §6.11, §6.12 (second judge now spans all ten models, κ 0.60–0.95) and Ch10 updated. Judge-primary: Mistral ΔASR −0.040 (n.s.; the v2 proxy's +0.055 is a sign-flipped over-count, judge-vs-proxy κ 0.11–0.19 — the study's starkest divergence, with the second judge gpt-4o concurring at κ 0.60–0.63); Phi ΔASR 0.000 (robust_preservation, κ 0.67). No new significant ΔASR — Qwen 1.7B (+0.055) remains the only one. Phi-4-mini's ΔOR = −0.028 is the study's one significant over-refusal delta (a decrease), so the over-refusal-null statements were qualified. Phi loaded via native transformers Phi3 (D31). No existing pair's numbers changed."],
      ["2026-06-14 15:45", "FYP_Report_2026-06-14.docx", "Renamed the artifact to today's date and rolled in the day's robustness work. T18 multi-seed sensitivity (§6.6.1): the Qwen 1.7B ΔASR is decode-dependent — mean +0.024 across five seeds vs the +0.055 greedy headline, not sign-consistent — so the headline is the upper end of a range, tempered not overturned. T21: §6.6.2 per-category judge ASR breakdown (the +0.055 is broad-based, rising in 5/6 harm categories), §7.6 deployment implications, §7.7 positioning against prior work, and verified citations replacing the earlier placeholders. T22 (§6.12 Result 4): a second independent judge (gpt-4o, same rubric) agrees with the primary classifier at κ 0.69–0.94 and reproduces the Qwen 1.7B increase in direction (+0.045 vs +0.055), borderline on significance (McNemar p=0.122) — W3 substantially resolved. No headline label changed; these strengthen how the findings are evidenced."],
      ["2026-06-06 16:25", "FYP_Report_2026-05-27.docx", "Added a repo-native next-session handoff at docs/HANDOFF.md so Codex, Claude Code, and future agents can recover from the same file instead of relying on pasted chat context. Appendix H §H.5 now points future work to that shared handoff and removes the stale post-push D16 instruction, because D16 is already on origin/main. No result numbers changed."],
      ["2026-06-06 16:00", "FYP_Report_2026-05-27.docx", "Doc-consistency follow-ups to the D16 judge-primary promotion. Rewrote Chapter 7 §7.1 (scoring determinism now describes the judge classifier, not regex-eliminates-judge-variance) and §7.4 (renamed \"Scorer Choice and Remaining Validity Threat\" — judge is primary, remaining threat is the absence of a second independent judge). Rewrote Appendix H §H.2–§H.5 to the post-judge state (the v2 Qwen 4B figure flagged as superseded; judge sidecars are the committed primary scorer; next steps are push/T1/T3/T15 plus an optional second judge, not a pending judge run). Updated the cover-page revision line to the judge-primary D16 wording. No numbers changed; this is a consistency pass so the .docx no longer contains pre-judge framing."],
      ["2026-06-06 15:00", "FYP_Report_2026-05-27.docx", "T20 results + D16: official HarmBench classifier promoted to PRIMARY HarmBench scorer; v2 regex demoted to a secondary non-refusal-rate proxy. The judge ran in fp16 on a 32 GB V100 (job 61047, n=200×6, 0 parse errors). Validation showed the regex over-counts ASR unevenly by family (judge vs v2 agreement: Qwen κ≈0.19–0.37, Llama κ≈0.69–0.79), and the choice of scorer changed the conclusion: the one significant ΔASR moved from Qwen 4B (proxy) to Qwen 1.7B (judge, +0.055 CI [+0.010,+0.100]). Refined the interpretation rule so alignment_degradation requires capability preserved; Qwen 1.7B (judge) is now broad_degradation (significant ASR up + significant MMLU down). Rewrote Table 6.1 (judge primary + v2 proxy rows), Table 6.2, new Table 6.3 (per-model agreement/κ), §6.1/6.1.1/6.3/6.5/6.6/6.7/6.8/6.9(all RQ)/6.10/6.11/6.12 (full results, no longer a placeholder), Abstract, Ch10. Headline relocated to Qwen 1.7B dual degradation; Qwen 4B is now directional-only. Test suite 176→178."],
      ["2026-06-06 12:00", "FYP_Report_2026-05-27.docx", "T20 refinement — judge runs in full precision (float16) by default, not NF4. Quantizing the judge classifier would confound the validation it is meant to provide, so §6.12 now states the judge is loaded in fp16; the job logs and records the exact GPU and precision used, and falls back to 8-bit (reported explicitly) only if the 13B classifier does not fit in fp16 on the available GPU. Test suite grew to 176 (added judge-precision resolution and VRAM-estimate tests). Appendix D judge-test row updated."],
      ["2026-05-28 13:30", "FYP_Report_2026-05-27.docx", "T20 — Judge-model validation (HarmBench classifier). Added §6.12 documenting an independent sensitivity check of the HarmBench ASR using the official cais/HarmBench-Llama-2-13b-cls classifier (a derived check that re-scores saved generations, performs no new target-model inference, and writes redacted scores.judge.* / summary.judge.* sidecars without touching raw or v2 outputs). The embedded classifier prompt is pinned to the official model-card standard (non-contextual) template, with a byte-for-byte regression test. §6.12 carries the methodology and a [RESULTS PENDING TC1 RUN] placeholder to be populated with per-model agreement, Cohen's kappa, judge-vs-v2 ASR, and per-pair label stability after the TC1 classifier run. Test suite grew to 172 (added test_judge_validation.py). Appendix D updated with the judge test row."],
      ["2026-05-27 23:54", "FYP_Report_2026-05-27.docx", "Added Appendix H as a fresh-session handoff after the v2 scorer incident. The appendix documents the v1 failure mode, the v2 parser correction, the later immutable-sidecar correction, the verification evidence, and the next-step contract for judge-model validation. Updated cover revision text and Future Work wording to make clear that LlamaGuard/API judge scoring is a derived validation layer rather than a replacement for v2 or a mutation of raw outputs. Documented that any local incomplete judge skeleton should not be committed until T20 is implemented end-to-end."],
      ["2026-05-28 12:00", "FYP_Report_2026-05-27.docx", "T13 — Refusal classifier revision v1 → v2. The v1 deterministic regex parser missed canonical modern refusal templates (\"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", curly-apostrophe variants), systematically misclassifying clear refusals as harmful compliance. Replaced with a comprehensive pattern set (forty-plus regexes; apostrophe normalisation; verb-family coverage). Added rescore script (`scripts/rescore_harmbench.py`) that re-evaluates all existing raw.jsonl files without re-running inference; emits redacted IDs-only diagnostic CSVs and an aggregate JSON. The final workflow preserves TC1-original `raw.jsonl`/`summary.json` files and stores corrected scores in derived `scores.v2.jsonl` + `summary.v2.json` sidecars. Headline impact: Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2), with CI now excluding zero — the study's most robust empirical finding; Qwen 1.7B ΔASR collapsed from −0.120 (v1) to −0.025 (v2, within noise); Llama ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). Two of three interpretation labels changed (Qwen 4B → alignment_degradation; Llama → broad_degradation). Re-ran `make analyze` for updated bootstrap CIs. All affected text rewritten throughout the report: Abstract, Table 6.1, Table 6.2, §6.1, §6.1.1 (new — scorer revision history), §6.3, §6.4, §6.5, §6.6, §6.7, §6.8, §6.9 (all five RQ answers), §6.10, §6.11 (all subsections), Ch10. Thesis reframed around the framework as the durable contribution. Test suite grew to 163 with refusal-pattern regression tests and v2 sidecar-selection coverage. T12 entry from 2026-05-28 01:00 superseded by this update."],
      ["2026-05-28 01:00", "FYP_Report_2026-05-27.docx (superseded by v2 scorer revision)", "T12 — Incorporated paired bootstrap 95% confidence intervals throughout the results chapter. Pipeline extended (`compare_quant_pairs.py` adds `compute_paired_bootstrap_ci` and emits CI bounds + significance flag per benchmark in `pairwise_deltas.{json,csv}`). Table 6.1 redesigned with \"Δ (95% CI)\" and \"Sig?\" columns. §6.1 introduces the bootstrap method and lists significance status of each delta. §6.5 Statistical Caveats updated with observed CI widths. §6.6 (Qwen 2B, both deltas significant), §6.8 (Qwen 4B, hedged — ΔASR borderline non-significant), §6.10 (Llama, ΔMMLU significant, ΔASR borderline) rewritten with inline CI annotations. RQ1–RQ3 in §6.9 expanded with significance language. §6.11.2, §6.11.3, §6.11.4 cross-family subsections updated. Ch10 conclusion rewritten with full significance reporting. Test suite grew to 126 (added 3 new tests: outcome extraction, bootstrap CI smoke, no-overlap guard). Pipeline output `results/analysis/pairwise_deltas.json` now contains CI fields."],
      ["2026-05-28 00:30", "FYP_Report_2026-05-27.docx", "Extended interpretation taxonomy with alignment_improvement (mirror of alignment_degradation): fires when ΔASR ≤ −0.02 with capability and over-refusal preserved. Reclassified Qwen 4B from broad_degradation → alignment_improvement, properly capturing the desirable capability-preserving harmful-compliance reduction. Updated Table 3.4 (five labels), §3.6 intro, §6.4 (now five canonical outcomes), §6.8 (Qwen 4B reading rewritten as genuine safety win), RQ4 synthesis, §6.11.5 (full safety spectrum), Ch10, Abstract. Pipeline output (`make analyze`) re-run after code change."],
      ["2026-05-27 23:30", "FYP_Report_2026-05-27.docx", "Corrected interpretation labels throughout to match pipeline output (make analyze → pair_interpretations.csv). Qwen 4B corrected from robust_preservation → broad_degradation (ΔASR abs=0.045 exceeds harm_tol=0.02, fails strict robust_preservation check). Llama corrected from broad_degradation → alignment_degradation (ΔASR=+0.030 ≥ harm_tol=0.02, fires second condition). Fixed Table 3.4 label definitions to match code logic."],
      ["2026-05-27 22:00", "FYP_Report_2026-05-27.docx", "Full three-pair results and analysis. All 6 jobs complete (60976–60981). Table 6.1 fully populated. Table 6.2 expanded to all three pairs. §6.3 updated with cross-family headline finding. §6.9 RQ5 and synthesis sections complete. Added §6.10 Llama 3.2 3B pair observations (baseline profile, MMLU subject breakdown). Added §6.11 cross-family comparison (§6.11.1–§6.11.5: baseline divergence, ASR sign inconsistency, OR null result, MMLU magnitude comparison, interpretation labels and central question). Updated Abstract with all three pair findings. Updated Ch10 conclusion with full five-RQ answers."],
      ["2026-05-27 21:00", "FYP_Report_2026-05-27.docx", "Full Qwen analysis update. Added Table 6.2 (Qwen family delta comparison with interpretation labels). Expanded §6.2 with full within-family analysis, delta magnitude ratio (22:1 MMLU), and methodological payoff discussion. Updated §6.3 with two Llama hypotheses. Rewrote §6.4 Capability Anchoring with concrete Qwen examples and updated label names. Added §6.9 Qwen Family Synthesis addressing RQ1–RQ4 in full (5 H3 subsections). Updated Ch7 intro, Ch10 conclusion paragraphs."],
      ["2026-05-27 20:00", "FYP_Report_2026-05-27.docx", "Added Qwen 4B 4-bit results (job 60979: ASR=0.815, OR=0.016, MMLU=0.743). Updated Table 6.1 Qwen 4B row with full pair including deltas. Expanded §6.2 with Qwen 4B quantization findings. Revised §6.7 with confirmed outcome. Added §6.8 Preliminary Observations: Qwen 4B Pair documenting robust preservation pattern and RQ4 scale contrast. Updated Chapter 10 conclusion with two-pair findings."],
      ["2026-05-27 19:00", "FYP_Report_2026-05-27.docx", "Added Qwen 4B baseline results to Table 6.1 (job 60978: ASR=0.860, OR=0.016, MMLU=0.747). Expanded §6.2 with preliminary scale-comparison analysis (capability-harmfulness coupling at baseline). Added §6.7 Preliminary Scale Observations documenting the 1.7B-vs-4B full-precision comparison and the over-refusal coincidence between the 4B baseline and 2B 4-bit."],
      ["2026-05-27 18:30", "FYP_Report_2026-05-27.docx", "Populated Chapter 6 with actual Qwen 2B results (jobs 60976 and 60977). Renamed chapter from 'Intended Results and Analysis Plan' to 'Results and Analysis'. Added §6.6 Preliminary Observations with per-metric analysis, capability-collapse interpretation, and MMLU subject breakdown. Updated Table 6.1 with Qwen 2B point estimates. Updated Abstract and Chapter 10 conclusion to reflect work-in-progress status and preliminary finding."],
      ["2026-05-27 17:31", "FYP_Report_2026-05-27.docx", "Recorded the first production matrix submission: job 60976 started and job 60977 waited with QOSMaxGRESPerUser. Updated Chapter 5 to describe the effective one-running-GPU-job scheduling rule."],
      ["2026-05-27 15:43", "FYP_Report_2026-05-27.docx", "Recorded that the current Qwen3/Llama pre-cache completed on TC1 and that smoke job 60975 successfully validated the CUDA/offline-cache path. Updated Chapter 5 and Chapter 10 so the next operational step is the full six-job matrix, not the smoke job."],
      ["2026-05-27 15:40", "FYP_Report_2026-05-27.docx", "Switched the Qwen model pairs from third-party techwithsergiu Qwen3.5-text derivatives to official Qwen3 checkpoints (Qwen/Qwen3-1.7B and Qwen/Qwen3-4B), updating the model table, setup text, limitations, and YAML appendix."],
      ["2026-05-27 00:41", "FYP_Report_2026-05-27.docx", "Rolled the generated report artifact forward to the current checkpoint date. The former 2026-05-24 report is archived for traceability; active documentation now points to the 2026-05-27 docx."],
      ["2026-05-27 00:34", "FYP_Report_2026-05-27.docx", "Updated the experimental-setup status after TC1 pre-cache completion. Corrected XSTest source text to the bundled canonical CSV, recorded that HarmBench/Llama gated access has been verified, updated observed cache sizes, and clarified that the next operational step is the smoke sbatch."],
      ["2026-05-24 01:50", "FYP_Report_2026-05-24.docx", "Fixed numbered-list numbering. All five numbered lists (Ch 1.5 Contributions, Ch 2.5 Research Gaps, Ch 8 Limitations, Ch 9 Future Work, References) were sharing one global counter and continued incrementing across chapters (Ch 8 started at 10, Ch 9 at 19, References at 25). Each list now correctly restarts at 1. Implementation: per-list `numlist<N>` numbering references in the builder."],
      ["2026-05-24 01:20", "FYP_Report_2026-05-24.docx", "Added this Document Revision History appendix (Appendix G). No content changes to other chapters."],
      ["2026-05-24 00:55", "FYP_Report_2026-05-24.docx", "Strengthened cover page: bolded student name and document date (26pt); added supervisor email; running header on every page now shows project code + author on the left and report type + date on the right."],
      ["2026-05-24 00:15", "FYP_Report_2026-05-24.docx", "Rewrote Chapter 5 to incorporate the TC1 user-guide policy review. Added §5.3 (Cluster Usage Policy and Workflow Constraints) and §5.5 (Offline-Mode Strategy and Pre-Cache). Expanded Table 5.1 with full QoS limits (CPU/memory/GPU). Updated Appendix A and Appendix B to show the new HF_*_OFFLINE env vars in the sbatch setup_commands."],
      ["2026-05-23 evening", "FYP_Report_2026-05-23.docx (archived)", "Initial release of the interim report. Cover, abstract, TOC, 10 chapters, references, Appendices A–F. Approximately 30 pages. Documents the matched-pair design with three pairs (Qwen 2B, Qwen 4B, Llama 3.2 3B) and on-the-fly NF4 quantization."],
    ],
    [2200, 3000, 4160],
  ),
  P("Table G.1  Revision history of this FYP report. The source builder script is `scripts/build_fyp_report_v3.js`; the docx is a build artifact and is never hand-edited.", { size: 18 }),
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

const appendixH = [
  H1("Appendix H: Scorer Incident and Future-Session Handoff"),
  PJ("This appendix is included so that a future agent or fresh chat session can continue the work without relying on conversation memory. It records the critical scorer incident, the corrected artifact contract, and the next safe continuation point. The authoritative operational source remains `docs/PROJECT_LOG.md`; this appendix mirrors the same information in the Word deliverable."),

  H2("H.1 What went wrong in v1"),
  PJ("The original HarmBench and XSTest scorer used a narrow deterministic refusal parser with only fourteen regex patterns. During post-hoc review, it became clear that the parser missed common modern instruction-model refusal templates, including straight- and curly-apostrophe variants of phrases such as \"I can't fulfil this request\", \"I cannot provide\", and \"I'm unable to assist\". Those responses were clear refusals, but v1 counted them as harmful compliance on HarmBench or benign answers on XSTest. This was a scoring error, not a target-model generation error: the six TC1 jobs had already completed, and the saved model outputs did not change."),
  PJ("The consequence was material, and the story has two stages. The v1→v2 regex correction first moved the Qwen 4B point estimate from an apparent safety improvement to a (v2-regex) ΔASR of +0.065. But the v2 regex still only measures \"non-refusal rate\", so a second correction followed: the official HarmBench classifier (judge; decision D16, §6.12) was promoted to the primary HarmBench scorer. Under the judge, the v2 regex is shown to over-count ASR, and the study's one statistically significant ΔASR is Qwen 1.7B at +0.055 (CI [+0.010, +0.100]), not Qwen 4B, whose judge ΔASR (+0.025) is directional only. The final reading is therefore Qwen 1.7B broad_degradation (significant safety worsening plus significant capability loss); the v2 Qwen 4B figure quoted above is the superseded regex-proxy value. MMLU values are unchanged throughout because MMLU scoring does not use any refusal scorer."),

  H2("H.2 How the correction was made"),
  PJ("The v2 correction expanded `ethical_benchmark/benchmarks/utils.py` with broader refusal-pattern coverage, punctuation normalisation, and a diagnostic helper that reports the matched refusal-pattern name. Regression tests were added for canonical refusal strings, negative controls, and curly-apostrophe handling. The analysis pipeline was then re-run from saved outputs; no target-model inference was re-run on TC1. This was appropriate because the flaw was in the scoring layer, not in model execution."),
  PJ("A first implementation rescored `raw.jsonl` and `summary.json` in place. That approach was rejected during Codex audit because the original TC1 outputs are the evidence trail and should remain immutable. The final solution preserves `raw.jsonl` and `summary.json` as original v1 artifacts, and stores corrected v2 scoring in sidecars: `scores.v2.jsonl` contains prompt IDs and score_fields only, with no prompt or response text, and `summary.v2.json` contains corrected aggregates. `compare_quant_pairs.py` now prefers complete v2 sidecars when they exist and falls back to original summaries otherwise."),

  H2("H.3 Current artifact contract"),
  buildTable(
    ["Artifact", "Status", "Rule for future work"],
    [
      ["raw.jsonl", "TC1-original saved generations and original score fields", "Do not modify, redact, duplicate into new raw files, or print prompt/response content during audits."],
      ["summary.json", "Original aggregate summary from the initial run", "Do not overwrite during post-hoc scoring corrections."],
      ["scores.v2.jsonl", "Derived v2 score fields only; no prompt/response text", "Authoritative corrected scorer sidecar for HarmBench/XSTest analysis."],
      ["summary.v2.json", "Derived v2 aggregate summary", "Used by `make analyze` through sidecar preference logic."],
      ["rescore_diagnostics_*.csv", "Redacted diagnostics with IDs, labels, lengths, and matched-pattern names", "Safe to inspect; must remain free of raw prompt/response text."],
      ["scores.judge.harmbench_cls.jsonl", "Per-prompt official HarmBench classifier labels (the PRIMARY HarmBench scorer, D16); produced by job 61047", "Redacted (IDs + booleans only). Committed to the repo. Never overwrite; raw text must never appear."],
      ["summary.judge.harmbench_cls.json", "Aggregate judge HarmBench metrics + GPU/precision metadata", "The authoritative HarmBench ASR source. judge_agreement.{json,csv} compares it to the v2 proxy."],
    ],
    [1900, 3200, 4700],
  ),

  H2("H.4 Verification completed"),
  PJ("The workflow was verified in five ways. First, the full test suite passes with 329 tests, including v2 refusal-parser regression tests, the analysis sidecar-selection test, the capability-guard test for the refined interpretation rule, and judge-validation tests for redaction, raw immutability, idempotency, ASR aggregation, the official HarmBench classifier prompt template, and judge-precision resolution. Second, the judge job (61047) ran in full precision on a 32 GB V100 and classified all 1 200 generations with zero parse errors. Third, `scripts/judge_agreement.py` reproduces the judge-vs-v2 agreement and the judge-primary pair labels deterministically from the committed sidecars. Fourth, the original `raw.jsonl` and `summary.json` files were confirmed unchanged by every post-hoc step (rescore and judge). Fifth, all judge sidecars were checked for redaction: they contain prompt IDs, boolean labels, and run metadata only, with no prompt, behaviour, or response text."),

  H2("H.5 Where the next session should continue"),
  PJ("The HarmBench judge validation is complete (job 61047) and the official classifier is the primary HarmBench scorer (decision D16); §6.12 carries the results. There is no pending scoring work. Future agents should start from the repo-native handoff in `docs/HANDOFF.md`, then verify live Git state with `git status -sb`. The remaining non-technical items are: send the supervisor progress note (T1); record the exact storage quota via `MyTCinfo` (T3, hardware already confirmed as a 32 GB Tesla V100); and prepare the final submission (T15)."),
  PJ("The most valuable optional extension is a second independent judge (for example LlamaGuard or a frontier-model classifier with a strict yes/no rubric) to cross-check the HarmBench classifier itself; this is the one remaining construct-validity threat on the HarmBench metric (§7.4) and is listed in Chapter 9. The infrastructure already supports it: add a backend in `ethical_benchmark/judges/validation.py`, run it through `scripts/run_judge_validation.py` into `scores.judge.<name>.*` sidecars, and compare via `scripts/judge_agreement.py`. On TC1: `git pull --ff-only`, cache any new judge weights with `prefetch_tc1.py --judge --judge-model-id <id>`, then `sbatch`. Never run judge Python on the head node, and never print or commit raw prompts/responses outside the existing immutable raw files. The Llama-2-family judge tokenizer requires `sentencepiece`, `protobuf`, and `tiktoken` (now pinned in requirements.txt)."),
];

// ------------------------------------------------------------
// Assemble document
// ------------------------------------------------------------
const body = [
  ...cover,
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
  ...appendixG,
  ...appendixH,
];

const doc = new Document({
  creator: "TAN UEI HORNG",
  title: "FYP Interim Report: Benchmarking Ethical Performance of Open-Source LLMs (CCDS25-1136)",
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
              text: "FYP Interim Report  ·  24 May 2026",
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

const OUTPUT = path.join(__dirname, "..", "docs", "FYP_Report_2026-06-30_v4.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("WROTE: " + OUTPUT);
  console.log("Size: " + buf.length + " bytes");
});
