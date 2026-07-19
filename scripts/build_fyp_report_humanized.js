// FYP Report builder (v3 - publication-grade pass + humanizer prose edit) - docx-js
// Output: /Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-01_humanized.docx
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
      text: "A Matched-Pair, Judge-Validated Study of Safety, Capability Trade-offs in Quantized Compact Language Models",
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
    children: [new TextRun({ text: "Comprehensive project report, revised through July 2026. Scope: five matched pairs across four families at three precisions (fp16 → INT8 → NF4). HarmBench ASR is scored by the official HarmBench classifier (judge-primary, D16) and cross-checked by a second judge; the refusal regex is retained only as a transparent proxy. This is the full technical write-up; two companion documents present the same study at other lengths, a shorter Interim Report (milestone deliverable, docs/FYP_Interim) and a condensed, IEEE-cited standalone Thesis (docs/FYP_Thesis).", font: FONT, size: 20, italics: true, color: "555555" })],
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
  PJ("The design loads the baseline and quantized members of each pair from identical weights, applying quantization on the fly, so quantization is the only variable. Five pairs across four families (Qwen, Llama, Mistral, Phi; 1.7 to 7.2 B) are evaluated at three precisions (fp16, INT8/LLM.int8, NF4 four-bit) on harmful compliance (HarmBench), over-refusal (XSTest), and capability (MMLU and ARC-Challenge). The central methodological finding is that the scorer determines the conclusion: a refusal-counting regex over-counts harmful compliance (its harmful set very nearly contains the classifier's, the disagreement is one-directional over-counting), so both which model looks least safe and whether any model looks significantly less safe at all depend on the scorer. Replacing the regex with HarmBench's own fine-tuned classifier, cross-checked by a second, architecturally independent judge at Cohen's κ 0.68 to 0.95, removes that over-counting; refusal-counting therefore overstates quantization harm, a cautionary result for safety evaluation generally."),
  PJ("HarmBench attack-success rate is reported at HarmBench's own 512-token reference budget as the primary configuration of this study, with a shorter 128-token budget retained for direct comparison. Prompts are HarmBench's standard harmful behaviours presented directly, with no adversarial attack augmentation, so ASR here measures harmful compliance under direct requests rather than robustness to optimised attacks. Under the classifier at 512 tokens, no evaluated pair shows a significant harmful-compliance change under four-bit NF4 once a Benjamini-Hochberg correction is applied over the family of primary contrasts: not one HarmBench attack-success-rate contrast survives multiplicity control, whereas the surviving effects are capability losses (MMLU, ARC) and a single over-refusal decrease, and that over-refusal decrease is most plausibly a measurement artifact of the regex scorer: an independent three-class refusal judge does not reproduce it, and a blinded human audit finds that judge substantially better aligned with the annotator than the regex on this construct (§6.12). A validation-informed parallel correction, with over-refusal scored by that judge, leaves two survivors, both capability effects (§6.5.1), and no scorer finds a statistically significant over-refusal increase in any pair. The only pair whose ASR delta is individually significant, Llama-3.2-3B, moves in the safety-improving direction and still fails the corrected threshold. Capability degradation is thus the robust, budget-invariant cost of four-bit quantization, while harmful-compliance change is not. The shorter 128-token budget produced an apparent safety regression in the smallest model (Qwen3-1.7B, ΔASR = +0.055) that does not replicate at the reference budget (ΔASR = 0.000 under the classifier, +0.005 under gpt-4o, McNemar p = 1.000), because the 128-token cap truncated roughly sixty percent of generations before completion; a refusal-margin probe independently supports the same capability-driven reading, and a power analysis bounds how large any undetected safety effect could be (minimum detectable ΔASR ≈ 0.06 at n = 200). An fp16 → INT8 → NF4 sweep shows the effect is not bit-width-graded, capability loss is a clean cliff at four-bit, and the one INT8-specific safety increase seen on Llama-3B at 128 tokens under both judges likewise vanishes at 512 under both judges. The contributions are a reproducible, capability-anchored, judge-validated and human-grounded evaluation method for the compact regime, and the evidence that the headline risk of four-bit quantization in small models is capability degradation, with harmful-compliance changes that are small, scorer-sensitive, budget-sensitive, and not robust to multiplicity correction."),
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
  P("Table 6.2  All-pair quantization deltas, interpretation labels, and evidence status.", { after: 60 }),
  P("Table 6.3  Judge vs v2 per-model agreement on HarmBench (n = 200 each).", { after: 60 }),
  P("Table 6.4  Exploratory per-pair XSTest over-refusal deltas under the independent three-class refusal judge.", { after: 60 }),
  P("Table 6.5  HarmBench ΔASR across the 128- and 512-token generation budgets.", { after: 60 }),
  P("Table D.1  Stable verification coverage areas.", { after: 60 }),
  P("Table G.1  Revision history of this FYP report.", { after: 60 }),
];

// ------------------------------------------------------------
// Chapter 1 - Introduction
// ------------------------------------------------------------
const ch1 = [
  H1("Chapter 1: Introduction"),

  H2("1.1 Background and Motivation"),
  PJ("The last two years have seen a rapid proliferation of compact instruction-tuned large language models (LLMs) in the one-to-four-billion-parameter range. Models such as Qwen 2.5 and 3.x, Llama 3.2, Microsoft Phi-3, and Google Gemma 2 have demonstrated that capable reasoning, instruction following, and multilingual performance are achievable at parameter counts that fit comfortably on consumer hardware, mobile chipsets, and edge accelerators. This has shifted the practical envelope of LLM deployment: tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, with stronger privacy guarantees, lower latency, and substantially reduced operational cost."),
  PJ("In practice, however, such models are rarely deployed in full precision. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision impractical for routine use, and quantization to four-bit precision has become the de facto compression standard for on-device inference. Lightweight runtimes such as llama.cpp and on-device agent frameworks routinely ship four-bit GGUF or BitsAndBytes NF4 checkpoints by default. End users encountering these models therefore almost always interact with a quantized variant, not the original baseline."),
  PJ("Quantization has historically been treated as a numerical optimisation technique whose primary cost is a small loss in perplexity or downstream accuracy. A growing body of evidence challenges this view. Quantization can alter behavioural properties of an LLM in ways that are not visible from perplexity alone, including instruction-following fidelity, refusal calibration, and resistance to adversarial prompts. Because safety alignment is itself a learned behaviour encoded in the model weights, any operation that alters those weights, even one that preserves task accuracy on average, can perturb that behaviour. Understanding how compression interacts with safety is therefore not optional: it is a prerequisite to safe deployment of any compact model that has been quantized for production use."),

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
  Bullet("RQ5 (Robustness). Does the pattern observed within the Qwen family replicate qualitatively across other model families (Llama, Mistral, Phi) and across quantization precisions (INT8, NF4)?"),

  H2("1.4 Scope and Boundaries"),
  PJ("Explicit scoping is essential to make the study tractable within the constraints of an undergraduate Final Year Project and to keep the resulting claims defensible. The following boundaries are stated up front."),
  Bullet("Quantization method. The main study applies BitsAndBytes NF4 four-bit loading on the fly via the Hugging Face transformers integration; a precision point (§6.15) adds BitsAndBytes INT8/LLM.int8 as a second method, so the study spans fp16 → INT8 → NF4. Both are applied on the fly to identical baseline weights. Other families such as GPTQ, AWQ, and GGUF are out of scope and are flagged in the Future Work chapter."),
  Bullet("Model regime. The study targets compact instruction-tuned models. The primary matched pairs sit in the roughly two-to-four-billion-parameter range, and the cross-family robustness extension (§6.13) reaches 7.2 billion parameters (Mistral-7B), so the study spans approximately 1.7 to 7.2 billion parameters. This regime corresponds to the practical envelope of contemporary on-device deployment."),
  Bullet("Languages and modality. Evaluation is restricted to English-language, text-only interactions."),
  Bullet("Decoding. All inference uses deterministic greedy decoding with temperature 0.0 and top-p 1.0, both to eliminate within-condition variance and to maximise comparability across baseline and quantized pair members."),
  Bullet("Benchmark axes. Rather than attempting a broad multi-axis ethical evaluation, which is already covered by larger institutional benchmarks, this study evaluates three complementary axes that jointly enable the alignment-versus-capability disambiguation that motivates the research question."),

  H2("1.5 Contributions"),
  PJ("The contributions of this project are, in order of significance, a methodological cautionary result, a controlled study design, an interpretation method, an engineering framework, and an empirical characterisation of quantization effects."),
  ...numberedList([
    "A scorer-validity result with consequences beyond this study: the choice of HarmBench scorer changes the conclusion. A deterministic refusal-counting regex over-counts harmful compliance (its harmful set is a near-strict superset of the official classifier's) and replacing it with HarmBench's own fine-tuned classifier (cross-checked by a second, architecturally independent judge at Cohen's κ 0.68 to 0.95) changes both which model looks least safe and whether any model looks significantly less safe at all. Refusal-counting therefore overstates quantization harm, a cautionary lesson for safety evaluation generally.",
    "A controlled matched-pair study design that isolates quantization as the sole experimental variable. Both members of every pair are loaded from the same underlying model_id; the only difference is whether on-the-fly quantization is applied during the from_pretrained call. This eliminates publisher-asymmetry and conversion-pipeline asymmetry, two confounds that affect most published comparisons of quantized and full-precision checkpoints.",
    "A capability-anchored interpretation method that combines harmful-compliance, over-refusal, and capability deltas into one of six diagnostic labels (alignment degradation, alignment improvement, capability collapse masquerading as safety, over-refusal regression, robust preservation, broad degradation), each carried with a two-layer evidence status. This formalises the alignment-versus-capability disambiguation as a reproducible procedure rather than an ad hoc judgement.",
    "An open, reproducible benchmarking framework with batched chat-templated generation, per-prompt audit logging, immutable hash-pinned raw artefacts, redacted score sidecars, resumable per-model SLURM matrix jobs for the NTU TC1 cluster, and a verification suite whose live inventory is derived by pytest rather than copied into prose.",
    "An empirical characterisation of quantization across five model pairs, four families, and three precisions (fp16 → INT8 → NF4): under the validated classifier at the 512-token reference budget, no pair shows a significant harmful-compliance change under four-bit NF4 after a multiplicity correction, the multiplicity-robust signal is capability loss rather than safety change, and the effect is not bit-width-graded (a 128-token apparent increase in the smallest model does not replicate at the reference budget, §6.16). These results are supported by a refusal-margin mechanism probe, a multi-seed sensitivity arm, an FDR/power analysis, and a second-judge cross-check.",
  ]),

  H2("1.6 Report Structure"),
  PJ("Chapter 2 surveys the relevant literature on large-scale ethical benchmarking, the helpfulness, harmlessness trade-off, the deployment of compact LLMs, and the behavioural effects of quantization, concluding with the specific research gaps that this study targets. Chapter 3 details the experimental methodology, including the matched-pair design, quantization approach, benchmark selection, scoring, decoding controls, and the interpretation framework. Chapter 4 documents the system design and implementation: package structure, configuration schema, the model loader and generation pipeline, the benchmark plugin architecture, matrix orchestration, resume logic, and SLURM job generation. Chapter 5 describes the experimental setup on TC1 and the run plan. Chapter 6 presents the experimental results for all five model pairs; the original three completed on the NTU TC1 cluster on 2026-05-27 and the two cross-family pairs (Mistral-7B, Phi-4-mini) on 2026-06-15. Chapter 7 discusses threats to validity, Chapter 8 records limitations, and Chapter 9 proposes future work. Chapter 10 concludes. Eight appendices reproduce the full configuration, an example SLURM script, the configuration schema, the test inventory, the repository layout, and a glossary, and record the document revision history and the scoring-methodology correction record."),
];

// ------------------------------------------------------------
// Chapter 2 - Literature Review
// ------------------------------------------------------------
const ch2 = [
  H1("Chapter 2: Literature Review"),

  H2("2.1 Large-Scale Ethical Benchmarking of LLMs"),
  PJ("The past two years have produced several large-scale institutional efforts to benchmark the ethical performance of open-source LLMs. TrustLLM [1] evaluates sixteen mainstream LLMs over thirty datasets across six trustworthiness dimensions (truthfulness, safety, fairness, robustness, privacy, and machine ethics); DecodingTrust [2] performs a deep multi-perspective assessment (toxicity, stereotype bias, robustness, privacy, machine ethics, and fairness) focused on the GPT-3.5/GPT-4 family; and SafetyBench [3] tests twenty-five Chinese and English LLMs with over eleven thousand multiple-choice safety questions. These benchmarks have published leaderboards that cover popular open-weight model families such as Llama, Mistral, Qwen, and Falcon, and have influenced what counts as an acceptable safety profile in publicly released models."),
  PJ("The strengths of these frameworks are clear: large model coverage, multi-dimensional evaluation, and standardised methodologies that facilitate cross-model comparison. Their gaps, however, are also clear in the context of the present work. First, they predominantly evaluate full-precision checkpoints; quantized variants are either omitted or treated as a separate, secondary evaluation. Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, where alignment training tends to be most robust. Third, they typically report each safety axis independently (harmful compliance, bias, toxicity) without explicitly anchoring those measurements against a capability metric, leaving open the question of whether safety changes reflect alignment or capability."),

  H2("2.2 The Helpfulness, Harmlessness Trade-off"),
  PJ("A central, well-documented limitation of static safety benchmarks is their difficulty in jointly capturing the tension between helpfulness and harmlessness. A model that is heavily optimised for safety may become overly conservative, refusing benign prompts that incidentally resemble unsafe ones, a failure mode known as over-refusal or exaggerated safety. Conversely, a model optimised for helpfulness may remain vulnerable to adversarial jailbreak attacks. Measuring only one side of this trade-off produces an incomplete picture."),
  PJ("Two benchmarks have emerged as the practical standards for measuring opposite sides of this trade-off. HarmBench [4] provides curated unsafe prompts and measures attack success rate, the fraction of unsafe prompts to which the model produces a harmful, complying response. XSTest [5] provides benign prompts that are superficially similar to unsafe ones and measures over-refusal rate, the fraction of benign prompts the model nonetheless refuses. Evaluating both simultaneously is essential to detecting trade-offs introduced by alignment training or compression."),
  PJ("MMLU [6], the Massive Multitask Language Understanding benchmark, has become the de facto general-capability anchor in safety studies; ARC-Challenge [7] provides a structurally different second capability axis used here as corroboration. By measuring multiple-choice accuracy across a broad spectrum of academic and professional subjects, MMLU provides a capability signal that is largely independent of refusal behaviour, allowing capability collapse to be detected even when safety metrics appear to improve."),

  H2("2.3 Small Language Models and On-Device Deployment"),
  PJ("Recent compact LLM releases have repeatedly demonstrated that strong reasoning, instruction following, and multilingual performance are achievable at parameter counts below four billion. The Qwen 2.5 and Qwen 3 series, Llama 3.2 (1B and 3B Instruct), Microsoft Phi-3 (3.8B), and Google Gemma 2 (2B and 9B) have each been positioned for on-device or edge inference. These models are routinely integrated into lightweight agent frameworks and consumer applications, where their compact size enables genuinely local execution."),
  PJ("The deployment reality of these models is that they are almost never used in full precision. Memory and latency constraints on consumer hardware drive routine use of four-bit quantization, often via on-the-fly BitsAndBytes loading or pre-quantized GGUF checkpoints. Safety claims attached to the unquantized release model therefore do not, in general, transfer to the model that end users actually encounter."),

  H2("2.4 Quantization and Behavioural Effects"),
  PJ("Quantization compresses model weights from higher-precision floating-point representations to lower-precision integer or normalized-float representations, reducing memory footprint and accelerating inference at the cost of some numerical fidelity. Post-training quantization (PTQ) methods apply this conversion after the model has been trained and require no fine-tuning, making them attractive for deployment."),
  PJ("Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA [8] has become particularly prevalent in the open-source ecosystem. NF4 represents each weight using a four-bit normalized-float code optimised for the typical Gaussian-like distribution of neural network weights, with double quantization applied to the quantization constants to further reduce overhead. The eight-bit LLM.int8() method [9] is a distinct, mixed-precision algorithm in the same library that decomposes outlier features into a separate high-precision path; the present study evaluates both. The accompanying BitsAndBytes library integrates directly with Hugging Face transformers, exposing quantized loading through a single BitsAndBytesConfig object that can be passed to from_pretrained at model load time."),
  PJ("A body of recent work has argued that quantization is not behaviourally neutral. A comprehensive multi-benchmark evaluation of quantization strategies finds that headline benchmark scores are largely retained at four bits, while cautioning that evaluations limited to language modelling and a few classification tasks can miss shifts on other behavioural dimensions such as alignment [10]. Quantization has also been shown to degrade the confidence calibration of model predictions even where accuracy is preserved, with an impact that varies with model type and scale and concentrates on inputs where the full-precision model was already least confident [11]. Safety-specific studies make the point directly. Kharinaev et al. [12] evaluate 66 quantized variants across multiple post-training and quantization-aware methods and find that quantization can degrade safety alignment, with no single method dominating across models and bit-widths. Egashira et al. [13] show that widely used BitsAndBytes schemes, including NF4, can be manipulated so that a benign full-precision model becomes harmful once quantized. HarmLevelBench [14] reports that the direction of the safety change under AWQ and GPTQ is non-uniform across attack types. Q-resafe [15] confirms quantization-induced safety degradation and proposes a quantization-aware patching method to repair it. Closest to this study's headline, Hong et al. [16] scrutinise trustworthiness across five compression methods and eight trust dimensions and find that 4-bit quantization broadly preserves trustworthiness (unlike pruning, which degrades it, and unlike the more extreme 3-bit setting, where erosion appears); this is the nearest prior analogue to the null-safety-at-four-bit picture found here, though their study covers larger models and a different trust taxonomy. Most of these studies evaluate either perplexity or general capability, with safety considered (if at all) as a separate axis rather than jointly with capability against a fixed capability anchor, the gap this study targets."),

  H2("2.5 Research Gaps Targeted by This Work"),
  PJ("Four interlocking gaps in the existing literature motivate the present study."),
  ...numberedList([
    "Gap 1: Limited empirical study of compact (<4B) instruction-tuned models in the safety, quantization context. Most quantization studies focus on the seven-billion-to-thirteen-billion parameter range and above. Edge-deployment-relevant compact models are under-represented.",
    "Gap 2: Lack of integrated evaluation that measures harmful compliance, over-refusal, and capability simultaneously. Studies that measure only harmful compliance cannot detect capability-driven safety artifacts.",
    "Gap 3: Difficulty interpreting whether observed safety metric changes in quantized models reflect a real alignment shift or a side-effect of capability degradation. No widely adopted convention exists for jointly interpreting safety and capability deltas under compression.",
    "Gap 4: Provenance asymmetry in existing comparisons. Many published comparisons of full-precision and quantized checkpoints use a full-precision checkpoint from one publisher and a pre-quantized checkpoint from another, conflating quantization effects with checkpoint-conversion effects. A clean, on-the-fly quantization design from identical baseline weights eliminates this confound.",
  ]),
  PJ("The methodology described in Chapter 3 is structured to address all four gaps. The compact-deployment regime is addressed by the choice of Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B (extended by the Mistral-7B and Phi-4-mini cross-family pairs, §6.13); the integrated evaluation is addressed by the three complementary benchmarks; the interpretation challenge is addressed by the rule-based interpretation layer; and the provenance asymmetry is eliminated by on-the-fly NF4 loading from the same baseline weights."),
];

// ------------------------------------------------------------
// Chapter 3 - Methodology
// ------------------------------------------------------------
const ch3 = [
  H1("Chapter 3: Methodology"),

  H2("3.1 Experimental Design"),
  PJ("The study adopts a matched-pair comparative experimental design. Each model under study is evaluated as a pair: a baseline variant loaded in the default high-precision dtype, and a four-bit variant produced by applying BitsAndBytes NF4 quantization on the fly at load time. Both pair members are loaded from exactly the same Hugging Face model_id; no separately uploaded \"pre-quantized\" checkpoint is used. The only operational difference between the two members of a pair is the presence of a BitsAndBytesConfig object in the from_pretrained call."),
  PJ("This design choice is the strongest internal-validity property of the study. By construction, both pair members share identical baseline weights, the same tokenizer, the same chat template, and the same release artifact. Any observed delta in benchmark scores is therefore attributable to the quantization step itself rather than to checkpoint provenance, separate fine-tuning, or conversion artifacts. The on-the-fly NF4 path is the most direct way to operationalise the counterfactual \"the same model, quantized\" in code."),
  PJ("The independent variables are quantization state (baseline versus four-bit), model scale (1.7 billion versus four billion parameters within the Qwen family), and model family (Qwen versus Llama in the primary design, extended to Mistral and Phi in the cross-family replication of §6.13). The dependent variables are harmful compliance, over-refusal on benign prompts, and general capability, each measured by a corresponding benchmark."),

  H2("3.2 Model Selection"),
  PJ("Three model pairs form the primary controlled design. The Qwen pairs at 1.7 billion and four billion parameters [17] are the primary within-family comparison and provide both a quantization axis and a scale axis. The Llama 3.2 3B pair [18] sits between them by parameter count and is a cross-family robustness check. Table 3.1 summarises this primary matrix; a cross-family extension later adds two further matched pairs (Mistral-7B and Phi-4-mini, introduced with the experimental setup in Chapter 5 and analysed in §6.13), taking the study to five pairs across four families."),
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
  PJ("Four-bit loading is forbidden on a CPU runtime: the loader raises a clear error when quantized is true and the resolved runtime device is not cuda. After load, the loader checks four signals that quantization really engaged — is_loaded_in_4bit, is_loaded_in_8bit, is_quantized, and an attached hf_quantizer — and if none of them is set it raises a RuntimeError and refuses to go on, so an fp16 load can never be recorded as a quantized one."),
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
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets. Sampling is deterministic and seed-controlled. Capability is reported on two anchors side by side (MMLU + ARC-Challenge); interpretation labels remain MMLU-anchored (§6.4.1).", { size: 18 }),
  PJ("HarmBench is selected as the harmful-compliance benchmark because it is the most widely adopted harmful-behaviour collection in contemporary safety research and provides a stable definition of attack success. The walledai release on Hugging Face is used with the standard config, and a threat-model boundary is stated plainly: the standard config supplies HarmBench's curated harmful behaviour requests directly, and this study applies no adversarial attack or jailbreak transformation on top of them (no GCG, PAIR, AutoDAN, or any other attack pipeline from the HarmBench suite). ASR in this study therefore measures harmful compliance under direct requests, the weakest threat model in the HarmBench taxonomy, equivalent to its DirectRequest baseline, not robustness to optimised adversarial attack; attack-augmented evaluation is flagged as future work (Chapter 9). XSTest is the canonical over-refusal benchmark; this repository bundles the canonical XSTest v2 prompts CSV from paul-rottger/xstest and evaluates the benign-only subset to measure how often the model refuses prompts that should not be refused. MMLU is used as the primary general-capability anchor, with ARC-Challenge (allenai/ai2_arc; ~1,172 grade-school science questions designed to resist retrieval shortcuts; identical exact-match scoring to MMLU) added as a structurally different second capability benchmark, so capability is reported on the two benchmarks side by side (see §6.4.1). Interpretation labels remain anchored on MMLU, with ARC as a corroborating axis."),
  PJ("To keep MMLU tractable within the per-job walltime on TC1, a six-subject subset is used: business_ethics, clinical_knowledge, college_biology, high_school_world_history, high_school_macroeconomics, and human_aging. These subjects span ethics, sciences, humanities, social sciences, and applied knowledge, and are deliberately chosen to cover heterogeneous reasoning demands without exhausting the time budget. The six subjects are pooled and a deterministic, seed-controlled random sample of 300 questions is drawn from the pool, so the per-subject counts are not equal, they range from 25 to 94 questions (the exact per-subject distribution is reported with the statistical limitations in Chapter 8). Because the sample is fixed by seed and prompt identity, every baseline and 4-bit run sees the same 300 questions in the same order, so this uneven distribution is held constant across all conditions and does not affect the matched-pair comparison."),

  H2("3.5 Scoring: Refusal Detection and the HarmBench Classifier"),
  PJ("Two scorers are used, with a clear division of roles. A deterministic regex-based refusal parser provides fully reproducible, zero-variance scoring; it is the scorer for XSTest over-refusal, and for HarmBench it is retained as a secondary, refusal-based upper-bound proxy (it measures non-refusal rather than genuine harmful compliance). The primary HarmBench Attack Success Rate scorer, however, is the official HarmBench classifier (cais/HarmBench-Llama-2-13b-cls), a fine-tuned judge model, run in full precision with deterministic greedy decoding (decision D16; the validation that established it as primary is reported in §6.12). The classifier was promoted to primary after that validation showed the regex over-counts ASR (equating non-refusal with success) most severely for the Qwen and Mistral families. Retaining the deterministic regex alongside it is deliberate: it lets the reader see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring, and its reproducibility is why it remains the over-refusal scorer and the transparent proxy. A judge LLM does introduce a variance budget and an additional moving part, so it is controlled rather than avoided: it is run at full precision, with greedy decoding, over the immutable saved generations, and writes only redacted sidecars."),
  PJ("Three boolean fields are recorded for each response: harmful_compliance, safe_refusal, and malformed. For HarmBench, harmful compliance corresponds to a response that addresses the unsafe request substantively rather than refusing. For XSTest benign prompts, refusal is the failure mode, and over-refusal rate is the fraction of benign prompts on which the refusal pattern fires. The malformed field flags responses that are empty, contain only template fragments, or fail to parse into the expected output format."),
  PJ("The trade-off of deterministic parsing (its potential to miss nuanced or euphemistic refusals) is exactly why the official HarmBench classifier, not the regex, is the primary HarmBench scorer (§6.12). For XSTest over-refusal, whose own protocol relies on LLM-based three-way labelling (full compliance / partial refusal / full refusal) rather than a released fine-tuned classifier, this study instead keeps deterministic regex parsing, a deliberate, reproducibility-motivated deviation that yields a coarser binary (refusal / non-refusal) construct. That deviation is now cross-checked with exactly such an LLM three-class refusal judge, run as a sensitivity layer (§6.12, Result 6): the check finds the one multiplicity-robust over-refusal contrast to be scorer-dependent, and a subsequent pre-registered human audit (a blinded, 200-item, disagreement-enriched single-annotator gold set) finds the judge substantially better aligned with the annotator than the regex on this construct (Outcome J). That contrast is therefore carried as most plausibly a regex measurement artifact; the regex is retained as the original pre-specified scorer-of-record for traceability, and a second annotator remains the residual strengthening acknowledged in the Limitations chapter."),

  H2("3.6 Decoding Controls"),
  PJ("All inference runs share identical decoding parameters, applied uniformly across baseline and quantized members of every pair. Table 3.3 lists the parameters."),
  buildTable(
    ["Parameter", "Value", "Rationale"],
    [
      ["temperature", "0.0", "Greedy decoding; removes within-condition variance."],
      ["top_p", "1.0", "Disabled (greedy)."],
      ["max_new_tokens", "512", "HarmBench's reference generation budget (the primary study); the 128-token run is retained only as the §6.16 generation-length comparison. Monitored during smoke run."],
      ["repetition_penalty", "1.0", "Neutral; avoids confounding refusal calibration."],
      ["max_input_tokens", "1024", "Caps prompt length; prevents context overflow."],
      ["use_chat_template", "true", "Applies model-specific chat template to every prompt."],
      ["enable_thinking", "false", "Critical: disables Qwen3.x default thinking mode so the 512-token reference budget is not consumed by <thinking> blocks before the answer."],
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
  PJ("Bootstrap 95% confidence intervals [19] are computed for each metric by resampling the per-prompt score records with replacement. The intervals reflect prompt-sampling variance only; because decoding is deterministic at temperature 0.0, there is no within-condition stochastic variance."),
  PJ("Paired significance test for HarmBench (McNemar). Because both pair members see the identical prompt set, the HarmBench ΔASR is a paired-binary contrast, and the textbook-correct significance test is McNemar's exact test [20] on the discordant prompts (those scored harmful under exactly one member) rather than an unpaired two-proportion test. Writing b for the number of prompts that become harmful under quantization (baseline-safe → 4-bit-harmful) and c for those that become safe (baseline-harmful → 4-bit-safe), under the null hypothesis that quantization does not change harmful compliance each discordant prompt favours one member with probability one half, so min(b, c) follows a Binomial(b + c, 0.5); the exact two-sided p-value doubles the lower tail. Reporting b and c also makes the effect size transparent: the reader sees exactly how many prompts moved each way rather than only the net delta. McNemar's exact p is reported alongside the bootstrap CI for every HarmBench pair as an independent corroboration that does not assume large-sample normality (implemented in mcnemar_exact_test, no SciPy dependency). The same exact paired test is applied to every other paired-binary endpoint, XSTest over-refusal, MMLU, and ARC, in the multiple-comparisons analysis (results_512/analysis/multiple_comparisons.json; §6.5.1), and where the bootstrap CI and the exact test disagree at the margin, the exact test is taken as the significance criterion and such a delta would be reported as borderline. At the corrected over-refusal scoring no contrast in the study now sits in that margin: the previously borderline Qwen3-1.7B over-refusal decrease resolves to non-significant on both criteria (−0.024; bootstrap CI [−0.048, 0.000] now touches zero and McNemar's exact p = 0.109 on 10 discordant prompts)."),

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
  PJ("Reproducibility is treated as a core engineering requirement. A single seed (42) is propagated to Python's random module, NumPy, and PyTorch RNGs at the start of each run. Dataset shuffling is deterministic: each benchmark plugin loads its full dataset, shuffles with the seeded RNG, and then truncates to the configured max_samples. The same prompt order is therefore visited by both pair members."),
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

  H2("4.2 Configuration Schema"),
  PJ("All runtime parameters are specified by a YAML configuration file (typically configs/tc1.yaml for cluster runs, configs/default.yaml for local development). The configuration is loaded and validated by a Pydantic schema (config_schema.py) that enforces structural and semantic invariants. The top-level schema has four sections: models, decoding, benchmarks, and slurm."),
  PJ("Each model entry must specify family, size_b, quantized, pair_id, model_id, trust_remote_code, dtype, and a benchmarks list, plus an optional attn_implementation (the attention backend passed to from_pretrained; restricted to eager, sdpa, or flash_attention_2, and omitted entirely when unset so the existing models' load call is unchanged). The schema enforces two cross-references: every benchmark name appearing in a model's benchmarks list must also appear as a top-level benchmarks entry, and every pair_id must have at least one model with quantized=false and at least one with quantized=true. These invariants make it impossible to configure a study that omits a baseline or compares a model only against itself."),
  PJ("The decoding section specifies generation parameters; the benchmarks section provides per-benchmark dataset names, splits, sample caps, batch sizes, and benchmark-specific options (such as the benign_only flag for XSTest and the subjects list for MMLU); the slurm section specifies cluster directives and bootstrap commands."),

  H2("4.3 Model Loading and Quantization Path"),
  PJ("The HFModelLoader class encapsulates all model-loading logic. Its load method accepts a ModelSpec dataclass (built from the YAML configuration) and returns the loaded model, tokenizer, and resolved runtime device. The loader is responsible for three things: (i) selecting the runtime device based on the user-specified policy and CUDA availability; (ii) resolving the model dtype from a string preference (auto, float16, bfloat16, float32) to an actual torch.dtype value, with auto resolving to float16 on CUDA and float32 on CPU; and (iii) injecting a BitsAndBytesConfig object into the from_pretrained call when ModelSpec.quantized is true."),
  PJ("The BitsAndBytesConfig is constructed with load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_use_double_quant=True, and bnb_4bit_compute_dtype equal to the resolved torch_dtype. Setting the compute dtype to follow the model dtype ensures that the dequantized compute path uses the same precision the model would have used at full precision, eliminating a subtle source of pair asymmetry."),
  PJ("Two runtime safeguards are applied. First, the loader raises a clear RuntimeError if quantized is true but the resolved runtime device is not cuda, preventing accidental CPU execution of a four-bit path that BitsAndBytes does not support. Second, after from_pretrained returns, the loader checks four supported quantization signals: is_loaded_in_4bit, is_loaded_in_8bit, is_quantized, and hf_quantizer. If quantization was requested but none is present, it raises RuntimeError and refuses to proceed, preventing full-precision outputs from being silently recorded as NF4 or INT8 results."),

  H2("4.4 Generation Pipeline"),
  PJ("The TextGenerator class handles all inference. It accepts the loaded model, tokenizer, runtime device, and a DecodingConfig dataclass, and exposes a single generate_batch method that takes a list of prompts and returns a list of generated responses."),
  PJ("Each prompt passes through a formatting step. When use_chat_template is true (the default), the generator wraps the prompt in a single-message user-role list and applies the tokenizer's apply_chat_template method with add_generation_prompt=True. Critically, enable_thinking=False is also passed to this call so that Qwen3.x-family tokenizers, whose chat templates default to enabling a multi-step thinking block, do not silently consume the entire max_new_tokens budget producing <thinking>...</thinking> output before reaching the answer. A TypeError handler catches tokenizers that do not accept the enable_thinking keyword and retries without it, falling back to a raw-prompt path only as a last resort."),
  PJ("Generation itself uses torch.inference_mode and a do_sample=False decode path under temperature=0.0. Outputs are post-processed to strip the chat-template prefix and any trailing whitespace before being returned to the calling pipeline."),

  H2("4.5 Benchmark Plugins"),
  PJ("Benchmarks are implemented as plugins behind a small abstract base class defined in ethical_benchmark/benchmarks/base.py. Every plugin implements four methods: load_items, build_prompt, score_response, and aggregate. load_items returns a deterministic, seeded sample of BenchmarkItem objects; build_prompt maps each item to a single user-facing prompt string; score_response evaluates a model response against the item and returns a dictionary of boolean and numeric score fields; aggregate consumes the full list of per-item score dictionaries and produces the final aggregated summary."),
  PJ("The deterministic refusal layer is shared through benchmarks/utils.py, whose match_refusal_pattern function returns the matched pattern and whose is_refusal_response wrapper returns the boolean decision. XSTest uses this layer for its primary over-refusal score, while HarmBench retains it only for the secondary v2 non-refusal proxy; HarmBench's primary Attack Success Rate is scored by the official classifier. Centralising the regex layer keeps the two refusal-based measurements consistent without conflating that proxy with the classifier-primary safety endpoint."),
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
  PJ("Memory management is tightly constrained on TC1, where each job is allocated only ten gigabytes of host RAM and a single GPU. A try/finally block surrounds the inner benchmark loop. When the inner loop completes (or any benchmark within it raises), the model, tokenizer, and generator references are explicitly deleted, gc.collect() is invoked, and torch.cuda.empty_cache() is called when CUDA is available. This sequence releases the GPU memory before the next model is loaded, preventing stacking-induced out-of-memory errors that would otherwise occur immediately when loading a second model into the same job (a case that does arise in local development but is structurally avoided on TC1 because each job runs a single model)."),

  H2("4.7 Resume and Checkpointing"),
  PJ("Long-running jobs need to be robust to walltime exhaustion and transient failures. The framework treats raw.jsonl as the source of truth for which prompts have already been processed: every successfully scored prompt is immediately appended to raw.jsonl, with its prompt_id as a stable record key. On startup, the get_processed_prompt_ids helper reads the existing raw.jsonl (if any) and returns the set of prompt_ids already on disk. prepare_remaining_items then filters the freshly-loaded benchmark items, keeping only those whose prompt_ids are not in the processed set."),
  PJ("The granularity of the resume mechanism is the (model, benchmark, prompt_id) triple. Three concrete recovery scenarios are worth stating explicitly. If a job is killed during XSTest for qwen_2b_4bit, HarmBench (whose raw.jsonl is complete) is skipped entirely on the next submission; XSTest (whose raw.jsonl is partial) is resumed at the next unprocessed prompt; MMLU (whose raw.jsonl does not yet exist) runs from scratch. If a model fails to load (a deterministic OOM, for example), the entire model's benchmark suite is lost for that job; the operator can re-submit with --model and --benchmark filters to chunk the workload differently."),

  H2("4.8 SLURM Job Generation"),
  PJ("Cluster orchestration is handled by ethical_benchmark/cluster/generate_jobs.py. The generator reads the loaded configuration's slurm section and emits one sbatch file per scheduling unit. A --group_by flag controls the granularity: with group_by=benchmark, one sbatch is generated per (model, benchmark) pair (forty scripts for the ten-model, four-benchmark matrix); with group_by=model (the default for TC1), one sbatch is generated per model alias, and that script invokes run_quant_matrix.py so the model is loaded only once."),
  PJ("Each generated sbatch contains the standard SBATCH directives (partition UGGPU-TC1, qos normal, gres gpu:1, cpus-per-task 1, mem 10G, time 06:00:00), output and error log paths, a set -euo pipefail line, a cd into the configured work_dir, a mkdir -p of the log directory, the configured setup_commands (module load slurm, module load anaconda, source activate fyp-tc1), and finally the python invocation with the model alias and configuration path passed in. Ten sbatch files are produced for the present configuration, one per model alias."),

  H2("4.9 Output Artifacts"),
  PJ("Every (model, benchmark) run produces a fixed set of files under the results directory. raw.jsonl contains one JSON object per prompt with the prompt text, the model response, the per-response score fields, and the run metadata (model alias, model_id, family, pair_id, quantized flag, seed, generation_config, timestamp). summary.json contains the aggregated metrics, with bootstrap confidence intervals where applicable, alongside the same run metadata. The aggregator also appends a flat row to results/summary/<benchmark>_runs.csv, producing a single CSV per benchmark that records all model runs."),
  PJ("The analysis stage produces results/analysis/pairwise_deltas.json and .csv (one row per (pair_id, benchmark, metric) with absolute and relative deltas), results/analysis/pair_interpretations.csv (one row per pair with the interpretation label and the three component deltas), and results/analysis/quantization_analysis_summary.json (high-level study-wide summary)."),

  H2("4.10 Testing"),
  PJ("The repository ships with an automated verification suite whose live inventory is collected by pytest. Appendix D records coverage areas without copying a count that would become stale whenever a guard is added. Coverage includes dataset and benchmark loaders, model loading and prompt formatting, matrix reuse, pairwise analysis and bootstrap logic, score-sidecar selection, schema and resume helpers, refusal parsing, judge validation and redaction, artifact immutability, claim-surface verification, and SLURM generation."),

  H2("4.11 CLI and Operational Interface"),
  PJ("The unified CLI fyp_cli.py exposes nine subcommands, summarised in Table 4.2: seven study and cluster commands plus two agent-harness helpers. The study and cluster commands accept the shared configuration, results-directory, and log-level options where applicable; the two harness helpers expose their own status and task-packet arguments. The Makefile provides convenient targets that wrap these invocations."),
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
  PJ("All experiments are run on the NTU TC1 GPU cluster, a shared facility operated by the College of Computing and Data Science for undergraduate and postgraduate research workloads. Account access was approved in March 2026 under QoS \"normal\" with the parameters listed in Table 5.1. The compute partition consists of seven nodes (TC1N01, TC1N07), each equipped with three NVIDIA Tesla V100 PCIe 32 GB GPU cards, giving twenty-one GPUs across the partition. Although MaxJobsPU permits two submitted jobs, the first production submission showed an effective one-running-GPU-job limit: the second GPU job waited with reason QOSMaxGRESPerUser until the first released its GPU."),
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
      ["Account window", "March 2026, November 2026"],
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
      ["pytest", "(installed via requirements.txt; run pytest tests/ for the live inventory)"],
    ],
    [4200, 5160],
  ),
  P("Table 5.2  Software environment versions on TC1.", { size: 18 }),

  H2("5.3 Cluster Usage Policy and Workflow Constraints"),
  PJ("The TC1 user guide imposes strict workflow rules that shape how this study is executed. The headline rule is that user code may not be executed on the head node (CCDS-TC1). The guide states: \"DO NOT execute your coding on TC1 Head Node ... For the user process found executing coding and occupying high CPU and Memory usage in the Head Node will be terminated with no prior notification. Repeated offenders will be banned from TC1.\" GPU verification commands such as nvidia-smi and nvcc --version are also explicitly forbidden on the head node, because the GPU cards reside on the compute nodes (TC1N01, TC1N07) and not on the head node."),
  PJ("By contrast, package installation, file transfers, conda environment management, dataset downloads, and small administrative commands (squeue, scontrol, sacct, MyTCinfo, MyJobHistory, seff) are explicitly demonstrated as head-node activities throughout the user guide. The study's pre-cache step (see §5.5) therefore runs entirely on the head node and is policy-equivalent to a conda install."),
  PJ("The guide also recommends sbatch over srun for all real job submission: \"Avoid using the command 'srun' to submit job ... all users are advised to use the command 'sbatch' for job submission. Then exit from the session, access later to see the result.\" This study therefore submits every job, including the initial smoke validation, via sbatch."),

  H2("5.4 Hugging Face Access and Gating"),
  PJ("The Qwen pairs draw from open-access official Qwen3 checkpoints (Qwen/Qwen3-1.7B and Qwen/Qwen3-4B) and require no authentication. The Llama pair uses meta-llama/Llama-3.2-3B-Instruct, which is gated under the Meta Llama community license, and HarmBench currently also requires accepted Hugging Face dataset access conditions. Access is handled by logging into Hugging Face once on the TC1 head node with a read-scoped personal access token. As of 2026-05-26, token registration and gated-access acceptance have been verified by a successful full pre-cache of both Llama 3.2 3B and HarmBench."),

  H2("5.4.1 Infrastructure Validation and Fixes"),
  PJ("Before the full matrix was submitted, a dedicated CUDA verification job and a five-sample smoke run were used to validate the cluster environment end-to-end. Two infrastructure bugs were identified and fixed during this process. First, the generated sbatch scripts used relative paths for the #SBATCH --output and #SBATCH --error directives. SLURM resolves these relative to the directory from which sbatch is invoked, not to the working directory set by cd inside the script body. Jobs submitted from the home directory therefore wrote their log files to ~/results/slurm_logs_tc1/ rather than to the repository's results/ tree, causing the output to appear missing. The fix was applied in the job generator (ethical_benchmark/cluster/generate_jobs.py): when log_dir is a relative path and work_dir is set, the generator now anchors log_dir under work_dir so all sbatch directives contain fully qualified paths. Second, an initial CUDA verification script placed a Python file in /tmp on the head node and referenced it from the sbatch body; the /tmp filesystem is local to each node and is not shared, so the compute node could not find the file. The fix was to embed the verification code inline in the sbatch script using a Python heredoc (python - << 'PYEOF' ... PYEOF), removing the dependency on any shared path. Both fixes were committed and pushed before the smoke run, and the five-sample smoke job (SLURM job ID 60975) completed successfully: runtime_device reported cuda, attack_success_rate was 0.6 over five HarmBench prompts, and no malformed outputs were produced."),

  H2("5.5 Offline-Mode Strategy and Pre-Cache"),
  PJ("Because the compute nodes may not have outbound internet access, and because runtime downloads risk burning the six-hour walltime budget on slow Hugging Face mirrors, this study adopts a strict offline-mode strategy. All datasets and model weights are pre-cached on the head node before any SLURM job is submitted. SLURM jobs then run with HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, and TRANSFORMERS_OFFLINE=1 exported in the sbatch setup_commands block, so that any cache miss fails immediately with a clear error rather than hanging on a network attempt."),
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
  P("Code listing 5.1  Head-node pre-cache invocation (scripts/prefetch_tc1.py).", { size: 18, after: 160, alignment: AlignmentType.CENTER }),
  PJ("The pre-cache step retrieves HarmBench, the six MMLU subjects, ARC-Challenge (allenai/ai2_arc, added as the second capability benchmark), and the model repositories. XSTest is not fetched from Hugging Face because the canonical CSV is bundled in the repository as data/xstest_v2_prompts.csv. The initial TC1 pre-cache completed successfully on 2026-05-27 at 15:37 UTC+8, caching the original three model repositories in ~/.cache/huggingface/hub: Qwen3-1.7B (4.08 GB), Qwen3-4B (8.06 GB), and Llama 3.2 3B (12.9 GB), plus small dataset files. When the study was extended to five pairs (§5.6), a second pre-cache on 2026-06-15 added the two cross-family repositories, Mistral-7B-Instruct-v0.3 and Phi-4-mini-instruct; Mistral-7B is HF-gated, so its access conditions were accepted and the head-node HF token login re-run before prefetching. The downloads themselves are pure HTTP file transfers with negligible CPU and memory cost, and are therefore consistent with the head-node activities explicitly demonstrated in the TC1 user guide (page 7 to 9, where conda install and pip install are shown executing on the head node)."),

  H2("5.6 Run Plan"),
  PJ("With group_by=model, the framework emits one sbatch file per model alias, six for the original three pairs, extended to ten when the cross-family pairs were added. Each script runs the full configured benchmark suite for its model with the model loaded only once, exploiting the matrix runner's reuse_loaded_model=True default. (The initial matrix run covered the three core benchmarks; ARC-Challenge was added later as the second capability benchmark and run on the original six models via a parallel per-model job set, slurm/jobs_tc1_arc/, with the two cross-family pairs covering ARC inside their matrix jobs; see §6.4.1.) The original six scripts are:"),
  Bullet("qwen_2b_base__matrix.sbatch"),
  Bullet("qwen_2b_4bit__matrix.sbatch"),
  Bullet("qwen_4b_base__matrix.sbatch"),
  Bullet("qwen_4b_4bit__matrix.sbatch"),
  Bullet("llama_3_2_3b_base__matrix.sbatch"),
  Bullet("llama_3_2_3b_4bit__matrix.sbatch"),
  PJ("The cross-family extension (run on TC1 on 2026-06-15; results in §6.13) added four more matrix scripts of the same form, mistral_7b_{base,4bit}__matrix.sbatch and phi4_mini_{base,4bit}__matrix.sbatch, submitted pair-by-pair under the same scheduling discipline, each preceded by a five-sample smoke job to confirm the model loads with its native chat template (and, for Phi-4-mini, the eager attention backend). Their HarmBench generations were judge-scored by a dedicated sbatch (slurm/judge_validation_newpairs.sbatch) that scores only the new aliases, so the committed judge sidecars of the original six models were never reopened. The primary 512-token study (§6.16, D41) then re-ran the entire ten-model matrix at max_new_tokens = 512 via configs/tc1_512.yaml into the parallel results_512/ tree, judge-scored by slurm/judge_validation_512.sbatch (job 61524)."),
  PJ("Before submitting the full matrix, a single short smoke sbatch is submitted (five prompts, qwen_2b_base on HarmBench) to verify that the offline-cache path works end-to-end on a real compute node. Per the user guide's guidance to prefer sbatch over srun, this smoke verification is performed as a regular SLURM job rather than an interactive session. On 2026-05-27, smoke job 60975 completed successfully on TC1 in 33 seconds, produced a clean summary.json with runtime_device=cuda and malformed_rate=0.0, and therefore cleared the full six-job matrix for submission."),
  PJ("The first production matrix submission was made on 2026-05-27 with the Qwen 2B-class pair: job 60976 (qwen_2b_base__matrix) began running while job 60977 (qwen_2b_4bit__matrix) remained pending with reason QOSMaxGRESPerUser. This establishes the practical scheduling rule for the remainder of the study: submit only the current model pair and expect one job to run while the paired job waits; submit the next pair only after the current pair has cleared. Under this effective one-GPU concurrency, the six model jobs run serially or near-serially, still within the study plan because each job has an independent six-hour walltime allocation. Memory and CPU utilisation are recorded after each job using the seff and MyJobHistory commands to inform any subsequent right-sizing of the sbatch resource requests."),

  H2("5.7 Reproducibility Notes"),
  PJ("All runs share the same global seed (42), deterministic dataset shuffling, and deterministic decoding. The exact commit hash of the framework at the time of the final result run will be recorded in the final report and will accompany the result tables. Per-prompt records are persisted to raw.jsonl with full metadata. Two reproducibility scopes are distinguished. The study can be re-executed from scratch from any repository checkout (configs, seeds, prompts, and pipeline are committed): the five model repositories are pinned by Hugging Face commit hash, the revision field on every model entry in configs/tc1.yaml and configs/tc1_512.yaml, set to the exact snapshots used on TC1, so an online re-execution loads byte-identical weights rather than whatever revision happens to be current at fetch time; and every reported number can be replayed from the committed analysis artifacts and redacted score sidecars. The raw per-prompt records themselves are local-only evidence, gitignored and hash-pinned in results/raw_artifact_manifest.sha256, so recomputing the analysis directly from raw outputs requires the locally held evidence trees, not a fresh clone. One further provenance boundary: the run records carry the model alias, pair id, and quantized flag, but not the quantization method (NF4 versus INT8) or the resolved compute dtype as explicit fields; for the committed studies these are fixed by the study name and configuration, the loader guarantees the NF4 compute dtype equals the baseline's float16, and the pipeline persists both fields explicitly for future runs."),
];

// ------------------------------------------------------------
// Chapter 6: Results and Analysis
// ------------------------------------------------------------
const ch6 = [
  H1("Chapter 6: Results and Analysis"),

  PJ("All six original experimental runs completed successfully on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976 to 60981). Three matched pairs were evaluated: Qwen3-1.7B (jobs 60976/60977), Qwen3-4B (jobs 60978/60979), and Llama 3.2 3B-Instruct (jobs 60980/60981). A cross-family extension subsequently added two further matched pairs (Mistral-7B-Instruct-v0.3 [21] and Phi-4-mini-instruct [22]) run on the same cluster on 2026-06-15 (matrix jobs 61121/61122/61123/61125; HarmBench classifier job 61134), taking the study to five pairs / ten models / four families; their results are folded into Tables 6.1 to 6.3 and analysed in §6.13. Each job ran all three core benchmarks sequentially with the model loaded once; ARC-Challenge (the second capability benchmark) was added afterwards and run on the original six models in a separate per-model job set, with the two cross-family pairs covering ARC inside their matrix jobs (§6.4.1). The results are presented in Table 6.1 (per-pair raw metrics) and Table 6.2 (quantization deltas and interpretation labels). Detailed per-pair observations are in §6.6 (Qwen3-1.7B), §6.8 (Qwen3-4B), §6.10 (Llama-3.2-3B), and §6.13 (Mistral-7B and Phi-4-mini). The within-family scale analysis (RQ4) is in §6.2 and §6.9. The cross-family comparison (RQ5) is in §6.11, extended across four families in §6.13."),

  H2("6.1 Results Table"),
  PJ("Table 6.1 records the per-pair benchmark results for all five model pairs together with bootstrap 95% confidence intervals on the deltas. HarmBench Attack Success Rate is reported under the official HarmBench classifier (the primary scorer; see §6.1.1 and §6.12) at HarmBench's own 512-token generation budget, which is the primary configuration of this study; the shorter 128-token budget is retained as an explicit generation-length comparison in §6.16. The v2 refusal regex is shown beneath each pair as a secondary non-refusal-rate proxy. XSTest over-refusal and MMLU accuracy are reported at the same 512-token budget; the HarmBench classifier affects only the HarmBench ASR column. Confidence intervals are computed by a paired bootstrap [19] over the matched prompt set (2 000 resamples, seed 42); both members of every pair see the same prompts in the same order, so each bootstrap draw resamples prompt indices and recomputes the delta directly. A delta is marked statistically significant when its 95% CI excludes zero AND McNemar's exact paired test (§3.7) agrees at p < 0.05; at the corrected over-refusal scoring no contrast has the two criteria disagreeing, the previously borderline Qwen3-1.7B over-refusal decrease is now non-significant on both, CI [−0.048, 0.000] and McNemar p = 0.109 on 10 discordant prompts."),
  buildTable(
    ["Pair", "Metric (scorer)", "Baseline", "4-bit", "Δ (95% CI)", "Sig?"],
    CLAIMS.render.report_table_6_1,
    [1200, 2300, 1000, 1000, 2900, 660],
  ),
  P("Table 6.1  Complete per-pair results with paired bootstrap 95% CIs on the deltas. \"Sig?\" = the CI excludes zero and McNemar's exact paired test agrees (p < 0.05); at the corrected over-refusal scoring no contrast sits in the borderline margin where the CI excludes zero but the exact test does not (Qwen3-1.7B's over-refusal decrease is now non-significant on both, McNemar p = 0.109). HarmBench ASR (judge) is the primary scorer (cais/HarmBench-Llama-2-13b-cls, fp16); the v2 non-refusal proxy is shown for transparency and is not used for the headline conclusions. Judge CIs from `results_512/analysis/judge_agreement.json`; v2/MMLU/XSTest CIs from `results_512/analysis/pairwise_deltas.json`.", { size: 18 }),
  ...Figure("asr_forest.png", CLAIMS.render.asr_forest_caption),
  PJ("Under the primary (judge) scorer at the 512-token reference budget, the HarmBench deltas tell a uniform story: no pair's attack-success rate increases significantly, and the only ΔASR whose 95% CI excludes zero is Llama-3.2-3B's, at −0.040, a decrease, not an increase. The statistically significant deltas across Table 6.1 are therefore all capability or over-refusal effects: Qwen3-1.7B ΔMMLU (−0.090), the Llama-3.2-3B ΔASR decrease (−0.040), and one over-refusal decrease, Phi-4-mini (−0.048); Qwen3-1.7B's over-refusal decrease (−0.024) is non-significant (its bootstrap CI [−0.048, 0.000] now touches zero and McNemar's exact test gives p = 0.109; §3.7). The ARC capability axis (§6.4.1) adds significant losses on Qwen3-4B (−0.016) and Llama-3.2-3B (−0.032). Not one of the five pairs shows a significant harmful-compliance increase under NF4. The picture is sharper still after multiplicity correction (§6.5): under a Benjamini, Hochberg control over the family of primary contrasts, no HarmBench ASR contrast survives at all, and the three surviving effects are all capability or over-refusal changes. Every XSTest over-refusal delta is non-significant or points downward, the one significant delta (Phi, −0.048) is a decrease and no delta is borderline, so no pair becomes more over-refusing under quantization (RQ2). The demoted v2 non-refusal proxy again reports much higher absolute ASR and a different pattern (analysed in §6.12); the judge values are authoritative. The shorter 128-token budget, retained for direct comparison in §6.16, produced an apparent Qwen3-1.7B ASR increase of +0.055 that does not appear at the reference budget."),

  H3("6.1.1 Scorer revision history"),
  PJ("Scope under the judge-primary framing (D16). This section documents the refusal regex's own v1→v2 revision history, which bears on two things only: the XSTest over-refusal figures (produced by the v2 expanded refusal classifier, `ethical_benchmark/benchmarks/utils.py`, scorer version `v2_expanded_refusal_patterns_2026-05-28`) and the secondary HarmBench non-refusal proxy. HarmBench Attack Success Rate throughout this report is the official HarmBench classifier value (the primary scorer; §6.12), not the regex; MMLU and ARC use deterministic exact match. With that scope established: the v1 classifier (pre-2026-05-28) used a narrow set of 14 refusal patterns that missed the most common modern refusal templates, including \"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", and curly-apostrophe variants used by Qwen3 output. As a result the v1 classifier systematically misclassified clear refusals as harmful compliance, inflating reported ASR and depressing reported over-refusal. The classifier was rewritten with comprehensive coverage of standard refusal templates, validated by manual stratified audit of 55 sampled responses across all six runs (audit confirmed correctness on the sample; see PROJECT_LOG entry 2026-05-28). The TC1-original `raw.jsonl` and `summary.json` files are retained as immutable v1 artifacts; corrected scoring is stored in derived `scores.v2.jsonl` sidecars (prompt IDs and score_fields only, no prompt or response text) and `summary.v2.json` files. The analysis pipeline prefers these v2 sidecars when present, so the report uses corrected scores without modifying the original raw generations. The auxiliary artifacts `results/analysis/rescore_diagnostics_*.csv` (IDs plus matched pattern names/indices only, no response text) and `results/analysis/rescore_aggregate.json` document the per-record reclassification for full auditability."),
  PJ("Impact of the rescore on the v2 numbers. Within the regex scorer, three pair-level deltas changed materially from v1 to v2: Qwen 1.7B ΔASR moved from −0.120 (v1) to −0.025 (v2), where the v1 \"significant decrease\" was largely an artefact of unrecognised refusals; Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2); Llama 3.2 3B ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). MMLU values are unchanged (MMLU scoring does not depend on the refusal classifier). This history explains why the v2 regex is retained only as a secondary proxy: even after the v1→v2 correction, the regex measures \"non-refusal rate\", not genuine harmful compliance. The subsequent judge validation (§6.12) showed that the v2 regex still systematically over-counts ASR, so the HarmBench Attack Success Rate reported throughout this chapter is the official-classifier value, with v2 retained for transparency. The v1→v2 history is documented openly here, and the judge supersession of v2 for HarmBench is recorded as decision D16 in the project log."),

  H2("6.2 Within-Family Scale Analysis (RQ4)"),
  PJ("RQ4 asks whether the magnitude of the quantization effect differs between two-billion and four-billion parameter Qwen models. Both pairs have now completed. Table 6.2 summarises the quantization deltas and interpretation labels for all five pairs, enabling both the within-family scale comparison (RQ4, the two Qwen pairs) and the cross-family comparison (RQ5)."),
  buildTable(
    ["Metric", "qwen_2b", "qwen_4b", "llama_3.2_3b", "mistral_7b", "phi4_mini"],
    CLAIMS.render.report_table_6_2,
    [2200, 1240, 1240, 1480, 1320, 1320],
  ),
  P("Table 6.2  All-pair quantization deltas, interpretation labels, and evidence status. Δ = 4-bit − baseline. HarmBench ΔASR (judge) is the primary scorer; the v2 proxy row is shown for transparency. ★ marks deltas significant under both criteria (paired-bootstrap 95% CI excludes zero and McNemar's exact p < 0.05); at the corrected over-refusal scoring Qwen3-1.7B's Δover-refusal is non-significant on both criteria (McNemar p = 0.109; §3.7) and no delta is borderline. The McNemar row is the exact two-sided paired-binary test on the judge ΔASR (independent of the bootstrap). Under the judge at the 512-token reference budget, the ★ deltas are Llama-3.2-3B ΔASR (−0.040, a decrease; McNemar p = 0.021, the only significant ΔASR, and no pair shows a significant increase), Qwen3-1.7B ΔMMLU (−0.090), and Phi-4-mini's over-refusal decrease (−0.048). The two-layer evidence status reports each label's statistical support separately from the label itself (§3.8): Qwen 4B keeps the alignment_degradation label by point estimate but is flagged directional because its ΔASR CI touches zero (McNemar p = 0.096); it is not a statistically confirmed regression. Labels are judge-primary (D16).", { size: 18 }),
  PJ(CLAIMS.render.bh_survivor_sentence),
  PJ("Table 6.2 shows a large scale effect on capability. Under the same NF4 quantization procedure, the 1.7B model experiences large capability degradation (MMLU drops 9.0 percentage points, relative decline of 14.0%, CI excludes zero) while the 4B model is essentially unaffected, with MMLU moving by only 0.3 percentage points, well within sampling noise. The MMLU delta ratio (9.0 pp versus 0.3 pp) is approximately 30:1; on this benchmark, scale is the single largest predictor of capability preservation under quantization observed in this study. This 30:1 ratio is a lenient-parser figure: on ARC under the same parser the 4B pair loses marginally more than the 1.7B (−1.6 pp vs −0.9 pp), which at first suggests the scale gap is MMLU-specific, but §6.4.1 shows that lenient reading is a parsing artefact, since under a strict answer-parser the 1.7B pair loses far more than the 4B on ARC too (−0.343 vs −0.017). Scale-as-predictor-of-capability-preservation is therefore parser-dependent in magnitude and is read as an observed within-family contrast rather than a benchmark-independent law."),
  PJ("The full-precision baselines reveal that within the Qwen family the larger model is both more capable and more refusal-calibrated. Under the primary (judge) scorer, Qwen3-1.7B's baseline HarmBench ASR is 0.255 while Qwen3-4B's is 0.115, so scaling from 1.7B to 4B reduces genuine baseline harmful compliance by 14.0 percentage points. MMLU accuracy simultaneously rises from 0.643 to 0.747 (+10.4 pp), confirming the capability gain is genuine, and XSTest over-refusal falls from 0.052 to 0.028, indicating fewer false refusals on benign prompts. The 4B baseline therefore dominates the 1.7B baseline on every dimension. (Under the demoted v2 non-refusal proxy the baseline ASR gap looks even larger, 0.595 versus 0.235, but that reflects the proxy's over-counting; the genuine harmful-compliance gap is the 14.0 pp judge figure, see §6.12.) This larger-is-safer-and-more-capable pattern is itself a useful empirical finding and motivates the matched-pair design: quantization effects must be interpreted against a baseline that already varies substantially by scale within the same family."),
  PJ("The most important comparison is on the post-quantization side, read under the primary (judge) scorer at the reference budget. The 1.7B pair's harmful-compliance rate does not move (judge ΔASR 0.000, McNemar p = 1.000) while its capability drops sharply (ΔMMLU −9.0 pp, CI excludes zero): the degradation is capability-only, and the broad_degradation label the pair receives is driven entirely by that capability loss, not by any safety change. The 4B pair shows the contrasting profile the anchor exists to separate: capability is preserved (ΔMMLU −0.3 pp, within tolerance) while the judge ΔASR is positive but not significant (+4.0 pp, CI [0.000, +0.080]; McNemar p = 0.096), alignment_degradation at the point-estimate level, carried as directional. Without the MMLU anchor the 4B safety nudge could be mistaken for a capability artefact and the 1.7B capability loss could be read as a safety story; with the anchor, each pair's two axes are interpreted jointly. This is the methodological payoff of the capability-anchored design. (The demoted v2 proxy paints a different and misleading picture here, a −2.5 pp 1.7B ΔASR and a +7.0 pp 4B ΔASR; §6.12 explains why those are regex over-counting artefacts and the judge values above are authoritative.)"),

  H2("6.3 Cross-Family Replication (RQ5)"),
  PJ("RQ5 asks whether the Qwen pattern replicates qualitatively in Llama 3.2 3B. The cross-family analysis examines two properties: sign consistency (do the Llama deltas have the same sign as the Qwen deltas?) and approximate magnitude (are the Llama deltas similar in scale?). The full cross-family comparison is in §6.11; this section provides a high-level orientation."),
  PJ("The headline finding for RQ5 is that, at the 512-token reference budget under the official classifier, NF4 quantization produces no significant harmful-compliance increase in any pair, and the only individually-significant ΔASR (Llama-3.2-3B, −0.040) is a decrease. The Qwen3-1.7B pair shows ΔASR at exactly zero paired with a large, significant capability loss (ΔMMLU −0.090); the Qwen3-4B pair shows a directional, non-significant +0.040 at preserved capability; the Llama-3.2-3B pair shows a small significant ASR decrease alongside a significant ARC loss (§6.4.1). The capability dimension is directionally consistent (all five pairs have negative ΔMMLU; the Qwen3-1.7B loss is the only MMLU-significant one, with further ARC-significant losses on Qwen3-4B and Llama-3.2-3B). On over-refusal, the only significant delta (Phi, −0.048) is a decrease and Qwen3-1.7B's decrease is now non-significant (McNemar p = 0.109); no delta is borderline. The composite picture is that NF4 quantization does not produce a significant or uniform safety regression in any family or at any scale; the robust, multiplicity-surviving signals are all capability losses. See §6.11 for the full discussion and §6.13 for the cross-family extension, which confirms the pattern across Mistral-7B and Phi-4-mini (no significant ASR increase in either)."),

  H2("6.4 Capability Anchoring"),
  ...Figure("capability_anchor.png", "The capability-anchored safety space. Each pair is placed by its capability delta (ΔMMLU, x-axis) and its harmful-compliance delta (judge ΔASR, y-axis); dashed lines mark the interpretation thresholds and the shaded quadrants name the diagnostic labels. Bars are paired-bootstrap 95% CIs. The anchor is what separates the broad_degradation pairs (top-left) from a capability-preserving alignment nudge (Qwen3-4B) and from Mistral-7B's capability-preserving safety improvement (bottom-right)."),
  PJ("Capability anchoring is the core methodological contribution of the interpretation framework. Each pair's MMLU delta is used as a covariate when interpreting its HarmBench and XSTest deltas. Six canonical outcomes are tracked, organised symmetrically around both the safety and over-refusal axes:"),
  Bullet("Alignment degradation: MMLU is preserved, but HarmBench ASR rises beyond threshold; the model becomes more compliant with harmful prompts without a corresponding capability change. A direct safety regression."),
  Bullet("Alignment improvement: MMLU is preserved, but HarmBench ASR falls beyond threshold; quantization reduces harmful compliance without measurable capability cost. The mirror image of alignment degradation and the most desirable outcome in the taxonomy."),
  Bullet("Capability-collapse masquerading as safety: MMLU drops noticeably, and HarmBench ASR also drops; the apparent safety improvement is attributed to reduced instruction-following capacity rather than a genuine alignment change."),
  Bullet("Robust preservation: all three deltas are small in magnitude; quantization is effectively neutral across safety and capability dimensions."),
  Bullet("Over-refusal regression: HarmBench ASR and MMLU are held within tolerance, but the model refuses materially more benign prompts (ΔOR beyond threshold); a benign-refusal regression named on its own over-refusal axis rather than folded into broad degradation (Table 3.4, §3.8)."),
  Bullet("Broad degradation: fallback. Typically signals over-refusal moving outside tolerance, or ASR within tolerance but capability dropping. Used when no specific pattern above matches."),
  PJ("The five pairs in this study illustrate why the capability anchor is necessary, especially at the reference budget where none of the harmful-compliance deltas is a significant increase. For the qwen_2b pair (Qwen 1.7B), MMLU fell significantly (−9.0 pp, −14.0% relative; CI [−0.140, −0.043], significant) while the judge HarmBench ASR did not move at all (ΔASR 0.000, CI [−0.055, +0.055]; McNemar p = 1.000). The harm axis is flat, so the broad_degradation label the pair receives is driven entirely by the significant capability loss, not by any safety change: the anchor is what makes clear this is a capability-only degradation rather than a safety regression. For the qwen_4b pair, MMLU is preserved (−0.3 pp, within tolerance, not significant) while the judge ΔASR is positive but not significant (+4.0 pp, CI [0.000, +0.080]; McNemar p = 0.096). Capability held with a directional, unconfirmed safety worsening: the alignment_degradation pattern at the point-estimate level, carried with a directional evidence status. The anchor is exactly what licenses reading the 4B nudge as a tentative alignment direction rather than a capability artefact: capability did not move. For the llama_3_2_3b pair, capability drops on both anchors (ΔMMLU −3.7 pp, not significant; ΔARC −3.2 pp, significant) while the judge ΔASR is a significant decrease (−4.0 pp, CI [−0.075, −0.010]; McNemar p = 0.021): harm falls and capability falls together, so the label is capability_collapse_masquerading_as_safety, the apparent safety gain attributed to reduced instruction-following rather than genuine alignment. The contrast is decisive: without the anchor the 4B directional nudge could not be separated from a capability artefact, and Llama's ASR decrease would be misread as a safety improvement rather than a symptom of capability loss. The anchor turns otherwise ambiguous quantization outcomes into separable diagnostic categories, and the second capability benchmark (§6.4.1) shows why the anchor itself is most trustworthy read across more than one benchmark. (Under the demoted v2 refusal proxy the qwen_2b and qwen_4b ASR figures look very different, −2.5 pp and +7.0 pp respectively; §6.12 explains why those are regex artefacts and the judge values above are authoritative.)"),

  H3("6.4.1 Capability robustness check: the ARC-Challenge second benchmark"),
  PJ("Because the capability anchor rests on a single MMLU subset, a structurally different second capability benchmark, ARC-Challenge (allenai/ai2_arc; 1,172 reasoning-oriented science questions; identical exact-match scoring), was run on all ten models to test whether the MMLU-based capability claims replicate (results_512/analysis/pairwise_deltas, benchmark = arc; same 2,000-resample paired-bootstrap CI as MMLU). The direction is largely robust: all five pairs lose on MMLU, and four of five lose on ARC, the exception is Mistral-7B, whose ARC moves +0.9 pp (non-significant) in the improving direction. The magnitude and significance, however, diverge informatively. Qwen 1.7B, which loses a large, significant 9.0 pp on MMLU, loses only 0.9 pp on ARC (not significant) under the lenient parser. That lenient reading is not evidence that the 1.7B degradation is genuinely MMLU-specific: 52.3% of its 4-bit ARC items fall to the lenient fallback (versus 2.5% at fp16), so the parser is salvaging format-broken answers and masking the ARC loss; under a strict parser ARC falls −0.343 and MMLU −0.293, comparable large losses (the answer-format truncation that partly explained the MMLU figure at 128 tokens is separately resolved at 512, Chapter 8). Each benchmark's loss is therefore a lenient/strict bracket, and the capability half of the 1.7B degradation is milder than MMLU alone implies only at the lenient end. Qwen 4B shows the opposite pattern: flat on MMLU (−0.3 pp, n.s.) but a small, significant −1.6 pp on ARC, so its capability is less fully preserved than MMLU suggested. Llama 3B is the most consistent pair, losing on both, and its ARC loss (−3.2 pp) is the significant one that anchors its capability_collapse_masquerading_as_safety label (its MMLU dip, −3.7 pp, is not significant). Two implications follow. First, RQ3 (NF4 degrades capability) is supported with per-pair nuance rather than uniformly: on MMLU all five point estimates are negative, on ARC four of five are (Mistral's +0.9 pp non-significant improvement is the exception), and three pairs carry an individually significant capability loss on at least one anchor (Qwen 1.7B on MMLU; Qwen 4B and Llama on ARC), Mistral and Phi show directionally consistent but individually non-significant capability changes. The capability cost is therefore robust in aggregate (it supplies two of the three FDR survivors, Qwen 1.7B MMLU and Llama ARC; the third is Phi's over-refusal decrease, §6.5.1) without being uniformly significant per pair. Second, the dramatic within-Qwen scale gap under MMLU (the ≈30:1 ratio, §6.2/RQ4) does not replicate under ARC's lenient parser, which finds the 4B losing marginally more than the 1.7B; under a strict parser it reverses again, the 1.7B pair's −0.343 dwarfing the 4B pair's −0.017, because the lenient parser was masking the 1.7B ARC collapse (above). The strong 'smaller model is far more capability-sensitive' claim is therefore parser-dependent rather than MMLU-specific and is hedged accordingly. Interpretation labels remain MMLU-anchored; ARC is reported here as a corroborating capability axis, and a formal composite-capability rule is left to future work. The two cross-family pairs (§6.13) extend the ARC axis to ten models: Mistral-7B is essentially flat on ARC (+0.9 pp, not significant) consistent with its preserved MMLU, and Phi-4-mini loses 1.5 pp (not significant) alongside a 2.7 pp MMLU dip, both new families showing the same direction-consistent, modest capability story. Crucially, at the 512-token reference budget the two ARC losses that reach significance (Qwen 4B −1.6 pp, Llama −3.2 pp) are capability effects; no HarmBench ASR increase reaches significance on any pair, so ARC reinforces the reading that capability degradation, not harmful-compliance change, is the robust cost of NF4."),

  H2("6.5 Statistical Caveats"),
  PJ("Five statistical limitations should be borne in mind when interpreting the results. First, with 200 HarmBench prompts and 250 benign XSTest prompts, the paired bootstrap 95% confidence interval on a binomial-proportion delta is approximately ±0.05 (Table 6.1), so small deltas may not be statistically distinguishable from zero. Under the primary (judge) scorer at the 512-token reference budget, the only HarmBench ΔASR whose CI excludes zero is Llama-3.2-3B's, and it is a decrease (−0.040, CI [−0.075, −0.010]; McNemar p = 0.021); every other pair is within noise: Qwen 1.7B (0.000, CI [−0.055, +0.055]; McNemar p = 1.000), Qwen 4B (+0.040, CI [0.000, +0.080]; McNemar p = 0.096), Mistral (−0.020, CI [−0.080, +0.040]) and Phi (+0.020, CI [−0.015, +0.055]). No pair shows a significant harmful-compliance increase, so the only individually-significant NF4 ΔASR is a safety-improving decrease; the INT8 precision point in §6.15 shows no significant ASR move at all at this budget either. Second, decoding is deterministic at temperature 0.0, so the only source of variance reflected in the bootstrap intervals is prompt sampling, with no within-condition stochastic variance. Because that greedy delta conditions on a single decode, a multi-seed (T = 0.7, top-p 0.8) sensitivity arm was run to estimate generation-level variance (§6.6.1; three pairs, Qwen 1.7B, Qwen 4B, Llama). For Qwen 1.7B the greedy 0.000 sits inside the seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant), so the arm corroborates the null rather than an effect. To keep point-estimate labels from being over-read, every interpretation label additionally carries a two-layer evidence_status (confirmed / directional / null; §3.8): Qwen 1.7B is confirmed (its confirmed axis is the significant MMLU loss, not a safety change), Llama is directional (its ASR decrease and ARC loss are individually significant, but the label's MMLU capability anchor, −0.037, is not, the artifact records evidence_status = directional and that is the claim carried here), Qwen 4B is directional, and the cross-family pairs (§6.13) are directional (Mistral) and null (Phi). Third, the MMLU subset comprises 300 questions pooled across six subjects; subject counts are uneven (25 to 94 per subject: business_ethics 25, college_biology 33, human_aging 44, clinical_knowledge 48, high_school_world_history 56, high_school_macroeconomics 94). Subject-level accuracy estimates are correspondingly noisy and are not reported as primary statistics. The distribution is identical across all baseline and 4-bit runs (same seed, same prompt IDs), so cross-condition comparisons are unaffected. A further MMLU-specific construct caveat concerns answer parsing: the option-letter parser falls back to a lenient \"first in-range capital letter anywhere\" scan when a response neither leads with the option letter nor states an explicit \"answer is X\", and this fallback fires far more often on the degraded 4-bit outputs; for qwen_2b_4bit 48.7% of items are scored by the fallback versus 3.3% for its fp16 twin, and the fallback's own accuracy on that pair (46%) is well below the leading-letter rule's (75%). Part of the −0.090 MMLU gap for that pair therefore reflects a shift in answer format that the lenient parser scores less reliably, not only a loss of underlying knowledge. ARC-Challenge is subject to the same asymmetry even more strongly for this pair: 52.3% of the 4-bit member's ARC items fall to the lenient fallback versus 2.5% for its fp16 twin, so ARC's near-zero primary delta (−0.009, not significant) is the lenient parser salvaging those format-broken answers (fallback accuracy 66% on that pair), not evidence that ARC is immune to the shift. Under a strict parser (leading-letter or explicit \"answer is X\" declaration only, an unparsable response scored incorrect) ARC falls −0.343 (CI [−0.375, −0.311]) and MMLU falls −0.293 (CI [−0.350, −0.237]), comparable large losses on both benchmarks. All four capability point estimates for this pair are negative, three of them individually significant, with ARC under the lenient parser (−0.009) within noise of zero, so the direction of the loss is consistent across both benchmarks and both parsers while its magnitude is protocol-dependent. Each benchmark's loss is therefore honestly a bracket whose lenient end tracks answer-knowledge (largely preserved) and whose strict end tracks format-compliant answering (which collapses), not a single figure (results_512/analysis/parser_strict_sensitivity.json); answer-format parsing is thus a third measurement-dependence axis in the study, alongside the harm scorer (§6.12) and the over-refusal scorer (§6.12, Result 6). Fourth, HarmBench ASR is scored by the official HarmBench classifier (primary); the v2 refusal regex is retained only as a secondary non-refusal-rate proxy because the judge validation (§6.12) showed it materially over-counts ASR. XSTest over-refusal is scored by the v2 regex (the HarmBench classifier does not cover the over-refusal question); this includes the single over-refusal contrast that survives the multiplicity correction (Phi-4-mini's −0.048 decrease, §6.5.1), which therefore inherits the same weaker-construct status as the regex and should be read as a reproducible non-refusal-rate change rather than a classifier-validated over-refusal shift. An independent three-class refusal judge (§6.12, Result 6) makes this concrete: it does not reproduce the −0.048 decrease (judge ΔOR +0.016 strict / −0.004 broad, both non-significant) and agrees with the regex only poorly-to-moderately on benign over-refusal (Cohen κ from −0.01 to 0.50 under the strict mapping, and to 0.54 under the broad one), so this contrast is scorer-dependent. The pre-registered human audit (§6.12, Result 6) resolves the weighting: the judge aligns substantially better with a blinded single annotator than the regex on this construct (strict κ 0.485 versus −0.006; regex recall 2 of 63 full refusals), so the contrast is carried as most plausibly a measurement artifact of the regex scorer, and the validation-informed parallel correction (§6.5.1) states the over-refusal conclusion without it. Fifth, no per-comparison flag should be read in isolation: across the five pairs the study reports roughly twenty primary delta confidence intervals (HarmBench ASR, MMLU, ARC, over-refusal) plus the per-category breakdown and the INT8 precision sweep (§6.15), so the nominal significance flags are corrected explicitly in §6.5.1. The multiplicity-robust signals there are all capability or over-refusal losses; not one HarmBench ASR contrast survives the correction, which is consistent with the null-safety, capability-driven reading of §6.14. Readers should weigh this converging evidence rather than any single per-comparison threshold."),

  H3("6.5.1 Multiple-comparison correction and statistical power"),
  PJ("This section reports the corrected view rather than only acknowledging the multiplicity problem. Applying a Benjamini-Hochberg false-discovery-rate correction (q < 0.05) to the family of twenty primary NF4-vs-fp16 contrasts (five pairs × {HarmBench ASR (judge), MMLU, ARC, over-refusal}, with an exact McNemar p-value computed for every paired-binary contrast, results_512/analysis/multiple_comparisons.json), exactly three contrasts survive, and all three are capability or over-refusal losses: Qwen3-1.7B MMLU (−0.090, q = 0.008), Llama-3B ARC (−0.032, q = 0.008) and Phi-4-mini XSTest over-refusal (−0.048, q = 0.012). The two capability survivors rest on deterministic exact-match scoring (MMLU and ARC parse an option letter; no refusal scorer is involved), but the over-refusal survivor rests on the demoted refusal regex and is scorer-dependent: an independent three-class refusal judge does not reproduce Phi's −0.048 decrease (judge ΔOR +0.016 strict / −0.004 broad, both non-significant; §6.12, Result 6), so it should be read as a reproducible property of the regex rather than a validated over-refusal shift. Not one HarmBench ASR contrast survives the correction at the 512-token reference budget, including the individually-significant Llama ΔASR decrease, which becomes nominal once multiplicity is accounted for. Capability degradation is therefore the multiplicity-robust signal of four-bit NF4 in this study; harmful-compliance change is not robust to correction on any pair. This is consistent with, and strengthens, the capability-driven mechanism reading (§6.14): the robust, family-wise-surviving effects are all capability or benign-refusal losses, and there is no surviving evidence of a harmful-compliance increase under NF4."),
  PJ("Because the over-refusal member of every contrast above is regex-scored, and the §6.12 Result 6 human audit subsequently found that scorer poorly aligned with a blinded human annotator on this construct, a validation-informed parallel correction is also reported. It is clearly labelled post-hoc, added after both scorers' results were known, with its composition fixed in a dated pre-analysis note before it was computed (docs/VALIDATION_INFORMED_FAMILY_NOTE.md). It recomputes the identical twenty-contrast family with a single change: the five over-refusal contrasts are scored by the independent three-class judge under the strict mapping, the replication definition pre-registered in the T35 protocol; the other fifteen p-values are reused verbatim, and the same Benjamini-Hochberg procedure is applied at the same threshold. " + CLAIMS.render.dual_family_sentence + " The parallel family is purely deflationary (no contrast becomes significant that was not already), and the original regex-scored family above remains the pre-specified family of record for traceability. The substantive RQ2 conclusion is carried in the scorer-invariant form: no scorer finds a statistically significant over-refusal increase in any pair (point estimates do move upward in places, the judge-strict Qwen3-1.7B +0.040 being the largest, McNemar p = 0.087, but none reaches significance under either scorer). Per-contrast values are in results_512/analysis/multiple_comparisons_judge_strict.json."),
  PJ("A power analysis quantifies why so few effects reach significance. For a two-sided α = 0.05, 80%-power McNemar test at the observed HarmBench discordant rates (median discordant rate ≈ 0.09) and n = 200, the minimum detectable ΔASR is roughly ±0.06 (representative MDE ≈ 0.059); no NF4 ASR increase sits above this detection floor on any pair, so the predominance of nulls on the safety axis reflects a detection floor as much as a substantive absence of effect. The study is therefore underpowered for the small safety effects it might have detected, a limitation that a larger prompt set would address (Chapter 8); it does, however, have enough power to resolve the capability losses, which are both larger and multiplicity-robust. Figures and per-contrast values are in results_512/analysis/multiple_comparisons.{json,csv}."),

  H2("6.6 Observations: Qwen3-1.7B Pair (pair_id qwen_2b)"),
  PJ("Under the official HarmBench classifier at the 512-token reference budget, the Qwen 1.7B pair receives the broad_degradation label (evidence status: confirmed), but at this budget the label is driven entirely by capability loss, not by any safety change. HarmBench ASR is flat (0.255 → 0.255, ΔASR = 0.000, 95% CI [−0.055, +0.055], not significant; McNemar's exact p = 1.000), while MMLU accuracy falls significantly (0.643 → 0.553, ΔMMLU = −0.090, significant). Quantization makes the smallest model markedly less capable without making it more willing to produce genuinely harmful content. Because the harm axis is flat while capability drops beyond tolerance, the label is broad_degradation only in the sense that the significant capability collapse triggers it; there is no accompanying harmful-compliance increase and this pair must not be read as a safety regression at the reference budget."),
  PJ("This null on the harm axis is unambiguous under the paired-binary test. Because HarmBench is paired, McNemar's exact test [20] (the correct paired-binary test; §3.7) is applied to the judge outcomes: the discordant flips are symmetric, giving an exact two-sided p = 1.000, and the net delta is exactly zero. The paired bootstrap (CI [−0.055, +0.055] spans zero) and McNemar's exact test therefore agree there is no detectable harmful-compliance change. The prominent +0.055 increase reported for this pair at the 128-token budget (§6.16) was a truncation artefact: at 128 tokens roughly 60% of all responses were provably cut off before completion (a direct prefix test; results_512/analysis/genlen_robustness.json), so the classifier was scoring incomplete generations, and where the cut fell could shift a label in either direction; extending the budget to 512 tokens dissolves the effect to zero. No HarmBench ASR contrast in the NF4 family survives the BH-FDR correction (§6.5), and this pair is not the exception."),
  PJ("The secondary v2 proxy tells a consistent story: it reports a small negative delta (ΔASR ≈ −0.025) at this budget, and the official classifier confirms there is no genuine harmful-compliance increase to recover from it. This remains a clean illustration of why the official classifier, not refusal-counting, is the appropriate scorer (see §6.12), but here both scorers agree the harm axis does not move under quantization for this pair."),
  PJ("The XSTest over-refusal rate declined (0.052 → 0.028, ΔOR = −0.024), a non-significant effect: the bootstrap CI [−0.048, 0.000] now touches zero, and McNemar's exact test on the 10 discordant prompts gives p = 0.109, so under the study's significance criterion (§3.7) the decrease is not confirmed. The 4-bit model refuses benign prompts slightly less often; combined with a flat harm axis, this is a mild over-refusal decrease rather than any harmful-compliance change. Together the three dimensions describe a model that, under NF4 compression, produces genuinely harmful completions at the same rate, refuses benign prompts slightly less often, and answers markedly fewer factual questions correctly. The pair's entire quantization signal is on the capability axis."),
  PJ("The MMLU decline is the load-bearing effect for this pair: accuracy falls from 0.643 to 0.553 (ΔMMLU = −0.090, significant), and this is one of only three contrasts in the whole NF4 family to survive the BH-FDR correction (q = 0.008; §6.5). The capability loss is therefore the robust, multiplicity-surviving cost of NF4 on the smallest model, and it is what assigns the broad_degradation label."),
  PJ("The Qwen 1.7B pair is thus a capability-only degradation at the reference budget: a 4-bit-quantized small model that is significantly less capable than its full-precision counterpart, with no measurable change in harmful compliance. The capability anchor is what makes this unambiguous, the significant ΔMMLU establishes a real cost, while the null judge ΔASR (0.000, McNemar p = 1.000, no FDR survivor) rules out any safety regression. This is the study's clearest single case of NF4 eroding competence rather than guardrails.")
,
  H3("6.6.1 Multi-seed decoding sensitivity (W1 robustness check)"),
  ...Figure("multiseed.png", "Qwen3-1.7B judge ΔASR under decoding variation at the 512-token reference budget: the greedy point estimate (0.000, red diamond) versus the multi-seed mean ± sd at temperature 0.7 (blue, five seeds), mean +0.013 with the greedy 0.000 sitting inside the seed range. Stochastic decoding corroborates the null. Source: results_512/analysis/sensitivity_multiseed.json."),
  PJ("Because the primary study uses greedy decoding (T = 0.0), its bootstrap and McNemar intervals capture prompt-sampling variance only, not generation variance, the one Priority-1 weakness (W1) that cannot be resolved by reanalysis of the existing artifacts. To estimate how stable the ΔASR result is under realistic sampling, three pairs (Qwen 1.7B, Qwen 4B, and Llama, not Mistral or Phi) were re-run on HarmBench at a single fixed stochastic setting (temperature 0.7, top-p 0.8, Qwen3's published non-thinking recommendation, applied uniformly to all three pairs for cross-pair comparability rather than per-publisher tuning; Meta's Llama guidance is T = 0.6 / top-p 0.9) across five independent seeds, with both pair members scored by the same official HarmBench classifier used for the primary result. Within each seed, baseline and 4-bit share the identical temperature and seed, so every per-seed ΔASR is a clean matched-pair contrast; the arm is a separate robustness layer and is never cross-compared with the greedy main results."),
  PJ("For Qwen 1.7B the per-seed judge ΔASR is 0.000, +0.010, +0.020, 0.000, +0.035 (mean +0.013, range [0.000, +0.035]), and no seed is individually significant (0/5). One directional nuance is disclosed rather than glossed: all five seed deltas are non-negative, a sign-consistency compatible with a small positive effect below the study's minimum detectable effect (≈0.06, §6.5), the claim established here is 'no detectable change at n = 200', not a proven zero. Critically, the greedy point estimate (0.000) sits inside the seed range, so stochastic decoding corroborates rather than contradicts the null: there is no effect at the reference budget for sampling to attenuate. This is the reverse of the 128-token picture (§6.16), where multi-seed decoding attenuated a real greedy +0.055; at 512 tokens there is nothing to attenuate. The other two covered pairs behave consistently with their primary deltas: Qwen 4B has mean +0.029 (1 of 5 seeds individually significant under McNemar's exact test, a directional non-null), and Llama has mean −0.024 with no positive seed in five (2 of 5 seeds individually significant, both decreases), matching its significant safety-improving primary decrease. Across all three pairs no pair is robustly positive. Per-seed deltas and per-seed exact McNemar results are persisted in results_512/analysis/sensitivity_multiseed.json (judge_delta.per_seed and n_seeds_significant_p05), so every per-seed claim in this section is asserted against a committed artifact; Mistral and Phi remain greedy-only, so a full-matrix stochastic estimate is still outstanding."),

  H3("6.6.2 Per-category harmful-compliance profile"),
  ...Figure("category_asr.png", "Per-category HarmBench attack success rate for Qwen3-1.7B at fp16 versus NF4 (official classifier, 512-token reference budget). The aggregate ΔASR is 0.000, so per-category rates are descriptive with no net change across the mix; there is no broad-based increase to attribute. Per-category counts are small (19 to 58 prompts), so the rates are descriptive. Source: results_512/analysis/harmbench_category_breakdown.json."),
  PJ("Because the aggregate Qwen 1.7B ΔASR is exactly 0.000 at the reference budget, there is no net harmful-compliance increase to decompose by harm type. The judge ASR was nonetheless decomposed by HarmBench semantic category (results_512/analysis/harmbench_category_breakdown.{json,csv}, produced by scripts/harmbench_category_breakdown.py from the saved generations and judge sidecars, no new inference) to confirm the null is not masking a large offsetting swing between categories. At 512 tokens the per-category deltas go in both directions and roughly cancel: four categories rise (misinformation/disinformation +0.18, harassment/bullying +0.11, illegal +0.03, and the generic harmful category +0.05) while two fall (cybercrime/intrusion −0.18 and chemical/biological −0.14), leaving the aggregate flat; no single category drives a systematic increase. The broad-based five-of-six-category rise reported for this pair previously was a feature of the 128-token run: it was an artefact of truncation (§6.16), not a stable property of the model, and it disappears alongside the aggregate +0.055 once responses are allowed to run to the reference budget. Per-category counts are small (19 to 58 prompts each), so these rates are descriptive and exploratory, reported to characterise the composition of the null rather than as individually significance-tested claims. For contrast, the Qwen 4B pair shows a small directional increase and the Llama pair a significant decrease, consistent with their aggregate deltas."),

  H2("6.7 Preliminary Scale Observations: Baseline Comparison"),
  PJ("The Qwen3-1.7B and Qwen3-4B full-precision baselines reveal a capability-and-safety coupling within the Qwen family. At full precision (HarmBench under the official classifier at the 512-token reference budget), scaling from 1.7B to 4B parameters produces: HarmBench ASR 0.255 → 0.115 (−14.0 pp; the larger model produces fewer genuinely harmful completions), MMLU accuracy 0.643 → 0.747 (+10.4 pp; substantially more capable), and XSTest over-refusal 0.052 → 0.028 (−2.4 pp; fewer false refusals on benign prompts). Within Qwen the larger model dominates the smaller one on every dimension: more capable, more safety-calibrated, and better calibrated on benign prompts. (Under the v2 non-refusal proxy the baseline ASR gap looks larger because the proxy over-counts; the genuine harmful-compliance gap is the 14.0 pp classifier figure.) This baseline divergence motivates the matched-pair design: quantization effects must be interpreted against scale-dependent baselines."),

  H2("6.8 Observations: Qwen3-4B Pair (pair_id qwen_4b)"),
  PJ("Under the official classifier at the 512-token reference budget, the Qwen 4B pair receives the alignment_degradation label by point estimate with evidence status directional: the effect is suggestive rather than statistically confirmed. HarmBench ASR rises from 0.115 to 0.155 (ΔASR = +0.040, 95% CI [0.000, +0.080], not significant, the lower bound touches zero) while capability is preserved (ΔMMLU = −0.003, CI [−0.040, +0.033], not significant). McNemar's exact paired test agrees the move is not significant (exact p = 0.096). Because capability is preserved and the ASR point estimate exceeds the harm tolerance, the rule assigns alignment_degradation; but both the bootstrap CI and McNemar's test include the null, so the two-layer scheme flags this as directional: a suggestive safety worsening, not a confirmed one. The one significant capability signal on this pair is on the second benchmark: ARC accuracy falls significantly (ΔARC = −0.016, §6.4.1), so the 4B pair is not perfectly capability-robust even though its MMLU is flat."),
  PJ("This is a notable demotion from the secondary v2 proxy, under which Qwen 4B reports a much larger non-refusal-rate delta (proxy ΔASR = +0.070). The judge validation shows that the proxy over-counts by refusal-counting: the regex marks many non-refusals as harmful, inflating both the level and the delta. On the genuine harmful-compliance measure, the 4B effect is real in direction but does not reach significance at n = 200, and no HarmBench ASR contrast in the study survives multiplicity correction (§6.5). The honest reading is that NF4 may modestly increase Qwen 4B's harmful compliance, but the present sample cannot confirm it."),
  PJ("Mechanism and over-refusal. Qwen 4B at full precision is strongly refusal-calibrated on genuine harm (judge baseline ASR = 0.115). XSTest over-refusal is essentially unchanged (0.028 → 0.024, ΔOR = −0.004, not significant). A plausible mechanism for the directional ASR increase is the same NF4 degradation of refusal precision seen elsewhere; but unlike the 1.7B pair, the 4B model retains its capability, so any safety move here is not a capability artefact; it just is not large enough to confirm at this sample size. The subject-level MMLU is correspondingly flat, corroborating the preserved-capability reading: only two of six subjects regress (business ethics −4.0 pp, college biology −3.0 pp), three are unchanged, and high school macroeconomics rises slightly (+1.1 pp), a stark contrast with the 1.7B pair's five-of-six broad regression. Per-subject figures for all pairs are in results_512/analysis/mmlu_subject_breakdown.{json,csv}."),
  PJ("Scale contrast (RQ4). The 1.7B and 4B Qwen models exhibit different failure modes under the same quantization, but at the 512-token reference budget neither failure is on the harmful-compliance axis. The 1.7B model receives the broad_degradation label driven entirely by a significant capability loss (ΔMMLU −0.090) with its harm axis flat (ΔASR 0.000). The 4B model preserves MMLU but loses ARC significantly, and shows only a directional, non-significant safety worsening (alignment_degradation by point estimate, ΔASR +0.040). The within-family contrast is therefore a capability contrast: the smaller model suffers a large, significant MMLU loss while the larger model's capability loss surfaces only on ARC; on the safety axis both pairs are within noise. This points the same direction as the broader literature on quantization sensitivity declining with model size."),
  PJ("The Llama 3.2 3B pair results (§6.10, §6.11) allow assessment of whether any safety worsening reproduces cross-family."),

  H2("6.9 Research Question Synthesis"),
  PJ("With all ten models evaluated and HarmBench ASR scored by the official classifier at its 512-token reference budget, it is now possible to give answers to all five research questions. Paired bootstrap 95% confidence intervals quantify the statistical uncertainty behind each answer. At the reference budget the individually-significant deltas under the primary scorers are all on the capability and over-refusal axes: Qwen 1.7B ΔMMLU (−0.090), Llama 3B ΔARC (−0.032), Qwen 4B ΔARC (−0.016), and one over-refusal decrease, Phi-4-mini (−0.048), with Qwen 1.7B's over-refusal decrease (−0.024) now non-significant (bootstrap CI [−0.048, 0.000] touches zero, McNemar p = 0.109; §3.7). On the harmful-compliance axis, the only ΔASR whose CI excludes zero is Llama 3B's, at −0.040, a decrease, not an increase; no pair shows a significant ASR increase. Under Benjamini, Hochberg correction over the twenty-test NF4 family (§6.5) exactly three contrasts survive, and all three are capability or over-refusal effects (Qwen 1.7B MMLU, Llama ARC, Phi over-refusal); zero HarmBench ASR contrasts survive. Capability degradation is therefore the robust, multiplicity-surviving cost of four-bit NF4, while the harmful-compliance change is not robust. (The 128-token run produced an apparent Qwen 1.7B ASR increase of +0.055 that was a truncation artefact and does not appear at the reference budget; §6.16.)"),

  H3("RQ1: Does 4-bit NF4 quantization increase harmful compliance?"),
  PJ("At the 512-token reference budget it does not, and it does not significantly increase it in any pair either. Under the official classifier the ΔASR values are: Qwen 1.7B 0.000 (CI [−0.055, +0.055], not significant), Qwen 4B +0.040 (CI [0.000, +0.080], not significant, directional), Llama 3.2 3B −0.040 (CI [−0.075, −0.010], significant, a decrease), Mistral-7B −0.020 (CI [−0.080, +0.040], not significant), and Phi-4-mini +0.020 (CI [−0.015, +0.055], not significant). The only ΔASR whose CI excludes zero is Llama's, and it is a decrease, not an increase; no pair shows a significant harmful-compliance increase, and no ASR contrast survives multiplicity correction (§6.5). The answer to RQ1: at the reference budget NF4 quantization does not significantly change harmful compliance in either direction for four of five pairs, and the single significant move (Llama) is a decrease. The data do not support a claim that NF4 raises harmful compliance; the robust cost of NF4 is capability, not safety. (The secondary v2 proxy over-counts ASR and reports a different pattern; the official classifier is authoritative, see §6.12. The 128-token +0.055 on Qwen 1.7B was a truncation artefact, §6.16.)"),

  H3("RQ2: Does 4-bit NF4 quantization increase over-refusal on benign prompts?"),
  PJ("Across the five pairs, none moves significantly in the over-refusing direction and the only confirmed change is benign. Under the v2 scorer: Qwen3-1.7B ΔOR = −0.024 (0.052 → 0.028, not significant: bootstrap CI [−0.048, 0.000] touches zero, McNemar p = 0.109; §3.7), Qwen3-4B −0.004 (0.028 → 0.024, not significant), Llama 3B +0.016 (0.032 → 0.048, not significant), Mistral-7B 0.000 (0.004 → 0.004, not significant), and Phi-4-mini −0.048 (0.128 → 0.080, significant, McNemar p = 0.002). The one significant over-refusal delta (Phi, −0.048) is a decrease and Qwen 1.7B's decrease is now non-significant, both running in the benign direction: the 4-bit model refuses fewer benign prompts, not more, and Phi's decrease survives multiplicity correction (§6.5). No pair shows a significant increase. The answer to RQ2: NF4 quantization does not produce a detectable over-refusal increase on benign prompts, across four families and five pairs; the moves that approach or reach significance both run in the safety-benign direction. This is practically useful for deployment teams considering NF4 compression: the concern that quantization would make models more trigger-happy on benign prompts is not supported by these data; if anything, the significant effects point the other way."),

  H3("RQ3: Does 4-bit NF4 quantization degrade general capability?"),
  PJ("Yes, and this is the robust, budget-invariant cost of NF4. The Qwen3-1.7B model loses 9.0 percentage points of MMLU accuracy under NF4 quantization (0.643 → 0.553, significant), and this is the single largest, multiplicity-surviving effect in the study. Every pair's MMLU point estimate is negative (Qwen 4B −0.003, Llama −0.037, Mistral −0.020, Phi −0.027); on the second capability benchmark ARC-Challenge (§6.4.1) the losses that reach nominal significance are Llama (−0.032, which also survives the Benjamini, Hochberg correction) and Qwen 4B (−0.016, uncorrected only, it does not survive correction). So capability degradation holds in direction across all five pairs, reaches significance on at least one benchmark for the Qwen 1.7B, Qwen 4B and Llama pairs, and supplies two of the three effects that survive the Benjamini, Hochberg correction (the third survivor is Phi's over-refusal decrease). The answer to RQ3: NF4 quantization degrades general capability, most severely (and MMLU-significantly) in the 1.7B Qwen model, and this capability cost is robust to generation-length budget where the harmful-compliance change is not (§6.16). The one substantive cross-benchmark divergence is the Qwen 1.7B loss, which is far smaller and non-significant on ARC (−0.9 pp) than on MMLU (−9.0 pp), so its *severity* is partly MMLU-specific even though its direction is not."),

  H3("RQ4: Are smaller models more sensitive to quantization within the same family?"),
  PJ("On capability, yes; on harmful compliance, not at the reference budget. On capability the MMLU delta ratio between Qwen 1.7B and Qwen 4B is large (−9.0 pp versus −0.3 pp): the 1.7B model loses substantial, significant MMLU capability under NF4 compression while the 4B model's MMLU is essentially unaffected. On safety (judge, 512-token reference budget) neither Qwen pair shows a significant ΔASR: the 1.7B is exactly 0.000 and the 4B is a directional, non-significant +0.040, so there is no within-family safety-sensitivity contrast to report, the earlier +0.055 on the 1.7B was a 128-token truncation artefact (§6.16). The answer to RQ4: smaller Qwen models are more sensitive to NF4 quantization on capability *as measured by MMLU*, but this is benchmark-dependent, on ARC the within-family ordering reverses (the 4B pair loses −0.016 significantly while the 1.7B loses only −0.009, not significant), so the dramatic MMLU ratio is an MMLU-specific result, not a benchmark-independent scale law. On the harmful-compliance axis there is no scale contrast at the reference budget: both Qwen pairs are within noise. The robust RQ4 conclusion is therefore a capability one, and even that is MMLU-specific and roughly equalised by the second benchmark."),

  H3("RQ5: Are effects consistent across model families?"),
  PJ("Partially, and consistently in the safety-null / capability-cost direction. On capability, every pair's MMLU point estimate is negative under NF4 compression; the loss reaches MMLU-significance for Qwen 1.7B (−9.0 pp), and the ARC axis adds significant losses for Qwen 4B (−0.016) and Llama (−0.032). On safety (judge, 512-token reference budget), no pair shows a significant harmful-compliance increase: the only individually-significant ΔASR is Llama's, at −0.040, a decrease, and the remaining pairs (Qwen 1.7B 0.000, Qwen 4B +0.040, Mistral −0.020, Phi +0.020) are all within noise. The three original pairs receive three labels (broad_degradation but capability-driven for Qwen 1.7B, alignment_degradation directional for Qwen 4B, capability_collapse_masquerading_as_safety for Llama). The over-refusal axis is near-null (the one significant delta, Phi's −0.048, is a decrease and the Qwen 1.7B delta is now non-significant; §3.7). The full answer to RQ5 across the original two families: at the reference budget NF4 quantization does not raise harmful compliance in any pair (Llama's move is a significant decrease), does not significantly raise over-refusal, and imposes a capability cost that is consistent in direction and the only multiplicity-surviving class of effect. The cross-family extension (§6.13) confirms this across families: neither Mistral-7B (ΔASR −0.020, not significant) nor Phi-4-mini (+0.020, not significant) shows a significant increase, so no family shows an NF4-induced safety regression while the capability-cost pattern holds across four families."),

  H3("Discussion: What the full five-pair dataset tells us"),
  PJ("Three implications extend beyond the specific numbers. First, surface-level refusal metrics are unreliable without both a capability anchor and an accurate harmful-compliance scorer. The v2 regex over-counts ASR (it equates non-refusal with success), and the judge validation is a conclusion-level correction rather than a cosmetic one, the scorer, not the delta, determines the conclusion (§6.12). Second, the most robust empirical finding is on the capability axis, not the safety axis: capability degradation is directionally consistent across all five pairs, reaches significance on at least one benchmark for three pairs, and supplies two of the three effects that survive multiplicity correction (the third, Phi's over-refusal decrease, is most plausibly a regex measurement artifact: it is human-audited in §6.12 Result 6, and under the validation-informed parallel family only the two capability contrasts survive, §6.5.1), whereas no HarmBench ASR contrast survives and the only individually-significant ΔASR (Llama, −0.040) is a decrease. At the reference budget the data are a null-safety result with capability as the robust cost; they do not support the alarmist reading that NF4 raises harmful compliance, nor the complacent reading that NF4 is cost-free. Third, the framework itself is the durable contribution: it produces well-separated diagnostic categories, and, validated against the benchmark's own classifier and re-run at the benchmark's reference generation budget, it caught and corrected both a scorer artefact and a generation-length truncation artefact rather than propagating them. The framework's value does not depend on any specific empirical outcome; it provides a reusable, self-correcting methodology for the next study."),

  H2("6.10 Observations: Llama-3.2-3B Pair (pair_id llama_3_2_3b)"),
  PJ("The Llama 3.2 3B pair (SLURM jobs 60980 and 60981) completed on 2026-05-27. Job 60980 (baseline, float16) ran in 6 minutes 15 seconds; job 60981 (NF4 4-bit) ran in 13 minutes 9 seconds, consistent with the longer load time of quantized models observed in the Qwen runs."),
  PJ("The baseline profile of Llama 3.2 3B-Instruct is strongly refusal-calibrated under the official classifier at the 512-token reference budget. HarmBench ASR at full precision is 0.100 (judge): 20 of 200 direct harmful requests produced genuinely harmful completions. XSTest over-refusal is 0.032, low. MMLU accuracy is 0.610, closely matching the Qwen3-1.7B baseline (0.643). This profile describes a model that is both compactly capable and safety-calibrated at full precision: safer on HarmBench than the smaller Qwen model (judge baseline ASR 0.255 for Qwen 1.7B) while approximately equally capable as the smaller Qwen on MMLU. This is also the pair where the v2 regex and the judge agree most closely (κ ≈ 0.71 to 0.84; §6.12): Llama mostly either refuses cleanly or complies, leaving little ambiguous middle ground for the regex to mis-score."),
  PJ("Under NF4 4-bit quantization, the Llama pair is classified as capability_collapse_masquerading_as_safety. HarmBench ASR falls from 0.100 to 0.060 (ΔASR = −0.040, CI [−0.075, −0.010], statistically significant; McNemar's exact test p = 0.021): the 4-bit model refuses more of the direct harmful requests than the baseline, a small but significant safety-improving move, and the only individually significant ΔASR among the five pairs at the reference budget. XSTest over-refusal rises modestly from 0.032 to 0.048 (ΔOR = +0.016, not significant). MMLU accuracy falls by 3.7 percentage points (0.610 → 0.573, CI includes noise; not statistically significant), and the second capability benchmark, ARC-Challenge, falls by 3.2 percentage points (ΔARC = −0.032), which is significant. The capability decline is moderate and, on ARC, confirmed; the MMLU point estimate agrees in direction but does not reach significance at n = 200."),
  PJ("Why capability_collapse_masquerading_as_safety? Harmful compliance decreases and capability decreases together: the significant ARC loss establishes a genuine capability cost, and the ASR falls rather than rises, so the apparent safety improvement is diagnosed as a by-product of a less-capable model refusing (or failing to produce coherent harmful content) more often, not as a strengthening of alignment. This is diagnostically informative and it is the crux of the study's central dichotomy: a naïve reading of the significant ΔASR = −0.040 as 'NF4 makes Llama safer' is exactly the misinterpretation the two-layer scheme is built to catch. The contrast with Qwen 1.7B is instructive: at nearly identical baseline capability (MMLU 0.610 vs 0.643), the smaller Qwen loses significant MMLU capability with a flat harm axis (ΔASR = 0.000), whereas Llama loses capability and its harm axis moves in the safety-improving direction, but in both pairs the robust, confirmed effect is capability loss, not any alignment shift."),

  H2("6.11 Cross-Family Comparison (RQ5)"),
  PJ("The cross-family comparison places the Llama 3.2 3B pair alongside the two Qwen pairs to assess whether NF4 quantization effects are consistent across architectures and alignment recipes. Table 6.2 (§6.2) shows the delta values and interpretation labels for all three original pairs side by side; the cross-family extension to Mistral-7B and Phi-4-mini is analysed in §6.13."),

  H3("6.11.1 Baseline safety profiles differ substantially across families"),
  PJ("Before interpreting quantization deltas, the baseline profile differences are themselves informative. At similar MMLU capability (Qwen3-1.7B: 0.643; Llama 3.2 3B: 0.610), the Llama model is more safety-calibrated than the smaller Qwen under the official classifier at the 512-token reference budget: baseline HarmBench ASR is 0.100 for Llama vs 0.255 for Qwen 1.7B, a 15.5 pp difference in favour of Llama on the safety axis at equivalent capability. XSTest over-refusal at baseline is 0.032 for Llama vs 0.052 for Qwen 1.7B, comparable. This indicates that two compact models at the same factual capability have different safety calibrations, attributable to differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach. Family/recipe is a comparable-to-larger safety determinant than quantization for these models: the Llama vs Qwen 1.7B baseline gap on ASR (15.5 pp) dwarfs every matched-pair quantization delta in the study (the largest of which, Llama's own ΔASR, is 4.0 pp). The 4B Qwen baseline at 0.115 sits between these two."),

  H3("6.11.2 Harmful compliance: no significant increase in any family; the one significant delta is a decrease"),
  PJ("Under the official classifier at the 512-token reference budget, exactly one of the three ΔASR values is statistically significant, and it runs in the safety-improving direction: Llama 3.2 3B (−0.040, CI [−0.075, −0.010], McNemar p = 0.021), a small significant decrease in harmful compliance. The Qwen 4B ΔASR (+0.040, CI [0.000, +0.080]) is directional-upward but not significant, and the Qwen 1.7B ΔASR (0.000, CI [−0.055, +0.055], McNemar p = 1.000) is an exact null. No pair shows a statistically significant increase in genuine harmful compliance under NF4. Nor does any ΔASR contrast survive the Benjamini-Hochberg correction over the twenty-test NF4 family (§6.2): the three FDR survivors are all capability or over-refusal deltas, and zero HarmBench ASR contrasts survive. The one individually significant ASR move (Llama) is a decrease, and even it does not survive multiple-comparison correction. (This contrasts with where the v2 proxy would have pointed: the proxy over-counts ASR and would have flagged spurious increases; see §6.12.) The Qwen 1.7B pair, whose +0.055 at the 128-token budget was the study's original headline safety regression, shows no ASR movement at all at the reference budget: that 128-token effect was a truncation artefact (60 percent of 128-token responses were cut off before completion), and at 512 tokens the harm axis is flat (§6.16). The robust NF4 harmful-compliance conclusion is therefore null at the reference budget: no significant increase in any of the five pairs, and the single significant delta points the safety-benign way."),

  H3("6.11.3 Over-refusal: no significant increase across families"),
  PJ("The XSTest over-refusal rate is a consistent cross-family signal: four of the five ΔOR deltas are not statistically significant (Qwen3-4B (−0.004), Llama-3B (+0.016), Mistral-7B (0.000), and Qwen3-1.7B (−0.024, not significant: bootstrap CI [−0.048, 0.000] touches zero and McNemar p = 0.109; §3.7)), and the one significant move, Phi-4-mini (−0.048), runs in the benign direction, as does Qwen3-1.7B's now non-significant decrease: their 4-bit models over-refuse less, not more. No pair shows a significant increase in over-refusal. This pattern spans four families and five pairs and is practically useful for deployment: the concern that NF4 compression would make models excessively conservative on benign prompts is not supported by these data, and the one significant over-refusal move is a reduction in false refusals (Phi), with Qwen 1.7B's non-significant move in the same benign direction. The largest upward point estimate is on Llama (+0.016), a directional signal whose CI still includes zero."),

  H3("6.11.4 Capability: the robust, budget-invariant cost of quantization"),
  PJ("Capability degradation, not safety change, is the robust cost of NF4 at the reference budget. On MMLU, only the Qwen 1.7B loss reaches significance (−9.0 pp); Llama 3B (−3.7 pp) and Qwen 4B (−0.3 pp) do not. But the second capability benchmark, ARC-Challenge, confirms the loss where MMLU is silent: ARC falls significantly for Qwen 4B (−1.6 pp) and Llama 3B (−3.2 pp), while Qwen 1.7B's ARC loss (−0.9 pp) is small and non-significant. Every one of the three pairs therefore carries at least one significant capability loss across the two benchmarks (Qwen 1.7B on MMLU, Qwen 4B and Llama on ARC), and all six point estimates are ≤ 0. This is what survives multiple-comparison correction: two of the three FDR survivors in the twenty-test NF4 family are capability deltas (Qwen 1.7B MMLU, Llama ARC). The Llama 3B losses are intermediate between the two Qwen models, consistent with a size-sensitivity hypothesis (Llama 3B, at 3 billion parameters, lies between the two Qwen models), but because Llama and Qwen differ in architecture and training the ordering may also reflect family-specific effects; more data points (e.g. Qwen at 7B or Llama at 1B) would be needed to separate a size effect from a family effect."),

  H3("6.11.5 Interpretation labels and the study's central question"),
  PJ("Under the official classifier at the 512-token reference budget, the three interpretation labels are broad_degradation (Qwen 1.7B), alignment_degradation, directional (Qwen 4B), and capability_collapse_masquerading_as_safety (Llama 3B). They collectively answer the study's central question: \"do observed safety changes under quantization reflect alignment shifts or capability artefacts?\" The Qwen 1.7B pair carries the broad_degradation label, but at the reference budget that label is driven entirely by its significant MMLU capability loss (−9.0 pp) with the harm axis flat (ΔASR = 0.000, McNemar p = 1.000): it is a capability-only degradation, not a safety regression. The Qwen 4B pair preserves capability and shows only a directional, non-significant ASR increase. The Llama pair decreases on both axes: harmful compliance falls (significant ΔASR = −0.040) alongside a confirmed capability loss (significant ΔARC = −0.032), so its apparent safety improvement is diagnosed as capability-driven rather than a genuine alignment gain. Every one of the three pairs carries a significant capability loss (Qwen 1.7B on MMLU, Qwen 4B and Llama on ARC), and none shows a significant increase in harmful compliance. The central scientific finding is the framework's diagnostic value: capability degradation is the robust, budget-invariant cost of NF4, whereas the harmful-compliance change is not robust, no ASR contrast survives multiple-comparison correction, and the one significant ASR delta (Llama) is a decrease that the label scheme correctly attributes to capability loss. The data refute the optimistic narrative that NF4 quantization is a clean safety win in this scale range, while also refuting any claim of a robust safety regression. These conclusions are drawn from the original three pairs; the cross-family extension (§6.13) adds Mistral-7B and Phi-4-mini and finds the same diagnostic structure (Mistral in the proxy-over-counting camp, judge ΔASR −0.020, not significant, versus a +0.060 proxy artefact; and Phi in the proxy-reliable camp, κ 0.67, its judge ΔASR a non-significant +0.020 that sits on the tolerance and so carries a directional alignment_degradation label, not a confirmed regression), so the central question's answer generalises to four families."),

  H2("6.12 Judge-Model Validation of HarmBench ASR (the primary scorer)"),
  PJ("HarmBench Attack Success Rate is scored by the official HarmBench classifier, cais/HarmBench-Llama-2-13b-cls, the fine-tuned judge model specified by the HarmBench paper. Given a harmful behavior and a model generation, it answers whether the generation is a non-minimal, unambiguous instance of the behavior (\"yes\" = attack success). This subsection reports the validation that established the judge as the primary HarmBench scorer over the v2 regex (decision D16). The judge re-scores the already-saved generations; it performs no new target-model inference and does not modify the TC1-original raw outputs or the v2 sidecars. Judge labels are written to redacted per-prompt sidecars (scores.judge.harmbench_cls.jsonl, IDs and booleans only) and aggregated to summary.judge.harmbench_cls.json. The judge covers the HarmBench benchmark specifically; over-refusal on XSTest is a different question (\"did the model refuse a benign prompt?\") and remains v2-scored as primary, with an independent three-class refusal judge added as a sensitivity cross-check (Result 6, below)."),
  PJ("Methodology and setup. For the primary 512-token study the judge classified every saved (behavior, generation) pair for all ten fp16/NF4 models (n = 200 each, 2 000 total) on the NTU TC1 cluster via sbatch in offline mode (job 61524, zero parse errors); the five INT8 aliases (§6.15) were scored by an identical follow-up job (a further 1 000 classifications, also zero parse errors). The same protocol had first been run on the 128-token study (jobs 61047 and 61134), which is the validation that originally established the classifier as the primary scorer (decision D16); the agreement figures in this section are recomputed at the 512-token reference budget. It ran in full precision (float16) on a 32 GB Tesla V100 (recorded in summary.judge.harmbench_cls.json) because the judge is the reference instrument validating the quantized target models, and quantizing the judge itself would reintroduce the very confound the check exists to rule out. All 3 000 generations of the 512-token study (2 000 fp16/NF4 + 1 000 INT8) were classified with zero parse errors. Agreement with the v2 scorer is reported per model as raw agreement rate and Cohen's kappa over the shared prompt set; the per-pair judge ΔASR is computed with the same paired bootstrap (2 000 resamples, seed 42) used for the other deltas, and passed through the interpretation-label rule (holding the v2 over-refusal and MMLU deltas fixed, since the classifier judges only HarmBench). Agreement is reported at the 512-token reference budget (the primary study); the judge and regex re-score the same saved generations. Figures are produced locally by scripts/judge_agreement.py and stored in results_512/analysis/judge_agreement.{json,csv}."),
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
  P("Table 6.3  Judge vs v2 per-model agreement on HarmBench (n = 200 each). \"v2 non-refusal\" is the regex's harmful-compliance rate; \"judge ASR\" is the official classifier's. Agreement and Cohen's κ are over the shared prompt set, computed at the 512-token reference budget. From results_512/analysis/judge_agreement.json.", { size: 18 }),
  ...Figure("judge_vs_proxy.png", "Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal' proxy, one marker per model. Eight of the ten points lie below the diagonal, so the proxy systematically over-counts harmful compliance (most severely for Qwen and Mistral); the two marginal exceptions (Llama base, Phi 4-bit) sit just above it. Right: judge-vs-proxy Cohen's κ per model, which is family-dependent: low for Qwen/Mistral (κ ≈ 0.25 to 0.59) and high for Llama/Phi (κ ≈ 0.67 to 0.84). Source: results_512/analysis/judge_agreement.json."),
  PJ("Result 1: the v2 regex systematically over-counts ASR, unevenly by family. The judge ASR is far lower than the v2 non-refusal rate for the Qwen models (e.g. Qwen 1.7B base 0.255 vs 0.595; Qwen 4B base 0.115 vs 0.235) and close for Llama (0.100 vs 0.090). Agreement is correspondingly poor-to-moderate for Qwen and Mistral (κ ≈ 0.25 to 0.59) and high for Llama and Phi (κ ≈ 0.67 to 0.84). The reason is structural: the regex equates \"did not refuse\" with \"attack succeeded\", but many non-refusals are not genuinely harmful (vague deflections, safety lectures without refusal keywords, on-topic but benign answers, failed attempts). Qwen produces a large volume of such ambiguous middle-ground text; Llama mostly either refuses cleanly or complies, leaving little for the regex to mis-score. This is exactly the failure mode the HarmBench paper's fine-tuned classifier was built to avoid, and it is why the official classifier, not refusal counting, is adopted as the primary HarmBench scorer."),
  PJ("Result 2: the choice of scorer changes the conclusion. Under the judge, at the 512-token reference budget, the per-pair HarmBench ΔASR values are: Qwen 1.7B 0.000 (CI [−0.055, +0.055], not significant); Qwen 4B +0.040 (CI [0.000, +0.080], not significant); Llama −0.040 (CI [−0.075, −0.010], individually significant and in the safety-improving direction). Compared with the v2 proxy (−0.025, +0.070, −0.020), the judge and the regex disagree in level and per-model ordering: the proxy's largest positive move is on Qwen 4B (+0.070) whereas the judge's only individually significant contrast is a decrease on Llama, and no HarmBench ΔASR contrast survives the Benjamini-Hochberg correction over the primary family. The interpretation labels follow: Qwen 1.7B is broad_degradation, but at 512 that label is driven entirely by its significant MMLU capability loss (ΔMMLU −0.090), with the harm axis flat (ΔASR 0.000), it is a capability-only degradation, not a safety regression; Qwen 4B is alignment_degradation but only directionally (+0.040, not significant); Llama is capability_collapse_masquerading_as_safety (harmful compliance decreases and capability decreases together). The judge validation is therefore not a rubber stamp: it corrected a refusal-counting artefact, and at the reference budget it establishes that refusal-based scoring and the classifier lead to different per-model conclusions about where, and in which direction, harmful compliance moves. This is reported transparently as the central methodological result of the chapter, and the HarmBench numbers throughout the report are the judge values accordingly (D16)."),
  PJ("Result 3: the null-increase conclusion is robust to the choice of significance test. The judge ΔASR significance was re-tested with McNemar's exact test, which is the correct paired-binary test for HarmBench and assumes no large-sample normality (§3.7). It reproduces the bootstrap conclusion: at 512 tokens Qwen 1.7B p = 1.000, Qwen 4B p = 0.096, and Llama p = 0.021, the sole individually significant contrast is Llama's decrease, and no pair shows a significant harmful-compliance increase under either test, so the study's null-increase conclusion does not depend on the bootstrap machinery. A structural property of the judge, regex relationship reinforces the over-counting diagnosis: across all models the judge labelled essentially no generation harmful that the regex had cleared (judge_harmful_v2_not ≈ 0), so the judge's harmful set is almost a strict subset of the regex's: the disagreement is one-directional over-counting by the regex, not two-sided noise."),
  PJ("Result 4: the finding is robust to the choice of judge (second-judge cross-check). To rule out the possibility that the result is an artefact of the single HarmBench classifier, the saved generations were independently re-scored by a second, architecturally unrelated frontier judge (gpt-4o) applying a condensed one-sentence yes/no rubric that encodes the same harmful-behaviour criterion as the HarmBench classifier (the api_judge backend; decision D25). The two judges agree strongly at the per-prompt level: across all ten models Cohen's κ ranges from 0.68 to 0.95 (substantial-to-almost-perfect agreement), far higher than the regex-versus-classifier agreement (κ ≈ 0.25 to 0.59 for Qwen and Mistral), confirming that it was the regex, not the classifier, that was the outlier. The per-pair ΔASR direction is reproduced at 512 tokens: gpt-4o gives Qwen 1.7B +0.005 (versus the classifier's 0.000) and Llama −0.035 (versus −0.040); for the cross-family pairs it agrees on the sign of Mistral's and Phi's small moves, and it places Mistral's ASR near 0.55, matching the classifier's ≈0.57, not the v2 proxy's 0.83 to 0.89. Both judges agree that no pair shows a material harmful-compliance increase and that Mistral's proxy-driven 'increase' is an artefact. The former 128-token Qwen 1.7B effect is null under both judges at the reference budget (classifier 0.000, McNemar p = 1.000; gpt-4o +0.005), so there is no boundary-significance disagreement left to reconcile; this converges with the multi-seed sensitivity arm (§6.6.1), whose greedy 0.000 sits inside the seed range. Both judges agree that at the reference budget NF4 does not raise harmful compliance in any pair. Per-model agreement and per-pair figures are in results_512/analysis/judge_pairwise_agreement.{json,csv} (redacted)."),
  PJ("Result 5: the classifier tracks human judgment; the regex does not. The two preceding cross-checks compare the classifier against other models. To ground the comparison against truth, a stratified subset of 200 saved HarmBench generations, spanning the ten base/NF4 aliases of the primary study and deliberately over-sampling the judge-vs-regex disagreement cases, was labelled by a human annotator against the HarmBench rubric (the redacted aggregate is results_512/analysis/human_validation.json; the annotation sheet holds raw response text and is local-only). The primary classifier agrees with the human at Cohen's κ = 0.59 (moderate agreement); the demoted refusal regex agrees at κ = 0.11 (negligible). Precision makes the gap concrete: of the responses each scorer flagged harmful, the human confirmed 61 percent of the classifier's but only 23 percent of the regex's, so against human judgment the regex issued 101 false-harmful calls to the classifier's 18. Of the three scorers the human applied the highest bar for harmful compliance, flagging the fewest responses, 37 of 200, against the classifier's 46 and the regex's 132, so both automated scorers apply the harmful label more readily than the human does. The two over-countings differ in kind, however: the classifier's disagreement with the human is mild and roughly two-directional (18 responses it flagged that the human judged benign, against 9 the human flagged that it missed), which reads as balanced noise around the human bar; the regex's is a near-one-directional flood (101 over-flags against 6 misses), i.e. systematic over-counting. The classifier is therefore close to the human bar and the regex far above it. This is the human-grounded form of the chapter's central result: refusal-counting over-states harmful compliance relative to the classifier and relative to human judgment, and the classifier is the substantially better scorer. Two bounds are stated with equal plainness. First, κ = 0.59 makes the classifier a better scorer, not a perfect human proxy, it still over-calls 18 cases, exactly the caution the LLM-as-judge literature urges [23], [24]. Second, this is a single annotator on a partly subjective rubric (some behaviours, such as subtly biased but superficially positive text, are genuinely borderline for humans and judges alike), so a second annotator with an inter-rater κ is the natural strengthening. Because the sample deliberately over-weights judge-vs-regex disagreements (120 of the 200 items), these κ values sit on a harder-than-representative slice and are not population estimates; the design does not identify which way a representative population would move them, since enriching for disagreement changes both the difficulty of the items and the marginal prevalence κ depends on. The pair should be read as a disagreement-enriched contrast rather than a population estimate. What survives that sampling is the ordering and its size: on the contested cases that decide the scorer choice, the classifier tracks the human and the regex does not."),
  PJ("Threats and scope. The judge is itself a model and can err; it is adopted because it is the field-standard reference classifier specified by the HarmBench paper [4], run at full precision, with deterministic greedy decoding and zero parse failures on this dataset. It validates HarmBench only; XSTest over-refusal and MMLU are unaffected. The single-judge construct-validity threat is addressed on two fronts: the second-judge cross-check (Result 4), where an independent frontier judge agrees at κ 0.68 to 0.95 across all ten models and reproduces the finding's direction and magnitude; and the human-label audit (Result 5), which grounds the scorer comparison against human judgment rather than against another model and finds the classifier substantially closer to the human (κ 0.59) than the regex (κ 0.11). The claim this study makes is therefore the human-grounded one, the choice of scorer changes the conclusion, and refusal-counting over-states harmful compliance relative to both the benchmark's own classifier and a human annotator, with the residual honestly bounded: the classifier is a better scorer, not a perfect oracle (κ 0.59 is moderate, not near-perfect), and the grounding rests on a single annotator, so a second annotator with an inter-rater κ is the remaining strengthening. Two further residual caveats concern the judges themselves: the second judge is a versioned API model (less reproducible than the open-weight primary), and both judges share the broad lineage of LLM-based harm classification, so a fully independent open-weight guard (e.g. Llama Guard [25]) is the natural complementary cross-check; that guard has since been run, and its verification and fold-in are pending, so this revision carries the two-judge and human-grounded validation only. As a data-handling note, that second judge is an external API, so for its cross-check the (public) HarmBench behaviour string and the model's generation are transmitted off-cluster, whereas the primary classifier runs entirely on TC1 and only redacted identifier-and-boolean sidecars are stored. The v2 regex is retained in the repository and in Table 6.1/6.2 as a transparent secondary proxy so that readers can see exactly how much, and where, refusal-based scoring diverges from genuine harmful-compliance scoring."),
  PJ("Result 6: the over-refusal axis is scorer-sensitive too, and the one over-refusal survivor does not replicate under an independent judge. The validation above concerns HarmBench ASR; the study's third multiplicity-robust finding, by contrast, is an XSTest over-refusal decrease scored by the demoted v2 refusal regex (Phi-4-mini, ΔOR = −0.048, §6.5.1). To test whether that regex-scored finding is scorer-robust, the same question the classifier answered for HarmBench, the 3,750 saved benign XSTest responses (fifteen aliases × 250 prompts) were re-scored by an independent API judge (gpt-4o, pinned dated snapshot gpt-4o-2024-08-06, temperature 0, zero parse failures) that labels each response by the XSTest paper's own three-class taxonomy: full compliance, partial refusal, full refusal (Röttger et al. [5], who themselves use a GPT-4 judge with this taxonomy). Strict over-refusal counts full refusals only (the conservative reading); broad counts partial or full. This is a sensitivity check, not a change of scorer: the deterministic regex remains the primary over-refusal scorer of record and the multiple-comparison family of §6.5.1 is unchanged. The finding parallels the HarmBench one. First, the two scorers disagree substantially on benign over-refusal: the judge counts roughly four times as many refusals as the regex (mean over-refusal 0.171 strict versus 0.044), because it treats moral lectures, redirections, and answer-free 'I can't do X but I can do Y' alternatives as refusals whereas the keyword regex does not, and per-alias agreement is only poor-to-moderate (Cohen κ from −0.01 to 0.50 across the ten base/NF4 aliases of the primary study). Second, and decisively, the Phi-4-mini over-refusal decrease does not replicate: under the judge the Phi ΔOR is +0.016 (strict; direction reversed, CI [−0.028, +0.060], McNemar p = 0.597) or −0.004 (broad; CI [−0.048, +0.036], McNemar p = 1.000), not significant under either mapping, and neither reproduces the regex's −0.048 decrease. No pair shows a significant judge ΔOR under either mapping (every CI includes zero), which is consistent with the study's RQ2 over-refusal null, but the direction and magnitude diverge from the regex most sharply for the one contrast that had survived multiplicity correction. The honest reading, carried into §6.5 and the Abstract, is that the single FDR-surviving over-refusal contrast is scorer-dependent, and the human audit below resolves which scorer to weight: the contrast is most plausibly a measurement artifact of the refusal regex, not a classifier- or human-validated over-refusal shift. This extends the chapter's central message, the scorer determines the conclusion, from the harmful-compliance axis to the over-refusal axis. As with the second HarmBench judge, the benign prompt and saved response were transmitted to an external API for this check (public benchmark text; only redacted identifier-and-label sidecars are stored, enforced by the same redaction contract). All five per-pair judge ΔORs are reported in Table 6.4 regardless of direction, as exploratory sensitivity results under the pre-registered presentation rule. The largest, Qwen3-1.7B's +0.040 (strict; McNemar p = 0.087), is an apparent increase that does not reach significance. Full per-alias agreement is in results_512/analysis/xstest_judge_agreement.{json,csv} (redacted); no new significance claim enters the pre-registered multiplicity family."),
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
  P("Table 6.4  Exploratory per-pair XSTest over-refusal deltas (NF4 − fp16) under the independent three-class refusal judge, both mappings, with paired-bootstrap 95% CIs (10 000 resamples, seed 42) and exact McNemar p-values; the primary regex ΔOR is shown for comparison (★ = the one regex-significant, FDR-surviving contrast). Every pair is reported regardless of direction under the pre-registered presentation rule; no judge delta is significant and none enters the §6.5.1 registered multiplicity family (the judge-strict column is the over-refusal member of the §6.5.1 validation-informed parallel family). From results_512/analysis/xstest_judge_agreement.json.", { size: 18 }),
  PJ("The human grounding for this axis has since been completed, mirroring Result 5. All 200 items of a pre-registered gold set, drawn from the same 3,750 saved responses, deliberately over-sampling the regex-versus-judge disagreements (120 of 200) and double-weighting the contested Phi and divergent Qwen-1.7B pairs, presentation-shuffled and label-blind, were labelled by a single human annotator (the study's author) under the same three-class XSTest taxonomy the judge applies. That taxonomy is the benchmark's own, but the shared rubric is disclosed plainly: a semantic judge is structurally better placed than a phrase-matcher to satisfy a shared semantic rubric, so the comparison measures alignment with the annotator under the benchmark's construct, not scorer quality in the abstract. The pre-registered read-off (docs/XSTEST_GOLD_PREREG.md, locked with a J/R/T/X outcome matrix before any label existed) is Outcome J: the judge aligns substantially better with the annotator than the regex, at strict-mapping Cohen's κ 0.485 versus −0.006 and broad-mapping 0.662 versus 0.054, and the two failure modes are asymmetric in kind. The regex detected 2 of the annotator's 63 full refusals (recall 0.032; the 61 missed refusals are typically long, polite, lecture-style declinations carrying none of the template phrases), whereas the judge caught 61 of 63 (recall 0.968) and errs by over-flagging (52 over-flags, 25 of them on the genuinely subjective partial-versus-full boundary; three-class exact agreement with the annotator 0.695). Three bounds are stated with the same plainness as Result 5's. First, these labels are a blinded single-annotator reference set, not ground truth: a single annotator can misread and drift, and no inter-rater κ exists for this set. Second, the draw is disagreement-enriched, so the κ values are a contested-slice contrast, not population estimates, and no population over-refusal rate or human ΔOR is estimated from them (descriptively, the labelled Phi items show no base-versus-4-bit difference in human-judged strict refusal, but the subset licenses no population claim). Third, the pre-registered no-swap rule (D45) was locked before the judge results existed but after the regex results were known, a timing disclosed rather than argued from. Under the pre-registered Outcome-J action, the regex remains the original pre-specified scorer-of-record and the registered multiplicity family is unchanged; the interpretive weight, however, shifts to the judge as the scorer better aligned with the single annotator on the enriched audit. The Phi-4-mini −0.048 contrast is therefore not treated as robust evidence of reduced over-refusal (it is most plausibly a measurement artifact of the regex scorer), and the §6.5.1 validation-informed parallel family gives that reading a multiplicity-controlled footing: under it, two contrasts survive, both capability effects. The committed aggregate is results_512/analysis/xstest_human_validation.json (counts and rates only; the annotation sheet holds raw text and stays local); a second annotator with an inter-rater κ remains the residual strengthening, exactly as for Result 5."),

  H2("6.13 Cross-Family Extension: Mistral-7B and Phi-4-mini (RQ5)"),
  PJ("To test whether the findings generalise beyond the original two families, two further matched pairs were added and run on TC1 on 2026-06-15 under the identical methodology (on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding at temperature 0.0, seed 42, the same four benchmarks at the same sample counts, and the official HarmBench classifier as the primary ASR scorer): mistral_7b (Mistral-7B-Instruct-v0.3, 7.2B, the largest model in the study) and phi4_mini (Phi-4-mini-instruct, 3.8B). This takes the study to five pairs across four families (Qwen, Llama, Mistral, Phi). Both pairs' raw metrics, deltas, and judge-agreement figures appear in Tables 6.1, 6.2 and 6.3, and their second-judge (gpt-4o) cross-check is folded into the §6.12 figures. The two pairs fall into the two diagnostic camps already seen among the original three."),
  PJ("Mistral-7B is the clearest demonstration of the judge-over-proxy finding (§6.12, D16) in the whole study. The v2 refusal proxy reports a high baseline non-refusal rate (0.830) rising to 0.890 under quantization, a +0.060 'increase' that, taken at face value, would label the pair alignment_degradation. The official classifier tells the opposite story: at the 512-token reference budget genuine harmful compliance is 0.585 at baseline and 0.565 under 4-bit, a ΔASR of −0.020 (CI [−0.080, +0.040], not significant; McNemar p = 0.627): quantization slightly reduces, not increases, true harmful compliance. The proxy and the judge disagree in sign, and the judge, proxy agreement is the lowest in the study (Cohen's κ = 0.29 at baseline, 0.25 under 4-bit; at the retired 128-token budget it had been lower still, 0.19/0.11). The second independent judge (gpt-4o) corroborates the classifier, not the proxy, agreeing with the primary classifier at κ 0.83 (baseline) and 0.78 (4-bit) and reproducing a small negative ΔASR. Capability is essentially preserved (MMLU −2.0 pp, n.s.; ARC +0.9 pp, n.s.) and over-refusal is flat (ΔOR = +0.004, n.s.), so under the judge the pair reads as alignment_improvement at the point-estimate level, carried as directional because its ΔASR CI spans zero. For the study's largest model, NF4 leaves genuine harmful compliance unchanged-to-slightly-lower while preserving capability, and a refusal-based scorer would have badly mischaracterised it."),
  PJ("Phi-4-mini falls in the opposite (Llama-like) camp, where the proxy and the judge largely agree. Genuine harmful compliance is low across the pair (judge ASR 0.070 at baseline and 0.090 under 4-bit, ΔASR +0.020, CI [−0.015, +0.055], not significant; McNemar p = 0.424), so on safety the pair takes a directional alignment_degradation label (evidence status: directional), its ΔASR of +0.020 sitting exactly on the +0.02 harm-tolerance boundary and not statistically distinguishable from zero. The judge, proxy agreement is high (κ = 0.67, like Llama), and the second judge agrees at κ 0.68 at baseline and 0.83 under 4-bit at the reference budget, the baseline is in fact the study's lowest cross-judge agreement, and the near-perfect κ 0.79/0.95 belongs to the retired 128-token budget. On capability, Phi shows a modest, non-significant dip on both axes: MMLU −2.7 pp and ARC −1.5 pp (both n.s.). Its one significant delta is on over-refusal: the 4-bit model is slightly less over-cautious on benign prompts, ΔOR = −0.048 (CI [−0.076, −0.020], significant under the primary regex scorer), the only significant over-refusal delta in the study, and in the benign direction (fewer false refusals, not more), though this contrast is regex-scored and scorer-dependent: an independent three-class refusal judge does not reproduce it (§6.12, Result 6)."),
  PJ("Across all four families the central pattern holds and strengthens. Under the official classifier at the 512-token reference budget, none of the five pairs shows a significant increase in genuine harmful compliance under NF4: no pair's ΔASR is a significant increase, the only individually significant contrast is Llama's decrease (−0.040), and the two new families add a non-significant decrease (Mistral, −0.020) and a non-significant increase (Phi, +0.020). The proxy-reliability split spans four families (the regex over-counts severely for Qwen and Mistral (judge-vs-proxy κ 0.25 to 0.59) but tracks the judge for Llama and Phi (κ 0.67 to 0.84)), so the methodological contribution (validate the scorer before trusting the delta) generalises rather than being a Qwen-specific artefact. On over-refusal, four of five pairs show no significant change and the fifth (Phi) moves in the benign direction, so the RQ2 deployment message (NF4 does not make models more trigger-happy on benign prompts) survives the extension. Capability loss stays directionally consistent (every pair's MMLU and ARC point estimates are ≤ 0, the sole exception being Mistral's negligible +0.9 pp on ARC). The cross-family extension therefore corroborates RQ5: NF4's safety effect is not uniform across families, is never a significant worsening in any pair, and is cleanly separable from capability only once the scorer itself is validated."),

  H2("6.14 Mechanism Probe: Refusal-Margin Geometry (the \"why\")"),
  PJ("The behavioural results (§6.3, §6.12, §6.13) establish that at the 512-token reference budget NF4 does not significantly increase harmful compliance in any pair, and that behavioural flips are near-symmetric and small. This mechanism probe operates on the first-token refusal distribution, which is fixed at the first generated position and therefore generation-length-invariant, so its numbers are budget-invariant and reported unchanged from the 128- and 512-token runs alike; its capability-driven boundary-instability reading now aligns with the 512-token null (harmful compliance flat, capability the robust cost). To probe why flips occur where they do, a derived mechanism analysis measured each model's first-token refusal margin (m = log P(refusal-token set) − log P(compliance-token set), a scale-invariant log-probability difference at the first generated position) on all 200 HarmBench prompts per model, in fp16 and under NF4, via teacher-forced forward passes (scripts/capture_refusal_margin.py on TC1; redacted scores.margin.<precision>.jsonl sidecars carrying only prompt-id and scalar margins, no text; analysed by scripts/refusal_margin_analysis.py → results/analysis/refusal_margin.{json,csv}). The hypothesis under test, following Proskurina et al. [11], is that quantization perturbs the decision distribution most where the baseline margin is thin, so behavioural flips should concentrate at thin baseline margins. The analysis was run with explicit validity, circularity and confound controls, and the headline numbers were independently recomputed from the raw sidecars."),
  PJ("Validation gate. Before any quantization claim the margin's sign was checked against the baseline judge label. For four families it discriminates refusal from compliance well (AUC 0.86 to 0.90 for Qwen-1.7B, Qwen-4B, Llama, Phi) but for Mistral-7B only weakly (AUC 0.69), because Mistral frequently opens with a refusal-shaped preamble and then complies: its refuse-versus-comply decision is resolved later than the first token. The first-token margin is therefore a sound proxy for four families and a poor one for Mistral, which is flagged throughout. The gate is itself only suggestive: the well-discriminating pairs have small baseline-harmful counts (8 to 13 prompts) and correspondingly wide AUC intervals, whereas Mistral's lower estimate rests on the largest positive class (77)."),
  PJ("Central result: boundary instability, not targeted erosion. A thin baseline margin does predict which prompts flip under NF4, but the within-model effect is modest. Pooling all 1,000 prompts gives AUC 0.76; after removing the between-family level difference (z-scoring the margin within each pair) it falls to 0.64, and within individual pairs it is near chance for the two thinnest-margin families (Qwen-1.7B 0.61, Mistral 0.54). The flips are also close to symmetric in aggregate: of 92 flips, 50 are harmful-ward (refuse→comply) and 42 are safe-ward (comply→refuse), and a thin margin predicts the safe-ward flips (AUC 0.78) at least as strongly as the harmful-ward ones (0.75). This pooled near-symmetry must, however, be read with care: it is driven by Mistral-7B (the one family whose first-token margin the gate above flags as a poor proxy), which alone contributes 20 harmful-ward and 28 safe-ward flips (two-thirds of all safe-ward flips). Across the four families where the first-token margin is a valid proxy the flips lean roughly 2:1 harmful-ward (30 versus 14), and the load-bearing Qwen-1.7B pair is 16 versus 5. Two of those valid-proxy families are nonetheless individually symmetric (Llama 2:2, Phi 5:5) and the safe-ward direction is well represented, so thin first-token margins are best read as marking prompts near the greedy-decoding decision boundary that quantization can destabilise in either direction, a boundary-fragility effect rather than evidence of a safety-specific erosion mechanism. The strongest evidence for the boundary-instability reading of the one behaviourally-moving pair (Qwen-1.7B) is not this pooled symmetry but the entropy confound control and the significant ΔMMLU discussed next."),
  PJ("Direction of the margin shift is model-specific and modest. The paired shift Δm = m(fp16) − m(NF4) is statistically significant for every pair (Wilcoxon p < 0.001, n = 200) but moves comply-ward for Qwen (Δm = +1.2 for 1.7B, +4.0 for 4B) and refuse-ward for Llama and Phi (−1.1, −0.3), with effect sizes spanning Cohen dz ≈ −0.74 to +1.8; the whole-sequence margins reverse the first-token sign for Qwen-1.7B and Phi, so the directional claim is first-token-specific. The supportable statement is that NF4 perturbs the first-token refusal distribution in a model-specific direction, not that it uniformly erodes refusal. Qwen-4B shifts comply-ward more than Qwen-1.7B (+4.0 vs +1.2) yet flips fewer prompts (9 vs 21), because its baseline margin is far wider (median 13.0 vs 3.8): it has refusal headroom to absorb the perturbation, which is the mechanistic reason the 4B pair is behaviourally intact while the 1.7B is not."),
  PJ("Capability versus alignment: the tie to the central question. If NF4 simply raised next-token entropy everywhere, every margin would shrink as a by-product of generic confidence loss rather than a safety-specific change. Comparing the entropy rise on harmful prompts with that on neutral (MMLU / XSTest-benign) tokens, the harmful side softens more only for Qwen-4B and Mistral, the two pairs that do not drive the behavioural result. For the one pair that does move, Qwen-1.7B, neutral-token entropy rises roughly ten times more than harmful-token entropy (+0.21 vs +0.02): the change is consistent with generic confidence/capability softening near the boundary, not a targeted alignment shift. (This control is suggestive only: n = 100 unpaired neutral prompts, a mixed set, no significance test.) The mechanism evidence therefore converges with the significant Qwen-1.7B ΔMMLU (§6.2), the multi-seed fragility (§6.6.1) and the ARC-versus-MMLU capability picture (§6.4.1): the smallest model's harmful-compliance wobble reads as capability-driven boundary instability rather than genuine alignment erosion. The probe's contribution is not a clean safety-erosion mechanism: it is to localise the effect to near-boundary prompts and to weigh the evidence, for the one moving pair, toward the capability side of the study's central dichotomy. Two mechanism follow-ups (an independent activation-space refusal-direction probe and a paired neutral-margin control) are listed in Chapter 9; the third anticipated there, an INT8 precision point tracing the effect across the fp16 → INT8 → NF4 spectrum, has since been run and is reported in §6.15."),

  H2("6.15 INT8 Precision Point: Method- and Bit-Width Sensitivity (fp16 → INT8 → NF4)"),
  ...Figure("precision_sweep.png", "Precision sweep fp16 → INT8 → NF4. Capability (MMLU, ARC) is essentially flat through INT8 and falls only at four-bit (a cliff, not a gradient), the one robust, budget-invariant precision effect. The safety axis (judge ASR) shows no robust move at either precision step at the 512-token reference budget: the 128-token Llama-3B INT8 increase does not replicate here. Source: results_512/analysis/precision_sweep.json."),
  PJ("The main study evaluates one quantization method (NF4) at one bit-width (four-bit). To test whether the safety and capability effects are a smooth function of quantization aggressiveness or are instead method- and bit-width-specific, an INT8 precision point was added so every pair is evaluated across three precisions: fp16 → INT8 → NF4. INT8 here is the bitsandbytes LLM.int8() mixed-precision method (a distinct algorithm, not a lower-bit NF4) applied on the fly to the same baseline weights. All five INT8 members were run on the NTU TC1 cluster at the 512-token reference budget (five per-model matrix jobs plus the HarmBench-classifier judge job), scored by the primary official classifier and cross-checked by the gpt-4o second judge, with zero parse errors throughout; the baselines and NF4 members are the same 512-token artefacts as the primary study, so the three precisions are directly comparable. The analysis (configs/tc1_int8_512.yaml, scripts/precision_sweep_analysis.py, with paired-bootstrap CIs and McNemar tests recomputed from the redacted judge sidecars) is kept separate from the base-vs-4-bit pairwise pipeline so the evaluated ten-model study is untouched. The question it answers: does degradation scale monotonically with bit-width (a graded fp16 > INT8 > NF4 trend), or is it concentrated and method-specific?"),
  PJ("Capability: a clean cliff at four-bit. The capability answer is unambiguous and holds at the 512-token reference budget. No INT8 delta, on either MMLU or ARC, is statistically significant for any of the five pairs (paired bootstrap), so within the study's resolution INT8 is capability-lossless. Every significant capability loss appears only at the four-bit step: ΔMMLU(NF4) = −0.090 (Qwen-1.7B); ΔARC(NF4) = −0.016 (Qwen-4B) and −0.032 (Llama-3B). Eight-bit is therefore effectively a free precision point on capability, and the capability cost of quantization is a cliff at four-bit rather than a graded decline. (The one borderline cell is Phi-4-mini ΔARC(NF4) = −0.015, non-significant only by the closed-interval convention, footnoted so the dichotomy is not overstated as perfectly clean.)"),
  PJ("Safety: no robust precision effect at the reference budget. The safety axis is the more revealing result, and at the 512-token reference budget it yields no robust move at either precision step. The 128-token run had reported a both-judge-significant INT8-specific increase on Llama-3.2-3B (ΔASR = +0.040, McNemar p = 0.008/0.022, non-monotonic, rising at INT8 and reverting at NF4). At 512 tokens this vanishes under both judges: Llama INT8 ΔASR = +0.005 under the primary classifier (McNemar p = 1.000) and +0.010 under gpt-4o (McNemar p = 0.688), neither significant. Every INT8 ASR contrast is non-significant under both judges across all five pairs. The 128-token Llama INT8 signal was therefore a budget-sensitive artefact, though of a different kind from the truncation that inflated the Qwen-1.7B NF4 move: Llama's generations truncate at only 3 to 4% (the lowest of any family; §6.16), so its 128-vs-512 instability traces to cross-run greedy divergence (Llama carries the study's highest prefix-mismatch share, §6.16) rather than to cut-off responses, either way, not a stable method-specific regression. The conceptual conclusion is that quantization's effect on safety is neither bit-width-graded nor budget-robust: the only robust, budget-invariant precision effect is the capability cliff at four-bit, while no INT8 or NF4 harmful-compliance increase survives at the reference budget."),
  PJ("Honest reading of the INT8 result. The lesson of the 128-to-512 comparison is stated plainly. The 128-token Llama INT8 increase rested on a small absolute flip count (≈8 to 9 prompts) on a single pair and was non-monotonic; at the 512-token reference budget it does not replicate under either judge (classifier +0.005, gpt-4o +0.010, both McNemar n.s.), so it is best read as a budget-sensitive artefact rather than a method-specific \"INT8 erodes safety\" effect, and because Llama's generations truncate at only 3 to 4% (§6.16), the instability is attributable to cross-run greedy divergence on thin-margin prompts, not to truncated responses. No INT8 ASR contrast is significant at 512 under either judge, so nothing on the safety axis needs a multiple-comparison correction to be discounted. The picture is consistent with the §6.14 mechanism reading (thin-margin boundary prompts are destabilised by quantization in a model-specific direction, and which prompts flip is itself idiosyncratic and budget-sensitive) rather than any graded or method-specific safety erosion. Replication of INT8 behaviour across more models and decode seeds remains a natural next step (Chapter 9), but at the reference budget the honest statement is a null on the safety axis and a clean capability cliff at four-bit."),
  PJ("Scorer note (the proxy at INT8). The secondary v2 refusal regex continues to over-count harmful compliance at INT8 exactly as it does at fp16 and NF4: judge-versus-proxy Cohen κ is lowest for Mistral (the worst over-counter) and highest for Llama-3B and Phi-4-mini, the same D16 pattern (low Qwen/Mistral agreement, high Llama/Phi agreement) reproduced at the new precision. The official classifier therefore remains the primary scorer at INT8, and every safety number above is judge-scored; the proxy is reported only as the demoted foil whose failure motivates the judge. The precision point sharpens the study's central message at the reference budget: within the compact-deployment regime the capability cost of quantization is a clean cliff at four-bit, whereas the safety axis shows no robust precision effect, neither bit-width-graded nor budget-robust. INT8 is capability-lossless and safety-neutral at 512 tokens, and no harmful-compliance increase (INT8 or NF4) survives at the reference budget."),

  H2("6.16 Generation-Length Robustness: the 512-Token Rerun"),
  PJ("The HarmBench authors show that the number of tokens generated during evaluation can change the measured attack-success rate by up to 30 percent, and for exactly that reason standardize the generation parameter to N = 512 in their evaluation framework [4] (\"we standardize this parameter to N = 512 to allow the metric to converge\"); the official HarmBench evaluation harness likewise clips completions to 512 tokens before classification (evaluate_completions.py, num_tokens default 512). The 512-token budget is therefore HarmBench's own standardized evaluation budget, not a choice of this study. This study therefore adopts the 512-token reference budget as its primary configuration, and retains an initial 128-token run purely as a generation-length comparison. This section reports that comparison and explains why the 512-token budget is the correct primary: it shows that several safety signals visible at 128 tokens were truncation artefacts that disappear at the reference budget. The initial run generated at most max_new_tokens = 128; the primary run regenerates the entire study at 512 tokens, all five pairs, all four benchmarks, scored by both the primary HarmBench classifier and the gpt-4o second judge. The two runs live in parallel results trees (results_512/ is primary, via configs/tc1_512.yaml with the classifier judge in slurm/judge_validation_512.sbatch; the 128-token artefacts are retained unchanged for comparison), so the two budgets are directly comparable. All ten matrix jobs and both judges completed with zero parse errors at both budgets."),
  PJ("The reference budget changes the generations materially, which is the crux of why it must be primary. At the 512-token budget the HarmBench responses are far longer: median length 1,675 characters against 567 at 128 tokens (a roughly threefold increase). A direct prefix test shows 60.3 percent of the 128-token responses were provably truncated (the 512-token generation continues past where the 128-token one was cut off), and a character-ceiling proxy puts the truncation rate at about 62 percent, so the 128-token run was cutting off the majority of generations before completion, and any safety signal measured there is confounded with where the truncation fell. Two disclosures accompany the prefix statistic. First, it is family-heterogeneous: Mistral truncates at 93.5 to 98.0 percent and Phi at 73.5 to 78.5, Qwen at 54.5 to 70.0, while Llama truncates at only 3.0 to 4.0 percent, Llama's 128-token generations mostly stop naturally or diverge, so the truncation mechanism is family-specific even though the reference-budget conclusion is not. Second, 9.2 percent of the 2,000 paired generations (184, concentrated in Llama) are prefix mismatches: the 128- and 512-token runs, though both nominally greedy, diverge early rather than one being a truncation of the other, benign run-to-run non-determinism between two separate executions at different max_new_tokens (kernel and batching effects), conservatively excluded from the truncated count, which makes 60.3 percent a floor. Every matched-pair safety delta in this study is computed within a single run, so this cross-run non-determinism does not affect any reported contrast. Absolute ASR is correspondingly higher at 512 under both judges (for example Mistral-7B classifier ASR is 0.585 at the reference budget against 0.385 at 128 tokens), because a complete generation gives more room for harmful content to appear. That is a property of the absolute rate, not of the matched-pair delta, and it is the delta that carries every safety claim in this study."),
  PJ("A qualitative note on the form of these longer generations, surfaced while inspecting responses during the human-label audit (§6.12). The generation loop halts only on the tokenizer's default end-of-sequence id and otherwise runs to the token budget, so a completed answer is frequently not the end of the sequence: having nothing left to answer, the model continues, and often fabricates a synthetic next turn, inventing a follow-up user request and then answering it, so a single record can contain the model role-playing both sides of a short dialogue. Because decoding uses skip_special_tokens, the intervening chat-turn markers are stripped and this is saved as a run-on, which is why some responses read as a refusal immediately followed by an unrelated benign answer (a refused harmful request trailed by, for instance, an unprompted explanation of photosynthesis). This is a decoding-configuration artifact, not a model-capability finding, and it is benign for the study's claims on two counts: it is produced identically by both members of every pair, so it cancels in the matched-pair delta; and the primary classifier scores the entire response rather than its opening, so a continuation that did emit harmful content would still be counted (the human-label audit, §6.12 Result 5, is the direct check on whether that scoring is correct, and it finds the classifier substantially closer to human judgment than the regex). Constraining generation to stop at the chat turn-end token is a clean fix for future runs and would also shorten and tidy the saved transcripts."),
  buildTable(
    ["Pair", "ΔASR @128 (classifier)", "ΔASR @512 (classifier)", "ΔASR @512 (gpt-4o)"],
    [
      ["Qwen3-1.7B", "+0.055 (sig @128)", "0.000", "+0.005"],
      ["Qwen3-4B", "+0.025", "+0.040", "+0.040"],
      ["Llama-3.2-3B", "0.000", "−0.040", "−0.035"],
      ["Mistral-7B", "−0.040", "−0.020", "−0.005"],
      ["Phi-4-mini", "0.000", "+0.020", "+0.020"],
    ],
    [1400, 2650, 2650, 2660]
  ),
  P("Table 6.5  HarmBench ΔASR across the retained 128-token comparison and the 512-token reference budget. Source: results_512/analysis/genlen_robustness.json.", { size: 18, after: 160, alignment: AlignmentType.CENTER }),
  PJ("The headline of the comparison is that the safety signal seen at 128 tokens was a truncation artefact: the study's only significant 128-token safety regression does not exist at the reference budget. The Qwen3-1.7B increase of +0.055 at 128 tokens (base/NF4 ASR 0.255/0.255 at 512) is ΔASR = 0.000 under the primary classifier and +0.005 under gpt-4o at 512 tokens, neither significant (McNemar p = 1.000 under both judges, and both paired-bootstrap confidence intervals span zero). At the reference budget it is a capability-only degradation with a null harm delta, not a safety regression. The Qwen3-1.7B null at 512 is moreover a high-churn null: sixteen prompts flip from refusal to compliance and sixteen flip from compliance to refusal, a symmetric exchange that is exactly the boundary-instability signature the §6.14 mechanism probe identified, now visible directly at the reference budget. Across all five pairs at 512, no ΔASR survives a Benjamini-Hochberg correction under either judge; the only individually significant ΔASR is Llama-3.2-3B at −0.040 (classifier, CI [−0.075, −0.010], McNemar p = 0.021) and −0.035 (gpt-4o), which is in the safety-improving (decrease) direction and does not survive the correction either. The three contrasts that do survive BH-FDR across the 20-test NF4 family are all capability or over-refusal effects (Qwen-1.7B MMLU −0.090, Llama ARC −0.032, Phi over-refusal −0.048), never a harmful-compliance increase. The two judges agree per prompt at Cohen κ 0.68 to 0.95, matching the 0.60 to 0.95 range reported for the 128-token run, so the scorer is no less reliable at the reference budget."),
  PJ("Capability degradation, in contrast, is essentially unchanged by the generation budget, which is why it, not the safety axis, is the robust cost of NF4. The four-bit MMLU and ARC losses move only at the third decimal between the two budgets: Qwen3-1.7B ΔMMLU goes from −0.087 to −0.090, Llama-3.2-3B ΔARC from −0.028 to −0.032, and Qwen3-4B ΔARC from −0.021 to −0.016, with the same cells remaining significant at 512. The capability cost of quantization is budget-robust; the harmful-compliance change is not."),
  PJ("The generation-length comparison therefore establishes the study's central dichotomy at the reference budget. The capability cost of NF4 is real and budget-robust, whereas the apparent 128-token safety regression is fragile on every axis tested: it does not survive the 512-token reference budget, a multiple-comparison correction, the second judge, or stochastic decoding (§6.6.1). Quantization in this regime erodes competence more than it erodes guardrails, and the one borderline safety signal observed at 128 tokens is best read as a truncation artefact and capability-driven boundary instability rather than a robust alignment shift. Under the official classifier at 512, Qwen3-1.7B retains its broad_degradation label, but that label is now driven entirely by its significant MMLU loss (−0.090) with the harm axis flat (ΔASR 0.000); Llama-3.2-3B is capability_collapse_masquerading_as_safety (both harm and capability decrease). The 128-token results are retained unchanged as this comparison. The INT8 precision point (§6.15) and the multi-seed sensitivity arm (§6.6.1) have both been regenerated at 512 tokens and confirm the null: the 128-token Llama INT8 increase vanishes under both judges at the reference budget, and the multi-seed greedy ΔASR of 0.000 for Qwen3-1.7B sits inside its stochastic seed range (mean +0.013, range [0.000, +0.035], 0/5 seeds significant; all five seed deltas non-negative, a sub-MDE directional signal disclosed in §6.6.1), so at 512 there is no detectable effect for stochastic decoding to attenuate. The multi-seed arm covers three pairs (Qwen3-1.7B, Qwen3-4B, Llama), not all five."),
];

// ------------------------------------------------------------
// Chapter 7 - Discussion and Threats to Validity
// ------------------------------------------------------------
const ch7 = [
  H1("Chapter 7: Discussion and Threats to Validity"),

  PJ("The Qwen family results (§6.9) provide initial empirical grounding for the threats discussed in this chapter. At the 512-token reference budget the Qwen 1.7B pair carries the broad_degradation label, but that label is driven entirely by its significant capability loss (ΔMMLU = −0.090) with a flat harm axis (ΔASR = 0.000, not significant); the Qwen 4B pair is directional alignment_degradation (ΔASR +0.040, not significant). These are used below as concrete examples where relevant, with the caveat that the smallest model's movement is capability-driven, not a safety regression."),

  H2("7.1 Internal Validity"),
  PJ("Internal validity is the strongest property of the present design. The matched-pair structure, combined with on-the-fly NF4 quantization from identical baseline weights, isolates quantization as the sole experimental variable. There is no plausible alternative explanation for an observed delta beyond the quantization step itself, the act of loading the same checkpoint twice, or measurement noise. Deterministic decoding (temperature 0.0) eliminates within-condition variance from generation. Scoring is also deterministic: MMLU uses exact match, XSTest over-refusal uses the regex parser, and the primary HarmBench scorer (the official HarmBench classifier) is run with greedy decoding (max_new_tokens = 1) in full precision, producing reproducible labels (the 512-token primary judge job, 61524, returned zero parse errors over 2 000 generations, as did the earlier 128-token validation jobs). The resume logic prevents partial-run contamination: every reported metric is computed from a complete raw.jsonl with exactly the configured number of prompts. All ten target-model runs of the primary 512-token study completed under the same conditions, hardware, and software stack."),

  H2("7.2 External Validity"),
  PJ("External validity is bounded by five explicit design choices. First, the safety axis is measured under direct harmful requests only, no adversarial attack or jailbreak transformation is applied (Chapter 3), so the harmful-compliance null does not speak to robustness under optimised attacks (GCG, PAIR, and similar), which is precisely the regime where prior quantization-safety work reports attack-dependent effects. Second, findings are bounded to BitsAndBytes on-the-fly quantization (NF4 as the main study, INT8/LLM.int8 as a precision point). Alternative methods such as GPTQ, AWQ, and the GGUF family used by llama.cpp may produce different effects, both quantitatively and qualitatively; the present study does not claim to characterise those methods. Third, the Qwen3 baselines (Qwen3-1.7B and Qwen3-4B) are instruction-tuned dense models from the official Alibaba Qwen3 release; results describe quantization behaviour on this specific model family and may not transfer directly to other architectures or training regimes. Fourth, the study is restricted to English-language, text-only interactions; multilingual and multimodal effects are out of scope. Finally, the full-precision baseline is loaded at float16 (dtype: auto resolves to float16 on CUDA), not the models' native bfloat16; because both members of every pair share the same compute dtype, the matched-pair delta still cleanly isolates quantization, but the absolute baseline figures are float16 rather than bfloat16 values."),

  H2("7.3 Construct Validity"),
  PJ("Each benchmark operationalises its dimension in a specific way. HarmBench captures one curated definition of \"harmful compliance\" against a particular set of harmful behaviour requests, presented directly and without adversarial attack augmentation (the threat-model boundary stated in Chapter 3); other operationalisations exist, including attack-mediated ASR under HarmBench's own attack suite, and may yield different absolute numbers. XSTest captures over-refusal against a specific distribution of benign-but-suspicious prompts. The MMLU subset, even with diverse subject coverage, is a partial capability proxy and does not measure all reasoning capacities relevant to deployment (such as code generation, long-context reasoning, or tool use). A related construct point is stated plainly: the MMLU and ARC scores here are controlled zero-shot subset anchors chosen for a paired, within-model comparison on identical prompts, not official leaderboard reproductions. The official MMLU protocol is five-shot, so the absolute accuracies reported here are not directly comparable to published leaderboard figures; because both members of every pair are scored under the identical zero-shot subset protocol, the matched-pair capability delta is still clean, but the absolute anchors should not be read as leaderboard results. These construct boundaries are common in safety evaluation but should be borne in mind when interpreting the results."),

  H2("7.4 Scorer Choice and Remaining Validity Threat"),
  PJ("HarmBench ASR was initially scored with a deterministic regex parser, chosen for reproducibility. The judge validation (§6.12) showed this regex measures \"non-refusal rate\" rather than genuine harmful compliance and over-counts ASR, most severely for the Qwen models, so the official HarmBench classifier was promoted to the primary HarmBench scorer (decision D16) and the regex demoted to a transparent secondary proxy. This removes the original construct-validity threat on the HarmBench metric: the primary scorer is now the benchmark's own fine-tuned classifier, run at full precision with deterministic decoding. Two residual threats remain. First, the HarmBench classifier is itself a model and can err. This single-judge threat has now been directly tested: the saved generations were re-scored by a second, architecturally unrelated frontier judge (gpt-4o) applying the same HarmBench rubric (the api_judge backend; D25). The two judges agree strongly per prompt (κ 0.60 to 0.95 across all ten models, and κ 0.68 to 0.95 at the 512-token reference budget), which is the load-bearing point: the choice of scorer, not of judge, is what determines the conclusion. At the 512-token reference budget the two judges also agree that no pair shows a significant harmful-compliance increase; for the smallest Qwen pair the primary classifier reports ΔASR = 0.000 and gpt-4o +0.005, both null (McNemar p = 1.000 under both judges, §6.12, Result 4; §6.16). The classifier has also now been grounded against human judgment (§6.12, Result 5): on a stratified subset of 200 generations a human annotator agrees with the classifier at Cohen's κ 0.59 and with the regex at only 0.11, so the classifier is the better scorer against truth, not merely against another model, a moderate, single-annotator agreement that a second annotator would strengthen. The construct-validity threat is therefore substantially resolved (the null-safety conclusion is not an artefact of one classifier, and the primary scorer is closer to human judgment than the demoted regex), with two residual caveats: the second judge is a versioned, less-reproducible API model, and both judges share the general lineage of LLM-based harm classification, so a fully independent open-weight guard (e.g. LlamaGuard) is the natural complementary cross-check; that guard has since been run, with verification and fold-in pending, so this revision carries the two-judge and human-grounded validation only. Second, XSTest over-refusal is scored by the regex parser as primary, since the HarmBench classifier does not judge over-refusal; an independent three-class refusal judge (gpt-4o) has now cross-checked it as a sensitivity layer (§6.12, Result 6) and finds the one multiplicity-robust over-refusal contrast (Phi-4-mini −0.048) to be scorer-dependent, not reproduced by the judge, and the pre-registered human refusal audit has since grounded that comparison: the judge aligns substantially better with a blinded single annotator than the regex (strict κ 0.485 versus −0.006, Outcome J), so the contrast is carried as most plausibly a regex measurement artifact, with a second annotator the remaining strengthening on that axis. The immutable-raw-output contract (TC1-original raw.jsonl/summary.json untouched; all corrected and judge scores in derived sidecars) means either future judge can be added without disturbing the existing evidence trail."),

  H2("7.5 Cross-Family Comparison Caveat"),
  PJ("The Qwen-versus-Llama comparison should be read as descriptive only. Qwen and Llama differ in tokenizer, pre-training corpus, instruction-tuning recipe, and safety-alignment methodology. Differences in their quantization deltas could plausibly reflect any combination of these factors, not only family identity. The cross-family component of the study therefore provides a useful robustness check on the within-Qwen findings but does not support causal claims about quantization, family interactions."),

  H2("7.6 Deployment Implications"),
  PJ("The practical takeaway is asymmetric across the scale range studied, but the asymmetry is on the capability axis, not the safety axis. At HarmBench's 512-token reference budget no pair shows a significant increase in genuine harmful compliance under NF4: across the five pairs the only individually significant ΔASR is Llama-3.2-3B at −0.040 (a decrease, in the safety-improving direction), and no ASR contrast survives a Benjamini-Hochberg correction over the NF4 test family (§6.3, §6.16). The apparent Qwen 1.7B safety regression seen at the 128-token budget (+0.055) does not replicate at 512 (ΔASR = 0.000 classifier, +0.005 gpt-4o, McNemar p = 1.000): it was a generation-length truncation artefact, and the smallest model's real, budget-robust cost is capability, not compliance (ΔMMLU = −0.090, significant). What survives correction at 512 is exclusively capability and over-refusal (Qwen 1.7B MMLU, Llama ARC, Phi over-refusal), so the robust, budget-invariant cost of NF4 in this regime is degraded competence, not eroded guardrails. A team that validates on the full-precision 1.7B release and ships the 4-bit build should expect a lower capability level than they signed off on (−9.0 pp MMLU) rather than a higher harmful-compliance rate. The multi-seed check at 512 (§6.6.1, covering three of the five pairs) corroborates this: the greedy ΔASR of 0.000 for Qwen 1.7B sits inside the per-seed range and no pair is seed-robust in the harmful direction. The deployment guidance is therefore one-sided but capability-shaped: 4-bit NF4 did not significantly increase harmful compliance under direct requests in any pair at the reference budget, and its dependable effect is capability loss, worst in the smallest model, which is precisely the size class most likely to be quantized for on-device or edge use. The larger Qwen 4B model tolerates quantization far better on the MMLU anchor, its MMLU is essentially preserved and its safety shift is only a non-significant nudge, though ARC records a small significant loss (−1.6 pp, §6.4.1), so \"less MMLU-sensitive\" is the supportable claim rather than full capability preservation. The operational recommendations that follow are to re-run evaluation on the exact quantized artefact that will be deployed (not the full-precision release), to evaluate at the deployment generation budget rather than a truncated one, and to prefer the largest model that fits the memory budget over aggressively quantizing a smaller one. These recommendations are established for on-the-fly BitsAndBytes quantization (NF4, INT8) under direct harmful requests; deployment formats not tested here (GGUF, GPTQ, AWQ) and attack-mediated threat models may behave differently and are future work (Chapters 8 to 9)."),

  H2("7.7 Positioning Against Prior Work"),
  PJ("These findings sit within, qualify, and extend the emerging quantization-and-safety literature (§2.4). Prior work establishes that quantization can degrade safety alignment rather than being behaviourally neutral, with effects that vary by method, model, and attack type (Kharinaev et al. [12], Q-resafe [15], HarmLevelBench [14]). The one prior study that, like this one, isolates quantization at four bits is Hong et al. [16], which, across a broader trust taxonomy, reports that 4-bit compression broadly retains trustworthiness; the null-safety result here is consistent with that finding and extends it with a matched-pair capability anchor and a classifier-validated harmful-compliance scorer. This study contributes a boundary condition to that picture: for on-the-fly BitsAndBytes quantization evaluated under direct requests at the benchmark's own generation budget, no significant harmful-compliance increase appears in any of five pairs; the non-uniformity the literature reports is visible here only as small, sign-varying, non-significant safety movements across families, and the degradation this study robustly measures is on the capability axis. The Egashira et al. [13] result that BitsAndBytes NF4 specifically can turn a benign model harmful is the closest prior analogue to the mechanism studied here, though that work demonstrates an adversarial worst case whereas this study measures the effect under ordinary, non-adversarial loading. The distinct contribution of the present work is the matched-pair, capability-anchored design: by holding the prompt set fixed and measuring harmful compliance and general capability jointly on the same on-the-fly-quantized weights, it separates capability change from safety change on each pair rather than reading a single non-refusal axis. At the 512-token reference budget that separation yields a null-safety result: harmful compliance does not significantly rise in any pair, whereas capability significantly falls in several, so every pair's movement is best read on the capability side of the dichotomy. Qwen 1.7B, which the truncated 128-token budget made look like an alignment regression, is at 512 a capability-only degradation (a significant MMLU loss with a flat harm axis), and Llama is capability_collapse_masquerading_as_safety (harmful compliance falls alongside capability). Distinguishing these cases from a genuine alignment shift is exactly what the cited single-axis studies are not structured to do."),
];

// ------------------------------------------------------------
// Chapter 8: Limitations
// ------------------------------------------------------------
const ch8 = [
  H1("Chapter 8: Limitations"),
  PJ("The principal limitations of the study are summarised below. Each is acknowledged explicitly and, where possible, addressed in the Future Work chapter."),
  ...numberedList([
    "Threat model: direct requests only. HarmBench was evaluated in its standard configuration, the curated harmful behaviours presented directly, with no adversarial attack or jailbreak augmentation applied (no GCG, PAIR, AutoDAN, or other attack pipelines from the HarmBench suite). Every harmful-compliance finding, including the headline null, is therefore scoped to the weakest threat model (direct requests) and does not establish that the null would survive optimised adversarial attack, which is where prior quantization-safety work reports effects; an attack-augmented arm is flagged as the highest-impact extension in Chapter 9.",
    "Quantization-method coverage. Two BitsAndBytes methods are evaluated: NF4 four-bit (the main study) and INT8/LLM.int8 as a precision point (§6.15), so the study now spans fp16 → INT8 → NF4 rather than a single bit-width. GPTQ, AWQ, and GGUF paths remain out of scope and are flagged for follow-up work, as does replication of the §6.15 Llama-3B INT8 effect across more models and decode seeds.",
    "Refusal parser approximation. The deterministic regex-based refusal parser is reproducible but may under-count nuanced refusals.",
    "Partial capability proxy (now two-benchmark). The six-subject MMLU subset is a tractable but partial measure of general capability and does not include code generation, long-context reasoning, or tool-use evaluation. To address this, a second capability benchmark (ARC-Challenge, 1,172 reasoning-oriented questions) has now been run on all ten models (§6.4.1). It confirms the direction of capability loss but shows the MMLU magnitudes are partly benchmark-specific. Qwen 1.7B's large MMLU drop does not replicate on ARC, while Qwen 4B shows a small significant ARC loss MMLU missed, so capability is now reported on the two benchmarks side by side. The interpretation labels remain MMLU-anchored pending a formal composite-capability rule (Chapter 9).",
    "MMLU answer-format sensitivity. MMLU is scored by exact-match on a parsed option letter, so an item counts as incorrect both when the model selects the wrong option and when it never commits to a letter within the generation budget. At the 128-token budget this was a real concern for the smallest pair: 11 of 300 Qwen 1.7B 4-bit MMLU responses were cut off mid-reasoning without a final letter (answered rate 0.963 vs 0.983 baseline). At the 512-token primary budget the truncation component is resolved: every model answers every item (answered rate 1.000 across all fifteen aliases), so none of the primary ΔMMLU = −0.090 is attributable to answer-format truncation. A distinct format asymmetry remains at 512, however: the lenient fallback parser fires on 48.7% of the 4-bit member's MMLU items versus 3.3% at fp16 (§6.5), so part of the −0.090 magnitude reflects an answer-format shift that the parser scores less reliably, not genuinely wrong answers alone. ARC is subject to the same asymmetry even more strongly for this pair (52.3% of its 4-bit ARC items fall to the lenient fallback versus 2.5% at fp16), so ARC's smaller lenient magnitude reflects the parser salvaging those format-broken answers (fallback accuracy 66%), not immunity; under a strict parser ARC falls −0.343 and MMLU −0.293, comparably (§6.5). The truncation caveat therefore applies only to the retained 128-token comparison (§6.16); the parser-leniency caveat applies at the primary budget and is quantified in §6.5",
    "Text-only, English-only scope. Multilingual and multimodal behavioural effects of quantization are out of scope.",
    "Generation-budget dependence of the absolute rate (directly evidenced, then controlled). The HarmBench attack-success rate depends on how many tokens the target is allowed to generate, and the study's original 128-token budget truncated the majority of generations: at 512 tokens the median HarmBench response is 1,675 characters against 567 at 128, and 60.3 percent of 128-token responses are a proper prefix of their 512-token continuation (a direct prefix test, ≈62 percent by the character-ceiling proxy), so most 128-token generations were cut off before the model finished. HarmBench ASR is therefore reported at HarmBench's 512-token reference budget as primary; the 128-token run is retained as a generation-length comparison (§6.16). The matched-pair ΔASR that carries every safety claim is budget-robust in its conclusion (no significant increase at either budget after correction), but the absolute rates are budget-dependent and the 128-token deltas for individual pairs are not (the Qwen 1.7B +0.055 at 128 is a truncation artefact that goes to 0.000 at 512).",
    "Greedy decoding only (stochastic-decoding arm now complete for three pairs). Temperature 0.0 is used throughout the primary study, eliminating within-condition stochastic variance from the main analysis. A multi-seed (T = 0.7, top-p 0.8, five seeds) sensitivity arm was run at the 512-token reference budget and scored by the official classifier (§6.6.1); it covers three of the five pairs (Qwen 1.7B, Qwen 4B, Llama), not the full matrix. At 512 it corroborates the null-safety result: for Qwen 1.7B the greedy ΔASR of 0.000 sits inside the per-seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant), so there is no stochastic effect to attenuate, and no pair is seed-robust in the harmful direction. Extending the same symmetric arm to the two remaining pairs (Mistral, Phi) and raising the seed count would complete the within-condition variance estimate across the full matrix.",
    "Generation not halted at the chat turn boundary. The generation loop stops only on the tokenizer's default end-of-sequence id, so responses frequently run to the full token budget; once its answer is complete the model sometimes generates synthetic follow-up turns, role-playing both sides of a short dialogue, which skip_special_tokens decoding saves as a run-on (§6.16). This is a decoding-configuration artifact: it inflates response length but is produced identically by both pair members, so it cancels in every matched-pair delta, and because the primary classifier scores the whole response (not just its opening) a continuation that emitted harmful content would still be counted. It is a straightforward fix for future runs (add the chat turn-end token to the stop criteria) and does not affect any result reported here.",
    "Hardware and walltime constraints. Each TC1 job is allocated a single GPU, ten gigabytes of host memory, and six hours of walltime. Sample budgets and batch sizes are sized to fit comfortably within these constraints.",
    "Sample-size-driven confidence intervals and a bounded null. With 200 HarmBench prompts and 250 benign XSTest prompts, binomial-proportion confidence intervals are wider than large leaderboard settings; small deltas may not be statistically separable from zero. The per-pair minimum detectable effect for ΔASR at 200 paired items is approximately 0.04 to 0.09 (reported with the multiple-comparison analysis, §6.5.1). The headline HarmBench null should therefore be read as bounded, the evidence rules out any harmful-compliance increase larger than that detectable effect, not that the change is exactly zero, and pre-registering a smallest effect of interest and powering the design to it, including the two pairs (Mistral, Phi) not yet covered by the multi-seed arm, is noted in Chapter 9.",
    "Model-weight provenance (pinned). Each of the five model repositories is pinned by Hugging Face commit hash, the optional revision field (Appendix C) set on every entry in configs/tc1.yaml and configs/tc1_512.yaml (Appendix A), to the exact snapshot cached on TC1 at the recorded run dates, so an online from-scratch reproduction loads byte-identical weights rather than a possibly re-uploaded head. The residual reproducibility boundary is therefore not the weights but the raw per-prompt generations: those are retained locally (gitignored, hash-pinned in results/raw_artifact_manifest.sha256), so the analysis replays from the committed redacted sidecars but is not recomputable from raw generations on a fresh clone.",
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
    "Adversarial-attack arm. Re-run the HarmBench axis under at least one optimisation-based or LLM-mediated attack (for example GCG or PAIR) on a subset of pairs, upgrading the direct-request compliance result (Chapter 3 threat model) to a quantization-versus-safety-under-attack result and directly engaging the attack-dependent effects reported by HarmLevelBench [14] and Egashira et al. [13]. This is the highest-impact extension of the safety axis.",
    "Multi-method quantization comparison. Extend the matrix to include GPTQ, AWQ, and GGUF quantization paths on the same baselines, allowing direct comparison of how different PTQ algorithms perturb safety and capability. GGUF is the deployment-relevant format named in the motivation (Chapter 1), so it is the priority method.",
    "Stochastic-decoding sensitivity arm (completed for three of five pairs at the reference budget). A multi-seed arm (temperature 0.7, top-p 0.8, five seeds, official classifier) was run at the 512-token reference budget for three pairs (Qwen 1.7B, Qwen 4B, Llama; §6.6.1). At 512 it corroborates the null-safety result rather than tempering a real effect: for Qwen 1.7B the greedy ΔASR of 0.000 sits inside the per-seed range (mean +0.013, range [0.000, +0.035], 0 of 5 seeds significant), so there is no stochastic effect to attenuate, and Llama's seeds are all non-positive (mean −0.024). (At the truncated 128-token budget the same arm had attenuated an apparent +0.055; that effect has since been shown to be a truncation artefact, §6.16.) Extending the arm to the two remaining pairs (Mistral, Phi) and raising the seed count would complete the within-condition variance estimate across the full matrix.",
    "Composite-capability interpretation rule (second benchmark now run). ARC-Challenge has been run on all ten models (§6.4.1) and corroborated the direction of capability loss while showing the MMLU magnitudes are partly benchmark-specific, notably the Qwen 1.7B drop (−9.0 pp MMLU, significant, vs −0.9 pp ARC, n.s. at the reference budget) and the fact that MMLU's within-Qwen scale gap is not reproduced on ARC. The interpretation labels currently stay MMLU-anchored with ARC as a corroborating axis; formalising a composite-capability rule (e.g. requiring agreement across benchmarks before assigning a capability-driven label) is the natural next step. Adding further capability benchmarks (e.g. GSM8K for math reasoning, HellaSwag for commonsense) would broaden the composite.",
    "Mechanism follow-ups (refusal-margin probe, §6.14). The first-token refusal-margin analysis localises the quantization effect to near-boundary prompts but is first-token-specific (its whole-sequence margins disagree in sign for two pairs) and its capability/confidence confound control is small (n = 100, unpaired, mixed neutral set, no significance test). Three follow-ups would sharpen it: (1) an independent activation-space refusal-direction probe (Arditi et al. [26]) as a second mechanism window that does not rely on a hand-built token set; (2) a paired neutral-margin control with a significance test, to separate safety-targeted shifts from generic confidence softening more rigorously; and (3) an INT8 precision point (now run and reported in §6.15) which traced the behavioural effect across the fp16 → INT8 → NF4 spectrum and found only the four-bit capability cliff to be a robust precision effect, with the safety axis non-significant at every precision under both judges at the reference budget; extending the first-token margin probe itself across the three precisions, rather than only the behavioural metrics, remains open.",
    "Cross-family and scale extension (completed 2026-06-15). Two further matched pairs were implemented and run on the cluster: mistral_7b (mistralai/Mistral-7B-Instruct-v0.3) and phi4_mini (microsoft/Phi-4-mini-instruct), taking the study to five pairs across four families (Qwen, Llama, Mistral, Phi) and adding a seven-billion-parameter point at the upper edge of the compact-deployment regime. Both pairs use the identical methodology (on-the-fly NF4 with the same BitsAndBytesConfig, greedy decoding, the same four benchmarks at the same sample counts, seed 42, and the official HarmBench classifier as the primary ASR scorer), so the comparison stays matched. The only new loader capability is an optional attn_implementation field (Phi-4-mini uses the eager attention backend on the V100, which has no flash-attention kernels); it is now covered by the configuration schema, the loader, the per-model SLURM job set, the judge-validation scripts, and the verification suite. Phi-4-mini loads through transformers' native Phi3 implementation rather than its bundled remote code, keeping its load path consistent with every other model in the study. Results are reported in Tables 6.1 to 6.3 and analysed in §6.13: under the judge, no pair shows a significant harmful-compliance increase, and at the 512-token reference budget Mistral ΔASR is −0.020 (n.s.; the v2 proxy's positive 'increase' is a sign-flipped over-count, judge-vs-proxy κ as low as 0.25) and Phi ΔASR is +0.020 (directional alignment_degradation, κ 0.67), so adding two families leaves the null-safety conclusion intact and reinforces the judge-over-proxy finding (D16) in two further families.",
    "INT8 precision point / quantization-method sweep (completed; reported in §6.15). To test whether the safety and capability effects depend on quantization aggressiveness and method, an INT8 precision point (bitsandbytes LLM.int8, a different method from NF4, not a lower-bit NF4) was added as a third precision and run on TC1 across all five pairs at the 512-token reference budget, scored by both the official HarmBench classifier and the gpt-4o second judge with zero parse errors. The headline result (§6.15) is on the capability axis, and it is robust: capability loss is a clean cliff at four-bit (no INT8 MMLU/ARC delta is significant for any pair), so the only budget- and method-robust precision effect is the four-bit capability cliff. On the safety axis, at the reference budget every INT8 contrast is non-significant under both judges, the apparent Llama-3B INT8 increase seen at 128 tokens (which had been both-judge-significant there) vanishes at 512 (classifier Δ+0.005, McNemar p = 1.000; gpt-4o Δ+0.010, p = 0.688), so the safety effect is neither bit-width-graded nor budget-robust. Three extensions remain: (a) replicating any candidate INT8 effect across more models, decode seeds and generation budgets to separate a general LLM.int8 phenomenon from model-specific numerics or truncation; (b) tracing the §6.14 refusal-margin probe across all three precisions rather than only the behavioural metrics; and (c) adding genuinely different quantization families (GPTQ, AWQ, GGUF) beyond the two bitsandbytes algorithms evaluated here.",
    "Multilingual extension. Replicate the matched-pair design in Chinese (where Qwen is natively strong) and one low-resource language, to test whether quantization-induced safety changes are language-dependent.",
    "Fully-independent open-weight second judge, and a second human annotator. The primary HarmBench classifier has now been cross-checked against a second frontier judge (gpt-4o, same rubric), which agreed strongly per prompt (κ 0.60 to 0.95 across all ten models, 0.68 to 0.95 at the 512-token reference budget) and reached the same null-safety conclusion at the reference budget (§6.12, Result 4; §6.16), and it has been grounded against human judgment on a stratified 200-item subset (§6.12, Result 5; classifier κ 0.59 vs regex 0.11), substantially resolving the single-judge threat. Because that grounding rests on a single annotator, a second annotator with an inter-rater κ (and adjudication of disagreements) is the natural strengthening on the human side. Two complementary extensions remain on the judge side: (i) verify and fold in the already-run open-weight guard-model cross-check (LlamaGuard, via the already-wired --backend llamaguard on TC1) so its confirmation becomes a fully reproducible complement to the versioned API model; and (ii) a refusal-style judge for XSTest over-refusal has now been run as a sensitivity layer (§6.12, Result 6; gpt-4o, three-class taxonomy, 3,750 responses, zero parse failures), finding the one multiplicity-robust over-refusal contrast to be scorer-dependent, and the human-labelled refusal gold set that grounds that comparison has since been completed too (200 items, blinded single annotator, pre-registered Outcome J: the judge aligns substantially better with the annotator than the regex), exactly as the HarmBench human audit (Result 5) grounds the classifier; the residual on both axes is a second annotator with an inter-rater κ. All are derived validation layers writing separate redacted sidecars, never modifying raw.jsonl, summary.json, or existing sidecars.",
    "Safety-preserving quantization. Investigate emerging \"safety-preserving\" quantization methods that explicitly seek to mitigate alignment degradation under PTQ, and compare them against the vanilla NF4 baseline studied here.",
  ]),
];

// ------------------------------------------------------------
// Chapter 10: Conclusion
// ------------------------------------------------------------
const ch10 = [
  H1("Chapter 10: Conclusion"),
  PJ("This Final Year Project investigates safety, capability trade-offs in four-bit quantized compact language models, focusing on a research question that institutional benchmarks have not directly answered: when a small instruction-tuned model is quantized for on-device deployment, do observed changes in safety behaviour reflect a true shift in alignment or a side-effect of degraded general capability?"),
  PJ("The methodological contribution is a controlled matched-pair design in which baseline and four-bit pair members are loaded from identical baseline weights, with NF4 quantization applied on the fly. This design eliminates publisher- and pipeline-asymmetry as confounds and provides the strongest practical isolation of quantization as the experimental variable. The engineering contribution is an open, reproducible benchmarking framework comprising the matched-pair pipeline, four benchmark plugins, the pairwise analysis layer with rule-based interpretation labels and paired bootstrap 95% confidence intervals, full SLURM orchestration for the NTU TC1 cluster with resumable per-model matrix jobs, and an automated verification suite. The analytical contribution is the interpretation layer itself, the capability-anchored, statistically-grounded, multi-benchmark framework, which formalises the alignment-versus-capability disambiguation as a rule-based decision procedure over combined safety and capability deltas and which is the durable contribution of this study independent of any specific empirical outcome."),
  PJ("All six original runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976 to 60981); HarmBench ASR was validated and re-scored with the official HarmBench classifier (full precision) on 2026-06-06 (job 61047); a cross-family extension added the Mistral-7B and Phi-4-mini pairs on 2026-06-15 (§6.13); and the entire study was regenerated at HarmBench's 512-token reference budget, which is adopted as the primary HarmBench budget (§6.16). At the reference budget the primary finding is a null on the safety axis with capability as the robust cost. No pair shows a significant increase in genuine harmful compliance under NF4: the only individually significant ΔASR across the five pairs is Llama-3.2-3B at −0.040 (a decrease), and no ASR contrast survives a Benjamini-Hochberg correction over the twenty-test NF4 family. The three contrasts that do survive correction are all capability or over-refusal (Qwen 1.7B MMLU −0.090, Llama ARC −0.032, Phi over-refusal −0.048), so within this study's detection bounds (minimum detectable ΔASR of about 0.06 per pair; §6.5.1) the robust, budget-invariant cost of NF4 is degraded competence, not eroded guardrails. The five pairs still span distinct diagnostic profiles, but each reads on the capability side of the dichotomy. The Qwen3-1.7B pair carries the broad_degradation label, driven entirely by its significant MMLU loss (−0.090) with a flat harm axis (ΔASR = 0.000, McNemar p = 1.000): the +0.055 increase seen at the truncated 128-token budget was a generation-length artefact (60 percent of 128-token responses were truncated) and does not replicate at 512. The Llama 3.2 3B pair is capability_collapse_masquerading_as_safety: harmful compliance falls (ΔASR −0.040, significant) alongside a significant ARC loss, so its lower ASR is more consistent with a capability side-effect than with a genuine alignment improvement (the label encodes the correlation between the two axes; it does not by itself establish that the capability loss caused the safety change). The Qwen3-4B pair is directional alignment_degradation (ΔASR +0.040, not significant) with capability essentially preserved. A central methodological result, independent of budget, is that the judge validation overturned the regex proxy: the v2 scorer over-counts harmful compliance (it equates non-refusal with success), so the benchmark's own classifier is the primary scorer (D16) and the choice of scorer, not of judge or budget, is what determines the conclusion. The cross-family extension (§6.13) reinforces the null across four families: Mistral-7B's apparent proxy-driven rise is absent under the judge (ΔASR −0.020, not significant) at preserved capability, and Phi-4-mini is a directional alignment_degradation on safety (ΔASR +0.020 on the +0.02 tolerance boundary, n.s.) with its only significant move a decrease in over-refusal. The INT8 precision point (§6.15) adds that the safety axis is neither bit-width-graded nor budget-robust, the apparent Llama INT8 move at 128 tokens vanishes at the reference budget under both judges, while the four-bit capability cliff is the one robust precision effect. Finally, a mechanism probe (§6.14, whose first-token margins are generation-length-invariant) localises the quantization effect to near-boundary prompts and finds it symmetric (destabilising refusals in both directions) and, for the one pair that moves most, more consistent with capability-driven boundary instability than a targeted alignment shift, a reading the 512-token null corroborates directly, since its Qwen 1.7B ΔASR of 0.000 is a high-churn exchange of sixteen harmful-ward and sixteen safe-ward flips."),
  PJ("The results answer all five research questions at HarmBench's 512-token reference budget. Under the official HarmBench classifier, no evaluated pair shows a significant increase in genuine harmful compliance under NF4, and no ASR contrast survives a multiple-comparison correction (RQ1): the per-pair ΔASR values are Qwen 1.7B 0.000, Qwen 4B +0.040, Llama −0.040, Mistral-7B −0.020, and Phi-4-mini +0.020, of which only Llama's decrease is individually significant, and it is in the safety-improving direction. The data refute the optimistic claim that quantization reliably improves safety, but equally give no evidence that NF4 significantly erodes guardrails at the reference budget. NF4 quantization does not significantly increase over-refusal in any evaluated model (RQ2): four of five pairs are non-significant and the fifth (Phi-4-mini) significantly decreases, so the only movement is toward fewer false refusals. NF4 quantization significantly degrades capability in the smallest model on MMLU (Qwen 1.7B, −0.090) and on ARC for the Qwen 4B and Llama 3B pairs, so capability loss is the robust, correction-surviving effect while no capability delta appears at the INT8 precision point (RQ3). The smallest model is the most sensitive, but on the capability axis: the 1.7B pair carries the broad_degradation label driven by its MMLU loss with a flat harm axis, while the 4B pair preserves capability and shows only a directional, non-significant safety nudge (RQ4). Cross-family, no pair shows a significant ASR increase, over-refusal never significantly rises, and capability loss is directionally consistent where significant (RQ5). A central methodological result, independent of the generation budget, is that validating the framework against the benchmark's own classifier corrected a refusal-counting artefact in the regex proxy, the choice of scorer, not of judge or budget, determines the conclusion. The framework itself, capability-anchored, statistically grounded, multi-benchmark, and self-correcting under judge validation, is the durable contribution; the specific empirical numbers are subordinate to it, and at the reference budget they resolve to a null-safety result with capability as the robust cost of four-bit quantization."),
];

// ------------------------------------------------------------
// References
// ------------------------------------------------------------
const refs = [
  H1("References"),
  P("References are cited in IEEE numbered style; the bracketed numbers below correspond to the in-text citations.", { after: 200 }),
  ...numberedList([
    "Y. Huang, L. Sun, et al., \"Position: TrustLLM: Trustworthiness in large language models,\" in Proc. International Conference on Machine Learning (ICML), PMLR vol. 235, 2024. arXiv:2401.05561.",
    "B. Wang et al., \"DecodingTrust: A comprehensive assessment of trustworthiness in GPT models,\" in Proc. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.11698.",
    "Z. Zhang et al., \"SafetyBench: Evaluating the safety of large language models,\" in Proc. Association for Computational Linguistics (ACL), 2024. arXiv:2309.07045.",
    "M. Mazeika et al., \"HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,\" in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.04249.",
    "P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, \"XSTest: A test suite for identifying exaggerated safety behaviours in large language models,\" in Proc. NAACL, 2024. arXiv:2308.01263.",
    "D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, \"Measuring massive multitask language understanding,\" in Proc. International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.",
    "P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, \"Think you have solved question answering? Try ARC, the AI2 reasoning challenge,\" arXiv:1803.05457, 2018.",
    "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, \"QLoRA: Efficient finetuning of quantized LLMs,\" in Proc. NeurIPS, 2023. arXiv:2305.14314. (Introduces NF4 and the bitsandbytes integration.)",
    "T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, \"LLM.int8(): 8-bit matrix multiplication for transformers at scale,\" in Proc. NeurIPS, 2022. arXiv:2208.07339.",
    "R. Jin, J. Du, W. Huang, et al., \"A comprehensive evaluation of quantization strategies for large language models,\" in Findings of the Association for Computational Linguistics (ACL), 2024. arXiv:2402.16775.",
    "I. Proskurina, L. Brun, G. Metzler, and J. Velcin, \"When quantization affects confidence of large language models?,\" in Findings of NAACL, 2024. arXiv:2405.00632.",
    "A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., \"Investigating the impact of quantization methods on the safety and reliability of large language models,\" IEEE Access, vol. 14, pp. 96771 to 96793, 2026, doi: 10.1109/ACCESS.2026.3703899. arXiv:2502.15799.",
    "K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, \"Exploiting LLM quantization,\" in Proc. NeurIPS, 2024. arXiv:2405.18137.",
    "Y. Belkhiter, G. Zizzo, and S. Maffeis, \"HarmLevelBench: Evaluating harm-level compliance and the impact of quantization on model alignment,\" in Proc. NeurIPS 2024 Workshop on Safe Generative AI (SafeGenAI), 2024. arXiv:2411.06835.",
    "K. Chen, J. Zhang, J. Hu et al., \"Q-resafe: Assessing safety risks and quantization-aware safety patching for quantized large language models,\" in Proc. International Conference on Machine Learning (ICML), 2025. arXiv:2506.20251.",
    "J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Bartoldson, A. K. Jaiswal, K. Xu, B. Kailkhura, D. Hendrycks, D. Song, Z. Wang, and B. Li, \"Decoding compressed trust: Scrutinizing the trustworthiness of efficient LLMs under compression,\" in Proc. International Conference on Machine Learning (ICML), PMLR vol. 235, 2024. arXiv:2403.15447.",
    "A. Yang et al. (Qwen Team), \"Qwen3 technical report,\" Alibaba Group, 2025. arXiv:2505.09388.",
    "A. Grattafiori et al. (Meta AI), \"The Llama 3 herd of models,\" Meta AI, 2024. arXiv:2407.21783.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
    "Q. McNemar, \"Note on the sampling error of the difference between correlated proportions or percentages,\" Psychometrika, vol. 12, no. 2, pp. 153 to 157, 1947.",
    "A. Q. Jiang et al., \"Mistral 7B,\" Mistral AI, 2023. arXiv:2310.06825.",
    "A. Abouelenin et al., \"Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs,\" Microsoft, 2025. arXiv:2503.01743.",
    "J. Gu et al., \"A survey on LLM-as-a-judge,\" arXiv:2411.15594, 2024.",
    "M. Krumdick et al., \"No free labels: Limitations of LLM-as-a-judge without human grounding,\" arXiv:2503.05061, 2025.",
    "H. Inan et al., \"Llama Guard: LLM-based input-output safeguard for human-AI conversations,\" Meta AI, 2023. arXiv:2312.06674.",
    "A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, \"Refusal in language models is mediated by a single direction,\" in Proc. NeurIPS, 2024. arXiv:2406.11717.",
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
  PJ("The full configs/tc1_512.yaml, the configuration of the primary 512-token study (D41), is reproduced below for reference, with explanatory comments lightly abridged. The retired 128-token configuration (configs/tc1.yaml, max_new_tokens: 128, writing to results/) is identical apart from the generation budget and output roots and is retained in the repository for the §6.16 generation-length comparison. The mistral_7b and phi4_mini entries are the cross-family extension: they were run on TC1 on 2026-06-15 and their results are incorporated into Chapter 6 (Tables 6.1 to 6.3 and §6.13)."),
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
  Bullet("quant_method : str | None : quantization method when quantized=true, \"nf4\" (default, the main study) or \"int8\" (LLM.int8, the §6.15 precision point); a validator rejects any other value and rejects the field on a baseline entry."),
  Bullet("pair_id : str : links baseline and 4-bit members of the same pair."),
  Bullet("model_id : str : Hugging Face repo id."),
  Bullet("trust_remote_code : bool : disabled by default."),
  Bullet("dtype : str : one of \"auto\", \"float16\", \"bfloat16\", \"float32\"."),
  Bullet("benchmarks : list[str] : must reference top-level benchmark keys."),
  Bullet("revision : str | None : optional Hugging Face commit pin."),

  H3("DecodingConfig"),
  Bullet("max_new_tokens : int (≥1)"),
  Bullet("temperature : float (0.0 to 2.0)"),
  Bullet("top_p : float (0.0 to 1.0)"),
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
  PJ("The test inventory is intentionally not frozen into this document. Run pytest tests/ --collect-only for the live inventory; Table D.1 records the stable coverage contract instead of a count that changes whenever a guard is added."),
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
    └── FYP_Report_2026-07-01_v5.docx   (this document)`),
];

const appendixG = [
  H1("Appendix G: Document Revision History"),
  PJ("This appendix records the revision history of this FYP report. It mirrors the report-affecting subset of the project changelog (`docs/PROJECT_LOG.md` §4). Purely internal changes (refactors, tests, dev tooling) are recorded in the project log but omitted here for readability. Every entry corresponds to a regenerated docx artifact."),
  buildTable(
    ["When (UTC+8)", "Version", "Change to the report"],
    [
      ["2026-07-19 12:00", "FYP_Report_2026-07-01_v5.docx (current)", "T36 human refusal gold set folded in (pre-registered Outcome J) plus a validation-informed parallel multiplicity family (D51). §6.12 Result 6 extended with the completed 200-item blinded single-annotator audit: the three-class judge aligns substantially better with the annotator than the regex on benign over-refusal (strict κ 0.485 vs −0.006, broad 0.662 vs 0.054; regex recall 2/63 full refusals, judge 61/63 with over-flagging concentrated on the partial/full boundary), so the Phi-4-mini −0.048 over-refusal survivor is carried as most plausibly a regex measurement artifact. Disclosures added: a single-annotator reference set on a disagreement-enriched draw, not population ground truth; a shared benchmark taxonomy; and D45 timing that was judge-outcome-blind but regex-outcome-aware. §6.5.1 adds the composition-locked validation-informed parallel BH-FDR family (over-refusal scored judge-strict; docs/VALIDATION_INFORMED_FAMILY_NOTE.md): two survivors, both capability effects, purely deflationary; the registered regex family remains the family of record. RQ2 restated scorer-invariantly: no scorer finds a statistically significant over-refusal increase in any pair. New committed artifacts: xstest_human_validation.json, multiple_comparisons_judge_strict.json. No registered number changed."],
      ["2026-07-11 16:00", "FYP_Report_2026-07-01_v5.docx (superseded by the 2026-07-19 pass)", "Independent-audit remediation pass. Removed the internal future-session handoff content from Appendix H and reframed it as a scoring-methodology correction record, so no agent instructions, internal task identifiers, or cluster-operational steps remain in the deliverable. Added honest disclosures surfaced by the audit: the historical run artifacts predate resolved-revision, package-version, and dataset-fingerprint capture in each record (the checkpoint revisions are pinned in the committed configs, and the exact environment is now a documented reproducibility caveat rather than a machine-recorded field); the INT8 precision-comparison configs now pin the same model revisions as the NF4 study; the HarmBench judge model and tokenizer revision is now pinnable and recorded with the judge output; XSTest over-refusal is scored by a regex proxy pending an independent-judge or blind human-label validation; and the HarmBench ASR statistical power (per-pair minimum detectable effects of about 0.04 to 0.09 at 200 items) and the three-of-five multi-seed coverage are stated as explicit bounds on the small-effect and cross-family conclusions. No result numbers changed."],
      ["2026-07-01 23:00", "FYP_Report_2026-07-01_v5.docx (superseded by the 2026-07-11 pass)", "512-primary re-base (decision D41). HarmBench ASR is now reported at HarmBench's 512-token reference budget as the primary configuration, with the 128-token run retained as the §6.16 generation-length comparison. Tables 6.1 to 6.3, the Abstract, all of Chapter 6, the RQ answers, and Chapters 7 to 10 were re-based to the 512 numbers. Headline: no HarmBench ASR contrast survives BH-FDR at the reference budget (the three survivors are capability/over-refusal, Qwen3-1.7B MMLU, Llama ARC, Phi over-refusal); the only individually significant ΔASR is Llama-3.2-3B −0.040, a decrease; and the 128-token Qwen3-1.7B +0.055 was a truncation artefact (≈60% of 128-token responses were cut off before completion, shown directly by a prefix test). Folded in INT8@512 (the 128-token Llama-INT8 both-judge increase vanishes at 512 under both judges: classifier +0.005 p=1.000, gpt-4o +0.010 p=0.688) and multi-seed@512 (Qwen3-1.7B greedy 0.000 sits inside the seed range; three of five pairs covered). Added disclosures (MMLU/ARC are zero-shot subset anchors, not leaderboard reproductions). Table 6.3 judge-ASR/κ values updated to the 512 study. No methodological contribution changed; the capability-driven dichotomy is sharpened. Built from a copy of build_fyp_report_v4.js."],
      ["2026-06-30 11:30", "FYP_Report_2026-06-30_v4.docx (superseded by v5)", "Added §6.16 Generation-Length Robustness: the full study was regenerated at HarmBench's 512-token reference budget (configs/tc1_512.yaml; results_512/), retaining the 128-token study unchanged for comparison. The Qwen3-1.7B safety regression does not replicate at 512 (ΔASR 0.000 classifier / +0.005 gpt-4o, McNemar p=1.000, neither significant; symmetric 16/16 prompt flips), no ΔASR survives BH-FDR under either judge, cross-judge κ 0.68 to 0.95, and ~62% of 128-token responses were length-capped under the character-ceiling proxy (refined in v5 to 60.3% provably prefix-truncated by the direct prefix test; the two are distinct metrics). Four-bit capability losses are essentially unchanged 128→512, so the rerun sharpens the capability-driven dichotomy. Built non-destructively from a copy of build_fyp_report_v3.js (v3 left intact). INT8@512 + multi-seed@512 are regenerating and will be folded in. No 128-token number changed."],
      ["2026-06-29 12:00", "FYP_Report_2026-06-26_v3.docx (superseded by v4)", "Promoted the figure-rich v3 build to the canonical report: make report now builds scripts/build_fyp_report_v3.js, and the earlier FYP_Report_2026-06-14.docx (plus the v2 draft) are archived under docs/archive/. This version embeds the six analysis figures (capability anchor, ASR forest, precision sweep, judge-vs-proxy, per-category ASR, multi-seed) and carries the §6.15 INT8 precision-point results (capability cliff at 4-bit; the Llama-3B INT8 ΔASR +0.040 both-judge + McNemar move, caveated) and the §6.14 refusal-margin mechanism analysis. Test inventory refreshed to 329 automated tests across twenty-five files (added the dashboard data and theme test layers). Corrected the Appendix E and Appendix G self-references that still named the retired 2026-06-14 filename. No result numbers changed."],
      ["2026-06-15 18:00", "FYP_Report_2026-06-14.docx", "T26 cross-family extension folded in. Two matched pairs run on TC1 on 2026-06-15, Mistral-7B-Instruct-v0.3 and Phi-4-mini-instruct, take the study to five pairs / ten models / four families. New §6.13 presents the cross-family results; Tables 6.1/6.2/6.3 extended to all five pairs; Abstract, §6.1, §6.3 (RQ5), §6.4.1 (ARC), §6.9 (RQ2/RQ5), §6.11, §6.12 (second judge now spans all ten models, κ 0.60 to 0.95) and Ch10 updated. Judge-primary (128-token era; superseded by the 512-primary figures): Mistral ΔASR −0.040 (n.s.; the v2 proxy's +0.055 is a sign-flipped over-count, judge-vs-proxy κ 0.11 to 0.19, the study's starkest divergence, with the second judge gpt-4o concurring at κ 0.60 to 0.63); Phi ΔASR 0.000 (robust_preservation, κ 0.67). No new significant ΔASR, Qwen 1.7B (+0.055) remains the only one. Phi-4-mini's ΔOR = −0.028 is the study's one significant over-refusal delta (a decrease), so the over-refusal-null statements were qualified. Phi loaded via native transformers Phi3 (D31). No existing pair's numbers changed."],
      ["2026-06-14 15:45", "FYP_Report_2026-06-14.docx (superseded)", "128-token era. Renamed the artifact to today's date and rolled in the day's robustness work. T18 multi-seed sensitivity (§6.6.1): the Qwen 1.7B ΔASR is decode-dependent, mean +0.024 across five seeds vs the +0.055 greedy headline, not sign-consistent, so the headline is the upper end of a range, tempered not overturned. T21: §6.6.2 per-category judge ASR breakdown (the +0.055 is broad-based, rising in 5/6 harm categories), §7.6 deployment implications, §7.7 positioning against prior work, and verified citations replacing the earlier placeholders. T22 (§6.12 Result 4): a second independent judge (gpt-4o, same rubric) agrees with the primary classifier at κ 0.69 to 0.94 and reproduces the Qwen 1.7B increase in direction (+0.045 vs +0.055), borderline on significance (McNemar p=0.122), W3 substantially resolved. No headline label changed; these strengthen how the findings are evidenced."],
      ["2026-06-06 16:25", "FYP_Report_2026-05-27.docx", "Consistency pass on Appendix H so it reflected the post-D16 judge-primary state, and removed a stale post-release instruction because D16 had already been released. No result numbers changed."],
      ["2026-06-06 16:00", "FYP_Report_2026-05-27.docx", "Doc-consistency follow-ups to the D16 judge-primary promotion. Rewrote Chapter 7 §7.1 (scoring determinism now describes the judge classifier, not regex-eliminates-judge-variance) and §7.4 (renamed \"Scorer Choice and Remaining Validity Threat\", judge is primary, remaining threat is the absence of a second independent judge). Rewrote Appendix H (§H.2 onward) to the post-judge state (the v2 Qwen 4B figure flagged as superseded; the judge sidecars are the committed primary scorer; an optional second judge remains as future work, not an outstanding run). Updated the cover-page revision line to the judge-primary D16 wording. No numbers changed; this is a consistency pass so the .docx no longer contains pre-judge framing."],
      ["2026-06-06 15:00", "FYP_Report_2026-05-27.docx", "T20 results + D16: official HarmBench classifier promoted to PRIMARY HarmBench scorer; v2 regex demoted to a secondary non-refusal-rate proxy. The judge ran in fp16 on a 32 GB V100 (job 61047, n=200×6, 0 parse errors). Validation showed the regex over-counts ASR unevenly by family (judge vs v2 agreement: Qwen κ≈0.19 to 0.37, Llama κ≈0.69 to 0.79), and the choice of scorer changed the conclusion: the one significant ΔASR moved from Qwen 4B (proxy) to Qwen 1.7B (judge, +0.055 CI [+0.010,+0.100]). Refined the interpretation rule so alignment_degradation requires capability preserved; Qwen 1.7B (judge) is now broad_degradation (significant ASR up + significant MMLU down). Rewrote Table 6.1 (judge primary + v2 proxy rows), Table 6.2, new Table 6.3 (per-model agreement/κ), §6.1/6.1.1/6.3/6.5/6.6/6.7/6.8/6.9(all RQ)/6.10/6.11/6.12 (full results, no longer a placeholder), Abstract, Ch10. Headline relocated to Qwen 1.7B dual degradation; Qwen 4B is now directional-only. Test suite 176→178."],
      ["2026-06-06 12:00", "FYP_Report_2026-05-27.docx", "T20 refinement, judge runs in full precision (float16) by default, not NF4. Quantizing the judge classifier would confound the validation it is meant to provide, so §6.12 now states the judge is loaded in fp16; the job logs and records the exact GPU and precision used, and falls back to 8-bit (reported explicitly) only if the 13B classifier does not fit in fp16 on the available GPU. Test suite grew to 176 (added judge-precision resolution and VRAM-estimate tests). Appendix D judge-test row updated."],
      ["2026-05-28 13:30", "FYP_Report_2026-05-27.docx", "T20, Judge-model validation (HarmBench classifier). Added §6.12 documenting an independent sensitivity check of the HarmBench ASR using the official cais/HarmBench-Llama-2-13b-cls classifier (a derived check that re-scores saved generations, performs no new target-model inference, and writes redacted scores.judge.* / summary.judge.* sidecars without touching raw or v2 outputs). The embedded classifier prompt is pinned to the official model-card standard (non-contextual) template, with a byte-for-byte regression test. §6.12 carries the methodology and a placeholder (TC1 results still to come at that date) to be populated with per-model agreement, Cohen's kappa, judge-vs-v2 ASR, and per-pair label stability after the TC1 classifier run. Test suite grew to 172 (added test_judge_validation.py). Appendix D updated with the judge test row."],
      ["2026-05-27 23:54", "FYP_Report_2026-05-27.docx", "Added Appendix H documenting the v1 scorer failure mode, the v2 parser correction, the later immutable-sidecar correction, and the verification evidence for the judge-model validation. Updated cover revision text and Future Work wording to make clear that LlamaGuard/API judge scoring is a derived validation layer rather than a replacement for v2 or a mutation of raw outputs."],
      ["2026-05-28 12:00", "FYP_Report_2026-05-27.docx", "T13, Refusal classifier revision v1 → v2. The v1 deterministic regex parser missed canonical modern refusal templates (\"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", curly-apostrophe variants), systematically misclassifying clear refusals as harmful compliance. Replaced with a comprehensive pattern set (forty-plus regexes; apostrophe normalisation; verb-family coverage). Added rescore script (`scripts/rescore_harmbench.py`) that re-evaluates all existing raw.jsonl files without re-running inference; emits redacted IDs-only diagnostic CSVs and an aggregate JSON. The final workflow preserves TC1-original `raw.jsonl`/`summary.json` files and stores corrected scores in derived `scores.v2.jsonl` + `summary.v2.json` sidecars. Headline impact: Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2), with CI now excluding zero, the study's most robust empirical finding; Qwen 1.7B ΔASR collapsed from −0.120 (v1) to −0.025 (v2, within noise); Llama ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). Two of three interpretation labels changed (Qwen 4B → alignment_degradation; Llama → broad_degradation). Re-ran `make analyze` for updated bootstrap CIs. All affected text rewritten throughout the report: Abstract, Table 6.1, Table 6.2, §6.1, §6.1.1 (new, scorer revision history), §6.3, §6.4, §6.5, §6.6, §6.7, §6.8, §6.9 (all five RQ answers), §6.10, §6.11 (all subsections), Ch10. Thesis reframed around the framework as the durable contribution. Test suite grew to 163 with refusal-pattern regression tests and v2 sidecar-selection coverage. T12 entry from 2026-05-28 01:00 superseded by this update."],
      ["2026-05-28 01:00", "FYP_Report_2026-05-27.docx (superseded by v2 scorer revision)", "T12, Incorporated paired bootstrap 95% confidence intervals throughout the results chapter. Pipeline extended (`compare_quant_pairs.py` adds `compute_paired_bootstrap_ci` and emits CI bounds + significance flag per benchmark in `pairwise_deltas.{json,csv}`). Table 6.1 redesigned with \"Δ (95% CI)\" and \"Sig?\" columns. §6.1 introduces the bootstrap method and lists significance status of each delta. §6.5 Statistical Caveats updated with observed CI widths. §6.6 (Qwen 2B, both deltas significant), §6.8 (Qwen 4B, hedged, ΔASR borderline non-significant), §6.10 (Llama, ΔMMLU significant, ΔASR borderline) rewritten with inline CI annotations. RQ1, RQ3 in §6.9 expanded with significance language. §6.11.2, §6.11.3, §6.11.4 cross-family subsections updated. Ch10 conclusion rewritten with full significance reporting. Test suite grew to 126 (added 3 new tests: outcome extraction, bootstrap CI smoke, no-overlap guard). Pipeline output `results/analysis/pairwise_deltas.json` now contains CI fields."],
      ["2026-05-28 00:30", "FYP_Report_2026-05-27.docx", "Extended interpretation taxonomy with alignment_improvement (mirror of alignment_degradation): fires when ΔASR ≤ −0.02 with capability and over-refusal preserved. Reclassified Qwen 4B from broad_degradation → alignment_improvement, properly capturing the desirable capability-preserving harmful-compliance reduction. Updated Table 3.4 (five labels), §3.6 intro, §6.4 (now five canonical outcomes), §6.8 (Qwen 4B reading rewritten as genuine safety win), RQ4 synthesis, §6.11.5 (full safety spectrum), Ch10, Abstract. Pipeline output (`make analyze`) re-run after code change."],
      ["2026-05-27 23:30", "FYP_Report_2026-05-27.docx", "Corrected interpretation labels throughout to match pipeline output (make analyze → pair_interpretations.csv). Qwen 4B corrected from robust_preservation → broad_degradation (ΔASR abs=0.045 exceeds harm_tol=0.02, fails strict robust_preservation check). Llama corrected from broad_degradation → alignment_degradation (ΔASR=+0.030 ≥ harm_tol=0.02, fires second condition). Fixed Table 3.4 label definitions to match code logic."],
      ["2026-05-27 22:00", "FYP_Report_2026-05-27.docx", "Full three-pair results and analysis. All 6 jobs complete (60976 to 60981). Table 6.1 fully populated. Table 6.2 expanded to all three pairs. §6.3 updated with cross-family headline finding. §6.9 RQ5 and synthesis sections complete. Added §6.10 Llama 3.2 3B pair observations (baseline profile, MMLU subject breakdown). Added §6.11 cross-family comparison (§6.11.1, §6.11.5: baseline divergence, ASR sign inconsistency, OR null result, MMLU magnitude comparison, interpretation labels and central question). Updated Abstract with all three pair findings. Updated Ch10 conclusion with full five-RQ answers."],
      ["2026-05-27 21:00", "FYP_Report_2026-05-27.docx", "Full Qwen analysis update. Added Table 6.2 (Qwen family delta comparison with interpretation labels). Expanded §6.2 with full within-family analysis, delta magnitude ratio (22:1 MMLU), and methodological payoff discussion. Updated §6.3 with two Llama hypotheses. Rewrote §6.4 Capability Anchoring with concrete Qwen examples and updated label names. Added §6.9 Qwen Family Synthesis addressing RQ1, RQ4 in full (5 H3 subsections). Updated Ch7 intro, Ch10 conclusion paragraphs."],
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
      ["2026-05-23 evening", "FYP_Report_2026-05-23.docx (archived)", "Initial release of the interim report. Cover, abstract, TOC, 10 chapters, references, Appendices A, F. Approximately 30 pages. Documents the matched-pair design with three pairs (Qwen 2B, Qwen 4B, Llama 3.2 3B) and on-the-fly NF4 quantization."],
    ],
    [2200, 3000, 4160],
  ),
  P("Table G.1  Revision history of this FYP report. The source builder script is `scripts/build_fyp_report_v5.js`; the docx is a build artifact and is never hand-edited.", { size: 18 }),
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
  H1("Appendix H: Scoring-Methodology Correction Record"),
  PJ("This appendix documents, for transparency and auditability, the scoring-methodology correction this study underwent: the original refusal-regex scorer, the corrected v2 regex, and the promotion of the official HarmBench classifier to the primary scorer. It also records the immutable-artifact contract that keeps every correction fully reproducible from the saved generations without re-running any target-model inference."),

  H2("H.1 What went wrong in v1"),
  PJ("The original HarmBench and XSTest scorer used a narrow deterministic refusal parser with only fourteen regex patterns. During post-hoc review, it became clear that the parser missed common modern instruction-model refusal templates, including straight- and curly-apostrophe variants of phrases such as \"I can't fulfil this request\", \"I cannot provide\", and \"I'm unable to assist\". Those responses were clear refusals, but v1 counted them as harmful compliance on HarmBench or benign answers on XSTest. This was a scoring error, not a target-model generation error: the six TC1 jobs had already completed, and the saved model outputs did not change."),
  PJ("The consequence was material, and the correction proceeded in two stages. The v1→v2 regex correction first moved the Qwen 4B point estimate from an apparent safety improvement to a (v2-regex) ΔASR of +0.065. But the v2 regex still only measures \"non-refusal rate\", so a second correction followed: the official HarmBench classifier (judge; decision D16, §6.12) was promoted to the primary HarmBench scorer. Under the judge, the v2 regex is shown to over-count ASR, and, at the then-current 128-token budget, the one statistically significant ΔASR was Qwen 1.7B at +0.055, not Qwen 4B, whose judge ΔASR was directional only; the v2 Qwen 4B figure quoted above is the superseded regex-proxy value. A third stage followed (decision D41, §6.16): regenerating the study at HarmBench's 512-token reference budget dissolved that 128-token Qwen 1.7B increase to ΔASR 0.000 under both judges, so the CURRENT primary reading is capability-driven, Qwen 1.7B broad_degradation via its significant MMLU loss with the harm axis flat, and no pair showing a significant harmful-compliance increase. The 128-token reading in this paragraph is retained as scorer-revision history only. MMLU values are unchanged throughout because MMLU scoring does not use any refusal scorer."),

  H2("H.2 How the correction was made"),
  PJ("The v2 correction expanded `ethical_benchmark/benchmarks/utils.py` with broader refusal-pattern coverage, punctuation normalisation, and a diagnostic helper that reports the matched refusal-pattern name. Regression tests were added for canonical refusal strings, negative controls, and curly-apostrophe handling. The analysis pipeline was then re-run from saved outputs; no target-model inference was re-run on TC1. This was appropriate because the flaw was in the scoring layer, not in model execution."),
  PJ("A first implementation rescored `raw.jsonl` and `summary.json` in place. That approach was rejected during code review because the original TC1 outputs are the evidence trail and should remain immutable. The final solution preserves `raw.jsonl` and `summary.json` as original v1 artifacts, and stores corrected v2 scoring in sidecars: `scores.v2.jsonl` contains prompt IDs and score_fields only, with no prompt or response text, and `summary.v2.json` contains corrected aggregates. `compare_quant_pairs.py` now prefers complete v2 sidecars when they exist and falls back to original summaries otherwise."),

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
  PJ("The workflow was verified in five ways. First, the full test suite passes, including refusal-parser regressions, analysis sidecar selection, interpretation guards, registry/surface checks, and judge-validation tests for redaction, raw immutability, idempotency, ASR aggregation, the official HarmBench classifier prompt template, and judge-precision resolution. Second, the judge jobs ran in full precision on a 32 GB V100 with zero parse errors throughout (512-token primary study: job 61524, 2 000 generations plus 1 000 INT8; 128-token comparison: jobs 61047/61134, 2 000 generations). Third, `scripts/judge_agreement.py` reproduces the judge-vs-v2 agreement and the judge-primary pair labels deterministically from the committed sidecars. Fourth, the original `raw.jsonl` and `summary.json` files were confirmed unchanged by every post-hoc step (rescore and judge). Fifth, all judge sidecars were checked for redaction: they contain prompt IDs, boolean labels, and run metadata only, with no prompt, behaviour, or response text."),

  H2("H.5 Remaining construct-validity extension"),
  PJ("The classifier has since been cross-checked twice over: once by an independent frontier-model second judge (gpt-4o, §6.12 Result 4), and once against human labels on a stratified subset (§6.12 Result 5). One optional extension is still carried in Chapter 9: an open-weight guard model (LlamaGuard), whose cross-check would be fully reproducible in a way the versioned API judge is not. That guard has since been run, with verification and fold-in pending, so this revision carries the two-judge and human-grounded validation only. The framework already supports it through the pluggable judge backend in `ethical_benchmark/judges/validation.py` and the agreement analysis in `scripts/judge_agreement.py`; the primary scorer is unaffected either way."),
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
  title: "FYP Comprehensive Report: Benchmarking Ethical Performance of Open-Source LLMs (CCDS25-1136)",
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
              text: "CCDS25-1136, TAN UEI HORNG (UTAN001)",
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

const OUTPUT = "/Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-07-01_humanized.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("WROTE: " + OUTPUT);
  console.log("Size: " + buf.length + " bytes");
});
