// FYP Report builder — Word document via docx-js
// Output: /Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-05-27.docx

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink,
  TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak,
} = require("docx");

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
  // Legacy single-list helper. Prefer numberedList(items) below — that
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
      text: "A Study of Safety–Capability Trade-offs in 4-bit Quantized Compact Language Models",
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
    children: [new TextRun({ text: "Document date:  27 May 2026", font: FONT, size: 26, bold: true })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Revision: incorporates v2 scorer correction and immutable sidecar handoff", font: FONT, size: 20, italics: true, color: "555555" })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------
// Abstract
// ------------------------------------------------------------
const abstract = [
  H1NoBreak("Abstract"),
  PJ("Compact instruction-tuned language models in the one-to-four-billion-parameter range are increasingly deployed on edge devices and consumer hardware, where four-bit quantization has become the de facto compression method for fitting them into available memory. However, quantization is not behaviourally neutral: emerging evidence suggests it can alter safety alignment, refusal calibration, and general capability in non-trivial ways. Existing large-scale ethical benchmarking efforts, such as TrustLLM, DecodingTrust, and SafetyBench, primarily evaluate full-precision mid-to-large models and do not systematically address how compression interacts with safety in the compact-deployment regime."),
  PJ("This work designs and implements a research-grade benchmarking framework to study safety–capability trade-offs in four-bit quantized compact language models. The study adopts a matched-pair experimental design in which baseline and four-bit variants are loaded from the same underlying weights, differing only by the application of on-the-fly BitsAndBytes NF4 quantization at load time. This design choice eliminates publisher- and pipeline-asymmetry as confounds, isolating quantization itself as the sole experimental variable. Three model pairs are evaluated: Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B-Instruct as a cross-family robustness check."),
  PJ("Each pair is scored on three complementary benchmarks: HarmBench (harmful compliance, measured as Attack Success Rate), XSTest (over-refusal on benign prompts), and a curated MMLU subset (general capability). A deterministic, regex-based scoring pipeline ensures reproducibility and removes judge-model variance. Pairwise deltas are combined through a rule-based interpretation layer that distinguishes genuine alignment shifts from capability collapse masquerading as safety. All six experimental runs completed on the NTU TC1 GPU cluster on 2026-05-27. The primary methodological contribution is the capability-anchored interpretation framework itself: a deterministic, reproducible procedure for distinguishing alignment shifts from capability artefacts that survives independent of any specific empirical outcome. Applied to the three pairs, the framework yields: the Qwen3-1.7B pair is classified as capability_collapse_masquerading_as_safety (ΔMMLU = −0.087 with CI excluding zero; ΔASR = −0.025 within noise); the Qwen3-4B pair is classified as alignment_degradation, the only statistically significant ΔASR in the study (+0.065, CI [+0.025, +0.110]) at preserved capability; and the Llama 3.2 3B pair is classified as broad_degradation (ΔASR within noise; ΔMMLU = −0.043 with CI excluding zero). Llama's full-precision baseline is the most refusal-calibrated of the three (ASR = 0.060) and that calibration is preserved under quantization, contradicting the simple hypothesis that NF4 universally weakens alignment. Over-refusal increases marginally for all pairs after quantization but no ΔOR is statistically significant. Headline conclusion: the framework reliably separates the three failure modes, and within that framework the strongest single empirical finding is the statistically robust Qwen3-4B alignment_degradation under NF4 compression."),
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
  P("Table 3.1  Model pair selection matrix.", { after: 60 }),
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets.", { after: 60 }),
  P("Table 3.3  Decoding controls used during inference.", { after: 60 }),
  P("Table 3.4  Interpretation labels derived from combined deltas.", { after: 60 }),
  P("Table 4.1  Repository module responsibilities.", { after: 60 }),
  P("Table 4.2  CLI subcommands exposed by fyp_cli.py.", { after: 60 }),
  P("Table 5.1  TC1 cluster hardware and policy parameters.", { after: 60 }),
  P("Table 5.2  Software environment versions on TC1.", { after: 60 }),
  P("Code listing 5.1  Head-node pre-cache invocation (scripts/prefetch_tc1.py).", { after: 60 }),
  P("Table 6.1  Per-pair benchmark results for all three model pairs with bootstrap 95% CIs.", { after: 60 }),
  P("Table D.1  Distribution of automated tests across modules.", { after: 60 }),
];

// ------------------------------------------------------------
// Chapter 1 — Introduction
// ------------------------------------------------------------
const ch1 = [
  H1("Chapter 1 — Introduction"),

  H2("1.1 Background and Motivation"),
  PJ("The last two years have seen a rapid proliferation of compact instruction-tuned large language models (LLMs) in the one-to-four-billion-parameter range. Models such as Qwen 2.5 and 3.x, Llama 3.2, Microsoft Phi-3, and Google Gemma 2 have demonstrated that capable reasoning, instruction following, and multilingual performance are achievable at parameter counts that fit comfortably on consumer hardware, mobile chipsets, and edge accelerators. This has shifted the practical envelope of LLM deployment: tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, with stronger privacy guarantees, lower latency, and substantially reduced operational cost."),
  PJ("In practice, however, such models are rarely deployed in full precision. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision impractical for routine use, and quantization to four-bit precision has become the de facto compression standard for on-device inference. Lightweight runtimes such as llama.cpp and on-device agent frameworks routinely ship four-bit GGUF or BitsAndBytes NF4 checkpoints by default. End users encountering these models therefore almost always interact with a quantized variant, not the original baseline."),
  PJ("Quantization has historically been treated as a numerical optimisation technique whose primary cost is a small loss in perplexity or downstream accuracy. A growing body of evidence challenges this view. Quantization can alter behavioural properties of an LLM in ways that are not visible from perplexity alone, including instruction-following fidelity, refusal calibration, and resistance to adversarial prompts. Because safety alignment is itself a learned behaviour encoded in the model weights, any operation that alters those weights — even one that preserves task accuracy on average — has the potential to perturb that behaviour. Understanding how compression interacts with safety is therefore not optional: it is a prerequisite to safe deployment of any compact model that has been quantized for production use."),

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
  PJ("The distinction matters. A model that becomes more refusal-prone after quantization could plausibly be reported as \"safer\" by a benchmark that measures only harmful compliance. The same model, measured on benign prompts, may also refuse those — exhibiting over-refusal — and on capability benchmarks may answer fewer questions correctly. Without a joint evaluation of harmful compliance, over-refusal, and capability, the safety community risks systematically misattributing capability collapse as safety improvement. The problem statement above directly motivates the multi-axis evaluation framework adopted in this study."),

  H2("1.3 Research Questions"),
  PJ("The study is organised around five research questions. The first three are primary questions answered directly by the matched-pair experiments. The remaining two examine secondary dimensions: scale sensitivity within a family, and replication across families."),
  Bullet("RQ1 (Primary). Does four-bit NF4 quantization measurably change harmful-compliance behaviour, as measured by HarmBench Attack Success Rate, relative to matched baseline weights in compact instruction-tuned models?"),
  Bullet("RQ2 (Primary). Does four-bit quantization change over-refusal behaviour on benign prompts, as measured by XSTest?"),
  Bullet("RQ3 (Primary). Does four-bit quantization degrade general capability, as measured by accuracy on a curated MMLU subset?"),
  Bullet("RQ4 (Secondary, scale). Does the magnitude of the quantization effect differ between Qwen 2B and Qwen 4B within the same model family?"),
  Bullet("RQ5 (Robustness). Does the pattern observed within the Qwen family replicate qualitatively in a different model family, namely Llama 3.2 3B?"),

  H2("1.4 Scope and Boundaries"),
  PJ("Explicit scoping is essential to make the study tractable within the constraints of an undergraduate Final Year Project and to keep the resulting claims defensible. The following boundaries are stated up front."),
  Bullet("Quantization method. The study evaluates a single quantization path: BitsAndBytes NF4 four-bit loading applied on the fly via the Hugging Face transformers integration. Alternative methods such as GPTQ, AWQ, and GGUF are out of scope and are flagged in the Future Work chapter."),
  Bullet("Model regime. The study targets compact instruction-tuned models in the two-to-four-billion-parameter range. This regime corresponds to the practical envelope of contemporary on-device deployment."),
  Bullet("Languages and modality. Evaluation is restricted to English-language, text-only interactions."),
  Bullet("Decoding. All inference uses deterministic greedy decoding with temperature 0.0 and top-p 1.0, both to eliminate within-condition variance and to maximise comparability across baseline and quantized pair members."),
  Bullet("Benchmark axes. Rather than attempting a broad multi-axis ethical evaluation, which is already covered by larger institutional benchmarks, this study evaluates three complementary axes that jointly enable the alignment-versus-capability disambiguation that motivates the research question."),

  H2("1.5 Contributions"),
  PJ("The contributions of this project span experimental design, software engineering, and analytical methodology."),
  ...numberedList([
    "A controlled matched-pair study design that isolates quantization as the sole experimental variable. Both members of every pair are loaded from the same underlying model_id; the only difference is whether on-the-fly NF4 quantization is applied during the from_pretrained call. This eliminates publisher-asymmetry and conversion-pipeline asymmetry — two confounds that affect most published comparisons of quantized and full-precision checkpoints.",
    "An open, reproducible benchmarking framework consisting of approximately 2,800 lines of production Python and approximately 1,800 lines of automated tests. The framework provides deterministic regex-based refusal scoring, configurable per-benchmark sampling, batched generation with chat-template application, and full per-prompt audit logging.",
    "A rule-based interpretation layer that combines harmful-compliance, over-refusal, and capability deltas into one of five diagnostic labels: alignment degradation, alignment improvement, capability collapse masquerading as safety, robust preservation, or broad degradation. This layer formalises the alignment-versus-capability disambiguation in code.",
    "SLURM cluster orchestration for the NTU TC1 GPU cluster, including resumable per-model matrix jobs that load each model only once and execute all three benchmarks sequentially. Resume granularity is per-prompt, so jobs that exhaust walltime can be re-submitted without redundant computation.",
    "A verification suite of 163 automated tests across twelve test files, covering schema validation, dataset loaders, the refusal parser (with comprehensive regression tests for the v2 expanded patterns), SLURM job generation, matrix reuse behaviour, analysis logic, and v2 score-sidecar selection.",
  ]),

  H2("1.6 Report Structure"),
  PJ("Chapter 2 surveys the relevant literature on large-scale ethical benchmarking, the helpfulness–harmlessness trade-off, the deployment of compact LLMs, and the behavioural effects of quantization, concluding with the specific research gaps that this study targets. Chapter 3 details the experimental methodology, including the matched-pair design, quantization approach, benchmark selection, scoring, decoding controls, and the interpretation framework. Chapter 4 documents the system design and implementation: package structure, configuration schema, the model loader and generation pipeline, the benchmark plugin architecture, matrix orchestration, resume logic, and SLURM job generation. Chapter 5 describes the experimental setup on TC1 and the run plan. Chapter 6 presents the experimental results for all three model pairs; runs completed on the NTU TC1 cluster on 2026-05-27. Chapter 7 discusses threats to validity, Chapter 8 records limitations, and Chapter 9 proposes future work. Chapter 10 concludes. Six appendices reproduce the full configuration, an example SLURM script, the configuration schema, the test inventory, the repository layout, and a glossary."),
];

// ------------------------------------------------------------
// Chapter 2 — Literature Review
// ------------------------------------------------------------
const ch2 = [
  H1("Chapter 2 — Literature Review"),

  H2("2.1 Large-Scale Ethical Benchmarking of LLMs"),
  PJ("The past two years have produced several large-scale institutional efforts to benchmark the ethical performance of open-source LLMs. TrustLLM, DecodingTrust, and SafetyBench each provide a standardised pipeline that evaluates dozens of widely used models across multiple trustworthiness dimensions, including toxicity, bias, hallucination, robustness, fairness, and ethics. These benchmarks have published leaderboards that cover popular open-weight model families such as Llama, Mistral, Qwen, and Falcon, and have meaningfully shifted the conversation around what constitutes an acceptable safety profile in publicly released models."),
  PJ("The strengths of these frameworks are clear: large model coverage, multi-dimensional evaluation, and standardised methodologies that facilitate cross-model comparison. Their gaps, however, are also clear in the context of the present work. First, they predominantly evaluate full-precision checkpoints; quantized variants are either omitted or treated as a separate, secondary evaluation. Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, where alignment training tends to be most robust. Third, they typically report each safety axis independently — harmful compliance, bias, toxicity — without explicitly anchoring those measurements against a capability metric, leaving open the question of whether safety changes reflect alignment or capability."),

  H2("2.2 The Helpfulness–Harmlessness Trade-off"),
  PJ("A central, well-documented limitation of static safety benchmarks is their difficulty in jointly capturing the tension between helpfulness and harmlessness. A model that is heavily optimised for safety may become overly conservative, refusing benign prompts that incidentally resemble unsafe ones — a failure mode known as over-refusal or exaggerated safety. Conversely, a model optimised for helpfulness may remain vulnerable to adversarial jailbreak attacks. Measuring only one side of this trade-off produces an incomplete picture."),
  PJ("Two benchmarks have emerged as the practical standards for measuring opposite sides of this trade-off. HarmBench and AdvBench provide curated unsafe prompts and measure attack success rate — the fraction of unsafe prompts to which the model produces a harmful, complying response. XSTest provides benign prompts that are superficially similar to unsafe ones and measures over-refusal rate — the fraction of benign prompts the model nonetheless refuses. Evaluating both simultaneously is essential to detecting trade-offs introduced by alignment training or compression."),
  PJ("MMLU, the Massive Multitask Language Understanding benchmark, has become the de facto general-capability anchor in safety studies. By measuring multiple-choice accuracy across a broad spectrum of academic and professional subjects, MMLU provides a capability signal that is largely independent of refusal behaviour, allowing capability collapse to be detected even when safety metrics appear to improve."),

  H2("2.3 Small Language Models and On-Device Deployment"),
  PJ("Recent compact LLM releases have repeatedly demonstrated that strong reasoning, instruction following, and multilingual performance are achievable at parameter counts below four billion. The Qwen 2.5 and Qwen 3 series, Llama 3.2 (1B and 3B Instruct), Microsoft Phi-3 (3.8B), and Google Gemma 2 (2B and 9B) have each been positioned for on-device or edge inference. These models are routinely integrated into lightweight agent frameworks and consumer applications, where their compact size enables genuinely local execution."),
  PJ("The deployment reality of these models is that they are almost never used in full precision. Memory and latency constraints on consumer hardware drive routine use of four-bit quantization, often via on-the-fly BitsAndBytes loading or pre-quantized GGUF checkpoints. Safety claims attached to the unquantized release model therefore do not, in general, transfer to the model that end users actually encounter."),

  H2("2.4 Quantization and Behavioural Effects"),
  PJ("Quantization compresses model weights from higher-precision floating-point representations to lower-precision integer or normalized-float representations, reducing memory footprint and accelerating inference at the cost of some numerical fidelity. Post-training quantization (PTQ) methods apply this conversion after the model has been trained and require no fine-tuning, making them attractive for deployment."),
  PJ("Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA has become particularly prevalent in the open-source ecosystem. NF4 represents each weight using a four-bit normalized-float code optimised for the typical Gaussian-like distribution of neural network weights, with double quantization applied to the quantization constants to further reduce overhead. The accompanying BitsAndBytes library integrates directly with Hugging Face transformers, exposing four-bit loading through a single BitsAndBytesConfig object that can be passed to from_pretrained at model load time."),
  PJ("A body of recent work has argued that quantization is not behaviourally neutral. Studies across a wide range of model sizes have shown that smaller models can suffer severe degradation under aggressive four-bit quantization while larger models remain comparatively stable, and that instruction-following fidelity and hallucination rates can shift even when general benchmark scores appear preserved. Safety-specific studies have flagged that PTQ can degrade alignment behaviour, and an emerging line of research proposes safety-preserving quantization methods explicitly designed to mitigate these effects. Importantly, most existing quantization studies evaluate either perplexity or general capability, with safety considered (if at all) as a separate axis rather than jointly with capability."),

  H2("2.5 Research Gaps Targeted by This Work"),
  PJ("Four interlocking gaps in the existing literature motivate the present study."),
  ...numberedList([
    "Gap 1: Limited empirical study of compact (<4B) instruction-tuned models in the safety–quantization context. Most quantization studies focus on the seven-billion-to-thirteen-billion parameter range and above. Edge-deployment-relevant compact models are under-represented.",
    "Gap 2: Lack of integrated evaluation that measures harmful compliance, over-refusal, and capability simultaneously. Studies that measure only harmful compliance cannot detect capability-driven safety artifacts.",
    "Gap 3: Difficulty interpreting whether observed safety metric changes in quantized models reflect a real alignment shift or a side-effect of capability degradation. No widely adopted convention exists for jointly interpreting safety and capability deltas under compression.",
    "Gap 4: Provenance asymmetry in existing comparisons. Many published comparisons of full-precision and quantized checkpoints use a full-precision checkpoint from one publisher and a pre-quantized checkpoint from another, conflating quantization effects with checkpoint-conversion effects. A clean, on-the-fly quantization design from identical baseline weights eliminates this confound.",
  ]),
  PJ("The methodology described in Chapter 3 is structured to address all four gaps. The compact-deployment regime is addressed by the choice of Qwen 2B, Qwen 4B, and Llama 3.2 3B; the integrated evaluation is addressed by the three complementary benchmarks; the interpretation challenge is addressed by the rule-based interpretation layer; and the provenance asymmetry is eliminated by on-the-fly NF4 loading from the same baseline weights."),
];

// ------------------------------------------------------------
// Chapter 3 — Methodology
// ------------------------------------------------------------
const ch3 = [
  H1("Chapter 3 — Methodology"),

  H2("3.1 Experimental Design"),
  PJ("The study adopts a matched-pair comparative experimental design. Each model under study is evaluated as a pair: a baseline variant loaded in the default high-precision dtype, and a four-bit variant produced by applying BitsAndBytes NF4 quantization on the fly at load time. Crucially, both pair members are loaded from exactly the same Hugging Face model_id; no separately uploaded \"pre-quantized\" checkpoint is used. The only operational difference between the two members of a pair is the presence of a BitsAndBytesConfig object in the from_pretrained call."),
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
  PJ("Quantization is applied entirely on the fly by injecting a BitsAndBytesConfig object into the AutoModelForCausalLM.from_pretrained call. The configuration sets load_in_4bit to True, selects the NF4 quantization type, enables double quantization, and sets the four-bit compute dtype to follow the resolved torch_dtype of the model — that is, float16 on CUDA when dtype is set to auto, and float32 on CPU (although the CPU path is not used for quantized runs since BitsAndBytes requires a CUDA device)."),
  PJ("Four-bit loading is forbidden on a CPU runtime: the loader raises a clear error when quantized is true and the resolved runtime device is not cuda. After load, the loader verifies that the resulting model object reports is_loaded_in_4bit, logging a warning otherwise."),
  PJ("The choice of NF4 over alternative PTQ methods such as GPTQ and AWQ is deliberate. NF4 is the dominant on-the-fly quantization format in the open-source ecosystem, is supported natively by transformers without an offline calibration pass, and applies symmetrically to any model that can be loaded with from_pretrained. This applicability symmetry is required by the matched-pair design: the quantization step must be applicable to exactly the same weights as the baseline, without an intermediate calibration corpus that would itself become a confound."),

  H2("3.4 Benchmark Selection and Mapping"),
  PJ("Three benchmarks are used. Each measures one of the three dimensions required to disambiguate alignment shifts from capability collapse."),
  buildTable(
    ["Dimension", "Benchmark (source)", "Primary metric", "Sampling"],
    [
      ["Harmful compliance", "HarmBench (walledai/HarmBench, standard)", "Attack Success Rate (ASR)", "200 prompts"],
      ["Over-refusal", "XSTest (local canonical CSV, benign-only)", "Over-refusal rate", "250 benign prompts"],
      ["General capability", "MMLU (cais/mmlu, 6-subject subset)", "Accuracy", "300 questions total"],
    ],
    [1900, 3260, 2300, 1900],
  ),
  P("Table 3.2  Benchmark selection, primary metrics, and sample budgets. Sampling is deterministic and seed-controlled.", { size: 18 }),
  PJ("HarmBench is selected as the harmful-compliance benchmark because it is the most widely adopted adversarial-prompt collection in contemporary safety research and provides a stable definition of attack success. The walledai release on Hugging Face is used with the standard config. XSTest is the canonical over-refusal benchmark; this repository bundles the canonical XSTest v2 prompts CSV from paul-rottger/xstest and evaluates the benign-only subset to measure how often the model refuses prompts that should not be refused. MMLU is used as the general-capability anchor."),
  PJ("To keep MMLU tractable within the per-job walltime on TC1, a six-subject subset is used: business_ethics, clinical_knowledge, college_biology, high_school_world_history, high_school_macroeconomics, and human_aging. These subjects span ethics, sciences, humanities, social sciences, and applied knowledge, and are deliberately chosen to cover heterogeneous reasoning demands without exhausting the time budget. The total sample cap of 300 questions is distributed evenly across the six subjects."),

  H2("3.5 Scoring: Deterministic Refusal Detection"),
  PJ("All refusal scoring is performed by deterministic regex-based pattern matching rather than by a separate judge LLM. The motivation for this design choice is reproducibility and transparency. Judge-LLM scoring introduces a non-trivial variance budget (different judges and different judge versions disagree about whether a particular response constitutes a refusal) and adds an additional moving part to the experiment that would itself need to be controlled."),
  PJ("Three boolean fields are recorded for each response: harmful_compliance, safe_refusal, and malformed. For HarmBench, harmful compliance corresponds to a response that addresses the unsafe request substantively rather than refusing. For XSTest benign prompts, refusal is the failure mode, and over-refusal rate is the fraction of benign prompts on which the refusal pattern fires. The malformed field flags responses that are empty, contain only template fragments, or fail to parse into the expected output format."),
  PJ("The trade-off of deterministic parsing — its potential to miss nuanced or euphemistic refusals — is explicitly acknowledged in the Limitations chapter and motivates one of the proposed future-work directions."),

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

  H2("3.8 Interpretation Framework"),
  PJ("Combining harmful-compliance, over-refusal, and capability deltas, each pair receives one of five rule-based interpretation labels. These labels are the central analytical output of the study and formalise the alignment-versus-capability disambiguation. The taxonomy is symmetric around the safety axis: it includes both alignment_degradation (ASR up beyond threshold) and its mirror, alignment_improvement (ASR down beyond threshold while capability is preserved), so that genuine safety wins are not misclassified as fallback degradations."),
  buildTable(
    ["Label", "Condition", "Interpretation"],
    [
      ["alignment_degradation", "ΔASR ≥ +0.02 (harm_tol)", "Harmful compliance worsens beyond threshold — a direct alignment concern; checked before capability signals."],
      ["capability_collapse_masquerading_as_safety", "ΔASR ≤ −0.02  AND  ΔMMLU ≤ −0.03", "Apparent safety improvement coincides with capability collapse; not a genuine alignment win."],
      ["alignment_improvement", "ΔASR ≤ −0.02  AND  ‖ΔOR‖ < 0.02  AND  ΔMMLU > −0.03", "Harmful compliance falls beyond threshold while capability and refusal calibration are preserved — the mirror image of alignment_degradation and the most desirable outcome."],
      ["robust_preservation", "‖ΔASR‖ < 0.02  AND  ‖ΔOR‖ < 0.02  AND  ΔMMLU > −0.03", "All three deltas within tolerance — quantization preserves both safety and capability."],
      ["broad_degradation", "Fallback (none of the above conditions met)", "Mixed or compound degradation not captured by the more specific labels; typically signals OR moves outside tolerance, or ASR is within tolerance but capability drops."],
    ],
    [2400, 3500, 3460],
  ),
  P("Table 3.4  Interpretation labels derived from combined deltas. Numerical thresholds for the qualitative terms (\"roughly flat\", \"small\") are set as configurable parameters in the analysis module.", { size: 18 }),

  H2("3.9 Reproducibility Controls"),
  PJ("Reproducibility is treated as a first-class engineering requirement. A single seed (42) is propagated to Python's random module, NumPy, and PyTorch RNGs at the start of each run. Dataset shuffling is deterministic: each benchmark plugin loads its full dataset, shuffles with the seeded RNG, and then truncates to the configured max_samples. The same prompt order is therefore visited by both pair members."),
  PJ("Every per-prompt record persisted to raw.jsonl includes the model alias, the resolved model_id, the model family, the pair_id, the quantized flag, the seed, the full generation_config, and an ISO-8601 timestamp. This per-record metadata supports later auditability without dependence on the configuration file at the time of analysis."),
  PJ("Long-running jobs are made robust by the resume logic described in §4.7: if a job is killed mid-benchmark by walltime exhaustion or transient cluster failure, the next submission reads the existing raw.jsonl, identifies which prompt_ids have already been processed, and skips them on the second pass. The granularity of the resume mechanism is per-prompt, not per-benchmark."),
];

// ------------------------------------------------------------
// Chapter 4 — System Design and Implementation
// ------------------------------------------------------------
const ch4 = [
  H1("Chapter 4 — System Design and Implementation"),

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
  PJ("Each model entry must specify family, size_b, quantized, pair_id, model_id, trust_remote_code, dtype, and a benchmarks list. The schema enforces two cross-references: every benchmark name appearing in a model's benchmarks list must also appear as a top-level benchmarks entry, and every pair_id must have at least one model with quantized=false and at least one with quantized=true. These invariants make it impossible to configure a study that omits a baseline or compares a model only against itself."),
  PJ("The decoding section specifies generation parameters; the benchmarks section provides per-benchmark dataset names, splits, sample caps, batch sizes, and benchmark-specific options (such as the benign_only flag for XSTest and the subjects list for MMLU); the slurm section specifies cluster directives and bootstrap commands."),

  H2("4.3 Model Loading and Quantization Path"),
  PJ("The HFModelLoader class encapsulates all model-loading logic. Its load method accepts a ModelSpec dataclass (built from the YAML configuration) and returns the loaded model, tokenizer, and resolved runtime device. The loader is responsible for three things: (i) selecting the runtime device based on the user-specified policy and CUDA availability; (ii) resolving the model dtype from a string preference (auto, float16, bfloat16, float32) to an actual torch.dtype value, with auto resolving to float16 on CUDA and float32 on CPU; and (iii) injecting a BitsAndBytesConfig object into the from_pretrained call when ModelSpec.quantized is true."),
  PJ("The BitsAndBytesConfig is constructed with load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_use_double_quant=True, and bnb_4bit_compute_dtype equal to the resolved torch_dtype. Setting the compute dtype to follow the model dtype ensures that the dequantized compute path uses the same precision the model would have used at full precision, eliminating a subtle source of pair asymmetry."),
  PJ("Two runtime safeguards are applied. First, the loader raises a clear RuntimeError if quantized is true but the resolved runtime device is not cuda, preventing accidental CPU execution of a four-bit path that BitsAndBytes does not support. Second, after the from_pretrained call returns, the loader checks model.is_loaded_in_4bit and logs a warning if the flag is false despite a quantization request — this protects against silent failures in which a checkpoint or installation issue causes BitsAndBytes to skip the quantization."),

  H2("4.4 Generation Pipeline"),
  PJ("The TextGenerator class handles all inference. It accepts the loaded model, tokenizer, runtime device, and a DecodingConfig dataclass, and exposes a single generate_batch method that takes a list of prompts and returns a list of generated responses."),
  PJ("Each prompt passes through a formatting step. When use_chat_template is true (the default), the generator wraps the prompt in a single-message user-role list and applies the tokenizer's apply_chat_template method with add_generation_prompt=True. Critically, enable_thinking=False is also passed to this call so that Qwen3.x-family tokenizers, whose chat templates default to enabling a multi-step thinking block, do not silently consume the entire max_new_tokens budget producing <thinking>...</thinking> output before reaching the answer. A TypeError handler catches tokenizers that do not accept the enable_thinking keyword and retries without it, falling back to a raw-prompt path only as a last resort."),
  PJ("Generation itself uses torch.inference_mode and a do_sample=False decode path under temperature=0.0. Outputs are post-processed to strip the chat-template prefix and any trailing whitespace before being returned to the calling pipeline."),

  H2("4.5 Benchmark Plugins"),
  PJ("Benchmarks are implemented as plugins behind a small abstract base class defined in ethical_benchmark/benchmarks/base.py. Every plugin implements four methods: load_items, build_prompt, score_response, and aggregate. load_items returns a deterministic, seeded sample of BenchmarkItem objects; build_prompt maps each item to a single user-facing prompt string; score_response evaluates a model response against the item and returns a dictionary of boolean and numeric score fields; aggregate consumes the full list of per-item score dictionaries and produces the final aggregated summary."),
  PJ("Refusal detection is shared between HarmBench and XSTest through the benchmarks/utils.py module, which exposes a single classify_refusal function backed by a curated set of regex patterns. Centralising the refusal logic in a single function guarantees that HarmBench and XSTest see exactly the same notion of \"refusal\" — a non-trivial property since these two benchmarks measure opposite sides of the same phenomenon."),
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
  PJ("Cluster orchestration is handled by ethical_benchmark/cluster/generate_jobs.py. The generator reads the loaded configuration's slurm section and emits one sbatch file per scheduling unit. A --group_by flag controls the granularity: with group_by=benchmark, one sbatch is generated per (model, benchmark) pair (eighteen scripts for the six-model, three-benchmark matrix); with group_by=model (the default for TC1), one sbatch is generated per model alias, and that script invokes run_quant_matrix.py so the model is loaded only once."),
  PJ("Each generated sbatch contains the standard SBATCH directives (partition UGGPU-TC1, qos normal, gres gpu:1, cpus-per-task 1, mem 10G, time 06:00:00), output and error log paths, a set -euo pipefail line, a cd into the configured work_dir, a mkdir -p of the log directory, the configured setup_commands (module load slurm, module load anaconda, source activate fyp-tc1), and finally the python invocation with the model alias and configuration path passed in. Six sbatch files are produced for the present configuration, one per model alias."),

  H2("4.9 Output Artifacts"),
  PJ("Every (model, benchmark) run produces a fixed set of files under the results directory. raw.jsonl contains one JSON object per prompt with the prompt text, the model response, the per-response score fields, and the run metadata (model alias, model_id, family, pair_id, quantized flag, seed, generation_config, timestamp). summary.json contains the aggregated metrics, with bootstrap confidence intervals where applicable, alongside the same run metadata. The aggregator also appends a flat row to results/summary/<benchmark>_runs.csv, producing a single CSV per benchmark that records all model runs."),
  PJ("The analysis stage produces results/analysis/pairwise_deltas.json and .csv (one row per (pair_id, benchmark, metric) with absolute and relative deltas), results/analysis/pair_interpretations.csv (one row per pair with the interpretation label and the three component deltas), and results/analysis/quantization_analysis_summary.json (high-level study-wide summary)."),

  H2("4.10 Testing"),
  PJ("The repository ships with a verification suite of 163 automated tests across twelve test files. The full distribution is recorded in Appendix D. Coverage areas include the dataset and benchmark loaders, the legacy evaluators (retained for backward compatibility), model loader specifics including dtype resolution and quantized-flag propagation, the prompt-formatting logic with explicit tests for enable_thinking handling, the matrix-reuse behaviour (verifying that reuse_loaded_model=True loads each model only once), the analysis module's pairwise delta computation, bootstrap CI logic, and v2 score-sidecar selection, the per-prompt schema validator, the resume-helper functions, the refusal-parser regex correctness (including regression tests for the v2 expanded patterns and curly-apostrophe handling), and the SLURM job generator including the per-benchmark and per-model grouping modes. All 163 tests pass on the current commit."),

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
// Chapter 5 — Experimental Setup
// ------------------------------------------------------------
const ch5 = [
  H1("Chapter 5 — Experimental Setup"),

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
      ["pytest", "(installed via requirements.txt; 163 tests passing)"],
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
  PJ("The pre-cache step retrieves HarmBench, the six MMLU subjects, and three model repositories (Qwen3-1.7B, Qwen3-4B, Llama 3.2 3B). XSTest is not fetched from Hugging Face because the canonical CSV is bundled in the repository as data/xstest_v2_prompts.csv. The current TC1 pre-cache completed successfully on 2026-05-27 at 15:37 UTC+8, caching the required current model repositories in ~/.cache/huggingface/hub: Qwen3-1.7B (4.08 GB), Qwen3-4B (8.06 GB), and Llama 3.2 3B (12.9 GB), plus small dataset files. The downloads themselves are pure HTTP file transfers with negligible CPU and memory cost, and are therefore consistent with the head-node activities explicitly demonstrated in the TC1 user guide (page 7–9, where conda install and pip install are shown executing on the head node)."),

  H2("5.6 Run Plan"),
  PJ("With group_by=model, the framework emits six sbatch files — one per model alias. Each script runs the full three-benchmark suite for its model with the model loaded only once, exploiting the matrix runner's reuse_loaded_model=True default. The six scripts are:"),
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
// Chapter 6 — Results and Analysis
// ------------------------------------------------------------
const ch6 = [
  H1("Chapter 6 — Results and Analysis"),

  PJ("All six experimental runs completed successfully on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976–60981). Three matched pairs were evaluated: Qwen3-1.7B (jobs 60976/60977), Qwen3-4B (jobs 60978/60979), and Llama 3.2 3B-Instruct (jobs 60980/60981). Each job ran all three benchmarks sequentially with the model loaded once. The results are presented in Table 6.1 (per-pair raw metrics) and Table 6.2 (quantization deltas and interpretation labels). Detailed per-pair observations are in §6.6 (Qwen 2B), §6.8 (Qwen 4B), and §6.10 (Llama 3.2 3B). The within-family scale analysis (RQ4) is in §6.2 and §6.9. The cross-family comparison (RQ5) is in §6.11."),

  H2("6.1 Results Table"),
  PJ("Table 6.1 records the per-pair benchmark results for all three model pairs together with bootstrap 95% confidence intervals on the deltas. Confidence intervals are computed by the analysis pipeline using a paired bootstrap over the matched prompt set (2 000 resamples, seed 42); the pairing exploits the matched-pair design — both members of every pair see the same prompts in the same order, so each bootstrap draw resamples prompt indices and recomputes the delta directly. A delta is marked statistically significant when its 95% CI excludes zero."),
  buildTable(
    ["Pair", "Metric", "Baseline", "4-bit", "Δ (95% CI)", "Sig?"],
    [
      ["qwen_2b", "HarmBench ASR", "0.600", "0.575", "−0.025 [−0.070, +0.020]", "no"],
      ["qwen_2b", "XSTest over-refusal", "0.052", "0.028", "−0.024 [−0.052, 0.000]", "no"],
      ["qwen_2b", "MMLU accuracy", "0.620", "0.533", "−0.087 [−0.137, −0.037]", "yes"],
      ["qwen_4b", "HarmBench ASR", "0.240", "0.305", "+0.065 [+0.025, +0.110]", "yes"],
      ["qwen_4b", "XSTest over-refusal", "0.028", "0.020", "−0.008 [−0.024, +0.008]", "no"],
      ["qwen_4b", "MMLU accuracy", "0.747", "0.743", "−0.003 [−0.040, +0.033]", "no"],
      ["llama_3_2_3b", "HarmBench ASR", "0.060", "0.060", "0.000 [−0.025, +0.025]", "no"],
      ["llama_3_2_3b", "XSTest over-refusal", "0.044", "0.060", "+0.016 [−0.004, +0.040]", "no"],
      ["llama_3_2_3b", "MMLU accuracy", "0.610", "0.567", "−0.043 [−0.080, −0.007]", "yes"],
    ],
    [1500, 1700, 1100, 1100, 3000, 660],
  ),
  P("Table 6.1  Complete per-pair results with paired bootstrap 95% CIs on the deltas. \"Sig?\" = whether the CI excludes zero. All values are produced by the v2 (expanded) refusal classifier; see §6.1.1 for the scorer-revision history. CIs from `results/analysis/pairwise_deltas.json`.", { size: 18 }),
  PJ("Three statistically significant deltas anchor the analysis: Qwen 4B ΔASR (+0.065, CI [+0.025, +0.110]), Qwen 2B ΔMMLU (−0.087, CI excludes zero), and Llama 3.2 3B ΔMMLU (−0.043, CI excludes zero). The Qwen 4B ΔASR is the only HarmBench result whose confidence interval excludes zero in either direction. The remaining ΔASR values are within sampling noise (Qwen 2B: −0.025, CI straddles zero; Llama: exactly zero with symmetric CI). All XSTest over-refusal deltas are non-significant, supporting the over-refusal null result for RQ2. The implications of significance status for each pair are discussed inline in §6.6, §6.8, §6.10, and the RQ synthesis in §6.9."),

  H3("6.1.1 Scorer revision history"),
  PJ("All HarmBench and XSTest figures in this report are produced by the v2 (expanded) refusal classifier (`ethical_benchmark/benchmarks/utils.py`, scorer version `v2_expanded_refusal_patterns_2026-05-28`). The v1 classifier (pre-2026-05-28) used a narrow set of 14 refusal patterns that missed the most common modern refusal templates — including \"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", and curly-apostrophe variants used by Qwen3 output. As a result the v1 classifier systematically misclassified clear refusals as harmful compliance, inflating reported ASR and depressing reported over-refusal. The classifier was rewritten with comprehensive coverage of standard refusal templates, validated by manual stratified audit of 55 sampled responses across all six runs (audit confirmed correctness on the sample; see PROJECT_LOG entry 2026-05-28). The TC1-original `raw.jsonl` and `summary.json` files are retained as immutable v1 artifacts; corrected scoring is stored in derived `scores.v2.jsonl` sidecars (prompt IDs and score_fields only, no prompt or response text) and `summary.v2.json` files. The analysis pipeline prefers these v2 sidecars when present, so the report uses corrected scores without modifying the original raw generations. The auxiliary artifacts `results/analysis/rescore_diagnostics_*.csv` (IDs plus matched pattern names/indices only, no response text) and `results/analysis/rescore_aggregate.json` document the per-record reclassification for full auditability."),
  PJ("Impact of the rescore on headline numbers. Three pair-level deltas changed materially: Qwen 1.7B ΔASR moved from −0.120 (v1) to −0.025 (v2) — the v1 \"significant decrease\" was largely an artefact of unrecognised refusals; Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2) and acquired statistical significance, becoming the study's most robust empirical finding; Llama 3.2 3B ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). MMLU values are unchanged (MMLU scoring does not depend on the refusal classifier). The interpretation labels for two of three pairs changed: Qwen 4B from alignment_improvement (v1) to alignment_degradation (v2); Llama from alignment_degradation (v1) to broad_degradation (v2). The Qwen 1.7B label (capability_collapse_masquerading_as_safety) is unchanged in spirit but is now driven entirely by the significant MMLU drop rather than the ASR move. The scorer revision is documented openly in this report as a matter of methodological transparency: the v1 numbers would have supported a more optimistic narrative about NF4 quantization, and that narrative is now retracted in favour of the v2 reading."),

  H2("6.2 Within-Family Scale Analysis (RQ4)"),
  PJ("RQ4 asks whether the magnitude of the quantization effect differs between two-billion and four-billion parameter Qwen models. Both pairs have now completed. Table 6.2 summarises the quantization deltas and interpretation labels for all three pairs, enabling both the within-family scale comparison (RQ4) and the cross-family comparison (RQ5)."),
  buildTable(
    ["Metric", "qwen_2b  (Qwen3-1.7B)", "qwen_4b  (Qwen3-4B)", "llama_3_2_3b  (Llama3.2-3B)"],
    [
      ["HarmBench ΔASR", "−0.025  (−4.2%)", "+0.065  (+27.1%) ★", "0.000  (0.0%)"],
      ["XSTest Δover-refusal", "−0.024  (−46.2%)", "−0.008  (−28.6%)", "+0.016  (+36.4%)"],
      ["MMLU Δaccuracy", "−0.087  (−14.0%) ★", "−0.003  (−0.5%)", "−0.043  (−7.0%) ★"],
      ["Interpretation label", "capability_collapse_masquerading_as_safety", "alignment_degradation", "broad_degradation"],
    ],
    [2100, 2753, 2753, 2753],
  ),
  P("Table 6.2  All-pair quantization deltas and interpretation labels. Δ = 4-bit − baseline. ★ marks deltas whose paired-bootstrap 95% CI excludes zero (statistically significant). Three deltas are significant: Qwen 4B ΔASR (the only significant HarmBench result), Qwen 2B ΔMMLU, and Llama ΔMMLU.", { size: 18 }),
  PJ("Table 6.2 reveals a striking scale effect on capability. Under the same NF4 quantization procedure, the 1.7B model experiences large capability degradation — MMLU drops 8.7 percentage points (relative decline of 14.0%, CI excludes zero) — while the 4B model is essentially unaffected — MMLU moves by only 0.3 percentage points, well within sampling noise. The MMLU delta ratio (8.7 pp versus 0.3 pp) is approximately 29:1, making scale the single largest predictor of capability preservation under quantization observed in this study."),
  PJ("The full-precision baselines reveal that within the Qwen family the larger model is both more capable and more refusal-calibrated. Qwen3-1.7B's baseline HarmBench ASR is 0.600 while Qwen3-4B's is 0.240, so scaling from 1.7B to 4B reduces baseline harmful compliance by 36 percentage points. MMLU accuracy simultaneously rises from 0.620 to 0.747 (+12.7 pp), confirming the capability gain is genuine, and XSTest over-refusal falls from 0.052 to 0.028, indicating fewer false refusals on benign prompts. The 4B baseline therefore dominates the 1.7B baseline on every dimension under the v2 scorer. This larger-is-safer-and-more-capable pattern is itself a useful empirical finding and motivates the matched-pair design: quantization effects must be interpreted against a baseline that already varies substantially by scale within the same family."),
  PJ("The most important comparison is on the post-quantization side. The 1.7B pair produces a small, non-significant ΔASR of −0.025 with a large, significant ΔMMLU of −0.087 — exactly the capability_collapse_masquerading_as_safety pattern (large capability loss, ASR within noise; the rule fires on the conjunction of the two thresholds even though the ASR direction is not significant on its own). The 4B pair produces an opposite and statistically robust pattern: ΔMMLU is within tolerance, while ΔASR increases by 6.5 percentage points (CI [+0.025, +0.110]). Without the MMLU anchor, both results might be ambiguously interpreted; with the anchor, the diagnostic separation is clean. This contrast is the methodological payoff of the capability-anchored design."),

  H2("6.3 Cross-Family Replication (RQ5)"),
  PJ("RQ5 asks whether the Qwen pattern replicates qualitatively in Llama 3.2 3B. The cross-family analysis examines two properties: sign consistency (do the Llama deltas have the same sign as the Qwen deltas?) and approximate magnitude (are the Llama deltas similar in scale?). The full cross-family comparison is in §6.11; this section provides a high-level orientation."),
  PJ("The headline finding for RQ5 is that NF4 quantization affects harmful compliance differently in the two model families. Within Qwen the effect depends on scale: the 1.7B pair shows a small, non-significant ΔASR of −0.025 alongside large capability loss, while the 4B pair shows a statistically significant ΔASR increase of +0.065 at preserved capability — the only HarmBench effect in the study whose CI excludes zero. The Llama 3.2 3B pair shows ΔASR exactly at zero with a symmetric noise band, paired with a significant ΔMMLU of −0.043. The capability dimension is directionally consistent (all three pairs have negative ΔMMLU; two are significant, the Qwen 4B value is within noise). On over-refusal, all three pairs show post-quantization changes within sampling noise; no ΔOR is significant. The composite picture is that NF4 quantization does not produce uniform safety effects across families and scales, and the most robust empirical signal (Qwen 4B ASR increase) goes in the safety-worsening direction. See §6.11 for the full discussion."),

  H2("6.4 Capability Anchoring"),
  PJ("Capability anchoring is the core methodological contribution of the interpretation framework. Each pair's MMLU delta is used as a covariate when interpreting its HarmBench and XSTest deltas. Five canonical outcomes are tracked, organised symmetrically around the safety axis:"),
  Bullet("Alignment degradation: MMLU is preserved, but HarmBench ASR rises beyond threshold — the model becomes more compliant with harmful prompts without a corresponding capability change. A direct safety regression."),
  Bullet("Alignment improvement: MMLU is preserved, but HarmBench ASR falls beyond threshold — quantization reduces harmful compliance without measurable capability cost. The mirror image of alignment degradation and the most desirable outcome in the taxonomy."),
  Bullet("Capability-collapse masquerading as safety: MMLU drops noticeably, and HarmBench ASR also drops; the apparent safety improvement is attributed to reduced instruction-following capacity rather than a genuine alignment change."),
  Bullet("Robust preservation: all three deltas are small in magnitude; quantization is effectively neutral across safety and capability dimensions."),
  Bullet("Broad degradation: fallback — typically signals over-refusal moving outside tolerance, or ASR within tolerance but capability dropping. Used when no specific pattern above matches."),
  PJ("The three pairs in this study exemplify three of these five outcomes and illustrate why the capability anchor is necessary. For the qwen_2b pair, MMLU fell by 8.7 pp (−14.0% relative; CI [−0.137, −0.037], significant). The contemporaneous ASR delta is small and not statistically significant (−2.5 pp, CI [−0.070, +0.020]) — but combined with the significant MMLU drop, the conjunction triggers the capability_collapse_masquerading_as_safety rule: large capability loss can mask any underlying alignment direction, so the safety dimension cannot be read as an alignment change in isolation. For the qwen_4b pair, MMLU is preserved (−0.3 pp, within tolerance) and ΔASR is positive and statistically significant (+6.5 pp, CI [+0.025, +0.110]) — exactly the alignment_degradation condition. The 4B model becomes more compliant with harmful prompts under quantization without any measurable capability change, so the safety move cannot be dismissed as a capability artefact. For the llama_3_2_3b pair, MMLU drops 4.3 pp (significant) while ΔASR is at exactly zero with a symmetric noise band; this combination — capability loss without a safety direction — falls through to broad_degradation. The contrast across the three pairs is decisive: without the MMLU anchor, the small Qwen 2B ASR move could be mistaken for safety improvement and the Llama capability cost would be invisible behind the unchanged ASR. The anchor turns three otherwise ambiguous quantization outcomes into three well-separated diagnostic categories."),

  H2("6.5 Statistical Caveats"),
  PJ("Three statistical limitations should be borne in mind when interpreting the results. First, with 200 HarmBench prompts and 250 benign XSTest prompts, the paired bootstrap 95% confidence interval on a binomial-proportion delta is approximately ±0.05 (Table 6.1) — small deltas may not be statistically distinguishable from zero. The Qwen 2B ΔASR (−0.025, CI [−0.070, +0.020]) illustrates the consequence: the capability_collapse_masquerading_as_safety rule fires on the combination of a small ASR move and a large MMLU drop, but the ASR component on its own is within sampling noise. The Llama ΔASR (exactly 0.000, CI [−0.025, +0.025]) is similarly within noise and the broad_degradation label is driven by the significant MMLU decline rather than any safety direction. The Qwen 4B ΔASR (+0.065, CI [+0.025, +0.110]) is the only HarmBench result whose CI excludes zero. Second, decoding is deterministic at temperature 0.0, so the only source of variance reflected in the bootstrap intervals is prompt sampling — there is no within-condition stochastic variance. A multi-seed (T = 0.7) sensitivity arm for one pair (Chapter 9, future work) would provide an independent variance estimate. Third, the MMLU subset comprises 300 questions pooled across six subjects; subject counts are uneven (25–94 per subject: business_ethics 25, college_biology 33, human_aging 44, clinical_knowledge 48, high_school_world_history 56, high_school_macroeconomics 94). Subject-level accuracy estimates are correspondingly noisy and are not reported as primary statistics. The distribution is identical across all baseline and 4-bit runs (same seed, same prompt IDs), so cross-condition comparisons are unaffected. Fourth (added 2026-05-28), all HarmBench and XSTest ASR/OR figures in this chapter are produced by the v2 (expanded) refusal classifier; the original v1 classifier under-detected modern refusal templates and is documented in §6.1.1 (scorer revision history) above."),

  H2("6.6 Observations: Qwen 2B Pair"),
  PJ("The Qwen 2B pair receives the capability_collapse_masquerading_as_safety label. The MMLU drop is the statistically robust component (−8.7 pp, CI [−0.137, −0.037], significant); the ASR component is small and within noise (−2.5 pp, CI [−0.070, +0.020], not significant). The combination — substantial capability loss with the ASR direction not statistically separable from zero — is exactly what the capability-collapse rule was designed to flag: large enough MMLU decline that any small ASR move on its own cannot be safely interpreted as an alignment change."),
  PJ("HarmBench ASR moved from 0.600 to 0.575 (−2.5 pp, −4.2% relative; 95% CI [−0.070, +0.020], not significant). Even though the point estimate is in the safety-improving direction, the bootstrap CI straddles zero and the result on its own provides no evidence for an alignment shift. MMLU accuracy fell by 8.7 percentage points (0.620 → 0.533, −14.0% relative; 95% CI [−0.137, −0.037], significant). The conjunction triggers the capability_collapse_masquerading_as_safety rule: the model became less capable overall, and the small ASR move cannot be safely separated from that capability loss. Reporting ASR alone — even if the direction looked encouraging — would have invited a misleading alignment-improvement reading that the data do not support."),
  PJ("The XSTest over-refusal rate declined modestly (0.052 → 0.028, ΔOR = −0.024, CI [−0.052, 0.000], not significant). This decline is unusual for a capability-collapse pattern: if reduced ASR were driven by increased over-refusal — the model refusing everything — XSTest OR would have been expected to rise. Its decline instead suggests the 4-bit model produces more compliant responses on benign prompts, which is consistent with a reduction in pattern-triggered refusal behaviour rather than an increase in genuine safety alignment. Together with the significant MMLU drop, the three deltas point toward broad capability degradation that manifests differently across benchmark types: the model complies slightly less with complex harmful instructions (within noise), refuses slightly fewer benign prompts (within noise), and answers fewer factual questions correctly (significant)."),
  PJ("Subject-level MMLU breakdown supports the capability-decline reading. Accuracy declined in five of six subjects: high school macroeconomics (0.596 → 0.479, −11.7 pp), business ethics (0.680 → 0.560, −12.0 pp), clinical knowledge (0.729 → 0.563, −16.7 pp), college biology (0.576 → 0.515, −6.1 pp), and high school world history (0.643 → 0.571, −7.1 pp). A small gain in human ageing (0.523 → 0.568, +4.6 pp) is within the noise range expected for a 44-sample subject subset. The overall accuracy decline of 8.7 percentage points has a 95% CI of [−0.137, −0.037] and is robustly distinguishable from zero by the paired bootstrap."),
  PJ("Of the three pairs in this study, the Qwen 2B pair is the cleanest empirical example of why a capability anchor is methodologically necessary. The non-significant ASR move could be read (mistakenly) as a small alignment gain if reported in isolation; combined with the significant MMLU loss, the capability-collapse rule correctly flags that no alignment-direction claim can be made from this data. The framework prevents the false-positive alignment-improvement reading that ASR-only reporting would have invited."),

  H2("6.7 Preliminary Scale Observations: Baseline Comparison"),
  PJ("The Qwen3-1.7B and Qwen3-4B full-precision baselines reveal a capability-and-safety coupling within the Qwen family. At full precision, scaling from 1.7B to 4B parameters produces: HarmBench ASR 0.600 → 0.240 (−36.0 pp; the larger model refuses many more harmful prompts), MMLU accuracy 0.620 → 0.747 (+12.7 pp; substantially more capable), and XSTest over-refusal 0.052 → 0.028 (−2.4 pp; fewer false refusals on benign prompts). Within Qwen the larger model dominates the smaller one on every dimension: more capable, more safety-calibrated, and better calibrated on benign prompts. This baseline divergence is itself an interesting empirical finding and motivates the matched-pair design — quantization effects must be interpreted against scale-dependent baselines."),

  H2("6.8 Observations: Qwen 4B Pair"),
  PJ("The Qwen 4B pair is the most empirically striking finding of the study. Quantization significantly increases harmful compliance (ΔASR = +0.065, CI [+0.025, +0.110]) while capability is preserved (ΔMMLU = −0.003, well within tolerance). The MMLU CI [−0.040, +0.033] straddles zero, ruling out capability collapse as an alternative explanation; the alignment-degradation reading is therefore not a capability artefact. This is the only HarmBench delta in the study whose 95% confidence interval excludes zero, and it goes in the safety-worsening direction."),
  PJ("HarmBench ASR increases from 0.240 to 0.305 (+6.5 pp, +27.1% relative; 95% CI [+0.025, +0.110], significant). MMLU accuracy is essentially unchanged at 0.747 → 0.743 (−0.5% relative; 95% CI [−0.040, +0.033], not significant). XSTest over-refusal moves marginally from 0.028 to 0.020 (ΔOR = −0.008, not significant). The combined pattern — significant ASR increase, preserved capability, slight OR drop — matches the alignment_degradation rule precisely: ΔASR ≥ +0.02 (harm_tol) with capability within tolerance. The pipeline assigns the alignment_degradation label."),
  PJ("Mechanism. Because Qwen 4B at full precision is the most refusal-calibrated model in the study (baseline ASR = 0.240, the lowest of the three baselines), it has substantial headroom for ASR to increase before reaching ceiling. The observed +6.5 pp move is small in absolute terms but large in relative terms (+27.1%) and is statistically robust. One plausible mechanism is that NF4 compression slightly degrades the precision of the learned refusal heuristics, with a larger relative effect on a model whose refusal threshold sits in a discriminating part of the distribution. The 1.7B model, by contrast, has weaker refusal calibration at baseline (ASR = 0.600), so further degradation of refusal heuristics has less room to manifest — and indeed its ASR is approximately stable under quantization."),
  PJ("Statistical confidence. The Qwen 4B ΔASR is the single most statistically robust HarmBench finding in the study: the 95% bootstrap CI of [+0.025, +0.110] entirely excludes zero on the positive side, and the lower bound is comfortably above the +0.02 harm tolerance threshold that triggers the alignment_degradation rule. Combined with the preserved MMLU (the capability anchor rules out a confound), this is the cleanest single-pair empirical signal in the dataset. It is also the most consequential for deployment: an NF4-quantized Qwen3-4B exhibits measurably weaker harmful-compliance refusal than its full-precision counterpart, at a difference that is unlikely to be due to sampling noise."),
  PJ("Scale contrast (RQ4). Under the same quantization procedure, the 1.7B and 4B Qwen models exhibit qualitatively different failure modes. The 1.7B model loses capability without a clear safety direction (capability_collapse_masquerading_as_safety: significant ΔMMLU, non-significant ΔASR). The 4B model preserves capability but loses alignment (alignment_degradation: non-significant ΔMMLU, significant ΔASR). Neither outcome is desirable, but they are diagnostically different: the 1.7B failure mode would be addressed by training a more robust quantization-aware model, while the 4B failure mode points specifically to NF4-induced degradation of refusal heuristics. RQ4 is therefore answered by these two failure modes being categorically distinct, not by one being a magnified version of the other."),
  PJ("The Llama 3.2 3B pair results (§6.10–§6.11) allow assessment of whether the Qwen 4B alignment-degradation pattern reproduces cross-family."),

  H2("6.9 Research Question Synthesis"),
  PJ("With all six jobs complete and all three pairs evaluated under the v2 (expanded) refusal scorer, it is now possible to give answers to all five research questions. Paired bootstrap 95% confidence intervals (Table 6.1) quantify the statistical uncertainty behind each answer. Three deltas are statistically significant: Qwen 4B ΔASR (+0.065), Qwen 2B ΔMMLU (−0.087), and Llama 3B ΔMMLU (−0.043). All other deltas — including the remaining two ΔASR values, all three ΔOR values, and Qwen 4B ΔMMLU — are within sampling noise. The Qwen 4B ΔASR is the only HarmBench result whose CI excludes zero in either direction; the alignment-degradation finding for Qwen 4B is therefore the single most statistically robust empirical result in the study."),

  H3("RQ1 — Does 4-bit NF4 quantization increase harmful compliance?"),
  PJ("Family- and scale-dependent. The only statistically significant ΔASR is positive: the Qwen 4B pair shows ΔASR = +0.065 (CI [+0.025, +0.110]), meeting the alignment_degradation threshold with capability preserved. The Qwen 1.7B pair shows ΔASR = −0.025 (CI [−0.070, +0.020]) — directionally negative but not separable from zero — combined with significant capability loss (the capability_collapse_masquerading_as_safety pattern). The Llama 3.2 3B pair shows ΔASR = 0.000 (CI [−0.025, +0.025]) — exactly at zero with a symmetric noise band, paired with significant ΔMMLU. The answer to RQ1: there is no universal direction. The only statistically robust ASR move under NF4 quantization in this study goes in the safety-worsening direction (Qwen 4B), so the data refute any blanket claim that quantization reduces harmful compliance. The remaining two pairs do not provide statistically separable ΔASR signals."),

  H3("RQ2 — Does 4-bit NF4 quantization increase over-refusal on benign prompts?"),
  PJ("No — all three ΔOR values are non-significant under the v2 scorer. Qwen 2B: ΔOR = −0.024 (CI [−0.052, 0.000], not significant). Qwen 4B: ΔOR = −0.008 (CI [−0.024, +0.008], not significant). Llama 3B: ΔOR = +0.016 (CI [−0.004, +0.040], not significant; this is the largest absolute ΔOR in the study but still within sampling noise). Two pairs trend slightly negative, one trends slightly positive, none reaches significance. The answer to RQ2: NF4 quantization does not produce a detectable over-refusal increase on benign prompts at this sample size, across two families and three scale points. This is the most robust null result in the study and is practically useful for deployment teams considering NF4 compression: concern that quantization would make models more trigger-happy on benign prompts is not supported by these data."),

  H3("RQ3 — Does 4-bit NF4 quantization degrade general capability?"),
  PJ("The answer depends strongly on model scale and family. The Qwen3-1.7B model loses 8.7 percentage points of MMLU accuracy under NF4 quantization (CI [−0.137, −0.037], significant), a decline spanning five of six evaluated subjects. The Llama 3.2 3B model loses 4.3 percentage points (CI [−0.080, −0.007], significant), spanning four of six subjects. The Qwen3-4B model loses only 0.3 percentage points (CI [−0.040, +0.033], not significant — within measurement noise). Scale is a strong moderator of quantization-induced capability loss within the Qwen family, and Llama 3B sits between the two Qwen models in sensitivity. The answer to RQ3: NF4 quantization significantly degrades capability in the 1.7B Qwen model and the 3B Llama model, but has negligible effect on the 4B Qwen model. This scale-and-family-dependent capability profile is the most practically important capability finding for deployment teams."),

  H3("RQ4 — Are smaller models more sensitive to quantization within the same family?"),
  PJ("Yes, but the sensitivity manifests as categorically different failure modes rather than as a single metric scaling continuously. On capability, the MMLU delta ratio between Qwen 1.7B and Qwen 4B is approximately 29:1 (−8.7 pp versus −0.3 pp): the 1.7B model loses substantial capability under NF4 compression while the 4B model is essentially unaffected. On safety, the two Qwen pairs exhibit *qualitatively different* labels: the 1.7B pair is capability_collapse_masquerading_as_safety (significant ΔMMLU, ΔASR within noise), while the 4B pair is alignment_degradation (preserved capability, significant ΔASR increase). The answer to RQ4: smaller Qwen models are more sensitive to quantization on the capability axis, but the smaller-versus-larger contrast within the Qwen family is best characterised not as a magnitude difference on a single dimension but as two different failure profiles. The framework correctly diagnoses them as separate categories."),

  H3("RQ5 — Are effects consistent across model families?"),
  PJ("Partially. On capability, both families lose MMLU accuracy under NF4 compression at the comparable scale points where loss is significant: Qwen 1.7B (−8.7 pp) and Llama 3B (−4.3 pp); the Qwen 4B model is the exception with non-significant loss. Direction is consistent (negative) where the effect is detectable. On safety, the three pairs produce three different labels (capability_collapse_masquerading_as_safety, alignment_degradation, broad_degradation), and the only statistically significant ΔASR (Qwen 4B) does not have a same-family or same-scale counterpart in the data for cross-validation. The cross-family pattern most likely to reproduce is the null on over-refusal: all three ΔOR are non-significant. The full answer to RQ5: capability and over-refusal effects show partial consistency (same direction where detectable; no significant over-refusal in either family), while harmful-compliance effects vary categorically by family and scale. The Qwen 4B alignment-degradation finding would benefit from replication on a same-family-different-scale pair (e.g. Qwen 7B) and a same-scale-different-family pair (e.g. Llama 3B at higher capability)."),

  H3("Discussion: What the full three-pair dataset tells us"),
  PJ("Three implications extend beyond the specific numbers. First, surface-level safety metrics are unreliable without a capability anchor and without statistical-significance reporting. The Qwen 1.7B pair has a directionally encouraging ΔASR (−0.025) that on its own would invite a misleading alignment-improvement interpretation; the framework correctly down-classifies this to capability_collapse_masquerading_as_safety because of the simultaneous large MMLU drop. The Llama pair has ΔASR exactly at zero with significant capability loss; the framework correctly assigns broad_degradation rather than incorrectly attributing the capability cost to a safety effect. Second, the most robust empirical finding goes in the safety-worsening direction. Qwen 4B's alignment_degradation (+0.065, CI excludes zero) is the only statistically significant ΔASR in the study and contradicts the simple narrative that NF4 compression is safety-neutral. Third, the framework itself is the durable contribution: it produces well-separated diagnostic categories from data that on a single metric (ASR) would have appeared to tell three different and partly contradictory stories. The framework's value does not depend on the specific empirical findings being reproduced — it provides a reusable methodology for the next study, regardless of the underlying numerical results."),

  H2("6.10 Observations: Llama 3.2 3B Pair"),
  PJ("The Llama 3.2 3B pair (SLURM jobs 60980 and 60981) completed on 2026-05-27. Job 60980 (baseline, float16) ran in 6 minutes 15 seconds; job 60981 (NF4 4-bit) ran in 13 minutes 9 seconds, consistent with the longer load time of quantized models observed in the Qwen runs."),
  PJ("The baseline profile of Llama 3.2 3B-Instruct (v2 scorer) is the most refusal-calibrated of the three models in the study. HarmBench ASR at full precision is 0.060 — only 12 out of 200 adversarial prompts produced compliant outputs by the v2 classifier, with the remainder correctly refused. XSTest over-refusal is 0.044, also low. MMLU accuracy is 0.610, closely matching the Qwen3-1.7B baseline (0.620). This profile describes a model that is both compactly capable and strongly refusal-calibrated at full precision: substantially safer on HarmBench than either Qwen model (Qwen 1.7B baseline ASR 0.600; Qwen 4B baseline ASR 0.240) while approximately equally capable as the smaller Qwen on MMLU."),
  PJ("Under NF4 4-bit quantization, the Llama pair is classified as broad_degradation. HarmBench ASR is essentially unchanged at 0.060 → 0.060 (ΔASR = 0.000, CI [−0.025, +0.025], not significant): the model retains its strong refusal behaviour under compression. XSTest over-refusal rises modestly from 0.044 to 0.060 (ΔOR = +0.016, CI [−0.004, +0.040], not significant). MMLU accuracy falls by 4.3 percentage points (0.610 → 0.567, −7.0% relative; CI [−0.080, −0.007], statistically significant). The MMLU drop spans four of six subjects: high school macroeconomics (0.543 → 0.468, −7.5 pp), college biology (0.667 → 0.545, −12.1 pp), clinical knowledge (0.667 → 0.583, −8.3 pp), and human aging (0.636 → 0.614, −2.3 pp). Two subjects show small gains within the noise range: high school world history (+3.6 pp) and business ethics (+4.0 pp). The MMLU decline of 4.3 percentage points is moderate — larger than Qwen 4B (−0.3 pp) but smaller than Qwen 2B (−8.7 pp), placing Llama between the two Qwen models in quantization-induced capability sensitivity."),
  PJ("Why broad_degradation? With ΔASR within noise and a significant ΔMMLU, the rule logic falls through to the fallback label. The capability_collapse_masquerading_as_safety rule requires ΔASR ≤ −0.02 (a meaningful safety move in the protective direction) alongside ΔMMLU ≤ −0.03; Llama's ΔASR is exactly zero, so that combination does not apply. The alignment_degradation rule requires ΔASR ≥ +0.02; that does not apply either. The alignment_improvement and robust_preservation rules also do not fit (capability does not satisfy preservation). The pair therefore matches the broad_degradation fallback: a meaningful capability cost without a detectable safety direction. This is itself diagnostically informative — the Llama pair refutes the simple narrative that NF4 compression universally weakens alignment. Llama's strong baseline refusal calibration is preserved under quantization at the cost of measurable capability loss, but no safety regression accompanies that cost."),

  H2("6.11 Cross-Family Comparison (RQ5)"),
  PJ("The cross-family comparison places the Llama 3.2 3B pair alongside the two Qwen pairs to assess whether NF4 quantization effects are consistent across architectures and alignment recipes. Table 6.2 (§6.2) shows the delta values and interpretation labels for all three pairs side by side."),

  H3("6.11.1 Baseline safety profiles differ substantially across families"),
  PJ("Before interpreting quantization deltas, the baseline profile differences are themselves informative. At similar MMLU capability (Qwen3-1.7B: 0.620; Llama 3.2 3B: 0.610), the Llama model is substantially more refusal-calibrated than the smaller Qwen: baseline HarmBench ASR is 0.060 for Llama vs 0.600 for Qwen 1.7B — a 54 pp difference in favour of Llama on the safety axis at equivalent capability. XSTest over-refusal at baseline is 0.044 for Llama vs 0.052 for Qwen 1.7B — comparable. This indicates that two compact models at the same factual capability can have fundamentally different safety calibrations, attributable to differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach. Family choice is therefore a larger safety determinant than quantization method for these models: the Llama vs Qwen 1.7B baseline gap on ASR (54 pp) is roughly an order of magnitude larger than the largest single quantization delta in the study (Qwen 4B, +6.5 pp). The 4B Qwen baseline at 0.240 sits between these two extremes."),

  H3("6.11.2 Harmful compliance: only one significant delta, in the worsening direction"),
  PJ("Only one of the three ΔASR values is statistically significant under the v2 scorer: Qwen 4B (+0.065, CI [+0.025, +0.110]) — a safety regression. The Qwen 1.7B ΔASR (−0.025, CI [−0.070, +0.020]) and the Llama ΔASR (0.000, CI [−0.025, +0.025]) are both within sampling noise. The data therefore do not support a universal directional claim about NF4 quantization and harmful compliance, but they do contain one robust counterexample to the optimistic narrative: in Qwen 4B, NF4 compression measurably increases harmful compliance at preserved capability. The remaining two ΔASR values are best described as null, not as evidence of safety improvement or worsening. The Llama baseline is already so strongly refusal-calibrated (ASR = 0.060) that little headroom exists for further reduction; whether NF4 would alter ASR more visibly in a less strongly-aligned Llama variant is an open question."),

  H3("6.11.3 Over-refusal: consistent null result across families"),
  PJ("The XSTest over-refusal rate is the most consistent cross-family null. None of the three ΔOR deltas is statistically significant: Qwen 2B (−0.024, CI [−0.052, 0.000], not significant), Qwen 4B (−0.008, CI [−0.024, +0.008], not significant), and Llama 3B (+0.016, CI [−0.004, +0.040], not significant). Two pairs trend slightly negative, one trends slightly positive, none is separable from zero. This null finding spans two families and three scale points and is practically useful for deployment: the concern that NF4 compression would make models excessively conservative on benign prompts is not supported by these data. The largest single over-refusal effect is on Llama (+0.016) and is itself a borderline-direction signal — quantization may slightly increase Llama's already-low rate of false refusal — but the CI does not exclude zero and the absolute magnitude is small."),

  H3("6.11.4 Capability: directionally consistent where significant, magnitude varies"),
  PJ("Two of three pairs lose MMLU accuracy significantly: Qwen 1.7B (−8.7 pp, CI [−0.137, −0.037]) and Llama 3B (−4.3 pp, CI [−0.080, −0.007]). The Qwen 4B pair is essentially neutral (−0.3 pp, CI [−0.040, +0.033], not significant). All three point estimates are negative; two are significant. The Llama 3B value (−4.3 pp) is intermediate between the two Qwen models. This positioning is consistent with a size-sensitivity hypothesis — Llama 3B, at 3 billion parameters, lies between the two Qwen models in the robustness spectrum. However, because Llama and Qwen differ in architecture and training, this apparent scale ordering may also reflect family-specific effects. More data points (e.g., Qwen at 7B or Llama at 1B) would be needed to confirm whether the ordering reflects a size effect or a family effect."),

  H3("6.11.5 Interpretation labels and the study's central question"),
  PJ("The three interpretation labels — capability_collapse_masquerading_as_safety (Qwen 1.7B), alignment_degradation (Qwen 4B), and broad_degradation (Llama 3B) — collectively answer the study's central question: \"do observed safety changes under quantization reflect alignment shifts or capability artefacts?\" None of the three pairs corresponds to a clean alignment-improvement or robust-preservation outcome. The Qwen 1.7B pair is the textbook capability-collapse case: significant capability loss with the safety direction not statistically separable, so no alignment claim is warranted. The Qwen 4B pair is the textbook alignment-regression case: preserved capability with significant ASR increase, the only such pattern in the study and the most direct evidence that NF4 compression can genuinely weaken a model's refusal behaviour. The Llama pair is a capability cost without a safety direction: NF4 reduces MMLU accuracy while leaving ASR unchanged. Taken together, the three pairs demonstrate that the framework reliably separates three different failure modes; the central scientific finding is the framework's diagnostic value, while the most consequential empirical finding for deployment is the Qwen 4B alignment_degradation result. The data refute the simple optimistic narrative that NF4 quantization is safety-neutral or safety-improving in this scale range."),
];

// ------------------------------------------------------------
// Chapter 7 — Discussion and Threats to Validity
// ------------------------------------------------------------
const ch7 = [
  H1("Chapter 7 — Discussion and Threats to Validity"),

  PJ("The Qwen family results (§6.9) provide initial empirical grounding for the threats discussed in this chapter. The qwen_2b capability-collapse finding and the qwen_4b alignment-degradation finding are used below as concrete examples where relevant."),

  H2("7.1 Internal Validity"),
  PJ("Internal validity is the strongest property of the present design. The matched-pair structure, combined with on-the-fly NF4 quantization from identical baseline weights, isolates quantization as the sole experimental variable. There is no plausible alternative explanation for an observed delta beyond the quantization step itself, the act of loading the same checkpoint twice, or measurement noise. Deterministic decoding eliminates within-condition variance from generation, and deterministic regex-based scoring eliminates judge-model variance from evaluation. The resume logic prevents partial-run contamination: every reported metric is computed from a complete raw.jsonl that contains exactly the configured number of prompts. The Qwen 2B and Qwen 4B pairs both completed under the same conditions, hardware, and software stack, providing strong internal consistency across the two within-family data points."),

  H2("7.2 External Validity"),
  PJ("External validity is bounded by three explicit design choices. First, findings are bounded to BitsAndBytes NF4 quantization. Alternative methods such as GPTQ, AWQ, and the GGUF family used by llama.cpp may produce different effects, both quantitatively and qualitatively; the present study does not claim to characterise those methods. Second, the Qwen3 baselines (Qwen3-1.7B and Qwen3-4B) are instruction-tuned dense models from the official Alibaba Qwen3 release; results describe quantization behaviour on this specific model family and may not transfer directly to other architectures or training regimes. Third, the study is restricted to English-language, text-only interactions; multilingual and multimodal effects are out of scope."),

  H2("7.3 Construct Validity"),
  PJ("Each benchmark operationalises its dimension in a specific way. HarmBench captures one curated definition of \"harmful compliance\" against a particular set of adversarial prompts; other operationalisations exist and may yield different absolute numbers. XSTest captures over-refusal against a specific distribution of benign-but-suspicious prompts. The MMLU subset, even with diverse subject coverage, is a partial capability proxy and does not measure all reasoning capacities relevant to deployment (such as code generation, long-context reasoning, or tool use). These construct boundaries are common in safety evaluation but should be borne in mind when interpreting the results."),

  H2("7.4 Refusal Parser Trade-offs"),
  PJ("The deterministic regex-based refusal parser was selected for reproducibility and transparency, and against the alternative of using a separate judge LLM. The trade-off is sensitivity. The parser may miss nuanced refusals — for example, a model that pivots to a related but non-harmful topic, or that produces a euphemistic refusal that does not match any of the configured patterns. The parser is therefore more conservative in its refusal estimates than a judge-LLM would likely be. After the v1→v2 correction, this limitation is handled through two safeguards: immutable raw outputs with derived score sidecars, and a planned judge-model validation pass that will produce separate judge sidecars rather than overwriting the current v2 results."),

  H2("7.5 Cross-Family Comparison Caveat"),
  PJ("The Qwen-versus-Llama comparison should be read as descriptive only. Qwen and Llama differ in tokenizer, pre-training corpus, instruction-tuning recipe, and safety-alignment methodology. Differences in their quantization deltas could plausibly reflect any combination of these factors, not just family identity. The cross-family component of the study therefore provides a useful robustness check on the within-Qwen findings but does not support causal claims about quantization–family interactions."),
];

// ------------------------------------------------------------
// Chapter 8 — Limitations
// ------------------------------------------------------------
const ch8 = [
  H1("Chapter 8 — Limitations"),
  PJ("The principal limitations of the study are summarised below. Each is acknowledged explicitly and, where possible, addressed in the Future Work chapter."),
  ...numberedList([
    "Single quantization method. Only BitsAndBytes NF4 four-bit quantization is evaluated. GPTQ, AWQ, GGUF, and INT8 paths are out of scope and are flagged for follow-up work.",
    "Refusal parser approximation. The deterministic regex-based refusal parser is reproducible but may under-count nuanced refusals.",
    "Partial capability proxy. The six-subject MMLU subset is a tractable but partial measure of general capability and does not include code generation, long-context reasoning, or tool-use evaluation.",
    "Text-only, English-only scope. Multilingual and multimodal behavioural effects of quantization are out of scope.",
    "Greedy decoding only. Temperature 0.0 is used throughout, eliminating within-condition stochastic variance from the analysis and precluding direct measurement of sample-to-sample variability.",
    "Hardware and walltime constraints. Each TC1 job is allocated a single GPU, ten gigabytes of host memory, and six hours of walltime. Sample budgets and batch sizes are sized to fit comfortably within these constraints.",
    "Sample-size-driven confidence intervals. With 200 HarmBench prompts and 250 benign XSTest prompts, binomial-proportion confidence intervals are wider than large leaderboard settings; small deltas may not be statistically separable from zero.",
    "Qwen baseline provenance. The Qwen baselines are text-only derivatives of a multimodal Qwen series. While both members of each pair inherit the same derivation, claims about \"quantization effects on Qwen\" are most safely interpreted as claims about quantization effects on these specific text-extracted derivatives.",
    "Gated-access dependency. The Llama 3.2 3B pair and HarmBench dataset depend on accepted Hugging Face access conditions and a valid token available to the TC1 environment. This precondition has been satisfied for the current run, but future reproductions must repeat the access setup.",
  ]),
];

// ------------------------------------------------------------
// Chapter 9 — Future Work
// ------------------------------------------------------------
const ch9 = [
  H1("Chapter 9 — Future Work"),
  PJ("The framework and methodology established by this study admit several natural extensions, listed in approximate order of practical impact."),
  ...numberedList([
    "Multi-method quantization comparison. Extend the matrix to include GPTQ, AWQ, and GGUF quantization paths on the same baselines, allowing direct comparison of how different PTQ algorithms perturb safety and capability.",
    "Stochastic-decoding sensitivity arm. Re-run a representative subset of pairs at temperature 0.7 with three to five distinct seeds to obtain an independent estimate of within-condition variance. This would meaningfully strengthen the statistical claims of the primary study.",
    "Expanded scale axis. Once primary results are in hand, extend the Qwen sweep to include a 0.5-billion or 7-billion parameter point, broadening the scale axis used in the RQ4 analysis.",
    "Multilingual extension. Replicate the matched-pair design in Chinese (where Qwen is natively strong) and one low-resource language, to test whether quantization-induced safety changes are language-dependent.",
    "Judge-LLM sensitivity check. Re-score saved responses with an independent judge LLM (HarmBench's official classifier, LlamaGuard, or a strict-schema API judge) and compare the resulting refusal/compliance rates against the deterministic v2 parser. This must be implemented as a derived validation layer with separate judge sidecars and redacted diagnostics, not by modifying raw.jsonl, summary.json, scores.v2.jsonl, or summary.v2.json.",
    "Safety-preserving quantization. Investigate emerging \"safety-preserving\" quantization methods that explicitly seek to mitigate alignment degradation under PTQ, and compare them against the vanilla NF4 baseline studied here.",
  ]),
];

// ------------------------------------------------------------
// Chapter 10 — Conclusion
// ------------------------------------------------------------
const ch10 = [
  H1("Chapter 10 — Conclusion"),
  PJ("This Final Year Project investigates safety–capability trade-offs in four-bit quantized compact language models, focusing on a research question that institutional benchmarks have not directly answered: when a small instruction-tuned model is quantized for on-device deployment, do observed changes in safety behaviour reflect a true shift in alignment or a side-effect of degraded general capability?"),
  PJ("The methodological contribution is a controlled matched-pair design in which baseline and four-bit pair members are loaded from identical baseline weights, with NF4 quantization applied on the fly. This design eliminates publisher- and pipeline-asymmetry as confounds and provides the strongest practical isolation of quantization as the experimental variable. The engineering contribution is an open, reproducible benchmarking framework comprising the matched-pair pipeline, three benchmark plugins, the pairwise analysis layer with rule-based interpretation labels and paired bootstrap 95% confidence intervals, full SLURM orchestration for the NTU TC1 cluster with resumable per-model matrix jobs, and a verification suite of 163 automated tests. The analytical contribution is the interpretation layer itself — the capability-anchored, statistically-grounded, multi-benchmark framework — which formalises the alignment-versus-capability disambiguation as a rule-based decision procedure over combined safety and capability deltas and which is the durable contribution of this study independent of any specific empirical outcome."),
  PJ("All six experimental runs completed on the NTU TC1 GPU cluster on 2026-05-27 (SLURM jobs 60976–60981); scoring under the v2 refusal classifier was completed on 2026-05-28. The three pairs produce three distinct diagnostic profiles. The Qwen3-1.7B pair (capability_collapse_masquerading_as_safety): MMLU accuracy falls 8.7 pp (CI [−0.137, −0.037], significant) with ΔASR within noise (−0.025, CI [−0.070, +0.020], not significant). The conjunction triggers the capability-collapse rule: substantial capability loss with no separable safety direction is the framework's clearest test case. The Qwen3-4B pair (alignment_degradation): ΔASR is positive and statistically significant (+0.065, CI [+0.025, +0.110]) while MMLU is within tolerance (−0.3 pp, not significant). This is the only HarmBench delta in the study whose CI excludes zero, and it goes in the safety-worsening direction — the most consequential empirical finding for deployment. The Llama 3.2 3B pair (broad_degradation): ΔASR is exactly zero with a symmetric noise band, ΔMMLU is significant (−0.043), and ΔOR is non-significant. Llama's strong baseline refusal calibration (ASR = 0.060) is preserved under quantization at a measurable capability cost — a different failure mode from either Qwen pair. Together the three pairs span three distinct diagnostic categories, illustrating that the framework's value is precisely its ability to separate them."),
  PJ("Taken together, the results answer all five research questions. NF4 quantization does not have a universal directional effect on harmful compliance (RQ1): only one of the three ΔASR values is statistically significant (Qwen 4B, +0.065, CI excludes zero), and the remaining two are within noise. The data therefore refute both the optimistic claim (\"quantization is safety-neutral or improving\") and the pessimistic claim (\"quantization universally degrades alignment\") — each pair tells a different story. NF4 quantization does not increase over-refusal in any evaluated model (RQ2) — the most robust null result in the study. NF4 quantization significantly degrades capability in the Qwen 1.7B and Llama 3B models but not in the Qwen 4B model (RQ3). Smaller Qwen models are more sensitive on capability, and the two Qwen failure modes are categorically distinct (capability collapse at 1.7B versus alignment degradation at 4B) (RQ4). Cross-family effects are partially consistent on capability and over-refusal directions, and not directly comparable on safety because each pair fell into a different diagnostic category (RQ5). The framework, methodology, and bootstrap CIs are available for replication and extension to additional model families, scale points, and quantization methods. The framework itself — capability-anchored, statistically grounded, multi-benchmark — is the durable contribution; the specific empirical numbers are subordinate to it."),
];

// ------------------------------------------------------------
// References
// ------------------------------------------------------------
const refs = [
  H1("References"),
  P("The following references are cited or contextually relevant. Final publication-grade formatting will be applied in the final FYP submission.", { after: 200 }),
  ...numberedList([
    "Sun, L., et al. TrustLLM: Trustworthiness in Large Language Models. arXiv:2401.05561, 2024.",
    "Wang, B., et al. DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models. NeurIPS 2023 Datasets and Benchmarks Track.",
    "Zhang, Z., et al. SafetyBench: Evaluating the Safety of Large Language Models. arXiv:2309.07045, 2023.",
    "Mazeika, M., et al. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249, 2024.",
    "Röttger, P., et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. arXiv:2308.01263, 2023.",
    "Hendrycks, D., et al. Measuring Massive Multitask Language Understanding. ICLR 2021. (MMLU)",
    "Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. (Introduces NF4 quantization and the bitsandbytes integration.)",
    "Qwen Team. Qwen Technical Report Series (Qwen 2.5 / Qwen 3.x). Alibaba DAMO Academy, 2024–2025.",
    "Meta AI. Llama 3.2 Model Family. Technical Report, 2024.",
    "Liu, S., et al. Behavioural Effects of Quantization on Instruction-Tuned Models. arXiv:2402.16775, 2024. (placeholder)",
    "Anonymous Authors. A Comprehensive Evaluation of Quantization Strategies for Large Language Models (1B–405B). arXiv:2409.11055, 2024. (placeholder)",
    "Anonymous Authors. Safety and Trustworthiness Effects of Post-Training Quantization. arXiv:2502.15799, 2025. (placeholder)",
    "Anonymous Authors. Safety-Preserving Quantization for Aligned LLMs. arXiv:2511.07842, 2025. (placeholder)",
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
    benchmarks: [harmbench, xstest, mmlu]

  qwen_2b_4bit:
    family: qwen
    size_b: 1.7
    quantized: true
    pair_id: qwen_2b
    model_id: Qwen/Qwen3-1.7B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu]

  qwen_4b_base:
    family: qwen
    size_b: 4.0
    quantized: false
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu]

  qwen_4b_4bit:
    family: qwen
    size_b: 4.0
    quantized: true
    pair_id: qwen_4b
    model_id: Qwen/Qwen3-4B
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu]

  llama_3_2_3b_base:
    family: llama
    size_b: 3.0
    quantized: false
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu]

  llama_3_2_3b_4bit:
    family: llama
    size_b: 3.0
    quantized: true
    pair_id: llama_3_2_3b
    model_id: meta-llama/Llama-3.2-3B-Instruct
    trust_remote_code: false
    dtype: auto
    benchmarks: [harmbench, xstest, mmlu]

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
  H1("Appendix A — Final TC1 Configuration"),
  PJ("The full configs/tc1.yaml is reproduced below for reference."),
  ...Code(tc1Yaml),
];

const appendixB = [
  H1("Appendix B — Example Generated SLURM Script"),
  PJ("The following is the generated sbatch script for qwen_2b_4bit, one of the six per-model jobs emitted by the cluster-generate command. The remaining five scripts differ only in the --model alias and the SBATCH --job-name, --output, and --error lines."),
  ...Code(sbatchExample),
];

const appendixC = [
  H1("Appendix C — Pydantic Configuration Schema Summary"),
  PJ("The configuration schema is defined in ethical_benchmark/quant/config_schema.py and is summarised below. All values are validated at load time; any invalid configuration raises a clear Pydantic validation error before any model is loaded."),

  H3("Top-level QuantizationConfig"),
  Bullet("study_name : str — human-readable study identifier."),
  Bullet("models : dict[str, ModelConfig] — keyed by model alias."),
  Bullet("decoding : DecodingConfig — generation parameters."),
  Bullet("benchmarks : dict[str, BenchmarkConfig] — keyed by benchmark name."),
  Bullet("slurm : SlurmConfig — cluster directives and bootstrap."),

  H3("ModelConfig"),
  Bullet("family : str — e.g. \"qwen\", \"llama\"."),
  Bullet("size_b : float — parameter count in billions."),
  Bullet("quantized : bool — true triggers NF4 loading."),
  Bullet("pair_id : str — links baseline and 4-bit members of the same pair."),
  Bullet("model_id : str — Hugging Face repo id."),
  Bullet("trust_remote_code : bool — disabled by default."),
  Bullet("dtype : str — one of \"auto\", \"float16\", \"bfloat16\", \"float32\"."),
  Bullet("benchmarks : list[str] — must reference top-level benchmark keys."),
  Bullet("revision : str | None — optional Hugging Face commit pin."),

  H3("DecodingConfig"),
  Bullet("max_new_tokens : int (≥1)"),
  Bullet("temperature : float (0.0–2.0)"),
  Bullet("top_p : float (0.0–1.0)"),
  Bullet("repetition_penalty : float (≥1.0)"),
  Bullet("max_input_tokens : int (≥1)"),
  Bullet("use_chat_template : bool"),

  H3("BenchmarkConfig"),
  Bullet("dataset_name : str — Hugging Face dataset id."),
  Bullet("split : str — dataset split."),
  Bullet("max_samples : int — sample cap."),
  Bullet("batch_size : int — generation batch size."),
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
  H1("Appendix D — Test Inventory"),
  PJ("The verification suite comprises 163 automated tests across twelve test files. The distribution is summarised in Table D.1; the test files are located under tests/ in the repository."),
  buildTable(
    ["Test file", "Test count", "Coverage area"],
    [
      ["test_datasets.py", "27", "Dataset loaders (toxicity, bias, factuality); type coercion helpers."],
      ["test_evaluators.py", "21", "Legacy evaluator parsing and aggregation."],
      ["test_matrix_reuse.py", "2", "Matrix runner reuse_loaded_model behaviour."],
      ["test_metrics_and_config.py", "20", "Bootstrap CI, JSONL/CSV round-trip, schema validation."],
      ["test_models.py", "18", "ModelSpec, device and dtype resolution, prompt formatting (incl. enable_thinking)."],
      ["test_pipeline.py", "12", "Batched generation and orchestration helpers."],
      ["test_quant_analysis.py", "9", "Pairwise delta computation, interpretation labels, bootstrap CI logic, and v2 score-sidecar selection."],
      ["test_quant_pipeline_utils.py", "4", "Record schema validation, resume helpers."],
      ["test_quant_smoke.py", "2", "End-to-end pipeline smoke test on a stub."],
      ["test_refusal_parser.py", "40", "v1 sanity tests plus v2 regression tests: 26 canonical refusal templates, 6 negative controls, curly-apostrophe handling, and the diagnostic match_refusal_pattern helper."],
      ["test_slurm_helpers.py", "8", "Generated sbatch contents and per-benchmark/per-model grouping."],
      ["(other small fixtures)", "—", "Conftest fixtures, shared mocks."],
      ["Total", "163", "All passing on the current commit."],
    ],
    [3000, 1200, 5160],
  ),
  P("Table D.1  Distribution of automated tests across modules.", { size: 18 }),
];

const appendixE = [
  H1("Appendix E — Repository Layout"),
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
├── tests/                    (163 tests across 12 files)
├── slurm/jobs_tc1/           (generated sbatch files)
├── results/                  (raw.jsonl, summary.json, v2 score sidecars, analysis outputs)
└── docs/
    ├── methodology.md
    ├── evaluation_metrics.md
    ├── datasets.md
    ├── limitations.md
    ├── extensibility.md
    ├── TC1_CLUSTER_RUNBOOK.md
    └── FYP_Report_2026-05-27.docx   (this document)`),
];

const appendixG = [
  H1("Appendix G — Document Revision History"),
  PJ("This appendix records the revision history of this FYP report. It mirrors the report-affecting subset of the project changelog (`docs/PROJECT_LOG.md` §4). Purely internal changes — refactors, tests, dev tooling — are recorded in the project log but omitted here for readability. Every entry corresponds to a regenerated docx artifact."),
  buildTable(
    ["When (UTC+8)", "Version", "Change to the report"],
    [
      ["2026-05-27 23:54", "FYP_Report_2026-05-27.docx (current)", "Added Appendix H as a fresh-session handoff after the v2 scorer incident. The appendix documents the v1 failure mode, the v2 parser correction, the later immutable-sidecar correction, the verification evidence, and the next-step contract for judge-model validation. Updated cover revision text and Future Work wording to make clear that LlamaGuard/API judge scoring is a derived validation layer rather than a replacement for v2 or a mutation of raw outputs. Documented that any local incomplete judge skeleton should not be committed until T20 is implemented end-to-end."],
      ["2026-05-28 12:00", "FYP_Report_2026-05-27.docx (current)", "T13 — Refusal classifier revision v1 → v2. The v1 deterministic regex parser missed canonical modern refusal templates (\"I can't fulfill this request\", \"I cannot provide\", \"I'm unable to assist\", curly-apostrophe variants), systematically misclassifying clear refusals as harmful compliance. Replaced with a comprehensive pattern set (forty-plus regexes; apostrophe normalisation; verb-family coverage). Added rescore script (`scripts/rescore_harmbench.py`) that re-evaluates all existing raw.jsonl files without re-running inference; emits redacted IDs-only diagnostic CSVs and an aggregate JSON. The final workflow preserves TC1-original `raw.jsonl`/`summary.json` files and stores corrected scores in derived `scores.v2.jsonl` + `summary.v2.json` sidecars. Headline impact: Qwen 4B ΔASR flipped sign from −0.045 (v1) to +0.065 (v2), with CI now excluding zero — the study's most robust empirical finding; Qwen 1.7B ΔASR collapsed from −0.120 (v1) to −0.025 (v2, within noise); Llama ΔASR moved from +0.030 (v1) to exactly 0.000 (v2). Two of three interpretation labels changed (Qwen 4B → alignment_degradation; Llama → broad_degradation). Re-ran `make analyze` for updated bootstrap CIs. All affected text rewritten throughout the report: Abstract, Table 6.1, Table 6.2, §6.1, §6.1.1 (new — scorer revision history), §6.3, §6.4, §6.5, §6.6, §6.7, §6.8, §6.9 (all five RQ answers), §6.10, §6.11 (all subsections), Ch10. Thesis reframed around the framework as the durable contribution. Test suite grew to 163 with refusal-pattern regression tests and v2 sidecar-selection coverage. T12 entry from 2026-05-28 01:00 superseded by this update."],
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
  P("Table G.1  Revision history of this FYP report. The source builder script is `scripts/build_fyp_report.js`; the docx is a build artifact and is never hand-edited.", { size: 18 }),
];

const appendixF = [
  H1("Appendix F — Glossary"),
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
  H1("Appendix H — Scorer Incident and Future-Session Handoff"),
  PJ("This appendix is included so that a future agent or fresh chat session can continue the work without relying on conversation memory. It records the critical scorer incident, the corrected artifact contract, and the next safe continuation point. The authoritative operational source remains `docs/PROJECT_LOG.md`; this appendix mirrors the same information in the Word deliverable."),

  H2("H.1 What went wrong in v1"),
  PJ("The original HarmBench and XSTest scorer used a narrow deterministic refusal parser with only fourteen regex patterns. During post-hoc review, it became clear that the parser missed common modern instruction-model refusal templates, including straight- and curly-apostrophe variants of phrases such as \"I can't fulfil this request\", \"I cannot provide\", and \"I'm unable to assist\". Those responses were clear refusals, but v1 counted them as harmful compliance on HarmBench or benign answers on XSTest. This was a scoring error, not a target-model generation error: the six TC1 jobs had already completed, and the saved model outputs did not change."),
  PJ("The consequence was material. Under v1, the report could have supported an overly optimistic interpretation of NF4 quantization, especially for Qwen 4B. Under the corrected v2 scorer, Qwen 4B changes from a point-estimate safety improvement to a statistically significant alignment_degradation result: ΔASR = +0.065 with 95% CI [+0.025, +0.110]. Qwen 1.7B's apparent ASR reduction shrinks to a non-significant −0.025, and Llama 3.2 3B's ΔASR becomes exactly 0.000. MMLU values are unchanged because MMLU scoring does not use the refusal parser."),

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
      ["scores.judge.<judge>.jsonl", "Future judge-model sidecar, not yet implemented", "Write only if T20 is implemented; keep separate from v2 sidecars."],
    ],
    [1900, 3200, 4700],
  ),

  H2("H.4 Verification already completed"),
  PJ("The corrected workflow was verified in four ways. First, the full test suite passed with 163 tests, including new v2 refusal-parser regression tests and a sidecar-selection test for the analysis layer. Second, `make analyze` reproduced the current v2 headline numbers from sidecars. Third, the rescore script was checked to leave all original `raw.jsonl` and `summary.json` hashes unchanged. Fourth, diagnostic outputs were checked for redaction: they contain IDs, old/new labels, response length, malformed/refusal flags, and matched pattern names, but no raw prompt or response text."),

  H2("H.5 Where the next session should continue"),
  PJ("The next methodological improvement is T20: judge-model validation. This should be treated as a sensitivity check rather than a replacement for the current report scorer. Acceptable options include HarmBench's official classifier, LlamaGuard, or an API judge with strict JSON output. Whichever option is chosen, the implementation must write separate judge sidecars and redacted diagnostics; it must not mutate raw outputs, v2 sidecars, or the current report numbers unless a later documented decision explicitly promotes judge scoring as the primary metric."),
  PJ("Operationally, a future TC1 session should first run `git pull --ff-only` in `/tc1home/FYP/utan001/fyp_quant/repo` to receive commit `aa451dc` or later. Do not run judge Python code on the TC1 head node; submit judge scoring through `sbatch`. If an API judge is used from the Mac instead of TC1, preserve the same artifact contract and log token/cost assumptions separately. If the judge refuses or fails to classify a benchmark prompt/response pair, record that as a judge-status field such as invalid, refused, or error, and keep the raw text out of logs. If a local untracked `ethical_benchmark/judges/__init__.py` skeleton is present, treat it as incomplete T20 scaffolding: complete the missing validation module in the same change or remove the skeleton before committing."),
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
  title: "FYP Interim Report — Benchmarking Ethical Performance of Open-Source LLMs (CCDS25-1136)",
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

const OUTPUT = "/Users/tanueihorng/fyp_quant/docs/FYP_Report_2026-05-27.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("WROTE: " + OUTPUT);
  console.log("Size: " + buf.length + " bytes");
});
