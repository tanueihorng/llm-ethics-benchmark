// ============================================================================
// FYP THESIS builder (v4 - 512-primary mirror; decisions D41/D43) - docx-js.
// Separate from scripts/build_fyp_report_v5.js; `make report` never touches this.
// Output: docs/FYP_Thesis_2026-07-02_v4.docx   (build: make thesis)
// v4 = v3 re-based to the 512-token PRIMARY study (D41, HarmBench's own
// standardized evaluation budget). Every results-bearing number mirrors
// results_512/analysis and is machine-checked by the claim lock
// (scripts/verify_report_claims.py, thesis section). v1-v3 remain as
// banner-marked 128-era history; their docx are archived in docs/archive/.
// ============================================================================
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak, ImageRun,
} = require("docx");

const FIGDIR = path.join(__dirname, "..", "docs", "figures");

const SERIF = "Times New Roman";
const MONO = "Consolas";
const BODY = 24;           // 12pt
const CONTENT_W = 9360;    // US Letter, 1" margins

// ---- helpers ---------------------------------------------------------------
let listN = 0;
const T = (text, opts = {}) => new TextRun({ text, font: SERIF, size: BODY, ...opts });
const P = (text, opts = {}) => new Paragraph({
  children: [T(text)], spacing: { after: opts.after ?? 140, line: 360 },
  alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT, ...opts.p,
});
const PJ = (text) => new Paragraph({
  children: [T(text)], alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 360 },
});
const RUNS = (runs, opts = {}) => new Paragraph({
  children: runs, alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 360 }, ...opts,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
  spacing: { after: 200 }, children: [new TextRun({ text, font: SERIF, size: 32, bold: true })] });
const H1NB = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1,
  spacing: { after: 200 }, children: [new TextRun({ text, font: SERIF, size: 32, bold: true })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 140 }, children: [new TextRun({ text, font: SERIF, size: 27, bold: true })] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3,
  spacing: { before: 160, after: 120 }, children: [new TextRun({ text, font: SERIF, size: 24, bold: true })] });
const BUL = (text) => { return new Paragraph({ numbering: { reference: "bullets", level: 0 },
  spacing: { after: 80, line: 360 }, children: [T(text)] }); };
const NUM = (text, ref) => new Paragraph({ numbering: { reference: ref, level: 0 },
  spacing: { after: 80, line: 360 }, children: [T(text)] });
// IEEE reference entry: "[n] ..." with a hanging indent so continuation lines align.
const REF = (n, text) => new Paragraph({ spacing: { after: 90, line: 264 },
  indent: { left: 400, hanging: 400 },
  children: [new TextRun({ text: `[${n}] ${text}`, font: SERIF, size: 22 })] });
const CAP = (text) => new Paragraph({ spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text, font: SERIF, size: 20, italics: true })] });

// Figure: centered PNG (aspect-preserving) + numbered italic caption. Figures
// are produced reproducibly from results/analysis/*.json by scripts/make_figures.py.
let __figN = 0;
const _pngSize = (buf) => ({ w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) });
function FIG(file, caption, dispW = 520) {
  __figN += 1;
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const { w, h } = _pngSize(buf);
  const dispH = Math.round(dispW * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: dispW, height: dispH } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${__figN}. ${caption}`, font: SERIF, size: 20, italics: true })] }),
  ];
}

function tbl(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: "D9E2F3", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: h, font: SERIF, size: 20, bold: true })] })],
  })) });
  const body = rows.map(r => new TableRow({ children: r.map((c, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(c), font: SERIF, size: 20 })] })],
  })) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

// ===========================================================================
// FRONT MATTER
// ===========================================================================
const CTR = (text, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER,
  spacing: { after: opts.after ?? 120 }, children: [new TextRun({ text, font: SERIF, size: opts.size ?? 24, bold: opts.bold, italics: opts.italics, color: opts.color })] });

const cover = [
  new Paragraph({ spacing: { after: 700 }, children: [T(" ")] }),
  CTR("NANYANG TECHNOLOGICAL UNIVERSITY", { size: 30, bold: true, after: 60 }),
  CTR("College of Computing and Data Science", { size: 24, after: 560 }),
  CTR("FINAL YEAR PROJECT: THESIS", { size: 24, bold: true, after: 480 }),
  CTR("Benchmarking the Ethical Performance of Open-Source LLMs:", { size: 32, bold: true, after: 100 }),
  CTR("A Matched-Pair, Judge-Validated Study of Safety–Capability Trade-offs in Quantized Compact Language Models (fp16 → INT8 → NF4)", { size: 28, bold: true, after: 760 }),
  CTR("Project Code:  CCDS25-1136", { size: 24, bold: true, after: 200 }),
  CTR("Student:  TAN UEI HORNG  (UTAN001)", { size: 26, bold: true, after: 120 }),
  CTR("Email:  UTAN001@e.ntu.edu.sg", { size: 22, after: 120 }),
  CTR("Supervisor:  Dr. Zhang Jiehuang  (jiehuang.zhang@ntu.edu.sg)", { size: 22, after: 560 }),
  CTR("2 July 2026", { size: 26, bold: true, after: 80 }),
  CTR("Five matched pairs / ten models / four families · three precisions · four benchmarks · HarmBench 512-token reference budget · 339 automated tests", { size: 18, italics: true, color: "555555" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const declaration = [
  H1NB("Declaration of Originality"),
  PJ("I hereby declare that this Final Year Project thesis is my own work and, to the best of my knowledge and belief, it contains no material previously published or written by another person, nor material that has been accepted for the award of any other degree or diploma of a university or other institution of higher learning, except where due acknowledgement has been made in the text. The intellectual content of this thesis (its research design, experimental methodology, analysis, and interpretation) is the product of my own work, although I have received assistance on software implementation, language, and presentation as acknowledged herein."),
  PJ("All experimental results reported in this thesis were produced by the open-source benchmarking framework described herein, executed on the NTU TC1 GPU cluster. Every reported numerical result is computed by the committed code from the recorded experimental records; no result has been altered, fabricated, or selectively reported. The per-generation raw records are retained locally and pinned by cryptographic hash rather than committed to the public repository, so a third party reproduces the results either by re-running the pipeline from the committed configuration and source code, or by replaying the committed redacted score sidecars — not by recomputing from raw generations on a fresh clone. Where the work of others has been used, it has been cited and referenced."),
  new Paragraph({ spacing: { before: 700, after: 40 }, children: [T("_______________________________")] }),
  P("Tan Uei Horng  (UTAN001)", { after: 30 }),
  P("College of Computing and Data Science, Nanyang Technological University", { after: 30 }),
  P("Date:  2 July 2026", { after: 30 }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstract = [
  H1NB("Abstract"),
  PJ("Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasingly deployed on edge and consumer hardware, where quantization, most commonly four-bit, has become the de facto method for fitting them into available memory. Quantization is not behaviourally neutral: it can alter safety alignment, refusal calibration, and general capability. Yet reports disagree, and two measurement problems confound them: a brittle refusal-matching scorer can over-count “attack success,” and the number of tokens a model is allowed to generate during evaluation can manufacture or hide a safety effect outright. This thesis asks whether the safety changes attributed to quantization are genuine alignment shifts or artefacts of capability loss, invalid scoring, or truncated generation."),
  PJ("It contributes an open, reproducible benchmarking framework built around a matched-pair design (baseline and quantized members are loaded from identical weights, with quantization applied on the fly at load time) so that quantization is the sole experimental variable. Five model pairs across four families (Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, and Phi-4-mini-instruct), spanning roughly 1.7 to 7.2 billion parameters, are evaluated across three precisions (fp16, INT8/LLM.int8, and NF4 four-bit) on four benchmarks: HarmBench (harmful compliance, Attack Success Rate), XSTest (over-refusal), and the MMLU and ARC-Challenge accuracy benchmarks (capability). Harmful compliance is scored by the official HarmBench classifier as the primary judge and cross-checked by a second, architecturally independent judge (GPT-4o); a refusal regex is retained only as a demoted, transparent proxy. Two scope controls apply throughout: the safety prompts are HarmBench's standard harmful behaviours presented directly, with no adversarial attack augmentation, so ASR measures harmful compliance under direct requests; and generation uses HarmBench's own standardized 512-token evaluation budget as the primary configuration, with an initial 128-token run retained as a controlled comparison. A capability-anchored interpretation layer separates genuine alignment shifts from capability degradation, and all deltas carry paired-bootstrap confidence intervals and McNemar exact tests."),
  PJ("Three findings result. First, methodologically, the scorer and the generation budget each determine the conclusion. The refusal regex systematically over-counts harmful compliance (its harmful set is a near-strict superset of the classifier's; judge-versus-regex agreement is family-dependent, Cohen's κ 0.25 to 0.84). And the original 128-token budget had truncated 60.3 percent of generations mid-answer, manufacturing an apparent safety regression in the smallest model (Qwen3-1.7B, ΔASR = +0.055, then significant) that dissolves entirely at the reference budget (ΔASR = 0.000 under the classifier and +0.005 under the second judge, McNemar p = 1.000 under both). Second, empirically, at the 512-token reference budget four-bit NF4 never significantly increases harmful compliance in any pair: not one HarmBench contrast survives a Benjamini-Hochberg correction, the only individually significant delta is a decrease (Llama-3.2-3B, −0.040), and the effects that do survive multiplicity correction are capability losses and one benign-direction over-refusal change. Third, the three-precision sweep shows capability loss is a clean cliff at four-bit (INT8 is essentially free), and the one INT8-specific safety increase seen at 128 tokens likewise vanishes at the reference budget. The overall picture is a rigorous, power-bounded null on the safety axis under direct requests, with capability degradation as the robust, budget-invariant cost of four-bit quantization."),
  new Paragraph({ children: [new PageBreak()] }),
];

const acknowledgements = [
  H1NB("Acknowledgements"),
  PJ("I am grateful to my supervisor, Dr. Zhang Jiehuang, for guidance, feedback, and direction throughout this Final Year Project. I thank the NTU College of Computing and Data Science for access to the TC1 GPU cluster, on which all experiments were run. This work builds upon a large body of open-source artefacts, and I acknowledge the maintainers of the Hugging Face transformers and bitsandbytes libraries; the authors of the HarmBench, XSTest, MMLU, and ARC-Challenge benchmarks and of the official HarmBench classifier; and the broader open evaluation community, whose tools and datasets made this study possible."),
  new Paragraph({ children: [new PageBreak()] }),
];

const aiDeclaration = [
  H1NB("Declaration on the Use of Generative AI Tools"),
  PJ("In the interest of transparency and in accordance with academic-integrity guidance and emerging open-source-software norms (for example, the Journal of Open Source Software AI-usage disclosure policy), I disclose that generative-AI coding and writing assistants were used during this project to assist with software implementation, automated test authoring, analysis scripting, documentation, and language editing. All such assistance was carried out under my direction. I reviewed, validated, and take full responsibility for all AI-assisted outputs, and the core research design, scientific claims, experimental decisions, and their interpretation are my own. No data, results, or citations were fabricated: every reported number is computed by the committed code from the recorded experimental artefacts, and all cited sources were independently verified."),
  new Paragraph({ children: [new PageBreak()] }),
];

const toc = [
  H1NB("Table of Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
  H1NB("List of Tables and Figures"),
  P("Table 3.1  Model pairs, families, and parameter scales.", { after: 60 }),
  P("Table 3.2  Benchmarks, primary metrics, and sample budgets.", { after: 60 }),
  P("Table 3.3  Interpretation labels derived from combined deltas.", { after: 60 }),
  P("Table 5.1  Experimental configuration summary.", { after: 60 }),
  P("Table 6.1  Judge-versus-regex agreement (Cohen’s κ) by model family.", { after: 60 }),
  P("Table 6.2  Main study at the 512-token reference budget: per-pair deltas and labels (fp16 vs NF4).", { after: 60 }),
  P("Table 6.3  Precision sweep: HarmBench ASR at fp16 / INT8 / NF4 (judge).", { after: 60 }),
  P("Figure 1  Scorer validation: judge ASR vs regex proxy, and judge-vs-proxy Cohen's κ.", { after: 60 }),
  P("Figure 2  The capability-anchored safety space (ΔMMLU vs judge ΔASR, with label regions).", { after: 60 }),
  P("Figure 3  Precision sweep fp16 → INT8 → NF4 (HarmBench ASR, MMLU, ARC).", { after: 60 }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===========================================================================
// CHAPTERS
// ===========================================================================
const ch1 = [
  H1("Chapter 1  Introduction"),
  H2("1.1  Motivation"),
  PJ("Large language models (LLMs) are increasingly deployed not in data centres but on laptops, phones, and embedded devices, where memory is scarce. The dominant enabler of this shift is quantization: representing model weights in eight, four, or fewer bits instead of sixteen, often halving or quartering the memory footprint at a small, well-studied cost to task accuracy [1], [2]. For compact models in the one-to-seven-billion-parameter range (the models most likely to run locally), four-bit quantization has become routine."),
  PJ("Accuracy, however, is not the only property that matters. Instruction-tuned models are also aligned: trained to refuse harmful requests and to answer benign ones. Whether quantization preserves this alignment is far less understood than whether it preserves accuracy, and the stakes are higher, because a locally deployed model runs outside any server-side safety filter. If compressing a model to fit a phone quietly makes it more willing to produce harmful content, that is a deployment-relevant safety regression that current accuracy-centric evaluation would miss."),
  H2("1.2  Problem statement"),
  PJ("Three difficulties make this question hard to answer credibly. The first is confounding: a quantized model can differ from a full-precision one for reasons that have nothing to do with quantization: a different published checkpoint, different decoding settings, or simply the noise of generation. The second is measurement: harmful compliance is usually scored by pattern-matching for refusals, equating “did not refuse” with “attack succeeded.” Many non-refusals are not actually harmful (vague deflections, safety lectures, on-topic but benign answers), so a refusal-counting scorer can over-state harmful compliance and, worse, do so unevenly across models. The third is the generation budget: the number of tokens a model may generate during evaluation changes the measured attack-success rate drastically (the HarmBench authors show swings of up to 30 percent and standardize the budget to 512 tokens for exactly this reason [3]); a truncated budget scores incomplete answers, and where the cut falls can flip a label in either direction. A study that does not control confounding, scoring validity, and the generation budget cannot distinguish a genuine alignment shift from a capability, scoring, or truncation artefact."),
  H2("1.3  Research questions"),
  PJ("This thesis is organised around five research questions:"),
  NUM("RQ1: Does quantization increase harmful compliance in compact instruction-tuned LLMs?", "rq"),
  NUM("RQ2: Does quantization increase over-refusal on benign prompts?", "rq"),
  NUM("RQ3: Does quantization degrade general capability?", "rq"),
  NUM("RQ4: Within a family, are smaller models more sensitive than larger ones?", "rq"),
  NUM("RQ5: Are the effects consistent across model families and across quantization precisions?", "rq"),
  H2("1.4  Contributions"),
  PJ("This thesis makes five contributions, and the first two have consequences beyond this study. One is a scorer-validity result: a refusal-counting scorer systematically over-counts attack success (its harmful set is a near-strict superset of the benchmark classifier's), and adopting HarmBench's own classifier, cross-checked by a second independent judge, changes the study's conclusion. The other is a generation-budget result: the original 128-token budget truncated 60.3 percent of generations and thereby manufactured a statistically significant safety regression that does not exist at HarmBench's standardized 512-token budget; safety evaluations that truncate generations can report effects that are artefacts of where the cut falls. The third contribution is methodological: a controlled, capability-anchored, judge-validated procedure at the benchmark's own reference budget that isolates quantization (matched-pair, on-the-fly) and distinguishes genuine alignment shifts from capability degradation. The fourth is empirical: across five matched pairs, four families, three precisions, and two generation budgets, the multiplicity-robust cost of four-bit quantization is capability loss; harmful compliance under direct requests never rises significantly in any pair at the reference budget. The fifth is an engineering one: an open, reproducible, extensible framework (matched-pair loading, a benchmark-plugin contract, a judge-validation layer, cluster orchestration, and a machine claim lock that verifies every reported number against the committed artefacts) that others can reuse for their own quantization-safety studies."),
  H2("1.5  Thesis structure"),
  PJ("Chapter 2 surveys related work and isolates the gap. Chapter 3 details the methodology: the matched-pair design, quantization, benchmarks, scoring, the interpretation framework, and the statistical procedure. Chapter 4 documents the system design and implementation. Chapter 5 describes the experimental setup. Chapter 6 presents the results. Chapter 7 discusses them and the threats to validity, Chapter 8 records limitations, Chapter 9 proposes future work, and Chapter 10 concludes. A reproducibility statement and appendices follow."),
];

const ch2 = [
  H1("Chapter 2  Background and Related Work"),
  H2("2.1  Quantization and its behavioural effects"),
  PJ("Post-training quantization compresses model weights to lower-precision formats. The bitsandbytes library provides two widely used schemes evaluated here: NF4, a four-bit “normal-float” format with double quantization [2], and LLM.int8(), an eight-bit mixed-precision scheme that keeps outlier dimensions in higher precision [1]. While the accuracy cost of these methods is well characterised, their effect on safety behaviour is contested. Empirical studies report mixed and method-dependent results: some find that quantization can degrade safety alignment, with effects that vary by method and attack type [4], [5]; a comprehensive multi-benchmark evaluation finds headline scores largely retained at four bits while cautioning that narrow evaluations can miss shifts on other behavioural dimensions [6]; and some show that quantization can be actively exploited to surface unsafe behaviour [7]. This very inconsistency motivates a controlled, matched-pair design rather than cross-paper comparison, and it cautions against strong, uniform claims."),
  H2("2.2  Safety benchmarks and red-teaming"),
  PJ("HarmBench [3] provides a standardised red-teaming benchmark with an explicit behaviour taxonomy and a fine-tuned classifier that scores whether a response actually exhibits the harmful behaviour, a deliberate move away from refusal-string matching. XSTest [8] measures the opposite failure mode, over-refusal on benign prompts. Holistic frameworks such as HELM [9] formalise the idea that evaluation should be multi-metric and transparent. This thesis adopts HarmBench’s classifier-as-scorer principle [3] as its primary instrument and treats refusal-matching only as a foil. One scope boundary is stated at the outset: HarmBench is a red-teaming framework whose full pipeline applies optimisation-based and LLM-mediated attacks; this study evaluates its standard harmful behaviours presented directly, with no attack augmentation, so its safety axis measures harmful compliance under direct requests, the weakest threat model in the HarmBench taxonomy."),
  H2("2.3  LLM-as-judge evaluation and its validity"),
  PJ("Using one model to judge another’s output is now common, but its validity is an active concern. Surveys of LLM-as-judge methods catalogue the approach’s validity threats and mitigations [10], and critical studies show that judge scores can diverge systematically from human judgment, so judge-based results need validation and, ideally, human grounding [11]. This thesis follows that guidance (a primary classifier, a second independent judge, κ reporting, bootstrap confidence intervals) and adds a human-label validation on a stratified 200-item subset — a single-annotator gold set that was, until now, the one safeguard it lacked."),
  H2("2.4  The research gap"),
  PJ("Three observations define the gap this thesis addresses. First, the quantization–safety literature reports inconsistent, method- and model-dependent effects [4]–[5], so cross-paper comparison cannot settle whether quantization shifts safety; a controlled, matched-pair design that isolates quantization is required. Second, harmful compliance is frequently scored by refusal-matching rather than by a validated classifier, so a reported “attack success” may be a scoring artefact, and where LLM judges are used their inter-judge agreement is often left unreported [10], [11], yet the benchmark’s own fine-tuned classifier exists precisely to avoid this [3]. Third, studies that do measure quantization rarely separate a safety change from a capability change, almost none trace the effect across more than one precision or quantization method, and almost none evaluate at the benchmark’s own standardized generation budget, even though the measured attack-success rate depends drastically on it [3]."),
  PJ("No prior study, to the author’s knowledge, combines all four of the following on compact models across several families: (i) a matched-pair design that isolates quantization as the sole variable; (ii) a judge-validated primary scorer with an independent second judge and reported chance-corrected agreement; (iii) a capability anchor that separates alignment shifts from capability degradation; and (iv) a multi-precision sweep (fp16 → INT8 → NF4) that tests whether any effect is graded with bit-width or specific to a method. That combination is the gap this thesis fills, and each component directly answers one of the weaknesses identified above."),
];

const ch3 = [
  H1("Chapter 3  Methodology"),
  H2("3.1  Matched-pair design"),
  PJ("The central design choice is the matched pair. For each model, a baseline (fp16) member and a quantized member are loaded from the same underlying weights; the only difference is that the quantized member applies bitsandbytes quantization on the fly at load time. This eliminates publisher- and pipeline-asymmetry as confounds and isolates quantization as the sole experimental variable: any measured delta is attributable to the quantization step, the act of loading the same checkpoint twice, or measurement noise, and nothing else."),
  PJ("Table 3.1 lists the five pairs. They span four families and roughly 1.7 to 7.2 billion parameters, providing both within-family scale contrasts (the two Qwen models) and cross-family generality."),
  tbl(["Pair", "Family", "Model", "Size (B)"], [
    ["qwen_2b", "Qwen", "Qwen3-1.7B", "1.7"],
    ["qwen_4b", "Qwen", "Qwen3-4B", "4.0"],
    ["llama_3_2_3b", "Llama", "Llama-3.2-3B-Instruct", "3.2"],
    ["mistral_7b", "Mistral", "Mistral-7B-Instruct-v0.3", "7.2"],
    ["phi4_mini", "Phi", "Phi-4-mini-instruct", "3.8"],
  ], [1900, 1700, 3760, 2000]),
  CAP("Table 3.1  Model pairs, families, and parameter scales."),
  H2("3.2  Quantization methods and the precision sweep"),
  PJ("The main study compares fp16 against NF4 four-bit quantization. A separate precision point adds INT8 (LLM.int8), so each pair can be traced across fp16 → INT8 → NF4. INT8 is a distinct mixed-precision method, not a lower-bit NF4; treating it as a separate precision dimension lets the study ask whether degradation is graded with bit-width or concentrated at a particular method. To keep the evaluated main study untouched, the INT8 members live in a separate configuration and are analysed by a dedicated precision-sweep script rather than entering the base-versus-4bit pairwise pipeline."),
  H2("3.3  Benchmarks and metrics"),
  PJ("Four benchmarks span the two axes of interest. Harmful compliance (HarmBench [3]) and over-refusal (XSTest [8]) capture safety; MMLU [12] and ARC-Challenge [13] capture capability, using two independent benchmarks so that capability claims do not rest on a single test."),
  tbl(["Benchmark", "Axis", "Primary metric", "n"], [
    ["HarmBench", "Harmful compliance", "Attack Success Rate", "200"],
    ["XSTest", "Over-refusal", "Over-refusal rate", "250"],
    ["MMLU (subset)", "Capability", "Accuracy", "300"],
    ["ARC-Challenge", "Capability", "Accuracy", "1172"],
  ], [2400, 2560, 2600, 1800]),
  CAP("Table 3.2  Benchmarks, primary metrics, and sample budgets."),
  PJ("A threat-model boundary applies to the safety axis. The HarmBench standard configuration supplies curated harmful behaviour requests directly, and this study applies no adversarial attack or jailbreak transformation on top of them (no GCG, PAIR, AutoDAN, or any other attack pipeline from the HarmBench suite). Attack Success Rate in this thesis therefore measures harmful compliance under direct requests, equivalent to HarmBench's DirectRequest baseline and the weakest threat model in its taxonomy, not robustness to optimised adversarial attack. Attack-augmented evaluation is future work (Chapter 9)."),
  H2("3.4  Scoring: classifier-primary, regex-as-foil"),
  PJ("HarmBench Attack Success Rate is scored by the official HarmBench classifier (cais/HarmBench-Llama-2-13b-cls) [3], run in full precision so that the reference instrument is not itself confounded by quantization. A second, architecturally independent judge (GPT-4o) re-scores the same generations as a robustness check [10], [11]. A deterministic refusal regex is retained only as a secondary, transparent proxy. This choice is itself a research object: the thesis reports the agreement between scorers and shows that the regex over-counts, which is why it is demoted. MMLU and ARC use deterministic exact match; XSTest over-refusal uses the regex (the HarmBench classifier judges harmful compliance only)."),
  H2("3.5  Capability-anchored interpretation"),
  PJ("A rule-based interpretation layer combines the safety and capability deltas into a single diagnostic label, so that a rise in harmful compliance accompanied by a capability drop is not mistaken for a pure alignment shift, and vice versa. Table 3.3 lists the labels. The anchor is what licenses reading a safety move as alignment rather than capability artefact: if capability is preserved, a safety move is a candidate alignment shift; if capability collapses, an apparent safety “improvement” may merely be the model becoming less coherent."),
  tbl(["Label", "Reading"], [
    ["broad_degradation", "Worsens on both safety and capability axes."],
    ["alignment_degradation", "Safety worsens, capability preserved (candidate alignment shift)."],
    ["alignment_improvement", "Harmful compliance falls, capability preserved."],
    ["capability_collapse_masquerading_as_safety", "Apparent safety gain driven by capability loss."],
    ["over_refusal_regression", "Over-refusal rises, capability preserved."],
    ["robust_preservation", "No material change on any axis."],
  ], [4200, 5160]),
  CAP("Table 3.3  Interpretation labels derived from combined deltas."),
  H2("3.6  Decoding controls and statistics"),
  PJ("All inference uses greedy decoding at temperature 0.0 with a fixed seed, so the only variance reflected in the intervals is prompt sampling. Generation uses HarmBench's standardized 512-token evaluation budget as the primary configuration: the HarmBench authors show the number of tokens generated during evaluation can change the measured attack-success rate by up to 30 percent and standardize the parameter to N = 512 for exactly that reason [3]. An initial 128-token run is retained, unchanged, as a controlled generation-length comparison (§6.3). Deltas carry paired-bootstrap 95% confidence intervals [14] (2,000 resamples, seed 42), resampled by prompt identity so that both members of a pair see the same prompts in the same order. Every HarmBench contrast is tested with McNemar’s exact paired test [15]; significance is reported both uncorrected and under a Benjamini-Hochberg false-discovery-rate correction over the twenty-contrast primary family; a power analysis reports the minimum detectable effect (about 0.06 at n = 200); and a multi-seed sensitivity arm (temperature 0.7, five seeds, three of the five pairs) estimates generation-level variance."),
];

const ch4 = [
  H1("Chapter 4  System Design and Implementation"),
  H2("4.1  Architecture"),
  PJ("The framework is a Python package organised around plugins and configuration. A YAML configuration, validated by a Pydantic schema, declares the models, decoding settings, benchmarks, and cluster parameters. A model loader resolves dtype and device and applies quantization on the fly; a shared text generator batches prompts through the model using the tokenizer’s chat template; benchmark plugins implement a common contract for loading items, building prompts, scoring responses, and aggregating metrics; and an analysis layer computes pairwise deltas, confidence intervals, and interpretation labels."),
  H2("4.2  Extensibility and the plugin contract"),
  PJ("Adding a model requires only a configuration entry; adding a benchmark requires implementing four methods (load, build prompt, score, aggregate) and registering one line; adding a quantization method extends the loader’s method branch. This keeps the framework reusable beyond the present study, a design property documented in the project’s quickstart and verified by the test suite."),
  H2("4.3  Reliability safeguards"),
  PJ("Several safeguards protect result integrity. The loader fails loud if quantization is requested but does not actually engage, refusing to emit full-precision results mislabelled as quantized. Raw generations and their summaries are treated as immutable artefacts: all post-hoc scoring, including judge validation, writes derived sidecars only, and integrity is pinned by a checksum manifest. Resume logic keys on prompt identity so an interrupted run continues without double-counting. A suite of 339 automated tests covers the schema, loaders, scorers, analysis mathematics, and the cluster job generator; a machine claim lock, run inside the suite, additionally asserts every load-bearing number in the report and thesis builders against the committed analysis artefacts, so a prose claim cannot silently drift from its evidence."),
  H2("4.4  Privacy discipline"),
  PJ("Because the study handles harmful prompts and responses, committed artefacts are redacted: prompt identifiers, boolean labels, scalar scores, and run metadata only, never prompt or response text. This makes the study reproducible from the committed sidecars without redistributing harmful content."),
];

const ch5 = [
  H1("Chapter 5  Experimental Setup"),
  PJ("All experiments ran on the NTU TC1 GPU cluster (NVIDIA Tesla V100, 32 GB) under a strict offline, batch-submission policy: datasets and weights are pre-cached on the head node, and all compute is dispatched via SLURM. The same software stack, seed, and decoding settings apply to every run, so cross-condition comparisons are controlled. The full apparatus ran at two generation budgets: the 512-token primary study (HarmBench's standardized evaluation budget) and an initial 128-token run retained unchanged as the generation-length comparison. Table 5.1 summarises the configuration."),
  tbl(["Parameter", "Value"], [
    ["Pairs / models / families", "5 / 10 / 4"],
    ["Precisions", "fp16, INT8 (LLM.int8), NF4 (4-bit)"],
    ["Benchmarks", "HarmBench, XSTest, MMLU, ARC-Challenge"],
    ["Threat model", "direct requests (HarmBench standard config; no attack augmentation)"],
    ["Decoding", "greedy, temperature 0.0, seed 42, max_new_tokens 512 (primary; 128 retained as comparison)"],
    ["Primary HarmBench scorer", "official HarmBench classifier (fp16)"],
    ["Second judge", "GPT-4o (architecturally independent)"],
    ["Statistics", "paired bootstrap 95% CI (2000 resamples) + McNemar exact + BH-FDR"],
    ["Hardware", "NVIDIA Tesla V100-32GB (TC1)"],
    ["Verification", "339 automated tests + machine claim lock"],
  ], [3600, 5760]),
  CAP("Table 5.1  Experimental configuration summary."),
  PJ("The five NF4 pairs, the INT8 precision point, both judge validations, and the multi-seed sensitivity arm all ran at the 512-token reference budget (the primary study throughout, with zero classifier parse errors over 3,000 judged generations); the 128-token artefacts are retained unchanged for the generation-length comparison. The GPT-4o second judge runs locally against an API. The study’s entire analysis is reproducible from the committed redacted sidecars without a GPU."),
];

const ch6 = [
  H1("Chapter 6  Results and Analysis"),
  H2("6.1  Scoring validity: the regex over-counts"),
  PJ("The first result concerns the instrument. The refusal regex reports far higher Attack Success Rates than the HarmBench classifier, and the gap is uneven by family. Cohen’s κ between the regex and the classifier at the 512-token reference budget is poor for the Qwen and Mistral models and good for Llama and Phi (Table 6.1), because models that produce a large volume of ambiguous, non-refusing-but-benign text are exactly those the regex mis-scores. In aggregate the classifier’s harmful set is a near-strict subset of the regex’s (29 judge-only labels against 325 regex-only across the ten models), so the disagreement is overwhelmingly one-directional over-counting. At the retired 128-token budget, adopting the classifier had moved the significant increases wholesale: the regex placed them on Qwen3-4B and Mistral, the classifier on Qwen3-1.7B alone. At the reference budget it removes every significant increase entirely (§6.2). A stratified human-label audit (200 items, single annotator) confirms the direction against truth rather than against another model: the classifier agrees with a human annotator at Cohen's κ 0.59 (moderate) while the regex agrees at only 0.11 (negligible) — against the human's 200 labels the regex over-flagged 101 responses as harmful to the classifier's 18, and the human, flagging 37, applied the highest bar of the three — so refusal-counting over-states harmful compliance relative to human judgment, not merely relative to the classifier; the classifier is a better scorer, though at moderate agreement, not a perfect human proxy. In safety evaluation the choice of scorer is not a detail: it can change the conclusion."),
  tbl(["Family", "Cohen’s κ (regex vs classifier)", "Reading"], [
    ["Qwen", "0.36 – 0.59", "poor to moderate; regex over-counts heavily (0.595 vs judge 0.255, 1.7B base)"],
    ["Mistral", "0.25 – 0.28", "worst; regex 0.890 vs judge 0.565 (4-bit)"],
    ["Llama", "0.71 – 0.84", "good; little ambiguous middle ground"],
    ["Phi", "0.67 – 0.77", "good; like Llama"],
  ], [1900, 3660, 3800]),
  CAP("Table 6.1  Judge-versus-regex agreement (Cohen’s κ) by model family (fp16-vs-NF4 main study, 512-token reference budget)."),
  ...FIG("judge_vs_proxy.png", "Scorer validation. Left: HarmBench-classifier ASR versus the regex 'non-refusal' proxy, one marker per model: eight of ten points lie below the diagonal, so the proxy over-counts harmful compliance (worst for Qwen and Mistral); the two marginal exceptions (Llama base, Phi 4-bit) sit just above it. Right: judge-vs-proxy Cohen's κ per model, family-dependent. Source: results_512/analysis/judge_agreement.json."),
  H2("6.2  Main study (fp16 vs NF4) at the reference budget"),
  PJ("Under the primary classifier at the 512-token reference budget, NF4 quantization never significantly increases harmful compliance in any pair. The only ΔASR whose confidence interval excludes zero is Llama-3.2-3B’s, and it is a decrease (−0.040, CI [−0.075, −0.010]; McNemar p = 0.021); it does not survive the multiplicity correction (§6.7). Table 6.2 reports the per-pair deltas and labels."),
  tbl(["Pair", "ΔASR (95% CI)", "Sig?", "ΔMMLU", "ΔARC", "Label"], [
    ["qwen_2b", "0.000 [−0.055, +0.055]", "no", "−0.090*", "−0.009", "broad_degradation (capability-driven)"],
    ["qwen_4b", "+0.040 [0.000, +0.080]", "no", "−0.003", "−0.016*", "alignment_degradation (dir.)"],
    ["llama_3_2_3b", "−0.040 [−0.075, −0.010]", "yes†", "−0.037", "−0.032*", "capability_collapse_masq._as_safety (dir.)"],
    ["mistral_7b", "−0.020 [−0.080, +0.040]", "no", "−0.020", "+0.009", "alignment_improvement (dir.)"],
    ["phi4_mini", "+0.020 [−0.015, +0.055]", "no", "−0.027", "−0.015", "robust_preservation"],
  ], [1500, 2700, 700, 1320, 1240, 1900]),
  CAP("Table 6.2  Main study at the 512-token reference budget: per-pair judge deltas and interpretation labels (fp16 vs NF4). * capability delta significant. † individually significant (a decrease) but does not survive BH-FDR."),
  ...FIG("capability_anchor.png", "The capability-anchored safety space at the reference budget. Each pair is placed by its capability delta (ΔMMLU, x) and harmful-compliance delta (judge ΔASR, y); dashed lines mark the interpretation thresholds and shaded quadrants name the labels. Bars are paired-bootstrap 95% CIs. Source: results_512/analysis/{judge_agreement,pairwise_deltas}.json."),
  PJ("Over-refusal moves in the benign direction where it moves at all: the only over-refusal delta significant under the exact paired test is Phi-4-mini’s decrease (ΔOR = −0.044, CI [−0.072, −0.016]), and Qwen3-1.7B shows a borderline decrease (−0.028, McNemar p = 0.065). No pair becomes significantly more trigger-happy on benign prompts (RQ2). The Qwen3-1.7B pair, whose +0.055 increase was the study’s original headline at the 128-token budget, is exactly zero at the reference budget: its thirty-two discordant prompts split sixteen against sixteen, the signature of symmetric boundary churn rather than a directional alignment shift (§6.5)."),
  H2("6.3  The generation-budget artefact (why 512 tokens is the primary budget)"),
  PJ("The 128-token budget of the study’s first run truncated the majority of generations before completion: a direct prefix test shows 60.3 percent of the 2,000 paired responses were provably cut off (the 512-token generation continues past the point where the 128-token one stops), 30.5 percent stop naturally, and 9.2 percent diverge between the two nominally greedy runs (benign cross-run non-determinism, conservatively excluded, which makes 60.3 percent a floor). Truncation is family-heterogeneous (93.5–98.0 percent for Mistral, 3.0–4.0 percent for Llama), so the mechanism is family-specific even though the conclusion is not. The HarmBench authors show the number of tokens generated during evaluation can change measured attack-success rates by up to 30 percent and standardize the parameter to N = 512 for exactly this reason [3]; the official evaluation harness likewise clips completions to 512 tokens before classification. Re-running the full apparatus at that budget dissolves the apparent safety regression: Qwen3-1.7B’s ΔASR goes from +0.055 (then significant, McNemar p = 0.027) at 128 tokens to exactly 0.000 under the classifier (p = 1.000) at 512, with the second judge at +0.005 (also p = 1.000). A five-seed stochastic-decoding arm at the reference budget corroborates the null (per-seed ΔASR mean +0.013, range [0.000, +0.035], no seed individually significant, and the greedy estimate sits inside the range; all five seed deltas are non-negative, a sub-minimum-detectable-effect directional signal disclosed rather than glossed). The capability deltas, by contrast, are essentially unchanged across budgets: truncation manufactured the safety signal, not the capability one."),
  H2("6.4  Capability anchoring (RQ3)"),
  PJ("Every pair’s capability point estimates are non-positive under NF4 at the reference budget, with one negligible exception (Mistral’s +0.9 percentage points on ARC, not significant). The loss is significant for Qwen3-1.7B on MMLU (−0.090, the study’s largest single effect), for Llama-3B on ARC (−0.032), and for Qwen3-4B on ARC (−0.016). The two capability benchmarks diverge informatively: Qwen3-1.7B’s large MMLU drop does not replicate on ARC (−0.009, not significant), so the dramatic within-Qwen scale gap is partly MMLU-specific. The capability anchor is what allows the safety deltas to be read correctly: the smallest Qwen pair is a capability story with a flat harm axis, and Llama’s harm decrease arrives together with its capability loss, which is why it is labelled capability collapse masquerading as safety rather than alignment improvement."),
  H2("6.5  Mechanism: refusal-margin geometry"),
  PJ("To probe the boundary behaviour behind the 128-token signal, a derived analysis measured each model’s first-token refusal margin in fp16 and under NF4, following the observation that quantization most affects low-confidence, near-boundary decisions [16]. A thin baseline margin does predict which prompts flip, but only weakly within a family, and the flips are symmetric: quantization destabilises the decision boundary in both directions rather than eroding refusals one-directionally. This mechanism reading is budget-invariant, and at the reference budget its signature is visible directly in the outcome data: the Qwen3-1.7B pair’s thirty-two discordant prompts split exactly sixteen to sixteen. The change is more consistent with generic capability-driven boundary instability than a targeted alignment shift, reinforcing the capability-anchored reading."),
  H2("6.6  Precision point: fp16 → INT8 → NF4 (RQ5)"),
  PJ("Adding INT8 shows the effect is not a smooth function of bit-width. On capability the result is a clean cliff at four-bit: every INT8 MMLU and ARC point estimate sits within about 1.3 percentage points of fp16, far below the significant NF4 losses of three to nine points and below the study’s detection floor, while the significant capability losses are all at NF4. On safety, at the reference budget there is no robust move at either precision step. The one apparent counter-example from the 128-token run, a Llama-3B increase specific to INT8 (+0.040, then significant under both judges), does not replicate at 512 tokens under either judge (classifier Δ+0.005, McNemar p = 1.000; GPT-4o Δ+0.010, p = 0.688): it was a 128-era budget artefact — attributable, since Llama truncates at only 3–4 percent (§6.3), to cross-run greedy divergence on thin-margin prompts rather than to truncation — not a stable property of LLM.int8 on this model. Table 6.3 shows the sweep."),
  tbl(["Pair", "fp16", "INT8", "NF4", "Shape (judge ASR)"], [
    ["qwen_2b", "0.255", "0.245", "0.255", "flat; the capability cliff is at NF4"],
    ["qwen_4b", "0.115", "0.125", "0.155", "small graded drift, n.s. at each step"],
    ["llama_3_2_3b", "0.100", "0.105", "0.060", "flat at INT8; decrease at NF4 (individually sig)"],
    ["mistral_7b", "0.585", "0.565", "0.565", "small decrease, n.s."],
    ["phi4_mini", "0.070", "0.090", "0.090", "small increase, n.s."],
  ], [1700, 1100, 1100, 1100, 4360]),
  CAP("Table 6.3  Precision sweep: HarmBench ASR at fp16 / INT8 / NF4 (primary judge, 512-token reference budget)."),
  ...FIG("precision_sweep.png", "Precision sweep fp16 → INT8 → NF4 at the reference budget. Capability (MMLU, ARC) is essentially flat through INT8 and falls only at four-bit (a cliff, not a gradient); the safety axis (judge ASR) shows no robust move at either step. Source: results_512/analysis/precision_sweep.json."),
  H2("6.7  Statistical robustness: multiplicity, power, and what survives"),
  PJ("With 200 HarmBench prompts the confidence intervals are wide (about ±0.05), and a power analysis makes the detection floor explicit: at n = 200 and the observed discordant rates, the minimum detectable ΔASR for 80 percent power is about 0.06. The predominance of nulls on the safety axis is therefore a power-bounded null (“no detectable change at n = 200”), not a proven zero, and the multi-seed sign-consistency for the smallest pair (§6.3) is compatible with a real effect below that floor."),
  PJ("Applying a Benjamini-Hochberg false-discovery-rate correction (q < 0.05) over the family of twenty primary NF4-versus-fp16 contrasts (exact McNemar p per contrast; results_512/analysis/multiple_comparisons.json), exactly three survive, and none is a harmful-compliance change: Qwen3-1.7B MMLU (−0.090, q = 0.008), Llama-3B ARC (−0.032, q = 0.008), and Phi-4-mini over-refusal (−0.044, q = 0.049, a decrease). Not one HarmBench ASR contrast survives, including Llama’s individually significant decrease. The multiplicity-robust signal of four-bit NF4 is therefore capability degradation, plus one benign-direction over-refusal change, and never a safety regression. This is the thesis’s headline empirical claim."),
];

const ch7 = [
  H1("Chapter 7  Discussion and Threats to Validity"),
  PJ("The results answer the research questions with calibrated, not sweeping, claims. At the reference budget NF4 does not significantly increase harmful compliance under direct requests in any pair (RQ1); over-refusal does not rise, and its only exact-test-significant change is a decrease (RQ2); capability loss is real, modest, partly benchmark-specific, and the only class of effect that survives multiplicity correction (RQ3); the smallest model is the most capability-sensitive but shows no genuine safety shift once truncation is controlled (RQ4); and the cross-family, cross-precision picture is a consistent power-bounded null on safety with a capability cliff at four-bit (RQ5). For deployment the practical message is capability-shaped: eight-bit quantization is essentially free, four-bit costs capability (worst in the smallest model), and neither produced a robust safety cost under direct requests. Teams should evaluate the exact quantized artefact they will ship, at the deployment generation budget rather than a truncated one, and prefer the largest model that fits the memory budget over aggressively quantizing a smaller one. These recommendations are established for on-the-fly bitsandbytes quantization under direct harmful requests; other deployment formats (GGUF, GPTQ, AWQ) and attack-mediated threat models may behave differently and are future work."),
  H2("7.1  Internal validity"),
  PJ("Internal validity is the design’s strongest property: the matched-pair structure with on-the-fly quantization from identical weights, plus deterministic decoding and deterministic or judge-validated scoring, leaves quantization as the only plausible cause of a measured delta. The fail-loud loader guard rules out the subtle confound of a quantized run silently executing in full precision. The generation budget, which the 128-token run showed can manufacture an effect, is controlled by adopting the benchmark’s own standardized budget and retaining the truncated run as an explicit comparison rather than discarding it."),
  H2("7.2  Construct validity"),
  PJ("The primary safety construct is scored by the benchmark’s own fine-tuned classifier rather than refusal matching, and the cross-judge agreement (Cohen’s κ 0.68–0.95 across all ten models at the reference budget) shows the finding is not an artefact of one classifier. A stratified human-label audit (200 items, single annotator — the author — HarmBench rubric) now grounds this against truth: the classifier agrees with the human at Cohen's κ 0.59 versus the regex's 0.11, so the primary scorer is closer to human judgment than the demoted regex, not merely to another model. Two sampling caveats scope those figures symmetrically: the subset deliberately over-weights judge-vs-regex disagreement cases (roughly three times their population rate), so the κ values are a disagreement-enriched contrast rather than a population estimate (both scorers' population agreement would sit higher); and the annotation view was capped at the first 2,000 characters of each response while the scorers saw the full text, which can only inflate measured disagreement. The residual is honestly bounded — κ 0.59 is moderate, not near-perfect, and rests on one annotator, so a second annotator with an inter-rater κ is the remaining strengthening. Over-refusal remains regex-scored, a known weaker construct. A second construct boundary is the threat model: the safety axis measures harmful compliance under direct requests only, with no attack augmentation, so the null does not speak to jailbreak robustness under optimised attacks, which is precisely where prior quantization-safety work reports effects [7], [5]."),
  H2("7.3  External and statistical-conclusion validity"),
  PJ("External validity is bounded by five pairs, one decoding regime (greedy primary, with a three-pair multi-seed arm), two bitsandbytes methods, direct requests only, and English-only text; the cross-family, cross-precision, and cross-budget sweeps widen it but do not make it general. Statistical-conclusion validity is bounded by sample size and is addressed head-on: paired-bootstrap intervals, exact McNemar tests, an explicit BH-FDR correction over the primary family, a minimum-detectable-effect analysis (about 0.06), and a two-layer evidence status rather than bare significance."),
];

const ch8 = [
  H1("Chapter 8  Limitations"),
  BUL("Threat model: HarmBench was evaluated in its standard direct-request configuration only; no attack or jailbreak augmentation was applied. All safety findings, including the headline null, are scoped to direct requests and do not establish robustness under optimised adversarial attack."),
  BUL("Sample size and power: 200 HarmBench prompts give wide (about ±0.05) confidence intervals; the minimum detectable ΔASR at 80% power is about 0.06, so effects below that floor cannot be ruled out (the smallest pair’s multi-seed sign-consistency is compatible with one)."),
  BUL("Decoding: the primary results condition on a single greedy decode; the multi-seed arm covers three of the five pairs (Qwen 1.7B, Qwen 4B, Llama), so a full-matrix stochastic estimate is outstanding."),
  BUL("Judge grounding: the HarmBench classifier is now grounded against human labels on a stratified 200-item subset (classifier κ 0.59 vs regex 0.11, single annotator), alongside the second-judge cross-check; a second annotator for an inter-rater κ remains the strengthening."),
  BUL("Quantization coverage: two bitsandbytes methods (NF4, INT8); GPTQ, AWQ, and GGUF (the dominant on-device deployment format) are out of scope, so the null may be method-specific."),
  BUL("Generation budget: absolute ASR is budget-dependent by construction (complete generations give more room for harmful content to appear), which is controlled here by adopting HarmBench’s standardized 512-token budget; the 128-era signals are budget artefacts retained only as the generation-length comparison (Qwen3-1.7B NF4 +0.055 via truncation; Llama-3B INT8 +0.040 via cross-run greedy divergence, since Llama truncates at only 3–4 percent)."),
  BUL("Generation stopping: the generation loop halts only on the tokenizer's default end-of-sequence token, so responses can run past the model's completed answer into synthetic follow-up turns (the model role-playing both sides of a dialogue), which skip_special_tokens decoding saves as a run-on. This is a decoding-configuration artefact: it inflates response length but is produced identically by both pair members, so it cancels in every matched-pair delta, and the full-response classifier still scores any harmful content a continuation might contain. Stopping at the chat turn-end token is a straightforward fix for future runs."),
  BUL("Capability anchor: interpretation labels are MMLU-anchored; a formal composite-capability rule across MMLU and ARC is left to future work."),
];

const ch9 = [
  H1("Chapter 9  Future Work"),
  NUM("An adversarial-attack arm: re-run the HarmBench axis under at least one optimisation-based or LLM-mediated attack (for example GCG or PAIR) on a subset of pairs, upgrading the direct-request compliance result to a quantization-versus-safety-under-attack result. This is the highest-leverage extension of the safety axis.", "fw"),
  NUM("A second human annotator (inter-rater κ and adjudication of disagreements) extending the completed single-annotator human validation (classifier κ 0.59 vs regex 0.11 against human labels), and an open-weight guard model (e.g. Llama Guard [17]) for a fully reproducible cross-check.", "fw"),
  NUM("Adding genuinely different quantization families (GPTQ, AWQ, and especially GGUF, the deployment-dominant format) beyond the two bitsandbytes methods.", "fw"),
  NUM("Extending the multi-seed sensitivity arm to the remaining two pairs (Mistral, Phi) for a full-matrix stochastic estimate.", "fw"),
  NUM("Extending the refusal-margin probe across all three precisions, and an independent activation-space refusal-direction probe with a paired neutral-margin control and a significance test.", "fw"),
];

const ch10 = [
  H1("Chapter 10  Conclusion"),
  PJ("This thesis set out to determine whether the safety changes attributed to quantization in compact LLMs are genuine alignment shifts or artefacts of capability loss, invalid scoring, or truncated generation. By combining a matched-pair design, a judge-validated primary scorer with an independent cross-check, a capability anchor, a multi-precision sweep, and the benchmark’s own standardized generation budget, it reaches a calibrated answer. At the 512-token reference budget, four-bit NF4 never significantly increases harmful compliance under direct requests in any of five pairs across four families; not one ASR contrast survives multiplicity correction; the only individually significant safety move is a decrease; and the effects that do survive correction are capability losses and one benign-direction over-refusal change. Capability degradation is a clean cliff at four-bit, is budget-invariant, and is the robust cost of NF4 in this regime."),
  PJ("Its most transferable lessons are methodological, and there are two. A refusal-counting scorer over-states harmful compliance and, uncorrected, can change a study’s conclusion; safety evaluation should validate its scorer against the benchmark’s own classifier and a second judge, and report chance-corrected agreement. And the generation budget is a first-class experimental parameter: a truncated budget manufactured this study’s original headline result, and only re-running at the benchmark’s standardized 512-token budget revealed it as an artefact. Both lessons argue for the same discipline: measure the instrument before trusting the measurement, and resist over-claiming on small, borderline effects. The open, reproducible framework that produced these results, together with the machine claim lock that ties every reported number to the committed artefacts, is the project’s durable contribution, designed for others to extend to their own models, benchmarks, and quantization methods."),
];

const refs = [
  H1("References"),
  REF(1, "T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2208.07339."),
  REF(2, "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023. arXiv:2305.14314."),
  REF(3, "M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.04249."),
  REF(4, "A. Kharinaev et al., “Investigating the impact of quantization methods on the safety and reliability of large language models,” arXiv preprint arXiv:2502.15799, 2025."),
  REF(5, "Y. Belkhiter et al., “HarmLevelBench: Evaluating harm-level compliance and the impact of quantization on model alignment,” arXiv preprint arXiv:2411.06835, 2024."),
  REF(6, "R. Jin et al., “A comprehensive evaluation of quantization strategies for large language models,” in Findings of the Association for Computational Linguistics (ACL Findings), 2024. arXiv:2402.16775."),
  REF(7, "K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2405.18137."),
  REF(8, "P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test suite for identifying exaggerated safety behaviours in large language models,” in Proc. NAACL, 2024. arXiv:2308.01263."),
  REF(9, "P. Liang et al., “Holistic evaluation of language models,” Transactions on Machine Learning Research (TMLR), 2023. arXiv:2211.09110."),
  REF(10, "J. Gu et al., “A survey on LLM-as-a-judge,” arXiv preprint arXiv:2411.15594, 2024."),
  REF(11, "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” arXiv preprint arXiv:2503.05061, 2025."),
  REF(12, "D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proc. International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300."),
  REF(13, "P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? Try ARC, the AI2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018."),
  REF(14, "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993."),
  REF(15, "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947."),
  REF(16, "I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence of large language models?,” arXiv preprint arXiv:2405.00632, 2024."),
  REF(17, "H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversations,” Meta AI, 2023. arXiv:2312.06674."),
  REF(18, "J. Pineau et al., “Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program),” Journal of Machine Learning Research (JMLR), vol. 22, 2021. arXiv:2003.12206."),
  REF(19, "G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, “Ten simple rules for reproducible computational research,” PLOS Computational Biology, vol. 9, no. 10, e1003285, 2013."),
  REF(20, "G. Wilson, J. Bryan, K. Cranston, J. Kitzes, L. Nederbragt, and T. K. Teal, “Good enough practices in scientific computing,” PLOS Computational Biology, vol. 13, no. 6, e1005510, 2017."),
  REF(21, "A. M. Smith et al., “Journal of Open Source Software (JOSS): Design and first-year review,” arXiv preprint arXiv:1707.02264, 2017."),
];

const appendix = [
  H1("Appendix A  Reproducibility Statement"),
  PJ("All code, configuration, and redacted result artefacts are in the project repository. The analysis reproduces byte-for-byte from the committed sidecars without a GPU; full experiments require the gated model weights and a CUDA GPU. The project maps to the ML Reproducibility Checklist [18]: models and algorithms are described (Chapters 3–4); datasets, splits, and sample counts are recorded in each run summary; code and declared dependencies are released; hyper-parameters and decoding controls are recorded in every summary; the compute infrastructure is recorded; and significance procedures are specified, with multiple comparisons disclosed. The dependency set is declared with minimum-version floors (a lockfile with exact pins is a future hardening step). The repository follows recognised research-software practice (scripts rather than manual steps, fixed seeds, explicit dependencies, and a documented project structure [19], [20]) and is packaged for reuse in line with open-source-software peer-review norms [21]. Raw generations are gitignored and hash-pinned; only redacted sidecars (identifiers, booleans, scalars) are released, so the study is reproducible without redistributing harmful text. The primary artefacts live under results_512/ (the 512-token study), with the 128-token run retained under results/ as the generation-length comparison; a checksum manifest pins 300 immutable raw files across both trees and the multi-seed arm."),
  H1("Appendix B  Artefacts and Test Suite"),
  PJ("The framework ships 339 automated tests across the schema, loaders (including the fail-loud quantization guard), scorers, analysis mathematics (delta computation, paired bootstrap, McNemar, Cohen’s κ), and the SLURM job generator; a machine claim lock (scripts/verify_report_claims.py, run inside the suite) additionally asserts every load-bearing number in the report and this thesis against the committed analysis artefacts under results_512/analysis/, so the documents cannot drift from the evidence without failing the build. Result artefacts comprise per-prompt redacted score sidecars, per-model judge summaries, and analysis files (pairwise deltas, judge agreement, the precision sweep). A checksum manifest pins the immutable raw artefacts."),
];

// ===========================================================================
// ASSEMBLE
// ===========================================================================
const doc = new Document({
  features: { updateFields: true },  // populate the TOC field on open
  styles: {
    default: { document: { run: { font: SERIF, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: SERIF }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: SERIF }, paragraph: { spacing: { before: 200, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: SERIF }, paragraph: { spacing: { before: 160, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ...["rq", "fw", "ref"].map(ref => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 360 } } } }] })),
  ] },
  sections: [{
    properties: { page: {
      size: { width: 11906, height: 16838 },
      margin: { top: 1273, right: 1273, bottom: 1273, left: 1273 },
    } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB", space: 4 } },
      children: [new TextRun({ text: "Safety–Capability Trade-offs in Quantized Compact LLMs", font: SERIF, size: 16, color: "666666" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Page ", font: SERIF, size: 18 }),
      new TextRun({ children: [PageNumber.CURRENT], font: SERIF, size: 18 }),
      new TextRun({ text: " of ", font: SERIF, size: 18 }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], font: SERIF, size: 18 }),
    ] })] }) },
    children: [
      ...cover, ...declaration, ...abstract, ...acknowledgements, ...aiDeclaration, ...toc,
      ...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...ch9, ...ch10,
      ...refs, ...appendix,
    ],
  }],
});

const out = "/Users/tanueihorng/fyp_quant/docs/FYP_Thesis_2026-07-02_v4.docx";
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(out, buf); console.log("WROTE:", out, "(" + buf.length + " bytes)"); });
