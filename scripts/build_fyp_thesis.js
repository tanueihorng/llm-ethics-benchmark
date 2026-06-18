// ============================================================================
// FYP THESIS builder — a NEW, standalone, research-grade thesis (docx-js).
// Separate from scripts/build_fyp_report.js; `make report` never touches this.
// Output: docs/FYP_Thesis_2026-06-18.docx   (build: node scripts/build_fyp_thesis.js)
// ============================================================================
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const SERIF = "Times New Roman";
const MONO = "Consolas";
const BODY = 24;           // 12pt
const CONTENT_W = 9360;    // US Letter, 1" margins

// ---- helpers ---------------------------------------------------------------
let listN = 0;
const T = (text, opts = {}) => new TextRun({ text, font: SERIF, size: BODY, ...opts });
const P = (text, opts = {}) => new Paragraph({
  children: [T(text)], spacing: { after: opts.after ?? 140, line: 276 },
  alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT, ...opts.p,
});
const PJ = (text) => new Paragraph({
  children: [T(text)], alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 276 },
});
const RUNS = (runs, opts = {}) => new Paragraph({
  children: runs, alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 276 }, ...opts,
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
  spacing: { after: 80, line: 276 }, children: [T(text)] }); };
const NUM = (text, ref) => new Paragraph({ numbering: { reference: ref, level: 0 },
  spacing: { after: 80, line: 276 }, children: [T(text)] });
// IEEE reference entry: "[n] ..." with a hanging indent so continuation lines align.
const REF = (n, text) => new Paragraph({ spacing: { after: 90, line: 264 },
  indent: { left: 400, hanging: 400 },
  children: [new TextRun({ text: `[${n}] ${text}`, font: SERIF, size: 22 })] });
const CAP = (text) => new Paragraph({ spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text, font: SERIF, size: 20, italics: true })] });

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
  CTR("FINAL YEAR PROJECT — THESIS", { size: 24, bold: true, after: 480 }),
  CTR("Benchmarking the Ethical Performance of Open-Source LLMs:", { size: 32, bold: true, after: 100 }),
  CTR("A Matched-Pair, Judge-Validated Study of Safety–Capability Trade-offs in Quantized Compact Language Models (fp16 → INT8 → NF4)", { size: 28, bold: true, after: 760 }),
  CTR("Project Code:  CCDS25-1136", { size: 24, bold: true, after: 200 }),
  CTR("Student:  TAN UEI HORNG  (UTAN001)", { size: 26, bold: true, after: 120 }),
  CTR("Email:  UTAN001@e.ntu.edu.sg", { size: 22, after: 120 }),
  CTR("Supervisor:  Dr. Zhang Jiehuang  (jiehuang.zhang@ntu.edu.sg)", { size: 22, after: 560 }),
  CTR("18 June 2026", { size: 26, bold: true, after: 80 }),
  CTR("Five matched pairs / ten models / four families · three precisions · four benchmarks · 295 automated tests", { size: 18, italics: true, color: "555555" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const declaration = [
  H1NB("Declaration of Originality"),
  PJ("I hereby declare that this Final Year Project thesis is my own work and, to the best of my knowledge and belief, it contains no material previously published or written by another person, nor material that has been accepted for the award of any other degree or diploma of a university or other institution of higher learning, except where due acknowledgement has been made in the text. The intellectual content of this thesis — its research design, experimental methodology, analysis, and interpretation — is the product of my own work, although I have received assistance on software implementation, language, and presentation as acknowledged herein."),
  PJ("All experimental results reported in this thesis were produced by the open-source benchmarking framework described herein, executed on the NTU TC1 GPU cluster, and are reproducible from the committed configuration, source code, and redacted result artefacts. Every reported numerical result is computed by the committed code from the recorded experimental records; no result has been altered, fabricated, or selectively reported. Where the work of others has been used, it has been cited and referenced."),
  new Paragraph({ spacing: { before: 700, after: 40 }, children: [T("_______________________________")] }),
  P("Tan Uei Horng  (UTAN001)", { after: 30 }),
  P("College of Computing and Data Science, Nanyang Technological University", { after: 30 }),
  P("Date:  18 June 2026", { after: 30 }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstract = [
  H1NB("Abstract"),
  PJ("Compact instruction-tuned language models in the one-to-seven-billion-parameter range are increasingly deployed on edge and consumer hardware, where quantization — most commonly four-bit — has become the de facto method for fitting them into available memory. Quantization is not behaviourally neutral: it can alter safety alignment, refusal calibration, and general capability. Yet reports disagree, and a deeper problem confounds them: a quantized model can differ from its baseline for reasons other than quantization (different checkpoints, decoding, or scoring), and a brittle refusal-matching scorer can over-count “attack success.” This thesis asks whether the safety changes attributed to quantization are genuine alignment shifts or artefacts of capability loss and invalid scoring."),
  PJ("It contributes an open, reproducible benchmarking framework built around a matched-pair design — baseline and quantized members are loaded from identical weights, with quantization applied on the fly at load time — so that quantization is the sole experimental variable. Five model pairs across four families (Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, and Phi-4-mini-instruct), spanning roughly 1.7 to 7.2 billion parameters, are evaluated across three precisions (fp16, INT8/LLM.int8, and NF4 four-bit) on four benchmarks: HarmBench (harmful compliance, Attack Success Rate), XSTest (over-refusal), and the MMLU and ARC-Challenge accuracy benchmarks (capability). Harmful compliance is scored by the official HarmBench classifier as the primary judge and cross-checked by a second, architecturally independent judge (GPT-4o); a refusal regex is retained only as a demoted, transparent proxy. A capability-anchored interpretation layer separates genuine alignment shifts from capability degradation, and all deltas carry paired-bootstrap confidence intervals and McNemar exact tests."),
  PJ("Three findings result. First, methodologically, the refusal regex systematically over-counts harmful compliance, and validating against the benchmark’s own classifier relocates the study’s single statistically significant effect from one model to another — a cautionary result for safety evaluation. Second, empirically, under four-bit NF4 the only significant increase in harmful compliance is in the smallest model (Qwen3-1.7B, ΔASR = +0.055), and it is modest, borderline, and judge-dependent; quantization never significantly reduces harmful compliance, and capability loss, where significant, accompanies it. Third, extending to a three-precision sweep shows the effect is not a smooth function of bit-width: capability loss is a clean cliff at four-bit (no INT8 capability delta is significant for any pair), whereas the safety effect is sparse and method-specific — a second, both-judge-significant increase appears in Llama-3B specifically at INT8 and vanishes at NF4. The framework reliably separates the two failure modes, and the overall picture is a rigorous null with small, model- and method-specific effects rather than a uniform “quantization breaks safety” narrative."),
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
  H1NB("List of Tables"),
  P("Table 3.1  Model pairs, families, and parameter scales.", { after: 60 }),
  P("Table 3.2  Benchmarks, primary metrics, and sample budgets.", { after: 60 }),
  P("Table 3.3  Interpretation labels derived from combined deltas.", { after: 60 }),
  P("Table 5.1  Experimental configuration summary.", { after: 60 }),
  P("Table 6.1  Judge-versus-regex agreement (Cohen’s κ) by model family.", { after: 60 }),
  P("Table 6.2  Main study: per-pair deltas and interpretation labels (fp16 vs NF4).", { after: 60 }),
  P("Table 6.3  Precision sweep: HarmBench ASR at fp16 / INT8 / NF4 (judge).", { after: 60 }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===========================================================================
// CHAPTERS
// ===========================================================================
const ch1 = [
  H1("Chapter 1  Introduction"),
  H2("1.1  Motivation"),
  PJ("Large language models (LLMs) are increasingly deployed not in data centres but on laptops, phones, and embedded devices, where memory is scarce. The dominant enabler of this shift is quantization: representing model weights in eight, four, or fewer bits instead of sixteen, often halving or quartering the memory footprint at a small, well-studied cost to task accuracy [1], [2]. For compact models in the one-to-seven-billion-parameter range — the models most likely to run locally — four-bit quantization has become routine."),
  PJ("Accuracy, however, is not the only property that matters. Instruction-tuned models are also aligned: trained to refuse harmful requests and to answer benign ones. Whether quantization preserves this alignment is far less understood than whether it preserves accuracy, and the stakes are higher, because a locally deployed model runs outside any server-side safety filter. If compressing a model to fit a phone quietly makes it more willing to produce harmful content, that is a deployment-relevant safety regression that current accuracy-centric evaluation would miss."),
  H2("1.2  Problem statement"),
  PJ("Two difficulties make this question hard to answer credibly. The first is confounding: a quantized model can differ from a full-precision one for reasons that have nothing to do with quantization — a different published checkpoint, different decoding settings, or simply the noise of generation. The second is measurement: harmful compliance is usually scored by pattern-matching for refusals, equating “did not refuse” with “attack succeeded.” Many non-refusals are not actually harmful (vague deflections, safety lectures, on-topic but benign answers), so a refusal-counting scorer can over-state harmful compliance and, worse, do so unevenly across models. A study that does not control both confounding and scoring validity cannot distinguish a genuine alignment shift from a capability artefact or a scoring artefact."),
  H2("1.3  Research questions"),
  PJ("This thesis is organised around five research questions:"),
  NUM("RQ1: Does quantization increase harmful compliance in compact instruction-tuned LLMs?", "rq"),
  NUM("RQ2: Does quantization increase over-refusal on benign prompts?", "rq"),
  NUM("RQ3: Does quantization degrade general capability?", "rq"),
  NUM("RQ4: Within a family, are smaller models more sensitive than larger ones?", "rq"),
  NUM("RQ5: Are the effects consistent across model families and across quantization precisions?", "rq"),
  H2("1.4  Contributions"),
  PJ("This thesis makes three contributions. First, a methodological one: a capability-anchored, judge-validated procedure for distinguishing genuine alignment shifts from capability degradation under quantization — and the finding that a refusal-counting scorer systematically over-counts attack success, relocating the study’s one significant effect once the benchmark’s own classifier is used. Second, an empirical one: across five matched pairs, four families, and three precisions, capability loss is a clean cliff at four-bit while the safety effect is sparse, model- and method-specific rather than a smooth function of bit-width. Third, an engineering one: an open, reproducible, extensible framework (matched-pair loading, a benchmark-plugin contract, a judge-validation layer, and cluster orchestration) that others can reuse for their own quantization-safety studies."),
  H2("1.5  Thesis structure"),
  PJ("Chapter 2 surveys related work and isolates the gap. Chapter 3 details the methodology: the matched-pair design, quantization, benchmarks, scoring, the interpretation framework, and the statistical procedure. Chapter 4 documents the system design and implementation. Chapter 5 describes the experimental setup. Chapter 6 presents the results. Chapter 7 discusses them and the threats to validity, Chapter 8 records limitations, Chapter 9 proposes future work, and Chapter 10 concludes. A reproducibility statement and appendices follow."),
];

const ch2 = [
  H1("Chapter 2  Background and Related Work"),
  H2("2.1  Quantization and its behavioural effects"),
  PJ("Post-training quantization compresses model weights to lower-precision formats. The bitsandbytes library provides two widely used schemes evaluated here: NF4, a four-bit “normal-float” format with double quantization [2], and LLM.int8(), an eight-bit mixed-precision scheme that keeps outlier dimensions in higher precision [1]. While the accuracy cost of these methods is well characterised, their effect on safety behaviour is contested. Empirical studies report mixed and method-dependent results: some find that quantization degrades safety or fairness [3], [6], others find no consistent trend across bit-widths [4], and some show that quantization can be actively exploited to surface unsafe behaviour [5]. This very inconsistency motivates a controlled, matched-pair design rather than cross-paper comparison, and it cautions against strong, uniform claims."),
  H2("2.2  Safety benchmarks and red-teaming"),
  PJ("HarmBench [7] provides a standardised red-teaming benchmark with an explicit behaviour taxonomy and, crucially, a fine-tuned classifier that scores whether a response actually exhibits the harmful behaviour — a deliberate move away from refusal-string matching. XSTest [8] measures the opposite failure mode, over-refusal on benign prompts. Holistic frameworks such as HELM [9] formalise the idea that evaluation should be multi-metric and transparent. This thesis adopts HarmBench’s classifier-as-scorer principle [7] as its primary instrument and treats refusal-matching only as a foil."),
  H2("2.3  LLM-as-judge evaluation and its validity"),
  PJ("Using one model to judge another’s output is now common, but its validity is an active concern. Surveys of LLM-as-judge methods [10] and critical studies [11] recommend reporting chance-corrected agreement (Cohen’s κ) rather than raw accuracy, using more than one judge, and acknowledging that without human grounding a judge can be systematically biased [11]. This thesis follows that guidance — a primary classifier, a second independent judge, κ reporting, and bootstrap confidence intervals — and is explicit about the one safeguard it does not have (human-label validation)."),
  H2("2.4  The research gap"),
  PJ("Three observations define the gap this thesis addresses. First, the quantization–safety literature reports inconsistent, method- and model-dependent effects [3]–[6], so cross-paper comparison cannot settle whether quantization shifts safety; a controlled, matched-pair design that isolates quantization is required. Second, harmful compliance is frequently scored by refusal-matching rather than by a validated classifier, so a reported “attack success” may be a scoring artefact, and where LLM judges are used their inter-judge agreement is often left unreported [10], [11] — yet the benchmark’s own fine-tuned classifier exists precisely to avoid this [7]. Third, studies that do measure quantization rarely separate a safety change from a capability change, and almost none trace the effect across more than one precision or quantization method."),
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
  PJ("Four benchmarks span the two axes of interest. Harmful compliance (HarmBench [7]) and over-refusal (XSTest [8]) capture safety; MMLU [12] and ARC-Challenge [13] capture capability, using two independent benchmarks so that capability claims do not rest on a single test."),
  tbl(["Benchmark", "Axis", "Primary metric", "n"], [
    ["HarmBench", "Harmful compliance", "Attack Success Rate", "200"],
    ["XSTest", "Over-refusal", "Over-refusal rate", "250"],
    ["MMLU (subset)", "Capability", "Accuracy", "300"],
    ["ARC-Challenge", "Capability", "Accuracy", "1172"],
  ], [2400, 2560, 2600, 1800]),
  CAP("Table 3.2  Benchmarks, primary metrics, and sample budgets."),
  H2("3.4  Scoring: classifier-primary, regex-as-foil"),
  PJ("HarmBench Attack Success Rate is scored by the official HarmBench classifier (cais/HarmBench-Llama-2-13b-cls) [7], run in full precision so that the reference instrument is not itself confounded by quantization. A second, architecturally independent judge (GPT-4o) re-scores the same generations as a robustness check [10], [11]. A deterministic refusal regex is retained only as a secondary, transparent proxy. This choice is itself a research object: the thesis reports the agreement between scorers and shows that the regex over-counts, which is why it is demoted. MMLU and ARC use deterministic exact match; XSTest over-refusal uses the regex (the HarmBench classifier judges harmful compliance only)."),
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
  PJ("All inference uses greedy decoding at temperature 0.0 with a fixed seed, so the only variance reflected in the intervals is prompt sampling. Deltas carry paired-bootstrap 95% confidence intervals (2,000 resamples, seed 42), resampled by prompt identity so that both members of a pair see the same prompts in the same order. The single load-bearing effect is additionally tested with McNemar’s exact paired test, and a multi-seed sensitivity arm (temperature 0.7) estimates generation-level variance for that pair. Significance flags are nominal and uncorrected for multiple comparisons; this is stated wherever significance is claimed."),
];

const ch4 = [
  H1("Chapter 4  System Design and Implementation"),
  H2("4.1  Architecture"),
  PJ("The framework is a Python package organised around plugins and configuration. A YAML configuration, validated by a Pydantic schema, declares the models, decoding settings, benchmarks, and cluster parameters. A model loader resolves dtype and device and applies quantization on the fly; a shared text generator batches prompts through the model using the tokenizer’s chat template; benchmark plugins implement a common contract for loading items, building prompts, scoring responses, and aggregating metrics; and an analysis layer computes pairwise deltas, confidence intervals, and interpretation labels."),
  H2("4.2  Extensibility and the plugin contract"),
  PJ("Adding a model requires only a configuration entry; adding a benchmark requires implementing four methods (load, build prompt, score, aggregate) and registering one line; adding a quantization method extends the loader’s method branch. This keeps the framework reusable beyond the present study — a design property documented in the project’s quickstart and verified by the test suite."),
  H2("4.3  Reliability safeguards"),
  PJ("Several safeguards protect result integrity. The loader fails loud if quantization is requested but does not actually engage, refusing to emit full-precision results mislabelled as quantized. Raw generations and their summaries are treated as immutable artefacts: all post-hoc scoring, including judge validation, writes derived sidecars only, and integrity is pinned by a checksum manifest. Resume logic keys on prompt identity so an interrupted run continues without double-counting. A suite of 295 automated tests covers the schema, loaders, scorers, analysis mathematics, and the cluster job generator."),
  H2("4.4  Privacy discipline"),
  PJ("Because the study handles harmful prompts and responses, committed artefacts are redacted: prompt identifiers, boolean labels, scalar scores, and run metadata only — never prompt or response text. This makes the study reproducible from the committed sidecars without redistributing harmful content."),
];

const ch5 = [
  H1("Chapter 5  Experimental Setup"),
  PJ("All experiments ran on the NTU TC1 GPU cluster (NVIDIA Tesla V100, 32 GB) under a strict offline, batch-submission policy: datasets and weights are pre-cached on the head node, and all compute is dispatched via SLURM. The same software stack, seed, and decoding settings apply to every run, so cross-condition comparisons are controlled. Table 5.1 summarises the configuration."),
  tbl(["Parameter", "Value"], [
    ["Pairs / models / families", "5 / 10 / 4"],
    ["Precisions", "fp16, INT8 (LLM.int8), NF4 (4-bit)"],
    ["Benchmarks", "HarmBench, XSTest, MMLU, ARC-Challenge"],
    ["Decoding", "greedy, temperature 0.0, seed 42, max_new_tokens 128"],
    ["Primary HarmBench scorer", "official HarmBench classifier (fp16)"],
    ["Second judge", "GPT-4o (architecturally independent)"],
    ["Statistics", "paired bootstrap 95% CI (2000 resamples) + McNemar"],
    ["Hardware", "NVIDIA Tesla V100-32GB (TC1)"],
    ["Verification", "295 automated tests"],
  ], [3600, 5760]),
  CAP("Table 5.1  Experimental configuration summary."),
  PJ("The five NF4 pairs and the judge validation ran first; the INT8 precision point and its judge ran subsequently on the same cluster with identical methodology. The GPT-4o second judge runs locally against an API. The study’s entire analysis is reproducible from the committed redacted sidecars without a GPU."),
];

const ch6 = [
  H1("Chapter 6  Results and Analysis"),
  H2("6.1  Scoring validity: the regex over-counts"),
  PJ("The first result concerns the instrument. The refusal regex reports far higher Attack Success Rates than the HarmBench classifier, and the gap is uneven by family. Cohen’s κ between the regex and the classifier is poor for the Qwen and Mistral models and good for Llama and Phi (Table 6.1), because models that produce a large volume of ambiguous, non-refusing-but-benign text are exactly those the regex mis-scores. Adopting the classifier relocates the study’s single significant effect from one model (under the regex) to another (under the classifier). This is the thesis’s central methodological result: in safety evaluation, the choice of scorer is not a detail but can change the conclusion."),
  tbl(["Family", "Cohen’s κ (regex vs classifier)", "Reading"], [
    ["Qwen", "0.19 – 0.37", "poor — regex over-counts heavily"],
    ["Mistral", "0.11", "worst — v2 0.885 vs judge 0.375"],
    ["Llama", "0.68 – 0.79", "good — little ambiguous middle ground"],
    ["Phi", "0.59 – 0.67", "moderate"],
  ], [1900, 3660, 3800]),
  CAP("Table 6.1  Judge-versus-regex agreement (Cohen’s κ) by model family."),
  H2("6.2  Main study (fp16 vs NF4)"),
  PJ("Under the primary classifier, NF4 quantization never significantly reduces harmful compliance, and the only significant increase is in the smallest model, Qwen3-1.7B (ΔASR = +0.055, CI [+0.010, +0.100]). That pair also loses significant capability, so it is labelled broad_degradation. The remaining pairs show no significant safety change. Table 6.2 reports the per-pair deltas and labels."),
  tbl(["Pair", "ΔASR (95% CI)", "Sig?", "ΔMMLU", "ΔARC", "Label"], [
    ["qwen_2b", "+0.055 [+0.010,+0.100]", "yes", "−0.087", "−0.013", "broad_degradation"],
    ["qwen_4b", "+0.025 [−0.000,+0.055]", "no", "−0.003", "−0.021", "alignment_degradation (dir.)"],
    ["llama_3_2_3b", "0.000 [−0.020,+0.020]", "no", "−0.043", "−0.028", "broad_degradation"],
    ["mistral_7b", "−0.040 [−0.110,+0.025]", "no", "−0.017", "+0.009", "alignment_improvement (dir.)"],
    ["phi4_mini", "0.000 [−0.030,+0.030]", "no", "−0.023", "−0.015", "robust_preservation"],
  ], [1500, 2700, 700, 1320, 1240, 1900]),
  CAP("Table 6.2  Main study: per-pair deltas and interpretation labels (fp16 vs NF4). Bold-zero CIs exclude zero."),
  PJ("The Qwen3-1.7B effect, though significant, is modest and fragile: it is corroborated by McNemar’s exact test (p = 0.027) but is not significant under the GPT-4o second judge, and a multi-seed arm attenuates it to roughly half the greedy estimate (mean +0.024). It should be read as the upper end of a decode-dependent range, not a fixed effect (RQ1, RQ4)."),
  H2("6.3  Capability anchoring (RQ3)"),
  PJ("Every pair loses capability under NF4 in direction; the loss is significant on at least one benchmark for Qwen3-1.7B (MMLU), Qwen3-4B (ARC), and Llama-3B (both). The two capability benchmarks diverge informatively: Qwen3-1.7B’s large MMLU drop does not replicate on ARC, so the dramatic within-Qwen scale gap is partly MMLU-specific. The capability anchor is what allows the safety deltas to be read correctly — the smallest Qwen pair degrades on both axes, while no pair shows a safety “improvement” that is really capability collapse."),
  H2("6.4  Mechanism: refusal-margin geometry"),
  PJ("To probe why the smallest model moves, a derived analysis measured each model’s first-token refusal margin in fp16 and under NF4, following the observation that quantization most affects low-confidence, near-boundary decisions [14]. A thin baseline margin does predict which prompts flip, but only weakly within a family, and the flips are symmetric — quantization destabilises the decision boundary in both directions rather than eroding refusals one-directionally. For the one moving pair, the change is more consistent with generic capability-driven boundary instability than a targeted alignment shift, reinforcing the capability-anchored reading."),
  H2("6.5  Precision point: fp16 → INT8 → NF4 (RQ5)"),
  PJ("Adding INT8 shows the effect is not a smooth function of bit-width. On capability the result is a clean cliff at four-bit: no INT8 MMLU or ARC delta is significant for any pair, while the significant capability losses are all at NF4. On safety the picture is two-peaked and method-specific: the study’s two significant ASR moves sit at different precisions. Qwen3-1.7B’s increase is a four-bit (NF4) effect, while a second increase appears in Llama-3B specifically at INT8 (ΔASR = +0.040) that is significant under both judges and McNemar (p = 0.022 classifier, 0.008 GPT-4o) yet non-monotonic — it reverts to baseline at NF4. Table 6.3 shows the sweep."),
  tbl(["Pair", "fp16", "INT8", "NF4", "Shape"], [
    ["qwen_2b", "0.135", "0.150", "0.190", "rising; only NF4 significant"],
    ["qwen_4b", "0.065", "0.065", "0.090", "INT8 = fp16"],
    ["llama_3_2_3b", "0.040", "0.080", "0.040", "INT8 spike, reverts at NF4"],
    ["mistral_7b", "0.385", "0.375", "0.345", "falling"],
    ["phi4_mini", "0.055", "0.060", "0.055", "flat"],
  ], [1700, 1100, 1100, 1100, 4360]),
  CAP("Table 6.3  Precision sweep: HarmBench ASR at fp16 / INT8 / NF4 (primary judge)."),
  PJ("The Llama INT8 effect is real but bounded: it rests on roughly eight to nine prompts, concentrated in the illegal and cybercrime categories, on a single pair, and it does not persist to the more aggressive four-bit method. It is therefore reported as a method-specific numerical effect of the LLM.int8 algorithm on this model, not a general law — the honest, calibrated reading."),
  H2("6.6  Statistical caveats"),
  PJ("Three caveats bound the conclusions. With 200 HarmBench prompts the confidence intervals are wide (about ±0.05), so small deltas are not distinguishable from zero. Significance is nominal and uncorrected for multiple comparisons — across the sweep roughly thirty Attack-Success-Rate comparisons are made, so the single significant NF4 delta (Qwen3-1.7B, p = 0.027) is a nominal result that would not survive a strict family-wise correction; it is carried as the headline only because it is independently corroborated by McNemar, a multi-seed arm, and (in direction) the second judge. The Llama INT8 effect, significant under two judges and two tests, is the more multiplicity-robust of the two. The reader is asked to weigh this converging evidence rather than any single per-comparison threshold."),
];

const ch7 = [
  H1("Chapter 7  Discussion and Threats to Validity"),
  PJ("The results answer the research questions with calibrated, not sweeping, claims. NF4 never makes these models safer on harmful compliance, and only the smallest model becomes significantly less safe; over-refusal does not rise (the only significant change is a decrease); capability loss is real but modest and partly benchmark-specific; and the cross-precision picture is that effects are method- and model-specific, not graded with bit-width. For deployment, the practical message is that eight-bit quantization is essentially free on these axes, whereas four-bit carries a small, model-specific risk that should be measured per model rather than assumed."),
  H2("7.1  Internal validity"),
  PJ("Internal validity is the design’s strongest property: the matched-pair structure with on-the-fly quantization from identical weights, plus deterministic decoding and deterministic or judge-validated scoring, leaves quantization as the only plausible cause of a measured delta. The fail-loud loader guard rules out the subtle confound of a quantized run silently executing in full precision."),
  H2("7.2  Construct validity"),
  PJ("The primary safety construct is scored by the benchmark’s own fine-tuned classifier rather than refusal matching, and the cross-judge agreement (κ 0.69–0.94) shows the finding is not an artefact of one classifier. The residual construct threat is that both judges are themselves LLMs without independent human-label validation; this is disclosed and flagged for future work. Over-refusal remains regex-scored, a known weaker construct."),
  H2("7.3  External and statistical-conclusion validity"),
  PJ("External validity is bounded by five pairs, one decoding regime, and two bitsandbytes methods; the cross-family and cross-precision sweeps widen it but do not make it general. Statistical-conclusion validity is bounded by sample size and uncorrected multiplicity, addressed by reporting intervals, McNemar corroboration, a multi-seed arm, and an explicit two-layer evidence status rather than bare significance."),
];

const ch8 = [
  H1("Chapter 8  Limitations"),
  BUL("Sample size: 200 HarmBench prompts give wide (±~0.05) confidence intervals; small deltas are not distinguishable from zero."),
  BUL("Multiple comparisons: significance flags are nominal and uncorrected; the one significant NF4 delta would not survive strict family-wise correction (it is corroborated by other tests)."),
  BUL("Decoding: results condition on a single greedy decode; the multi-seed arm covers only the load-bearing pair."),
  BUL("Judge grounding: the HarmBench classifier is treated as reference without independent human-label validation; the second judge is a partial cross-check."),
  BUL("Quantization coverage: two bitsandbytes methods (NF4, INT8); GPTQ, AWQ, and GGUF are out of scope."),
  BUL("The INT8 Llama effect rests on ~8–9 prompts on one pair and is non-monotonic; it is reported as suggestive and method-specific, not general."),
  BUL("Capability anchor: interpretation labels are MMLU-anchored; a formal composite-capability rule across MMLU and ARC is left to future work."),
];

const ch9 = [
  H1("Chapter 9  Future Work"),
  NUM("Human-grounded judge validation, to remove the residual construct threat on the primary scorer.", "fw"),
  NUM("Replication of the INT8 Llama effect across more models and decode seeds, to establish whether it is a general LLM.int8 phenomenon or model-specific numerics.", "fw"),
  NUM("Extending the refusal-margin probe across all three precisions, not only the behavioural metrics.", "fw"),
  NUM("Adding genuinely different quantization families (GPTQ, AWQ, GGUF) beyond the two bitsandbytes methods.", "fw"),
  NUM("An independent activation-space refusal-direction probe and a paired neutral-margin control with a significance test.", "fw"),
];

const ch10 = [
  H1("Chapter 10  Conclusion"),
  PJ("This thesis set out to determine whether the safety changes attributed to quantization in compact LLMs are genuine alignment shifts or artefacts of capability loss and invalid scoring. By combining a matched-pair design, a judge-validated primary scorer with an independent cross-check, a capability anchor, and a multi-precision sweep, it reaches a calibrated answer. The dominant result is a rigorous null with small, model- and method-specific effects: NF4 never makes these models safer; only the smallest model becomes significantly more harmful, and modestly so; capability loss is a clean cliff at four-bit; and the safety effect is two-peaked and method-specific rather than graded with bit-width."),
  PJ("Its most transferable lesson is methodological: a refusal-counting scorer over-states harmful compliance and, uncorrected, can place the study’s significant finding on the wrong model — so safety evaluation should validate its scorer against the benchmark’s own classifier and a second judge, report chance-corrected agreement, and resist over-claiming on small, borderline effects. The open, reproducible framework that produced these results is the project’s durable contribution, designed for others to extend to their own models, benchmarks, and quantization methods."),
];

const refs = [
  H1("References"),
  REF(1, "T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication for transformers at scale,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2208.07339."),
  REF(2, "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023. arXiv:2305.14314."),
  REF(3, "A. Kharinaev et al., “Investigating the impact of quantization methods on the safety and reliability of large language models,” arXiv preprint arXiv:2502.15799, 2025."),
  REF(4, "R. Jin et al., “A comprehensive evaluation of quantization strategies for large language models,” in Findings of the Association for Computational Linguistics (ACL Findings), 2024. arXiv:2402.16775."),
  REF(5, "K. Egashira et al., “Exploiting LLM quantization,” arXiv preprint arXiv:2405.18137, 2024."),
  REF(6, "Y. Belkhiter et al., “HarmLevelBench: Evaluating harm-level compliance and the impact of quantization on model alignment,” arXiv preprint arXiv:2411.06835, 2024."),
  REF(7, "M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,” in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.04249."),
  REF(8, "P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test suite for identifying exaggerated safety behaviours in large language models,” in Proc. NAACL, 2024. arXiv:2308.01263."),
  REF(9, "P. Liang et al., “Holistic evaluation of language models,” Transactions on Machine Learning Research (TMLR), 2023. arXiv:2211.09110."),
  REF(10, "J. Gu et al., “A survey on LLM-as-a-judge,” arXiv preprint arXiv:2411.15594, 2024."),
  REF(11, "M. Krumdick et al., “No free labels: Limitations of LLM-as-a-judge without human grounding,” arXiv preprint arXiv:2503.05061, 2025."),
  REF(12, "D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proc. International Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300."),
  REF(13, "P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? Try ARC, the AI2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018."),
  REF(14, "I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence of large language models?,” arXiv preprint arXiv:2405.00632, 2024."),
  REF(15, "J. Pineau et al., “Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program),” Journal of Machine Learning Research (JMLR), vol. 22, 2021. arXiv:2003.12206."),
  REF(16, "G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, “Ten simple rules for reproducible computational research,” PLOS Computational Biology, vol. 9, no. 10, e1003285, 2013."),
  REF(17, "G. Wilson, J. Bryan, K. Cranston, J. Kitzes, L. Nederbragt, and T. K. Teal, “Good enough practices in scientific computing,” PLOS Computational Biology, vol. 13, no. 6, e1005510, 2017."),
  REF(18, "A. M. Smith et al., “Journal of Open Source Software (JOSS): Design and first-year review,” arXiv preprint arXiv:1707.02264, 2017."),
];

const appendix = [
  H1("Appendix A  Reproducibility Statement"),
  PJ("All code, configuration, and redacted result artefacts are in the project repository. The analysis reproduces byte-for-byte from the committed sidecars without a GPU; full experiments require the gated model weights and a CUDA GPU. The project maps to the ML Reproducibility Checklist [15]: models and algorithms are described (Chapters 3–4); datasets, splits, and sample counts are recorded in each run summary; code and pinned dependencies are released; hyper-parameters and decoding controls are recorded in every summary; the compute infrastructure is recorded; and significance procedures are specified, with multiple comparisons disclosed. The repository follows recognised research-software practice — scripts rather than manual steps, fixed seeds, explicit dependencies, and a documented project structure [16], [17] — and is packaged for reuse in line with open-source-software peer-review norms [18]. Raw generations are gitignored and hash-pinned; only redacted sidecars (identifiers, booleans, scalars) are released, so the study is reproducible without redistributing harmful text."),
  H1("Appendix B  Artefacts and Test Suite"),
  PJ("The framework ships 295 automated tests across the schema, loaders (including the fail-loud quantization guard), scorers, analysis mathematics (delta computation, paired bootstrap, McNemar, Cohen’s κ), and the SLURM job generator. Result artefacts comprise per-prompt redacted score sidecars, per-model judge summaries, and analysis files (pairwise deltas, judge agreement, the precision sweep). A checksum manifest pins the immutable raw artefacts."),
];

// ===========================================================================
// ASSEMBLE
// ===========================================================================
const doc = new Document({
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
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
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

const out = "/Users/tanueihorng/fyp_quant/docs/FYP_Thesis_2026-06-18.docx";
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(out, buf); console.log("WROTE:", out, "(" + buf.length + " bytes)"); });
