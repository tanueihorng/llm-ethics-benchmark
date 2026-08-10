# Provenance — the published figures in Table 6.2

Date: 2026-08-02 (Asia/Singapore)
Task: T47 gap 6 — "prior work is never given a number"
Artifact: `docs/rewritten_fyp_report_v1.docx` §6.2, built by `scripts/build_rewritten_fyp_report_v1.py`
Selection rule: fixed in advance, `SELECTION_PREREG.md`
Pinned by: `tests/test_rewritten_report_v1.py::test_t47_gap6_prior_work_figures_are_present`
and `::test_t47_gap6_refuted_figures_stay_out`, both mutation-proven to fire

## Method

Reading was batched, judgement was not (D57). One agent read each of the five primary sources
end to end and returned every importable figure with locating evidence. The pre-registered rule
then selected one figure per study. Each selected figure was checked by **two further independent
agents** under separate lenses — entailment ("does the source report this value with this meaning?")
and comparability ("would placing it beside our number mislead?") — with no sibling figures visible
to either. Three more agents established the publication status of the three non-obvious citations.
Thirteen verification agents in total, none of them the agent that extracted the figure.

Sources were read as primary artifacts, not summaries: local PDFs for [37] and [39], arXiv full
text for [10], [35] and [36].

## Outcome

**No transcription error survived.** All ten verification verdicts returned
`supported_but_needs_rewording`: the values were right, the wording was not. Every row was
rewritten before it entered the builder.

### Defects found in the draft and fixed

| # | Source | Defect | Fix |
|---|---|---|---|
| 1 | [36] | Range floor **wrong, and from the wrong model**. Draft said the Llama gap runs "+5.7 to +17.3"; +5.7 is a Gemma 8-bit gap. Table 4's six Llama gaps are +7.3, +4.5, +12.1, +16.8, +13.0, +17.3. | Corrected to 4.5 to 17.3. Caught independently by both lenses. |
| 2 | [10] | **Fabricated quotation.** "no method dominates" was set in quotation marks as the authors' phrase. That string appears in neither arXiv version. | Reworded to the paper's actual sense: no single method universally outperforms the others. Caught by both lenses. |
| 3 | [36] | **Absence claim over-reached.** Draft asserted the paper contains no attack-free matched contrast "at all"; Table 3 scores the non-AWQ methods directly on AdvBench, and Table 9 pairs fp16 with INT4/INT8. | Claim narrowed and removed from the table cell. |
| 4 | [37] | "fp16 dense" is **not entailed**. The source says only "16-bit dense" / "Bits 16, Method none" and never states the float format — on the very axis the row contrasts. | Changed to "16-bit dense". |
| 5 | [39] | "applied at load time" and "unmodified release checkpoints" were **inferences**, not statements. The paper says only that its default precision is int8, and its Gemma citation resolves to a third-party repository. | Reworded to the paper's own framing. |
| 6 | [39] | **Internal inconsistency**: the caveat cited the 0.00→0.97 figure, which is the NF4 cell, while the row's setting said int8. | The bare delta was removed entirely; the headline is now described without a number that could be misread as a quantization effect. |
| 7 | all | **Scale hazard.** Prior work reports percentage points on 0–100; this study reports proportions. Uncorrected, −0.090 reads as *smaller* than −1.5, inverting the comparison. | The intro paragraph states the conversion, and this study's row carries both forms. |
| 8 | [37] | **Size-regime non-overlap unstated.** Hong's smallest model is 7B; this study's surviving capability effect is on a 1.7B model, and Hong's own text reports compression is less reliable for smaller models. A reader could take −1.5 as refuting −0.090. | Named explicitly in the obstacle column. |
| 9 | [10] | **Polarity inversion unstated.** The source metric is a safety ratio, so positive means *less* safe — the opposite orientation to the neighbouring ASR rows. | Stated in the cell and again in the following paragraph. |
| 10 | [10] | Scorer's "error" label fires on incoherent output, so a fall in safety ratio can be **pure capability loss** — the exact confound this study exists to separate. | Named in the obstacle column. |
| 11 | [35] | "points" applied to the levels as well as the difference; the levels are percentages of a 56-query set. | Reworded; the six-query magnitude is now stated. |
| 12 | [36] | The 29.8-point fp16 reference is **almost entirely attack effect** — the same checkpoint scores 0.3 un-attacked. Without that, the row reads as a direct-request base rate. | Stated in the obstacle column. |

### Figures as imported

| Study | Value | Located at | Verified |
|---|---|---|---|
| Decoding Compressed Trust [37] | MMLU 53.6 → 52.1 (−1.5) at 13B; −1.6 at 7B; −0.9 at 70B | Table 3 p.16, MMLU column, all three size blocks | local PDF, both lenses |
| Kharinaev et al. [10] | "Difference from FP 16": XSafety +0.571 / +0.536 / +4.500; OpenSafetyMini +3.561 / −0.187 / +8.622 | v2 PDF Table 9 (= v1 Table 2), Llama-3.1-8B-Instruct block, int4 rows | arXiv v1 and v2, values identical; both lenses |
| HarmLevelBench [35] | Human ASR, Simple Query: 33.9 → 44.6 (+10.7) | Table 2 vs Table 3, "Simple Query" row, Human ASR column | arXiv HTML, both lenses |
| Q-resafe [36] | ASR 29.8 → 37.1 (+7.3) at τ(0.95); range +4.5 to +17.3 | Table 4, Llama sub-table, Baseline and AWQ 4-bit rows | arXiv HTML, both lenses |
| Q-Misalign [39] | Control column: 0.07→0.07, 0.05→0.05, 0.00→0.00 | Table 1 p.8, "Pre-trained Model Mp" column | local PDF, both lenses |

No study was excluded. The pre-registered priority order selected the control condition for [39]
over its far more quotable headline, and the direct-query condition for [35] over its attack rows.

## Version and venue checks

- **[10] Kharinaev.** arXiv v2 HTML returns 404; the values were read from v1 HTML and the v2 PDF
  and are **byte-identical between the two versions**. The published IEEE Access text could not be
  read (bot-blocked), but the venue is confirmed via the Crossref record for
  doi:10.1109/ACCESS.2026.3703899 — *IEEE Access*, vol. 14, pp. 96771–96793, matching title and
  authors. The imported figures are from the version-stable Llama block, not the v2-only Qwen rows.
- **[36] Q-resafe.** Genuinely archival, and the citation was **wrong in two ways**. Verified
  against the PMLR v267 proceedings listing and publisher page: the version of record is titled
  "Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language
  Models" — without the "Q-resafe:" prefix — at pp. 9728–9746. The entry also omitted the volume
  and pages that the report's own house style supplies for other ICML references. Both fixed.
- **[35] HarmLevelBench.** **Non-archival.** Six independent checks (Crossref: no record; DBLP:
  indexed as CoRR "Informal and Other Publications"; PMLR: no such volume; NeurIPS publishes no
  workshop proceedings; the workshop's own call for papers states accepted papers are not archived;
  OpenReview confirms genuine acceptance and presentation). "in Proc." asserted a proceedings volume
  that does not exist, so under D56 the entry is now labelled a preprint with the workshop named as
  a presentation venue. The table caption repeats the evidence grade at the point of use.
- **[39] Q-Misalign.** Citation **accurate as printed** — confirmed as an accepted ICLR 2025
  main-conference paper via the Curran archival proceedings ToC (ISBN 979-8-3313-2085-0, p. 6869),
  the official ICLR proceedings page, the ICLR virtual program, and DBLP `conf/iclr/DongL025`.

The two citation corrections were applied to `scripts/build_fyp_report_v6.js`, which is the shared
reference source the rewrite builder reads. **`scripts/build_fyp_report_v5.js` carries the same two
defects at its own numbering ([14] and [15]) and was deliberately not touched**, because v5 is frozen
by the student's explicit instruction. The LaTeX mirrors cite HarmLevelBench as an arXiv preprint
already and so are unaffected.

## Disclosed deviation from the pre-registered rule

The pre-registered priority order resolves ties between *models* but is silent on ties between
*scorers*. HarmLevelBench reports three scorers for the same two rows, disagreeing widely (+16.1
string match, +10.7 human, +1.7 GPT judge). The human-annotated figure was selected, on the grounds
that this study's own methodological conclusion is that human grounding adjudicates between scorers,
and that Table 2.1 already labels this source "Human / mixed". The selected figure is the middle of
the three, so it is not the reading most favourable to this study's null; all three are printed in
the row's scorer cell so nothing is concealed. Recorded here rather than silently absorbed.

## What this does not establish

Verification reduces the risk that a sentence outruns its source. It does not make these five
studies commensurable with this one, and the table does not claim they are. Every row measures a
different intervention, prompt set or scorer; the obstacle column names the chief break in each
case; and no row is offered as a replication or a refutation of any other.
