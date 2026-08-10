#!/usr/bin/env python3
"""Atomic remediation of the 26 claim-to-source audit findings (T48) in v6.

Content-keyed exact-string replacement. Nothing is written unless EVERY gate
passes, so a partial application is impossible.

Gates:
  G1  each old string occurs exactly the expected number of times
  G2  no model name is lost, swapped, or invented
  G3  no number disappears from the document (additions are listed for review)
  G4  no reference is orphaned (every [n] in 1..N still cited in the body)
  G5  reference numbering is still strictly by first use
"""
import collections
import re
import sys

TARGET = "/Users/tanueihorng/fyp_quant/scripts/build_fyp_report_v6.js"

# (finding_id, expected_occurrences, old, new)
EDITS = [

    # ---- Chapter 1: uncited motivation ---------------------------------------
    ("C12", 1,
     "Tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, with stronger privacy guarantees, lower latency, and substantially reduced operational cost.",
     "Tasks that previously required cloud-hosted seven-to-seventy-billion-parameter models can now be executed locally, which keeps user data on the device and can reduce inference latency [5]."),

    ("C11+C03", 1,
     "Consequently, quantization to four-bit precision has become the de facto compression standard for on-device inference. Lightweight runtimes and on-device agent frameworks routinely ship four-bit GGUF or BitsAndBytes NF4 checkpoints by default, and four bits is where the compression literature concentrates its own evaluation effort [8], [9]. End users encountering these models therefore typically interact with a quantized variant rather than the original baseline, whereas the safety evidence published alongside a model release is measured on the full-precision checkpoint [10].",
     "Consequently, four bits is the operating point that compression research targets for on-device inference. AWQ is designed explicitly for on-device compression and ships four-bit weights in its own desktop and mobile runtime [5], and four bits is where the compression literature concentrates its own evaluation effort [8], [9]. The BitsAndBytes NF4 scheme evaluated in this study is one such four-bit format. How often an end user in fact receives a quantized rather than a full-precision checkpoint is not something this study measures, and no prevalence figure is claimed for it. The premise that motivates the work is narrower and is established in the literature: quantization can change safety behaviour [10], so a safety evaluation carried out on a full-precision checkpoint does not automatically transfer to the quantized artefact derived from it."),

    ("C26", 1,
     "Safety alignment is a learned behaviour, instilled by preference optimisation over human feedback [14], and it therefore resides in the model weights as any other learned behaviour does.",
     "Safety alignment is a learned behaviour, instilled by preference optimisation over human feedback [14]. The premise of this study is that it is therefore carried in the model weights as other learned behaviour is, and is exposed to any operation that modifies them."),

    ("C13a", 1,
     "fine-tuning on as few as ten adversarially chosen examples can strip alignment outright [17]",
     "fine-tuning on as few as ten adversarially chosen examples can substantially compromise safety guardrails [17]"),

    # ---- Chapter 2: source fit ------------------------------------------------
    ("C04", 1,
     "The evaluated object in all three frameworks is the released full-precision checkpoint.",
     "None of the three frameworks makes quantization a controlled experimental variable, and for the API-served systems that DecodingTrust covers the serving precision is not disclosed at all."),

    ("C02", 1,
     "Its authors evaluate twenty-five models across eight families.",
     "Its authors evaluate thirty-two models across eight families."),

    ("C06", 1,
     "XSTest is retained as the instrument here because its two hundred and fifty benign prompts are human-curated and because its published scoring protocol is the one the present study needs to reproduce, but the existence of OR-Bench establishes",
     "XSTest is retained as the instrument here because its two hundred and fifty benign prompts are human-curated and because its three-class response taxonomy transfers directly to the design used here. Its own primary evaluation was manual annotation by three of its authors, with automated string-matching and a GPT-4 classifier reported only as cheaper alternatives, so the deterministic regex used in this study is an adaptation of that taxonomy rather than a reproduction of the published protocol. The existence of OR-Bench nevertheless establishes"),

    ("C20", 1,
     "The threat model matters as much as the instrument. HarmBench's standard configuration presents harmful behaviours directly, with no adversarial augmentation.",
     "The threat model matters as much as the instrument. This study presents HarmBench behaviours directly, with no adversarial augmentation, which corresponds to the DirectRequest baseline in HarmBench's own taxonomy of red-teaming methods rather than to any of its attack methods. The label 'standard' in HarmBench denotes a functional category of behaviour, alongside contextual, copyright and multimodal, and not an absence of attack."),

    ("C13b", 1,
     "Qi et al. [17] demonstrate that safety alignment can be stripped by fine-tuning on as few as ten adversarially designed examples, at a cost of well under a dollar.",
     "Qi et al. [17] demonstrate that safety guardrails can be substantially compromised by fine-tuning on as few as ten adversarially designed examples, at a cost of well under a dollar."),

    # ---- Contributions --------------------------------------------------------
    ("C05", 1,
     "two confounds that affect most published comparisons of quantized and full-precision checkpoints.",
     "two confounds that affect comparisons assembled from separately published checkpoints."),

    ("C18a", 1,
     "The signal that survives the multiplicity correction is capability loss rather than safety change.",
     "Three contrasts survive the multiplicity correction in the registered family: two capability losses and one over-refusal decrease that later scorer validation attributes most plausibly to the regex. No harmful-compliance contrast survives on any pair."),

    # ---- Chapter 6: statistics -----------------------------------------------
    ("C15", 1,
     "It does, however, have enough power to resolve the capability losses, which are both larger and survive the multiplicity correction.",
     "The capability axis was checked against the same detection floor rather than assumed to be adequately powered. Running the identical procedure on the MMLU and ARC contrasts gives per-contrast minimum detectable effects of 0.052–0.072 on MMLU (n = 300) and 0.021–0.033 on ARC (n = 1 172), and only two of the ten capability contrasts have an observed effect at or above their own floor: Qwen3-1.7B MMLU (−0.090 against a floor of 0.072) and Llama-3.2-3B ARC (−0.032 against 0.026). Those two are precisely the two that survive the multiplicity correction. The remaining eight capability contrasts are underpowered in the same way the safety axis is, so the capability finding should be read as two adequately resolved effects rather than as a well-powered survey of the whole capability axis."),

    ("C14a", 1,
     "Exact McNemar tests recomputed from the paired per-item records give ten contrasts, none of them significant, with a largest |Δ| of 1.3 pp. Within the study's resolution, INT8 is therefore capability-lossless.",
     "Exact McNemar tests recomputed from the paired per-item records give ten contrasts, none of them significant, with a largest |Δ| of 1.3 pp. Absence of significance is not by itself evidence of equivalence, so the same ten contrasts were also submitted to a two-one-sided-tests (TOST) equivalence procedure against a margin of ±0.05, fixed on deployment-relevance grounds before the test was run. All ten are statistically equivalent to fp16 at that margin, with a largest p of 0.007, and the widest 90% confidence interval closes at ±0.038. INT8 capability is therefore equivalent to full precision to within about four percentage points, which is a stronger and more precise statement than the absence of a significant difference. The equivalence analysis is post hoc, having been added after the INT8 results were known."),

    ("C14b", 1,
     "INT8 is capability-lossless and safety-neutral at 512 tokens, and no harmful-compliance increase, whether INT8 or NF4, survives at the reference budget.",
     "INT8 capability is statistically equivalent to full precision within ±0.05 at 512 tokens, its safety contrasts are null, and no harmful-compliance increase, whether INT8 or NF4, survives at the reference budget."),

    # ---- Chapter 6: unsupported guarantees ------------------------------------
    ("C16", 1,
     "However, the matched-pair deltas, scored identically for both members of every pair, are unaffected.",
     "However, because both members of every pair are scored under the identical template, the matched-pair deltas are internally controlled for it. Whether a delta computed under the harness variant would take the same value was not tested."),

    ("C08", 1,
     "For this reason a fully independent open-weight guard, Llama Guard [51], is the natural complementary cross-check.",
     "For this reason an open-weight third judge applying a different safety-classification construct, Llama Guard [51], is the natural complementary cross-check. Its independence is partial: Llama-Guard-3-8B is itself a Llama-family model, so for the Llama-3.2-3B pair in particular it is not lineage-independent of the system it scores."),

    ("C07b", 1,
     "The saved generations were re-scored by a second, architecturally unrelated frontier judge (gpt-4o) applying the same HarmBench rubric.",
     "The saved generations were re-scored by a second frontier judge from a different provider (gpt-4o), applying the same HarmBench rubric. Its architecture is undisclosed, so the independence that can be claimed is of provider, training pipeline and invocation, not of architecture."),

    # ---- Chapter 6: mechanism inference ---------------------------------------
    ("C09", 1,
     "This is what the mechanistic literature predicts. Refusal is mediated by a single activation direction [15], and safety alignment is concentrated over the first few generated tokens [16], so a weight perturbation should move prompts across a refusal boundary rather than erode the behaviour uniformly.",
     "This is what an extrapolation from the mechanistic literature would suggest. Refusal is mediated by a single activation direction [15], and safety alignment is concentrated over the first few generated tokens [16]. Neither study examines quantization, so the step from those findings to the expectation that a weight perturbation should move prompts across a refusal boundary rather than erode the behaviour uniformly is this study's own, and the probe below is what tests it."),

    ("C10", 1,
     "That headroom is the mechanistic reason the Qwen3-4B pair is behaviourally intact while the Qwen3-1.7B pair is not.",
     "That headroom is consistent with the Qwen3-4B pair remaining behaviourally intact where the Qwen3-1.7B pair does not, although a two-pair comparison is observational and does not isolate headroom as the cause."),

    # ---- Discussion / conclusion ---------------------------------------------
    ("C19", 1,
     "Therefore no family shows an NF4-induced safety regression, while the capability-cost pattern holds across four families.",
     "Therefore no pair shows a statistically significant NF4-induced increase in harmful compliance, while the capability-cost pattern holds across four families."),

    ("C21", 1,
     "The one prior study that, like this one, isolates quantization at four bits is Hong et al. [37], which works across a broader trust taxonomy.",
     "The closest prior comparison in trustworthiness scope is Hong et al. [37], which also isolates quantization at four bits and works across a broader trust taxonomy."),

    ("C22+C24b", 1,
     "Badshah and Sajjad [29] report that larger models are notably resilient to four-bit precision reduction, and the present result is the complementary half of that observation: in the compact regime the capability cost is the effect that survives correction, while the safety axis stays null. Capability degradation is directionally consistent across all five pairs, reaches significance on at least one benchmark for three pairs, and is the only class of effect that survives multiplicity correction.",
     "Badshah and Sajjad [29] report in a preprint that larger models are notably resilient to four-bit precision reduction, and the present result is the complementary half of that observation: in the compact regime the capability cost is the effect that survives correction, while the safety axis stays null. Capability degradation is directionally consistent on MMLU across all five pairs and on ARC in four of five, reaches significance on at least one benchmark for three pairs, and accounts for two of the three contrasts that survive multiplicity correction."),

    ("C23", 1,
     "Therefore it provides the strongest practical isolation of quantization as the experimental variable.",
     "Therefore it provides controlled isolation of the quantization step as the experimental variable under the recorded configuration."),

    ("C17+C25", 1,
     "Whether benchmark items appear in the evaluated models' pretraining data is not measured, and there is at present no community-standard instrument for measuring it per benchmark [54]. Contamination would shift absolute scores. However, it applies identically to both members of each pair (same weights, same data history), so the paired deltas, which are the study's unit of inference, are unaffected. Only absolute levels carry this caveat.",
     "Whether benchmark items appear in the evaluated models' pretraining data is not measured here, and the literature has called for per-benchmark contamination measurement precisely because such instruments are not yet routine [54]. Contamination would shift absolute scores. Because both members of a pair share one set of weights and one data history, the pairing controls exposure, which is what the study's unit of inference requires. It does not rule out an interaction between quantization and memorised items, and no test in this study measures one."),

    # ---- Bibliography: evidence grade (C24 + the eleven Codex missed) ---------
    ("N1-bai", 1, "Anthropic, 2022. arXiv:2204.05862.", "Anthropic, 2022. arXiv:2204.05862 (preprint)."),
    ("N1-zou", 1, "2023. arXiv:2307.15043.", "2023. arXiv:2307.15043 (preprint)."),
    ("N1-badshah", 1, "2024. arXiv:2405.03146.", "2024. arXiv:2405.03146 (preprint)."),
    ("N1-llamaguard", 1, "Meta AI, 2023. arXiv:2312.06674.", "Meta AI, 2023. arXiv:2312.06674 (preprint)."),
    ("N1-gu", 1, "arXiv:2411.15594, 2024.", "arXiv:2411.15594, 2024 (preprint)."),
    ("N1-krumdick", 1, "arXiv:2503.05061, 2025.", "arXiv:2503.05061, 2025 (preprint)."),
]

MODEL_PATTERNS = [
    (r'Qwen\s?3?[- ]?\d?\.?\d*\s?B?', 'QWEN'), (r'Llama[- ]?\d(\.\d)?[- ]?\d*\.?\d*\s?B?', 'LLAMA'),
    (r'Mistral[- ]?\d*\.?\d*\s?B?', 'MISTRAL'), (r'Phi[- ]?\d*[- ]?\w*', 'PHI'),
    (r'Gemma\s?\d', 'GEMMA'), (r'GPT[- ]?\d(\.\d)?o?', 'GPT'),
]
BARE_SIZE = r'(?<![\d.])(1\.7|3\.2|7\.2|2\.5|13)\s?[Bb]\b'


def body(s):
    """Document strings only, with the bibliography array removed."""
    i0 = s.find("const refs = [")
    i1 = s.find("];", i0)
    return " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', s[:i0] + s[i1:]))


def mask(x):
    for p, t in MODEL_PATTERNS:
        x = re.sub(p, ' ' + t + ' ', x, flags=re.I)
    return re.sub(BARE_SIZE, ' SZ ', x)


def numbers(s):
    return collections.Counter(re.findall(r'\d+(?:\.\d+)?', mask(body(s))))


# Every model identity the report is allowed to name. An edit that introduces a
# model outside this set has invented one; an edit that empties a family that was
# previously present has dropped one. Counting mentions is NOT the right test —
# these edits legitimately add mentions (e.g. naming the two resolved contrasts).
VALID_MODELS = {
    ("qwen", "1.7"), ("qwen", "2.5"), ("qwen", "3"), ("qwen", "4"),
    ("llama", "2"), ("llama", "3"), ("llama", "3.1"), ("llama", "3.2"),
    ("mistral", "7"), ("phi", "3"), ("phi", "4"), ("gemma", "2"),
    ("gpt", "3.5"), ("gpt", "4"),
}


def model_identities(s):
    """Distinct (family, version) pairs actually named in the document."""
    found = set()
    txt = body(s)
    for fam, pat in (
        ("qwen", r'Qwen\s?3?[- ]?(\d+(?:\.\d+)?)\s?B'),
        ("llama", r'Llama[- ]?(\d+(?:\.\d+)?)'),
        ("mistral", r'Mistral[- ]?(\d+(?:\.\d+)?)\s?B'),
        ("phi", r'Phi[- ]?(\d+)'),
        ("gemma", r'Gemma\s?(\d+)'),
        ("gpt", r'GPT[- ]?(\d+(?:\.\d+)?)'),
    ):
        for m in re.finditer(pat, txt, flags=re.I):
            found.add((fam, m.group(1)))
    return found


def models(s):
    """Kept for the family-presence check: which families are named at all."""
    return {fam for fam, _ in model_identities(s)}


def refs_of(s):
    i0 = s.find("const refs = [")
    i1 = s.find("];", i0)
    return re.findall(r'\n    "((?:[^"\\]|\\.)+)",', s[i0:i1])


def citation_order(s):
    i0 = s.find("const refs = [")
    i1 = s.find("];", i0)
    doc = s[:i0] + s[i1:]
    seen, order = set(), []
    for m in re.finditer(r'\[(\d{1,3})\]', doc):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            order.append(n)
    return order


def main():
    before = open(TARGET, encoding="utf-8").read()
    after = before
    problems = []

    # ---- G1: apply, checking occurrence counts -------------------------------
    for fid, want, old, new in EDITS:
        got = after.count(old)
        if got != want:
            problems.append(f"{fid}: expected {want} occurrence(s), found {got}")
            continue
        after = after.replace(old, new)
    if problems:
        print("ABORT (G1 anchor mismatch), nothing written:")
        for p in problems:
            print("   " + p)
        sys.exit(1)

    # ---- G2: models: none lost, none invented --------------------------------
    fam_b, fam_a = models(before), models(after)
    ident_b, ident_a = model_identities(before), model_identities(after)
    dropped_fam = fam_b - fam_a
    new_ident = ident_a - ident_b
    # Only NEWLY introduced identities are this pass's responsibility. The
    # extractor picks up a couple of pre-existing artifacts (e.g. 'GPT-3' from
    # 'GPT-3.5'), which are not defects these edits created.
    invented = new_ident - VALID_MODELS
    if dropped_fam or invented:
        print("ABORT (G2 model drift), nothing written:")
        if dropped_fam:
            print(f"   families no longer named: {sorted(dropped_fam)}")
        if invented:
            print(f"   model identities not in the allowlist: {sorted(invented)}")
        sys.exit(1)

    # ---- G3: numbers ---------------------------------------------------------
    nb, na = numbers(before), numbers(after)
    lost = nb - na
    if lost:
        print("ABORT (G3 numbers removed), nothing written:")
        print("   " + str(dict(lost)))
        sys.exit(1)
    gained = na - nb

    # ---- G4: no orphaned reference -------------------------------------------
    n_refs = len(refs_of(after))
    cited = set(citation_order(after))
    orphans = [n for n in range(1, n_refs + 1) if n not in cited]
    if orphans:
        print(f"ABORT (G4 orphaned references {orphans}), nothing written")
        sys.exit(1)

    # ---- G5: first-use ordering ----------------------------------------------
    order = citation_order(after)
    if order != sorted(order):
        bad = [(i, order[i]) for i in range(1, len(order)) if order[i] < order[i - 1]]
        print(f"ABORT (G5 first-use order broken at {bad[:5]}), nothing written")
        sys.exit(1)

    open(TARGET, "w", encoding="utf-8").write(after)
    print(f"applied {len(EDITS)} edits, all gates passed")
    print(f"  G2 families preserved: {sorted(fam_a)}; new identities: {sorted(new_ident) or 'none'}")
    print(f"  G3 numbers removed: 0; newly introduced (review): {dict(gained)}")
    print(f"  G4 references: {n_refs}, all cited")
    print(f"  G5 first-use order: strictly increasing over {len(order)} refs")


if __name__ == "__main__":
    main()
