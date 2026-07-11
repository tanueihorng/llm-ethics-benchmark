# pair_interpretations.{json,csv} — SECONDARY / v2-regex proxy (superseded)

These per-pair interpretation labels are computed from the **v2 refusal-regex**
HarmBench ASR, which the judge validation (decision D16) showed materially
over-counts attack success. This file is retained ONLY as a transparent foil.

The **authoritative, judge-primary** per-pair labels live in
`judge_agreement.json` (field `judge_label`). Where the two disagree — e.g.
Mistral-7B reads `alignment_degradation` here (regex) but `alignment_improvement`
under the judge — the judge label is the headline. See report §6.12 (D16).
