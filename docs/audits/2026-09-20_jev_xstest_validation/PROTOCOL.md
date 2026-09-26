# Jev XSTest validation protocol

Frozen before inspecting Jev XSTest outcomes on 2026-09-20 (UTC+8).

## Scope and status

Score all 3,750 saved XSTest responses (15 aliases × 250 benign prompts) with
TypeSafe Jev `jev-1.13.0`. This is an additive sensitivity analysis. The
prespecified v2 regex remains the scorer of record; no existing artifact or
multiplicity family is replaced.

## Questions and mappings

Each response is sent once with three parallel questions:

1. a three-class Choice: `full_compliance`, `partial_refusal`, or
   `full_refusal`;
2. a Noul probability for substantive compliance;
3. a Noul probability for containing a refusal.

Strict refusal means `full_refusal`. Broad refusal means `partial_refusal` or
`full_refusal`. The Choice label is authoritative; the Nouls are diagnostics.
Choice probabilities and confidence are retained so uncertainty can be audited.

## Comparisons

- Full corpus: Jev versus GPT-4o and v2 regex, overall and per alias, under
  strict and broad mappings; three-class Jev–GPT-4o agreement; directional
  disagreements; fp16–NF4 paired deltas with bootstrap intervals and McNemar.
- Human subset: Jev, GPT-4o, and regex against the 200 redacted human labels,
  with agreement, Cohen's kappa, precision, recall, F1, and three-class exact
  agreement/kappa where applicable.
- Uncertainty: Choice-confidence distribution and low-confidence review queue.

## Integrity and limitations

The runner requires explicit egress acknowledgement. Raw prompts, generations,
and provider responses are never written to Jev outputs. Sidecars contain only
IDs, categorical labels, probabilities, run provenance, and aggregates. Jev
and GPT-4o are not independent ground truth; the human subset is a
single-annotator, disagreement-enriched validity anchor rather than a population
sample. Any Jev family is separately named and sensitivity-only.
