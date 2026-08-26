# Deliberately Left Technical — v5 Language Pass (2026-08-25)

Occurrences that were reviewed and intentionally kept because simplifying them
would lose statistical precision, break a repo-locked scope sentence, or remove
necessary terminology. Each is stated with its reason.

1. **RQ5 verbatim** — "Does the substantive conclusion remain credible under
   alternative scorers, generation lengths, and decoding seeds?" (§1.3).
   Research questions are preserved exactly; "substantive conclusion" and
   "remain credible" stay. The surrounding prose paraphrases RQ5 in plainer
   words ("RQ5 asks whether the primary conclusions hold…").

2. **"Preservation under direct requests is not evidence that the same model
   will remain safe under a targeted attack."** (§2.6) — a locked scope
   sentence (`tests/test_report_scope_invariants.py`); its rigidity is
   load-bearing for the threat-model boundary.

3. **"A result obtained with one method is evidence about that intervention…"**
   (§2.3) — the named-intervention scope rule; "intervention" is the precise
   word for "the method actually applied", and the sentence exists to stop
   generalisation from NF4 to all four-bit methods.

4. **"sensitivity analysis" / §5.4 heading "XSTest Scorer Sensitivity and
   Human Validation"** — standard methodological vocabulary for the
   scorer-robustness checks; the prose around them now explains what varies.

5. **Statistical machinery in §3.5** — paired bootstrap confidence intervals,
   exact McNemar tests, Benjamini–Hochberg false-discovery-rate control,
   q-values, Cohen's kappa, two one-sided tests / equivalence testing, minimum
   detectable effect, statistical power, null hypothesis, exact-match accuracy.
   These are the genuinely necessary technical terms; the pass removed the
   shorthand *around* them, not the terms.

6. **"audit trail" (§3.8)** — conventional record-keeping term for keeping
   original and derived scoring files separately; plain and precise.

7. **"diagnostic procedures/outputs" (§3.8)** — adjective use ("serving to
   identify the source of a discrepancy"), not compressed jargon.

8. **Figure 4.2's in-image title** — "Precision sweep: capability cliff at
   four-bit; safety is method-specific" remains as the literal figure label;
   figures were not regenerated in this language pass (per task rules). The
   caption and body text around it are plain.

9. **Table row/cell labels left as-is**: "Contrast set only" (Table 2.1,
   OR-Bench row), "unsafe items are a calibration contrast" (Table 2.1, XSTest
   row), "True practical preservation" (Table 5.3 row label), and
   "Attack-dependent construct" (Table 2.1, HarmLevelBench row). These are
   table labels, not prose; changing them risks inconsistency with the
   registered analysis outputs they name.

10. **"broad degradation", "alignment degradation/improvement", "capability
    loss with lower harmful compliance"** — retained only as defined
    interpretation labels (§3.9 definition, §4.5 usage with in-sentence
    explanations). The prose no longer relies on them unexplained.

11. **"the effect disappears at 512" (Table 5.2 cell)** — "disappears" is kept
    in the table cell for compactness; the body text states the same finding
    without metaphor ("the 128-token Qwen3-1.7B anomaly does not reappear at
    the 512-token reference budget").
