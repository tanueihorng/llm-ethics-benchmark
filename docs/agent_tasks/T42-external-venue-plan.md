# T42 — External venue plan (post-FYP workshop paper / preprint)

**Status:** PLANNED (scoping packet; no external action taken). Created 2026-07-19 by Claude (Fable 5).
**Owner of every outward-facing step:** the user. An agent must never submit, post, or email anything external for this task — it drafts and formats only.
**Precondition:** the FYP itself is submitted first (T15). Nothing here may delay or destabilize the FYP deliverables.

---

## 1. Why this is worth doing

The study's rigor is above typical venue-workshop bar: locked pre-registration with a
disclosed deviation (D52), three independent HarmBench judges (official classifier,
gpt-4o, revision-pinned LlamaGuard), human grounding on both axes (T30/T36), BH-FDR
families incl. a composition-locked validation-informed parallel family (D51),
5-pair × 3-seed sensitivity, and a scorer-invariant headline. The two publishable
stories:

1. **A clean negative result:** at the official 512-token budget, no statistically
   significant safety change from NF4/INT8 quantization survives FDR anywhere,
   under any of three judges.
2. **A measurement-validity cautionary tale:** the 128-token era "significant"
   ΔASR (+0.055) was a truncation artifact, and the one FDR-surviving over-refusal
   contrast was a regex measurement artifact (strict κ 0.485 vs −0.006 against a
   blinded human annotator). "Your scorer and your token budget can manufacture
   findings" is the more novel contribution.

## 2. Related work — positioning is now REQUIRED (found 2026-07-19)

A live arXiv literature exists on exactly this topic; any submission must position
against it (none of these were in the FYP's frame when the study was designed):

- **The Joint Effect of Quantization and Sampling Temperature on LLM Safety
  Alignment: A Factorial Analysis** — arXiv:2606.29581 (June 2026). Finds standard
  non-adversarial quantization "usually safety-neutral" — **agrees with our null**.
  Closest neighbour; must be cited and differentiated.
- **Alignment-Aware Quantization for LLM Safety** — arXiv:2511.07842 (claims
  quantization can erase RLHF guardrails; proposes CAQ).
- **Preserving Fairness and Safety in Quantized LLMs Through Critical Weight
  Protection** — arXiv:2601.12033 (QAT degrades safety; PTQ inconclusive).
- **Q-realign** — arXiv:2601.08089 (realignment during quantization).

**Our differentiators vs 2606.29581 and the field:** (a) the measurement-validity
layer — three judges + human gold sets + pre-registration; nobody else grounds their
scorer; (b) the 128-token truncation-artifact demonstration (a "significant" safety
finding that dissolves at the official budget); (c) matched-pair on-the-fly NF4/INT8
across 4 families with per-pair McNemar + FDR discipline. Frame the paper around
(a)+(b), with the null as the payload — a pure "quantization is safe" null is no
longer novel on its own.

## 3. Candidate venues (verified 2026-07-19; re-verify dates before acting)

| Venue | Deadline | Fit | Notes |
|---|---|---|---|
| **NeurIPS 2026 safety workshops** (Sydney, workshops Dec 11–12) | Workshop list not yet out; paper deadlines typically late Aug–early Sep 2026 | **Best effort/reward.** Workshop papers are short (4–9 pp), often non-archival (keeps journal/conference options open), reviewer pool is exactly this community | Watch neurips.cc for the accepted-workshop list (~Aug); target a safety/trustworthy-LM workshop (SoLaR-style, Safe Generative AI successor) |
| **IEEE SaTML 2027** (Reykjavik, May 8–10 2027) | **Sep 29, 2026 AoE** (confirmed on satml.org) | **Best archival fit.** Secure & trustworthy ML is the exact scope; welcomes research + position papers | Full-conference bar; would benefit most from T30b landing first |
| **ICLR 2027** | ~Sep 16–24, 2026 (predicted, unconfirmed) | Stretch — main-conference bar for a negative result is high | Only if supervisor pushes for it |
| **arXiv preprint** (cs.CL or cs.CR) | Anytime | Zero-cost priority stake; citable | May need endorsement for a first-time submitter; supervisor can endorse. Post after (or simultaneous with) FYP submission, with supervisor's blessing |

## 4. Required work (in order)

1. **User/supervisor gate (blocking, not agent work):** discuss with Dr. Zhang —
   co-authorship, NTU IP/publication policy for FYP work, and which venue tier.
   Natural moment: fold one paragraph into the staged July email (T1) or the
   next meeting.
2. **T30b second annotator** — the single biggest reviewer-proofing step
   (inter-annotator κ ceiling for both gold sets). Do before SaTML; nice-to-have
   before a workshop deadline.
3. **Condense to paper form:** the LaTeX mirrors in `fyp_submission/report_latex/`
   are the source; a workshop paper is a 4–9 pp cut (headline: measurement validity
   + null; move the harness/agent chapters out entirely). New related-work section
   per §2 — none of the four papers above are currently cited anywhere in the repo.
4. **Reproducibility statement:** results trees + sidecars are already committed
   and redacted; decide whether to point at a public snapshot of the repo
   (check for anything NTU-internal first: TC1 paths, emails, supervisor info).
5. **Venue-date re-verification** at execution time (the ICLR dates above are
   predictions; the NeurIPS workshop list does not exist yet).

## 5. Recommended path (agent's recommendation, user decides)

After T15: (1) supervisor conversation; (2) arXiv preprint of the condensed paper
once approved; (3) submit to a NeurIPS 2026 safety workshop when the list lands in
August; (4) SaTML 2027 (Sep 29 deadline) as the archival target if T30b lands and
the workshop reception is good. If nothing else happens, do step 2 alone — it is
cheap and makes the work citable.

## 6. Sources (checked 2026-07-19)

- SaTML 2027 CFP: https://satml.org/call-for-papers/ (deadline Sep 29, 2026)
- NeurIPS 2026 dates: https://neurips.cc/Conferences/2026/Dates (Sydney, Dec 6–12; workshops Dec 11–12)
- ICLR 2027 predicted dates: https://mlciv.com/ai-deadlines/conference/?id=iclr27
- arXiv:2606.29581, arXiv:2511.07842, arXiv:2601.12033, arXiv:2601.08089 (related work, §2)
