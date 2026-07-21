# T44 Phase 5 — omission audit results

Session S6, 2026-07-21. Workflow `wf_94d4c07f-ae6`: 5 Opus-xhigh absence-direction seats, every
finding → 2 adversarial refuters (absence claims re-searched under phrasings the finder didn't
try). 17 agents, 0 errors, ~1.7M subagent tokens. Already-adjudicated candidates (FS-1..FS-16 +
Phase 2B's refuted absence list) excluded up front.

## Seat verdicts

| Seat | Verdict | Notes |
|---|---|---|
| artifact-limitations | **no_material_omissions, 0 findings** | 8 candidate classes checked and all found disclosed — seed scope, GPT-4o API egress (explicit data-handling note), judge over-flagging both axes, the mechanism probe's unfavourable 2-of-5 slice, the qwen_2b category-masking (+0.18/−0.18), parser-sensitivity scoping, multiseed greedy-outside-range, power floors. Seat's own words: "one of the most thoroughly hedged FYP write-ups I have audited" |
| code-vs-prose | minor_omissions | 2 P3s, both killed (parser tier-4 bare-numeric fallback already bounded by the reported strict/lenient bracket; judge input caps inert at 512 tokens) |
| negative-arms | minor_omissions | 1 P3 killed (thesis/interim compress the probe's validity failures the report details — legitimate compression) |
| prereg-outcomes | **no_material_omissions, 0 findings** | every clause of both preregs has its outcome stated |
| repro-checklist | minor_omissions | 3 findings: 2 killed (bitsandbytes pin surfaced adequately; compute budget standard-practice omission), 1 SPLIT → adjudicated → **FS-20** |

## The one survivor — FS-20 (P3, orchestrator-adjudicated split)

The HarmBench gpt-4o second judge (Result 4, cross-judge κ 0.68–0.95) is recorded as the bare
floating "gpt-4o" alias — prose AND committed sidecars carry no snapshot date — while the
sibling XSTest judge is pinned to `gpt-4o-2024-08-06`. Real internal inconsistency (both
refuters concur); not load-bearing (open-weight primary scorer; revision-pinned LlamaGuard
explicitly retires the versioned-API caveat). Fix: state the run date at Result 4; retroactive
snapshot pinning is impossible, so honest bounding is the correct remediation.

## Killed-findings record (why each died)
- Parser tier-4 numeric fallback unnamed: bounded by the reported strict/lenient bracket; no number moves.
- Judge 2048-token/6000-char input caps: inert at the 512 budget.
- Thesis omits probe validity failures: report carries them; thesis compresses legitimately.
- bitsandbytes version unpinned in prose: surfaced adequately elsewhere; standard practice.
- No GPU-hours/compute budget: standard-practice omission, not load-bearing.

**Phase 5 verdict: no material omissions.** The disclosure surface — limitations, prereg
outcomes, negative arms — is complete to an unusual standard; the single real gap is a
provenance pin on a non-primary judge.
