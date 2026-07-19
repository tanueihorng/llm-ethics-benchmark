# TODO — working backlog (not the source of truth)

> **Source of truth is `docs/PROJECT_LOG.md`.** This file is a short, executable
> working queue so the next session/agent can resume mid-task without re-doing
> research. When an item here is finished, record it in PROJECT_LOG.md (§4
> changelog + §2/§3 as needed) and tick/remove it here.

---

## Current state (2026-07-08)

The study is **complete and submission-ready**. D44 pre-submission audit is DONE,
committed + pushed (`0abb641`): report + thesis are audit-clean, significance
criterion unified on McNemar's exact test (Qwen3-1.7B XSTest ΔOR is *borderline*,
not significant), per-seed multiseed evidence persisted, claim lock **58/58**,
pytest **339**, `make agent-check` **8/8**. Everything below is a remaining
*action* (mostly the user's), not blocking correctness.

Durable record: PROJECT_LOG §3 D44 + §4 (2026-07-08 row). Numbers source:
`results_512/analysis/` (`python scripts/number_bible_512.py`).

---

## [2026-07-19] DONE + pruned: Phase-C fold-in of T36 + T37 + T39

Landed in two commits (part A: T36 + D51 validation-informed family, PROJECT_LOG
2026-07-19 11:17; part B: T37 LlamaGuard + T39 5-pair multiseed, 2026-07-19 11:50).
Durable outcomes, decisions (D51), and gate states are recorded in PROJECT_LOG §1–§4;
T36/T37/T39 are ticked in §2. Remaining related backlog: T30b second annotator only.

## ⚠️ [security] ROTATE the OpenAI key

The user pasted an OpenAI API key into chat (twice, earlier sessions). It is **not
in the repo** (redaction scan clean; secret-pattern grep of the pushed tree clean),
but a key exposed in a chat log should be rotated as hygiene. **User decision
2026-07-12: the existing key is used as-is for T35 (assessed not compromised;
account holds only a few dollars, a natural spend cap) and will be DELETED
entirely once T35 completes.** Remaining action: delete the key after T35.

---

## [2026-07-12] NEXT: T36–T39 hardening batch (planned, ready for Opus 4.8)

**Why:** user selected 4 of the 8 assessed weaknesses to close post-T35: the XSTest
human gold set (T36 — "scorer-dependent, but who's right?"), LlamaGuard open-weight
third judge (T37 — kills the versioned-API caveat), strict-parser capability
sensitivity (T38 — bounds the MMLU parser inflation), multi-seed completion for
Mistral/Phi (T39 — 5/5 pairs in §6.6.1). Durable record: PROJECT_LOG §3 **D46** +
§2.1 **T36–T39**.

**Everything an executor needs (do NOT re-research):**
- Execution packet: `docs/agent_tasks/T36-T39-hardening-batch.md` — 4 workstreams,
  Phase A (infra, local) → adversarial review → Phase B (user labels; TC1 jobs B
  then D; T38 local) → Phase C (ONE combined fold-in, one D42 sweep, one commit set).
- T36 pre-reg **LOCKED before any sampling/labeling**: `docs/XSTEST_GOLD_PREREG.md`
  (blind 3-class labeling, ±0.15 κ margin, J/R/T/X matrix). Sheet tool = sibling of
  `scripts/human_label_audit.py` (T30), reusing `xstest_judge_agreement.py` loaders.
- Scout-verified traps (already encoded in the packet): T39 dual hardcoded pair
  lists (`generate_sensitivity_jobs.py:33` + `sensitivity_analysis.py:31`, sync
  test mandated; fix T24 in the same pass); T37 Llama-Guard-3-8B is HF-GATED
  (license + head-node token BEFORE prefetch); T38 `parser_strict` sidecars are
  NOT covered by existing policy globs (add 8 entries + self-test);
  `tc1_sensitivity_512.yaml` predates T26 (verify/add mistral+phi entries).
- First executor step: read the packet end-to-end, then Phase A WS-A.

**Batch rule (D46):** single combined fold-in unless a workstream stalls >1 week.
Current gates baseline: verify-claims 67, pytest 353, agent-check green.

---

## [2026-07-12] DONE: T35 — XSTest independent API refusal judge (sensitivity layer)

**Why:** the one BH-FDR-surviving over-refusal contrast (Phi-4-mini −0.048, q=0.0122)
rests on the v2 refusal regex — the scorer class the study itself demoted for HarmBench.
An independent judge either strengthens RQ2 or honestly exposes scorer-dependence.
Planned by Fable 5; **execution assigned to Opus 4.8**. Durable record: PROJECT_LOG
§3 **D45** + §2.1 **T35**.

**Decided (don't re-litigate):**
- **Sensitivity layer, NOT primary-scorer swap** (D45) — regex stays primary; the
  BH-FDR 20-contrast family and `multiple_comparisons.json` are untouched. Blast
  radius = one §6.12 result + one table.
- **Pre-registration is LOCKED**: `docs/XSTEST_JUDGE_PREREG.md` — STRICT (full refusal
  only; primary for the Phi replication verdict) + BROAD (partial+full; primary for
  regex-agreement κ); binding outcome→reporting matrix A/B/C/D. No post-hoc framing;
  only its Outcome section may be edited, append-only.
- Judge = pinned **dated** gpt-4o snapshot (e.g. `gpt-4o-2024-08-06`), temperature 0,
  max_tokens 10, 3-class XSTest taxonomy (Röttger et al. [5] used a GPT-4 judge with
  this exact taxonomy — cite it in §6.12).
- **Additive sibling code path**: new backend `xstest_api` subclassing
  `OpenAICompatibleJudgeBackend` + new `scripts/run_xstest_judge.py` +
  `scripts/xstest_judge_agreement.py`. Existing HarmBench runner byte-untouched.
- New sidecars `results_512/*/xstest/{scores,summary}.judge.xstest_api.*` with own
  version string `judge_xstest_refusal_api_v1_2026-07-12`.

**Rejected (don't re-litigate):**
- Reusing the HarmBench classifier for XSTest — it answers "is this harmful?", not
  "did the model refuse a benign prompt?".
- Modifying `run_judge_validation.py` — it hard-couples HarmBench field/metric names
  (`judge_harmful`, `attack_success_rate`); sibling path is zero-regression.
- Scoring the 128-token `results/` tree — historical comparison only.
- Open-weight judge (LlamaGuard) — wrong construct for over-refusal; stays future work.

**Verification already done (planning scouts, 2026-07-12):**
- `configs/artifact_policy.yaml` judge globs are **harmbench-dir-anchored** → a new
  xstest judge sidecar would silently escape BOTH `allowed_derived_artifacts` and
  `redaction.scan_paths`. Globs + extended synthetic-leak self-test are Phase 1a,
  before any sidecar exists.
- `openai>=1.0.0` already pinned (requirements.txt:24); key via `OPENAI_API_KEY` env
  only (SDK reads it; existing convention).
- 15 aliases have `results_512/<alias>/xstest/raw.jsonl` (incl. `_8bit`), 250 records
  each; refusal ground truth for κ = `score_fields.is_refusal` in `scores.v2.jsonl`.
- Phi contrast verbatim from `multiple_comparisons.json`: n=250, b=1, c=13,
  delta=−0.048, p=0.00183, q=0.0122, direction=down.
- verify-claims `Checker.check(name, snippets, fn)` pattern confirmed for adding
  Result-6 locks; next free IDs were D45/T35 (now taken).

**Next steps (ordered — full detail in `docs/agent_tasks/T35-xstest-api-judge-sensitivity.md`):**
1. Use the EXISTING key (user decision 2026-07-12; delete it after T35) →
   `export OPENAI_API_KEY=...` (env only, never argv/source/logs).
2. Phase 1: policy globs + self-test; backend + runner + `tests/test_xstest_judge.py`
   (fake client, no network); `pytest tests/ -q` green BEFORE any API call.
3. Phase 2 pilot: `python scripts/run_xstest_judge.py --results-dir results_512 --models phi4_mini_base phi4_mini_4bit --max-samples 20`
   → manually inspect all 40 labels vs local raw text; 0 parse errors required.
4. Phase 3 full run: `python scripts/run_xstest_judge.py --results-dir results_512`
   (~3,750 calls, ~USD 10–20, 1–2 h, local Mac).
5. Phase 4: `python scripts/xstest_judge_agreement.py --results-dir results_512` →
   `results_512/analysis/xstest_judge_agreement.{json,csv}`; determine outcome letter;
   append to prereg Outcome section.
6. Phase 5 (outcome-scoped): §6.12 "Result 6" + table in all 6 builders + LaTeX
   `mythesis.tex` (+ tectonic recompile + rebuild `final_thesis_overleaf.zip`, 24 files,
   exclude pdf) + `docs/RESULTS_CARD.md` + CLAUDE.md/AGENTS.md scope sentence (in sync,
   same commit) + ≥4 new verify-claims checks (self-test one fires) + Appendix D test
   inventory + PROJECT_LOG (tick T35, D45 outcome, §1 sentence, changelog rows).

**Watch items / guardrails:**
- Immutable: `raw.jsonl`, `summary.json`, `scores.v2.*`, all `harmbench/` judge
  sidecars, the 300-file manifest. Sidecars are additive; rollback = delete new files.
- Every reporting sentence must be licensed by the pre-registered outcome letter.
- Egress disclosure goes in the §6.12 setup discussion, NOT Limitations (2026-07-11
  placement convention).
- parse_error_rate > 2% or nonsense pilot ⇒ Outcome D: abort, commit only prereg +
  PROJECT_LOG row.
- No raw prompt/response text in any committed file or in this todo.

---

## [T1] Send the July progress email to Dr. Zhang (READY — user reviews + sends)

**Why:** deliberate supervisor-update pacing. June update SENT 2026-06-13
(`docs/email_drZhang_june.md`); the follow-up reporting the completed 512 rerun +
revised headline is drafted and its deck is built — held for a July send.

**Ready-to-paste:**
- To: `jiehuang.zhang@ntu.edu.sg`
- Subject: `FYP July Progress Update: full 512-token rerun, revised headline (CCDS25-1136)`
- Body: `docs/email_drZhang_july.md` (gitignored; re-based to 512-primary, user's framing)
- Attach: `docs/fyp_status_2026-07.html` **or** `_v2` (both refreshed to 512; pick the design). If NTU mail blocks the `.html`, zip or export to PDF.

**Guardrails:** don't cross the decks (July email ↔ `fyp_status_2026-07.html`; June ↔ archived `fyp_status_2026-06-13.html`). After sending: PROJECT_LOG §4 row + tick T1 + mark the email file header SENT.

---

## [T30b] Second annotator for the human gold set (OPTIONAL strengthening)

T30 is **DONE** (2026-07-09, n=200, single annotator): classifier κ 0.59 vs regex
κ 0.11 against human labels, folded into report §6.12 Result 5 + thesis §6.1,
claim-locked (`results_512/analysis/human_validation.json`). See PROJECT_LOG §4
(2026-07-09 22:40). The report/thesis disclose the single-annotator caveat honestly.

**Optional follow-up:** a second annotator on a ≥40-item overlap slice for an
inter-rater κ + adjudication, lifting the result from single-annotator (moderate κ)
toward a higher-confidence gold standard. Tooling ready: `python
scripts/human_label_audit.py --make-html` (defaults to `--results-dir results_512`);
then `--apply-labels` and re-fold. Not blocking submission.

---

## [T3] Run `MyTCinfo` on TC1 (quick, optional)

Confirm the actual storage quota (report Table 5.1 assumes ~300 GB). Effectively
confirmed by the 25.5 GB prefetch; still worth the exact number.

---

## Not blocking submission (decisions / future work — reference only)

- **LICENSE decision (user).** Repo is currently all-rights-reserved (blocks reuse).
  Recommend MIT/Apache-2.0 for the code, but confirm against NTU FYP IP policy first
  (FYP IP may be co-owned). Then set `license` in `pyproject.toml` + `CITATION.cff`.
  See `docs/REPRODUCIBILITY.md §6`.
- **Optional viva-tightening disclosures** (already partly in §7.5/Ch8): XSTest-ΔOR
  construct-validity note; "classifier adopted without human-label validation" (T30
  closes this); 128-token ASR-floor note.
- **Ch9 future work** (not blocking): replicate the §6.15 Llama-INT8 effect across
  more models/seeds (rests on ~8–9 prompts); an attack-mediated arm (GCG/PAIR subset);
  a different quant family (GPTQ/AWQ/GGUF); Arditi activation-direction probe.
- **Housekeeping:** stale merged branches prunable (`t26-add-mistral-phi-pairs`,
  `mechanism-refusal-margin`, `int8-precision-point`, `harden/latent-six-audit`).
