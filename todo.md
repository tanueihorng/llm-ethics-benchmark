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

## ⚠️ [security] ROTATE the OpenAI key

The user pasted an OpenAI API key into chat (twice, earlier sessions). It is **not
in the repo** (redaction scan clean; secret-pattern grep of the pushed tree clean),
but a key exposed in a chat log should be rotated as hygiene. One-off user action:
revoke + reissue at platform.openai.com, update `~/.zshrc` (`OPENAI_API_KEY`).

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

## [T30] Human-annotation gold set (READY — user labels)

**Why:** closes the one construct-validity gap the report discloses — the HarmBench
classifier is validated against the regex + a 2nd LLM judge, but not against human
labels. A small human-labelled subset turns "classifier vs regex disagree" into
"classifier matches human ground truth."

**Next steps:**
1. Open `human_labels/annotate.html` (200 rows drawn from the primary `results_512`
   tree: 120 judge-vs-proxy disagreement + 80 agreement, all 10 NF4 aliases).
2. Label each 0/1 → Finish & export → Copy results → paste back.
3. `python scripts/human_label_audit.py --apply-labels ... --score`
   → `results_512/analysis/human_validation.json` → fold classifier-vs-human +
   regex-vs-human agreement into report §6.12 / §7 + `make report`.

(Regenerate the sheet if needed: `python scripts/human_label_audit.py --make-sheet --make-html` — defaults to `--results-dir results_512`.)

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
