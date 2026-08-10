# CLAUDE_CODE_SESSION_ARCHIVE.md — every Claude Code session on this repo

**Written 2026-08-11 (UTC+8) by Claude (Opus 5), at the student's request, ahead of an account change that will wipe the live chat history.**

This file is a **narrative archive**, not a source of truth. `docs/PROJECT_LOG.md` remains authoritative for status, decisions (D-numbers), open tasks (T-numbers), and the changelog. What this file adds is the layer PROJECT_LOG deliberately does not carry: *what was asked, in what order, why, what was tried and abandoned, and what the conversation felt like.* Where the two disagree, PROJECT_LOG wins.

---

## 0. How this was built, and what it cannot tell you

**Source:** the raw Claude Code transcripts on this Mac, at

```
~/.claude/projects/-Users-tanueihorng-fyp-quant/
```

plus one worktree directory (`…-fyp-quant--claude-worktrees-goofy-haibt-585ee4/`). 43 `.jsonl` files, ≈642 MB.

**Method:** a script parsed every transcript and extracted, per session — the AI/custom title, first and last timestamp, git branch, model(s) used, tool-call histogram, every human-authored turn (tool results and sub-agent turns filtered out), and the last two assistant messages. Prompts were truncated to 3 000 characters for reading. The summaries below were then written from that digest.

**Honest limits — read these before trusting anything here:**

1. **I did not read all 642 MB.** I read the human side of every conversation and the closing assistant message of each. The middle of each session — the tool calls, the reasoning, the intermediate results — is summarised from titles, tool histograms and endings, not from full transcript reads.
2. **Prompt text was truncated at 3 000 characters**, so long pasted audits and terminal dumps are represented by their opening only.
3. **This is a summary of what was *said*, not proof of what is *true*.** Every technical claim below traces back to a conversation, not to a re-verification. To check any number, run `make verify-claims` (344 checks) or read the artifact — do not cite this file.
4. **Older sessions may already be missing.** `~/.claude/projects` rotates. The earliest transcript here starts **2026-06-13**, but the project began in **May 2026** (first TC1 matrix run 2026-05-27, per PROJECT_LOG). Everything before 13 June is already gone from the local transcript store and is recoverable only through PROJECT_LOG and git history.

> **⚠ The backup, not this file, is what actually preserves the record.**
> The raw transcripts are **ordinary local files**. Deleting the account wipes what the app shows you; it does not, by itself, delete `~/.claude/projects/`. This summary preserves perhaps 2% of what is in them.
>
> **DONE 2026-08-11 01:17 (UTC+8).** The archive exists at:
>
> ```
> ~/claude_transcripts_fyp_quant_20260811.tar.gz          289 MB, mode 600
> ~/claude_transcripts_fyp_quant_20260811.tar.gz.sha256   a46081028b2898da…f80489
> ```
>
> Contents and verification: **4 475 files** (both project directories, including every per-session `subagents/` and `tasks/` directory), of which **1 908 are `.jsonl`** — the 43 top-level session transcripts plus the sub-agent and workflow journals, which is where most of the audit work actually lives. Counts were read back **out of the finished archive** rather than taken from the source tree, `gzip -t` passes, and three transcripts (largest, smallest, and session 14) were extracted and `cmp`-ed byte-for-byte against the originals.
>
> To rebuild it later, or to back up the other 19 project directories (≈312 MB more, work projects included):
>
> ```bash
> tar czf ~/claude_transcripts_20260811.tar.gz -C ~/.claude/projects .
> ```
>
> **Two cautions on the archive.** It contains unredacted HarmBench prompts and model outputs, so it belongs on an encrypted or private destination — never a shared folder. And it contains **an OpenAI API key pasted in plaintext**: **9 occurrences, all in session 14** (`4bf15b9f…`, 2026-06-30). Session 23 refers back to that key but does not repeat it, and the key appears **nowhere in the repository itself**. Revocation was stated in-session and never confirmed — treat it as live until you have confirmed otherwise on the OpenAI dashboard.

---

## 1. The whole thing in numbers

| Measure | Value |
|---|---|
| Sessions | **43** (42 in the main project dir, 1 in a git worktree) |
| Transcript volume | ≈**642 MB** of JSONL |
| Date span | **2026-06-13 → 2026-08-10** (59 days) |
| Days with activity | **40 of 59** |
| Assistant messages | **21 981** |
| Human turn records | **1 024** — of which **721 authored** and **303 harness-inserted** (skill loads, task notifications, `/model`, `/compact`, interrupt markers) |
| Median authored prompt | **114 characters** (25th percentile: 39) |
| Tool calls (main loop) | **10 005** |
| Output tokens | **35.7 M** (main loop; excludes sub-agent and workflow tokens, which are the larger share) |
| Cache reads | **7.59 B** |
| Models | Opus 4.8 (13 052 msgs) · Fable 5 (5 306) · Opus 5 (3 554) |
| Busiest single day | **2026-07-11** — 1 678 assistant messages |

**Tool profile:** Bash 4 837 · Edit 2 328 · Read 945 · browser-preview tools 825 · Write 209 · Workflow 82 · Agent 57.

Bash-and-Edit dominance is the signature of this project: the work was overwhelmingly *verify-then-patch* against committed artifacts, not greenfield writing.

> For measured resource figures — TC1 GPU hours, gpt-4o spend, token economics — see **`docs/PROJECT_METRICS.md`**, which was itself produced in session 35 below and carries per-figure MEASURED / DERIVED / ESTIMATED confidence tags. This file does not duplicate it.

---

## 2. Session index

Chronological by first message. "P" = authored + harness prompt records; "A" = assistant messages.

| # | Dates | Session ID (short) | Title | P | A |
|---|---|---|---|---|---|
| 1 | 06-13 → 06-14 | `40309c23` | Documentation and state verification | 54 | 821 |
| 2 | 06-13 | `641061e3` | fyp diagram | 3 | 114 |
| 3 | 06-13 → 06-14 | `531b5e9e` | Code quality audit | 30 | 460 |
| 4 | 06-17 → 06-19 | `d50b7a6c` | Continue running session (INT8) | 43 | 772 |
| 5 | 06-19 → 06-23 | `1f927a90` | Interactive FYP timeline showcase | 13 | 847 |
| 6 | 06-20 | `de604af4` | HTML pages for AI-agent documentation | 10 | 339 |
| 7 | 06-20 | `70242e9d` | Install humanizer skill | 13 | 96 |
| 8 | 06-26 | `0844ef7d` | Research-grade repository audit | 10 | 424 |
| 9 | 06-26 | `e2f218d6` | Agentic-AI workflow report | 7 | 280 |
| 10 | 06-26 → 06-28 | `76733ef8` | Professor email + submission folder | 32 | 552 |
| 11 | 06-26 | `65469807` | McNemar tests for precision sweep *(worktree branch)* | 2 | 173 |
| 12 | 06-28 | `ec7ed3e0` | Project-deviation analysis + dashboard | 10 | 294 |
| 13 | 06-28 | `168ed3a7` | Judge model-ID bug + README redesign | 7 | 152 |
| 14 | 06-28 → 07-03 | `4bf15b9f` | **T30 → the 512-token discovery → D41 re-base** | 143 | 2 157 |
| 15 | 06-29 → 06-30 | `b1765bd2` | Report identity and stale metadata (D40) | 5 | 212 |
| 16 | 07-03 → 07-07 | `e9618ac9` | FYP showcase — 3D / Home Planet / Blue Marble | 29 | 1 121 |
| 17 | 07-03 | `6e57c69f` | Submission-readiness audit (Codex claims refuted) | 1 | 114 |
| 18 | 07-06 → 07-07 | `fbd3e61a` | Understand-Anything → Codebase Atlas | 18 | 191 |
| 19 | 07-07 → 07-09 | `df09358c` | Full repo audit + human labelling + Codex skill mirror | 72 | 1 092 |
| 20 | 07-08 | `49ebc286` | Project status (one question) | 1 | 11 |
| 21 | 07-09 → 07-18 | `a8e5b8ab` | Human-label findings on model behaviour | 2 | 18 |
| 22 | 07-10 → 07-11 | `0692b0b8` | **Fable 5 25-agent adversarial audit** | 83 | 2 010 |
| 23 | 07-11 → 07-14 | `9d0d6561` | "Don't be a yes-man" → T35–T40 | 61 | 1 441 |
| 24 | 07-12 → 07-13 | `af6bf29d` | T37/T39 TC1 execution + strengths/weaknesses | 63 | 784 |
| 25 | 07-14 → 07-15 | `01971a11` | Stale `precision_sweep.json` *(worktree)* | 22 | 457 |
| 26 | 07-15 → 07-16 | `31e3ff13` | Pre-submission audit, 4 rounds + code review | 31 | 735 |
| 27 | 07-16 → 07-19 | `87666a5d` | Verifier hardening → D51 → T41 | 53 | 1 319 |
| 28 | 07-16 | `f8cde9ec` | Stale `genlen_robustness.json` *(worktree)* | 1 | 63 |
| 29 | 07-18 | `05b6dc89` | "Is Fable 5 high enough?" | 1 | 2 |
| 30 | 07-19 | `ec279925` | Phase-C framing audit — no findings | 7 | 78 |
| 31 | 07-19 | `58db0052` | Citation-fitness sweep (31 papers, 83 sites) | 11 | 201 |
| 32 | 07-20 → 07-24 | `cbd02232` | **T44 full-sweep audit (8 phases) + emails** | 65 | 1 085 |
| 33 | 07-24 | `c649d459` | "My worst habit" — candid cross-session read | 1 | 17 |
| 34 | 07-24 | `2711dfc0` | June/July/August email staging | 7 | 118 |
| 35 | 07-24 | `a7aea457` | → `docs/PROJECT_METRICS.md` | 8 | 149 |
| 36 | 07-25 | `f22cae72` | Which report is which + architecture diagrams | 9 | 187 |
| 37 | 07-25 → 07-28 | `5b90599c` | **15 past-FYP papers → `rewritten_fyp_report_v1`** | 43 | 1 515 |
| 38 | 07-25 | `5e557384` | "Am I AI-native?" | 2 | 43 |
| 39 | 07-28 → 07-29 | `a392df59` | Handoff pickup — three fixes | 4 | 109 |
| 40 | 08-01 → 08-10 | `0726bd4b` | T47, rewrite selected, papers corpus excluded | 16 | 303 |
| 41 | 08-10 | `8e43a48a` | Interim deadline → repo made **private** | 14 | 292 |
| 42 | 08-10 | `6b70e056` | **Last-mile durability before access lapses** | 16 | 794 |
| 43 | 08-10 | `2f61cd73` | This archive | 1 | 20 |

---

## 3. The seven eras

### Era 1 — Finishing the first experiment set (13–23 Jun)

**Sessions 1–7.** The project already had results; this era added arms and hardened the code underneath them.

- **[1] `40309c23` — Documentation and state verification.** Opened with the house rule (`read HANDOFF.md then PROJECT_LOG.md, verify with git status and pytest`). Ran the **T18 multi-seed sensitivity** jobs on TC1 (jobs 61078–61084), then built and ran **ARC-Challenge** as a second capability anchor (jobs 61086–61099; the first ARC job FAILED at 6 s and needed a `git pull` on the cluster first). Set up `gh` CLI. Decided to add a **second judge** and chose the **API route (gpt-4o)** over LlamaGuard for T22 — the student had a ChatGPT Plus account, topped up $10, and created the key themselves. Closed with a report audit the student pushed for after spotting one stale section: *"only recently you found this issue right. I'm worried there are more."* That instinct was correct — the audit found §6.2's paragraphs carried the same v2-as-primary bug as §6.4, a whole region the D16 judge-primary rewrite had missed.
- **[2] `641061e3` — fyp diagram.** Installed an HTML-diagram skill and built an architecture diagram. Notable for the verification method: text overflow was measured with `getBBox` in a headless browser rather than eyeballed — 7 subtitle lines were spilling, one by 73 px.
- **[3] `531b5e9e` — Code quality audit.** *"which of your findings is significant and is affecting my experiment/fyp result accuracy… or is it only a code quality issue"* — the framing that recurs all project. Six latent correctness bugs found and fixed (degenerate κ returning a fake 1.0, MMLU parser, taxonomy, loader guard) **before** the multi-seed run, which is why they never contaminated results. Then a long research-scoping conversation: is a few hundred prompts enough for research grade? How does this compare to Kharinaev et al. 2025? Should the extension be more models or more quantization methods? Landed on **more model families** — Phi-4-mini and Mistral-7B-Instruct-v0.3 — after checking both HuggingFace pages directly for gating, licence, and transformers-version compatibility. (Gemma-3 was considered and dropped; it is gated.)
- **[4] `d50b7a6c` — Continue running session.** Ran the whole **INT8 precision point** across five models on TC1. Two important things surfaced here. First, a process complaint that shaped everything after: *"why you never give command to check and verify output and following steps and commands?"* Second, a real latent bug — `summary.json` holds **v1** regex scores for the original 10 models while v2 lives in a separate sidecar, and the precision sweep was reading the wrong one for old models. The student caught the *category* of the problem themselves: *"if regex method is flawed, why compute and run it, and include the flawed result with others? do we even need to include it in the report?"* The session also produced the first **standalone thesis** document with IEEE citations, verified by actually browsing the sources after the student said *"Are you sure these sources are valid? Can you use browser to actually browse it? Don't hallucinate the sources."*
- **[5] `1f927a90` — Interactive FYP timeline.** Modelled on an ESPN feature page. 15 slides, full-screen scroll-snap, dark/light toggle, 3D transitions. This is where the showcase-deck thread starts: high-craft, iterated hard, reviewed adversarially by a sub-agent workflow.
- **[6] `de604af4` — AI-agent documentation deck.** *"it's meant for presentation so be thoughtful about the design. It must be very very good. Even better than the FYP showcase."* Installed **hallmark** here. Diagnosed a real UX bug: trackpad momentum was skipping slides because v2's transition lock (480 ms) was shorter than the inertia tail; fixed by treating one gesture as exactly one slide.
- **[7] `70242e9d` — Skills installation.** humanizer, stop-slop (compared, then one removed), timesfm-forecasting, frontend-slides, html-diagram, hallmark. Includes an honest moment worth keeping: asked where `hallmark` came from, Claude checked filesystem timestamps and stated plainly that it had **not** created it.

---

### Era 2 — Publication-grade push, and the first outward-facing artifacts (26–28 Jun)

**Sessions 8–13.** The pivot from "does the experiment work" to "is this defensible and presentable."

- **[8] `0844ef7d` — Research-grade repository audit.** *"Make sure everything is research-paper grade, publishable to magazine level."* Produced the **v2** documents (figures, IEEE citations, FDR/power, reframed lead) and then **v3** (v2 + a de-AI prose pass via `/humanizer`). Established a convention that held for the rest of the project after the student interrupted with *"u should keep old version, dont directly edit on old files so that i can compare"* — **every major revision became a new numbered file, never an in-place edit.**
- **[9] `e2f218d6` — Agentic-AI workflow report.** A separate deliverable about *how* agentic tooling was used, with the explicit framing constraint: *"Don't frame it as everything done by AI, but detail how I use this program to accelerate my workflow."* Shipped as both HTML and a Word document with six embedded figures (the harness control plane, memory layers, test-suite growth 186 → 306, changelog authorship Claude 83 / Codex 16).
- **[10] `76733ef8` — Professor email + submission folder.** Retrieved the stale draft, humanized it, rewrote it around the actual June state. Built the NTU submission folder, an **Overleaf-portable LaTeX** version, and ran an internet-grounding workflow over every thesis reference to catch hallucinated citations. `fyp-submission/` was removed from GitHub on the student's instruction. This session also received a finding from a *parallel Codex session* and had to reconcile with a second Claude worktree — the first real dual-agent collision.
- **[11] `65469807` — McNemar for the precision sweep.** *(Ran on branch `claude/wizardly-bouman-21c318`.)* The thesis claimed Llama-3.2-3B's INT8 ASR rise was "significant under both judges and McNemar (p=0.022 / 0.008)" but **no committed artifact computed a McNemar test at all** — the only source was a hand-derived footnote. Claude reconstructed it from the judge sidecars, confirmed the claim was *true*, and then made it *traceable* by adding `harmbench_asr_int8_mcnemar` to `precision_sweep.json` — verified purely additive, no pre-existing value changed. A clean example of the project's standard: a number in a thesis must trace to a committed file.
- **[12] `ec7ed3e0` — Deviation analysis + dashboard.** Compared the delivered project against the original allocated title. The student's real worry: *"whether my project is significant and brings real contribution, and also something that other people can reuse."* Answer took the form of a **Streamlit dashboard** — visualize results, add a model through a schema-validated form, launch runs. Then a v2 status deck in a liquid-glass aesthetic, with the original left byte-untouched for comparison.
- **[13] `168ed3a7` — Judge model-ID bug + README.** `judge_model_id` recorded `null` for api_judge summaries; patched and 15 summaries backfilled to `gpt-4o`. README redesigned purpose-first with live Mermaid diagrams and result figures. Pushed with all 8 gates green and 329 tests passing.

---

### Era 3 — The 512-token discovery (28 Jun – 3 Jul)

**Session 14 is the single most consequential conversation in the archive**, and the reason the project's headline number changed.

- **[14] `4bf15b9f` — T30 execution steps.** 143 prompts, 2 157 assistant messages, five days. It began as human labelling of the HarmBench gold set. The student built momentum immediately by rejecting the tooling — *"excel is very hard for me to annotate as the text is too long can you build a html for me to annotate instead"* — and then, while labelling, noticed something in the data:

  > *"why do the responses are half baked not full?"*
  > *"omg is this a risk to my experiment and study? do other academic researchers use this truncation also?"*

  The generation budget was **128 tokens**; HarmBench's own reference budget is **512**. Roughly 60% of generations were being cut off mid-answer. The student's own decision followed within hours — *"for harmbench wise, promote 512 to primary as official classifier recommends 512, I want to rerun everything, the full apparatus for 512"* — and the rest of the session executed it: 10 models × 4 benchmarks re-run on TC1 (jobs 61474–61557), the HarmBench classifier re-scored (job 61524, 3 000 classifications, 0 parse errors), gpt-4o second judge, INT8@512, multi-seed@512.

  **The result overturned the project's own headline.** The 128-era Qwen-1.7B **+0.055 ASR increase** — the finding the whole safety story had rested on — went to **zero** at the correct budget. It was a truncation artifact. The student found it themselves, mid-labelling, and chose to rerun rather than ship it.

  The back half of the session was the **D41 re-base**: making 512 genuinely canonical across the report (v5), thesis, README, decks, LaTeX and the claim-verification oracle. This is where the project's hardest process lesson was learned — three successive Codex audits kept finding stale 128-era claims that diff-scoped verification had missed, prompting the student's sharpest question of the project:

  > *"why keep having issue even after you told me u had checked and pass? was there some structural issue with agent?"*

  That question produced **D42** ("when a RESULT changes, sweep the claim surface"), now in CLAUDE.md.

- **[15] `b1765bd2` — Report identity and stale metadata.** Handled the Codex finding that the repo had *split report identity* — PROJECT_LOG and `make report` pointed at the June 14 docx while newer v3 existed. Produced **D40** ("sweep the instruction layer"). Ended with restraint worth noting: asked to build v4, Claude built it but deliberately left it **untracked**, keeping v3 canonical until the TC1 jobs finished.
- **[17] `6e57c69f` — Submission-readiness audit.** A one-prompt session that pasted a Codex verdict claiming a dirty working tree and a failing log-discipline gate. Claude verified against primary artifacts and found **both P1 headline claims were wrong** — `git status` was clean, the named files were never modified, and what dirtied the tree appeared *during* the session from a concurrent sibling session. A few real content nits were buried inside the false alarm. This is the origin of a rule the project applies to *itself*: **verify audit findings against primary artifacts, not against the audit summary.**

---

### Era 4 — Showcases, understanding, and the human-label payoff (3–9 Jul)

- **[16] `e9618ac9` — FYP showcase, escalating.** Six escalating requests across four days: enhance → real 3D → *"each component is an island on its own"* → *"I want each island on an earth… if I zoom out, I can see stars"* → realistic continents, city lights, a ship that sails between islands with **Singapore as the final base** → **"Blue Marble — the Cinematic Cut"**, an engine rewrite with an HDR bloom pipeline, ACES filmic tonemapping, and 1 000+ real stars baked from the 33 MB HYG astronomy catalog. Every version was adversarially reviewed by a workflow before landing. The best debugging moment in the archive is here: *"something is still wrong the ship doesn't sail"* — the ship **was** sailing; the chase camera locked it dead-centre so it never moved on screen. Cinematography bug, not a mechanics bug.
- **[18] `fbd3e61a` — Understand-Anything → Codebase Atlas.** The tool rendered the project as a 429-node force-directed graph. Verdict: unreadable at that scale, and no tweak fixes a hairball. Replaced with a **readable document view of the same data** — research question, 12-step guided tour following the data flow, 14 layer cards, all 139 files.
- **[19] `df09358c` — Full repo audit + human labelling.** The audit half ran adversarial workflows over defensibility, citations and LLM-eval standards. The labelling half is where **T30 paid off**: the student finished 200 items and Claude compared them against both scorers.

  | Scorer vs the student's labels | Agreement | Cohen's κ | Over-calls | Misses |
  |---|---|---|---|---|
  | HarmBench classifier | 86.5% | **0.591** | 18 | 9 |
  | v2 regex | 46.5% | **0.110** | 101 | 6 |

  The regex over-called harm on **101 of 200** items. That single table is the empirical ground under D16 (classifier as primary scorer). The session also contains a genuine methodological argument the student won partly and lost partly — whether a *fake or broken* harmful recipe should be labelled 0 (the classifier has no "does it work?" rule; it does not). Finally, sensing the subscription ending, the student asked for **all 15 skills mirrored into `~/.codex/skills/`**.
- **[21] `a8e5b8ab` — Model behaviour from labelling.** Short but substantive: the disclaimer-then-comply pattern, and where the student's judgement and the classifier's split systematically in two opposite directions.

---

### Era 5 — The audit machine (10–19 Jul)

The densest stretch of the project. Five audits, four verification rounds, four new experimental arms.

- **[22] `0692b0b8` — Fable 5 25-agent adversarial audit.** *"IMPORTANT this is essentially a full project review as you are fable 5, feel free to use whatever method, tools to accomplish the goal, I don't want to restrict you with your reasoning."* A 25-agent workflow across code, results recompute, methods, reports and citations — run over three days through repeated context exhaustion, Monitor timeouts and model switches. Findings were remediated in P0/P1 then P2 tiers. Two things worth carrying forward: a **real scoring bug** was found (the refusal parser's unanchored `"unable to <verb>"` matched third-person text — *"the victim is unable to give consent"* — over-counting XSTest over-refusal on 11 prompts), and the nearest prior paper (**Hong et al., Decoding Compressed Trust, ICML 2024**) was not cited, which an examiner would flag first. The student's closing question — *"so did I close all the blockers and ready to submit without any error or mistakes"* — got the correct answer: the audit reviewed ~9% of files in depth; "all blockers closed" means *all this audit found*.
- **[23] `9d0d6561` — "Don't be a yes-man."** *"tell me honestly how my project compares with other papers… as my deadline is close, im very worried and this will affect greatly affect my grade."* The honest answer named a real weakness: **XSTest over-refusal was scored by the very regex the project had just proven unreliable on HarmBench.** That produced **T35** (gpt-4o XSTest judge as a sensitivity layer, D45), and then **T36–T39** — human gold set, LlamaGuard third judge, ARC/parser sensitivity, five-pair multi-seed. Notable process choice, entirely the student's: *"okay fable 5, plan all these in detail so that i can ask opus 4.8 to execute"* — Fable plans, Opus executes.
- **[24] `af6bf29d` — T37/T39 on TC1 + strengths and weaknesses.** Ran LlamaGuard (job 61844, 15 aliases, 0 parse errors, revision-pinned) and the five-seed judges. Between jobs, the most useful self-assessment conversation in the archive: strong points (statistical practice — paired bootstrap + McNemar exact + BH-FDR + explicit MDE beats several cited papers; and two genuine methodological findings), then eight weaknesses with feasibility ratings. The student's own footnote deserves preserving: *"tbh the 3 things caught, generation length knob is me myself discovered it after doing human label."* Also caught here: five modified sbatch files would have **re-scored and overwritten committed judge sidecars**, violating the zero-drift convention. Trimmed to the four new aliases before submission.
- **[25] `01971a11`, [28] `f8cde9ec` — worktree fix sessions.** Two narrow, well-executed jobs: a stale XSTest column in `precision_sweep.json` (v1 vs v2 sidecar preference), and a stale `capability_128_vs_512` sub-block in `genlen_robustness.json`. The second confirmed by repo-wide grep that **nothing consumes** the stale block before regenerating it.
- **[26] `31e3ff13` — Pre-submission audit, four rounds.** The student wrote the orchestration prompt themselves after asking how to structure it: Fable 5 as orchestrator-only, Opus as executor, mechanical sweeps as plain Bash. Rounds found 17 → 15 → 24 → 12 findings, **with P1s appearing in rounds 3 and 4**. The honest read given at the time still holds: *"It is not converging to zero, and I'd expect a fifth round to find a few more. What has been stable throughout is the science: no result, significance verdict, label, hedge, or conclusion has moved once in any round."* The session then ran an **execution-based code review** — 104 agents, every artifact script actually re-run; 10 of 11 reproduced byte-for-byte.
- **[27] `87666a5d` — Verifier hardening → D51 → T41.** Three rounds of Codex breaking a new semantic verifier with false-negative DOCX fixtures (`Mistral-7B MMLU (-0.090) is a BH-FDR survivor` slipping past an adjacency regex because parentheses terminated it), until both agents agreed the right acceptance standard is the verifier's **declared contract** — an enumerated drift detector, not a natural-language theorem prover. Then the XSTest human gold set (**T36**) landed and forced two hard calls:
  1. The regex detected **2 of 63** full refusals (κ = −0.006); gpt-4o detected 61 (κ = 0.485). Rather than swap the registered scorer after seeing the outcome, **D51** added a *validation-informed parallel family* alongside the pre-registered one, with the headline stated scorer-invariantly.
  2. The blindness exclusion had run **prompt-wide** instead of on the ≤10 pre-specified items — a genuine protocol deviation. **The clean "pre-registered" badge was withdrawn rather than brazened out** (T41), footprint ≤14/200, κ gap unchanged.
- **[30] `ec279925` — Phase-C framing audit.** Scoped deliberately narrow because five prior audits had covered the older surface. Verdict: **no findings.** The first clean audit of the project.
- **[31] `58db0052` — Citation fitness.** *"my biggest concern is hallucination in my report/sources/citation/content/fitness. The gate checks numbering consistency, not fitness."* Six Opus verifiers fetched all **31 unique papers** across **83 in-text citation sites**. Headline: **zero hallucinated citations.**

---

### Era 6 — T44, the metrics ledger, and learning to write like an FYP (20–29 Jul)

- **[32] `cbd02232` — T44 full-sweep audit, in eight phases.** Started as an email task and turned into the largest audit of the project after a discovery mid-conversation: **Benjamini–Hochberg was used throughout but never cited.** The student's reaction shaped the whole design — *"omg here we go and discover something that is risk before fixing. how to possibly detect and prevent this error from happening?"* and *"I would like to explore ways of why previous full audit didn't catch this."* The answer became **D54**: presence-audits enumerate outward from the bibliography and structurally cannot see a method named with no citation at all; every audit needs an absence-direction pass. Also here, a piece of project management that worked: *"i dont want the full audit plan run in one session, as it's token intensive — is it possible to do multiple days run, but no record is lost?"* → eight resumable phases, run across several days as limits reset. Result: 25 findings, **zero conclusion-touching**.
- **[33] `c649d459` — "My worst habit."** One prompt, asked candidly, answered candidly from evidence across a dozen sessions: **re-verifying finished work instead of shipping it, and deferring the one irreversible step.** Cited the 103-agent audit whose own verdict was *"No finding changes any reported number, verdict, label, or conclusion."*
- **[34] `2711dfc0` — Email staging.** Discovered the repo recorded the June email as sent 2026-06-13 when the real header said **26 June**. Every number in the July email was verified against artifacts before sending (60.3% truncation, Δ0.055 p=0.0266 at 128, → 0.000 p=1.0 at 512). The student set the voice rules explicitly: *"don't mention anything to do with agent helping me do stuff. Agent is assistant. Judgment is mine. I spotted issue. Agent is executor."*
- **[35] `a7aea457` — `docs/PROJECT_METRICS.md`.** *"Can you use data to justify the time and cost efficiency of the agentic harness? I need you to ground it, not just your opinion."* 41.1 GPU-hours measured, gpt-4o ≈ $13.60, 6.98 B agent tokens ≈ $8 960 API-equivalent, every figure tagged MEASURED / DERIVED / ESTIMATED, with counter-evidence included. Also: *"i want u to write into separate md file not project log, project log is to keep track."*
- **[37] `5b90599c` — 15 past NTU FYP papers → the rewrite.** The most decisive quality intervention after session 14. The student supplied a corpus of real submitted NTU FYP reports and asked Claude to learn *structure, table of contents, writing style, citation craft*, separating research reports from development reports. Then: rewrite into a **new** document, preserving every number exactly. Along the way the student made the sharpest self-diagnosis of the project — the report *"looks like an AI-written report,"* the literature review is undercited, *"there's no citation supporting any of the claims I've made in the introduction. This is a major risk."* Result: **`docs/rewritten_fyp_report_v1.docx`** — 57 pages, 64 references in strict first-use order, Chapter 2 grown from 3.7% to 20.6% of the body. Multiple adversarial passes then checked each *new* citation against what its source actually says, producing **D56**: a real, correctly formatted reference proves nothing about the sentence citing it. 26 claims were found stretching past their sources, **19 of them sitting in v5** — the report of record, under all 93 claim locks, after six prior audits.
- **[38] `5e557384` — "Am I AI-native?"** Answered from 110 transcripts across 19 projects, with the scope limit stated up front. Verdict: yes, but on one specific axis — *engineering the agent's environment rather than its prompts*. Includes a self-correction worth keeping: the "long prompts" cited in the first answer were mostly harness-inserted; the actual authored median is **109 characters**.
- **[36] `f22cae72`, [39] `a392df59`** — housekeeping with teeth: disambiguating six coexisting report files, and adding three hand-authored SVG architecture diagrams built under a deliberate constraint — **structural constants only, no result values**, because figures aren't covered by `verify_report_claims.py` and a κ printed in a figure would be an unlocked claim surface.

---

### Era 7 — Last mile (1–10 Aug)

- **[40] `0726bd4b` — T47 and the rewrite decision.** `rewritten_fyp_report_v1` selected as the submission candidate (D59/T52). One judgement call the student should know about: **`Examples of Past FYP Papers/` was excluded and gitignored** — 110 MB, 47 PDFs of other students' final reports with names in the filenames, plus copyrighted IEEE/ICML/NeurIPS papers, in a then-public repo.
- **[41] `8e43a48a` — Deadline, then privacy.** Confirmed the official CCDS dates (interim **24 Aug**, final **19 Oct**, amended **13 Nov**, oral **Dec**). Ran an additive edit pass over Chapters 2 and 6 to lift academic register while diff-protecting every number. Then: *"Wait. Is my report on GitHub?"* → 26 files untracked (24 docx + 2 NTU PDFs), `.gitignore` updated, and **the repository set to PRIVATE** — with the honest note that privacy is what actually withdraws the text; the gitignore is complementary.
- **[42] `6b70e056` — Before access lapses.** *"As I'm losing my Max Plan access soon, I need you to see what else you can do here."* Four tracks: **A** viva package (deck, timed speaker notes, artifact-grounded Q&A bank for December), **B** durability (Makefile targets, pinned deps, green CI, `make backup`), **C** submission runbook, **E** closing T46/T47. Track E's outcome is the best single illustration of a rule this project learned the hard way: both tickets closed **as superseded, not by doing the work they described** — every proposal had been diagnosed against v5/v6, whose structure the rewrite doesn't have. But re-measuring found something neither ticket described: Table 3.1 listed Mistral-7B at **7.0 B** directly above its own sentence saying the models span "1.7–**7.2** billion." Root cause: `params_b` was hand-typed in a display map while the config and all fifteen run summaries said 7.2. Fixed by **deriving it from the config** instead of retyping it.
- **[43] `2f61cd73`** — this file.

---

## 4. What lives only in the chat history

Things that shaped the project but appear nowhere in code, git history, or PROJECT_LOG.

**Working preferences, stated and enforced.**
- **One command per code block.** Said in session 4 (*"why you never give command to check and verify output and following steps"*) and again bluntly in session 14: *"That's why you should provide me with one command in one box only. Remember this, please."*
- **Never edit the old version.** Every revision is a new numbered file so versions can be compared side by side. Set in session 8, held to the end.
- **Push before giving TC1 instructions** — otherwise `git pull` on the cluster fetches nothing. Learned the hard way when the first ARC job failed in 6 seconds.
- **Agent is executor; judgement is the student's.** Explicit in session 34, and the reason the supervisor emails never mention agent assistance.
- **Fable plans, Opus executes.** Adopted in session 23 for token economy, reused for every large audit after.

**The intellectual through-line.** The same question recurs in almost every session, phrased differently: *does this affect my result, or is it only cosmetic?* Sessions 3, 4, 14, 22, 26, 32, 42 all turn on it. The project's answer stabilised into a habit — classify every finding by whether it touches a number, a verdict, a label, or a conclusion, and be willing to say "no" out loud.

**Three findings the student made personally, not the agent:**
1. The **128-token truncation** — noticed while hand-labelling, and it overturned the project's own headline (session 14).
2. The **v1/v2 scorer mismatch** across old and new models (session 4).
3. That **§6.4 was stale** after the judge-primary rewrite — which, when pushed on, revealed the entire §6.2 region had the same bug (session 1).

**Things that were tried and dropped:** GPTQ/AWQ and GGUF k-quant extensions (scoped out as future work); Gemma-3 (gated); a `stop-slop` skill (superseded by humanizer); drag-to-navigate in the showcase deck (reverted at the student's request); the Understand-Anything force graph (unreadable at 429 nodes, replaced with a document view).

**Open at the time of writing:** T30b, a second annotator for both human gold sets — the one genuinely open piece of research work. Interim report due **24 Aug 2026**.

---

## 5. If you are a future agent reading this

Do not treat this file as evidence. Read `docs/PROJECT_LOG.md` §1–§3 first, then `docs/HANDOFF.md`, then verify against artifacts. This file exists to tell you *why* things are the way they are — which decision came from which mistake, which audits were run and what each one could not see, and which conventions were bought with real pain. That context makes PROJECT_LOG's D-numbers legible. It does not replace them.
