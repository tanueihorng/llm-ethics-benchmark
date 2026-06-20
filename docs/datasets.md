# Datasets

## 1. HarmBench
Purpose: evaluate harmful compliance under unsafe prompts.

Usage in this framework:
- benchmark key: `harmbench`
- source: `walledai/HarmBench`
- config: `standard`
- access: Hugging Face gated dataset; accept the dataset terms before pre-cache
- output focus: harmful compliance vs safe refusal

## 2. XSTest
Purpose: evaluate over-refusal on benign prompts.

Usage in this framework:
- benchmark key: `xstest`
- source: canonical XSTest v2 CSV from `paul-rottger/xstest`
- local file: `data/xstest_v2_prompts.csv`
- sample pool: 450 prompts total; `benign_only=true` evaluates the 250 safe prompts
- scoring focus: refusal behavior on benign requests

## 3. MMLU (Subset)
Purpose: evaluate broad capability for safety-capability interpretation (the primary capability anchor).

Usage in this framework:
- benchmark key: `mmlu`
- source: `cais/mmlu`
- fixed six-subject subset (300 questions) for reproducible and tractable runs

## 4. ARC-Challenge
Purpose: a structurally different second capability benchmark that corroborates the MMLU anchor (convergent validity).

Usage in this framework:
- benchmark key: `arc`
- source: `allenai/ai2_arc`, config `ARC-Challenge`
- ~1,172 reasoning-oriented science questions; identical exact-match scoring to MMLU
- interpretation labels remain MMLU-anchored, with ARC reported as a corroborating capability axis

## 5. Sampling Strategy
- deterministic shuffle with fixed seed
- benchmark-specific `max_samples` caps
- consistent sampling policy across baseline and 4-bit within each pair

## 6. Licensing and Access Notes
Dataset licenses and access requirements vary by source and version. Users should verify current usage terms before publication.

## 7. Ethical Considerations
- Harmful prompts may contain unsafe content and require controlled handling.
- Refusal and safety labels are operational measurements, not absolute ethical truths.
- Capability benchmarks can be culturally and educationally biased.
