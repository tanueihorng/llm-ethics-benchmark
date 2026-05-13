# Datasets

## 1. HarmBench
Purpose: evaluate harmful compliance under unsafe prompts.

Usage in this framework:
- benchmark key: `harmbench`
- output focus: harmful compliance vs safe refusal

## 2. XSTest
Purpose: evaluate over-refusal on benign prompts.

Usage in this framework:
- benchmark key: `xstest`
- scoring focus: refusal behavior on benign requests

## 3. MMLU (Subset)
Purpose: evaluate broad capability for safety-capability interpretation.

Usage in this framework:
- benchmark key: `mmlu`
- fixed subject subset for reproducible and tractable runs

## 4. Sampling Strategy
- deterministic shuffle with fixed seed
- benchmark-specific `max_samples` caps
- consistent sampling policy across baseline and 4-bit within each pair

## 5. Licensing and Access Notes
Dataset licenses and access requirements vary by source and version. Users should verify current usage terms before publication.

## 6. Ethical Considerations
- Harmful prompts may contain unsafe content and require controlled handling.
- Refusal and safety labels are operational measurements, not absolute ethical truths.
- Capability benchmarks can be culturally and educationally biased.
