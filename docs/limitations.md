# Limitations

## 1. Dataset and Annotation Limits
Benchmark datasets represent specific annotation regimes and may not generalize to all real-world safety contexts.

## 2. Refusal Parsing Approximation
Over-refusal scoring uses deterministic refusal-pattern parsing. This improves reproducibility but may under-capture nuanced refusals.

## 3. Capability Proxy Limits
MMLU subset is a practical capability proxy, not a complete competence measure.

## 4. Quantization Coverage
The study evaluates publicly released 4-bit checkpoints and does not isolate differences from training, publishing, or conversion pipelines.

## 5. Text-only Scope
Evaluation is restricted to text-only interactions. Multimodal safety and capability effects are out of scope.

## 6. Infrastructure Constraints
Single-GPU and batch settings chosen for feasibility can influence throughput and may limit large-scale replication speed.
