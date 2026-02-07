# Limitations

## 1. Dataset-Related Bias
All benchmark datasets are socially constructed artifacts. Labels and categories may reflect annotator assumptions, linguistic priors, and cultural context. Consequently, metric values should be interpreted as comparative indicators rather than absolute ethical judgments.

## 2. Toxicity Classifier Dependence
Toxicity scores rely on proxy classifiers (e.g., Detoxify or transformer classifiers). These classifiers may:

- under-detect contextual harm
- over-penalize identity terms
- transfer biases from training data

Therefore, toxicity metrics should be accompanied by qualitative inspection of raw generations.

## 3. Output Parsing Fragility
Bias and factuality tasks require extracting option letters from generated responses. Although constrained prompting is used, verbose or malformed outputs can produce parsing errors, affecting measured accuracy.

## 4. Subjective Evaluation Risk
Optional LLM-as-judge scoring introduces model-dependent subjectivity. Judge outputs are sensitive to prompt wording and calibration drift, and should not be conflated with objective benchmark metrics.

## 5. Model Size and Hardware Constraints
The framework targets sub-10B models to remain feasible on CPU and single-GPU hardware. This improves accessibility but excludes larger models that may exhibit different ethical behavior patterns.

## 6. Benchmark Scope
The framework evaluates three dimensions only:

- toxicity
- social bias
- factuality

Other important dimensions (e.g., privacy leakage, jailbreak resistance, legal compliance, multilingual harms) are out of scope in the current implementation.
