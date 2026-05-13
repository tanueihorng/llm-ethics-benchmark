"""Shared constants for quantization safety-capability benchmarking."""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Generation Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_NEW_TOKENS: int = 128
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_TOP_P: float = 1.0
DEFAULT_REPETITION_PENALTY: float = 1.0
DEFAULT_MAX_INPUT_TOKENS: int = 1024
DEFAULT_GENERATION_BATCH_SIZE: int = 4

# ---------------------------------------------------------------------------
# Quantization Study Benchmarks
# ---------------------------------------------------------------------------
BENCHMARK_HARMBENCH: str = "harmbench"
BENCHMARK_XSTEST: str = "xstest"
BENCHMARK_MMLU: str = "mmlu"
SUPPORTED_BENCHMARKS = [BENCHMARK_HARMBENCH, BENCHMARK_XSTEST, BENCHMARK_MMLU]

# ---------------------------------------------------------------------------
# Default Dataset IDs
# ---------------------------------------------------------------------------
DEFAULT_HARMBENCH_DATASET: str = "walledai/HarmBench"
DEFAULT_XSTEST_DATASET: str = "allenai/xstest-response"
DEFAULT_MMLU_DATASET: str = "cais/mmlu"

# ---------------------------------------------------------------------------
# Output File Names
# ---------------------------------------------------------------------------
DEFAULT_RAW_FILENAME: str = "raw.jsonl"
DEFAULT_SUMMARY_FILENAME: str = "summary.json"
DEFAULT_ANALYSIS_DIRNAME: str = "analysis"
