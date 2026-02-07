"""Shared constants for the ethical benchmarking framework.

Centralizes magic numbers and default values to improve maintainability
and ensure consistency across modules.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Model / Generation Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_NEW_TOKENS: int = 128
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_TOP_P: float = 1.0
DEFAULT_REPETITION_PENALTY: float = 1.0
DEFAULT_MAX_INPUT_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Batch Sizes
# ---------------------------------------------------------------------------
DEFAULT_GENERATION_BATCH_SIZE: int = 4
DEFAULT_CLASSIFIER_BATCH_SIZE: int = 16

# ---------------------------------------------------------------------------
# Toxicity Evaluation
# ---------------------------------------------------------------------------
DEFAULT_TOXICITY_THRESHOLD: float = 0.5
DEFAULT_TOXICITY_CLASSIFIER: str = "unitary/toxic-bert"
DEFAULT_TOXICITY_BACKEND: str = "hf_classifier"

# ---------------------------------------------------------------------------
# Bootstrap / Statistics
# ---------------------------------------------------------------------------
DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000
DEFAULT_CONFIDENCE_LEVEL: float = 0.95

# ---------------------------------------------------------------------------
# Dataset Defaults
# ---------------------------------------------------------------------------
DEFAULT_TOXICITY_DATASET: str = "allenai/real-toxicity-prompts"
DEFAULT_BIAS_DATASET: str = "heegyu/bbq"
DEFAULT_FACTUALITY_DATASET: str = "truthful_qa"
DEFAULT_FACTUALITY_CONFIG: str = "multiple_choice"

# ---------------------------------------------------------------------------
# Task Sample Limits
# ---------------------------------------------------------------------------
DEFAULT_TOXICITY_MAX_SAMPLES: int = 500
DEFAULT_BIAS_MAX_SAMPLES: int = 500
DEFAULT_FACTUALITY_MAX_SAMPLES: int = 300

# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_SCALE_MIN: int = 1
DEFAULT_JUDGE_SCALE_MAX: int = 5

# ---------------------------------------------------------------------------
# Output File Formats
# ---------------------------------------------------------------------------
JSONL_EXTENSION: str = ".jsonl"
JSON_EXTENSION: str = ".json"
CSV_EXTENSION: str = ".csv"
