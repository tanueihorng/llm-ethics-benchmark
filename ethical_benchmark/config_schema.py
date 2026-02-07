"""Pydantic models for validating YAML configuration before pipeline execution.

Validates the full configuration schema so errors surface early, before
model downloading or dataset fetching begins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ModelEntry(BaseModel):
    """Schema for a single model registry entry."""

    hf_id: str = Field(..., min_length=1, description="Hugging Face model identifier.")
    trust_remote_code: bool = Field(default=False)
    dtype: str = Field(default="auto")
    revision: Optional[str] = Field(default=None)

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str) -> str:
        allowed = {"auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
        if value.strip().lower() not in allowed:
            raise ValueError(f"dtype must be one of {sorted(allowed)}, got '{value}'.")
        return value


class DecodingSchema(BaseModel):
    """Schema for global decoding parameters."""

    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    max_input_tokens: int = Field(default=1024, ge=1)
    use_chat_template: bool = Field(default=True)


class ToxicityEvaluationSchema(BaseModel):
    """Schema for toxicity evaluation settings."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classifier_type: str = Field(default="hf_classifier")
    classifier_model_name: str = Field(default="unitary/toxic-bert")
    classifier_batch_size: int = Field(default=16, ge=1)
    bootstrap_resamples: int = Field(default=1000, ge=1)

    @field_validator("classifier_type")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        allowed = {"hf_classifier", "detoxify"}
        if value.strip().lower() not in allowed:
            raise ValueError(f"classifier_type must be one of {sorted(allowed)}, got '{value}'.")
        return value


class BiasEvaluationSchema(BaseModel):
    """Schema for bias evaluation settings."""

    answer_letter_pattern: str = Field(default=r"\b([A-Z])\b")
    bootstrap_resamples: int = Field(default=1000, ge=1)


class FactualityEvaluationSchema(BaseModel):
    """Schema for factuality evaluation settings."""

    enable_llm_judge: bool = Field(default=False)
    judge_scale_min: int = Field(default=1, ge=0)
    judge_scale_max: int = Field(default=5, ge=1)

    @field_validator("judge_scale_max")
    @classmethod
    def _scale_ordering(cls, value: int, info: Any) -> int:
        min_val = info.data.get("judge_scale_min", 1)
        if value <= min_val:
            raise ValueError(
                f"judge_scale_max ({value}) must be greater than judge_scale_min ({min_val})."
            )
        return value


class TaskSchema(BaseModel):
    """Schema for a single task configuration block."""

    dataset_name: str = Field(..., min_length=1)
    split: str = Field(..., min_length=1)
    config_name: Optional[str] = Field(default=None)
    max_samples: Optional[int] = Field(default=None, ge=1)
    batch_size: int = Field(default=4, ge=1)
    evaluation: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Top-level schema for the benchmark YAML configuration file."""

    models: Dict[str, ModelEntry] = Field(..., min_length=1)
    decoding: DecodingSchema = Field(default_factory=DecodingSchema)
    tasks: Dict[str, TaskSchema] = Field(..., min_length=1)


def validate_config(raw_config: Dict[str, Any]) -> BenchmarkConfig:
    """Validates a raw configuration dictionary against the schema.

    Args:
        raw_config: Parsed YAML configuration.

    Returns:
        Validated ``BenchmarkConfig`` instance.

    Raises:
        pydantic.ValidationError: If configuration violates schema constraints.
    """

    return BenchmarkConfig.model_validate(raw_config)
