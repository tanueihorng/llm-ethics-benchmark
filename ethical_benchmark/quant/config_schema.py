"""Pydantic schema for quantization-focused benchmark configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


SUPPORTED_BENCHMARKS = {"harmbench", "xstest", "mmlu"}

# SLURM accepts MM:SS, HH:MM:SS, or DD-HH:MM:SS. Bare integers (e.g. "360") are
# ambiguous across sites and have been seen to silently truncate; require an
# explicit colon-form to avoid the footgun.
_SLURM_TIME_RE = re.compile(r"^(\d+-)?\d+:\d+(:\d+)?$")


class DecodingSchema(BaseModel):
    """Schema for global decoding settings.

    Args:
        max_new_tokens: Maximum generated token count.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        repetition_penalty: Repetition penalty factor.
        max_input_tokens: Input truncation length.
        use_chat_template: Whether to apply tokenizer chat templates.
    """

    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    max_input_tokens: int = Field(default=1024, ge=1)
    use_chat_template: bool = Field(default=True)


class ModelEntry(BaseModel):
    """Schema for one model in the quantization matrix."""

    family: str = Field(..., min_length=1)
    size_b: float = Field(..., gt=0)
    quantized: bool
    pair_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    benchmarks: List[str] = Field(default_factory=list)
    trust_remote_code: bool = Field(default=False)
    dtype: str = Field(default="auto")
    revision: Optional[str] = Field(default=None)

    @field_validator("benchmarks")
    @classmethod
    def _validate_benchmarks(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Each model must specify at least one benchmark.")
        normalized = [item.strip().lower() for item in value]
        invalid = [item for item in normalized if item not in SUPPORTED_BENCHMARKS]
        if invalid:
            raise ValueError(f"Unsupported benchmark names: {invalid}")
        return normalized

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str) -> str:
        allowed = {"auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"dtype must be one of {sorted(allowed)}, got '{value}'.")
        return normalized


class BenchmarkEntry(BaseModel):
    """Schema for benchmark-level settings."""

    dataset_name: str = Field(..., min_length=1)
    split: str = Field(..., min_length=1)
    max_samples: int = Field(default=300, ge=1)
    batch_size: int = Field(default=4, ge=1)
    subjects: Optional[List[str]] = None
    refusal_patterns: Optional[List[str]] = None
    benign_only: Optional[bool] = None


class SlurmConfig(BaseModel):
    """Schema for configurable SLURM defaults."""

    partition: str = Field(default="gpu")
    qos: str = Field(default="normal")
    account: Optional[str] = Field(default=None)
    gpus: int = Field(default=1, ge=0)
    cpus_per_task: int = Field(default=4, ge=1)
    mem: str = Field(default="32G")
    time: str = Field(default="04:00:00")
    log_dir: str = Field(default="results/slurm_logs")

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        if not _SLURM_TIME_RE.match(value.strip()):
            raise ValueError(
                f"slurm.time must be MM:SS, HH:MM:SS, or DD-HH:MM:SS (got {value!r}). "
                "Bare-minute integers like '360' are ambiguous on SLURM."
            )
        return value.strip()
    work_dir: Optional[str] = Field(
        default=None,
        description="Optional working directory to cd into before running the job command.",
    )
    setup_commands: List[str] = Field(
        default_factory=list,
        description="Optional shell commands to bootstrap the cluster environment before execution.",
    )


class QuantizationConfig(BaseModel):
    """Top-level schema for quantization study configuration."""

    study_name: str = Field(default="quantization_safety_capability_study")
    models: Dict[str, ModelEntry] = Field(..., min_length=1)
    benchmarks: Dict[str, BenchmarkEntry] = Field(..., min_length=1)
    decoding: DecodingSchema = Field(default_factory=DecodingSchema)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

    @field_validator("benchmarks")
    @classmethod
    def _validate_benchmark_keys(cls, value: Dict[str, BenchmarkEntry]) -> Dict[str, BenchmarkEntry]:
        normalized = {key.strip().lower(): val for key, val in value.items()}
        invalid = [key for key in normalized if key not in SUPPORTED_BENCHMARKS]
        if invalid:
            raise ValueError(f"Unsupported benchmark config keys: {invalid}")
        return normalized

    @model_validator(mode="after")
    def _cross_validate(self) -> "QuantizationConfig":
        benchmark_names = set(self.benchmarks.keys())
        pair_map: Dict[str, List[ModelEntry]] = {}

        for alias, entry in self.models.items():
            model_benchmarks = set(entry.benchmarks)
            missing = sorted(model_benchmarks - benchmark_names)
            if missing:
                raise ValueError(
                    f"Model '{alias}' references benchmarks not defined in top-level config: {missing}"
                )
            pair_map.setdefault(entry.pair_id, []).append(entry)

        for pair_id, members in pair_map.items():
            quantized_count = sum(int(member.quantized) for member in members)
            baseline_count = len(members) - quantized_count
            if quantized_count < 1 or baseline_count < 1:
                raise ValueError(
                    f"pair_id '{pair_id}' must include at least one baseline and one quantized model."
                )

        return self


def load_quant_config(path: str | Path) -> QuantizationConfig:
    """Loads and validates quantization configuration from YAML.

    Args:
        path: YAML file path.

    Returns:
        Validated quantization configuration object.

    Side Effects:
        Reads YAML from filesystem.

    Raises:
        ValidationError: If configuration is invalid.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValidationError.from_exception_data(
            "QuantizationConfig",
            [{"type": "value_error", "loc": ("root",), "msg": "Config must be a mapping.", "input": raw}],
        )

    return QuantizationConfig.model_validate(raw)
