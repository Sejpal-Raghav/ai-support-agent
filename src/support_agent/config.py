"""Strict, versioned configuration for the support-agent pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    """Reject unspecified configuration keys to keep decisions auditable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SeedSettings(StrictConfig):
    global_: int = Field(alias="global", ge=0)
    golden_sampling: int = Field(ge=0)
    bootstrap: int = Field(ge=0)


class CohortSettings(StrictConfig):
    threads: int = Field(gt=0)
    split_ratios: tuple[float, float, float]

    @model_validator(mode="after")
    def split_ratios_sum_to_one(self) -> CohortSettings:
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("cohort.split_ratios must sum to 1.0")
        if self.split_ratios != (0.70, 0.15, 0.15):
            raise ValueError("cohort.split_ratios must be exactly [0.70, 0.15, 0.15]")
        return self


class ValidationSettings(StrictConfig):
    model_selection_size: int = Field(gt=0)
    calibration_size: int = Field(gt=0)
    policy_size: int = Field(gt=0)


class GoldenSettings(StrictConfig):
    size: int = Field(gt=0)
    edge_size: int = Field(gt=0)
    non_edge_size: int = Field(gt=0)

    @model_validator(mode="after")
    def partitions_sum_to_size(self) -> GoldenSettings:
        if self.edge_size + self.non_edge_size != self.size:
            raise ValueError("golden edge_size + non_edge_size must equal golden size")
        return self


class EmbeddingSettings(StrictConfig):
    model_id: str
    revision: str
    dimension: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: Literal["cpu", "cuda"]


class LLMModelSettings(StrictConfig):
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)


class ModelSettings(StrictConfig):
    embedding: EmbeddingSettings
    drafter: LLMModelSettings
    judge: LLMModelSettings


class PolicySettings(StrictConfig):
    target_error_upper: float = Field(gt=0.0, le=1.0)
    minimum_escalation_recall: float = Field(ge=0.0, le=1.0)
    minimum_auto_count: int = Field(gt=0)


class PathSettings(StrictConfig):
    raw_csv: Path
    artifacts_dir: Path
    frozen_dir: Path


class PipelineSettings(StrictConfig):
    schema_version: Literal["1"]
    seeds: SeedSettings
    cohort: CohortSettings
    validation: ValidationSettings
    golden: GoldenSettings
    models: ModelSettings
    policy: PolicySettings
    paths: PathSettings


def load_settings(path: Path) -> PipelineSettings:
    """Load a YAML file and validate every pipeline decision."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return PipelineSettings.model_validate(raw)
