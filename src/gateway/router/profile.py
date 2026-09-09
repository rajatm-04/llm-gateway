"""Load a small, model-specific suitability profile; never infer validation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    max_input_words: int = Field(gt=0)


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    validated: bool = False
    notes: str = ""
    tasks: dict[Literal["rewrite", "summary", "extraction"], TaskPolicy]


def load_profile(path: Path) -> ModelProfile:
    """A missing or invalid configured profile is a startup/configuration error."""
    return ModelProfile.model_validate_json(path.read_text(encoding="utf-8"))
