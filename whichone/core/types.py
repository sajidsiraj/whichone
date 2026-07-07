"""Domain types shared by every layer.

Objectives are data, not code: the engine operates over whatever ObjectiveSpec
list a snapshot declares, so adding or replacing objectives never requires a
code change. DEFAULT_OBJECTIVES ships the six-objective set used in the paper.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    MAX = "max"
    MIN = "min"


class ObjectiveSpec(BaseModel):
    id: str
    name: str
    direction: Direction
    description: str = ""


DEFAULT_OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec(id="capability", name="General capability", direction=Direction.MAX,
                  description="Composite of public benchmark scores (0-100)"),
    ObjectiveSpec(id="tool_reliability", name="Tool-calling reliability", direction=Direction.MAX,
                  description="Function-calling success rate on a standardised suite (0-1)"),
    ObjectiveSpec(id="cost", name="Cost", direction=Direction.MIN,
                  description="Blended price per million tokens, USD"),
    ObjectiveSpec(id="latency", name="Latency", direction=Direction.MIN,
                  description="Median time-to-first-token under probe load, seconds"),
    ObjectiveSpec(id="context", name="Context capacity", direction=Direction.MAX,
                  description="Maximum context window, thousands of tokens"),
    ObjectiveSpec(id="safety", name="Safety and robustness", direction=Direction.MAX,
                  description="Composite of published safety evaluations (0-100)"),
]


class MeasuredValue(BaseModel):
    """An objective value with provenance; provenance is part of the audit record."""
    value: float
    source: str = "unspecified"
    measured_at: Optional[datetime] = None


class ModelOffering(BaseModel):
    """A model as offered by one provider (same weights, two providers: two offerings)."""
    id: str
    name: str
    provider: str = ""
    deployment_class: str = "api"  # api | vpc | on_prem
    objectives: dict[str, MeasuredValue]

    def raw(self, objective_id: str) -> float:
        return self.objectives[objective_id].value


class Constraints(BaseModel):
    """Stage-0 hard constraints; non-compensatory by construction."""
    min_context: Optional[float] = None
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    allowed_deployment_classes: Optional[list[str]] = None
    provider_allowlist: Optional[list[str]] = None
    provider_denylist: Optional[list[str]] = None


class TaskProfile(BaseModel):
    """What the caller tells us about the task.

    Exactly one of `weights` (explicit, objective id to weight) or
    `intensities` (criterion id to ordinal 1-5, mapped through the loading
    matrix) should be provided; explicit weights win if both are present.
    `veto` sets raw-value aspiration levels: for a benefit objective the model
    must reach at least the given value, for a cost objective at most.
    """
    weights: Optional[dict[str, float]] = None
    intensities: Optional[dict[str, int]] = None
    constraints: Constraints = Field(default_factory=Constraints)
    veto: Optional[dict[str, float]] = None
    method: str = "saw"


class ScoredModel(BaseModel):
    model_id: str
    score: float
    normalised: dict[str, float]


class StageReport(BaseModel):
    """Which models each stage removed, and why the survivors survived."""
    infeasible: list[str] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)
    vetoed: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """The full audit record; the top pick is just its first line."""
    recommended: Optional[str]
    ranking: list[ScoredModel]
    frontier: list[str]
    weights: dict[str, float]
    method: str
    cross_check_method: Optional[str] = None
    cross_check_agrees: Optional[bool] = None
    stages: StageReport
    snapshot_version: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
