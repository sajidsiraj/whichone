"""Domain types shared by every layer.

A model is described per step of a task family: its per-step reliability and
the energy, cost and time it uses per step (paper, Table "Model quantities").
A task is described by its profile: length L, retry limit K and the required
probability of completion beta. The engine turns the two into resources per
successfully completed task, which is what the decision is made on.

Objectives are data, not code: a snapshot declares its ObjectiveSpec list.
The spec with id `reliability` is the per-step reliability; every other spec
is a resource used per step (direction min). DEFAULT_OBJECTIVES is the set
used in the paper.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

RELIABILITY = "reliability"


class Direction(str, Enum):
    MAX = "max"
    MIN = "min"


class ObjectiveSpec(BaseModel):
    id: str
    name: str
    direction: Direction
    description: str = ""


DEFAULT_OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec(id=RELIABILITY, name="Per-step reliability", direction=Direction.MAX,
                  description="Share of steps completed correctly in runs of the task family (0-1)"),
    ObjectiveSpec(id="energy", name="Energy per step", direction=Direction.MIN,
                  description="Wh per step; measured for open-weight models, estimated with bounds otherwise"),
    ObjectiveSpec(id="cost", name="Cost per step", direction=Direction.MIN,
                  description="US cents per step: tokens per step times the provider's price"),
    ObjectiveSpec(id="time", name="Time per step", direction=Direction.MIN,
                  description="Seconds per step: measured response time"),
]


class MeasuredValue(BaseModel):
    """A value with provenance; provenance is part of the audit record."""
    value: float
    source: str = "unspecified"
    measured_at: Optional[datetime] = None
    estimated: bool = False  # True for estimates (e.g. energy of closed models)


class ModelOffering(BaseModel):
    """A model as offered by one provider (same weights, two providers: two offerings).

    `objectives` holds the per-step quantities. `attributes` holds values that
    enter only as Stage-0 constraints, such as context capacity (thousands of
    tokens) and a safety score.
    """
    id: str
    name: str
    provider: str = ""
    deployment_class: str = "api"  # api | vpc | on_prem
    objectives: dict[str, MeasuredValue]
    attributes: dict[str, MeasuredValue] = Field(default_factory=dict)

    def raw(self, key: str) -> float:
        if key in self.objectives:
            return self.objectives[key].value
        return self.attributes[key].value

    def has(self, key: str) -> bool:
        return key in self.objectives or key in self.attributes


class Constraints(BaseModel):
    """Stage-0 hard constraints; non-compensatory by construction.

    The ceilings apply to resources per successfully completed task, in the
    snapshot's units (for the default objectives: Wh, US cents, seconds).
    """
    min_context: Optional[float] = None
    min_safety: Optional[float] = None
    allowed_deployment_classes: Optional[list[str]] = None
    provider_allowlist: Optional[list[str]] = None
    provider_denylist: Optional[list[str]] = None
    max_per_task: Optional[dict[str, float]] = None


class TaskProfile(BaseModel):
    """What the caller tells us about the task.

    length:                 steps in one attempt (L)
    max_attempts:           retry limit K; None means failed attempts can be
                            retried freely, 1 means the task is irreversible
    completion_probability: required chance beta of completing the task within
                            K attempts; None means no chance constraint
    weights / intensities:  preferences over the resources, used only when more
                            than one model is efficient for this task (Stage 2).
                            Explicit weights win if both are present.
    """
    length: int = Field(default=1, ge=1)
    max_attempts: Optional[int] = Field(default=None, ge=1)
    completion_probability: Optional[float] = Field(default=None, gt=0, le=1)
    weights: Optional[dict[str, float]] = None
    intensities: Optional[dict[str, int]] = None
    constraints: Constraints = Field(default_factory=Constraints)
    method: str = "saw"


class ScoredModel(BaseModel):
    model_id: str
    score: float
    normalised: dict[str, float]


class StageReport(BaseModel):
    """Which models each stage removed. Lists are in candidate order."""
    infeasible: list[str] = Field(default_factory=list)            # Stage 0 constraints
    dominated_per_step: list[str] = Field(default_factory=list)    # Stage 1: out for every length
    below_completion: list[str] = Field(default_factory=list)      # chance constraint at this L
    over_ceiling: list[str] = Field(default_factory=list)          # per-task ceilings at this L
    dominated_at_length: list[str] = Field(default_factory=list)   # not efficient at this L


class Recommendation(BaseModel):
    """The full audit record; the top pick is just its first line.

    `preferences_needed` is True when more than one model is efficient and no
    weights were given: the efficient set is then the answer, and the choice
    within it is the decision maker's. `recommended` is None in that case, and
    when no model is feasible.
    """
    recommended: Optional[str]
    efficient: list[str]
    per_success: dict[str, dict[str, float]]
    attempt_success: dict[str, float]
    preferences_needed: bool = False
    ranking: list[ScoredModel] = Field(default_factory=list)
    weights: Optional[dict[str, float]] = None
    method: Optional[str] = None
    cross_check_method: Optional[str] = None
    cross_check_agrees: Optional[bool] = None
    length: int
    max_attempts: Optional[int] = None
    completion_probability: Optional[float] = None
    stages: StageReport
    note: str = ""
    snapshot_version: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
