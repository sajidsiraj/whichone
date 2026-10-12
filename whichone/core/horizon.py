"""Horizon-aware selection: resources per successfully completed task.

An agent completes a task of L steps only if every step succeeds. With
per-step reliability rho, one attempt succeeds with probability S = rho^L
(the basic case), and failed attempts still use energy, cost and time. The
unit of comparison is therefore the resource used per *successful* task:

    R_k(L) = r_k * L / S(L)        for k in {energy, cost, time}

which does not depend on the retry limit K. The retry limit enters only
through the chance constraint P(L, K) = 1 - (1 - S)^K >= beta, which is the
same as S >= 1 - (1 - beta)^(1/K).

Consequences used by the framework (paper, Section 4):
  * a model worse on every per-step quantity is dominated for every L
    (per_step_dominates), so it can be pruned once for all tasks;
  * between two models, each resource has one crossover length
    (crossover_length), with the same slope ln(rho_b / rho_a) for all three;
  * the efficient set changes only at crossover lengths
    (frontier_by_length), so it can be stored per interval of L.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .pareto import efficient_set
from .types import Direction, MeasuredValue, ModelOffering, ObjectiveSpec

RESOURCES = ("energy", "cost", "time")

_PER_SUCCESS_SPECS = [ObjectiveSpec(id=k, name=f"{k} per successful task",
                                    direction=Direction.MIN) for k in RESOURCES]


@dataclass(frozen=True)
class StepProfile:
    """Per-step quantities of one model on one task family."""
    id: str
    reliability: float  # rho: probability that one step is completed correctly
    energy: float       # per step
    cost: float         # per step
    time: float         # per step

    def resource(self, k: str) -> float:
        return getattr(self, k)


def attempt_success(rho: float, length: int) -> float:
    """Probability that one attempt of `length` steps succeeds (basic case)."""
    return rho ** length


def completion_probability(success: float, max_attempts: float) -> float:
    """Probability of completing the task within `max_attempts` attempts."""
    if math.isinf(max_attempts):
        return 1.0 if success > 0 else 0.0
    return 1.0 - (1.0 - success) ** max_attempts


def min_attempt_success(beta: float, max_attempts: float) -> float:
    """Smallest single-attempt success that meets P(L, K) >= beta."""
    if math.isinf(max_attempts):
        return 0.0
    return 1.0 - (1.0 - beta) ** (1.0 / max_attempts)


def per_success(resource_per_step: float, length: int, success: float) -> float:
    """Expected resource per successfully completed task, for any retry limit."""
    if success <= 0:
        return math.inf
    return resource_per_step * length / success


def crossover_length(rho_a: float, rho_b: float, r_a: float, r_b: float
                     ) -> Optional[float]:
    """Task length beyond which the more reliable model b uses less of a
    resource per successful task than model a.

    Defined when a is less reliable but cheaper per step on that resource
    (rho_a < rho_b, r_a < r_b). Returns None otherwise: then one model is
    better on that resource at every length.
    """
    if not (rho_a < rho_b and r_a < r_b):
        return None
    return math.log(r_b / r_a) / math.log(rho_b / rho_a)


def per_step_dominates(a: StepProfile, b: StepProfile) -> bool:
    """True iff a is at least as good as b on every per-step quantity and
    strictly better on one. Then a dominates b at every task length."""
    ge = (a.reliability >= b.reliability
          and all(a.resource(k) <= b.resource(k) for k in RESOURCES))
    strict = (a.reliability > b.reliability
              or any(a.resource(k) < b.resource(k) for k in RESOURCES))
    return ge and strict


def per_step_screen(models: list[StepProfile]
                    ) -> tuple[list[StepProfile], list[StepProfile]]:
    """Stage 1: split into (kept, pruned for every task length)."""
    kept, pruned = [], []
    for m in models:
        if any(per_step_dominates(o, m) for o in models if o.id != m.id):
            pruned.append(m)
        else:
            kept.append(m)
    return kept, pruned


def feasible(model: StepProfile, length: int, beta: Optional[float] = None,
             max_attempts: float = math.inf) -> bool:
    """Chance constraint P(L, K) >= beta (always true when beta is None)."""
    if beta is None:
        return True
    return attempt_success(model.reliability, length) >= min_attempt_success(beta, max_attempts)


def _as_offering(m: StepProfile, length: int) -> ModelOffering:
    s = attempt_success(m.reliability, length)
    return ModelOffering(id=m.id, name=m.id, objectives={
        k: MeasuredValue(value=per_success(m.resource(k), length, s), source="horizon")
        for k in RESOURCES})


def efficient_at(models: list[StepProfile], length: int, beta: Optional[float] = None,
                 max_attempts: float = math.inf) -> list[str]:
    """Ids of the feasible models that are efficient per successful task."""
    cands = [_as_offering(m, length) for m in models
             if feasible(m, length, beta, max_attempts)]
    eff, _ = efficient_set(cands, _PER_SUCCESS_SPECS)
    return sorted(m.id for m in eff)


def frontier_by_length(models: list[StepProfile], max_length: int,
                       beta: Optional[float] = None, max_attempts: float = math.inf
                       ) -> list[tuple[int, int, tuple[str, ...]]]:
    """Efficient sets as intervals (first L, last L, ids) for L = 1..max_length."""
    out: list[tuple[int, int, tuple[str, ...]]] = []
    for length in range(1, max_length + 1):
        ids = tuple(efficient_at(models, length, beta, max_attempts))
        if out and out[-1][2] == ids:
            out[-1] = (out[-1][0], length, ids)
        else:
            out.append((length, length, ids))
    return out
