"""The pipeline of the paper (Section 4), for one task profile:

  Stage 0  feasibility: hard constraints (context, safety, deployment, provider)
  Stage 1  per-step screen: a model worse on every per-step quantity is
           dominated for every task length (Observation 2), so it is removed once
  at L     resources per successful task R_k = r_k L / rho^L; the chance
           constraint P(L, K) >= beta and any per-task ceilings; the efficient
           set E(L) over (R_energy, R_cost, R_time)
  Stage 2  only if E(L) has more than one model: weights over the resources,
           SAW on values rescaled within E(L), with TOPSIS as a check

Pure functions over in-memory data; no I/O, no caching, no web. Callers own
persistence and caching (see whichone.api.service).
"""
from __future__ import annotations

import math
from typing import Optional

from .horizon import attempt_success, min_attempt_success, per_success
from .madm import METHODS, normalise, rank
from .pareto import efficient_set
from .types import (RELIABILITY, Constraints, Direction, MeasuredValue,
                    ModelOffering, ObjectiveSpec, Recommendation, ScoredModel,
                    StageReport, TaskProfile)
from .weights import intensities_to_weights, validate_weights


def resource_ids(specs: list[ObjectiveSpec]) -> list[str]:
    """The resources of a snapshot: every objective except reliability.
    Checks that the snapshot has the shape the model needs."""
    ids = [s.id for s in specs]
    if RELIABILITY not in ids:
        raise ValueError(f"objectives must include {RELIABILITY!r} (per-step reliability)")
    res = [s for s in specs if s.id != RELIABILITY]
    if not res:
        raise ValueError("objectives must include at least one resource per step")
    bad = [s.id for s in res if s.direction != Direction.MIN]
    if bad:
        raise ValueError(f"resources per step must have direction 'min': {bad}")
    return [s.id for s in res]


def apply_constraints(models: list[ModelOffering], c: Constraints
                      ) -> tuple[list[ModelOffering], list[ModelOffering]]:
    """Stage 0. A model that lacks a value a constraint needs is infeasible:
    an unknown context or safety level cannot be assumed to be enough."""
    def ok(m: ModelOffering) -> bool:
        if c.min_context is not None and not (m.has("context") and m.raw("context") >= c.min_context):
            return False
        if c.min_safety is not None and not (m.has("safety") and m.raw("safety") >= c.min_safety):
            return False
        if c.allowed_deployment_classes is not None and m.deployment_class not in c.allowed_deployment_classes:
            return False
        if c.provider_allowlist is not None and m.provider not in c.provider_allowlist:
            return False
        if c.provider_denylist is not None and m.provider in c.provider_denylist:
            return False
        return True
    feasible = [m for m in models if ok(m)]
    return feasible, [m for m in models if not ok(m)]


def at_length(m: ModelOffering, resources: list[str], length: int
              ) -> tuple[float, dict[str, float]]:
    """(attempt success S(L), resource k per successful task) in the basic case."""
    s = attempt_success(m.raw(RELIABILITY), length)
    return s, {k: per_success(m.raw(k), length, s) for k in resources}


def _per_success_offering(m: ModelOffering, values: dict[str, float]) -> ModelOffering:
    return ModelOffering(id=m.id, name=m.name, provider=m.provider,
                         deployment_class=m.deployment_class,
                         objectives={k: MeasuredValue(value=v, source="per successful task")
                                     for k, v in values.items()})


def _per_success_specs(specs: list[ObjectiveSpec]) -> list[ObjectiveSpec]:
    return [ObjectiveSpec(id=s.id, name=f"{s.name} per successful task",
                          direction=Direction.MIN) for s in specs if s.id != RELIABILITY]


def efficient_at_length(models: list[ModelOffering], specs: list[ObjectiveSpec],
                        length: int, completion_probability: Optional[float] = None,
                        max_attempts: Optional[int] = None,
                        max_per_task: Optional[dict[str, float]] = None) -> dict:
    """Everything the decision at one task length needs, without preferences."""
    resources = resource_ids(specs)
    if max_per_task:
        unknown = set(max_per_task) - set(resources)
        if unknown:
            raise ValueError(f"max_per_task names unknown resources: {sorted(unknown)}")
    k_attempts = math.inf if max_attempts is None else max_attempts
    floor = (min_attempt_success(completion_probability, k_attempts)
             if completion_probability is not None else 0.0)

    success, values = {}, {}
    below, over, candidates = [], [], []
    for m in models:
        s, r = at_length(m, resources, length)
        success[m.id] = s
        if s <= 0 or s < floor:
            below.append(m.id)
            continue
        values[m.id] = r
        if max_per_task and any(r[k] > lim for k, lim in max_per_task.items()):
            over.append(m.id)
            continue
        candidates.append(m)
    eff, dom = efficient_set([_per_success_offering(m, values[m.id]) for m in candidates],
                             _per_success_specs(specs))
    return {"efficient": [m.id for m in eff], "dominated": [m.id for m in dom],
            "below_completion": below, "over_ceiling": over,
            "attempt_success": success, "per_success": values,
            "offerings": {m.id: m for m in eff}}


def frontier_intervals(models: list[ModelOffering], specs: list[ObjectiveSpec],
                       max_length: int, completion_probability: Optional[float] = None,
                       max_attempts: Optional[int] = None
                       ) -> list[tuple[int, int, list[str]]]:
    """Efficient sets as intervals (first L, last L, ids) for L = 1..max_length.
    The set changes only at crossover lengths, so this is what a service stores."""
    out: list[tuple[int, int, list[str]]] = []
    for length in range(1, max_length + 1):
        ids = sorted(efficient_at_length(models, specs, length, completion_probability,
                                         max_attempts)["efficient"])
        if out and out[-1][2] == ids:
            out[-1] = (out[-1][0], length, ids)
        else:
            out.append((length, length, ids))
    return out


NO_MODEL_NOTE = ("No model meets the requirements for this task. If the required chance "
                 "of completion rules out every model, consider changing the task, for "
                 "example by adding a human check before irreversible steps.")
PREFERENCES_NOTE = ("More than one model is efficient for this task: each is better on "
                    "some resource. Give weights over the resources to choose among them.")


def recommend(models: list[ModelOffering], profile: TaskProfile,
              specs: list[ObjectiveSpec], snapshot_version: str = "",
              cross_check: str | None = "topsis") -> Recommendation:
    """Run the full pipeline for one task profile and return the audit record."""
    resources = resource_ids(specs)

    # Preferences first: fail fast on a bad profile before any screening work.
    if profile.weights is not None:
        weights = validate_weights(profile.weights, resources)
    elif profile.intensities is not None:
        weights = intensities_to_weights(profile.intensities, resources)
    else:
        weights = None
    if profile.method not in METHODS:
        raise ValueError(f"unknown method {profile.method}; available: {sorted(METHODS)}")

    feasible, infeasible = apply_constraints(models, profile.constraints)
    kept, pruned = efficient_set(feasible, specs)  # per-step screen
    task = efficient_at_length(kept, specs, profile.length, profile.completion_probability,
                               profile.max_attempts, profile.constraints.max_per_task)
    efficient = task["efficient"]

    rec = dict(efficient=efficient, per_success=task["per_success"],
               attempt_success=task["attempt_success"], length=profile.length,
               max_attempts=profile.max_attempts,
               completion_probability=profile.completion_probability,
               snapshot_version=snapshot_version,
               stages=StageReport(infeasible=[m.id for m in infeasible],
                                  dominated_per_step=[m.id for m in pruned],
                                  below_completion=task["below_completion"],
                                  over_ceiling=task["over_ceiling"],
                                  dominated_at_length=task["dominated"]))
    if not efficient:
        return Recommendation(recommended=None, note=NO_MODEL_NOTE, **rec)
    if len(efficient) == 1:
        return Recommendation(recommended=efficient[0], weights=weights, **rec)
    if weights is None:
        return Recommendation(recommended=None, preferences_needed=True,
                              note=PREFERENCES_NOTE, **rec)

    # Stage 2: rescale within E(L) so the best model on each resource scores 1.
    pool = [task["offerings"][mid] for mid in efficient]
    normalised = normalise(pool, _per_success_specs(specs))
    scores = METHODS[profile.method](normalised, weights)
    order = rank(scores)
    agrees = None
    if cross_check and cross_check != profile.method:
        agrees = rank(METHODS[cross_check](normalised, weights))[0] == order[0]
    return Recommendation(
        recommended=order[0], weights=weights, method=profile.method,
        ranking=[ScoredModel(model_id=mid, score=scores[mid], normalised=normalised[mid])
                 for mid in order],
        cross_check_method=cross_check if agrees is not None else None,
        cross_check_agrees=agrees, **rec)
