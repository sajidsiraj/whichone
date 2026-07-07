"""The pipeline: feasibility, efficiency, veto, ranking, audit.

Pure functions over in-memory data; no I/O, no caching, no web. Callers own
persistence and caching (see whichone.api.service).
"""
from __future__ import annotations

from .madm import METHODS, Normalised, normalise, rank
from .pareto import efficient_set
from .types import (Constraints, Direction, ModelOffering, ObjectiveSpec,
                    Recommendation, ScoredModel, StageReport, TaskProfile)
from .weights import intensities_to_weights, validate_weights


def apply_constraints(models: list[ModelOffering], c: Constraints
                      ) -> tuple[list[ModelOffering], list[ModelOffering]]:
    feasible, infeasible = [], []
    for m in models:
        ok = True
        if c.min_context is not None and m.raw("context") < c.min_context:
            ok = False
        if c.max_cost is not None and m.raw("cost") > c.max_cost:
            ok = False
        if c.max_latency is not None and m.raw("latency") > c.max_latency:
            ok = False
        if c.allowed_deployment_classes is not None and \
                m.deployment_class not in c.allowed_deployment_classes:
            ok = False
        if c.provider_allowlist is not None and m.provider not in c.provider_allowlist:
            ok = False
        if c.provider_denylist is not None and m.provider in c.provider_denylist:
            ok = False
        (feasible if ok else infeasible).append(m)
    return feasible, infeasible


def apply_veto(models: list[ModelOffering], specs: list[ObjectiveSpec],
               veto: dict[str, float]
               ) -> tuple[list[ModelOffering], list[ModelOffering]]:
    """Raw-value aspiration levels: floor for benefit objectives, ceiling for
    cost objectives. Guaranteed independent of any weight choice."""
    by_id = {s.id: s for s in specs}
    unknown = set(veto) - set(by_id)
    if unknown:
        raise ValueError(f"veto targets unknown objectives: {sorted(unknown)}")
    passed, vetoed = [], []
    for m in models:
        ok = True
        for obj, level in veto.items():
            v = m.raw(obj)
            if by_id[obj].direction == Direction.MAX and v < level:
                ok = False
            if by_id[obj].direction == Direction.MIN and v > level:
                ok = False
        (passed if ok else vetoed).append(m)
    return passed, vetoed


def recommend(models: list[ModelOffering], profile: TaskProfile,
              specs: list[ObjectiveSpec], snapshot_version: str = "",
              cross_check: str | None = "topsis") -> Recommendation:
    """Run the full two-stage pipeline and return the audit record."""
    objective_ids = [s.id for s in specs]

    # Weights first: fail fast on a bad profile before any screening work.
    if profile.weights is not None:
        weights = validate_weights(profile.weights, objective_ids)
    elif profile.intensities is not None:
        weights = intensities_to_weights(profile.intensities, objective_ids)
    else:
        raise ValueError("profile must supply weights or intensities")

    if profile.method not in METHODS:
        raise ValueError(f"unknown method {profile.method}; "
                         f"available: {sorted(METHODS)}")

    # Stage 0: feasibility.
    feasible, infeasible = apply_constraints(models, profile.constraints)

    # Stage 1: efficiency (preference-free).
    efficient, dominated = efficient_set(feasible, specs)

    # Normalise within the efficient set (the paper's reference frame), then
    # apply the veto before ranking.
    normalised: Normalised = normalise(efficient, specs) if efficient else {}
    if profile.veto:
        ranked_pool, vetoed = apply_veto(efficient, specs, profile.veto)
    else:
        ranked_pool, vetoed = efficient, []

    scores = METHODS[profile.method](
        {m.id: normalised[m.id] for m in ranked_pool}, weights)
    order = rank(scores)

    agrees = None
    if cross_check and cross_check != profile.method and order:
        alt = METHODS[cross_check](
            {m.id: normalised[m.id] for m in ranked_pool}, weights)
        agrees = rank(alt)[0] == order[0]

    return Recommendation(
        recommended=order[0] if order else None,
        ranking=[ScoredModel(model_id=mid, score=scores[mid],
                             normalised=normalised[mid]) for mid in order],
        frontier=[m.id for m in efficient],
        weights=weights,
        method=profile.method,
        cross_check_method=cross_check if agrees is not None else None,
        cross_check_agrees=agrees,
        stages=StageReport(
            infeasible=[m.id for m in infeasible],
            dominated=[m.id for m in dominated],
            vetoed=[m.id for m in vetoed],
        ),
        snapshot_version=snapshot_version,
    )
