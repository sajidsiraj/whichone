"""Stage 2: preference ranking (MADM) over the efficient set.

Aggregation methods live in a registry so new MCDA methods (PROMETHEE,
ELECTRE, ...) plug in without touching the engine or the API: implement
`fn(normalised, weights) -> dict[model_id, score]` and register it.

Normalisation is min-max within the ranked set and benefit-oriented, so every
normalised value reads "1.0 is the best available on this objective".
A constant objective normalises to 1.0 for everyone (it cannot discriminate).
"""
from __future__ import annotations

import math
from typing import Callable

from .types import Direction, ModelOffering, ObjectiveSpec

Normalised = dict[str, dict[str, float]]  # model_id -> objective_id -> [0, 1]
MethodFn = Callable[[Normalised, dict[str, float]], dict[str, float]]

METHODS: dict[str, MethodFn] = {}


def register_method(name: str):
    def deco(fn: MethodFn) -> MethodFn:
        METHODS[name] = fn
        return fn
    return deco


def normalise(models: list[ModelOffering], specs: list[ObjectiveSpec]) -> Normalised:
    out: Normalised = {m.id: {} for m in models}
    for spec in specs:
        values = [m.raw(spec.id) for m in models]
        lo, hi = min(values), max(values)
        span = hi - lo
        for m in models:
            if span == 0:
                v = 1.0
            elif spec.direction == Direction.MAX:
                v = (m.raw(spec.id) - lo) / span
            else:
                v = (hi - m.raw(spec.id)) / span
            out[m.id][spec.id] = v
    return out


@register_method("saw")
def saw(normalised: Normalised, weights: dict[str, float]) -> dict[str, float]:
    """Simple Additive Weighting: the default, and the paper's primary rule."""
    return {
        model_id: sum(weights[obj] * v for obj, v in values.items())
        for model_id, values in normalised.items()
    }


@register_method("topsis")
def topsis(normalised: Normalised, weights: dict[str, float]) -> dict[str, float]:
    """TOPSIS closeness coefficient on the weighted normalised matrix.

    Because normalisation is already benefit-oriented, the ideal point is the
    per-objective maximum and the anti-ideal the minimum.
    """
    if not normalised:
        return {}
    objective_ids = list(next(iter(normalised.values())).keys())
    weighted = {
        mid: {obj: weights[obj] * v for obj, v in values.items()}
        for mid, values in normalised.items()
    }
    ideal = {obj: max(w[obj] for w in weighted.values()) for obj in objective_ids}
    anti = {obj: min(w[obj] for w in weighted.values()) for obj in objective_ids}
    scores: dict[str, float] = {}
    for mid, w in weighted.items():
        d_pos = math.sqrt(sum((w[obj] - ideal[obj]) ** 2 for obj in objective_ids))
        d_neg = math.sqrt(sum((w[obj] - anti[obj]) ** 2 for obj in objective_ids))
        scores[mid] = 0.5 if d_pos + d_neg == 0 else d_neg / (d_pos + d_neg)
    return scores


def rank(scores: dict[str, float]) -> list[str]:
    """Model ids by descending score; ties broken lexically for determinism."""
    return sorted(scores, key=lambda mid: (-scores[mid], mid))
