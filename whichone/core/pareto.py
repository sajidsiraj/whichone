"""Stage 1: preference-free efficiency screen (MODM).

Non-dominated sorting over oriented objective values. O(p * m^2), which is
microseconds at marketplace scale (m of order 10^2); the win from caching is
avoiding re-reads of the catalogue, not the sort itself.

The efficient set depends only on within-objective orderings, so it is
invariant under any strictly increasing rescaling of any objective. No
normalisation or weight choice can change it.
"""
from __future__ import annotations

from .types import Direction, ModelOffering, ObjectiveSpec


def oriented(model: ModelOffering, spec: ObjectiveSpec) -> float:
    """Objective value oriented so that larger is always better."""
    v = model.raw(spec.id)
    return v if spec.direction == Direction.MAX else -v


def dominates(a: ModelOffering, b: ModelOffering, specs: list[ObjectiveSpec],
              eps: dict[str, float] | None = None) -> bool:
    """True iff `a` dominates `b`: at least as good on every objective and
    strictly better on at least one.

    With `eps` (objective id to tolerance), differences within the tolerance
    count as ties, giving epsilon-dominance for pruning near-duplicates.
    """
    at_least_one_strict = False
    for spec in specs:
        tol = (eps or {}).get(spec.id, 0.0)
        va, vb = oriented(a, spec), oriented(b, spec)
        if va < vb - tol:
            return False
        if va > vb + tol:
            at_least_one_strict = True
    return at_least_one_strict


def efficient_set(models: list[ModelOffering], specs: list[ObjectiveSpec],
                  eps: dict[str, float] | None = None
                  ) -> tuple[list[ModelOffering], list[ModelOffering]]:
    """Split `models` into (efficient, dominated)."""
    efficient: list[ModelOffering] = []
    dominated: list[ModelOffering] = []
    for candidate in models:
        if any(dominates(other, candidate, specs, eps)
               for other in models if other.id != candidate.id):
            dominated.append(candidate)
        else:
            efficient.append(candidate)
    return efficient, dominated
