"""Service layer between HTTP and core: owns the repository and the caches.

The API is stateless apart from these process-local caches, so instances
scale horizontally behind a load balancer with no coordination. Frontier
results are memoised on (snapshot version, constraints); a catalogue update
means a new version string, which naturally invalidates every cached entry.
"""
from __future__ import annotations

import json
from functools import lru_cache

from ..catalogue.repository import CatalogueRepository, Snapshot
from ..core.engine import apply_constraints, recommend
from ..core.pareto import efficient_set
from ..core.types import Constraints, Recommendation, TaskProfile


class DecisionService:
    def __init__(self, repository: CatalogueRepository):
        self.repository = repository
        self._frontier_cached = lru_cache(maxsize=256)(self._frontier_uncached)

    def snapshot(self, version: str | None = None) -> Snapshot:
        return self.repository.get(version) if version else self.repository.latest()

    @staticmethod
    def _constraints_key(constraints: Constraints) -> str:
        return json.dumps(constraints.model_dump(), sort_keys=True, default=str)

    def _frontier_uncached(self, version: str, constraints_key: str
                           ) -> tuple[list[str], list[str], list[str]]:
        snap = self.repository.get(version)
        constraints = Constraints.model_validate(json.loads(constraints_key))
        feasible, infeasible = apply_constraints(snap.models, constraints)
        efficient, dominated = efficient_set(feasible, snap.objectives)
        return ([m.id for m in efficient], [m.id for m in dominated],
                [m.id for m in infeasible])

    def frontier(self, constraints: Constraints, version: str | None = None
                 ) -> dict:
        snap = self.snapshot(version)
        efficient, dominated, infeasible = self._frontier_cached(
            snap.version, self._constraints_key(constraints))
        return {
            "snapshot_version": snap.version,
            "frontier": efficient,
            "dominated": dominated,
            "infeasible": infeasible,
        }

    def recommend(self, profile: TaskProfile, version: str | None = None
                  ) -> Recommendation:
        snap = self.snapshot(version)
        return recommend(snap.models, profile, snap.objectives,
                         snapshot_version=snap.version)
