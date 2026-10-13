"""Service layer between HTTP and core: owns the repository and the caches.

The API is stateless apart from these process-local caches, so instances
scale horizontally behind a load balancer with no coordination. Efficient
sets are memoised on (snapshot version, constraints, task parameters); a
catalogue update means a new version string, which invalidates every cached
entry.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from ..catalogue.repository import CatalogueRepository, Snapshot
from ..core.engine import (apply_constraints, efficient_at_length,
                           frontier_intervals, recommend)
from ..core.pareto import efficient_set
from ..core.types import Constraints, Recommendation, TaskProfile


class DecisionService:
    def __init__(self, repository: CatalogueRepository):
        self.repository = repository
        self._frontier_cached = lru_cache(maxsize=256)(self._frontier_uncached)
        self._intervals_cached = lru_cache(maxsize=64)(self._intervals_uncached)

    def snapshot(self, version: str | None = None) -> Snapshot:
        return self.repository.get(version) if version else self.repository.latest()

    @staticmethod
    def _constraints_key(constraints: Constraints) -> str:
        return json.dumps(constraints.model_dump(), sort_keys=True, default=str)

    def _screen(self, snap: Snapshot, constraints_key: str):
        constraints = Constraints.model_validate(json.loads(constraints_key))
        feasible, infeasible = apply_constraints(snap.models, constraints)
        kept, pruned = efficient_set(feasible, snap.objectives)
        return constraints, kept, pruned, infeasible

    def _frontier_uncached(self, version: str, constraints_key: str, length: int,
                           beta: Optional[float], max_attempts: Optional[int]) -> dict:
        snap = self.repository.get(version)
        constraints, kept, pruned, infeasible = self._screen(snap, constraints_key)
        task = efficient_at_length(kept, snap.objectives, length, beta, max_attempts,
                                   constraints.max_per_task)
        return {"efficient": task["efficient"],
                "per_success": task["per_success"],
                "dominated_per_step": [m.id for m in pruned],
                "dominated_at_length": task["dominated"],
                "below_completion": task["below_completion"],
                "over_ceiling": task["over_ceiling"],
                "infeasible": [m.id for m in infeasible]}

    def frontier(self, constraints: Constraints, length: int = 1,
                 completion_probability: Optional[float] = None,
                 max_attempts: Optional[int] = None,
                 version: str | None = None) -> dict:
        """The efficient set for a task, without preferences. Same answer for
        every caller with the same task, so it is served from cache."""
        snap = self.snapshot(version)
        body = self._frontier_cached(snap.version, self._constraints_key(constraints),
                                     length, completion_probability, max_attempts)
        return {"snapshot_version": snap.version, "length": length,
                "max_attempts": max_attempts,
                "completion_probability": completion_probability, **body}

    def _intervals_uncached(self, version: str, constraints_key: str, max_length: int,
                            beta: Optional[float], max_attempts: Optional[int]) -> list:
        snap = self.repository.get(version)
        _, kept, _, _ = self._screen(snap, constraints_key)
        return frontier_intervals(kept, snap.objectives, max_length, beta, max_attempts)

    def intervals(self, constraints: Constraints, max_length: int,
                  completion_probability: Optional[float] = None,
                  max_attempts: Optional[int] = None,
                  version: str | None = None) -> dict:
        """Efficient sets for every task length up to max_length, as intervals."""
        snap = self.snapshot(version)
        rows = self._intervals_cached(snap.version, self._constraints_key(constraints),
                                      max_length, completion_probability, max_attempts)
        return {"snapshot_version": snap.version, "max_attempts": max_attempts,
                "completion_probability": completion_probability,
                "intervals": [{"from_length": a, "to_length": b, "efficient": ids}
                              for a, b, ids in rows]}

    def recommend(self, profile: TaskProfile, version: str | None = None
                  ) -> Recommendation:
        snap = self.snapshot(version)
        return recommend(snap.models, profile, snap.objectives,
                         snapshot_version=snap.version)
