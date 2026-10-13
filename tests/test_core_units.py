"""Unit tests for pareto, weights and engine edge cases."""
import numpy as np
import pytest

from whichone.core import (Constraints, MeasuredValue, ModelOffering,
                           ObjectiveSpec, TaskProfile, ahp_weights,
                           dominates, efficient_set, intensities_to_weights,
                           recommend, validate_weights)
from whichone.core.types import Direction


def make(mid: str, rho: float, cost: float, **kw) -> ModelOffering:
    return ModelOffering(
        id=mid, name=mid,
        objectives={"reliability": MeasuredValue(value=rho),
                    "cost": MeasuredValue(value=cost)}, **kw)


SPECS = [
    ObjectiveSpec(id="reliability", name="rho", direction=Direction.MAX),
    ObjectiveSpec(id="cost", name="cost", direction=Direction.MIN),
]


class TestPareto:
    def test_strict_domination(self):
        a, b = make("a", 0.99, 5), make("b", 0.98, 10)
        assert dominates(a, b, SPECS)
        assert not dominates(b, a, SPECS)

    def test_equal_on_all_is_not_domination(self):
        a, b = make("a", 0.99, 5), make("b", 0.99, 5)
        assert not dominates(a, b, SPECS)
        assert not dominates(b, a, SPECS)

    def test_tradeoff_pair_both_efficient(self):
        a, b = make("a", 0.99, 20), make("b", 0.97, 2)
        efficient, dominated = efficient_set([a, b], SPECS)
        assert {m.id for m in efficient} == {"a", "b"}
        assert dominated == []

    def test_epsilon_dominance_prunes_near_duplicate(self):
        a, b = make("a", 0.990, 5.0), make("b", 0.992, 5.5)
        # Without tolerance both are efficient.
        efficient, _ = efficient_set([a, b], SPECS)
        assert len(efficient) == 2
        # With reliability differences under 0.005 treated as ties, a dominates b.
        efficient, dominated = efficient_set([a, b], SPECS,
                                             eps={"reliability": 0.005})
        assert [m.id for m in efficient] == ["a"]
        assert [m.id for m in dominated] == ["b"]


class TestWeights:
    def test_ahp_recovers_known_weights(self):
        # Perfectly consistent matrix built from w = (0.6, 0.3, 0.1).
        w = np.array([0.6, 0.3, 0.1])
        matrix = w[:, None] / w[None, :]
        weights, cr = ahp_weights(matrix)
        assert weights == pytest.approx(w, abs=1e-9)
        assert cr == pytest.approx(0.0, abs=1e-9)

    def test_ahp_rejects_nonsquare(self):
        with pytest.raises(ValueError):
            ahp_weights(np.ones((2, 3)))

    def test_intensities_map_and_normalise(self):
        w = intensities_to_weights(
            {"cost_sensitivity": 5, "carbon_sensitivity": 3, "time_sensitivity": 2},
            ["energy", "cost", "time"])
        assert sum(w.values()) == pytest.approx(1.0)
        assert w == pytest.approx({"energy": 0.3, "cost": 0.5, "time": 0.2})

    def test_unknown_criterion_rejected(self):
        with pytest.raises(ValueError, match="unknown criteria"):
            intensities_to_weights({"vibes": 5}, ["energy", "cost"])

    def test_validate_weights_requires_exact_cover(self):
        with pytest.raises(ValueError, match="missing"):
            validate_weights({"energy": 1.0}, ["energy", "cost"])


class TestEngine:
    def test_constraints_remove_infeasible(self):
        a = make("a", 0.99, 5, attributes={"context": MeasuredValue(value=32)})
        b = make("b", 0.95, 1, attributes={"context": MeasuredValue(value=200)})
        rec = recommend([a, b], TaskProfile(constraints=Constraints(min_context=100)), SPECS)
        assert rec.stages.infeasible == ["a"]
        assert rec.recommended == "b"

    def test_unknown_context_is_infeasible(self):
        rec = recommend([make("a", 0.99, 5)],
                        TaskProfile(constraints=Constraints(min_context=8)), SPECS)
        assert rec.stages.infeasible == ["a"] and rec.recommended is None

    def test_per_task_ceiling(self):
        a, b = make("a", 0.99, 5), make("b", 0.95, 1)
        rec = recommend([a, b], TaskProfile(length=40, constraints=Constraints(
            max_per_task={"cost": 300})), SPECS)
        # b uses 1 * 40 / 0.95^40, about 311 per successful task
        assert rec.stages.over_ceiling == ["b"]
        assert rec.recommended == "a"

    def test_ceiling_on_unknown_resource_rejected(self):
        with pytest.raises(ValueError, match="unknown resources"):
            recommend([make("a", 0.99, 5)], TaskProfile(
                constraints=Constraints(max_per_task={"water": 1})), SPECS)

    def test_crossover_switches_the_recommendation(self):
        a, b = make("a", 0.99, 5), make("b", 0.95, 1)
        # one resource: crossover at ln 5 / ln(0.99/0.95), about 39 steps
        assert recommend([a, b], TaskProfile(length=38), SPECS).recommended == "b"
        assert recommend([a, b], TaskProfile(length=40), SPECS).recommended == "a"

    def test_trade_off_without_weights_asks_for_them(self):
        specs = SPECS + [ObjectiveSpec(id="time", name="time", direction=Direction.MIN)]
        a = make("a", 0.99, 5, )
        a.objectives["time"] = MeasuredValue(value=1)
        b = make("b", 0.95, 1)
        b.objectives["time"] = MeasuredValue(value=2)
        rec = recommend([a, b], TaskProfile(length=10), specs)
        assert rec.preferences_needed and rec.recommended is None
        rec = recommend([a, b], TaskProfile(length=10, weights={"cost": 0.9, "time": 0.1}), specs)
        assert rec.recommended == "b"
        assert rec.cross_check_method == "topsis"
        assert rec.cross_check_agrees in (True, False)

    def test_snapshot_without_reliability_rejected(self):
        specs = [ObjectiveSpec(id="cost", name="cost", direction=Direction.MIN)]
        with pytest.raises(ValueError, match="reliability"):
            recommend([make("a", 0.99, 5)], TaskProfile(), specs)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            recommend([make("a", 0.99, 5)], TaskProfile(method="promethee"), SPECS)
