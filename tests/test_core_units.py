"""Unit tests for pareto, weights and engine edge cases."""
import numpy as np
import pytest

from whichone.core import (Constraints, MeasuredValue, ModelOffering,
                           ObjectiveSpec, TaskProfile, ahp_weights,
                           dominates, efficient_set, intensities_to_weights,
                           recommend, validate_weights)
from whichone.core.types import Direction


def make(mid: str, cap: float, cost: float, **kw) -> ModelOffering:
    return ModelOffering(
        id=mid, name=mid,
        objectives={"capability": MeasuredValue(value=cap),
                    "cost": MeasuredValue(value=cost)}, **kw)


SPECS = [
    ObjectiveSpec(id="capability", name="cap", direction=Direction.MAX),
    ObjectiveSpec(id="cost", name="cost", direction=Direction.MIN),
]


class TestPareto:
    def test_strict_domination(self):
        a, b = make("a", 90, 5), make("b", 80, 10)
        assert dominates(a, b, SPECS)
        assert not dominates(b, a, SPECS)

    def test_equal_on_all_is_not_domination(self):
        a, b = make("a", 90, 5), make("b", 90, 5)
        assert not dominates(a, b, SPECS)
        assert not dominates(b, a, SPECS)

    def test_tradeoff_pair_both_efficient(self):
        a, b = make("a", 90, 20), make("b", 70, 2)
        efficient, dominated = efficient_set([a, b], SPECS)
        assert {m.id for m in efficient} == {"a", "b"}
        assert dominated == []

    def test_epsilon_dominance_prunes_near_duplicate(self):
        a, b = make("a", 90.0, 5.0), make("b", 90.2, 5.5)
        # Without tolerance both are efficient.
        efficient, _ = efficient_set([a, b], SPECS)
        assert len(efficient) == 2
        # With capability differences under 0.5 treated as ties, a dominates b.
        efficient, dominated = efficient_set([a, b], SPECS,
                                             eps={"capability": 0.5})
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
            {"cost_sensitivity": 5, "accuracy": 1},
            ["capability", "cost", "safety"])
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["cost"] > w["capability"] > 0
        assert w["safety"] > 0  # accuracy loads 0.3 onto safety

    def test_unknown_criterion_rejected(self):
        with pytest.raises(ValueError, match="unknown criteria"):
            intensities_to_weights({"vibes": 5}, ["capability", "cost"])

    def test_validate_weights_requires_exact_cover(self):
        with pytest.raises(ValueError, match="missing"):
            validate_weights({"capability": 1.0}, ["capability", "cost"])


class TestEngine:
    def test_constraints_remove_infeasible(self):
        a = make("a", 90, 5)
        b = make("b", 95, 50)
        profile = TaskProfile(weights={"capability": 0.5, "cost": 0.5},
                              constraints=Constraints(max_cost=10))
        rec = recommend([a, b], profile, SPECS)
        assert rec.stages.infeasible == ["b"]
        assert rec.recommended == "a"

    def test_empty_candidate_pool_yields_no_recommendation(self):
        a = make("a", 90, 5)
        profile = TaskProfile(weights={"capability": 0.5, "cost": 0.5},
                              constraints=Constraints(max_cost=1))
        rec = recommend([a], profile, SPECS)
        assert rec.recommended is None
        assert rec.ranking == []

    def test_profile_without_weights_or_intensities_rejected(self):
        with pytest.raises(ValueError, match="weights or intensities"):
            recommend([make("a", 90, 5)], TaskProfile(), SPECS)

    def test_unknown_method_rejected(self):
        profile = TaskProfile(weights={"capability": 1.0, "cost": 0.0},
                              method="promethee")
        with pytest.raises(ValueError, match="unknown method"):
            recommend([make("a", 90, 5)], profile, SPECS)

    def test_cross_check_reported(self):
        a, b = make("a", 90, 20), make("b", 70, 2)
        profile = TaskProfile(weights={"capability": 0.5, "cost": 0.5})
        rec = recommend([a, b], profile, SPECS)
        assert rec.cross_check_method == "topsis"
        assert rec.cross_check_agrees in (True, False)
