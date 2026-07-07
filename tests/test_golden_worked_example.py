"""Golden tests: the code must reproduce the paper's worked example exactly
(methodology section, Tables on the illustrative candidate set and SAW
scores). If these fail, either the code or the paper is wrong; find out which
before touching the assertions.
"""
import pytest

from whichone.catalogue import default_repository
from whichone.core import TaskProfile, recommend

SNAPSHOT = "2026-07-06-worked-example"

# Paper values, 3 decimal places.
T1_WEIGHTS = {"capability": 0.30, "tool_reliability": 0.15, "cost": 0.05,
              "latency": 0.05, "context": 0.10, "safety": 0.35}
T1_EXPECTED = {"model-a": 0.855, "model-b": 0.723, "model-c": 0.509,
               "model-e": 0.750, "model-f": 0.100}

T2_WEIGHTS = {"capability": 0.10, "tool_reliability": 0.20, "cost": 0.30,
              "latency": 0.25, "context": 0.05, "safety": 0.10}
T2_EXPECTED = {"model-a": 0.502, "model-b": 0.669, "model-c": 0.701,
               "model-e": 0.487}


@pytest.fixture(scope="module")
def snapshot():
    return default_repository().get(SNAPSHOT)


def test_model_d_dominated_by_model_c(snapshot):
    rec = recommend(snapshot.models, TaskProfile(weights=T1_WEIGHTS),
                    snapshot.objectives)
    assert rec.stages.dominated == ["model-d"]
    assert sorted(rec.frontier) == ["model-a", "model-b", "model-c",
                                    "model-e", "model-f"]


def test_t1_compliance_profile_selects_model_a(snapshot):
    rec = recommend(snapshot.models, TaskProfile(weights=T1_WEIGHTS),
                    snapshot.objectives)
    assert rec.recommended == "model-a"
    scores = {s.model_id: s.score for s in rec.ranking}
    for mid, expected in T1_EXPECTED.items():
        assert scores[mid] == pytest.approx(expected, abs=5e-4)


def test_t2_triage_profile_selects_model_c_with_veto(snapshot):
    profile = TaskProfile(weights=T2_WEIGHTS,
                          veto={"tool_reliability": 0.85})
    rec = recommend(snapshot.models, profile, snapshot.objectives)
    assert rec.recommended == "model-c"
    assert rec.stages.vetoed == ["model-f"]
    scores = {s.model_id: s.score for s in rec.ranking}
    for mid, expected in T2_EXPECTED.items():
        assert scores[mid] == pytest.approx(expected, abs=5e-4)


def test_veto_excludes_f_for_any_weight_vector(snapshot):
    """The paper's non-compensatory guarantee: no weight choice can push a
    model below the reliability floor to the top."""
    extreme_cost_weights = {"capability": 0.0, "tool_reliability": 0.0,
                            "cost": 0.6, "latency": 0.4, "context": 0.0,
                            "safety": 0.0}
    profile = TaskProfile(weights=extreme_cost_weights,
                          veto={"tool_reliability": 0.85})
    rec = recommend(snapshot.models, profile, snapshot.objectives)
    assert "model-f" in rec.stages.vetoed
    assert rec.recommended != "model-f"
