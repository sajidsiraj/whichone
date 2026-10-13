"""Golden tests: the engine behind the API must reproduce the paper's
illustrative example (Section 4, "An illustrative example") from the bundled
snapshot. If these fail, either the code or the paper is wrong; find out
which before touching the assertions.
"""
import pytest

from whichone.catalogue import default_repository
from whichone.core import Constraints, TaskProfile, frontier_intervals, recommend

SNAPSHOT = "2026-10-13-paper-example"
CARBON = {"energy": 0.6, "cost": 0.2, "time": 0.2}
SPEED = {"energy": 0.1, "cost": 0.1, "time": 0.8}


@pytest.fixture(scope="module")
def snap():
    return default_repository().get(SNAPSHOT)


def run(snap, **kw):
    return recommend(snap.models, TaskProfile(**kw), snap.objectives)


def ids(*letters):
    return [f"model-{x}" for x in letters]


def test_stage1_prunes_d_and_e_for_every_length(snap):
    rec = run(snap, length=1)
    assert rec.stages.dominated_per_step == ids("d", "e")


def test_efficient_sets_by_length_match_paper_table(snap):
    kept = [m for m in snap.models if m.id not in ids("d", "e")]
    rows = frontier_intervals(kept, snap.objectives, 250)
    assert [(a, b, [x[-1].upper() for x in e]) for a, b, e in rows] == [
        (1, 12, ["F"]), (13, 16, ["C", "F"]), (17, 21, ["B", "C", "F"]),
        (22, 49, ["B", "C"]), (50, 76, ["A", "B", "C"]), (77, 181, ["A", "B"]),
        (182, 250, ["A"])]


def test_short_triage_task_needs_no_weights(snap):
    rec = run(snap, length=3)
    assert rec.efficient == ids("f")
    assert rec.recommended == "model-f"
    assert not rec.preferences_needed
    assert rec.per_success["model-f"]["energy"] == pytest.approx(0.11, abs=0.005)
    assert rec.per_success["model-a"]["energy"] == pytest.approx(1.22, abs=0.005)


def test_service_task_is_a_trade_off(snap):
    rec = run(snap, length=40)
    assert sorted(rec.efficient) == ids("b", "c")
    assert rec.preferences_needed and rec.recommended is None
    c, b, f = (rec.per_success[f"model-{x}"] for x in "cbf")
    assert (c["energy"], c["cost"], c["time"]) == pytest.approx((8.1, 34, 203), rel=0.01)
    assert (b["energy"], b["cost"], b["time"]) == pytest.approx((13.2, 72, 126), rel=0.01)
    assert f["energy"] == pytest.approx(21.9, rel=0.01)


def test_weights_decide_only_inside_the_band(snap):
    assert run(snap, length=40, weights=CARBON).recommended == "model-c"
    assert run(snap, length=40, weights=SPEED).recommended == "model-b"
    # outside the band the weights cannot change the answer
    assert run(snap, length=3, weights=SPEED).recommended == "model-f"


def test_irreversible_payment_needs_model_a(snap):
    rec = run(snap, length=40, max_attempts=2, completion_probability=0.95, weights=CARBON)
    assert rec.efficient == ids("a")
    assert rec.recommended == "model-a"
    assert sorted(rec.stages.below_completion) == ids("b", "c", "f")
    assert rec.per_success["model-a"]["energy"] == pytest.approx(19.6, rel=0.01)


def test_no_feasible_model_says_so(snap):
    rec = run(snap, length=200, max_attempts=1, completion_probability=0.99)
    assert rec.recommended is None and rec.efficient == []
    assert "changing the task" in rec.note


def test_large_context_brings_back_e(snap):
    """Stage 0 then removes B, C, D and F; A does not dominate E, which is
    cheaper per step on energy and cost."""
    rec = run(snap, length=10, constraints=Constraints(min_context=201))
    assert sorted(rec.stages.infeasible) == ids("b", "c", "d", "f")
    assert rec.stages.dominated_per_step == []
    assert sorted(rec.efficient) == ids("a", "e")
