"""Golden tests for the horizon-aware model: the code must reproduce the
numbers in the paper's motivation, variants and illustrative example
(Sections 3 and 4). If these fail, either the code or the paper is wrong;
find out which before touching the assertions.
"""
import math

import pytest

from whichone.core.horizon import (StepProfile, attempt_success, completion_probability,
                                   crossover_length, efficient_at, frontier_by_length,
                                   min_attempt_success, per_step_screen, per_success)

# Illustrative candidates (paper, Table "Illustrative candidate models").
MODELS = [
    StepProfile("A", 0.995, 0.400, 3.00, 2.7),
    StepProfile("B", 0.990, 0.220, 1.20, 2.1),
    StepProfile("C", 0.970, 0.060, 0.25, 1.5),
    StepProfile("D", 0.960, 0.075, 0.30, 1.8),
    StepProfile("E", 0.985, 0.320, 1.80, 3.6),
    StepProfile("F", 0.930, 0.030, 0.10, 0.9),
]


def test_motivation_twenty_steps():
    assert attempt_success(0.95, 20) == pytest.approx(0.36, abs=0.005)
    assert attempt_success(0.99, 20) == pytest.approx(0.82, abs=0.005)
    assert 1 / attempt_success(0.95, 20) == pytest.approx(2.8, abs=0.05)
    assert 1 / attempt_success(0.99, 20) == pytest.approx(1.2, abs=0.05)


def test_per_success_does_not_depend_on_retry_limit():
    s, r, length = 0.4, 2.0, 10
    for k in (1, 2, 5, 50):
        p = completion_probability(s, k)
        expected_attempts = (1 - (1 - s) ** k) / s
        assert r * length * expected_attempts / p == pytest.approx(per_success(r, length, s))


def test_crossovers_of_worked_pair():
    # 10% energy, 20% cost, 40% time per step; rho 0.95 vs 0.99.
    assert crossover_length(0.95, 0.99, 0.4, 1.0) == pytest.approx(22.2, abs=0.05)
    assert crossover_length(0.95, 0.99, 0.2, 1.0) == pytest.approx(39.0, abs=0.05)
    assert crossover_length(0.95, 0.99, 0.1, 1.0) == pytest.approx(55.8, abs=0.05)
    assert crossover_length(0.99, 0.95, 0.1, 1.0) is None
    assert crossover_length(0.0, 0.95, 0.1, 1.0) == 0.0  # a never completes a step


def test_chance_constraint_examples():
    assert min_attempt_success(0.9, 3) == pytest.approx(0.54, abs=0.005)
    assert min_attempt_success(0.95, 2) == pytest.approx(0.78, abs=0.005)
    # V2 with K = 1, beta = 0.9: feasible up to 2 and 10 steps.
    def max_len(rho):
        return max(L for L in range(1, 200) if attempt_success(rho, L) >= 0.9)
    assert max_len(0.95) == 2
    assert max_len(0.99) == 10


def test_stage1_prunes_d_and_e_for_every_length():
    kept, pruned = per_step_screen(MODELS)
    assert sorted(m.id for m in pruned) == ["D", "E"]
    assert sorted(m.id for m in kept) == ["A", "B", "C", "F"]


def test_frontier_by_length_matches_paper_table():
    got = frontier_by_length(MODELS, 250)
    assert got == [
        (1, 12, ("F",)),
        (13, 16, ("C", "F")),
        (17, 21, ("B", "C", "F")),
        (22, 49, ("B", "C")),
        (50, 76, ("A", "B", "C")),
        (77, 181, ("A", "B")),
        (182, 250, ("A",)),
    ]


def test_three_task_profiles():
    by_id = {m.id: m for m in MODELS}
    # Short triage task, free retries.
    assert efficient_at(MODELS, 3) == ["F"]
    f3 = per_success(by_id["F"].energy, 3, attempt_success(0.93, 3))
    a3 = per_success(by_id["A"].energy, 3, attempt_success(0.995, 3))
    assert f3 == pytest.approx(0.11, abs=0.005)
    assert a3 == pytest.approx(1.22, abs=0.005)
    # Customer-service task, 40 steps, free retries.
    assert efficient_at(MODELS, 40) == ["B", "C"]
    c40 = {k: per_success(by_id["C"].resource(k), 40, attempt_success(0.97, 40))
           for k in ("energy", "cost", "time")}
    assert c40["energy"] == pytest.approx(8.1, abs=0.05)
    assert c40["cost"] == pytest.approx(34, abs=0.5)
    assert c40["time"] == pytest.approx(203, abs=0.5)
    f40 = per_success(by_id["F"].energy, 40, attempt_success(0.93, 40))
    assert f40 == pytest.approx(21.9, abs=0.05)
    # Same task with an irreversible step: K = 2, beta = 0.95.
    assert efficient_at(MODELS, 40, beta=0.95, max_attempts=2) == ["A"]
    a40 = per_success(by_id["A"].energy, 40, attempt_success(0.995, 40))
    assert a40 == pytest.approx(19.6, abs=0.05)


def test_unlimited_retries_always_feasible():
    assert completion_probability(0.01, math.inf) == 1.0
    assert min_attempt_success(0.99, math.inf) == 0.0


def test_stage2_example_at_sixty_steps():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
    from reproduce import stage2
    st = stage2()
    assert st["efficient"] == ["A", "B", "C"]
    tops = {row["label"]: (row["saw_top"], row["topsis_top"]) for row in st["weights"]}
    assert tops["Equal"] == ("B", "B")
    assert tops["Energy and cost"] == ("C", "C")
    assert tops["Energy, little cost"] == ("B", "C")
    assert st["simplex"]["agree_share"] == pytest.approx(0.973, abs=0.001)
