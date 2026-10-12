"""Reproduce every computed number, table and figure in the paper's motivation
and framework sections (Sections 3 and 4). No API keys or network access needed.

Usage: python paper/reproduce.py [--out paper/output]

Outputs:
  numbers.json                   every number quoted in the text
  table_frontier_by_length.tex   Table "Efficient models per successful task, by task length"
  fig_parallel_lines.pdf         per-success log-ratio lines of the worked pair (vector)

The same numbers are asserted in tests/test_horizon.py, so `pytest` fails if
the code and the paper disagree.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from whichone.core.horizon import (StepProfile, attempt_success, crossover_length,
                                   efficient_at, frontier_by_length, min_attempt_success,
                                   per_step_screen, per_success)

HERE = Path(__file__).resolve().parent

# Paper, Table "Illustrative candidate models" (per step: rho, Wh, US cents, seconds).
MODELS = [
    StepProfile("A", 0.995, 0.400, 3.00, 2.7),
    StepProfile("B", 0.990, 0.220, 1.20, 2.1),
    StepProfile("C", 0.970, 0.060, 0.25, 1.5),
    StepProfile("D", 0.960, 0.075, 0.30, 1.8),
    StepProfile("E", 0.985, 0.320, 1.80, 3.6),
    StepProfile("F", 0.930, 0.030, 0.10, 0.9),
]
# Worked pair: per step, a uses 10% energy, 20% cost, 40% time of b.
PAIR = {"rho_a": 0.95, "rho_b": 0.99, "ratio": {"energy": 0.1, "cost": 0.2, "time": 0.4}}


def numbers() -> dict:
    n: dict = {}
    s20a, s20b = attempt_success(0.95, 20), attempt_success(0.99, 20)
    n["motivation"] = {"S_a_20": s20a, "S_b_20": s20b,
                       "attempts_a": 1 / s20a, "attempts_b": 1 / s20b,
                       "energy_advantage_at_20": 10 / ((0.99 / 0.95) ** 20)}
    n["crossovers"] = {k: crossover_length(PAIR["rho_a"], PAIR["rho_b"], r, 1.0)
                       for k, r in PAIR["ratio"].items()}
    n["chance_constraint"] = {
        "S_min_beta0.9_K3": min_attempt_success(0.9, 3),
        "S_min_beta0.95_K2": min_attempt_success(0.95, 2),
        "max_L_rho0.95_K1_beta0.9": max(L for L in range(1, 500) if attempt_success(0.95, L) >= 0.9),
        "max_L_rho0.99_K1_beta0.9": max(L for L in range(1, 500) if attempt_success(0.99, L) >= 0.9),
    }
    kept, pruned = per_step_screen(MODELS)
    n["stage1"] = {"kept": [m.id for m in kept], "pruned": [m.id for m in pruned]}
    n["frontier_by_length"] = frontier_by_length(MODELS, 250)
    by = {m.id: m for m in MODELS}

    def ps(mid, L):
        m = by[mid]
        s = attempt_success(m.reliability, L)
        return {k: per_success(m.resource(k), L, s) for k in ("energy", "cost", "time")}

    n["profiles"] = {
        "triage_L3": {"efficient": efficient_at(MODELS, 3), "F": ps("F", 3), "A": ps("A", 3)},
        "service_L40": {"efficient": efficient_at(MODELS, 40), "B": ps("B", 40), "C": ps("C", 40),
                        "F": ps("F", 40)},
        "service_L40_irreversible_K2_beta0.95": {
            "efficient": efficient_at(MODELS, 40, beta=0.95, max_attempts=2), "A": ps("A", 40)},
    }
    return n


def frontier_table(rows) -> str:
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Task length $L$ & Efficient models \\", r"\midrule"]
    for i, (lo, hi, ids) in enumerate(rows):
        last = i == len(rows) - 1
        span = f"{lo} or more" if last else (f"{lo} to {hi}" if lo != hi else f"{lo}")
        lines.append(f"{span} & {', '.join(ids)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra, rb = PAIR["rho_a"], PAIR["rho_b"]
    Ls = list(range(0, 81))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    styles = {"time": "-", "cost": "--", "energy": ":"}
    cross = {}
    for k, r in PAIR["ratio"].items():
        ax.plot(Ls, [math.log(r) + L * math.log(rb / ra) for L in Ls], styles[k], color="black", label=k)
        cross[k] = crossover_length(ra, rb, r, 1.0)
    lo, hi = min(cross.values()), max(cross.values())
    ax.axvspan(lo, hi, color="0.9")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Task length $L$ (steps)")
    ax.set_ylabel(r"$\ln\,(R_a / R_b)$ per successful task")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "output"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = numbers()
    (out / "numbers.json").write_text(json.dumps(n, indent=1))
    (out / "table_frontier_by_length.tex").write_text(frontier_table(n["frontier_by_length"]))
    figure(out / "fig_parallel_lines.pdf")
    print(f"wrote numbers.json, table_frontier_by_length.tex, fig_parallel_lines.pdf to {out}")


if __name__ == "__main__":
    main()
