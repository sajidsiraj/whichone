"""Reproduce every computed number, table and figure in the paper's motivation
and framework sections (Sections 3 and 4). No API keys or network access needed.

Usage: python paper/reproduce.py [--out paper/output]

Outputs:
  numbers.json                   every number quoted in the text
  table_frontier_by_length.tex   Table "Efficient models per successful task, by task length"
  table_stage2_matrix.tex        Stage 2 example: values per successful task at L = 60, rescaled
  table_stage2_scores.tex        Stage 2 example: SAW and TOPSIS scores for several sets of weights
  fig_parallel_lines.pdf         per-success log-ratio lines of models C and B (vector)

The same numbers are asserted in tests/test_horizon.py, so `pytest` fails if
the code and the paper disagree.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from whichone.core.horizon import (_PER_SUCCESS_SPECS, StepProfile, _as_offering,
                                   attempt_success, crossover_length, efficient_at,
                                   frontier_by_length, min_attempt_success, per_step_screen,
                                   per_success)
from whichone.core.madm import normalise, rank, saw, topsis

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
RESOURCES = ("energy", "cost", "time")
# Stage 2 example: the one interval of Table "Efficient models" with three
# efficient models, and weight sets (energy, cost, time) that a deployment might state.
STAGE2_L = 60
STAGE2_WEIGHTS = [("Equal", (1 / 3, 1 / 3, 1 / 3)),
                  ("Energy first", (0.6, 0.2, 0.2)),
                  ("Speed first", (0.2, 0.2, 0.6)),
                  ("Energy and cost", (0.45, 0.45, 0.10)),
                  ("Energy, little cost", (0.80, 0.04, 0.16))]
SIMPLEX_STEP = 0.01

# Motivation pair (Section 3): per step, a uses 10% energy, 20% cost, 40% time of b.
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
    n["stage2"] = stage2()
    return n


def stage2() -> dict:
    """Stage 2 at STAGE2_L: the decision matrix of the efficient models, the
    values rescaled within E(L), SAW and TOPSIS scores for each weight set,
    and how the top choice varies over a grid of all weights."""
    eff = efficient_at(MODELS, STAGE2_L)
    offs = [_as_offering(m, STAGE2_L) for m in MODELS if m.id in eff]
    norm = normalise(offs, _PER_SUCCESS_SPECS)
    out = {"L": STAGE2_L, "efficient": eff,
           "per_success": {o.id: {k: o.raw(k) for k in RESOURCES} for o in offs},
           "normalised": norm, "weights": []}
    for label, w in STAGE2_WEIGHTS:
        wd = dict(zip(RESOURCES, w))
        sa, to = saw(norm, wd), topsis(norm, wd)
        out["weights"].append({"label": label, "w": w, "saw": sa, "topsis": to,
                               "saw_top": rank(sa)[0], "topsis_top": rank(to)[0]})
    n = round(1 / SIMPLEX_STEP)
    tops = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            wd = dict(zip(RESOURCES, (i / n, j / n, (n - i - j) / n)))
            tops.append((rank(saw(norm, wd))[0], rank(topsis(norm, wd))[0]))
    share = lambda f: sum(1 for t in tops if f(t)) / len(tops)
    out["simplex"] = {"points": len(tops),
                      "saw_top_share": {m: share(lambda t, m=m: t[0] == m) for m in eff},
                      "topsis_top_share": {m: share(lambda t, m=m: t[1] == m) for m in eff},
                      "agree_share": share(lambda t: t[0] == t[1])}
    return out


def stage2_tables(st: dict) -> tuple[str, str]:
    ids = st["efficient"]
    m = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & Energy (Wh) & Cost (cents) & Time (s) & $v_e$ & $v_c$ & $v_t$ \\", r"\midrule"]
    for mid in ids:
        r, v = st["per_success"][mid], st["normalised"][mid]
        m.append(f"{mid} & {r['energy']:.1f} & {r['cost']:.0f} & {r['time']:.0f} & "
                 f"{v['energy']:.2f} & {v['cost']:.2f} & {v['time']:.2f} \\\\")
    m += [r"\bottomrule", r"\end{tabular}"]
    head = " & ".join(ids)
    s = [r"\begin{tabular}{lc" + "r" * len(ids) + "c" + "r" * len(ids) + "c}", r"\toprule",
         r"Weights & $(w_e, w_c, w_t)$ & \multicolumn{" + str(len(ids)) + r"}{c}{SAW} & & \multicolumn{"
         + str(len(ids)) + r"}{c}{TOPSIS} & Top choice \\",
         r" & & " + head + " & & " + head + r" & SAW / TOPSIS \\", r"\midrule"]
    for row in st["weights"]:
        w = ", ".join(f"{x:.2f}" for x in row["w"])
        sa = " & ".join(f"{row['saw'][i]:.2f}" for i in ids)
        to = " & ".join(f"{row['topsis'][i]:.2f}" for i in ids)
        s.append(f"{row['label']} & ({w}) & {sa} & & {to} & {row['saw_top']} / {row['topsis_top']} \\\\")
    s += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(m) + "\n", "\n".join(s) + "\n"


def frontier_table(rows) -> str:
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Task length $L$ & Efficient models \\", r"\midrule"]
    for i, (lo, hi, ids) in enumerate(rows):
        last = i == len(rows) - 1
        span = f"{lo} or more" if last else (f"{lo} to {hi}" if lo != hi else f"{lo}")
        lines.append(f"{span} & {', '.join(ids)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure(path: Path):
    """Parallel log-ratio lines for models C (smaller) against B (larger) of
    the example: ln(R_C / R_B) per successful task for each resource. Each
    line crosses zero at that resource's crossover; the band between the first
    and last crossing is shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {m.id: m for m in MODELS}
    a, b = by["C"], by["B"]
    Ls = list(range(0, 101))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    styles = {"time": "-", "energy": ":", "cost": "--"}
    cross = {}
    for k, ls in styles.items():
        r = a.resource(k) / b.resource(k)
        ax.plot(Ls, [math.log(r) + L * math.log(b.reliability / a.reliability) for L in Ls],
                linestyle=ls, color="black", label=k)
        cross[k] = crossover_length(a.reliability, b.reliability, a.resource(k), b.resource(k))
    ax.axvspan(min(cross.values()), max(cross.values()), color="0.9")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task length $L$ (steps)")
    ax.set_ylabel(r"$\ln\,(R_C / R_B)$ per successful task")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    return cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "output"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = numbers()
    (out / "table_frontier_by_length.tex").write_text(frontier_table(n["frontier_by_length"]))
    matrix, scores = stage2_tables(n["stage2"])
    (out / "table_stage2_matrix.tex").write_text(matrix)
    (out / "table_stage2_scores.tex").write_text(scores)
    n["figure_crossovers_C_vs_B"] = figure(out / "fig_parallel_lines.pdf")
    (out / "numbers.json").write_text(json.dumps(n, indent=1))
    print(f"wrote numbers.json, table_frontier_by_length.tex, table_stage2_*.tex, "
          f"fig_parallel_lines.pdf to {out}")


if __name__ == "__main__":
    main()
