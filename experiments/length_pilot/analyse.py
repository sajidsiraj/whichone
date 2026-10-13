"""Analyse pilot logs: success curves, per-step reliability, resources per
successful task, predicted and observed crossovers, single-crossing check (H1).

Every pilot number in the paper is produced by this script from the raw logs in
results/, so the analysis can be repeated exactly without calling any API.

Usage:
  python experiments/length_pilot/analyse.py [--results DIR] [--out DIR] [--boot 1000]

Inputs (all under version control):
  results/*.jsonl  raw run logs written by run.py
  prices.json      {"model": [usd_per_M_input, usd_per_M_output]}, used when a
                   provider does not report billed cost
  energy.json      {"model": {"j_per_output_token": x, "source": "..."}}:
                   third-party estimates for API models, reported as estimates.
                   Local runs carry measured GPU energy per step and use that.
Outputs:
  output/summary.json      all numbers quoted in the paper
  output/table_pilot.tex   per-model table
  output/table_pilot_crossovers.tex
  output/fig_pilot_energy.pdf  energy per successful task against task length

Three crossover lengths are reported for each pair (a less reliable than b):
  observed     the shortest L from which b uses less per successful task than a
               at every longer length, from the empirical success curves
  closed_form  ln(r_b/r_a) / ln(rho_b/rho_a), with r the mean per-step resource
               over all steps: the paper's formula, which assumes r is constant
  fitted       the same rule applied to the measured cumulative resource C(L)
               and the fitted S(L) = rho^L, so it allows r to grow with context
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from whichone.core.horizon import crossover_length

HERE = Path(__file__).resolve().parent
RESOURCES = ("energy", "cost", "time")
REPORT_L = (5, 10, 20, 50, 100, 150)


def load(results: Path):
    runs = defaultdict(list)
    for p in sorted(results.glob("*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            runs[f"{r['model']}__d{r.get('difficulty', 1)}"].append(r)
    return runs


def survival(rs, lmax):
    n = len(rs)
    return [sum(1 for r in rs if (r["first_fail"] or lmax + 1) > L) / n for L in range(1, lmax + 1)]


def rho_hat(rs, lmax):
    """Maximum-likelihood per-step reliability from first-failure steps."""
    fails = sum(1 for r in rs if r["first_fail"])
    ok = sum((r["first_fail"] - 1) if r["first_fail"] else lmax for r in rs)
    return ok / (ok + fails) if ok + fails else float("nan")


def step_resources(rs, lmax, prices, energy, model):
    """Mean resource used at step j, over runs (j = 1..lmax)."""
    sums = {k: [0.0] * lmax for k in RESOURCES}
    pin, pout = prices.get(model, (None, None))
    e = (energy.get(model) or {}).get("j_per_output_token")
    for r in rs:
        for s in r["steps"][:lmax]:
            j = s["step"] - 1
            if s.get("cost") is not None:
                c = s["cost"]
            elif pin is not None:
                c = (s["prompt_tokens"] * pin + s["completion_tokens"] * pout) / 1e6
            else:
                c = float("nan")
            sums["cost"][j] += c
            sums["time"][j] += s["provider_time"] if s.get("provider_time") is not None else s["wall_time"]
            if s.get("gpu_energy_j") is not None:  # measured on the local GPU
                sums["energy"][j] += s["gpu_energy_j"]
            else:  # third-party estimate per output token, flagged as such
                sums["energy"][j] += s["completion_tokens"] * e if e is not None else float("nan")
    n = len(rs)
    return {k: [x / n for x in v] for k, v in sums.items()}


def cumulative(xs):
    out, c = [], 0.0
    for x in xs:
        c += x
        out.append(c)
    return out


def per_success_curve(step_means, S):
    return [c / s if s > 0 else math.inf for c, s in zip(cumulative(step_means), S)]


def summarise(rs, lmax, prices, energy, model):
    S = survival(rs, lmax)
    rho = rho_hat(rs, lmax)
    steps = step_resources(rs, lmax, prices, energy, model)
    fit = [rho ** L for L in range(1, lmax + 1)]
    return {"n_runs": len(rs), "rho_hat": rho, "S": S,
            "first_failures": sorted(r["first_fail"] for r in rs if r["first_fail"]),
            "max_abs_S_minus_fit": max(abs(a - b) for a, b in zip(S, fit)),
            "per_step_mean": {k: sum(v) / len(v) for k, v in steps.items()},
            "per_step_by_band": {k: {f"{a + 1}-{b}": sum(v[a:b]) / len(v[a:b])
                                     for a, b in ((0, 10), (10, 50), (50, 100), (100, lmax)) if b <= lmax}
                                 for k, v in steps.items()},
            "cumulative": {k: cumulative(v) for k, v in steps.items()},
            "per_success": {k: per_success_curve(v, S) for k, v in steps.items()},
            "per_success_fitted": {k: per_success_curve(v, fit) for k, v in steps.items()}}


def switch_length(ra, rb):
    """Shortest L from which b's curve is below a's at every longer length.
    Lengths where both are infinite carry no information and are skipped.
    None if b is not below a at the last informative length."""
    last = None
    for L in range(len(ra), 0, -1):
        x, y = ra[L - 1], rb[L - 1]
        if math.isinf(x) and math.isinf(y):
            continue
        if x > y:
            last = L
        else:
            break
    return last


def crossovers(A, B, k):
    xa, xb = A["per_step_mean"][k], B["per_step_mean"][k]
    if math.isnan(xa) or math.isnan(xb):
        return None
    return {"per_step_ratio": xa / xb,
            "closed_form": crossover_length(A["rho_hat"], B["rho_hat"], xa, xb),
            "fitted": switch_length(A["per_success_fitted"][k], B["per_success_fitted"][k]),
            "observed": switch_length(A["per_success"][k], B["per_success"][k]),
            "sign_changes": sign_changes(A["per_success"][k], B["per_success"][k])}


def sign_changes(ra, rb):
    """Number of times the cheaper model per successful task changes as L grows
    (H1 predicts at most one). Ties and doubly infinite lengths are skipped."""
    signs = [x > y for x, y in zip(ra, rb) if x != y and not (math.isinf(x) and math.isinf(y))]
    return sum(1 for u, v in zip(signs, signs[1:]) if u != v)


def quantiles(xs, n):
    """2.5% and 97.5% points, and the share of resamples with no crossover."""
    have = sorted(x for x in xs if x is not None)
    if not have:
        return {"ci95": None, "share_none": 1.0}
    q = lambda p: have[min(len(have) - 1, int(p * len(have)))]
    return {"ci95": [q(0.025), q(0.975)], "share_none": 1 - len(have) / n}


def bootstrap(runs, keys, lmax, prices, energy, n, rng):
    """Resample runs within each model; return CI for rho and pair crossovers."""
    rho = defaultdict(list)
    cross = defaultdict(list)
    for _ in range(n):
        s = {}
        for key in keys:
            rs = runs[key]
            s[key] = summarise([rng.choice(rs) for _ in rs], lmax, prices, energy, rs[0]["model"])
            rho[key].append(s[key]["rho_hat"])
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                for k in RESOURCES:
                    c = crossovers(s[ka], s[kb], k)
                    if c is None:
                        continue
                    for m in ("closed_form", "fitted", "observed"):
                        cross[(ka, kb, k, m)].append(c[m])
    rho_ci = {k: quantiles(v, n)["ci95"] for k, v in rho.items()}
    return rho_ci, {key: quantiles(v, n) for key, v in cross.items()}


def fmt_L(x):
    return "--" if x is None else (f"{x:.0f}" if not math.isinf(x) else r"$\infty$")


def ci(b):
    if not b or not b["ci95"]:
        return "--"
    lo, hi = b["ci95"]
    return f"{lo}--{hi}" + (f" ({b['share_none'] * 100:.0f}\\% none)" if b["share_none"] > 0.025 else "")


def display_name(key):
    """Model name as printed in the paper: qwen3:1.7b -> Qwen3 1.7B."""
    model = key.split("__")[0]
    m = re.fullmatch(r"([a-z]+[0-9.]*)[:-]([0-9.]+b)", model)
    if not m:  # other names are printed as logged
        return model.replace("_", r"\_")
    return f"{m.group(1).capitalize()} {m.group(2).upper()}"


def tables(summ, keys, pairs, out: Path, lmax):
    """Tables are written without float or caption, so the paper can
    \\input them inside its own table environment."""
    name = display_name
    size = lambda k: summ[k]["per_step_mean"]["energy"]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & $\hat\rho$ & 95\% CI & J/step & s/step & \multicolumn{3}{c}{kJ per successful task} \\",
             r" & & & & & $L=10$ & $L=20$ & $L=50$ \\", r"\midrule"]
    for k in sorted(keys, key=lambda k: summ[k]["per_step_mean"]["energy"]):
        m = summ[k]
        lo, hi = m["rho_ci95"]
        e = [m["per_success"]["energy"][L - 1] / 1000 for L in (10, 20, 50)]
        lines.append(f"{name(k)} & {m['rho_hat']:.4f} & {lo:.4f}--{hi:.4f} & "
                     f"{m['per_step_mean']['energy']:.0f} & {m['per_step_mean']['time']:.2f} & "
                     + " & ".join(r"$\infty$" if math.isinf(x) else f"{x:.1f}" for x in e) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_pilot.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrrrrc}", r"\toprule",
             r"Smaller model & Larger model & Energy ratio & \multicolumn{4}{c}{Energy crossover $L^\ast$ (steps)} \\",
             r" & & per step & closed form & fitted & observed & observed 95\% CI \\", r"\midrule"]
    for p in sorted(pairs, key=lambda p: (size(p["a"]), size(p["b"]))):
        c = p["resources"].get("energy")
        if not c or p["status"] != "crossover":
            continue
        lines.append(f"{name(p['a'])} & {name(p['b'])} & {c['per_step_ratio']:.2f} & "
                     f"{fmt_L(c['closed_form'])} & {fmt_L(c['fitted'])} & {fmt_L(c['observed'])} & {ci(c['observed_boot'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_pilot_crossovers.tex").write_text("\n".join(lines) + "\n")


def figure(summ, keys, path: Path, lmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
    Ls = list(range(1, lmax + 1))
    for k, ls in zip(sorted(keys, key=lambda k: summ[k]["per_step_mean"]["energy"]), styles):
        m = summ[k]
        obs = [x / 1000 if not math.isinf(x) else float("nan") for x in m["per_success"]["energy"]]
        fit = [x / 1000 for x in m["per_success_fitted"]["energy"]]
        line, = ax.plot(Ls, fit, linestyle=ls, color="black", linewidth=1, label=display_name(k))
        ax.plot(Ls[4::5], obs[4::5], "o", markersize=2.5, color=line.get_color(), alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_xlabel("Task length $L$ (steps)")
    ax.set_ylabel("GPU energy per successful task (kJ)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    prices = json.loads((HERE / "prices.json").read_text()) if (HERE / "prices.json").exists() else {}
    energy = json.loads((HERE / "energy.json").read_text()) if (HERE / "energy.json").exists() else {}
    runs = load(Path(a.results))
    if not runs:
        raise SystemExit(f"No run logs in {a.results}")
    lmax = min(r["lmax"] for rs in runs.values() for r in rs)
    rng = random.Random(a.seed)

    summ = {key: summarise(rs, lmax, prices, energy, rs[0]["model"]) for key, rs in runs.items()}
    # order by reliability, ties by per-step energy, so a is always the less reliable
    keys = sorted(summ, key=lambda k: (summ[k]["rho_hat"], summ[k]["per_step_mean"]["energy"]))
    rho_ci, cross_ci = bootstrap(runs, keys, lmax, prices, energy, a.boot, rng)
    for key in keys:
        rs, m = runs[key], summ[key]
        m["rho_ci95"] = rho_ci[key]
        measured = all(s.get("gpu_energy_j") is not None for r in rs for s in r["steps"])
        m["energy_source"] = "measured (GPU board, NVML)" if measured else (
            "estimate (energy.json)" if rs[0]["model"] in energy else "none")
        m["at"] = {L: {"S": m["S"][L - 1], **{f"{k}_per_success": m["per_success"][k][L - 1]
                                              for k in RESOURCES}} for L in REPORT_L if L <= lmax}
        print(f"{key}: runs={len(rs)} rho_hat={m['rho_hat']:.4f} CI95={tuple(round(x, 4) for x in m['rho_ci95'])} "
              f"J/step={m['per_step_mean']['energy']:.1f} s/step={m['per_step_mean']['time']:.3f}")

    pairs = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            A, B = summ[ka], summ[kb]
            pair = {"a": ka, "b": kb, "resources": {}}
            for k in RESOURCES:
                c = crossovers(A, B, k)
                if c is None:
                    continue
                for m in ("closed_form", "fitted", "observed"):
                    c[m + "_boot"] = cross_ci.get((ka, kb, k, m))
                pair["resources"][k] = c
            e = pair["resources"]["energy"]
            # a closed-form crossover below one step means b is better at every length
            pair["status"] = ("crossover" if e["closed_form"] is not None and e["closed_form"] >= 1
                              else "b dominates")
            pairs.append(pair)
            print(f"  {ka} vs {kb} [{pair['status']}]: " + "; ".join(
                f"{k} ratio={v['per_step_ratio']:.2f} closed={fmt_L(v['closed_form'])} "
                f"fitted={fmt_L(v['fitted'])} obs={fmt_L(v['observed'])} changes={v['sign_changes']} "
                f"obsCI={v['observed_boot'] and v['observed_boot']['ci95']}"
                for k, v in pair["resources"].items() if k != "cost"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slim = {k: {x: y for x, y in v.items() if x not in ("cumulative", "per_success", "per_success_fitted", "S")}
            for k, v in summ.items()}
    for k in slim:
        slim[k]["S"] = summ[k]["S"]
        slim[k]["energy_per_success"] = summ[k]["per_success"]["energy"]
    (out / "summary.json").write_text(json.dumps(
        {"lmax": lmax, "bootstrap": a.boot, "seed": a.seed, "order": keys, "models": slim, "pairs": pairs},
        indent=1))
    tables(summ, keys, pairs, out, lmax)
    if not a.no_figure:
        figure(summ, keys, out / "fig_pilot_energy.pdf", lmax)
    print(f"wrote summary.json, table_pilot.tex, table_pilot_crossovers.tex, fig_pilot_energy.pdf to {out}")


if __name__ == "__main__":
    main()
