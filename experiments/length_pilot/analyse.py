"""Analyse pilot logs: success curves, per-step reliability, resources per
successful task, predicted and observed crossovers, single-crossing check (H1).

Every pilot number in the paper is produced by this script from the raw logs in
results/, so the analysis can be repeated exactly without calling any API.

Usage:
  python experiments/length_pilot/analyse.py [--task d0|tools] [--results DIR] [--out DIR] [--boot 1000]

Each task family is analysed on its own. The order task (d0) writes the file
names below; any other task adds its name, e.g. table_pilot_tools.tex.

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
  output/table_pilot_crossovers_all.tex  every pair that survives the per-step screen
  output/fig_pilot_energy.pdf  energy per successful task against task length
  output/fig_pilot_models.pdf  all models on energy per step and steps before an error

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
            task = "tools" if r.get("task") == "tools" else f"d{r.get('difficulty', 1)}"
            runs[f"{r['model']}__{task}"].append(r)
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
                                     for a, b in ((0, 10), (10, 50), (50, 100), (100, lmax)) if a < b <= lmax}
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


FAMILIES = {"qwen3": "Qwen3", "qwen3.8": "Qwen3.8", "gemma3": "Gemma 3",
            "llama3.1": "Llama 3.1", "llama3.2": "Llama 3.2",
            "ministral-3": "Ministral 3", "glm4": "GLM-4"}


def display_name(key):
    """Model name as printed in the paper: qwen3:1.7b -> Qwen3 1.7B,
    ministral-3:8b -> Ministral 3 8B. Unknown names are printed as logged."""
    model = key.split("__")[0]
    m = re.fullmatch(r"(.+?)[:-]([0-9.]+b)", model)
    if not m or m.group(1) not in FAMILIES:
        return model.replace("_", r"\_")
    return f"{FAMILIES[m.group(1)]} {m.group(2).upper()}"


def family(key):
    """Developer family of a model, for grouping in tables and figures."""
    name = display_name(key)
    return name.rsplit(" ", 1)[0] if " " in name else name


def per_step_dominated(summ, keys):
    """Models another model beats per step: at least as reliable and using no
    more energy or time per step, better on one (cost is not measured locally).
    Such a model is worse at every task length (Observation 2)."""
    q = lambda k: (-summ[k]["rho_hat"], summ[k]["per_step_mean"]["energy"],
                   summ[k]["per_step_mean"]["time"])
    beats = lambda a, b: all(x <= y for x, y in zip(q(a), q(b))) and q(a) != q(b)
    return {k: sorted(o for o in keys if beats(o, k)) for k in keys
            if any(beats(o, k) for o in keys)}


def shown_models(summ, keys, lmax):
    """Models drawn in the length figure and listed in the paper's crossover
    table: each model that uses the least energy per successful task at some
    length (fitted curve), and the most reliable model as a reference."""
    shown = {max(keys, key=lambda k: summ[k]["rho_hat"])}
    for L in range(1, lmax + 1):
        shown.add(min(keys, key=lambda k: summ[k]["per_success_fitted"]["energy"][L - 1]))
    return [k for k in keys if k in shown]


def cheapest_by_length(summ, keys, lmax, curve="per_success_fitted"):
    """Intervals of task length (first L, last L, model) over which one model
    uses the least energy per successful task. Lengths where every model has
    infinite energy per success (no successful run) are left out."""
    out = []
    for L in range(1, lmax + 1):
        k = min(keys, key=lambda k: summ[k][curve]["energy"][L - 1])
        if math.isinf(summ[k][curve]["energy"][L - 1]):
            continue
        if out and out[-1][2] == k and out[-1][1] == L - 1:
            out[-1] = (out[-1][0], L, k)
        else:
            out.append((L, L, k))
    return out


def crossover_table(pairs, name, size, path: Path):
    lines = [r"\begin{tabular}{llrrrrc}", r"\toprule",
             r"Less reliable model & More reliable model & Energy ratio & \multicolumn{4}{c}{Energy crossover $L^\ast$ (steps)} \\",
             r" & & per step & closed form & fitted & observed & observed 95\% CI \\", r"\midrule"]
    for p in sorted(pairs, key=lambda p: (size(p["a"]), size(p["b"]))):
        c = p["resources"].get("energy")
        if not c or p["status"] != "crossover":
            continue
        lines.append(f"{name(p['a'])} & {name(p['b'])} & {c['per_step_ratio']:.2f} & "
                     f"{fmt_L(c['closed_form'])} & {fmt_L(c['fitted'])} & {fmt_L(c['observed'])} & {ci(c['observed_boot'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def tables(summ, keys, pairs, out: Path, lmax, shown, dominated, sfx=""):
    """Tables are written without float or caption, so the paper can
    \\input them inside its own table environment."""
    name = display_name
    size = lambda k: summ[k]["per_step_mean"]["energy"]
    Ls = [L for L in (10, 20, 50) if L <= lmax]
    lines = [r"\begin{tabular}{lrrrr" + "r" * len(Ls) + "}", r"\toprule",
             r"Model & $\hat\rho$ & 95\% CI & J/step & s/step & \multicolumn{" + str(len(Ls))
             + r"}{c}{kJ per successful task} \\",
             r" & & & & & " + " & ".join(f"$L={L}$" for L in Ls) + r" \\", r"\midrule"]
    for k in sorted(keys, key=lambda k: summ[k]["per_step_mean"]["energy"]):
        m = summ[k]
        lo, hi = m["rho_ci95"]
        e = [m["per_success"]["energy"][L - 1] / 1000 for L in Ls]
        mark = r"$^\dagger$" if k in dominated else ""
        lines.append(f"{name(k)}{mark} & {m['rho_hat']:.4f} & {lo:.4f}--{hi:.4f} & "
                     f"{m['per_step_mean']['energy']:.0f} & {m['per_step_mean']['time']:.2f} & "
                     + " & ".join(r"$\infty$" if math.isinf(x) else f"{x:.1f}" for x in e) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"table_pilot{sfx}.tex").write_text("\n".join(lines) + "\n")

    crossover_table([p for p in pairs if p["a"] in shown and p["b"] in shown], name, size,
                    out / f"table_pilot_crossovers{sfx}.tex")
    # every pair of models that survive the per-step screen, for the appendix
    crossover_table([p for p in pairs if p["a"] not in dominated and p["b"] not in dominated],
                    name, size, out / f"table_pilot_crossovers_all{sfx}.tex")


MIN_SUCCESSES = 3  # observed points from fewer successful runs are drawn hollow


def figure(summ, keys, path: Path, lmax):
    """Energy per successful task against L: fitted lines and observed points,
    one colour and marker per model (Okabe-Ito colours, readable in grey)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (3, 1)), (0, (1, 2))]
    if len(keys) > len(colours):
        raise ValueError(f"{len(keys)} models to draw but only {len(colours)} styles")
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    Ls = list(range(1, lmax + 1))
    for k, col, mk, ls in zip(sorted(keys, key=lambda k: summ[k]["per_step_mean"]["energy"]),
                              colours, markers, styles):
        m = summ[k]
        fit = [x / 1000 for x in m["per_success_fitted"]["energy"]]
        ax.plot(Ls, fit, linestyle=ls, color=col, linewidth=1.2)
        full, thin = [], []
        for L in range(5, lmax + 1, 5):
            x = m["per_success"]["energy"][L - 1]
            if math.isinf(x):
                continue
            (full if m["S"][L - 1] * m["n_runs"] >= MIN_SUCCESSES - 1e-9 else thin).append((L, x / 1000))
        if full:
            ax.plot(*zip(*full), mk, color=col, markersize=3.5, linestyle="none")
        if thin:
            ax.plot(*zip(*thin), mk, markerfacecolor="none", markeredgecolor=col,
                    markersize=3.5, linestyle="none")
        ax.plot([], [], linestyle=ls, marker=mk, color=col, markersize=3.5,
                label=display_name(k))
    ax.set_yscale("log")
    # top of the axis: a power of ten above twice the lowest curve at lmax
    best = min(summ[k]["per_success_fitted"]["energy"][lmax - 1] for k in keys) / 1000
    ax.set_ylim(0.01, 10 ** math.ceil(math.log10(2 * best)))
    ax.set_xlim(0, lmax + 2)
    ax.set_xlabel("Task length $L$ (steps)")
    ax.set_ylabel("GPU energy per successful task (kJ)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)


DEVELOPERS = {"Qwen3": "Qwen (Alibaba)", "Qwen3.8": "Qwen (Alibaba)", "GLM-4": "GLM (Zhipu)",
              "Llama 3.1": "Llama (Meta)", "Llama 3.2": "Llama (Meta)",
              "Gemma 3": "Gemma (Google)", "Ministral 3": "Ministral (Mistral)"}


# label positions (points), per task, where the default would overlap a neighbour
LABEL_OFFSET = {
    "d0": {"llama3.2:3b": (-5, 6), "qwen3:1.7b": (-5, -3), "ministral-3:3b": (-5, -12),
           "gemma3:4b": (5, -11), "llama3.1:8b": (5, 4), "ministral-3:8b": (-5, -9),
           "qwen3:14b": (5, -7), "ministral-3:14b": (-5, 4), "qwen3.8-27b": (-5, 4)},
    "tools": {"llama3.2:1b": (-5, 4), "qwen3:0.6b": (5, -6), "qwen3:1.7b": (-5, -3),
              "llama3.2:3b": (-5, 6), "ministral-3:3b": (-5, 6), "gemma3:4b": (5, -9),
              "qwen3:8b": (-5, -10), "glm4:9b": (5, -3), "llama3.1:8b": (5, 4),
              "ministral-3:8b": (-5, 4), "qwen3:14b": (5, -7), "gemma3:12b": (-5, 4),
              "ministral-3:14b": (-5, 4), "qwen3.8-27b": (-5, 4)}}


def scatter(summ, keys, dominated, path: Path, task="d0"):
    """The models on two per-step axes: energy per step and expected steps
    before the first error, 1/(1 - rho). One colour and marker per developer;
    models beaten per step by another model are drawn hollow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    devs = sorted({DEVELOPERS.get(family(k), family(k)) for k in keys})
    style = dict(zip(devs, zip(["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#000000", "#D55E00"],
                               ["o", "s", "^", "D", "v", "P"])))
    offset = LABEL_OFFSET.get(task, {})
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for k in keys:
        m = summ[k]
        col, mk = style[DEVELOPERS.get(family(k), family(k))]
        x, y = m["per_step_mean"]["energy"], 1 / max(1 - m["rho_hat"], 1e-4)
        face = "none" if k in dominated else col
        ax.plot([x], [y], mk, markerfacecolor=face, markeredgecolor=col, markersize=6)
        ax.annotate(display_name(k).replace("\\_", "_"), (x, y), textcoords="offset points",
                    xytext=offset.get(k.split("__")[0], (4, 3)), fontsize=6.5,
                    ha="right" if offset.get(k.split("__")[0], (1,))[0] < 0 else "left")
    for d in devs:
        col, mk = style[d]
        ax.plot([], [], mk, color=col, markersize=6, linestyle="none", label=d)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):  # plain numbers at 1, 2, 5, 10, ...
        axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("GPU energy per step (J)")
    ax.set_ylabel(r"Expected steps before first error, $1/(1-\hat\rho)$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--task", default="d0",
                    help="task family to analyse: d0, d1, d2 (order task at that difficulty) or tools")
    a = ap.parse_args()
    prices = json.loads((HERE / "prices.json").read_text()) if (HERE / "prices.json").exists() else {}
    energy = json.loads((HERE / "energy.json").read_text()) if (HERE / "energy.json").exists() else {}
    runs = {k: v for k, v in load(Path(a.results)).items() if k.endswith(f"__{a.task}")}
    if not runs:
        raise SystemExit(f"No run logs for task {a.task} in {a.results}")
    sfx = "" if a.task == "d0" else f"_{a.task}"
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

    dominated = per_step_dominated(summ, keys)
    shown = shown_models(summ, keys, lmax)
    print("dominated per step:", dominated)
    print("shown in the length figure:", shown)
    print("least energy per successful task (fitted):", cheapest_by_length(summ, keys, lmax))

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
    (out / f"summary{sfx}.json").write_text(json.dumps(
        {"lmax": lmax, "bootstrap": a.boot, "seed": a.seed, "order": keys, "models": slim,
         "dominated_per_step": dominated, "shown": shown,
         "cheapest_energy_fitted": cheapest_by_length(summ, keys, lmax),
         "cheapest_energy_observed": cheapest_by_length(summ, keys, lmax, "per_success"),
         "pairs": pairs},
        indent=1))
    tables(summ, keys, pairs, out, lmax, shown, dominated, sfx)
    if not a.no_figure:
        figure(summ, shown, out / f"fig_pilot_energy{sfx}.pdf", lmax)
        scatter(summ, keys, dominated, out / f"fig_pilot_models{sfx}.pdf", a.task)
    print(f"wrote summary{sfx}.json, table_pilot{sfx}.tex, table_pilot_crossovers{sfx}.tex, "
          f"table_pilot_crossovers_all{sfx}.tex, fig_pilot_energy{sfx}.pdf, fig_pilot_models{sfx}.pdf to {out}")


if __name__ == "__main__":
    main()
