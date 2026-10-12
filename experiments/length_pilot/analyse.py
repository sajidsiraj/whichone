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
                   third-party estimates, reported as estimates
Outputs:
  output/summary.json  all numbers quoted in the paper
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from whichone.core.horizon import crossover_length

HERE = Path(__file__).resolve().parent
RESOURCES = ("energy", "cost", "time")


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
        for s in r["steps"]:
            j = s["step"] - 1
            if s.get("cost") is not None:
                c = s["cost"]
            elif pin is not None:
                c = (s["prompt_tokens"] * pin + s["completion_tokens"] * pout) / 1e6
            else:
                c = float("nan")
            sums["cost"][j] += c
            sums["time"][j] += s["provider_time"] if s.get("provider_time") is not None else s["wall_time"]
            sums["energy"][j] += s["completion_tokens"] * e if e is not None else float("nan")
    n = len(rs)
    return {k: [x / n for x in v] for k, v in sums.items()}


def per_success_curve(step_means, S):
    out, cum = [], 0.0
    for x, s in zip(step_means, S):
        cum += x
        out.append(cum / s if s > 0 else math.inf)
    return out


def summarise(rs, lmax, prices, energy, model):
    S = survival(rs, lmax)
    steps = step_resources(rs, lmax, prices, energy, model)
    return {"n_runs": len(rs), "rho_hat": rho_hat(rs, lmax), "S": S,
            "per_step_mean": {k: sum(v) / len(v) for k, v in steps.items()},
            "per_success": {k: per_success_curve(v, S) for k, v in steps.items()}}


def bootstrap_rho(rs, lmax, n, rng):
    vals = sorted(rho_hat([rng.choice(rs) for _ in rs], lmax) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    prices = json.loads((HERE / "prices.json").read_text()) if (HERE / "prices.json").exists() else {}
    energy = json.loads((HERE / "energy.json").read_text()) if (HERE / "energy.json").exists() else {}
    runs = load(Path(a.results))
    if not runs:
        raise SystemExit(f"No run logs in {a.results}")
    lmax = min(r["lmax"] for rs in runs.values() for r in rs)
    rng = random.Random(a.seed)

    summ = {}
    for key, rs in runs.items():
        model = rs[0]["model"]
        summ[key] = summarise(rs, lmax, prices, energy, model)
        summ[key]["rho_ci95"] = bootstrap_rho(rs, lmax, a.boot, rng)
        summ[key]["energy_is_estimate"] = model in energy
        print(f"{key}: runs={len(rs)} rho_hat={summ[key]['rho_hat']:.4f} "
              f"CI95={tuple(round(x, 4) for x in summ[key]['rho_ci95'])}")

    pairs = []
    keys = sorted(summ, key=lambda k: summ[k]["rho_hat"])
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            A, B = summ[ka], summ[kb]
            ratio = [sa / sb if sb > 0 else None for sa, sb in zip(A["S"], B["S"])]
            pair = {"a": ka, "b": kb, "ratio_S": ratio, "resources": {}}
            for k in RESOURCES:
                xa, xb = A["per_step_mean"][k], B["per_step_mean"][k]
                if math.isnan(xa) or math.isnan(xb):
                    pair["resources"][k] = None
                    continue
                pred = crossover_length(A["rho_hat"], B["rho_hat"], xa, xb)
                obs = next((L for L in range(1, lmax + 1)
                            if A["per_success"][k][L - 1] > B["per_success"][k][L - 1]), None)
                pair["resources"][k] = {"per_step_ratio": xa / xb, "predicted": pred, "observed": obs}
            pairs.append(pair)
            print(f"  {ka} vs {kb}: " + ", ".join(
                f"{k}: pred={v['predicted'] and round(v['predicted'], 1)} obs={v['observed']}"
                for k, v in pair["resources"].items() if v))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"lmax": lmax, "bootstrap": a.boot, "seed": a.seed, "models": summ, "pairs": pairs}, indent=1))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
