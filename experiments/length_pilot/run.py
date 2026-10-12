"""Controlled-length pilot: per-step reliability and resources of LLM agents.

Each run is one multi-turn agent task of `lmax` steps. At every step the model
receives a tool result (an order line to add or remove) and must reply with the
updated running total as JSON. The full history is kept, so the model sees its
own earlier answers and its errors carry forward. A run is never stopped early,
so the cost and time of every step are observed. Task success at length L means
that all of the first L totals are correct, so one run of `lmax` steps gives
success at every length from 1 to `lmax`.

Tasks are generated from fixed seeds, so every model sees the same tasks.
Raw logs (one JSON line per run) are written to results/ and are the data from
which every pilot number in the paper is computed (see analyse.py).

Keys are read from the environment (GROQ_API_KEY, OPENROUTER_API_KEY) or from
the repository's .env file, which is not under version control.

Usage:
  python experiments/length_pilot/run.py --config experiments/length_pilot/pilot.json
  python experiments/length_pilot/run.py --provider groq --model llama-3.1-8b-instant --runs 15 --lmax 50
  python experiments/length_pilot/run.py --provider mock --model mock --mock-rho 0.97   # tests the harness only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARNESS_VERSION = "1"

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
KEY_VARS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

SYSTEM = (
    "You are an order-processing agent. You keep the running total value of an "
    "order, in whole pounds. At each step you receive a tool result describing "
    "one order line. For action 'add', increase the total by qty * unit_price. "
    "For action 'remove', decrease it by qty * unit_price. "
    "Reply with JSON only, in the form {\"total\": <integer>}, and nothing else."
)


def load_key(provider: str) -> str:
    var = KEY_VARS[provider]
    if os.environ.get(var):
        return os.environ[var].strip()
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(var + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"No key: set {var} in the environment or in {env}.")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_task(seed: int, lmax: int, difficulty: int = 1):
    """Deterministic task: start total, step list and the true running totals."""
    rng = random.Random(seed)
    total = rng.randint(200, 900)
    start = total
    steps, truth = [], []
    for _ in range(lmax):
        if difficulty == 1:
            qty, price = rng.randint(1, 9), rng.randint(11, 99)
        else:
            qty, price = rng.randint(2, 19), rng.randint(101, 999)
        action = "add" if (rng.random() < 0.65 or total < qty * price) else "remove"
        total += qty * price if action == "add" else -qty * price
        steps.append({"item": f"SKU-{rng.randint(1000, 9999)}", "action": action,
                      "qty": qty, "unit_price": price})
        truth.append(total)
    return start, steps, truth


def parse_total(text: str | None):
    m = re.search(r'"total"\s*:\s*(-?\d+)', text or "")
    if m:
        return int(m.group(1))
    m = re.findall(r"-?\d+", text or "")
    return int(m[-1]) if m else None


def call_api(provider, model, messages, key, provider_pin=None, max_retries=8):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 40}
    if provider == "openrouter":
        body["usage"] = {"include": True}
        if provider_pin:
            body["provider"] = {"order": [provider_pin], "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(max_retries):
        t0 = time.time()
        r = requests.post(ENDPOINTS[provider], headers=headers, json=body, timeout=120)
        wall = time.time() - t0
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(float(r.headers.get("retry-after", 2 ** attempt)), 60))
            continue
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {}) or {}
        return {
            "text": d["choices"][0]["message"]["content"],
            "served_model": d.get("model"),
            "served_provider": d.get("provider"),
            "system_fingerprint": d.get("system_fingerprint"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "provider_time": u.get("total_time"),  # Groq: server-side time
            "cost": u.get("cost"),                  # OpenRouter: billed cost, USD
            "wall_time": wall,
        }
    raise RuntimeError("too many retries")


def call_mock(messages, truth_value, rho, rng):
    """Tests the harness only. Never used for results in the paper."""
    val = truth_value if rng.random() < rho else truth_value + rng.choice([-10, 10, 100])
    n_in = sum(len(m["content"]) for m in messages) // 4
    return {"text": json.dumps({"total": val}), "served_model": "mock",
            "served_provider": "mock", "system_fingerprint": None,
            "prompt_tokens": n_in, "completion_tokens": 8, "provider_time": None,
            "cost": None, "wall_time": 0.0005 * n_in}


def run_one(cfg, seed, key, rng):
    start, steps, truth = make_task(seed, cfg["lmax"], cfg["difficulty"])
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"New order. Current total: {start}. "
                                            f"Reply with {{\"total\": {start}}} to confirm."},
                {"role": "assistant", "content": json.dumps({"total": start})}]
    records = []
    for j, (step, true_total) in enumerate(zip(steps, truth), start=1):
        messages.append({"role": "user", "content": "Tool result: " + json.dumps(step)})
        if cfg["provider"] == "mock":
            res = call_mock(messages, true_total, cfg["mock_rho"], rng)
        else:
            res = call_api(cfg["provider"], cfg["model"], messages, key, cfg.get("provider_pin"))
        got = parse_total(res["text"])
        messages.append({"role": "assistant", "content": res["text"] or ""})
        records.append({"step": j, "true": true_total, "got": got, "correct": got == true_total,
                        **{k: v for k, v in res.items() if k != "text"},
                        "raw": (res["text"] or "")[:80]})
        if cfg.get("sleep"):
            time.sleep(cfg["sleep"])
    return records


def run_model(cfg, out_dir: Path):
    key = None if cfg["provider"] == "mock" else load_key(cfg["provider"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", f"{cfg['provider']}__{cfg['model']}__d{cfg['difficulty']}")
    path = out_dir / f"{tag}.jsonl"
    done = {json.loads(l)["seed"] for l in path.open()} if path.exists() else set()
    rng = random.Random(42)
    commit = git_commit()
    with path.open("a") as f:
        for i in range(cfg["runs"]):
            seed = cfg["seed0"] + i
            if seed in done:
                continue
            started = datetime.now(timezone.utc).isoformat()
            recs = run_one(cfg, seed, key, rng)
            first_fail = next((r["step"] for r in recs if not r["correct"]), None)
            f.write(json.dumps({
                "harness_version": HARNESS_VERSION, "git_commit": commit,
                "started_utc": started, "provider": cfg["provider"], "model": cfg["model"],
                "provider_pin": cfg.get("provider_pin"), "difficulty": cfg["difficulty"],
                "seed": seed, "lmax": cfg["lmax"], "temperature": 0, "system_prompt": SYSTEM,
                "first_fail": first_fail, "steps": recs}) + "\n")
            f.flush()
            print(f"{cfg['model']} seed {seed}: first failure at step {first_fail}", flush=True)


DEFAULTS = {"runs": 15, "lmax": 50, "seed0": 1000, "difficulty": 1, "sleep": 0.0,
            "mock_rho": 0.97, "provider_pin": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON file with a list of model runs (see pilot.json)")
    ap.add_argument("--provider", choices=["groq", "openrouter", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--provider-pin")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--lmax", type=int)
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--difficulty", type=int, choices=[1, 2])
    ap.add_argument("--sleep", type=float)
    ap.add_argument("--mock-rho", type=float)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    if a.config:
        spec = json.loads(Path(a.config).read_text())
        common = {**DEFAULTS, **spec.get("common", {})}
        cfgs = [{**common, **m} for m in spec["models"]]
    else:
        if not (a.provider and a.model):
            ap.error("give --config, or --provider and --model")
        cfgs = [{**DEFAULTS, "provider": a.provider, "model": a.model}]
    overrides = {k: v for k, v in {"runs": a.runs, "lmax": a.lmax, "seed0": a.seed0,
                                   "difficulty": a.difficulty, "sleep": a.sleep,
                                   "mock_rho": a.mock_rho, "provider_pin": a.provider_pin}.items()
                 if v is not None}
    for cfg in cfgs:
        run_model({**cfg, **overrides}, Path(a.out))


if __name__ == "__main__":
    main()
