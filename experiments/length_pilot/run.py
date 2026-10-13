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
HARNESS_VERSION = "5"  # 5 adds the tools task; the order task is as in 4

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "ollama": "http://127.0.0.1:11434/api/chat",
}
KEY_VARS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

SYSTEM = (
    "You are an order-processing agent. You keep the running total value of an "
    "order, in whole pounds. At each step you receive a tool result describing "
    "one order line. For action 'add', increase the total by qty * unit_price. "
    "For action 'remove', decrease it by qty * unit_price. "
    "Reply with JSON only, in the form {\"total\": <integer>}, and nothing else."
)


# Prompt variants. "answer-only" asks for the total alone; "short-working"
# asks for one line of working before it, which small models need in order
# to attempt the task at all. Every run logs the variant and the full text.
PROMPTS = {
    "answer-only": SYSTEM,
    "line-total": (
        "You are an order-processing agent. You keep the running total value of an "
        "order, in whole pounds. At each step you receive a tool result describing "
        "one order line, including its line_total. For action 'add', the new total is "
        "the previous total plus line_total. For action 'remove', it is the previous "
        "total minus line_total. Reply with JSON only, in the form "
        "{\"working\": \"<previous total> + <line_total> = <new total>\", \"total\": <new total>} "
        "(use - for remove), and nothing else. Do not explain."
    ),
    "short-working": (
        "You are an order-processing agent. You keep the running total value of an "
        "order, in whole pounds. At each step you receive a tool result describing "
        "one order line. For action 'add', the new total is the previous total plus "
        "qty * unit_price. For action 'remove', it is the previous total minus "
        "qty * unit_price. Reply with JSON only, in the form "
        "{\"working\": \"<previous total> + <qty>*<unit_price> = <new total>\", \"total\": <new total>} "
        "(use - for remove), and nothing else. Do not explain."
    ),
}


def load_keys(provider: str) -> list[str]:
    """All keys for a provider: VAR, VAR2, VAR3, ... from the environment,
    then from the repository's .env. Several keys are rotated on rate limits."""
    var = KEY_VARS[provider]
    found: dict[str, str] = {n: v.strip() for n, v in os.environ.items()
                             if re.fullmatch(var + r"\d*", n) and v.strip()}
    env = REPO / ".env"
    if not found and env.exists():
        for line in env.read_text().splitlines():
            n, _, v = line.partition("=")
            if re.fullmatch(var + r"\d*", n.strip()) and v.strip():
                found[n.strip()] = v.strip().strip('"').strip("'")
    if not found:
        sys.exit(f"No key: set {var} in the environment or in {env}.")
    return [found[n] for n in sorted(found)]


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
        if difficulty in (0, 1):
            qty, price = rng.randint(1, 9), rng.randint(11, 99)
        else:
            qty, price = rng.randint(2, 19), rng.randint(101, 999)
        action = "add" if (rng.random() < 0.65 or total < qty * price) else "remove"
        total += qty * price if action == "add" else -qty * price
        step = {"item": f"SKU-{rng.randint(1000, 9999)}", "action": action,
                "qty": qty, "unit_price": price}
        if difficulty == 0:  # the tool also returns the line total
            step["line_total"] = qty * price
        steps.append(step)
        truth.append(total)
    return start, steps, truth


# Second task family: tool calls in a customer-service conversation. The
# account starts with a few orders; at each step the customer asks for one
# change and the model must reply with the one tool call that makes it. About
# half the requests refer back to earlier ones ("that same order", "the same
# city as order X", "refund every lamp"), so the model has to track the
# state of the account through the conversation. Tool results only say "ok",
# so they give nothing away.

TOOLS_SYSTEM = (
    "You are a customer-service agent for an online shop. You change orders only "
    "by calling tools. Tools:\n"
    "set_address(order, city): deliver the order to that city\n"
    "add_item(order, item, qty): add qty units of an item to the order\n"
    "remove_item(order, item, qty): take qty units of an item out of the order\n"
    "refund_item(order, item, qty): refund an item and take it out of the order; "
    "qty must be the number of units of that item the order holds now\n"
    "Rules: \"that same order\" means the order of the customer's previous request. "
    "Keep track of every change, because tool results only say ok. "
    "For each customer message, reply with exactly one tool call as JSON only, in the form "
    "{\"tool\": \"<name>\", \"order\": \"<order>\", ...arguments}, for example "
    "{\"tool\": \"add_item\", \"order\": \"B7\", \"item\": \"mug\", \"qty\": 2}. Do not explain."
)
ITEMS = ["mug", "lamp", "towel", "kettle", "pillow", "candle", "plate", "rug",
         "clock", "vase", "jug", "bowl"]
CITIES = ["Leeds", "York", "Bath", "Derby", "Exeter", "Hull", "Bristol", "Leicester",
          "Norwich", "Oxford", "Preston", "Durham"]
TOOL_ARGS = {"set_address": ("order", "city"), "add_item": ("order", "item", "qty"),
             "remove_item": ("order", "item", "qty"), "refund_item": ("order", "item", "qty")}


def make_tools_task(seed: int, lmax: int):
    """Deterministic tools task: the opening account, then one customer
    message and the one correct tool call per step."""
    rng = random.Random(seed)
    ids = rng.sample([f"{c}{d}" for c in "ABCDEFGH" for d in range(1, 10)], 5)
    orders = {o: {"city": rng.choice(CITIES),
                  "items": {i: rng.randint(1, 4) for i in rng.sample(ITEMS, rng.randint(2, 3))}}
              for o in ids}
    intro = "Account opened. Orders:\n" + "\n".join(
        f"{o}: " + ", ".join(f"{q} x {i}" for i, q in v["items"].items()) + f"; deliver to {v['city']}"
        for o, v in orders.items())
    last = None
    steps, truth = [], []
    kinds = ["address", "same_address", "add", "add_same", "remove", "refund"]
    weights = [0.12, 0.16, 0.20, 0.16, 0.16, 0.20]
    while len(steps) < lmax:
        kind = rng.choices(kinds, weights)[0]
        o = last if kind == "add_same" else rng.choice(ids)
        if o is None:
            continue
        held = orders[o]["items"]
        if kind in ("remove", "refund") and not held:
            continue
        if kind == "address":
            city = rng.choice([c for c in CITIES if c != orders[o]["city"]])
            text, call = f"Please deliver order {o} to {city} instead.", {"tool": "set_address", "order": o, "city": city}
            orders[o]["city"] = city
        elif kind == "same_address":
            p = rng.choice([x for x in ids if x != o])
            city = orders[p]["city"]
            text, call = f"Send order {o} to the same city as order {p}.", {"tool": "set_address", "order": o, "city": city}
            orders[o]["city"] = city
        elif kind in ("add", "add_same"):
            item, q = rng.choice(ITEMS), rng.randint(1, 3)
            where = "that same order" if kind == "add_same" else f"order {o}"
            text, call = f"Add {q} x {item} to {where}.", {"tool": "add_item", "order": o, "item": item, "qty": q}
            held[item] = held.get(item, 0) + q
        elif kind == "remove":
            item = rng.choice(sorted(held))
            q = rng.randint(1, held[item])
            text, call = f"Take {q} x {item} out of order {o}.", {"tool": "remove_item", "order": o, "item": item, "qty": q}
            held[item] -= q
            if not held[item]:
                del held[item]
        else:
            item = rng.choice(sorted(held))
            text, call = f"Refund every {item} in order {o}.", {"tool": "refund_item", "order": o, "item": item, "qty": held[item]}
            del held[item]
        last = o
        steps.append(text)
        truth.append(call)
    return intro, steps, truth


def parse_call(text: str | None):
    """The first JSON object in a reply (code fences allowed), or None."""
    t = re.sub(r"```(?:json)?", "", text or "")
    start = t.find("{")
    if start < 0:
        return None
    try:
        d, _ = json.JSONDecoder().raw_decode(t[start:])
    except ValueError:
        return None
    if isinstance(d, dict) and isinstance(d.get("args"), dict):  # {"tool":..., "args": {...}}
        d = {"tool": d.get("tool"), **d["args"]}
    return d if isinstance(d, dict) else None


def call_correct(got, want) -> bool:
    """Right tool and right arguments. Names compare without case or spaces,
    quantities as integers; extra keys are ignored."""
    if not isinstance(got, dict):
        return False
    norm = lambda v: str(v).strip().lower()
    if norm(got.get("tool")) != want["tool"]:
        return False
    for k in TOOL_ARGS[want["tool"]]:
        g = got.get(k)
        if k == "qty":
            try:
                if int(g) != want[k] or float(g) != want[k]:
                    return False
            except (TypeError, ValueError):
                return False
        elif norm(g) != norm(want[k]):
            return False
    return True


def parse_total(text: str | None):
    """Read the total as a JSON parser would: if the key appears twice, the
    last value wins. Falls back to the last "total" match, then the last
    integer, for replies that are not valid JSON."""
    t = (text or "").strip()
    try:
        d = json.loads(t)
        if isinstance(d, dict) and isinstance(d.get("total"), int):
            return d["total"]
    except ValueError:
        pass
    m = re.findall(r'"total"\s*:\s*(-?\d+)', t)
    if m:
        return int(m[-1])
    m = re.findall(r"-?\d+", t)
    return int(m[-1]) if m else None


def call_api(provider, model, messages, keys, provider_pin=None, max_tokens=40,
             extra=None, max_retries=12):
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, **(extra or {})}
    if provider == "openrouter":
        body["usage"] = {"include": True}
        if provider_pin:
            body["provider"] = {"order": [provider_pin], "allow_fallbacks": False}
    for attempt in range(max_retries):
        headers = {"Authorization": f"Bearer {keys[attempt % len(keys)]}"}
        t0 = time.time()
        r = requests.post(ENDPOINTS[provider], headers=headers, json=body, timeout=120)
        wall = time.time() - t0
        if r.status_code == 429 or r.status_code >= 500:
            if r.status_code >= 500 or attempt >= len(keys) - 1:
                time.sleep(min(float(r.headers.get("retry-after", 2 ** min(attempt, 6))), 60))
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
            "reasoning_tokens": ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")),
            "finish_reason": d["choices"][0].get("finish_reason"),
            "provider_time": u.get("total_time"),  # Groq: server-side time
            "cost": u.get("cost"),                  # OpenRouter: billed cost, USD
            "wall_time": wall,
        }
    raise RuntimeError("too many retries")


class PowerSampler:
    """Samples GPU board power through NVML (ctypes, no extra packages) and
    integrates it over time. NVML reports power averaged over about one
    second, so per-step values are smoothed; whole-run totals are not
    affected. Measures the GPU board only, not CPU or memory."""

    def __init__(self, interval=0.05, index=0):
        import ctypes
        import threading
        self.ct = ctypes
        self.nv = ctypes.CDLL("libnvidia-ml.so.1")
        self.nv.nvmlInit_v2()
        self.h = ctypes.c_void_p()
        self.nv.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.h))
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def watts(self) -> float:
        p = self.ct.c_uint()
        self.nv.nvmlDeviceGetPowerUsage(self.h, self.ct.byref(p))
        return p.value / 1000.0

    def _loop(self):
        while not self.stop.is_set():
            w = self.watts()
            with self.lock:
                self.samples.append((time.time(), w))
                if len(self.samples) > 200000:
                    self.samples = self.samples[-100000:]
            time.sleep(self.interval)

    def energy(self, t0: float, t1: float) -> float:
        """Joules between t0 and t1 (trapezoid rule over the samples)."""
        with self.lock:
            pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
            before = [(t, w) for t, w in self.samples if t < t0][-1:]
        pts = before + pts
        if len(pts) < 2:
            return (pts[0][1] if pts else 0.0) * (t1 - t0)
        e = 0.0
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            a, b = max(ta, t0), min(tb, t1)
            if b > a:
                e += (wa + wb) / 2 * (b - a)
        tl, wl = pts[-1]
        if t1 > tl:
            e += wl * (t1 - max(tl, t0))
        return e

    def idle_watts(self, seconds=5.0) -> float:
        t0 = time.time()
        time.sleep(seconds)
        return self.energy(t0, time.time()) / (time.time() - t0)


def gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv,noheader"], text=True).strip()
        name, driver, mem, plimit = [x.strip() for x in out.split(",")]
        return {"gpu": name, "driver": driver, "gpu_memory": mem, "power_limit": plimit}
    except Exception:
        return {}


def ollama_meta(model: str) -> dict:
    meta = {}
    try:
        meta["ollama_version"] = requests.get("http://127.0.0.1:11434/api/version", timeout=10).json().get("version")
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json().get("models", [])
        for t in tags:
            if model in (t.get("name"), t.get("model")) or f"{model}:latest" in (t.get("name"), t.get("model")):
                meta.update({"model_digest": t.get("digest"), "model_size": t.get("size"),
                             "model_details": t.get("details")})
    except Exception as exc:
        meta["meta_error"] = str(exc)
    return meta


def ollama_loaded() -> list[dict]:
    return requests.get("http://127.0.0.1:11434/api/ps", timeout=10).json().get("models", [])


def ollama_unload_all():
    """Unload every resident model so the next one has the GPU to itself."""
    for m in ollama_loaded():
        requests.post("http://127.0.0.1:11434/api/generate",
                      json={"model": m["name"], "keep_alive": 0}, timeout=60)
    for _ in range(30):
        if not ollama_loaded():
            return
        time.sleep(1)
    raise RuntimeError("could not unload resident models")


def ollama_check_on_gpu(model: str) -> dict:
    """Refuse to measure a model that is not fully in GPU memory: part of
    it would run on the CPU, whose energy is not measured."""
    for m in ollama_loaded():
        if model in (m.get("name"), m.get("model")) or f"{model}:latest" == m.get("name"):
            if m.get("size_vram", 0) < m.get("size", 1):
                raise RuntimeError(f"{model} is not fully on the GPU "
                                   f"({m.get('size_vram')} of {m.get('size')} bytes)")
            return {"size_loaded": m.get("size"), "size_vram": m.get("size_vram"),
                    "context_length": m.get("context_length")}
    raise RuntimeError(f"{model} is not loaded")


def call_ollama(model, messages, max_tokens, extra, sampler):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens, "num_ctx": 8192}}
    if extra:
        body.update({k: v for k, v in extra.items() if k != "options"})
        body["options"].update(extra.get("options", {}))
    t0 = time.time()
    r = requests.post(ENDPOINTS["ollama"], json=body, timeout=600)
    t1 = time.time()
    r.raise_for_status()
    d = r.json()
    ns = 1e9
    return {
        "text": d["message"]["content"],
        "served_model": d.get("model"), "served_provider": "ollama-local",
        "system_fingerprint": None,
        "prompt_tokens": d.get("prompt_eval_count"),
        "completion_tokens": d.get("eval_count"),
        "reasoning_tokens": None,
        "finish_reason": d.get("done_reason"),
        "provider_time": (d.get("total_duration") or 0) / ns,
        "prompt_eval_time": (d.get("prompt_eval_duration") or 0) / ns,
        "eval_time": (d.get("eval_duration") or 0) / ns,
        "load_time": (d.get("load_duration") or 0) / ns,
        "cost": None,
        "wall_time": t1 - t0,
        "gpu_energy_j": sampler.energy(t0, t1) if sampler else None,
        "t_start": t0, "t_end": t1,
    }


def call_mock(messages, truth_value, rho, rng):
    """Tests the harness only. Never used for results in the paper."""
    if isinstance(truth_value, dict):  # tools task: a wrong quantity
        val = truth_value if rng.random() < rho else {**truth_value, "qty": 99}
    else:
        val = {"total": truth_value if rng.random() < rho else truth_value + rng.choice([-10, 10, 100])}
    n_in = sum(len(m["content"]) for m in messages) // 4
    return {"text": json.dumps(val), "served_model": "mock",
            "served_provider": "mock", "system_fingerprint": None,
            "prompt_tokens": n_in, "completion_tokens": 8, "provider_time": None,
            "cost": None, "wall_time": 0.0005 * n_in}


def system_prompt(cfg) -> str:
    return TOOLS_SYSTEM if cfg.get("task") == "tools" else PROMPTS[cfg.get("prompt", "answer-only")]


def run_one(cfg, seed, keys, rng, sampler=None):
    tools = cfg.get("task") == "tools"
    if tools:
        intro, steps, truth = make_tools_task(seed, cfg["lmax"])
        messages = [{"role": "system", "content": system_prompt(cfg)}]
        turns = [(intro + "\nCustomer: " if j == 0 else "Tool result: ok\nCustomer: ") + t
                 for j, t in enumerate(steps)]
    else:
        start, steps, truth = make_task(seed, cfg["lmax"], cfg["difficulty"])
        messages = [{"role": "system", "content": system_prompt(cfg)},
                    {"role": "user", "content": f"New order. Current total: {start}. "
                                                f"Reply with {{\"total\": {start}}} to confirm."},
                    {"role": "assistant", "content": json.dumps({"total": start})}]
        turns = ["Tool result: " + json.dumps(step) for step in steps]
    num_ctx = ((cfg.get("extra") or {}).get("options") or {}).get("num_ctx", 8192)
    records = []
    for j, (turn, true_total) in enumerate(zip(turns, truth), start=1):
        messages.append({"role": "user", "content": turn})
        if cfg["provider"] == "mock":
            res = call_mock(messages, true_total, cfg["mock_rho"], rng)
        elif cfg["provider"] == "ollama":
            res = call_ollama(cfg["model"], messages, cfg.get("max_tokens", 40), cfg.get("extra"), sampler)
        else:
            res = call_api(cfg["provider"], cfg["model"], messages, keys, cfg.get("provider_pin"),
                           cfg.get("max_tokens", 40), cfg.get("extra"))
        if tools:
            got = parse_call(res["text"])
            correct = call_correct(got, true_total)
        else:
            got = parse_total(res["text"])
            correct = got == true_total
        messages.append({"role": "assistant", "content": res["text"] or ""})
        # a prompt that fills the context window would be cut by the server
        over = (cfg["provider"] == "ollama" and res.get("prompt_tokens") is not None
                and res["prompt_tokens"] + cfg.get("max_tokens", 40) > num_ctx)
        records.append({"step": j, "true": true_total, "got": got, "correct": correct,
                        "context_full": over,
                        **{k: v for k, v in res.items() if k != "text"},
                        "raw": res["text"] or ""})
        if cfg.get("sleep"):
            time.sleep(cfg["sleep"])
    return records


def run_model(cfg, out_dir: Path):
    keys = None if cfg["provider"] in ("mock", "ollama") else load_keys(cfg["provider"])
    sampler, env_meta = None, {}
    if cfg["provider"] == "ollama":
        ollama_unload_all()
        sampler = PowerSampler()
        env_meta = {**gpu_info(), **ollama_meta(cfg["model"])}
        # Warm-up call so that model loading is not charged to the first run.
        call_ollama(cfg["model"], [{"role": "user", "content": "Reply with OK."}], 5, cfg.get("extra"), None)
        env_meta.update(ollama_check_on_gpu(cfg["model"]))
        env_meta["idle_watts"] = sampler.idle_watts(5.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = "tools" if cfg.get("task") == "tools" else f"d{cfg['difficulty']}"
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", f"{cfg['provider']}__{cfg['model']}__{task}")
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
            run_t0 = time.time()
            recs = run_one(cfg, seed, keys, rng, sampler)
            run_t1 = time.time()
            run_energy = None
            if sampler:
                time.sleep(1.5)  # let the one-second power average catch up
                run_energy = sampler.energy(run_t0, run_t1 + 1.5)
            first_fail = next((r["step"] for r in recs if not r["correct"]), None)
            f.write(json.dumps({
                "harness_version": HARNESS_VERSION, "git_commit": commit,
                "started_utc": started, "provider": cfg["provider"], "model": cfg["model"],
                "provider_pin": cfg.get("provider_pin"), "difficulty": cfg["difficulty"],
                "seed": seed, "lmax": cfg["lmax"], "temperature": 0,
                "max_tokens": cfg.get("max_tokens", 40), "extra": cfg.get("extra"),
                "task": cfg.get("task", "order-total"),
                "prompt": cfg.get("prompt", "answer-only"),
                "system_prompt": system_prompt(cfg),
                "env": env_meta, "run_wall_time": run_t1 - run_t0,
                "run_gpu_energy_j": run_energy,
                "first_fail": first_fail, "steps": recs}) + "\n")
            f.flush()
            print(f"{cfg['model']} seed {seed}: first failure at step {first_fail}", flush=True)
    if cfg["provider"] == "ollama":
        ollama_unload_all()


DEFAULTS = {"runs": 15, "lmax": 50, "seed0": 1000, "difficulty": 1, "sleep": 0.0,
            "mock_rho": 0.97, "provider_pin": None, "max_tokens": 40, "extra": None,
            "prompt": "answer-only", "task": "order-total"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON file with a list of model runs (see pilot.json)")
    ap.add_argument("--provider", choices=["groq", "openrouter", "ollama", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--provider-pin")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--lmax", type=int)
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--difficulty", type=int, choices=[0, 1, 2])
    ap.add_argument("--sleep", type=float)
    ap.add_argument("--mock-rho", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--prompt", choices=sorted(PROMPTS))
    ap.add_argument("--task", choices=["order-total", "tools"])
    ap.add_argument("--only", help="run only this model id from the config")
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
                                   "mock_rho": a.mock_rho, "provider_pin": a.provider_pin,
                                   "max_tokens": a.max_tokens, "prompt": a.prompt, "task": a.task}.items()
                 if v is not None}
    if a.only:
        cfgs = [c for c in cfgs if c["model"] == a.only]
    for cfg in cfgs:
        run_model({**cfg, **overrides}, Path(a.out))


if __name__ == "__main__":
    main()
