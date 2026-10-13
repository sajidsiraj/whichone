# Controlled-length pilot

Tests hypothesis H1 of the paper: task success falls with task length in a way that gives one crossover per resource between two models.

## Task

Each run is one multi-turn agent task with an exact number of steps (`lmax`). At each step the model receives a tool result, an order line to add or remove, and must reply with the new order total as JSON. The full history is kept, so errors carry forward. Runs are never stopped early, so the cost and time of every step are recorded. Success at length L means the first L totals are all correct, so one run gives success at every length from 1 to `lmax`. Tasks come from fixed seeds, so every model sees the same tasks. Temperature is 0.

## Files

| File | Role |
|---|---|
| `run.py` | Runs the models and writes raw logs to `results/` (needs API keys) |
| `pilot_local.json` | The model list and settings of the main pilot in the paper (local GPU) |
| `pilot.json` | Settings for a hosted-API run (Groq), not used for the paper results so far |
| `results/*.jsonl` | Raw logs, one line per run, with model, provider, date, git commit and every step. Under version control. |
| `analyse.py` | Computes every pilot number, table and figure in the paper from `results/` (no keys or GPU needed) |
| `output/` | Written by `analyse.py`: `summary.json`, `table_pilot.tex`, `table_pilot_crossovers.tex`, `fig_pilot_energy.pdf`. Not under version control. |
| `prices.json` | Provider prices per million tokens, used when billed cost is not returned |
| `energy.json` | Third-party energy estimates per output token, with sources. Reported as estimates. |

## Reproducing

From the repository root:

```bash
make pilot-analysis   # recompute all pilot numbers from the committed logs
make pilot-run-local  # re-run the main pilot on a local GPU through Ollama
make pilot-run        # hosted-API run (needs GROQ_API_KEY or OPENROUTER_API_KEY)
```

Keys are read from the environment or from the repository's `.env`, which is not under version control. Re-running calls live services, whose models and serving can change, so new runs will not match the committed logs exactly. The committed logs are the data behind the paper.

`--provider mock` runs the harness against a simulated model with a fixed per-step reliability. It exists only to test the code and is never used for results.

## Main pilot: setting

The committed logs `results/ollama__*__d0.jsonl` were produced on 13 October 2026 by harness version 4 (commit 079af42), with:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB), driver 595.99.02 |
| Server | Ollama 0.35.1, each model alone on the GPU and fully in GPU memory (checked before each model) |
| Models (Ollama digest) | qwen3:0.6b 7df6b6e09427, qwen3:1.7b 8f68893c685c, qwen3:8b 500a1f067a9f, qwen3:14b bdbd181c33f2, qwen3.8-27b 99322ec1f845 |
| Task | difficulty 0 (the tool result gives `line_total`), prompt `line-total`, 150 steps, 15 runs per model (seeds 1000 to 1014), temperature 0, at most 60 output tokens, context 16384 |
| Energy | GPU board power from NVML, sampled at 20 Hz and integrated over each call. Covers the GPU board only, not CPU, memory or cooling. |
| Time | server-reported time per call |

Energy per step grows with context length (about twofold over 150 steps for the small models), so `analyse.py` reports three crossovers: the paper's closed form with the mean per-step energy, a fitted one that uses the measured cumulative energy, and the observed one from the empirical success curves, with bootstrap intervals over runs.
