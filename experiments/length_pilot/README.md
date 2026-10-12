# Controlled-length pilot

Tests hypothesis H1 of the paper: task success falls with task length in a way that gives one crossover per resource between two models.

## Task

Each run is one multi-turn agent task with an exact number of steps (`lmax`). At each step the model receives a tool result, an order line to add or remove, and must reply with the new order total as JSON. The full history is kept, so errors carry forward. Runs are never stopped early, so the cost and time of every step are recorded. Success at length L means the first L totals are all correct, so one run gives success at every length from 1 to `lmax`. Tasks come from fixed seeds, so every model sees the same tasks. Temperature is 0.

## Files

| File | Role |
|---|---|
| `run.py` | Runs the models and writes raw logs to `results/` (needs API keys) |
| `pilot.json` | The model list and settings used for the paper |
| `results/*.jsonl` | Raw logs, one line per run, with model, provider, date, git commit and every step. Under version control. |
| `analyse.py` | Computes every pilot number in the paper from `results/` (no keys needed) |
| `prices.json` | Provider prices per million tokens, used when billed cost is not returned |
| `energy.json` | Third-party energy estimates per output token, with sources. Reported as estimates. |

## Reproducing

From the repository root:

```bash
make pilot-analysis   # recompute all pilot numbers from the committed logs
make pilot-run        # re-run the models (needs GROQ_API_KEY or OPENROUTER_API_KEY)
```

Keys are read from the environment or from the repository's `.env`, which is not under version control. Re-running calls live services, whose models and serving can change, so new runs will not match the committed logs exactly. The committed logs are the data behind the paper.

`--provider mock` runs the harness against a simulated model with a fixed per-step reliability. It exists only to test the code and is never used for results.
