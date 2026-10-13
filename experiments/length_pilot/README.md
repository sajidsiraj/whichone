# Controlled-length pilot

Tests hypothesis H1 of the paper: task success falls with task length in a way that gives one crossover per resource between two models.

## Tasks

There are two task families, so that per-step reliability and the crossovers can be compared across kinds of work.

**Order task** (`task: order-total`, the default). At each step the model receives a tool result, an order line to add or remove, and must reply with the new order total as JSON.

**Tools task** (`task: tools`). A customer-service conversation. The account starts with five orders, each with a few items and a delivery city. At each step the customer asks for one change, and the model must reply with the one correct tool call (`set_address`, `add_item`, `remove_item` or `refund_item`) as JSON. About half the requests refer back to earlier ones ("that same order", "the same city as order F4", "refund every vase", which needs the current quantity), and tool results only say "ok", so the model has to track the account itself. A step is correct when the tool and every argument match the generated answer; case, code fences and extra keys are ignored.

In both tasks the full history is kept, so errors carry forward. Runs are never stopped early, so the cost and time of every step are recorded. Success at length L means the first L steps are all correct, so one run gives success at every length from 1 to `lmax`. Tasks come from fixed seeds, so every model sees the same tasks. Temperature is 0.

## Files

| File | Role |
|---|---|
| `run.py` | Runs the models and writes raw logs to `results/` (needs API keys) |
| `pilot_local.json` | The model list and settings of the main pilot in the paper (local GPU): five Qwen models |
| `pilot_mixed.json` | The second part of the main pilot: ten models from four other developers, same settings |
| `pilot_tools.json` | The tools task for all fifteen models, same settings |
| `pilot.json` | Settings for a hosted-API run (Groq), not used for the paper results so far |
| `results/*.jsonl` | Raw logs, one line per run, with model, provider, date, git commit and every step. Under version control. |
| `analyse.py` | Computes every pilot number, table and figure in the paper from `results/` (no keys or GPU needed) |
| `output/` | Written by `analyse.py`: `summary.json`, `table_pilot.tex`, `table_pilot_crossovers.tex`, `table_pilot_crossovers_all.tex`, `fig_pilot_energy.pdf`, `fig_pilot_models.pdf`. Not under version control. |
| `prices.json` | Provider prices per million tokens, used when billed cost is not returned |
| `energy.json` | Third-party energy estimates per output token, with sources. Reported as estimates. |

## Reproducing

From the repository root:

```bash
make pilot-analysis   # recompute all pilot numbers from the committed logs (both tasks)
make pilot-run-local  # re-run the main pilot on a local GPU through Ollama (both configurations)
make pilot-run        # hosted-API run (needs GROQ_API_KEY or OPENROUTER_API_KEY)
```

Keys are read from the environment or from the repository's `.env`, which is not under version control. Re-running calls live services, whose models and serving can change, so new runs will not match the committed logs exactly. The committed logs are the data behind the paper.

`--provider mock` runs the harness against a simulated model with a fixed per-step reliability. It exists only to test the code and is never used for results.

## Main pilot: setting

The committed logs `results/ollama__*__d0.jsonl` were produced on 13 October 2026 by harness version 4: the five Qwen models with `pilot_local.json` (commit 079af42), and the ten other models with `pilot_mixed.json` (commit 1a2c2a4; `run.py` is the same in both commits). Both used:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB), driver 595.99.02 |
| Server | Ollama 0.35.1, each model alone on the GPU and fully in GPU memory (checked before each model) |
| Models (Ollama digest) | qwen3:0.6b 7df6b6e09427, qwen3:1.7b 8f68893c685c, qwen3:8b 500a1f067a9f, qwen3:14b bdbd181c33f2, qwen3.8-27b 99322ec1f845; gemma3:1b 97558784bdc9, gemma3:4b 68ceab7e468a, gemma3:12b b92b218582c5, llama3.2:1b baf6a787fdff, llama3.2:3b a80c4f17acd5, llama3.1:8b 46e0c10c039e, ministral-3:3b f04aa1c738f6, ministral-3:8b 1922accd5827, ministral-3:14b 4760c35aeb9d, glm4:9b 5b699761eca5 |
| Task | difficulty 0 (the tool result gives `line_total`), prompt `line-total`, 150 steps, 15 runs per model (seeds 1000 to 1014), temperature 0, at most 60 output tokens, context 16384 |
| Energy | GPU board power from NVML, sampled at 20 Hz and integrated over each call. Covers the GPU board only, not CPU, memory or cooling. |
| Time | server-reported time per call |

Energy per step grows with context length (about twofold over 150 steps for the small models), so `analyse.py` reports three crossovers: the paper's closed form with the mean per-step energy, a fitted one that uses the measured cumulative energy, and the observed one from the empirical success curves, with bootstrap intervals over runs.
