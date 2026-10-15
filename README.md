# whichone

Two-stage decision support for choosing which LLM should power an agentic
AI task. Companion code to Siraj and Barbati, *When Is Smaller Greener?
Right-Sizing Language Models for Agentic AI* ([arXiv:2610.19345](https://arxiv.org/abs/2610.19345)).

Every computed number, table and figure in the paper can be regenerated with
`make reproduce`. See [REPRODUCE.md](REPRODUCE.md).

## The idea

An agent completes a task only if every step succeeds, and failed attempts
still use energy, cost and time. So models are compared per successfully
completed task, not per call. With per-step reliability rho and resource r
per step, a task of L steps uses on average `r L / rho^L` of that resource per
success. A smaller model is better for short tasks; past a crossover length,
one per resource, a more reliable model uses less of everything.

The pipeline (paper, Section 4) separates questions of fact from questions
of value:

0. **Feasibility** (Stage 0): hard constraints (context, safety, deployment
   class, providers, ceilings per task) remove infeasible models.
1. **Per-step screen** (Stage 1): a model worse on every per-step quantity
   (reliability, energy, cost, time) is dominated at every task length and is
   removed once.
2. **Efficient set at the task length**: resources per successful task for
   the task's length L; the chance constraint that the task is completed with
   probability at least beta within K attempts; the Pareto-efficient set.
   Often one model is left, and no preferences are needed.
3. **Preference** (Stage 2): only if several models are efficient, weights
   over energy, cost and time (explicit, AHP-derived, or mapped from
   sensitivities) rank them by SAW, with TOPSIS as a check. Without weights
   the API returns the efficient set and says that a preference is needed.

Every response is an audit record: the efficient set, the per-stage
exclusions, the weights if Stage 2 ran, and the catalogue snapshot version, so
any recommendation can be reproduced exactly.

## Layout

```
whichone/core        pure decision engine (pydantic + numpy, no web deps)
whichone/core/horizon.py  resources per successful task, crossovers, efficient sets by task length
whichone/catalogue   versioned marketplace snapshots (JSON repository)
whichone/api         FastAPI service, /v1, OpenAPI docs at /docs
tests/               includes golden tests reproducing the paper's example (Section 4)
paper/               reproduce.py regenerates the paper's numbers, tables and figures
experiments/         controlled-length pilot: run.py (live models), analyse.py (from logs)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn whichone.api.main:app --reload
```

Then:

```bash
# The catalogue: the six illustrative models of the paper's example
curl localhost:8000/v1/models

# Efficient models for a 40-step task (no preferences involved)
curl "localhost:8000/v1/frontier?length=40"

# The same task when it may be tried at most twice and must succeed with
# probability 0.95 (an irreversible payment step)
curl "localhost:8000/v1/frontier?length=40&max_attempts=2&completion_probability=0.95"

# Efficient sets for every task length up to 250, as intervals
curl "localhost:8000/v1/frontier/intervals?max_length=250"

# A recommendation for a 40-step task by a deployment that cares mostly about carbon
curl -X POST localhost:8000/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"length": 40, "weights": {"energy": 0.6, "cost": 0.2, "time": 0.2}}'
```
