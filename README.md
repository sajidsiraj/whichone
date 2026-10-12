# whichone

Two-stage decision support for choosing which LLM should power an agentic
AI task. Companion code to Siraj and Barbati, *When Is Smaller Greener?
Right-Sizing Language Models for Agentic AI* (in preparation).

Every computed number, table and figure in the paper can be regenerated with
`make reproduce`. See [REPRODUCE.md](REPRODUCE.md).

## The idea

Model selection is a multi-criteria decision problem, not a benchmark
lookup. The pipeline separates efficiency from preference:

1. **Feasibility screen** (Stage 0): hard constraints (context floor,
   governance class, cost/latency ceilings) remove infeasible models.
2. **Efficiency screen** (Stage 1, MODM): preference-free non-dominated
   sorting over measurable objectives (capability, tool-calling
   reliability, cost, latency, context, safety) yields the Pareto
   frontier. No weight or normalisation choice can change it, and it is
   cached marketplace-wide.
3. **Preference ranking** (Stage 2, MADM): only the frontier is ranked
   under task-specific weights (explicit, AHP-derived, or mapped from
   criterion intensities), aggregated by SAW with a TOPSIS cross-check,
   subject to non-compensatory aspiration levels (veto) so that, for
   example, an unreliable tool-caller can never top a high-risk
   recommendation regardless of weights.

Every response is an audit record: ranking, frontier, weight vector,
per-stage exclusions, and the catalogue snapshot version, so any
recommendation can be reproduced exactly.

## Layout

```
whichone/core        pure decision engine (pydantic + numpy, no web deps)
whichone/core/horizon.py  resources per successful task, crossovers, efficient sets by task length
whichone/catalogue   versioned marketplace snapshots (JSON repository)
whichone/api         FastAPI service, /v1, OpenAPI docs at /docs
tests/               includes golden tests reproducing the paper's tables
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
# The catalogue and the preference-free frontier
curl localhost:8000/v1/models
curl "localhost:8000/v1/frontier?max_cost=10"

# A recommendation for a high-volume triage agent
curl -X POST localhost:8000/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "weights": {"capability": 0.10, "tool_reliability": 0.20,
                    "cost": 0.30, "latency": 0.25,
                    "context": 0.05, "safety": 0.10},
        "veto": {"tool_reliability": 0.85}
      }'
```

Interactive docs: `http://localhost:8000/docs`.

## Authentication

Unset `WHICHONE_API_KEYS` = development mode, no auth. Set it to a
comma-separated list of keys and every request must carry a matching
`X-API-Key` header.

## Extending

* **New MCDA method**: implement `fn(normalised, weights) -> scores` and
  decorate with `@register_method("name")` in `whichone/core/madm.py`.
* **New objective**: add it to a snapshot's `objectives` list; the engine
  is objective-set agnostic.
* **New storage backend**: implement the `CatalogueRepository` protocol
  (e.g. Postgres) and pass it to `create_app(repository=...)`.
* **Custom criteria loading matrix**: pass `loading=` to
  `intensities_to_weights`; treat your copy as a versioned governance
  artefact.

## Roadmap

* v1.1: `/v1/profile`, free-text task description to criterion
  intensities via a pluggable LLM backend (currently 501).
* Live measurement layer: scheduled probes for price, latency and
  benchmark ingestion producing dated snapshots.
* Web front-end on the same API.
