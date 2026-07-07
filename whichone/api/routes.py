"""Versioned API surface. Everything lives under /v1; breaking changes get
/v2 rather than mutating this router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core.types import Constraints, Recommendation, TaskProfile
from .auth import require_api_key
from .pro_auth import optional_pro_key
from ..pro.tenancy import ApiKeyRecord

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def _service(request: Request):
    return request.app.state.service


class RedactedModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    deployment_class: str


class RedactedSnapshot(BaseModel):
    """What the free tier sees: enough to interpret a recommendation or a
    frontier (which ids exist, what the objectives mean), never the raw
    measured values or their provenance -- that measurement effort is the
    moat, and it is a Pro-tier benefit."""
    version: str
    description: str
    objectives: list
    models: list[RedactedModelInfo]


@router.get("/models")
def list_models(request: Request, snapshot: str | None = None,
                pro: ApiKeyRecord | None = Depends(optional_pro_key)):
    """Full provenance (values, sources, timestamps) for an authenticated
    Pro caller; a redacted, values-free view for everyone else."""
    try:
        snap = _service(request).snapshot(snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if pro is not None:
        return snap
    return RedactedSnapshot(
        version=snap.version, description=snap.description,
        objectives=snap.objectives,
        models=[RedactedModelInfo(id=m.id, name=m.name, provider=m.provider,
                                  deployment_class=m.deployment_class)
                for m in snap.models])


@router.get("/frontier")
def get_frontier(request: Request, snapshot: str | None = None,
                 min_context: float | None = None,
                 max_cost: float | None = None,
                 max_latency: float | None = None):
    """The Pareto-efficient frontier, optionally under hard constraints.

    Preference-free: no weights are involved, so this answer is the same for
    every caller sharing the constraints, and is served from cache.
    """
    constraints = Constraints(min_context=min_context, max_cost=max_cost,
                              max_latency=max_latency)
    try:
        return _service(request).frontier(constraints, snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/recommend", response_model=Recommendation)
def post_recommend(request: Request, profile: TaskProfile,
                   snapshot: str | None = None):
    """Full pipeline: feasibility, efficiency, veto, ranking.

    The response is the audit record: ranking with normalised values, the
    frontier, the weight vector actually used, per-stage exclusions, and the
    snapshot version, so any recommendation can be reproduced exactly.
    """
    try:
        return _service(request).recommend(profile, snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/profile")
def post_profile():
    """v1.1: free-text task description to criterion intensities via a
    pluggable LLM backend (the paper's automated channel)."""
    raise HTTPException(status_code=501,
                        detail="automated task profiling ships in v1.1; "
                               "supply intensities or weights to /v1/recommend")
