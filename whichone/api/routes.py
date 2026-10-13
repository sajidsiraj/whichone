"""Versioned API surface. Everything lives under /v1; breaking changes get
/v2 rather than mutating this router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


MAX_INTERVAL_LENGTH = 1000


def redact(rec: Recommendation) -> Recommendation:
    """Free callers get the decision and its reasons (efficient set, stages,
    ranking scores) but not resources per successful task or attempt success:
    from those at two task lengths the measured per-step values could be
    recovered, and those are a Pro-tier benefit."""
    return rec.model_copy(update={
        "per_success": {}, "attempt_success": {},
        "ranking": [r.model_copy(update={"normalised": {}}) for r in rec.ranking]})


@router.get("/frontier")
def get_frontier(request: Request, snapshot: str | None = None,
                 length: int = Query(1, ge=1),
                 max_attempts: int | None = Query(None, ge=1),
                 completion_probability: float | None = Query(None, gt=0, le=1),
                 min_context: float | None = None,
                 pro: ApiKeyRecord | None = Depends(optional_pro_key)):
    """The efficient set for a task of `length` steps, per successful task.

    Preference-free: no weights are involved, so this answer is the same for
    every caller with the same task, and is served from cache.
    """
    constraints = Constraints(min_context=min_context)
    try:
        body = _service(request).frontier(constraints, length, completion_probability,
                                          max_attempts, snapshot)
        return body if pro is not None else {**body, "per_success": None}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/frontier/intervals")
def get_frontier_intervals(request: Request, snapshot: str | None = None,
                           max_length: int = Query(100, ge=1, le=MAX_INTERVAL_LENGTH),
                           max_attempts: int | None = Query(None, ge=1),
                           completion_probability: float | None = Query(None, gt=0, le=1),
                           min_context: float | None = None):
    """Efficient sets for every task length from 1 to `max_length`, as
    intervals of length over which the set stays the same."""
    constraints = Constraints(min_context=min_context)
    try:
        return _service(request).intervals(constraints, max_length, completion_probability,
                                           max_attempts, snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/recommend", response_model=Recommendation)
def post_recommend(request: Request, profile: TaskProfile,
                   snapshot: str | None = None,
                   pro: ApiKeyRecord | None = Depends(optional_pro_key)):
    """Full pipeline for one task profile: constraints, per-step screen,
    efficient set at the task length, and Stage 2 only if it is needed.

    The response is the audit record: the efficient set with resources per
    successful task, the ranking and weights if Stage 2 ran, per-stage
    exclusions, and the snapshot version, so any recommendation can be
    reproduced exactly.
    """
    try:
        rec = _service(request).recommend(profile, snapshot)
        return rec if pro is not None else redact(rec)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/profile")
def post_profile():
    """v1.1: free-text task description to a task profile (length, retry
    limit, required completion, sensitivities) via a pluggable LLM backend
    (the paper's automated way)."""
    raise HTTPException(status_code=501,
                        detail="automated task profiling ships in v1.1; "
                               "supply the task profile to /v1/recommend")
