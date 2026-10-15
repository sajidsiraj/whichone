"""Versioned API surface. Everything lives under /v1; breaking changes get
/v2 rather than mutating this router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ..core.types import Constraints, Recommendation, TaskProfile
from .auth import require_api_key

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def _service(request: Request):
    return request.app.state.service


@router.get("/models")
def list_models(request: Request, snapshot: str | None = None):
    """The catalogue snapshot: models, measured values, sources and dates."""
    try:
        return _service(request).snapshot(snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


MAX_INTERVAL_LENGTH = 1000


@router.get("/frontier")
def get_frontier(request: Request, snapshot: str | None = None,
                 length: int = Query(1, ge=1),
                 max_attempts: int | None = Query(None, ge=1),
                 completion_probability: float | None = Query(None, gt=0, le=1),
                 min_context: float | None = None):
    """The efficient set for a task of `length` steps, per successful task.

    Preference-free: no weights are involved, so this answer is the same for
    every caller with the same task, and is served from cache.
    """
    constraints = Constraints(min_context=min_context)
    try:
        body = _service(request).frontier(constraints, length, completion_probability,
                                          max_attempts, snapshot)
        return body
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
                   snapshot: str | None = None):
    """Full pipeline for one task profile: constraints, per-step screen,
    efficient set at the task length, and Stage 2 only if it is needed.

    The response is the audit record: the efficient set with resources per
    successful task, the ranking and weights if Stage 2 ran, per-stage
    exclusions, and the snapshot version, so any recommendation can be
    reproduced exactly.
    """
    try:
        return _service(request).recommend(profile, snapshot)
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
