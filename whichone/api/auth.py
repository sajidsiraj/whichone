"""API-key scaffold: open in development, gated in production.

If the WHICHONE_API_KEYS environment variable is unset or empty, every
request is allowed (development mode). If set (comma-separated keys), the
X-API-Key header must match one of them. Per-key metering and rate limits
attach here later without touching any route.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException


def _configured_keys() -> set[str]:
    raw = os.environ.get("WHICHONE_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    keys = _configured_keys()
    if not keys:
        return None  # development mode: auth disabled
    if x_api_key in keys:
        return x_api_key
    raise HTTPException(status_code=401, detail="missing or invalid API key")
