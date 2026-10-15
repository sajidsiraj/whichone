"""Application factory.

Run locally:
    uvicorn whichone.api.main:app --reload
Interactive docs at /docs; the OpenAPI spec at /openapi.json is the contract
client SDKs are generated from.
"""
from __future__ import annotations

from fastapi import FastAPI

from .. import __version__
from ..catalogue.repository import CatalogueRepository, default_repository
from .routes import router
from .service import DecisionService


def create_app(repository: CatalogueRepository | None = None) -> FastAPI:
    app = FastAPI(
        title="whichone",
        version=__version__,
        description="Two-stage MODM+MADM decision support for selecting "
                    "LLMs in agentic AI systems.",
    )
    app.state.service = DecisionService(repository or default_repository())
    app.include_router(router)

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/version")
    def version():
        return {"version": __version__,
                "snapshots": app.state.service.repository.versions()}

    return app


app = create_app()
