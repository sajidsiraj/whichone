"""Marketplace data layer behind a repository interface.

v1 backs the interface with versioned JSON snapshot files: reproducible (a
snapshot file is an experiment input), diffable, and free of infrastructure.
A Postgres-backed implementation can replace JsonFileRepository behind the
same Protocol without touching core or API code.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from ..core.types import ModelOffering, ObjectiveSpec


class Snapshot(BaseModel):
    """One versioned view of the marketplace."""
    version: str
    created_at: Optional[datetime] = None
    description: str = ""
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    models: list[ModelOffering] = Field(default_factory=list)


class CatalogueRepository(Protocol):
    def latest(self) -> Snapshot: ...
    def get(self, version: str) -> Snapshot: ...
    def versions(self) -> list[str]: ...


class JsonFileRepository:
    """Snapshots as JSON files in a directory; `latest` is the greatest
    version string, so use sortable versions (e.g. dates: 2026-07-06)."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def _path(self, version: str) -> Path:
        return self.directory / f"{version}.json"

    def versions(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def get(self, version: str) -> Snapshot:
        path = self._path(version)
        if not path.exists():
            raise KeyError(f"no snapshot {version!r} in {self.directory}")
        return Snapshot.model_validate(json.loads(path.read_text()))

    def latest(self) -> Snapshot:
        versions = self.versions()
        if not versions:
            raise KeyError(f"no snapshots in {self.directory}")
        return self.get(versions[-1])

    def save(self, snapshot: Snapshot) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(snapshot.version)
        path.write_text(snapshot.model_dump_json(indent=2))
        return path


def default_repository() -> JsonFileRepository:
    """The snapshots bundled with the package."""
    return JsonFileRepository(Path(__file__).parent / "snapshots")
