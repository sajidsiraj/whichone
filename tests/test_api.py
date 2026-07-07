"""API-level tests via the ASGI test client, including the auth scaffold."""
import pytest
from fastapi.testclient import TestClient

from whichone.api.main import create_app

T2_BODY = {
    "weights": {"capability": 0.10, "tool_reliability": 0.20, "cost": 0.30,
                "latency": 0.25, "context": 0.05, "safety": 0.10},
    "veto": {"tool_reliability": 0.85},
}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("WHICHONE_API_KEYS", raising=False)
    return TestClient(create_app())


def test_health_and_version(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    body = client.get("/v1/version").json()
    assert "2026-07-06-worked-example" in body["snapshots"]


def test_models_are_redacted_without_a_pro_key(client):
    """Unauthenticated/free callers get ids and objective definitions but
    never raw measured values or provenance -- that measurement effort is
    the Pro-tier moat (see whichone/api/routes.py RedactedSnapshot, and
    tests/test_pro_api.py::TestFreeTierModelsRedaction for the Pro-side
    full-provenance counterpart)."""
    body = client.get("/v1/models").json()
    assert body["version"] == "2026-07-06-worked-example"
    first = body["models"][0]
    assert "objectives" not in first
    assert first["id"] == "model-a"


def test_frontier_is_preference_free(client):
    body = client.get("/v1/frontier").json()
    assert sorted(body["frontier"]) == ["model-a", "model-b", "model-c",
                                        "model-e", "model-f"]
    assert body["dominated"] == ["model-d"]


def test_frontier_respects_constraints(client):
    body = client.get("/v1/frontier", params={"max_cost": 10}).json()
    assert sorted(body["infeasible"]) == ["model-a", "model-b", "model-e"]


def test_recommend_returns_audit_record(client):
    resp = client.post("/v1/recommend", json=T2_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended"] == "model-c"
    assert body["stages"]["vetoed"] == ["model-f"]
    assert body["snapshot_version"] == "2026-07-06-worked-example"
    assert sum(body["weights"].values()) == pytest.approx(1.0)


def test_recommend_rejects_bad_profile(client):
    resp = client.post("/v1/recommend", json={})
    assert resp.status_code == 422


def test_unknown_snapshot_404(client):
    resp = client.post("/v1/recommend", json=T2_BODY,
                       params={"snapshot": "nope"})
    assert resp.status_code == 404


def test_profile_stub_is_501(client):
    assert client.post("/v1/profile").status_code == 501


def test_api_key_gate(monkeypatch):
    monkeypatch.setenv("WHICHONE_API_KEYS", "sekrit-1, sekrit-2")
    gated = TestClient(create_app())
    assert gated.get("/v1/models").status_code == 401
    ok = gated.get("/v1/models", headers={"X-API-Key": "sekrit-2"})
    assert ok.status_code == 200
