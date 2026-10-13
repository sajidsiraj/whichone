"""API-level tests via the ASGI test client, including the auth scaffold."""
import pytest
from fastapi.testclient import TestClient

from whichone.api.main import create_app

SNAPSHOT = "2026-10-13-paper-example"
SERVICE_BODY = {"length": 40, "weights": {"energy": 0.6, "cost": 0.2, "time": 0.2}}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("WHICHONE_API_KEYS", raising=False)
    return TestClient(create_app())


def test_health_and_version(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    body = client.get("/v1/version").json()
    assert SNAPSHOT in body["snapshots"]


def test_models_are_redacted_without_a_pro_key(client):
    """Unauthenticated/free callers get ids and objective definitions but
    never raw measured values or provenance -- that measurement effort is
    the Pro-tier moat (see whichone/api/routes.py RedactedSnapshot, and
    tests/test_pro_api.py::TestFreeTierModelsRedaction for the Pro-side
    full-provenance counterpart)."""
    body = client.get("/v1/models").json()
    assert body["version"] == SNAPSHOT
    first = body["models"][0]
    assert "objectives" not in first
    assert first["id"] == "model-a"


def test_frontier_is_preference_free(client):
    body = client.get("/v1/frontier", params={"length": 40}).json()
    assert sorted(body["efficient"]) == ["model-b", "model-c"]
    assert body["dominated_per_step"] == ["model-d", "model-e"]
    assert body["per_success"] is None  # values are a Pro-tier benefit


def test_frontier_with_completion_requirement(client):
    body = client.get("/v1/frontier", params={
        "length": 40, "max_attempts": 2, "completion_probability": 0.95}).json()
    assert body["efficient"] == ["model-a"]
    assert sorted(body["below_completion"]) == ["model-b", "model-c", "model-f"]


def test_frontier_respects_constraints(client):
    body = client.get("/v1/frontier", params={"min_context": 300}).json()
    assert sorted(body["infeasible"]) == ["model-b", "model-c", "model-d", "model-f"]


def test_frontier_intervals(client):
    body = client.get("/v1/frontier/intervals", params={"max_length": 30}).json()
    assert [(r["from_length"], r["to_length"]) for r in body["intervals"]] == [
        (1, 12), (13, 16), (17, 21), (22, 30)]
    assert client.get("/v1/frontier/intervals",
                      params={"max_length": 10**6}).status_code == 422


def test_recommend_returns_audit_record(client):
    resp = client.post("/v1/recommend", json=SERVICE_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended"] == "model-c"
    assert sorted(body["efficient"]) == ["model-b", "model-c"]
    assert body["stages"]["dominated_per_step"] == ["model-d", "model-e"]
    assert body["snapshot_version"] == SNAPSHOT
    assert sum(body["weights"].values()) == pytest.approx(1.0)
    assert body["per_success"] == {} and body["attempt_success"] == {}


def test_recommend_without_weights_returns_the_efficient_set(client):
    body = client.post("/v1/recommend", json={"length": 40}).json()
    assert body["preferences_needed"] is True
    assert body["recommended"] is None
    body = client.post("/v1/recommend", json={"length": 3}).json()
    assert body["recommended"] == "model-f"


def test_recommend_rejects_bad_profile(client):
    assert client.post("/v1/recommend", json={"length": 0}).status_code == 422
    resp = client.post("/v1/recommend", json={"length": 40, "weights": {"capability": 1}})
    assert resp.status_code == 422


def test_unknown_snapshot_404(client):
    resp = client.post("/v1/recommend", json=SERVICE_BODY,
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
