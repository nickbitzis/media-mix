import pytest
from fastapi.testclient import TestClient

from mediamix.app import create_app
from mediamix.world import CHANNELS


@pytest.fixture(scope="module")
def client(study):
    return TestClient(create_app(study))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Media Mix" in r.text


def test_fit(client):
    j = client.get("/api/fit").json()
    assert len(j["roi"]) == 7 and j["gate"]["pass"] is True and "lift_test" in j


def test_decomposition(client):
    j = client.get("/api/decomposition").json()
    assert len(j["week"]) == 156 and set(j["series"]) == {"baseline", *CHANNELS}
    assert client.get("/api/decomposition?model=base").status_code == 200
    assert client.get("/api/decomposition?model=nope").status_code == 422


def test_curves(client):
    j = client.get("/api/curves").json()
    assert set(j) == set(CHANNELS) and "marginal_roi_at_current" in j["tv"]


def test_optimize_default_bounds(client):
    r = client.post("/api/optimize", json={"quarter_budget": 4_000_000})
    assert r.status_code == 200
    j = r.json()
    assert sum(j["optimized"]["weekly_spend"].values()) * 13 == pytest.approx(4_000_000, rel=1e-4)
    assert "lift_pct_ci90" in j


def test_optimize_custom_bounds(client):
    bounds = {"tv": {"min": 0, "max": 0}, "email": {"min": 500_000, "max": 600_000}}
    j = client.post("/api/optimize", json={"quarter_budget": 4_000_000, "bounds": bounds, "uncertainty": False}).json()
    assert j["optimized"]["weekly_spend"]["tv"] == pytest.approx(0, abs=1e-6)
    assert 500_000 / 13 - 1 <= j["optimized"]["weekly_spend"]["email"] <= 600_000 / 13 + 1


@pytest.mark.parametrize("body", [
    {"quarter_budget": 4_000_000, "bounds": {"radio": {"min": 0, "max": 1}}},
    {"quarter_budget": 4_000_000, "bounds": {"tv": {"min": 10, "max": 5}}},
    {"quarter_budget": 1_000, "bounds": {c: {"min": 1_000, "max": 2_000} for c in CHANNELS}},
    {"quarter_budget": -5},
])
def test_optimize_rejects_bad_input(client, body):
    assert client.post("/api/optimize", json=body).status_code == 422


def test_scenarios(client):
    j = client.post("/api/scenarios", json={"user": {"tv": 1_000_000}}).json()
    assert set(j) == {"current", "optimized", "user"}
    assert j["user"]["weekly_spend"]["tv"] == pytest.approx(1_000_000 / 13, rel=1e-6)
    assert client.post("/api/scenarios", json={}).json().keys() == {"current", "optimized"}


def test_scenarios_reject_bad_input(client):
    assert client.post("/api/scenarios", json={"user": {"radio": 5}}).status_code == 422
    assert client.post("/api/scenarios", json={"user": {"tv": -5}}).status_code == 422


def test_request_id_and_metrics(client):
    r = client.get("/api/fit", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    client.post("/api/optimize", json={"quarter_budget": 4_000_000, "uncertainty": False})
    m = client.get("/metrics").text
    assert "mediamix_http_requests_total" in m and "mediamix_optimizations_total" in m
    assert "mediamix_gate_pass 1" in m and "mediamix_holdout_mape" in m
