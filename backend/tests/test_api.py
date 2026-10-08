"""
Automated Integration Tests for FastAPI Backend Endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database_connected" in data

def test_forecast_endpoints():
    response = client.get("/api/forecast?location=chennai&variable=temperature")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        item = data[0]
        assert "location_id" in item
        assert "blended_forecast" in item
        assert "weights" in item

def test_weights_map_endpoint():
    response = client.get("/api/weights/map?variable=temperature&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0

def test_models_registry_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_backtest_endpoint():
    response = client.get("/api/backtest?variable=temperature")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "skill_improvement_pct" in data

def test_skills_endpoint():
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_extremes_endpoint():
    response = client.get("/api/extremes")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_integrity_endpoint():
    response = client.get("/api/integrity")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "tables" in data

def test_dashboard_and_root_endpoints():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Hybrid AI" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "Hybrid AI" in res_dash.text

