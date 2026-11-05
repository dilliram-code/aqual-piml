"""
Unit Tests: FastAPI REST Endpoints, Pydantic Validation, and Responses.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["available_catchments"] >= 5


def test_catchments_endpoint():
    response = client.get("/api/catchments")
    assert response.status_code == 200
    catchments = response.json()
    assert len(catchments) >= 5
    gauge_ids = [c["gauge_id"] for c in catchments]
    assert "54057" in gauge_ids


def test_catchment_detail_endpoint():
    response = client.get("/api/catchments/54057")
    assert response.status_code == 200
    detail = response.json()
    assert "metadata" in detail
    assert "data_summary" in detail
    assert detail["metadata"]["gauge_id"] == "54057"


def test_catchment_detail_not_found():
    response = client.get("/api/catchments/9999999")
    assert response.status_code == 404


def test_data_explorer_timeseries():
    response = client.get("/api/data/54057?downsample_factor=10")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert "precipitation" in data
    assert "discharge_spec" in data
    assert len(data["dates"]) > 0


def test_data_explorer_stats():
    response = client.get("/api/data/54057/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "statistics" in stats
    assert "precipitation" in stats["statistics"]
    assert "discharge_spec" in stats["statistics"]


def test_physics_equations_endpoint():
    response = client.get("/api/physics/equations")
    assert response.status_code == 200
    eq_data = response.json()
    assert "Delta S" in eq_data["equation"]
    assert "conversion_factor" in eq_data["summary_metrics"]


def test_physics_water_balance_endpoint():
    response = client.get("/api/physics/water-balance/54057")
    assert response.status_code == 200
    wb_data = response.json()
    assert "actual_et_mm" in wb_data
    assert "storage_mm" in wb_data
    assert "daily_residual_mm" in wb_data


def test_launch_experiment_and_poll():
    req_payload = {
        "gauge_id": "54057",
        "data_proportion": 0.5,
        "noise_level": 0.0,
        "lambda_physics": 0.1,
        "lookback_window": 15,
        "epochs": 2,
        "batch_size": 32,
        "hidden_dim": 16,
    }
    response = client.post("/api/experiments/launch", json=req_payload)
    assert response.status_code == 200
    res = response.json()
    job_id = res["job_id"]

    # Poll status
    status_resp = client.get(f"/api/experiments/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id
