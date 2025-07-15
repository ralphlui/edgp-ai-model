"""
Tests for the EDGP AI Model service.
"""
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data

def test_status_endpoint():
    """Test the status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_info_endpoint():
    """Test the info endpoint."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
    assert "supported_checks" in data

def test_analyze_endpoint_with_sample_data():
    """Test the analyze endpoint with sample data."""
    sample_data = {
        "data": [
            {"age": 25, "income": 50000, "name": "Alice"},
            {"age": 30, "income": 60000, "name": "Bob"},
            {"age": 35, "income": 70000, "name": "Charlie"},
            {"age": 1000, "income": 1000000, "name": "Outlier"},  # Anomaly
            {"age": 25, "income": 50000, "name": "Alice"}  # Duplicate
        ],
        "check_type": "both"
    }
    
    response = client.post("/api/v1/analyze", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "total_rows" in data
    assert "anomalies" in data
    assert "duplications" in data
    assert "summary" in data

def test_analyze_endpoint_with_invalid_data():
    """Test the analyze endpoint with invalid data."""
    invalid_data = {
        "data": [],
        "check_type": "both"
    }
    
    response = client.post("/api/v1/analyze", json=invalid_data)
    assert response.status_code == 400

def test_analyze_anomaly_only():
    """Test anomaly detection only."""
    sample_data = {
        "data": [
            {"value": 1}, {"value": 2}, {"value": 3}, {"value": 1000}
        ],
        "check_type": "anomaly"
    }
    
    response = client.post("/api/v1/analyze", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data["duplications"]) == 0  # No duplication check performed

def test_analyze_duplication_only():
    """Test duplication detection only."""
    sample_data = {
        "data": [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Alice", "age": 25}  # Duplicate
        ],
        "check_type": "duplication"
    }
    
    response = client.post("/api/v1/analyze", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data["anomalies"]) == 0  # No anomaly check performed

if __name__ == "__main__":
    pytest.main([__file__])
