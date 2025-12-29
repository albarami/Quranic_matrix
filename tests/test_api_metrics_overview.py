"""
Tests for /api/metrics/overview endpoint.

Validates:
- Returns 200 and serves JSON verbatim when truth metrics file exists
- Returns 503 when file is missing / corrupted / not ready
"""

import json
import hashlib
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_truth_metrics() -> dict:
    payload = {
        "schema_version": "metrics_v1",
        "status": "ready",
        "generated_at": "2025-01-01T00:00:00Z",
        "metrics": {
            "totals": {"spans": 1, "tafsir_sources_count": 7},
            "agent_distribution": {"items": [{"key": "AGT_ALLAH", "count": 1, "percentage": 100.0}]},
            "behavior_forms": {"items": []},
            "evaluations": {"items": []},
        },
    }
    payload_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["checksum"] = hashlib.sha256(payload_bytes).hexdigest()
    return payload


@pytest.fixture
def truth_metrics_file(tmp_path, sample_truth_metrics) -> Path:
    path = tmp_path / "truth_metrics_v1.json"
    path.write_text(json.dumps(sample_truth_metrics, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.unit
class TestMetricsOverviewEndpoint:
    """Tests for /api/metrics/overview endpoint."""
    
    def test_returns_200_when_file_exists(self, client, truth_metrics_file, monkeypatch):
        """Endpoint returns 200 when truth metrics file exists."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        assert response.status_code == 200
    
    def test_returns_schema_version(self, client, truth_metrics_file, monkeypatch):
        """Response includes schema_version."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        data = response.json()
        
        assert "schema_version" in data
        assert data["schema_version"] == "metrics_v1"
    
    def test_returns_status_ready(self, client, truth_metrics_file, monkeypatch):
        """Response has status='ready'."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "ready"
    
    def test_returns_checksum(self, client, truth_metrics_file, monkeypatch):
        """Response includes checksum."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        data = response.json()
        
        assert "checksum" in data
        assert len(data["checksum"]) == 64
    
    def test_returns_metrics_object(self, client, truth_metrics_file, monkeypatch):
        """Response includes metrics object with required keys."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        data = response.json()
        
        assert "metrics" in data
        metrics = data["metrics"]
        
        assert "totals" in metrics
        assert "agent_distribution" in metrics
        assert "behavior_forms" in metrics
        assert "evaluations" in metrics
    
    def test_agent_distribution_has_real_data(self, client, truth_metrics_file, monkeypatch):
        """Agent distribution contains real data."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        response = client.get("/api/metrics/overview")
        data = response.json()
        
        agent_dist = data["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        
        # Must have AGT_ALLAH as largest
        assert items[0]["key"] == "AGT_ALLAH"
        assert items[0]["percentage"] > 40
        
    def test_response_matches_file_exactly(self, client, truth_metrics_file, sample_truth_metrics, monkeypatch):
        """Response matches the truth file exactly (verbatim serving)."""
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", truth_metrics_file)
        
        # Get from API
        response = client.get("/api/metrics/overview")
        api_content = response.json()
        
        # Must be identical
        assert api_content == sample_truth_metrics


@pytest.mark.unit
class TestMetricsOverview503:
    """Tests for 503 responses when file is missing."""
    
    def test_503_includes_error_key(self, client, tmp_path, monkeypatch):
        """503 response includes error key."""
        # Temporarily point to non-existent file
        fake_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", fake_path)
        
        response = client.get("/api/metrics/overview")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"] == "metrics_truth_missing"
    
    def test_503_includes_how_to_fix(self, client, tmp_path, monkeypatch):
        """503 response includes how_to_fix instructions."""
        fake_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("src.api.routers.metrics.TRUTH_METRICS_FILE", fake_path)
        
        response = client.get("/api/metrics/overview")
        
        assert response.status_code == 503
        data = response.json()
        assert "how_to_fix" in data
        assert "build_truth_metrics_v1.py" in data["how_to_fix"]
