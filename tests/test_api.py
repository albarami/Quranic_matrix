#!/usr/bin/env python3
"""
Test suite for QBM REST API.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_returns_health(self):
        """GET / should return health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "dataset_loaded" in data

    def test_health_endpoint(self):
        """GET /health should return same as root."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDatasetEndpoints:
    """Test dataset retrieval endpoints."""

    def test_get_silver_dataset(self):
        """GET /datasets/silver should return data."""
        response = client.get("/datasets/silver?limit=10")
        # May be 404 if no data, or 200 with data
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "meta" in data
            assert "spans" in data
            assert len(data["spans"]) <= 10

    def test_get_invalid_tier(self):
        """GET /datasets/invalid should fail validation."""
        response = client.get("/datasets/invalid")
        assert response.status_code == 422  # Validation error

    def test_pagination(self):
        """Dataset pagination should work."""
        response = client.get("/datasets/silver?limit=5&offset=0")
        if response.status_code == 200:
            data = response.json()
            assert len(data["spans"]) <= 5


class TestSpanEndpoints:
    """Test span search and retrieval."""

    def test_search_spans_no_filter(self):
        """GET /spans should return spans."""
        response = client.get("/spans?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "spans" in data
        assert "total" in data

    def test_search_by_surah(self):
        """GET /spans?surah=1 should filter by surah."""
        response = client.get("/spans?surah=1&limit=50")
        assert response.status_code == 200
        data = response.json()
        assert "spans" in data
        # All returned spans should be from surah 1
        for span in data["spans"]:
            ref = span.get("reference", {})
            assert ref.get("surah") == 1

    def test_search_by_agent_type(self):
        """GET /spans?agent_type=AGT_BELIEVER should filter."""
        response = client.get("/spans?agent_type=AGT_BELIEVER&limit=10")
        assert response.status_code == 200
        data = response.json()
        for span in data["spans"]:
            assert "BELIEVER" in span.get("agent", {}).get("type", "").upper()

    def test_search_by_behavior_form(self):
        """GET /spans?behavior_form=inner_state should filter."""
        response = client.get("/spans?behavior_form=inner_state&limit=10")
        assert response.status_code == 200

    def test_search_by_evaluation(self):
        """GET /spans?evaluation=praise should filter."""
        response = client.get("/spans?evaluation=praise&limit=10")
        assert response.status_code == 200

    def test_search_by_deontic(self):
        """GET /spans?deontic_signal=amr should filter."""
        response = client.get("/spans?deontic_signal=amr&limit=10")
        assert response.status_code == 200

    def test_get_nonexistent_span(self):
        """GET /spans/INVALID should return 404."""
        response = client.get("/spans/NONEXISTENT_SPAN_ID")
        assert response.status_code == 404


class TestSurahEndpoints:
    """Test surah-specific endpoints."""

    def test_get_surah_spans(self):
        """GET /surahs/1 should return Al-Fatiha spans."""
        response = client.get("/surahs/1?limit=50")
        assert response.status_code == 200
        data = response.json()
        assert "spans" in data

    def test_invalid_surah_number(self):
        """GET /surahs/200 should fail validation."""
        response = client.get("/surahs/200")
        assert response.status_code == 422


class TestStatsEndpoints:
    """Test statistics endpoints."""

    def test_get_stats(self):
        """GET /stats should return statistics."""
        response = client.get("/stats")
        # May be 404 if no data
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "total_spans" in data
            assert "agent_distribution" in data
            assert "behavior_distribution" in data
            assert "evaluation_distribution" in data

    def test_stats_by_tier(self):
        """GET /stats?tier=silver should work."""
        response = client.get("/stats?tier=silver")
        assert response.status_code in [200, 404]


class TestVocabularyEndpoints:
    """Test vocabulary reference endpoints."""

    def test_get_vocabularies(self):
        """GET /vocabularies should return controlled vocabularies."""
        response = client.get("/vocabularies")
        assert response.status_code == 200
        data = response.json()
        # Should have key vocabulary lists
        assert "agent_types" in data or "agents" in data
        assert "behavior_forms" in data or "behaviors" in data


class TestOpenAPISchema:
    """Test OpenAPI documentation."""

    def test_openapi_schema(self):
        """GET /openapi.json should return schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Quranic Behavior Matrix API"
        assert "paths" in data

    def test_docs_available(self):
        """GET /docs should return Swagger UI."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """GET /redoc should return ReDoc."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
