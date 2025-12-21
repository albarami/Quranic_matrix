"""Tests for QBM REST API."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test GET / returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "dataset_loaded" in data


class TestDatasets:
    """Test dataset endpoints."""
    
    def test_get_silver_dataset(self):
        """Test GET /datasets/silver returns dataset."""
        response = client.get("/datasets/silver")
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "spans" in data
        assert data["metadata"]["tier"] == "silver"
    
    def test_invalid_tier(self):
        """Test invalid tier returns 400."""
        response = client.get("/datasets/invalid")
        assert response.status_code == 400


class TestSpans:
    """Test spans endpoints."""
    
    def test_search_spans(self):
        """Test GET /spans returns spans."""
        response = client.get("/spans")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "spans" in data
    
    def test_filter_by_surah(self):
        """Test filtering by surah."""
        response = client.get("/spans?surah=1")
        assert response.status_code == 200
        data = response.json()
        for span in data["spans"]:
            assert span["reference"]["surah"] == 1
    
    def test_filter_by_agent(self):
        """Test filtering by agent type."""
        response = client.get("/spans?agent=AGT_ALLAH")
        assert response.status_code == 200
        data = response.json()
        for span in data["spans"]:
            assert span["agent"]["type"] == "AGT_ALLAH"
    
    def test_pagination(self):
        """Test pagination with limit and offset."""
        response = client.get("/spans?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["spans"]) <= 10


class TestSurahs:
    """Test surah endpoints."""
    
    def test_get_surah_spans(self):
        """Test GET /surahs/{num} returns spans."""
        response = client.get("/surahs/1")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "spans" in data
        for span in data["spans"]:
            assert span["reference"]["surah"] == 1
    
    def test_invalid_surah(self):
        """Test invalid surah returns 400."""
        response = client.get("/surahs/0")
        assert response.status_code == 400
        
        response = client.get("/surahs/115")
        assert response.status_code == 400


class TestStats:
    """Test statistics endpoint."""
    
    def test_get_stats(self):
        """Test GET /stats returns statistics."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_spans" in data
        assert "unique_surahs" in data
        assert "unique_ayat" in data
        assert "agent_types" in data
        assert "behavior_forms" in data
        assert "evaluations" in data
        assert "deontic_signals" in data


class TestVocabularies:
    """Test vocabularies endpoint."""
    
    def test_get_vocabularies(self):
        """Test GET /vocabularies returns controlled vocabularies."""
        response = client.get("/vocabularies")
        assert response.status_code == 200
        data = response.json()
        assert "agent_types" in data
        assert "behavior_forms" in data
        assert "evaluations" in data
        assert "deontic_signals" in data
        assert "speech_modes" in data
        assert "systemic" in data
        
        # Check expected values
        assert "AGT_ALLAH" in data["agent_types"]
        assert "physical_act" in data["behavior_forms"]
        assert "praise" in data["evaluations"]
        assert "amr" in data["deontic_signals"]


class TestDocs:
    """Test documentation endpoints."""
    
    def test_openapi_docs(self):
        """Test /docs endpoint exists."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self):
        """Test /redoc endpoint exists."""
        response = client.get("/redoc")
        assert response.status_code == 200
