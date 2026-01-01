"""
Phase 7.3 Fix: Q25 Genome Export Tests

Tests for:
- Genome export counts match canonical entities
- Edges have provenance (chunk_id, verse_key, char_start, char_end)
- Checksum is reproducible (same inputs → same checksum)
- No fabrication (no edge without validated quote endpoints)
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for API."""
    from src.api.main import app
    return TestClient(app)


class TestGenomeCountsMatchCanonical:
    """Test that genome export counts match canonical entity registry."""
    
    def test_behaviors_count_is_87(self, client):
        """Canonical registry defines exactly 87 behaviors."""
        response = client.get("/api/genome/status")
        assert response.status_code == 200
        data = response.json()
        assert data["statistics"]["canonical_behaviors"] == 87
    
    def test_agents_count_is_14(self, client):
        """Canonical registry defines exactly 14 agents."""
        response = client.get("/api/genome/status")
        assert response.status_code == 200
        data = response.json()
        assert data["statistics"]["canonical_agents"] == 14
    
    def test_heart_states_count_is_12(self, client):
        """Canonical registry defines exactly 12 heart states."""
        response = client.get("/api/genome/status")
        assert response.status_code == 200
        data = response.json()
        assert data["statistics"]["canonical_heart_states"] == 12
    
    def test_consequences_count_is_16(self, client):
        """Canonical registry defines exactly 16 consequences."""
        response = client.get("/api/genome/status")
        assert response.status_code == 200
        data = response.json()
        assert data["statistics"]["canonical_consequences"] == 16
    
    def test_semantic_edges_exist(self, client):
        """Semantic graph must have edges."""
        response = client.get("/api/genome/status")
        assert response.status_code == 200
        data = response.json()
        assert data["statistics"]["semantic_edges"] > 0
    
    def test_export_behaviors_match_status(self, client):
        """Export behaviors count must match status count."""
        status_resp = client.get("/api/genome/status")
        export_resp = client.get("/api/genome/export?mode=light")
        
        assert status_resp.status_code == 200
        assert export_resp.status_code == 200
        
        status_count = status_resp.json()["statistics"]["canonical_behaviors"]
        export_count = len(export_resp.json()["behaviors"])
        
        assert export_count == status_count == 87


class TestGenomeEdgesHaveProvenance:
    """Test that semantic edges have proper provenance."""
    
    def test_edges_have_required_fields(self, client):
        """Each edge must have source, target, edge_type, confidence."""
        response = client.get("/api/genome/relationships?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        for edge in data["relationships"]:
            assert "source" in edge, "Edge missing source"
            assert "target" in edge, "Edge missing target"
            assert "edge_type" in edge, "Edge missing edge_type"
            assert "confidence" in edge, "Edge missing confidence"
    
    def test_edges_have_evidence_count(self, client):
        """Each edge must have evidence_count."""
        response = client.get("/api/genome/relationships?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        for edge in data["relationships"]:
            assert "evidence_count" in edge
            assert edge["evidence_count"] >= 0
    
    def test_full_export_edges_have_provenance(self, client):
        """Full export edges must have chunk_id, verse_key, char_start, char_end."""
        response = client.get("/api/genome/export?mode=full")
        assert response.status_code == 200
        data = response.json()
        
        # Check first 5 edges with evidence
        edges_with_evidence = [e for e in data["semantic_edges"] if e.get("evidence")][:5]
        
        for edge in edges_with_evidence:
            for ev in edge.get("evidence", []):
                assert "chunk_id" in ev, f"Evidence missing chunk_id: {ev}"
                assert "verse_key" in ev, f"Evidence missing verse_key: {ev}"
                assert "char_start" in ev, f"Evidence missing char_start: {ev}"
                assert "char_end" in ev, f"Evidence missing char_end: {ev}"


class TestGenomeChecksumReproducible:
    """Test that genome checksum is reproducible."""
    
    def test_checksum_present(self, client):
        """Export must include checksum."""
        response = client.get("/api/genome/export?mode=light")
        assert response.status_code == 200
        data = response.json()
        assert "checksum" in data
        assert len(data["checksum"]) == 64  # SHA256 hex
    
    def test_checksum_reproducible(self, client):
        """Same inputs must produce same checksum."""
        response1 = client.get("/api/genome/export?mode=light")
        response2 = client.get("/api/genome/export?mode=light")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        checksum1 = response1.json()["checksum"]
        checksum2 = response2.json()["checksum"]
        
        assert checksum1 == checksum2, "Checksum not reproducible"
    
    def test_checksum_includes_version(self, client):
        """Export must include source versions for reproducibility."""
        response = client.get("/api/genome/export?mode=light")
        assert response.status_code == 200
        data = response.json()
        
        assert "source_versions" in data
        assert "canonical_entities" in data["source_versions"]
        assert "semantic_graph" in data["source_versions"]


class TestGenomeNoFabrication:
    """Test that no edge exists without validated evidence."""
    
    def test_edges_have_validation_field(self, client):
        """Each edge must have validation metadata."""
        response = client.get("/api/genome/relationships?limit=20")
        assert response.status_code == 200
        data = response.json()
        
        for edge in data["relationships"]:
            assert "validation" in edge, f"Edge missing validation: {edge['source']} -> {edge['target']}"
    
    def test_full_export_evidence_has_endpoints_validated(self, client):
        """Evidence must have endpoints_validated field."""
        response = client.get("/api/genome/export?mode=full")
        assert response.status_code == 200
        data = response.json()
        
        # Check edges with evidence
        edges_with_evidence = [e for e in data["semantic_edges"] if e.get("evidence")][:5]
        
        for edge in edges_with_evidence:
            for ev in edge.get("evidence", []):
                assert "endpoints_validated" in ev, f"Evidence missing endpoints_validated"


class TestGenomeModes:
    """Test light vs full export modes."""
    
    def test_light_mode_no_evidence_payloads(self, client):
        """Light mode should not include evidence payloads."""
        response = client.get("/api/genome/export?mode=light")
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "light"
        assert data["semantic_edges"] == []  # No edges in light mode
    
    def test_full_mode_includes_evidence(self, client):
        """Full mode should include evidence payloads."""
        response = client.get("/api/genome/export?mode=full")
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "full"
        assert len(data["semantic_edges"]) > 0
    
    def test_invalid_mode_returns_400(self, client):
        """Invalid mode should return 400."""
        response = client.get("/api/genome/export?mode=invalid")
        assert response.status_code == 400


class TestGenomeEndpoints:
    """Test individual entity endpoints."""
    
    def test_behaviors_endpoint(self, client):
        """Behaviors endpoint returns canonical behaviors."""
        response = client.get("/api/genome/behaviors")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 87
        assert len(data["behaviors"]) == 87
    
    def test_behaviors_filter_by_category(self, client):
        """Behaviors can be filtered by category."""
        response = client.get("/api/genome/behaviors?category=speech")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        for b in data["behaviors"]:
            assert b["category"] == "speech"
    
    def test_agents_endpoint(self, client):
        """Agents endpoint returns canonical agents."""
        response = client.get("/api/genome/agents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 14
    
    def test_heart_states_endpoint(self, client):
        """Heart states endpoint returns canonical heart states."""
        response = client.get("/api/genome/heart-states")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 12
    
    def test_consequences_endpoint(self, client):
        """Consequences endpoint returns canonical consequences."""
        response = client.get("/api/genome/consequences")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 16
    
    def test_relationships_filter_by_edge_type(self, client):
        """Relationships can be filtered by edge type."""
        response = client.get("/api/genome/relationships?edge_type=CAUSES&limit=10")
        assert response.status_code == 200
        data = response.json()
        for rel in data["relationships"]:
            assert rel["edge_type"] == "CAUSES"
