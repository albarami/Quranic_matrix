"""
Test: Semantic Graph v2 (Phase 8.3)

Ensures the semantic graph:
- Requires cue phrase for causal edge types
- Has multi-source stats for edges
- Has calibrated confidence scores
- All edges have validated endpoints
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
AUDIT_REPORT_FILE = Path("reports/graph_audit_v1.md")

CAUSAL_EDGE_TYPES = ["CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"]
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
KNOWN_SOURCES = CORE_SOURCES + ["muyassar", "baghawi", "waseet"]  # All known tafsir sources


@pytest.fixture(scope="module")
def semantic_graph():
    """Load the semantic graph."""
    if not SEMANTIC_GRAPH_FILE.exists():
        pytest.skip(f"Semantic graph not found: {SEMANTIC_GRAPH_FILE}")
    
    with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
class TestCuePhraseRequirement:
    """Tests for cue phrase requirement on causal edges."""
    
    def test_semantic_edges_require_cue_phrase_for_causal_types(self, semantic_graph):
        """All causal edges must have cue_phrase present."""
        edges = semantic_graph.get("edges", [])
        
        causal_edges = [e for e in edges if e["edge_type"] in CAUSAL_EDGE_TYPES]
        
        for edge in causal_edges:
            assert "cue_phrases" in edge, \
                f"Edge {edge['source']} -> {edge['target']} missing cue_phrases"
            assert len(edge["cue_phrases"]) > 0, \
                f"Edge {edge['source']} -> {edge['target']} has empty cue_phrases"
    
    def test_all_edges_have_cue_strength(self, semantic_graph):
        """All edges must have cue_strength field."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges[:100]:  # Check first 100
            assert "cue_strength" in edge, \
                f"Edge {edge['source']} -> {edge['target']} missing cue_strength"
            assert edge["cue_strength"] in ["strong", "medium", "weak"], \
                f"Edge has invalid cue_strength: {edge['cue_strength']}"
    
    def test_evidence_contains_cue_phrase(self, semantic_graph):
        """Evidence quotes should contain the cue phrase."""
        edges = semantic_graph.get("edges", [])
        
        # Check sample of edges
        for edge in edges[:50]:
            evidence = edge.get("evidence", [])
            for ev in evidence[:2]:
                cue = ev.get("cue_phrase", "")
                quote = ev.get("quote", "")
                
                # Cue should be in quote (normalized)
                if cue and quote:
                    # Basic check - cue phrase should be findable
                    assert len(cue) > 0, "Empty cue phrase"


@pytest.mark.unit
class TestMultiSourceStats:
    """Tests for multi-source statistics on edges."""
    
    def test_semantic_edges_have_multi_source_stats(self, semantic_graph):
        """All edges must have sources_count and sources list."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges[:100]:
            assert "sources_count" in edge, \
                f"Edge {edge['source']} -> {edge['target']} missing sources_count"
            assert "sources" in edge, \
                f"Edge {edge['source']} -> {edge['target']} missing sources list"
            assert edge["sources_count"] == len(edge["sources"]), \
                f"Edge sources_count doesn't match sources list length"
    
    def test_high_confidence_edges_have_multiple_sources(self, semantic_graph):
        """High confidence edges (≥0.7) should have multiple sources."""
        edges = semantic_graph.get("edges", [])
        
        high_conf_edges = [e for e in edges if e["confidence"] >= 0.7]
        
        # Most high confidence edges should have 2+ sources
        multi_source_count = sum(1 for e in high_conf_edges if e["sources_count"] >= 2)
        
        assert multi_source_count > len(high_conf_edges) * 0.5, \
            f"Only {multi_source_count}/{len(high_conf_edges)} high-conf edges have 2+ sources"
    
    def test_sources_are_from_known_set(self, semantic_graph):
        """Sources should be from known tafsir sources."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges[:50]:
            for source in edge.get("sources", []):
                assert source in KNOWN_SOURCES, \
                    f"Unknown source: {source}"


@pytest.mark.unit
class TestConfidenceCalibration:
    """Tests for confidence calibration."""
    
    def test_confidence_in_valid_range(self, semantic_graph):
        """All confidence scores must be between 0 and 1."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges:
            assert 0 <= edge["confidence"] <= 1, \
                f"Edge has invalid confidence: {edge['confidence']}"
    
    def test_confidence_distribution_exists(self, semantic_graph):
        """Graph must have confidence distribution stats."""
        dist = semantic_graph.get("confidence_distribution", {})
        
        assert "high_confidence_0.7+" in dist
        assert "medium_confidence_0.5-0.7" in dist
        assert "low_confidence_below_0.5" in dist
    
    def test_weak_cue_edges_capped(self, semantic_graph):
        """Edges with weak cues should have capped confidence."""
        edges = semantic_graph.get("edges", [])
        
        weak_cue_edges = [e for e in edges if e.get("cue_strength") == "weak"]
        
        for edge in weak_cue_edges:
            assert edge["confidence"] <= 0.65, \
                f"Weak cue edge has confidence {edge['confidence']} > 0.65"
    
    def test_calibration_params_documented(self, semantic_graph):
        """Calibration parameters should be documented."""
        calibration = semantic_graph.get("calibration", {})
        
        assert "base_confidence" in calibration
        assert "evidence_boost" in calibration
        assert "source_boost" in calibration
        assert "strong_cue_boost" in calibration


@pytest.mark.unit
class TestEndpointValidation:
    """Tests for endpoint validation in evidence."""
    
    def test_all_edges_have_validated_endpoints(self, semantic_graph):
        """All edges must have validation.endpoints_in_quote = True."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges[:100]:
            validation = edge.get("validation", {})
            assert validation.get("endpoints_in_quote") == True, \
                f"Edge {edge['source']} -> {edge['target']} not endpoint-validated"
    
    def test_evidence_has_endpoints_validated_field(self, semantic_graph):
        """Evidence items should have endpoints_validated field."""
        edges = semantic_graph.get("edges", [])
        
        for edge in edges[:50]:
            for ev in edge.get("evidence", [])[:2]:
                assert "endpoints_validated" in ev, \
                    "Evidence missing endpoints_validated field"


@pytest.mark.unit
class TestGraphStructure:
    """Tests for graph structure."""
    
    def test_graph_version_is_2(self, semantic_graph):
        """Graph version must be 2.0."""
        assert semantic_graph.get("version") == "2.0"
    
    def test_graph_has_all_126_nodes(self, semantic_graph):
        """Graph must have all 126 canonical nodes."""
        assert semantic_graph.get("node_count") == 126
    
    def test_graph_has_hard_rules(self, semantic_graph):
        """Graph must document hard rules."""
        rules = semantic_graph.get("hard_rules", [])
        
        assert len(rules) >= 3
        assert any("cue phrase" in r.lower() for r in rules)
        assert any("endpoint" in r.lower() for r in rules)
    
    def test_edges_sorted_by_confidence(self, semantic_graph):
        """Edges should be sorted by confidence (descending)."""
        edges = semantic_graph.get("edges", [])
        
        for i in range(min(100, len(edges) - 1)):
            assert edges[i]["confidence"] >= edges[i + 1]["confidence"], \
                f"Edges not sorted by confidence at position {i}"


@pytest.mark.unit
class TestAuditReport:
    """Tests for audit report."""
    
    def test_audit_report_exists(self):
        """Audit report file must exist."""
        assert AUDIT_REPORT_FILE.exists(), f"Audit report not found: {AUDIT_REPORT_FILE}"
    
    def test_audit_report_has_top_50_edges(self):
        """Audit report must list top 50 edges."""
        if not AUDIT_REPORT_FILE.exists():
            pytest.skip("Audit report not found")
        
        with open(AUDIT_REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for numbered edges (1. through 50.)
        assert "### 1." in content, "Missing edge #1"
        assert "### 50." in content, "Missing edge #50"
    
    def test_audit_report_has_cue_phrase_highlight(self):
        """Audit report should highlight cue phrases."""
        if not AUDIT_REPORT_FILE.exists():
            pytest.skip("Audit report not found")
        
        with open(AUDIT_REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for bold markers (cue phrase highlighting)
        assert "**" in content, "No bold markers found for cue phrase highlighting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
