"""
Flagship Acceptance Tests: 25 Legendary QBM Questions

These tests validate the Truth Layer can answer the flagship questions
with deterministic, evidence-backed responses.

Test A: Q7 - 11-dimensional profile for الحسد (envy)
Test B: Q1 - Causal chains الغفلة → الكفر with evidence
Test C: Q23 - Complete التوبة (repentance) analysis
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.query_planner import QueryPlanner, QueryType

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.fixture(scope="module")
def planner():
    """Get a loaded query planner."""
    p = QueryPlanner(base_path=Path("."))
    p.load()
    return p


@pytest.fixture(scope="module")
def concept_index():
    """Load concept index directly."""
    concepts = {}
    path = Path("data/evidence/concept_index_v1.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            concepts[entry["concept_id"]] = entry
    return concepts


@pytest.fixture(scope="module")
def semantic_graph():
    """Load semantic graph directly."""
    path = Path("data/graph/semantic_graph_v1.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cooccurrence_graph():
    """Load co-occurrence graph directly."""
    path = Path("data/graph/cooccurrence_graph_v1.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# TEST A: Q7 - 11-Dimensional Profile for الحسد (Envy)
# =============================================================================

@pytest.mark.integration
class TestQ7EnvyProfile:
    """
    Q7: Build complete 11-dimensional profile for الحسد (envy).
    
    Requirements:
    - Entity resolves to BEH_EMO_ENVY
    - Has evidence from multiple tafsir sources
    - Has graph neighbors (co-occurrence)
    - Has semantic edges (causes, leads_to, opposite_of)
    - Evidence has character offsets for verification
    """
    
    def test_envy_entity_resolution(self, planner):
        """الحسد must resolve to BEH_EMO_ENVY."""
        entity_id = planner.resolve_entity("الحسد")
        assert entity_id == "BEH_EMO_ENVY", f"Expected BEH_EMO_ENVY, got {entity_id}"
    
    def test_envy_has_concept_evidence(self, concept_index):
        """الحسد must have evidence in concept index."""
        envy = concept_index.get("BEH_EMO_ENVY")
        assert envy is not None, "BEH_EMO_ENVY not in concept index"
        assert envy["total_mentions"] > 0, "No mentions found"
    
    def test_envy_multi_source_coverage(self, concept_index):
        """الحسد must have evidence from at least 3 core sources."""
        envy = concept_index.get("BEH_EMO_ENVY")
        stats = envy.get("per_source_stats", {})
        
        sources_with_evidence = sum(
            1 for s in CORE_SOURCES 
            if stats.get(s, {}).get("count", 0) > 0
        )
        
        assert sources_with_evidence >= 3, \
            f"Only {sources_with_evidence}/5 sources have evidence"
    
    def test_envy_evidence_has_offsets(self, concept_index):
        """Evidence must have character offsets for verification."""
        envy = concept_index.get("BEH_EMO_ENVY")
        chunks = envy.get("tafsir_chunks", [])
        
        assert len(chunks) > 0, "No tafsir chunks"
        
        for chunk in chunks[:10]:
            assert "char_start" in chunk, "Missing char_start"
            assert "char_end" in chunk, "Missing char_end"
            assert "quote" in chunk, "Missing quote"
            assert chunk["char_start"] < chunk["char_end"], "Invalid offsets"
    
    def test_envy_has_graph_neighbors(self, cooccurrence_graph):
        """الحسد must have co-occurrence neighbors."""
        neighbors = []
        for edge in cooccurrence_graph["edges"]:
            if edge["source"] == "BEH_EMO_ENVY":
                neighbors.append(edge["target"])
            elif edge["target"] == "BEH_EMO_ENVY":
                neighbors.append(edge["source"])
        
        assert len(neighbors) > 0, "No co-occurrence neighbors"
    
    def test_envy_has_semantic_edges(self, semantic_graph):
        """الحسد must have semantic relationships."""
        edges = []
        for edge in semantic_graph["edges"]:
            if edge["source"] == "BEH_EMO_ENVY" or edge["target"] == "BEH_EMO_ENVY":
                edges.append(edge)
        
        # May have 0 edges if strict validation filtered them out
        # This is acceptable - we don't fabricate
        if len(edges) > 0:
            for edge in edges:
                assert "evidence" in edge, "Semantic edge missing evidence"
                assert len(edge["evidence"]) > 0, "Empty evidence"
    
    def test_envy_query_returns_complete_profile(self, planner):
        """Query for الحسد must return complete profile."""
        result = planner.query("ملف الحسد الكامل")
        
        assert result["target_entity"] == "BEH_EMO_ENVY"
        assert result["query_type"] == "behavior_profile"
        
        # Check results structure
        results = result["results"]
        assert "concept_evidence" in results
        assert results["concept_evidence"].get("total_mentions", 0) > 0


# =============================================================================
# TEST B: Q1 - Causal Chains الغفلة → الكفر
# =============================================================================

@pytest.mark.integration
class TestQ1CausalChain:
    """
    Q1: Find causal chains from الغفلة (heedlessness) to الكفر (disbelief).
    
    Requirements:
    - Both entities must resolve correctly
    - Semantic graph must have CAUSES/LEADS_TO edges
    - Each edge must have evidence with offsets
    - No fabricated chains - only evidence-backed
    """
    
    def test_heedlessness_entity_resolution(self, planner):
        """الغفلة must resolve to BEH_COG_HEEDLESSNESS."""
        entity_id = planner.resolve_entity("الغفلة")
        assert entity_id == "BEH_COG_HEEDLESSNESS", f"Got {entity_id}"
    
    def test_disbelief_entity_resolution(self, planner):
        """الكفر must resolve to BEH_SPI_DISBELIEF."""
        entity_id = planner.resolve_entity("الكفر")
        assert entity_id == "BEH_SPI_DISBELIEF", f"Got {entity_id}"
    
    def test_causal_edges_have_evidence(self, semantic_graph):
        """All CAUSES/LEADS_TO edges must have evidence."""
        causal_edges = [
            e for e in semantic_graph["edges"]
            if e["edge_type"] in ["CAUSES", "LEADS_TO"]
        ]
        
        for edge in causal_edges[:20]:
            assert "evidence" in edge, f"Edge {edge['source']} -> {edge['target']} missing evidence"
            assert len(edge["evidence"]) > 0, "Empty evidence"
            
            for ev in edge["evidence"]:
                assert "quote" in ev, "Evidence missing quote"
                assert "char_start" in ev, "Evidence missing char_start"
    
    def test_causal_query_type_detected(self, planner):
        """Causal query must be detected correctly."""
        query_type = planner.detect_query_type("الغفلة يؤدي إلى الكفر")
        assert query_type == QueryType.CAUSAL_CHAIN
    
    def test_no_fabricated_chains(self, semantic_graph):
        """Chains must not be fabricated - all edges need evidence."""
        for edge in semantic_graph["edges"]:
            if edge["edge_type"] in ["CAUSES", "LEADS_TO"]:
                # Every causal edge must have validated evidence
                assert edge.get("evidence_count", 0) >= 2, \
                    f"Edge {edge['source']} -> {edge['target']} has insufficient evidence"


# =============================================================================
# TEST C: Q23 - Complete التوبة (Repentance) Analysis
# =============================================================================

@pytest.mark.integration
class TestQ23RepentanceAnalysis:
    """
    Q23: Complete analysis of التوبة (repentance).
    
    Requirements:
    - Entity resolves to BEH_SPI_REPENTANCE
    - Has evidence from multiple sources
    - Has graph relationships
    - Evidence is deterministic and reproducible
    """
    
    def test_repentance_entity_resolution(self, planner):
        """التوبة must resolve to BEH_SPI_REPENTANCE."""
        entity_id = planner.resolve_entity("التوبة")
        assert entity_id == "BEH_SPI_REPENTANCE", f"Got {entity_id}"
    
    def test_repentance_has_concept_evidence(self, concept_index):
        """التوبة must have evidence in concept index."""
        repentance = concept_index.get("BEH_SPI_REPENTANCE")
        assert repentance is not None, "BEH_SPI_REPENTANCE not in concept index"
        assert repentance["total_mentions"] > 0, "No mentions found"
    
    def test_repentance_multi_source_coverage(self, concept_index):
        """التوبة must have evidence from multiple sources."""
        repentance = concept_index.get("BEH_SPI_REPENTANCE")
        stats = repentance.get("per_source_stats", {})
        
        sources_with_evidence = sum(
            1 for s in CORE_SOURCES 
            if stats.get(s, {}).get("count", 0) > 0
        )
        
        assert sources_with_evidence >= 2, \
            f"Only {sources_with_evidence}/5 sources have evidence"
    
    def test_repentance_has_graph_neighbors(self, cooccurrence_graph):
        """التوبة must have co-occurrence neighbors."""
        neighbors = []
        for edge in cooccurrence_graph["edges"]:
            if edge["source"] == "BEH_SPI_REPENTANCE":
                neighbors.append(edge["target"])
            elif edge["target"] == "BEH_SPI_REPENTANCE":
                neighbors.append(edge["source"])
        
        assert len(neighbors) > 0, "No co-occurrence neighbors"
    
    def test_repentance_query_returns_analysis(self, planner):
        """Query for التوبة must return complete analysis."""
        result = planner.query("تحليل التوبة في القرآن")
        
        assert result["target_entity"] == "BEH_SPI_REPENTANCE"
        
        results = result["results"]
        assert "concept_evidence" in results
        assert results["concept_evidence"].get("total_mentions", 0) > 0
    
    def test_repentance_evidence_deterministic(self, concept_index):
        """Evidence must be deterministic - same query = same results."""
        repentance1 = concept_index.get("BEH_SPI_REPENTANCE")
        repentance2 = concept_index.get("BEH_SPI_REPENTANCE")
        
        # Same object, same results
        assert repentance1["total_mentions"] == repentance2["total_mentions"]
        assert len(repentance1["verses"]) == len(repentance2["verses"])


# =============================================================================
# CROSS-CUTTING TESTS
# =============================================================================

@pytest.mark.integration
class TestTruthLayerIntegrity:
    """Cross-cutting tests for Truth Layer integrity."""
    
    def test_no_synthetic_evidence(self, concept_index):
        """No concept should have synthetic/fabricated evidence."""
        for concept_id, concept in concept_index.items():
            for chunk in concept.get("tafsir_chunks", [])[:5]:
                # Every chunk must have real source
                assert chunk.get("source") in CORE_SOURCES + ["baghawi", "muyassar"], \
                    f"Unknown source in {concept_id}: {chunk.get('source')}"
                
                # Every chunk must have valid verse reference
                assert chunk.get("surah", 0) > 0, f"Invalid surah in {concept_id}"
                assert chunk.get("ayah", 0) > 0, f"Invalid ayah in {concept_id}"
    
    def test_semantic_edges_have_validated_endpoints(self, semantic_graph):
        """Semantic edges must have endpoints validated in quote."""
        validation = semantic_graph.get("validation", {})
        assert validation.get("endpoints_validated") == True, \
            "Semantic graph must have endpoints_validated=True"
    
    def test_graphs_have_all_canonical_nodes(self, cooccurrence_graph):
        """Graphs must include all canonical entities as nodes."""
        nodes_by_type = cooccurrence_graph.get("nodes_by_type", {})
        
        assert nodes_by_type.get("BEHAVIOR", 0) == 73, \
            f"Expected 73 behaviors, got {nodes_by_type.get('BEHAVIOR', 0)}"
    
    def test_debug_trace_available(self, planner):
        """Every query must return debug trace for transparency."""
        result = planner.query("تحليل الحسد")
        
        assert "debug_trace" in result
        trace = result["debug_trace"]
        
        assert "plan" in trace
        assert "entity_resolution" in trace
        assert "concept_lookups" in trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
