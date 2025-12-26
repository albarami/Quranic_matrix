"""
Test: Evidence-Backed Graphs (Phase 6.2)

Ensures graphs follow hard rules:
- No semantic edge without evidence offsets
- No causal chain may use co-occurrence edges
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

COOCCURRENCE_FILE = Path("data/graph/cooccurrence_graph_v1.json")
SEMANTIC_FILE = Path("data/graph/semantic_graph_v1.json")

SEMANTIC_EDGE_TYPES = [
    "CAUSES", "LEADS_TO", "PREVENTS", 
    "OPPOSITE_OF", "COMPLEMENTS", "CONDITIONAL_ON"
]


def load_cooccurrence_graph():
    with open(COOCCURRENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_semantic_graph():
    with open(SEMANTIC_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class TestGraphFilesExist:
    """Tests that graph files exist."""
    
    def test_cooccurrence_graph_exists(self):
        """cooccurrence_graph_v1.json must exist."""
        assert COOCCURRENCE_FILE.exists()
    
    def test_semantic_graph_exists(self):
        """semantic_graph_v1.json must exist."""
        assert SEMANTIC_FILE.exists()


class TestCooccurrenceGraph:
    """Tests for co-occurrence graph."""
    
    @pytest.fixture(scope="class")
    def graph(self):
        return load_cooccurrence_graph()
    
    def test_graph_type_is_cooccurrence(self, graph):
        """Graph type must be 'cooccurrence'."""
        assert graph['graph_type'] == 'cooccurrence'
    
    def test_edge_type_is_co_occurs_with(self, graph):
        """All edges must be CO_OCCURS_WITH."""
        for edge in graph['edges']:
            assert edge['edge_type'] == 'CO_OCCURS_WITH'
    
    def test_edges_have_count_and_pmi(self, graph):
        """Edges must have count and PMI scores."""
        for edge in graph['edges'][:10]:
            assert 'count' in edge
            assert 'pmi' in edge
            assert edge['count'] >= 3  # Minimum threshold
    
    def test_no_causal_reasoning_on_cooccurrence_graph(self, graph):
        """Co-occurrence graph must NOT have causal edge types."""
        for edge in graph['edges']:
            assert edge['edge_type'] not in SEMANTIC_EDGE_TYPES, \
                f"Found causal edge type in co-occurrence graph: {edge['edge_type']}"
    
    def test_graph_marked_for_discovery_only(self, graph):
        """Graph description must indicate discovery-only use."""
        assert 'discovery' in graph.get('description', '').lower() or \
               'NOT for causal' in graph.get('description', '')


class TestSemanticGraph:
    """Tests for semantic graph."""
    
    @pytest.fixture(scope="class")
    def graph(self):
        return load_semantic_graph()
    
    def test_graph_type_is_semantic(self, graph):
        """Graph type must be 'semantic'."""
        assert graph['graph_type'] == 'semantic'
    
    def test_semantic_edges_require_evidence_offsets(self, graph):
        """Every semantic edge must have evidence with offsets."""
        for edge in graph['edges']:
            assert 'evidence' in edge, f"Edge {edge['source']} -> {edge['target']} missing evidence"
            assert len(edge['evidence']) > 0, f"Edge {edge['source']} -> {edge['target']} has empty evidence"
            
            for ev in edge['evidence']:
                assert 'char_start' in ev, f"Evidence missing char_start"
                assert 'char_end' in ev, f"Evidence missing char_end"
                assert 'quote' in ev, f"Evidence missing quote"
                assert ev['char_start'] >= 0
                assert ev['char_end'] > ev['char_start']
    
    def test_edges_have_typed_edge_types(self, graph):
        """All edges must have valid semantic edge types."""
        for edge in graph['edges']:
            assert edge['edge_type'] in SEMANTIC_EDGE_TYPES, \
                f"Invalid edge type: {edge['edge_type']}"
    
    def test_edges_have_confidence(self, graph):
        """All edges must have confidence scores."""
        for edge in graph['edges']:
            assert 'confidence' in edge
            assert 0 <= edge['confidence'] <= 1
    
    def test_edges_have_extractor_version(self, graph):
        """All edges must have extractor_version for traceability."""
        for edge in graph['edges']:
            assert 'extractor_version' in edge
    
    def test_evidence_has_source_and_verse(self, graph):
        """Evidence must include source and verse reference."""
        for edge in graph['edges']:
            for ev in edge['evidence']:
                assert 'source' in ev
                assert 'surah' in ev
                assert 'ayah' in ev
                assert 'chunk_id' in ev


class TestGraphHardRules:
    """Tests for graph hard rules."""
    
    def test_no_semantic_edge_without_evidence(self):
        """Hard rule: No semantic edge without evidence offsets."""
        graph = load_semantic_graph()
        
        edges_without_evidence = []
        for edge in graph['edges']:
            if not edge.get('evidence') or len(edge['evidence']) == 0:
                edges_without_evidence.append(f"{edge['source']} -> {edge['target']}")
        
        assert len(edges_without_evidence) == 0, \
            f"Found edges without evidence: {edges_without_evidence}"
    
    def test_cooccurrence_not_used_for_causal(self):
        """Hard rule: Co-occurrence edges cannot be used for causal reasoning."""
        cooc = load_cooccurrence_graph()
        
        # Verify co-occurrence graph only has statistical edges
        for edge in cooc['edges']:
            assert edge['edge_type'] == 'CO_OCCURS_WITH'
            assert 'CAUSES' not in str(edge)
            assert 'LEADS_TO' not in str(edge)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
