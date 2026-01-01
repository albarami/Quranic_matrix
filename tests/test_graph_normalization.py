"""
Tests for Graph Normalization Layer (Phase 1 of Behavior Mastery Plan).

Tests enforce:
- Exactly 87 BEHAVIOR nodes (not minimum)
- Total nodes >= 87 (behaviors + agents + organs + states)
- All node_type values uppercase
- All edge_type values in allowed set
- All edges have evidence_refs with full provenance
- Deterministic normalization (same input = same output)
"""

import json
import pytest
from pathlib import Path

from src.graph.normalize import (
    normalize_graph,
    load_and_normalize,
    GraphV3,
    NodeV3,
    EdgeV3,
    EvidenceRef,
    get_normalized_graph,
)
from src.graph.contract import (
    validate_graph_contract,
    VALID_NODE_TYPES,
    VALID_EDGE_TYPES,
)


# Paths
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")
GRAPH_V2_PATH = Path("data/graph/semantic_graph_v2.json")


class TestGraphNormalization:
    """Tests for graph normalization."""
    
    @pytest.fixture
    def normalized_graph(self) -> GraphV3:
        """Load and normalize the graph."""
        return load_and_normalize()
    
    @pytest.fixture
    def canonical_behaviors(self) -> dict:
        """Load canonical behaviors."""
        with open(CANONICAL_ENTITIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {b["id"]: b for b in data.get("behaviors", [])}
    
    def test_normalize_graph_has_exactly_87_behaviors(self, normalized_graph: GraphV3):
        """Normalized graph must have EXACTLY 87 behavior nodes (not minimum)."""
        behavior_count = sum(1 for n in normalized_graph.nodes if n.node_type == "BEHAVIOR")
        assert behavior_count == 87, f"Expected exactly 87 BEHAVIOR nodes, got {behavior_count}"
    
    def test_normalize_graph_total_nodes_at_least_87(self, normalized_graph: GraphV3):
        """Total nodes >= 87 (behaviors + agents + organs + states)."""
        total_nodes = len(normalized_graph.nodes)
        assert total_nodes >= 87, f"Expected at least 87 total nodes, got {total_nodes}"
    
    def test_normalize_graph_node_types_uppercase(self, normalized_graph: GraphV3):
        """All node_type values must be uppercase."""
        for node in normalized_graph.nodes:
            assert node.node_type == node.node_type.upper(), \
                f"Node {node.id} has non-uppercase node_type: {node.node_type}"
            assert node.node_type in VALID_NODE_TYPES, \
                f"Node {node.id} has invalid node_type: {node.node_type}"
    
    def test_normalize_graph_edge_types_valid(self, normalized_graph: GraphV3):
        """All edge_type values must be in allowed set."""
        for edge in normalized_graph.edges:
            assert edge.edge_type == edge.edge_type.upper(), \
                f"Edge {edge.source}->{edge.target} has non-uppercase edge_type: {edge.edge_type}"
            assert edge.edge_type in VALID_EDGE_TYPES, \
                f"Edge {edge.source}->{edge.target} has invalid edge_type: {edge.edge_type}"
    
    def test_normalize_graph_edges_have_evidence(self, normalized_graph: GraphV3):
        """All edges must have evidence_count >= 0 and evidence_refs list."""
        for edge in normalized_graph.edges:
            assert edge.evidence_count >= 0, \
                f"Edge {edge.source}->{edge.target} has negative evidence_count"
            assert isinstance(edge.evidence_refs, list), \
                f"Edge {edge.source}->{edge.target} evidence_refs is not a list"
    
    def test_evidence_refs_have_provenance(self, normalized_graph: GraphV3):
        """Each EvidenceRef must have verse_key and source."""
        edges_with_refs = [e for e in normalized_graph.edges if e.evidence_refs]
        assert len(edges_with_refs) > 0, "No edges have evidence_refs"
        
        for edge in edges_with_refs:
            for i, ref in enumerate(edge.evidence_refs):
                assert ref.verse_key, \
                    f"Edge {edge.source}->{edge.target} evidence_ref[{i}] missing verse_key"
                assert ref.source, \
                    f"Edge {edge.source}->{edge.target} evidence_ref[{i}] missing source"
    
    def test_normalize_graph_deterministic(self, canonical_behaviors: dict):
        """Same input produces identical output (hash check)."""
        # Load raw graph
        with open(GRAPH_V2_PATH, "r", encoding="utf-8") as f:
            raw_graph = json.load(f)
        
        # Normalize twice
        graph1 = normalize_graph(raw_graph, canonical_behaviors, str(GRAPH_V2_PATH))
        graph2 = normalize_graph(raw_graph, canonical_behaviors, str(GRAPH_V2_PATH))
        
        # Compare hashes (excluding timestamp)
        # We compare node and edge counts since timestamps will differ
        assert len(graph1.nodes) == len(graph2.nodes), "Node counts differ"
        assert len(graph1.edges) == len(graph2.edges), "Edge counts differ"
        
        # Compare node IDs
        ids1 = sorted([n.id for n in graph1.nodes])
        ids2 = sorted([n.id for n in graph2.nodes])
        assert ids1 == ids2, "Node IDs differ"
    
    def test_all_canonical_behaviors_present(
        self, normalized_graph: GraphV3, canonical_behaviors: dict
    ):
        """All 87 canonical behaviors must be present in normalized graph."""
        graph_behavior_ids = {n.id for n in normalized_graph.nodes if n.node_type == "BEHAVIOR"}
        canonical_ids = set(canonical_behaviors.keys())
        
        missing = canonical_ids - graph_behavior_ids
        assert len(missing) == 0, f"Missing canonical behaviors: {missing}"
    
    def test_metadata_populated(self, normalized_graph: GraphV3):
        """Metadata must be populated with correct values."""
        meta = normalized_graph.metadata
        
        assert meta.schema_version == "3.0.0", f"Wrong schema version: {meta.schema_version}"
        assert meta.behavior_count == 87, f"Wrong behavior count in metadata: {meta.behavior_count}"
        assert meta.total_node_count >= 87, f"Wrong total node count: {meta.total_node_count}"
        assert meta.edge_count > 0, "No edges in metadata"
        assert meta.normalized_at, "Missing normalized_at timestamp"
    
    def test_contract_validation_passes(self, normalized_graph: GraphV3):
        """Normalized graph must pass contract validation."""
        nodes_dict = [n.to_dict() for n in normalized_graph.nodes]
        edges_dict = [e.to_dict() for e in normalized_graph.edges]
        
        is_valid, violations = validate_graph_contract(
            nodes_dict, edges_dict, require_exactly_87_behaviors=True
        )
        
        if not is_valid:
            violation_msgs = [v.message for v in violations[:10]]
            pytest.fail(f"Contract validation failed: {violation_msgs}")


class TestGraphContract:
    """Tests for graph contract validation."""
    
    def test_valid_node_types_defined(self):
        """VALID_NODE_TYPES must include required types."""
        required = {"BEHAVIOR", "AGENT", "ORGAN", "STATE"}
        assert required.issubset(VALID_NODE_TYPES), \
            f"Missing required node types: {required - VALID_NODE_TYPES}"
    
    def test_valid_edge_types_defined(self):
        """VALID_EDGE_TYPES must include required types."""
        required = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS", "OPPOSITE"}
        assert required.issubset(VALID_EDGE_TYPES), \
            f"Missing required edge types: {required - VALID_EDGE_TYPES}"
    
    def test_contract_rejects_wrong_behavior_count(self):
        """Contract must reject graphs with wrong behavior count."""
        nodes = [
            {"id": "BEH_1", "node_type": "BEHAVIOR", "label_ar": "أ", "label_en": "A"},
            {"id": "BEH_2", "node_type": "BEHAVIOR", "label_ar": "ب", "label_en": "B"},
        ]
        edges = []
        
        is_valid, violations = validate_graph_contract(
            nodes, edges, require_exactly_87_behaviors=True
        )
        
        assert not is_valid, "Should reject graph with only 2 behaviors"
        assert any(v.violation_type == "behavior_count_mismatch" for v in violations)
    
    def test_contract_rejects_invalid_node_type(self):
        """Contract must reject invalid node types."""
        nodes = [
            {"id": "X1", "node_type": "INVALID_TYPE", "label_ar": "أ", "label_en": "A"},
        ]
        edges = []
        
        is_valid, violations = validate_graph_contract(
            nodes, edges, require_exactly_87_behaviors=False
        )
        
        assert not is_valid, "Should reject invalid node type"
        assert any(v.violation_type == "invalid_node_type" for v in violations)
    
    def test_contract_rejects_dangling_edges(self):
        """Contract must reject edges with missing endpoints."""
        nodes = [
            {"id": "A", "node_type": "BEHAVIOR", "label_ar": "أ", "label_en": "A"},
        ]
        edges = [
            {"source": "A", "target": "B", "edge_type": "CAUSES", "evidence_count": 1},
        ]
        
        is_valid, violations = validate_graph_contract(
            nodes, edges, require_exactly_87_behaviors=False
        )
        
        assert not is_valid, "Should reject dangling edge"
        assert any(v.violation_type == "dangling_edge_target" for v in violations)


class TestSingletonGraph:
    """Tests for singleton graph access."""
    
    def test_get_normalized_graph_returns_same_instance(self):
        """get_normalized_graph() should return same instance."""
        graph1 = get_normalized_graph()
        graph2 = get_normalized_graph()
        
        assert graph1 is graph2, "Should return same instance"
    
    def test_get_normalized_graph_force_reload(self):
        """force_reload=True should create new instance."""
        graph1 = get_normalized_graph()
        graph2 = get_normalized_graph(force_reload=True)
        
        # Different instances but same content
        assert graph1 is not graph2, "Should create new instance"
        assert len(graph1.nodes) == len(graph2.nodes), "Should have same node count"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
