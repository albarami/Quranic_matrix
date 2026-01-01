#!/usr/bin/env python3
"""
Phase 7 Tests: Graph as SSOT Projection

These tests validate that:
1. Graph file exists and has correct structure
2. All behaviors from registry are in graph
3. Edges have correct provenance
4. Patience behavior has non-zero edges
5. Node labels are in Arabic and English

Run with: pytest tests/phase7/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.behavior_registry import get_behavior_registry, clear_registry
from src.core.data_profile import expected_behavior_count, is_fixture_mode


# ============================================================================
# Graph File Tests
# ============================================================================

class TestGraphFile:
    """Test graph file exists and has correct structure."""

    @pytest.fixture
    def graph(self):
        """Load graph from file."""
        graph_path = Path("data/graph/graph_v3.json")
        assert graph_path.exists(), "graph_v3.json not found"
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_graph_exists(self, graph):
        """Test that graph exists and loads."""
        assert graph is not None

    def test_graph_has_version(self, graph):
        """Test that graph has version field."""
        assert "version" in graph
        assert graph["version"] == "3.0"

    def test_graph_has_statistics(self, graph):
        """Test that graph has statistics."""
        assert "statistics" in graph
        stats = graph["statistics"]
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "nodes_by_type" in stats
        assert "edges_by_type" in stats

    def test_graph_has_nodes(self, graph):
        """Test that graph has nodes."""
        assert "nodes" in graph
        assert len(graph["nodes"]) > 0

    def test_graph_has_edges(self, graph):
        """Test that graph has edges."""
        assert "edges" in graph
        assert len(graph["edges"]) > 0


# ============================================================================
# Node Tests
# ============================================================================

class TestGraphNodes:
    """Test graph node structure and content."""

    @pytest.fixture
    def graph(self):
        """Load graph from file."""
        graph_path = Path("data/graph/graph_v3.json")
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_nodes_have_required_fields(self, graph):
        """Test that all nodes have required fields."""
        required_fields = ["id", "type", "labelAr", "labelEn"]

        for node in graph["nodes"]:
            for field in required_fields:
                assert field in node, f"Node {node.get('id', 'unknown')} missing {field}"

    def test_behavior_nodes_have_arabic_labels(self, graph):
        """Test that behavior nodes have Arabic labels."""
        behavior_nodes = [n for n in graph["nodes"] if n["type"] == "behavior"]

        for node in behavior_nodes:
            assert node["labelAr"], f"Behavior {node['id']} has empty Arabic label"

    def test_behavior_nodes_match_registry(self, graph):
        """Test that all registry behaviors are in graph."""
        clear_registry()
        registry = get_behavior_registry()
        registry_ids = {b.behavior_id for b in registry.get_all()}

        behavior_node_ids = {n["id"] for n in graph["nodes"] if n["type"] == "behavior"}

        missing = registry_ids - behavior_node_ids
        assert len(missing) == 0, f"Behaviors missing from graph: {missing}"

    def test_verse_nodes_have_correct_format(self, graph):
        """Test that verse nodes have VERSE_surah_ayah format."""
        verse_nodes = [n for n in graph["nodes"] if n["type"] == "verse"]

        for node in verse_nodes:
            node_id = node["id"]
            assert node_id.startswith("VERSE_"), f"Invalid verse ID format: {node_id}"
            parts = node_id.split("_")
            assert len(parts) == 3, f"Invalid verse ID format: {node_id}"


# ============================================================================
# Edge Tests
# ============================================================================

class TestGraphEdges:
    """Test graph edge structure and content."""

    @pytest.fixture
    def graph(self):
        """Load graph from file."""
        graph_path = Path("data/graph/graph_v3.json")
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_edges_have_required_fields(self, graph):
        """Test that all edges have required fields."""
        required_fields = ["source", "target", "type", "evidenceType"]

        for edge in graph["edges"]:
            for field in required_fields:
                assert field in edge, \
                    f"Edge {edge.get('source', '?')}->{edge.get('target', '?')} missing {field}"

    def test_edges_have_provenance(self, graph):
        """Test that edges have provenance."""
        for edge in graph["edges"]:
            assert "provenance" in edge

    def test_mentioned_in_edges_point_to_verses(self, graph):
        """Test that MENTIONED_IN edges target verse nodes."""
        mentioned_in_edges = [e for e in graph["edges"] if e["type"] == "MENTIONED_IN"]

        for edge in mentioned_in_edges:
            target = edge["target"]
            assert target.startswith("VERSE_"), \
                f"MENTIONED_IN edge targets non-verse: {target}"


# ============================================================================
# Patience Tests
# ============================================================================

class TestPatienceGraph:
    """Test patience behavior in graph specifically."""

    @pytest.fixture
    def graph(self):
        """Load graph from file."""
        graph_path = Path("data/graph/graph_v3.json")
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_patience_node_exists(self, graph):
        """Test that patience node exists."""
        patience_nodes = [n for n in graph["nodes"] if n["id"] == "BEH_EMO_PATIENCE"]
        assert len(patience_nodes) == 1

    def test_patience_has_arabic_label(self, graph):
        """Test that patience has Arabic label."""
        patience_node = next(n for n in graph["nodes"] if n["id"] == "BEH_EMO_PATIENCE")
        assert patience_node["labelAr"] == "الصبر"

    def test_patience_has_edges(self, graph):
        """Test that patience has non-zero edges."""
        patience_edges = [e for e in graph["edges"] if e["source"] == "BEH_EMO_PATIENCE"]
        assert len(patience_edges) > 0, "Patience has no edges"

    def test_patience_has_reasonable_edge_count(self, graph):
        """Test that patience has reasonable number of edges."""
        patience_edges = [e for e in graph["edges"] if e["source"] == "BEH_EMO_PATIENCE"]
        # Should have 60-120 edges (same as verse count)
        assert len(patience_edges) >= 60, f"Too few patience edges: {len(patience_edges)}"
        assert len(patience_edges) <= 120, f"Too many patience edges: {len(patience_edges)}"

    def test_patience_edges_are_lexical(self, graph):
        """Test that patience edges are lexical type."""
        patience_edges = [e for e in graph["edges"] if e["source"] == "BEH_EMO_PATIENCE"]

        for edge in patience_edges:
            assert edge["evidenceType"] == "lexical", \
                f"Patience edge to {edge['target']} is not lexical"


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestGraphSSotAlignment:
    """Test that graph aligns with SSOT sources."""

    @pytest.fixture
    def graph(self):
        """Load graph from file."""
        graph_path = Path("data/graph/graph_v3.json")
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @pytest.fixture
    def concept_index(self):
        """Load concept index."""
        entries = {}
        with open("data/evidence/concept_index_v3.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entries[entry["concept_id"]] = entry
        return entries

    def test_edge_count_matches_concept_index(self, graph, concept_index):
        """Test that edge count is consistent with concept index."""
        # Total edges should be related to total verses across all behaviors
        total_index_verses = sum(
            len(entry.get("verses", [])) for entry in concept_index.values()
        )
        total_graph_edges = len(graph["edges"])

        # Graph edges should be non-trivial and related to index verses
        assert total_graph_edges > 0, "Graph has no edges"
        # Allow some variance due to edge deduplication
        assert total_graph_edges >= total_index_verses * 0.5, \
            f"Graph has too few edges: {total_graph_edges} vs {total_index_verses} index verses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
