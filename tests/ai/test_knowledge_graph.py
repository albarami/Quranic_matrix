"""
Test suite for Phase 1: Knowledge Graph.

Tests the QBMKnowledgeGraph class with NetworkX + SQLite persistence.
"""

import json
import pytest
from pathlib import Path

from src.ai.graph.qbm_graph import QBMKnowledgeGraph


class TestNodeOperations:
    """Test node creation and retrieval."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))

    def test_add_behavior_node(self, graph):
        """Test adding a behavior node."""
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        
        assert "BEH_COG_ARROGANCE" in graph.G.nodes
        assert graph.G.nodes["BEH_COG_ARROGANCE"]["name_ar"] == "الكبر"
        assert graph.G.nodes["BEH_COG_ARROGANCE"]["name_en"] == "Arrogance"
        assert graph.G.nodes["BEH_COG_ARROGANCE"]["category"] == "cognitive"
        assert graph.G.nodes["BEH_COG_ARROGANCE"]["node_type"] == "Behavior"

    def test_add_ayah_node(self, graph):
        """Test adding an ayah node."""
        graph.add_ayah(2, 7, "ختم الله على قلوبهم وعلى سمعهم")
        
        assert "2:7" in graph.G.nodes
        assert graph.G.nodes["2:7"]["surah"] == 2
        assert graph.G.nodes["2:7"]["ayah"] == 7
        assert graph.G.nodes["2:7"]["node_type"] == "Ayah"

    def test_add_agent_node(self, graph):
        """Test adding an agent node."""
        graph.add_agent("AGENT_BELIEVER", "مؤمن", "human")
        
        assert "AGENT_BELIEVER" in graph.G.nodes
        assert graph.G.nodes["AGENT_BELIEVER"]["name_ar"] == "مؤمن"

    def test_add_organ_node(self, graph):
        """Test adding an organ node."""
        graph.add_organ("ORGAN_HEART", "قلب", "Heart")
        
        assert "ORGAN_HEART" in graph.G.nodes
        assert graph.G.nodes["ORGAN_HEART"]["name_ar"] == "قلب"

    def test_add_generic_node(self, graph):
        """Test adding a generic node with type."""
        graph.add_node("CONSEQ_HELLFIRE", "Consequence", name_ar="جهنم", name_en="Hellfire")
        
        assert "CONSEQ_HELLFIRE" in graph.G.nodes
        assert graph.G.nodes["CONSEQ_HELLFIRE"]["node_type"] == "Consequence"

    def test_invalid_node_type_raises(self, graph):
        """Test that invalid node type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid node type"):
            graph.add_node("TEST", "InvalidType")

    def test_get_node(self, graph):
        """Test retrieving node attributes."""
        graph.add_behavior("BEH_TEST", "اختبار", "Test", "test")
        
        node = graph.get_node("BEH_TEST")
        assert node is not None
        assert node["name_ar"] == "اختبار"

    def test_get_nonexistent_node(self, graph):
        """Test retrieving non-existent node returns None."""
        assert graph.get_node("NONEXISTENT") is None

    def test_get_nodes_by_type(self, graph):
        """Test filtering nodes by type."""
        graph.add_behavior("BEH_1", "سلوك1", "Behavior1", "cat1")
        graph.add_behavior("BEH_2", "سلوك2", "Behavior2", "cat2")
        graph.add_agent("AGENT_1", "فاعل", "human")
        
        behaviors = graph.get_nodes_by_type("Behavior")
        agents = graph.get_nodes_by_type("Agent")
        
        assert len(behaviors) == 2
        assert len(agents) == 1


class TestEdgeOperations:
    """Test edge/relationship operations."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))

    def test_add_causal_relationship(self, graph):
        """Test adding CAUSES edge."""
        graph.add_behavior("BEH_COG_HEEDLESSNESS", "الغفلة", "Heedlessness", "cognitive")
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_relationship("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES")
        
        edges = list(graph.G.edges(data=True))
        assert len(edges) == 1
        assert edges[0][0] == "BEH_COG_HEEDLESSNESS"
        assert edges[0][1] == "BEH_COG_ARROGANCE"
        assert edges[0][2]["edge_type"] == "CAUSES"

    def test_add_causal_with_confidence(self, graph):
        """Test adding causal relationship with confidence score."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_causal_relationship("BEH_A", "BEH_B", confidence=0.85)
        
        edges = graph.get_relationships("BEH_A", "CAUSES", "out")
        assert len(edges) == 1
        assert edges[0][2]["confidence"] == 0.85

    def test_opposite_relationship_bidirectional(self, graph):
        """Test OPPOSITE_OF creates bidirectional edge."""
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "cognitive")
        graph.add_opposite_relationship("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY")
        
        assert graph.G.has_edge("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY")
        assert graph.G.has_edge("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE")

    def test_invalid_edge_type_raises(self, graph):
        """Test that invalid edge type raises ValueError."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        
        with pytest.raises(ValueError, match="Invalid edge type"):
            graph.add_relationship("BEH_A", "BEH_B", "INVALID_TYPE")

    def test_get_relationships_outgoing(self, graph):
        """Test getting outgoing relationships."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_behavior("BEH_C", "ج", "C", "test")
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        graph.add_relationship("BEH_A", "BEH_C", "SIMILAR_TO")
        
        rels = graph.get_relationships("BEH_A", direction="out")
        assert len(rels) == 2

    def test_get_relationships_incoming(self, graph):
        """Test getting incoming relationships."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        
        rels = graph.get_relationships("BEH_B", direction="in")
        assert len(rels) == 1
        assert rels[0][0] == "BEH_A"

    def test_get_relationships_filtered(self, graph):
        """Test filtering relationships by type."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_behavior("BEH_C", "ج", "C", "test")
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        graph.add_relationship("BEH_A", "BEH_C", "SIMILAR_TO")
        
        causes = graph.get_relationships("BEH_A", rel_type="CAUSES", direction="out")
        assert len(causes) == 1
        assert causes[0][1] == "BEH_B"


class TestPathFinding:
    """Test path finding and causal chain discovery."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))

    def test_find_causal_chain(self, graph):
        """Test finding causal chain between behaviors."""
        # Build chain: الغفلة → الكبر → الظلم
        graph.add_behavior("BEH_COG_HEEDLESSNESS", "الغفلة", "Heedlessness", "cognitive")
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_SOC_OPPRESSION", "الظلم", "Oppression", "social")
        graph.add_relationship("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES")
        graph.add_relationship("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION", "RESULTS_IN")
        
        chains = graph.find_causal_chain("BEH_COG_HEEDLESSNESS", "BEH_SOC_OPPRESSION")
        assert len(chains) >= 1
        assert "BEH_COG_ARROGANCE" in chains[0]

    def test_find_all_paths(self, graph):
        """Test finding all paths between nodes."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_behavior("BEH_C", "ج", "C", "test")
        graph.add_relationship("BEH_A", "BEH_B", "RELATED")
        graph.add_relationship("BEH_B", "BEH_C", "RELATED")
        graph.add_relationship("BEH_A", "BEH_C", "RELATED")  # Direct path
        
        paths = graph.find_all_paths("BEH_A", "BEH_C")
        assert len(paths) == 2  # Direct and via B

    def test_find_chain_no_path(self, graph):
        """Test finding chain when no path exists."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        # No edge between them
        
        chains = graph.find_causal_chain("BEH_A", "BEH_B")
        assert len(chains) == 0

    def test_find_chain_nonexistent_node(self, graph):
        """Test finding chain with non-existent node."""
        graph.add_behavior("BEH_A", "أ", "A", "test")
        
        chains = graph.find_causal_chain("BEH_A", "NONEXISTENT")
        assert len(chains) == 0


class TestAnalytics:
    """Test graph analytics functions."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))

    def test_hub_behaviors_centrality(self, graph):
        """Test finding hub behaviors."""
        # Create hub structure: HUB connects to 5 spokes
        graph.add_behavior("HUB", "مركز", "Hub", "test")
        for i in range(5):
            graph.add_behavior(f"SPOKE_{i}", f"فرع_{i}", f"Spoke_{i}", "test")
            graph.add_relationship("HUB", f"SPOKE_{i}", "RELATED")
        
        hubs = graph.get_hub_behaviors(top_n=1)
        # Hub should have highest centrality
        assert len(hubs) >= 1

    def test_community_detection(self, graph):
        """Test behavioral clustering."""
        # Create two distinct clusters
        for i in range(3):
            graph.add_behavior(f"CLUSTER_A_{i}", f"أ_{i}", f"A_{i}", "test")
        for i in range(3):
            graph.add_behavior(f"CLUSTER_B_{i}", f"ب_{i}", f"B_{i}", "test")
        
        # Connect within clusters
        graph.add_relationship("CLUSTER_A_0", "CLUSTER_A_1", "RELATED")
        graph.add_relationship("CLUSTER_A_1", "CLUSTER_A_2", "RELATED")
        graph.add_relationship("CLUSTER_B_0", "CLUSTER_B_1", "RELATED")
        graph.add_relationship("CLUSTER_B_1", "CLUSTER_B_2", "RELATED")
        
        communities = graph.find_communities()
        assert len(communities) >= 2

    def test_behavior_statistics(self, graph):
        """Test getting behavior statistics."""
        graph.add_behavior("BEH_1", "سلوك1", "Behavior1", "cognitive")
        graph.add_behavior("BEH_2", "سلوك2", "Behavior2", "cognitive")
        graph.add_behavior("BEH_3", "سلوك3", "Behavior3", "emotional")
        graph.add_relationship("BEH_1", "BEH_2", "CAUSES")
        graph.add_relationship("BEH_2", "BEH_3", "RESULTS_IN")
        
        stats = graph.get_behavior_statistics()
        
        assert stats["total_behaviors"] == 3
        assert stats["total_edges"] == 2
        assert stats["categories"]["cognitive"] == 2
        assert stats["categories"]["emotional"] == 1
        assert stats["edge_types"]["CAUSES"] == 1
        assert stats["edge_types"]["RESULTS_IN"] == 1


class TestPersistence:
    """Test graph persistence to SQLite."""

    def test_save_and_load(self, tmp_path):
        """Test graph persistence to SQLite."""
        db_path = str(tmp_path / "test_graph.db")
        
        # Create and populate graph
        graph1 = QBMKnowledgeGraph(db_path=db_path)
        graph1.add_behavior("BEH_TEST", "اختبار", "Test", "test")
        graph1.add_behavior("BEH_TEST2", "اختبار2", "Test2", "test")
        graph1.add_relationship("BEH_TEST", "BEH_TEST2", "CAUSES")
        graph1.save()
        
        # Create new graph and load
        graph2 = QBMKnowledgeGraph(db_path=db_path)
        graph2.load()
        
        assert "BEH_TEST" in graph2.G.nodes
        assert "BEH_TEST2" in graph2.G.nodes
        assert graph2.G.has_edge("BEH_TEST", "BEH_TEST2")
        assert graph2.G.nodes["BEH_TEST"]["name_ar"] == "اختبار"

    def test_save_arabic_text(self, tmp_path):
        """Test that Arabic text is preserved correctly."""
        db_path = str(tmp_path / "test_graph.db")
        
        graph1 = QBMKnowledgeGraph(db_path=db_path)
        graph1.add_behavior("BEH_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph1.add_ayah(2, 7, "ختم الله على قلوبهم وعلى سمعهم")
        graph1.save()
        
        graph2 = QBMKnowledgeGraph(db_path=db_path)
        graph2.load()
        
        assert graph2.G.nodes["BEH_ARROGANCE"]["name_ar"] == "الكبر"
        assert "ختم الله" in graph2.G.nodes["2:7"]["text_uthmani"]

    def test_export_graphml(self, tmp_path):
        """Test exporting to GraphML format."""
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        
        export_path = str(tmp_path / "export.graphml")
        graph.export_graphml(export_path)
        
        assert Path(export_path).exists()


class TestBulkLoading:
    """Test bulk loading from vocabulary files."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))

    def test_load_behaviors_from_vocab(self, graph):
        """Test loading behaviors from vocab file."""
        vocab_path = Path("vocab/behavior_concepts.json")
        if not vocab_path.exists():
            pytest.skip("vocab/behavior_concepts.json not found")
        
        count = graph.load_behaviors_from_vocab(str(vocab_path))
        
        assert count > 0
        behaviors = graph.get_nodes_by_type("Behavior")
        assert len(behaviors) == count

    def test_load_all_behaviors(self, graph):
        """Test loading all behaviors from vocab."""
        vocab_path = Path("vocab/behavior_concepts.json")
        if not vocab_path.exists():
            pytest.skip("vocab/behavior_concepts.json not found")
        
        count = graph.load_behaviors_from_vocab(str(vocab_path))
        
        # Should have at least 70 behaviors (current vocab has 73)
        assert count >= 70

    def test_loaded_behaviors_have_arabic(self, graph):
        """Test that loaded behaviors have Arabic names."""
        vocab_path = Path("vocab/behavior_concepts.json")
        if not vocab_path.exists():
            pytest.skip("vocab/behavior_concepts.json not found")
        
        graph.load_behaviors_from_vocab(str(vocab_path))
        behaviors = graph.get_nodes_by_type("Behavior")
        
        # Check that at least some behaviors have Arabic names
        arabic_count = sum(1 for _, attrs in behaviors if attrs.get("name_ar"))
        assert arabic_count > 0


# Run with: pytest tests/ai/test_knowledge_graph.py -v
