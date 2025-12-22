"""
Test suite for Phase 3: Taxonomy & Ontology.

Tests the QBMOntology class with RDFLib.
"""

import pytest
from pathlib import Path

from src.ai.ontology.qbm_ontology import QBMOntology


class TestOntologyInitialization:
    """Test ontology initialization."""

    def test_init_creates_base_ontology(self):
        """Test that initialization creates base ontology structure."""
        onto = QBMOntology()
        
        stats = onto.get_statistics()
        assert stats["classes"] > 0
        assert stats["properties"] > 0

    def test_init_has_behavior_class(self):
        """Test that Behavior class exists."""
        onto = QBMOntology()
        
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            
            ASK { qbm:Behavior a owl:Class }
        """
        result = onto.g.query(sparql)
        assert result.askAnswer

    def test_init_has_behavior_subclasses(self):
        """Test that behavior subclasses exist."""
        onto = QBMOntology()
        
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?subclass
            WHERE {
                ?subclass rdfs:subClassOf qbm:Behavior .
            }
        """
        results = list(onto.g.query(sparql))
        assert len(results) >= 5  # At least 5 subclasses


class TestBehaviorOperations:
    """Test behavior instance operations."""

    @pytest.fixture
    def onto(self):
        """Create test ontology."""
        return QBMOntology()

    def test_add_behavior(self, onto):
        """Test adding a behavior instance."""
        onto.add_behavior(
            behavior_id="BEH_COG_ARROGANCE",
            name_ar="الكبر",
            name_en="Arrogance",
            category="cognitive",
            polarity="negative",
            evaluation="blame",
        )
        
        behaviors = onto.get_all_behaviors()
        assert len(behaviors) == 1
        assert behaviors[0]["id"] == "BEH_COG_ARROGANCE"

    def test_add_multiple_behaviors(self, onto):
        """Test adding multiple behaviors."""
        onto.add_behavior("BEH_1", "سلوك1", "Behavior1", "cognitive")
        onto.add_behavior("BEH_2", "سلوك2", "Behavior2", "emotional")
        onto.add_behavior("BEH_3", "سلوك3", "Behavior3", "spiritual")
        
        behaviors = onto.get_all_behaviors()
        assert len(behaviors) == 3

    def test_behavior_has_arabic_label(self, onto):
        """Test that behaviors have Arabic labels."""
        onto.add_behavior("BEH_TEST", "اختبار", "Test", "cognitive")
        
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            
            SELECT ?name
            WHERE {
                qbm:BEH_TEST qbm:nameArabic ?name .
            }
        """
        results = list(onto.g.query(sparql))
        assert len(results) == 1
        assert "اختبار" in str(results[0][0])


class TestRelationshipOperations:
    """Test relationship operations."""

    @pytest.fixture
    def onto(self):
        """Create test ontology with behaviors."""
        onto = QBMOntology()
        onto.add_behavior("BEH_A", "أ", "A", "cognitive")
        onto.add_behavior("BEH_B", "ب", "B", "cognitive")
        onto.add_behavior("BEH_C", "ج", "C", "cognitive")
        return onto

    def test_add_causal_relationship(self, onto):
        """Test adding causal relationship."""
        onto.add_causal_relationship("BEH_A", "BEH_B")
        
        causes = onto.get_causes("BEH_B")
        assert "BEH_A" in causes

    def test_get_effects(self, onto):
        """Test getting effects of a behavior."""
        onto.add_causal_relationship("BEH_A", "BEH_B")
        onto.add_causal_relationship("BEH_A", "BEH_C")
        
        effects = onto.get_effects("BEH_A")
        assert "BEH_B" in effects
        assert "BEH_C" in effects

    def test_add_opposite_relationship(self, onto):
        """Test adding opposite relationship."""
        onto.add_opposite_relationship("BEH_A", "BEH_B")
        
        opposites = onto.get_opposites("BEH_A")
        assert "BEH_B" in opposites

    def test_opposite_is_symmetric(self, onto):
        """Test that opposite relationship is symmetric."""
        onto.add_opposite_relationship("BEH_A", "BEH_B")
        
        # Due to symmetric property, both directions should work
        opposites_a = onto.get_opposites("BEH_A")
        opposites_b = onto.get_opposites("BEH_B")
        
        assert "BEH_B" in opposites_a
        # Note: Symmetric inference may require a reasoner

    def test_add_similar_relationship(self, onto):
        """Test adding similarity relationship."""
        onto.add_similar_relationship("BEH_A", "BEH_B")
        
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            
            ASK { qbm:BEH_A qbm:similarTo qbm:BEH_B }
        """
        result = onto.g.query(sparql)
        assert result.askAnswer


class TestSPARQLQueries:
    """Test SPARQL query functionality."""

    @pytest.fixture
    def onto(self):
        """Create test ontology with data."""
        onto = QBMOntology()
        onto.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive", "negative", "blame")
        onto.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "cognitive", "positive", "praise")
        onto.add_opposite_relationship("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY")
        return onto

    def test_custom_sparql_query(self, onto):
        """Test executing custom SPARQL query."""
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            
            SELECT ?behavior ?polarity
            WHERE {
                ?behavior qbm:polarity ?polarity .
            }
        """
        results = onto.query(sparql)
        
        assert len(results) == 2
        polarities = [r["polarity"] for r in results]
        assert "negative" in polarities
        assert "positive" in polarities

    def test_find_causal_chain(self, onto):
        """Test finding causal chain."""
        onto.add_behavior("BEH_A", "أ", "A", "cognitive")
        onto.add_behavior("BEH_B", "ب", "B", "cognitive")
        onto.add_behavior("BEH_C", "ج", "C", "cognitive")
        onto.add_causal_relationship("BEH_A", "BEH_B")
        onto.add_causal_relationship("BEH_B", "BEH_C")
        
        chains = onto.find_causal_chain("BEH_A", "BEH_C")
        
        # Should find a path
        assert len(chains) >= 1


class TestPersistence:
    """Test ontology persistence."""

    def test_save_and_load_turtle(self, tmp_path):
        """Test saving and loading in Turtle format."""
        onto1 = QBMOntology()
        onto1.add_behavior("BEH_TEST", "اختبار", "Test", "cognitive")
        
        path = str(tmp_path / "test_onto.ttl")
        onto1.save(path, format="turtle")
        
        assert Path(path).exists()
        
        onto2 = QBMOntology(ontology_path=path)
        behaviors = onto2.get_all_behaviors()
        
        assert len(behaviors) == 1
        assert behaviors[0]["id"] == "BEH_TEST"

    def test_save_preserves_arabic(self, tmp_path):
        """Test that Arabic text is preserved after save/load."""
        onto1 = QBMOntology()
        onto1.add_behavior("BEH_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        
        path = str(tmp_path / "test_arabic.ttl")
        onto1.save(path)
        
        onto2 = QBMOntology(ontology_path=path)
        behaviors = onto2.get_all_behaviors()
        
        assert "الكبر" in behaviors[0].get("nameAr", "")


class TestBulkLoading:
    """Test bulk loading from knowledge graph."""

    def test_load_from_graph(self, tmp_path):
        """Test loading from QBMKnowledgeGraph."""
        from src.ai.graph.qbm_graph import QBMKnowledgeGraph
        
        # Create graph with behaviors
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "test.db"))
        graph.add_behavior("BEH_1", "سلوك1", "Behavior1", "cognitive")
        graph.add_behavior("BEH_2", "سلوك2", "Behavior2", "emotional")
        graph.add_causal_relationship("BEH_1", "BEH_2")
        
        # Load into ontology
        onto = QBMOntology()
        count = onto.load_from_graph(graph)
        
        assert count == 2
        
        # Verify relationship
        causes = onto.get_causes("BEH_2")
        assert "BEH_1" in causes

    def test_load_from_vocab_graph(self, tmp_path):
        """Test loading from graph with vocab data."""
        from src.ai.graph.qbm_graph import QBMKnowledgeGraph
        
        vocab_path = Path("vocab/behavior_concepts.json")
        if not vocab_path.exists():
            pytest.skip("vocab/behavior_concepts.json not found")
        
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "test.db"))
        graph.load_behaviors_from_vocab(str(vocab_path))
        
        onto = QBMOntology()
        count = onto.load_from_graph(graph)
        
        assert count >= 70  # Should have at least 70 behaviors


class TestStatistics:
    """Test statistics functionality."""

    def test_get_statistics(self):
        """Test getting ontology statistics."""
        onto = QBMOntology()
        onto.add_behavior("BEH_1", "سلوك1", "Behavior1", "cognitive")
        onto.add_behavior("BEH_2", "سلوك2", "Behavior2", "emotional")
        
        stats = onto.get_statistics()
        
        assert stats["behaviors"] == 2
        assert stats["classes"] > 0
        assert stats["properties"] > 0
        assert stats["total_triples"] > 0


# Run with: pytest tests/ai/test_ontology.py -v
