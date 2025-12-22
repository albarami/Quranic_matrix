"""
Test suite for Phase 2: RAG Pipeline.

Tests the QBMRAGPipeline class with Azure OpenAI.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.ai.rag.qbm_rag import QBMRAGPipeline
from src.ai.vectors.qbm_vectors import QBMVectorStore
from src.ai.graph.qbm_graph import QBMKnowledgeGraph


class TestRAGInitialization:
    """Test RAG pipeline initialization."""

    @pytest.fixture
    def mock_azure(self):
        """Mock Azure OpenAI client."""
        with patch("src.ai.rag.qbm_rag.AzureOpenAI") as mock:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock.return_value = mock_client
            yield mock

    def test_init_with_defaults(self, tmp_path, mock_azure):
        """Test initialization with default components."""
        with patch.dict("os.environ", {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        }):
            rag = QBMRAGPipeline()
            
            assert rag.vector_store is not None
            assert rag.graph is not None

    def test_init_with_custom_components(self, tmp_path):
        """Test initialization with custom components."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        rag = QBMRAGPipeline(vector_store=store, graph=graph)
        
        assert rag.vector_store is store
        assert rag.graph is graph


class TestGraphExpansion:
    """Test graph context expansion."""

    @pytest.fixture
    def rag(self, tmp_path):
        """Create RAG pipeline with test data."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        # Add test behaviors
        graph.add_behavior("BEH_A", "الغفلة", "Heedlessness", "cognitive")
        graph.add_behavior("BEH_B", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_C", "الظلم", "Oppression", "social")
        graph.add_behavior("BEH_D", "التواضع", "Humility", "cognitive")
        
        # Add relationships
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        graph.add_relationship("BEH_B", "BEH_C", "RESULTS_IN")
        graph.add_relationship("BEH_B", "BEH_D", "OPPOSITE_OF")
        graph.add_relationship("BEH_D", "BEH_B", "OPPOSITE_OF")
        
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_expand_graph_context_causes(self, rag):
        """Test expanding causes from graph."""
        expanded = rag._expand_graph_context(["BEH_B"])
        
        assert "الغفلة" in expanded["causes"]

    def test_expand_graph_context_effects(self, rag):
        """Test expanding effects from graph."""
        expanded = rag._expand_graph_context(["BEH_B"])
        
        assert "الظلم" in expanded["effects"]

    def test_expand_graph_context_opposites(self, rag):
        """Test expanding opposites from graph."""
        expanded = rag._expand_graph_context(["BEH_B"])
        
        assert "التواضع" in expanded["opposites"]

    def test_expand_nonexistent_behavior(self, rag):
        """Test expanding non-existent behavior."""
        expanded = rag._expand_graph_context(["NONEXISTENT"])
        
        assert expanded["causes"] == []
        assert expanded["effects"] == []


class TestContextBuilding:
    """Test context building for LLM."""

    @pytest.fixture
    def rag(self, tmp_path):
        """Create RAG pipeline."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_build_context_with_ayat(self, rag):
        """Test building context with ayat."""
        ayat = {
            "ids": [["2:7", "2:10"]],
            "documents": [["ختم الله على قلوبهم", "في قلوبهم مرض"]],
            "metadatas": [[{"surah": 2}, {"surah": 2}]],
        }
        
        context = rag._build_context(ayat, {}, {}, {})
        
        assert "الآيات" in context
        assert "2:7" in context

    def test_build_context_with_expansion(self, rag):
        """Test building context with graph expansion."""
        expanded = {
            "causes": ["الغفلة"],
            "effects": ["الظلم"],
            "opposites": ["التواضع"],
        }
        
        context = rag._build_context({}, {}, expanded, {})
        
        assert "الأسباب" in context
        assert "الغفلة" in context
        assert "النتائج" in context


class TestQueryInterface:
    """Test main query interface."""

    @pytest.fixture
    def rag_with_data(self, tmp_path):
        """Create RAG pipeline with test data."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        # Add test data
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        store.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "التعالي")
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2, "ayah": 7})
        
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_query_returns_structure(self, rag_with_data):
        """Test that query returns expected structure."""
        result = rag_with_data.query("ما هو الكبر؟")
        
        assert "answer" in result
        assert "sources" in result
        assert "graph_expansion" in result
        assert "context_used" in result

    def test_query_sources_structure(self, rag_with_data):
        """Test that sources have correct structure."""
        result = rag_with_data.query("الكبر")
        
        assert "ayat" in result["sources"]
        assert "behaviors" in result["sources"]
        assert "tafsir" in result["sources"]


class TestBehaviorAnalysis:
    """Test behavior analysis functionality."""

    @pytest.fixture
    def rag_with_behaviors(self, tmp_path):
        """Create RAG pipeline with behaviors."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        # Add behaviors with relationships
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "cognitive")
        graph.add_opposite_relationship("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY")
        
        store.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "")
        store.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "")
        
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_analyze_behavior_found(self, rag_with_behaviors):
        """Test analyzing existing behavior."""
        result = rag_with_behaviors.analyze_behavior("الكبر")
        
        assert "behavior_id" in result
        assert "opposites" in result
        assert result["behavior_id"] == "BEH_COG_ARROGANCE"

    def test_analyze_behavior_not_found(self, rag_with_behaviors):
        """Test analyzing non-existent behavior."""
        result = rag_with_behaviors.analyze_behavior("سلوك غير موجود")
        
        # Should still return a result (might find closest match)
        assert "behavior_id" in result or "error" in result


class TestChainDiscovery:
    """Test causal chain discovery."""

    @pytest.fixture
    def rag_with_chains(self, tmp_path):
        """Create RAG pipeline with causal chains."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        # Build chain: A → B → C
        graph.add_behavior("BEH_A", "أ", "A", "test")
        graph.add_behavior("BEH_B", "ب", "B", "test")
        graph.add_behavior("BEH_C", "ج", "C", "test")
        graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        graph.add_relationship("BEH_B", "BEH_C", "CAUSES")
        
        store.add_behavior("BEH_A", "أ", "A", "")
        
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_discover_chains_by_id(self, rag_with_chains):
        """Test discovering chains by behavior ID."""
        chains = rag_with_chains.discover_chains("BEH_A", max_depth=3)
        
        # Should find chain to BEH_C
        assert any("BEH_C" in chain for chain in chains)

    def test_discover_chains_empty(self, rag_with_chains):
        """Test discovering chains from isolated behavior."""
        rag_with_chains.graph.add_behavior("BEH_ISOLATED", "معزول", "Isolated", "test")
        
        chains = rag_with_chains.discover_chains("BEH_ISOLATED")
        
        assert chains == []


class TestStats:
    """Test statistics functionality."""

    @pytest.fixture
    def rag(self, tmp_path):
        """Create RAG pipeline."""
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "graph.db"))
        
        graph.add_behavior("BEH_1", "سلوك", "Behavior", "test")
        store.add_ayah("2:7", "آية", {"surah": 2})
        
        return QBMRAGPipeline(vector_store=store, graph=graph)

    def test_get_stats(self, rag):
        """Test getting pipeline statistics."""
        stats = rag.get_stats()
        
        assert "vector_store" in stats
        assert "graph" in stats
        assert "llm_available" in stats
        assert "deployment" in stats


# Run with: pytest tests/ai/test_rag.py -v
