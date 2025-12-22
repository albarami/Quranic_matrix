"""
Test suite for Phase 2: Vector Embeddings.

Tests the QBMVectorStore class with ChromaDB.
"""

import pytest
from pathlib import Path

from src.ai.vectors.qbm_vectors import QBMVectorStore


class TestEmbedding:
    """Test embedding functionality."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_embed_arabic_text(self, store):
        """Test Arabic text embedding."""
        embedding = store.embed("الكبر من أمراض القلب")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_consistency(self, store):
        """Test same text produces same embedding."""
        text = "ختم الله على قلوبهم"
        emb1 = store.embed(text)
        emb2 = store.embed(text)
        
        assert emb1 == emb2

    def test_embed_batch(self, store):
        """Test batch embedding."""
        texts = ["الكبر", "التواضع", "الصبر"]
        embeddings = store.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)


class TestAyatOperations:
    """Test ayat collection operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_add_ayah(self, store):
        """Test adding ayah to collection."""
        store.add_ayah(
            ayah_id="2:7",
            text="ختم الله على قلوبهم وعلى سمعهم",
            metadata={"surah": 2, "ayah": 7}
        )
        
        assert store.ayat.count() == 1

    def test_add_multiple_ayat(self, store):
        """Test batch adding ayat."""
        ayat = [
            {"id": "2:6", "text": "إن الذين كفروا سواء عليهم", "metadata": {"surah": 2, "ayah": 6}},
            {"id": "2:7", "text": "ختم الله على قلوبهم", "metadata": {"surah": 2, "ayah": 7}},
            {"id": "2:8", "text": "ومن الناس من يقول آمنا", "metadata": {"surah": 2, "ayah": 8}},
        ]
        
        count = store.add_ayat_batch(ayat)
        
        assert count == 3
        assert store.ayat.count() == 3

    def test_search_ayat(self, store):
        """Test searching ayat."""
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2, "ayah": 7})
        store.add_ayah("2:10", "في قلوبهم مرض", {"surah": 2, "ayah": 10})
        store.add_ayah("3:14", "زين للناس حب الشهوات", {"surah": 3, "ayah": 14})
        
        results = store.search_ayat("القلب", n=2)
        
        assert "ids" in results
        assert len(results["ids"][0]) <= 2


class TestBehaviorOperations:
    """Test behavior collection operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_add_behavior(self, store):
        """Test adding behavior to collection."""
        store.add_behavior(
            behavior_id="BEH_COG_ARROGANCE",
            name_ar="الكبر",
            name_en="Arrogance",
            definition="التعالي على الناس",
            metadata={"category": "cognitive"}
        )
        
        assert store.behaviors.count() == 1

    def test_search_behaviors(self, store):
        """Test searching behaviors."""
        store.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "التعالي")
        store.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "خفض الجناح")
        store.add_behavior("BEH_EMO_PATIENCE", "الصبر", "Patience", "تحمل المشاق")
        
        results = store.search_behaviors("الكبر", n=2)
        
        assert "ids" in results
        assert len(results["ids"][0]) <= 2

    def test_add_behaviors_from_graph(self, store, tmp_path):
        """Test loading behaviors from graph."""
        from src.ai.graph.qbm_graph import QBMKnowledgeGraph
        
        graph = QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))
        graph.add_behavior("BEH_1", "سلوك1", "Behavior1", "cat1")
        graph.add_behavior("BEH_2", "سلوك2", "Behavior2", "cat2")
        
        count = store.add_behaviors_from_graph(graph)
        
        assert count == 2
        assert store.behaviors.count() == 2


class TestTafsirOperations:
    """Test tafsir collection operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_add_tafsir(self, store):
        """Test adding tafsir entry."""
        store.add_tafsir(
            tafsir_id="ibn_kathir_2_7",
            text="قال ابن كثير في تفسير هذه الآية...",
            surah=2,
            ayah=7,
            source="ibn_kathir"
        )
        
        assert store.tafsir.count() == 1

    def test_search_tafsir(self, store):
        """Test searching tafsir."""
        store.add_tafsir("ibn_kathir_2_7", "الختم على القلوب", 2, 7, "ibn_kathir")
        store.add_tafsir("qurtubi_2_7", "معنى الختم في اللغة", 2, 7, "qurtubi")
        
        results = store.search_tafsir("الختم", n=2)
        
        assert "ids" in results
        assert len(results["ids"][0]) <= 2

    def test_search_tafsir_by_source(self, store):
        """Test filtering tafsir by source."""
        store.add_tafsir("ibn_kathir_2_7", "تفسير ابن كثير", 2, 7, "ibn_kathir")
        store.add_tafsir("qurtubi_2_7", "تفسير القرطبي", 2, 7, "qurtubi")
        
        results = store.search_tafsir("تفسير", source="ibn_kathir", n=10)
        
        # Should only return ibn_kathir results
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                assert meta.get("source") == "ibn_kathir"


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_get_collection_stats(self, store):
        """Test getting collection statistics."""
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2})
        store.add_behavior("BEH_1", "سلوك", "Behavior", "")
        
        stats = store.get_collection_stats()
        
        assert stats["ayat"] == 1
        assert stats["behaviors"] == 1
        assert stats["tafsir"] == 0

    def test_clear_collection(self, store):
        """Test clearing a collection."""
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2})
        store.add_ayah("2:8", "ومن الناس", {"surah": 2})
        
        assert store.ayat.count() == 2
        
        store.clear_collection("ayat")
        
        assert store.ayat.count() == 0


# Run with: pytest tests/ai/test_vectors.py -v
