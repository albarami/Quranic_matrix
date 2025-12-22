"""
Test suite for Phase 4: Multi-Tafsir Integration.

Tests the CrossTafsirAnalyzer class.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ai.tafsir.cross_tafsir import CrossTafsirAnalyzer


class TestTafsirInitialization:
    """Test tafsir analyzer initialization."""

    def test_init_creates_cache_dir(self, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "tafsir_cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        
        assert cache_dir.exists()

    def test_get_available_sources(self):
        """Test getting available tafsir sources."""
        analyzer = CrossTafsirAnalyzer()
        sources = analyzer.get_available_sources()
        
        assert "ibn_kathir" in sources
        assert "tabari" in sources
        assert "qurtubi" in sources
        assert len(sources) >= 5

    def test_get_source_info(self):
        """Test getting source information."""
        analyzer = CrossTafsirAnalyzer()
        info = analyzer.get_source_info("ibn_kathir")
        
        assert info is not None
        assert "name_ar" in info
        assert "name_en" in info
        assert "id" in info


class TestAyatCounts:
    """Test ayat count functionality."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create test analyzer."""
        return CrossTafsirAnalyzer(cache_dir=str(tmp_path / "cache"))

    def test_get_ayat_count_fatiha(self, analyzer):
        """Test ayat count for Al-Fatiha."""
        assert analyzer.get_ayat_count(1) == 7

    def test_get_ayat_count_baqara(self, analyzer):
        """Test ayat count for Al-Baqara."""
        assert analyzer.get_ayat_count(2) == 286

    def test_get_ayat_count_nas(self, analyzer):
        """Test ayat count for An-Nas."""
        assert analyzer.get_ayat_count(114) == 6

    def test_get_ayat_count_invalid(self, analyzer):
        """Test ayat count for invalid surah."""
        assert analyzer.get_ayat_count(0) == 0
        assert analyzer.get_ayat_count(115) == 0


class TestTafsirRetrieval:
    """Test tafsir retrieval functionality."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create test analyzer with cache."""
        cache_dir = tmp_path / "cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        
        # Create mock cached tafsir
        ibn_kathir_dir = cache_dir / "ibn_kathir"
        ibn_kathir_dir.mkdir(parents=True)
        
        import json
        with open(ibn_kathir_dir / "2_7.json", "w", encoding="utf-8") as f:
            json.dump({
                "surah": 2,
                "ayah": 7,
                "source": "ibn_kathir",
                "source_name": "تفسير ابن كثير",
                "text": "قال ابن كثير في تفسير الختم على القلوب..."
            }, f, ensure_ascii=False)
        
        return analyzer

    def test_get_tafsir_from_cache(self, analyzer):
        """Test retrieving tafsir from cache."""
        tafsir = analyzer.get_tafsir(2, 7, "ibn_kathir")
        
        assert tafsir is not None
        assert tafsir["surah"] == 2
        assert tafsir["ayah"] == 7
        assert "الختم" in tafsir["text"]

    def test_get_tafsir_invalid_source(self, analyzer):
        """Test retrieving tafsir with invalid source."""
        tafsir = analyzer.get_tafsir(2, 7, "invalid_source")
        
        assert tafsir is None

    def test_get_all_tafsir(self, analyzer):
        """Test retrieving all tafsir for an ayah."""
        # Add another cached tafsir
        import json
        qurtubi_dir = analyzer.cache_dir / "qurtubi"
        qurtubi_dir.mkdir(parents=True)
        with open(qurtubi_dir / "2_7.json", "w", encoding="utf-8") as f:
            json.dump({
                "surah": 2,
                "ayah": 7,
                "source": "qurtubi",
                "source_name": "تفسير القرطبي",
                "text": "قال القرطبي في معنى الختم..."
            }, f, ensure_ascii=False)
        
        all_tafsir = analyzer.get_all_tafsir(2, 7, ["ibn_kathir", "qurtubi"])
        
        assert "ibn_kathir" in all_tafsir
        assert "qurtubi" in all_tafsir


class TestCrossTafsirAnalysis:
    """Test cross-tafsir analysis functionality."""

    @pytest.fixture
    def analyzer_with_data(self, tmp_path):
        """Create analyzer with multiple tafsir sources."""
        cache_dir = tmp_path / "cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        
        import json
        
        # Ibn Kathir
        ibn_kathir_dir = cache_dir / "ibn_kathir"
        ibn_kathir_dir.mkdir(parents=True)
        with open(ibn_kathir_dir / "2_7.json", "w", encoding="utf-8") as f:
            json.dump({
                "surah": 2,
                "ayah": 7,
                "source": "ibn_kathir",
                "source_name": "تفسير ابن كثير",
                "text": "الختم على القلوب يعني الطبع عليها فلا يدخلها الإيمان"
            }, f, ensure_ascii=False)
        
        # Qurtubi
        qurtubi_dir = cache_dir / "qurtubi"
        qurtubi_dir.mkdir(parents=True)
        with open(qurtubi_dir / "2_7.json", "w", encoding="utf-8") as f:
            json.dump({
                "surah": 2,
                "ayah": 7,
                "source": "qurtubi",
                "source_name": "تفسير القرطبي",
                "text": "الختم في اللغة هو الطبع والغلق على الشيء"
            }, f, ensure_ascii=False)
        
        return analyzer

    def test_find_consensus(self, analyzer_with_data):
        """Test finding consensus across scholars."""
        consensus = analyzer_with_data.find_consensus(2, 7, "الختم", ["ibn_kathir", "qurtubi"])
        
        assert "mentions_topic" in consensus
        assert "ibn_kathir" in consensus["mentions_topic"]
        assert "qurtubi" in consensus["mentions_topic"]

    def test_compare_interpretations(self, analyzer_with_data):
        """Test comparing interpretations."""
        comparison = analyzer_with_data.compare_interpretations(2, 7, ["ibn_kathir", "qurtubi"])
        
        assert "interpretations" in comparison
        assert "ibn_kathir" in comparison["interpretations"]
        assert "qurtubi" in comparison["interpretations"]


class TestBehaviorSearch:
    """Test behavior search in tafsir."""

    @pytest.fixture
    def analyzer_with_behaviors(self, tmp_path):
        """Create analyzer with behavior-related tafsir."""
        cache_dir = tmp_path / "cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        
        import json
        
        ibn_kathir_dir = cache_dir / "ibn_kathir"
        ibn_kathir_dir.mkdir(parents=True)
        
        # Multiple ayat mentioning الكبر
        for i, (surah, ayah) in enumerate([(2, 34), (4, 173), (7, 146)]):
            with open(ibn_kathir_dir / f"{surah}_{ayah}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "surah": surah,
                    "ayah": ayah,
                    "source": "ibn_kathir",
                    "source_name": "تفسير ابن كثير",
                    "text": f"في هذه الآية ذكر الكبر والتكبر على أمر الله"
                }, f, ensure_ascii=False)
        
        return analyzer

    def test_search_behavior_in_tafsir(self, analyzer_with_behaviors):
        """Test searching for behavior in tafsir."""
        results = analyzer_with_behaviors.search_behavior_in_tafsir("الكبر", "ibn_kathir")
        
        assert len(results) >= 3
        assert all("الكبر" in r["text"] for r in results)

    def test_behavioral_emphasis(self, analyzer_with_behaviors):
        """Test analyzing behavioral emphasis."""
        emphasis = analyzer_with_behaviors.behavioral_emphasis("الكبر", ["ibn_kathir"])
        
        assert "ibn_kathir" in emphasis
        assert emphasis["ibn_kathir"]["frequency"] >= 3


class TestStatistics:
    """Test statistics functionality."""

    @pytest.fixture
    def analyzer_with_cache(self, tmp_path):
        """Create analyzer with cached data."""
        cache_dir = tmp_path / "cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        
        import json
        
        # Create some cached files
        ibn_kathir_dir = cache_dir / "ibn_kathir"
        ibn_kathir_dir.mkdir(parents=True)
        for i in range(5):
            with open(ibn_kathir_dir / f"2_{i+1}.json", "w", encoding="utf-8") as f:
                json.dump({"surah": 2, "ayah": i+1, "text": "test"}, f)
        
        qurtubi_dir = cache_dir / "qurtubi"
        qurtubi_dir.mkdir(parents=True)
        for i in range(3):
            with open(qurtubi_dir / f"2_{i+1}.json", "w", encoding="utf-8") as f:
                json.dump({"surah": 2, "ayah": i+1, "text": "test"}, f)
        
        return analyzer

    def test_get_statistics(self, analyzer_with_cache):
        """Test getting cache statistics."""
        stats = analyzer_with_cache.get_statistics()
        
        assert stats["sources"]["ibn_kathir"] == 5
        assert stats["sources"]["qurtubi"] == 3
        assert stats["total_cached"] == 8


class TestVectorStoreIntegration:
    """Test integration with vector store."""

    @pytest.fixture
    def analyzer_and_store(self, tmp_path):
        """Create analyzer and vector store."""
        from src.ai.vectors.qbm_vectors import QBMVectorStore
        
        cache_dir = tmp_path / "cache"
        analyzer = CrossTafsirAnalyzer(cache_dir=str(cache_dir))
        store = QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
        
        import json
        
        # Create cached tafsir
        ibn_kathir_dir = cache_dir / "ibn_kathir"
        ibn_kathir_dir.mkdir(parents=True)
        for i in range(3):
            with open(ibn_kathir_dir / f"2_{i+1}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "surah": 2,
                    "ayah": i+1,
                    "source": "ibn_kathir",
                    "text": f"تفسير الآية {i+1} من سورة البقرة"
                }, f, ensure_ascii=False)
        
        return analyzer, store

    def test_load_into_vector_store(self, analyzer_and_store):
        """Test loading tafsir into vector store."""
        analyzer, store = analyzer_and_store
        
        count = analyzer.load_into_vector_store(store, ["ibn_kathir"])
        
        assert count == 3
        assert store.tafsir.count() == 3


# Run with: pytest tests/ai/test_tafsir.py -v
