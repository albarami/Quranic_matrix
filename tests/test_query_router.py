"""
Test: QueryRouter and RoutedEvidenceRetriever

Phase 5.5 Gate: These tests MUST pass before commit.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.query_router import QueryRouter, QueryIntent, get_query_router
from src.ml.routed_evidence_retriever import RoutedEvidenceRetriever, get_routed_retriever

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


class TestQueryRouter:
    """Tests for QueryRouter intent classification."""
    
    @pytest.fixture
    def router(self):
        return get_query_router()
    
    def test_ayah_ref_numeric(self, router):
        """Numeric verse references should be classified as AYAH_REF."""
        result = router.route("2:255")
        assert result.intent == QueryIntent.AYAH_REF
        assert result.extracted_ref == "2:255"
        assert result.surah_num == 2
        assert result.ayah_num == 255
    
    def test_ayah_ref_arabic_name(self, router):
        """Arabic surah name + ayah should be classified as AYAH_REF."""
        result = router.route("البقرة:255")
        assert result.intent == QueryIntent.AYAH_REF
        assert result.extracted_ref == "2:255"
    
    def test_ayah_ref_in_question(self, router):
        """Verse reference in a question should be classified as AYAH_REF."""
        result = router.route("ما تفسير الآية 2:255")
        assert result.intent == QueryIntent.AYAH_REF
        assert result.extracted_ref == "2:255"
    
    def test_surah_ref_with_keyword(self, router):
        """سورة + name should be classified as SURAH_REF."""
        result = router.route("سورة البقرة")
        assert result.intent == QueryIntent.SURAH_REF
        assert result.surah_num == 2
    
    def test_surah_ref_exact_match(self, router):
        """Exact surah name should be classified as SURAH_REF."""
        result = router.route("الفاتحة")
        assert result.intent == QueryIntent.SURAH_REF
        assert result.surah_num == 1
    
    def test_concept_ref_with_id(self, router):
        """Concept IDs should be classified as CONCEPT_REF."""
        result = router.route("BEH_SABR")
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.concept_term == "BEH_SABR"
    
    def test_concept_ref_arabic_term(self, router):
        """Arabic behavior terms should be classified as CONCEPT_REF."""
        result = router.route("الصبر")
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.concept_term == "صبر"
    
    def test_concept_ref_with_keyword(self, router):
        """Behavior term with keyword should be classified as CONCEPT_REF."""
        result = router.route("آيات التقوى")
        assert result.intent == QueryIntent.CONCEPT_REF
    
    def test_free_text_open_question(self, router):
        """Open questions should be classified as FREE_TEXT."""
        result = router.route("كيف أتعامل مع الابتلاء")
        assert result.intent == QueryIntent.FREE_TEXT
    
    def test_free_text_complex_query(self, router):
        """Complex queries without specific refs should be FREE_TEXT."""
        result = router.route("العلاقة بين الإيمان والعمل الصالح")
        assert result.intent == QueryIntent.FREE_TEXT
    
    def test_no_false_surah_match(self, router):
        """Behavior terms should not match single-letter surahs."""
        # الصبر should NOT match surah ص
        result = router.route("الصبر")
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.surah_num is None


class TestRoutedEvidenceRetriever:
    """Tests for RoutedEvidenceRetriever."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        return get_routed_retriever()
    
    def test_ayah_ref_full_coverage(self, retriever):
        """AYAH_REF queries should achieve 5/5 core source coverage."""
        response = retriever.search("2:255", top_k=20)
        
        assert response.intent == QueryIntent.AYAH_REF
        assert response.retrieval_mode == "deterministic_ayah"
        assert response.core_sources_count == 5, f"Expected 5/5 coverage, got {response.core_sources_count}"
        
        for source in CORE_SOURCES:
            assert source in response.sources_covered, f"Missing source: {source}"
    
    def test_surah_ref_has_results(self, retriever):
        """SURAH_REF queries should return results."""
        response = retriever.search("سورة الفاتحة", top_k=20)
        
        assert response.intent == QueryIntent.SURAH_REF
        assert response.retrieval_mode == "deterministic_surah"
        assert len(response.results) > 0
    
    def test_concept_ref_has_results(self, retriever):
        """CONCEPT_REF queries should return results."""
        response = retriever.search("الصبر", top_k=20)
        
        assert response.intent == QueryIntent.CONCEPT_REF
        assert len(response.results) > 0
    
    def test_free_text_uses_hybrid(self, retriever):
        """FREE_TEXT queries should use hybrid retrieval."""
        response = retriever.search("كيف أتعامل مع الابتلاء", top_k=20)
        
        assert response.intent == QueryIntent.FREE_TEXT
        assert response.retrieval_mode == "hybrid_free_text"
    
    def test_no_synthetic_evidence(self, retriever):
        """Results should never contain synthetic evidence."""
        response = retriever.search("2:255", top_k=20)
        
        for r in response.results:
            # All results should have valid chunk IDs from real sources
            assert r.chunk_id is not None
            assert r.source in CORE_SOURCES + ["baghawi", "muyassar"]
            assert len(r.text) > 0
    
    def test_fallback_not_used_for_valid_queries(self, retriever):
        """Valid queries should not trigger fallback."""
        response = retriever.search("2:255", top_k=20)
        
        assert response.fallback_used == False


class TestCoverageGate:
    """Gate tests for Phase 5.5 coverage requirements."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        return get_routed_retriever()
    
    def test_ayah_ref_coverage_gate(self, retriever):
        """AYAH_REF must achieve ≥85% 5/5 coverage on benchmark verses."""
        # Test on a sample of benchmark verses
        test_verses = [
            "2:255", "1:1", "112:1", "2:1", "3:1",
            "4:1", "5:1", "18:1", "36:1", "55:1",
        ]
        
        full_coverage_count = 0
        for verse_key in test_verses:
            response = retriever.search(verse_key, top_k=20)
            if response.core_sources_count == 5:
                full_coverage_count += 1
        
        coverage_rate = full_coverage_count / len(test_verses)
        assert coverage_rate >= 0.85, f"Coverage rate {coverage_rate:.1%} < 85% target"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
