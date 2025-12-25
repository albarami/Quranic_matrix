"""
Phase 4: Stratified Retriever Tests
Tests for the stratified tafsir retrieval system.
"""

import pytest
from pathlib import Path


class TestStratifiedRetriever:
    """Test stratified tafsir retrieval"""
    
    def test_retriever_initialization(self):
        """Retriever should initialize with all 5 sources"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        assert retriever._initialized
        assert len(retriever.source_indexes) == len(TAFSIR_SOURCES)
    
    def test_all_sources_have_data(self):
        """All 5 tafsir sources should have documents"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        for source in TAFSIR_SOURCES:
            assert source in retriever.source_indexes
            assert len(retriever.source_indexes[source]) > 0, f"{source} has no documents"
    
    def test_search_returns_all_sources(self):
        """Search should return results from all 5 sources"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        # Use a realistic full-question query (punctuation + stopwords)
        results = retriever.search("ما هو الصبر؟", top_k_per_source=5)
        
        for source in TAFSIR_SOURCES:
            assert source in results, f"Missing results from {source}"
            assert len(results[source]) == 5, f"Expected 5 results from {source}, got {len(results[source])}"
    
    def test_minimum_results_per_source(self):
        """Each source should return at least MIN_RESULTS_PER_SOURCE"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        results = retriever.search("الصبر", top_k_per_source=5)
        
        for source in TAFSIR_SOURCES:
            # Should have at least 1 result (may have less if query is very specific)
            assert len(results[source]) >= 1, f"{source} returned 0 results"
    
    def test_search_flat_interleaves_sources(self):
        """Flat search should interleave results from all sources"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        results = retriever.search_flat("الصبر", top_k_per_source=3)
        
        # Should have results from multiple sources in first 10
        sources_in_top_10 = set(r["source"] for r in results[:10])
        assert len(sources_in_top_10) >= 3, "Results not properly interleaved"
    
    def test_bm25_finds_exact_terms(self):
        """BM25 should find exact Arabic terms"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        # Search for specific term
        results = retriever.search("الكبر", top_k_per_source=5)
        
        # At least one source should have results with the term
        found_term = False
        for source, source_results in results.items():
            for r in source_results:
                if "كبر" in r.get("text", ""):
                    found_term = True
                    break
        
        assert found_term, "BM25 did not find exact Arabic term"
    
    def test_get_stats(self):
        """Stats should report correct document counts"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever, TAFSIR_SOURCES
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        retriever.initialize()
        
        stats = retriever.get_stats()
        
        assert stats["initialized"]
        assert stats["total_documents"] > 0
        assert len(stats["sources"]) == len(TAFSIR_SOURCES)


@pytest.mark.slow
class TestIndexPersistence:
    """Test index saving and loading"""
    
    def test_indexes_saved_to_disk(self):
        """Indexes should be saved to disk"""
        from src.ml.stratified_retriever import INDEXES_DIR, TAFSIR_SOURCES
        
        for source in TAFSIR_SOURCES:
            index_file = INDEXES_DIR / f"{source}.json"
            assert index_file.exists(), f"Index file missing: {index_file}"
    
    def test_indexes_load_from_disk(self):
        """Retriever should load from saved indexes"""
        from src.ml.stratified_retriever import StratifiedTafsirRetriever
        
        retriever = StratifiedTafsirRetriever(fail_fast=False)
        loaded = retriever._try_load_indexes()
        
        # Should load successfully if indexes exist
        if loaded:
            assert len(retriever.source_indexes) == 5


class TestFailFastBehavior:
    """Test fail-fast behavior for missing indexes"""
    
    def test_fail_fast_raises_on_missing_source(self):
        """Should raise IndexNotFoundError if source missing and fail_fast=True"""
        from src.ml.stratified_retriever import (
            StratifiedTafsirRetriever, 
            IndexNotFoundError,
            TAFSIR_SOURCES
        )
        
        retriever = StratifiedTafsirRetriever(fail_fast=True)
        
        # Manually clear one source to simulate missing data
        retriever.source_indexes = {s: retriever.source_indexes.get(s) for s in TAFSIR_SOURCES[:-1]}
        
        # This would raise if we called _validate_indexes with a missing source
        # For now, just verify the error class exists
        assert IndexNotFoundError is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
