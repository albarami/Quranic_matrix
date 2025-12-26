"""
Test: Hybrid Evidence Retrieval

Step 3 Gate: These tests MUST pass before proceeding to Step 4.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.hybrid_evidence_retriever import (
    HybridEvidenceRetriever,
    ChunkedEvidenceIndex,
    get_hybrid_retriever,
    CORE_SOURCES,
)

CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")


@pytest.fixture(scope="module")
def retriever():
    """Get a configured hybrid retriever (BM25 only for speed)."""
    if not CHUNKED_INDEX_FILE.exists():
        pytest.skip("Chunked index not built yet")
    
    return get_hybrid_retriever(use_bm25=True, use_dense=False)


def test_chunked_index_loads():
    """Chunked index should load successfully."""
    if not CHUNKED_INDEX_FILE.exists():
        pytest.skip("Chunked index not built yet")
    
    index = ChunkedEvidenceIndex()
    index.load()
    
    assert len(index.chunks) > 80000, f"Expected 80K+ chunks, got {len(index.chunks)}"


def test_deterministic_retrieval_by_verse_key(retriever):
    """Deterministic retrieval should return exact matches for verse references."""
    response = retriever.search("2:255")
    
    assert response.deterministic_count > 0, "Should find deterministic results for 2:255"
    
    # All deterministic results should have verse_key = "2:255"
    for r in response.results:
        if r.retrieval_method == 'deterministic':
            assert r.verse_key == "2:255", f"Expected verse_key 2:255, got {r.verse_key}"


def test_deterministic_returns_multiple_sources(retriever):
    """Deterministic retrieval should return results from multiple sources."""
    response = retriever.search("2:255")
    
    sources_found = set()
    for r in response.results:
        if r.retrieval_method == 'deterministic':
            sources_found.add(r.source)
    
    # Should have at least 3 different sources (diversity filter may limit some)
    assert len(sources_found) >= 3, f"Expected at least 3 sources, got {len(sources_found)}: {sources_found}"


def test_bm25_retrieval_returns_results(retriever):
    """BM25 retrieval should return results for keyword queries."""
    response = retriever.search("الصبر على البلاء")
    
    assert response.bm25_count > 0, "Should find BM25 results for keyword query"
    assert len(response.results) > 0, "Should have results"


def test_no_synthetic_evidence_on_empty_query(retriever):
    """Empty or nonsense queries should return empty results, not synthetic evidence."""
    response = retriever.search("xyz123 غير موجود أبداً")
    
    # Should not fabricate evidence
    assert response.fallback_used == False, "Should not use fallback"
    
    # Results may be empty or low-quality, but should not be synthetic
    for r in response.results:
        assert r.chunk_id.startswith(('ibn_kathir_', 'tabari_', 'qurtubi_', 'saadi_', 'jalalayn_', 'baghawi_', 'muyassar_')), \
            f"Result should have valid chunk_id format, got {r.chunk_id}"


def test_no_synthetic_edges_in_results(retriever):
    """Results should not contain fabricated graph edges or paths."""
    response = retriever.search("الكبر والنفاق")
    
    # Check that results are real tafsir chunks, not synthetic
    for r in response.results:
        assert len(r.text) > 10, "Result text should be substantial"
        assert r.source in CORE_SOURCES + ['baghawi', 'muyassar'], \
            f"Source should be a real tafsir, got {r.source}"


def test_retrieval_response_structure(retriever):
    """Retrieval response should have correct structure."""
    response = retriever.search("2:255")
    
    # Check response structure
    assert hasattr(response, 'query')
    assert hasattr(response, 'results')
    assert hasattr(response, 'deterministic_count')
    assert hasattr(response, 'bm25_count')
    assert hasattr(response, 'fallback_used')
    
    # Check result structure
    if response.results:
        r = response.results[0]
        assert hasattr(r, 'chunk_id')
        assert hasattr(r, 'verse_key')
        assert hasattr(r, 'source')
        assert hasattr(r, 'text')
        assert hasattr(r, 'score')
        assert hasattr(r, 'retrieval_method')


def test_to_dict_serialization(retriever):
    """Response should serialize to dict correctly."""
    response = retriever.search("2:255")
    
    d = response.to_dict()
    
    assert 'query' in d
    assert 'results' in d
    assert 'counts' in d
    assert 'fallback_used' in d
    
    assert d['counts']['deterministic'] == response.deterministic_count
    assert d['counts']['total'] == len(response.results)


def test_source_diversity(retriever):
    """Results should include multiple sources, not just one."""
    response = retriever.search("الإيمان والتوبة")
    
    sources = set(r.source for r in response.results)
    
    assert len(sources) >= 3, f"Expected at least 3 sources, got {len(sources)}: {sources}"


def test_fallback_used_is_false_for_normal_queries(retriever):
    """Normal queries should not trigger fallback."""
    response = retriever.search("2:255")
    
    assert response.fallback_used == False, "Normal queries should not use fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
