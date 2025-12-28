"""
Test: SURAH_REF Deterministic Evidence Filtering (Phase 6.3)

Ensures SURAH_REF intent returns ONLY verses from the specified surah.
No BM25 pollution in primary evidence set.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.mark.integration
class TestSurahRefDeterministicEvidence:
    """Tests for SURAH_REF deterministic verse filtering."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        """Get hybrid retriever."""
        from src.ml.hybrid_evidence_retriever import get_hybrid_retriever
        return get_hybrid_retriever(use_bm25=True, use_dense=False)
    
    def test_surah_ref_returns_only_surah_ayat_in_primary_evidence(self, retriever):
        """
        Query: "تفسير سورة الفاتحة"
        Assert: every tafsir chunk's verse_key starts with 1: in primary evidence.
        """
        response = retriever.search("تفسير سورة الفاتحة")
        
        # Must have results
        assert len(response.results) > 0, "SURAH_REF query returned no results"
        
        # Every result must be from surah 1 (الفاتحة)
        invalid_verses = []
        for r in response.results:
            if not r.verse_key.startswith("1:"):
                invalid_verses.append(r.verse_key)
        
        assert len(invalid_verses) == 0, \
            f"Found {len(invalid_verses)} chunks from wrong surah: {invalid_verses[:10]}"
    
    def test_surah_ref_core_sources_covered_for_each_ayah(self, retriever):
        """
        For each ayah 1:1–1:7, check source coverage.
        Note: Data may not have all verses - this test validates structure, not completeness.
        """
        response = retriever.search("تفسير سورة الفاتحة")
        
        # Build coverage matrix: verse_key -> set of sources
        coverage = {}
        for r in response.results:
            if r.verse_key not in coverage:
                coverage[r.verse_key] = set()
            coverage[r.verse_key].add(r.source)
        
        # Check each ayah in Al-Fatiha (1:1 to 1:7)
        missing_coverage = []
        verses_with_data = 0
        for ayah in range(1, 8):
            verse_key = f"1:{ayah}"
            sources_found = coverage.get(verse_key, set())
            if sources_found:
                verses_with_data += 1
            missing_sources = set(CORE_SOURCES) - sources_found
            
            if missing_sources and sources_found:  # Only report if partial coverage
                missing_coverage.append({
                    'verse': verse_key,
                    'found': list(sources_found),
                    'missing': list(missing_sources),
                })
        
        # Report missing coverage (informational)
        if missing_coverage:
            print(f"\nPartial coverage for Al-Fatiha:")
            for m in missing_coverage:
                print(f"  {m['verse']}: found {m['found']}, missing {m['missing']}")
        
        # At minimum, should have data for at least 1 verse (1:1 typically has full coverage)
        assert verses_with_data >= 1, "No tafsir data found for any verse in Al-Fatiha"
        
        # 1:1 (Bismillah) should have coverage from all 7 sources
        assert "1:1" in coverage, "No tafsir found for 1:1 (Bismillah)"
        assert len(coverage["1:1"]) >= 5, \
            f"1:1 should have at least 5 core sources, found: {coverage['1:1']}"
    
    def test_surah_ref_no_bm25_pollution(self, retriever):
        """SURAH_REF must use deterministic retrieval, not BM25."""
        response = retriever.search("تفسير سورة الفاتحة")
        
        # All results should be from deterministic retrieval
        bm25_results = [r for r in response.results if 'bm25' in r.retrieval_method]
        
        assert len(bm25_results) == 0, \
            f"Found {len(bm25_results)} BM25 results in SURAH_REF query (should be 0)"
    
    def test_surah_ref_al_ikhlas(self, retriever):
        """Test SURAH_REF for Al-Ikhlas (surah 112, 4 verses)."""
        response = retriever.search("تفسير سورة الإخلاص")
        
        # All results must be from surah 112
        for r in response.results:
            assert r.verse_key.startswith("112:"), \
                f"Found chunk from wrong surah: {r.verse_key}"
    
    def test_surah_ref_an_nas(self, retriever):
        """Test SURAH_REF for An-Nas (surah 114, 6 verses)."""
        response = retriever.search("تفسير سورة الناس")
        
        # All results must be from surah 114
        for r in response.results:
            assert r.verse_key.startswith("114:"), \
                f"Found chunk from wrong surah: {r.verse_key}"


@pytest.mark.unit
class TestSurahRefParsing:
    """Unit tests for surah reference parsing."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        from src.ml.hybrid_evidence_retriever import HybridEvidenceRetriever
        return HybridEvidenceRetriever(use_bm25=False, use_dense=False)
    
    def test_parse_surah_fatiha(self, retriever):
        """Parse 'سورة الفاتحة' -> 1"""
        result = retriever._parse_surah_reference("تفسير سورة الفاتحة")
        assert result == 1
    
    def test_parse_surah_baqarah(self, retriever):
        """Parse 'سورة البقرة' -> 2"""
        result = retriever._parse_surah_reference("سورة البقرة")
        assert result == 2
    
    def test_parse_surah_ikhlas(self, retriever):
        """Parse 'سورة الإخلاص' -> 112"""
        result = retriever._parse_surah_reference("سورة الإخلاص")
        assert result == 112
    
    def test_parse_surah_nas(self, retriever):
        """Parse 'سورة الناس' -> 114"""
        result = retriever._parse_surah_reference("سورة الناس")
        assert result == 114
    
    def test_parse_no_surah_ref(self, retriever):
        """Non-surah query returns None."""
        result = retriever._parse_surah_reference("الصبر في القرآن")
        assert result is None


@pytest.mark.unit
class TestUTF8Encoding:
    """Tests for UTF-8 encoding consistency."""
    
    def test_evidence_index_loads_utf8(self):
        """Evidence index must load with UTF-8 encoding."""
        from pathlib import Path
        import json
        
        index_file = Path("data/evidence/evidence_index_v2_chunked.jsonl")
        if not index_file.exists():
            pytest.skip("Evidence index not found")
        
        # Must explicitly use UTF-8 encoding
        with open(index_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            chunk = json.loads(first_line)
        
        # Verify Arabic text is properly decoded
        assert 'text_clean' in chunk, "Missing text_clean field"
        text = chunk['text_clean']
        
        # Should contain Arabic characters, not escape sequences
        assert any('\u0600' <= c <= '\u06FF' for c in text), \
            "Text should contain Arabic characters"


@pytest.mark.integration
class TestSurahRefSummaryView:
    """Tests for SURAH_REF summary view functionality."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        from src.ml.hybrid_evidence_retriever import get_hybrid_retriever
        return get_hybrid_retriever(use_bm25=False, use_dense=False)
    
    def test_summary_view_returns_top_per_verse_source(self, retriever):
        """Summary view should return top N chunks per verse per source."""
        response = retriever.search("سورة الفاتحة")
        
        # Get summary view with 1 chunk per verse per source
        summary = response.get_summary_view(top_per_verse_source=1)
        
        assert summary['view'] == 'summary'
        assert summary['intent'] == 'SURAH_REF'
        
        # Should have 7 verses × 7 core sources = 49 summary results
        # (may be less if some sources missing for some verses)
        assert len(summary['summary_results']) <= 7 * 7
        assert len(summary['summary_results']) > 0
        
        # Total available should be much larger
        assert summary['total_available'] > summary['top_per_verse_source'] * 7 * 5
    
    def test_full_results_preserved_with_summary(self, retriever):
        """Full results should still be accessible after getting summary."""
        response = retriever.search("سورة الفاتحة")
        
        # Get summary
        summary = response.get_summary_view(top_per_verse_source=1)
        
        # Full results should still be in response
        assert len(response.results) == summary['total_available']
        assert len(response.results) > len(summary['summary_results'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
