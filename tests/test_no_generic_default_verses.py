"""
Phase 0 Tests: No Generic Default Verses

These tests ensure the system is fail-closed and does NOT insert generic
opening verses (Surah 1 or early Baqarah) as fallback when retrieval fails.

Requirements:
- For a query known to fail retrieval, assert no verses are returned
- For a query that should retrieve, assert verses are not dominated by {1:1-7, 2:1-20}
  unless the query explicitly asks for those
"""

import pytest
from typing import Set

# Generic default verses that should NOT be used as fallback
GENERIC_DEFAULT_VERSES: Set[str] = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}


class TestNoGenericDefaultVerses:
    """Test suite for Phase 0: No generic default verses."""
    
    def test_generic_default_verses_set_defined(self):
        """Verify the generic default verses set is properly defined."""
        # Surah 1 (Al-Fatiha): 7 verses
        assert "1:1" in GENERIC_DEFAULT_VERSES
        assert "1:7" in GENERIC_DEFAULT_VERSES
        assert "1:8" not in GENERIC_DEFAULT_VERSES
        
        # Surah 2 (Al-Baqarah): first 20 verses
        assert "2:1" in GENERIC_DEFAULT_VERSES
        assert "2:20" in GENERIC_DEFAULT_VERSES
        assert "2:21" not in GENERIC_DEFAULT_VERSES
        
        # Total: 7 + 20 = 27 verses
        assert len(GENERIC_DEFAULT_VERSES) == 27
    
    def test_scoring_has_generic_default_check(self):
        """Verify scoring.py has the generic default verses check."""
        from src.benchmarks.scoring import GENERIC_DEFAULT_VERSES as SCORING_GENERIC
        from src.benchmarks.scoring import _generic_opening_default_fail
        
        # Verify the set matches
        assert SCORING_GENERIC == GENERIC_DEFAULT_VERSES
        
        # Verify the function exists and is callable
        assert callable(_generic_opening_default_fail)
    
    def test_fail_closed_no_verses_for_nonsense_query(self):
        """
        For a query known to fail retrieval, assert no verses are returned.
        
        This tests the fail-closed behavior: when retrieval fails, the system
        should NOT insert generic fallback verses.
        """
        # Import the scoring function
        from src.benchmarks.scoring import score_benchmark_item
        
        # Create a mock response with no evidence (simulating failed retrieval)
        mock_response = {
            "answer": "No evidence found",
            "proof": {
                "quran": [],  # Empty - no verses retrieved
                "tafsir": {},
            },
            "debug": {
                "intent": "FREE_TEXT",
                "fallback_used": False,
            }
        }
        
        result = score_benchmark_item(
            benchmark_item={"expected": {}},
            response=mock_response,
            http_status=200,
            request_payload={"question": "xyzzy gibberish nonsense query"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )
        
        # Should FAIL with "no_evidence" reason (fail-closed)
        assert result["verdict"] == "FAIL"
        assert "no_evidence" in result["reasons"]
    
    def test_generic_verses_detected_as_fail(self):
        """
        For a query that returns mostly generic verses, assert it fails.
        
        If >50% of returned verses are from {1:1-7, 2:1-20} and the query
        is not explicitly about those verses, it should FAIL.
        """
        from src.benchmarks.scoring import score_benchmark_item
        
        # Create a mock response with mostly generic verses
        mock_response = {
            "answer": "Some answer",
            "proof": {
                "quran": [
                    {"surah": 1, "ayah": 1, "text": "بسم الله"},
                    {"surah": 1, "ayah": 2, "text": "الحمد لله"},
                    {"surah": 1, "ayah": 3, "text": "الرحمن الرحيم"},
                    {"surah": 2, "ayah": 1, "text": "الم"},
                    {"surah": 2, "ayah": 2, "text": "ذلك الكتاب"},
                    # Only 1 non-generic verse
                    {"surah": 49, "ayah": 13, "text": "يا أيها الناس"},
                ],
                "tafsir": {
                    "ibn_kathir": [{"chunk_id": "c1", "verse_key": "1:1", "char_start": 0, "char_end": 10, "text": "test"}],
                },
            },
            "debug": {
                "intent": "CONCEPT_REF",  # Not SURAH_REF or AYAH_REF
                "fallback_used": False,
            }
        }
        
        result = score_benchmark_item(
            benchmark_item={"expected": {}},
            response=mock_response,
            http_status=200,
            request_payload={"question": "ما هو سلوك الكبر؟"},  # Query about pride behavior
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )
        
        # Should FAIL because >50% of verses are generic defaults
        # and the query is not about Fatiha or early Baqarah
        assert result["verdict"] == "FAIL"
        assert "generic_opening_verses_default" in result["reasons"]
    
    def test_fatiha_query_allows_fatiha_verses(self):
        """
        For a query explicitly about Surah Al-Fatiha, generic verses are allowed.
        """
        from src.benchmarks.scoring import score_benchmark_item
        
        # Create a mock response with Fatiha verses
        mock_response = {
            "answer": "تفسير سورة الفاتحة",
            "proof": {
                "quran": [
                    {"surah": 1, "ayah": 1, "text": "بسم الله"},
                    {"surah": 1, "ayah": 2, "text": "الحمد لله"},
                    {"surah": 1, "ayah": 3, "text": "الرحمن الرحيم"},
                ],
                "tafsir": {
                    "ibn_kathir": [{"chunk_id": "c1", "verse_key": "1:1", "char_start": 0, "char_end": 10, "text": "test"}],
                },
            },
            "debug": {
                "intent": "SURAH_REF",  # Explicit surah reference
                "fallback_used": False,
            }
        }
        
        result = score_benchmark_item(
            benchmark_item={"expected": {}},
            response=mock_response,
            http_status=200,
            request_payload={"question": "سورة الفاتحة"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )
        
        # Should NOT fail for generic_opening_verses_default
        # because the query is explicitly about Fatiha (SURAH_REF intent)
        assert "generic_opening_verses_default" not in result.get("reasons", [])
    
    def test_ayah_ref_allows_specific_verses(self):
        """
        For a query explicitly about a specific verse, that verse is allowed.
        """
        from src.benchmarks.scoring import score_benchmark_item
        
        # Create a mock response with specific verse
        mock_response = {
            "answer": "تفسير آية الكرسي",
            "proof": {
                "quran": [
                    {"surah": 2, "ayah": 255, "text": "الله لا إله إلا هو"},
                ],
                "tafsir": {
                    "ibn_kathir": [{"chunk_id": "c1", "verse_key": "2:255", "char_start": 0, "char_end": 10, "text": "test"}],
                },
            },
            "debug": {
                "intent": "AYAH_REF",
                "fallback_used": False,
            }
        }
        
        result = score_benchmark_item(
            benchmark_item={"expected": {}},
            response=mock_response,
            http_status=200,
            request_payload={"question": "2:255"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )
        
        # Should NOT fail for generic_opening_verses_default
        assert "generic_opening_verses_default" not in result.get("reasons", [])
    
    def test_fallback_used_for_structured_intent_fails(self):
        """
        If fallback is used for a structured intent (not FREE_TEXT), it should FAIL.
        """
        from src.benchmarks.scoring import score_benchmark_item
        
        mock_response = {
            "answer": "Some answer",
            "proof": {
                "quran": [{"surah": 49, "ayah": 13, "text": "يا أيها الناس"}],
                "tafsir": {
                    "ibn_kathir": [{"chunk_id": "c1", "verse_key": "49:13", "char_start": 0, "char_end": 10, "text": "test"}],
                },
            },
            "debug": {
                "intent": "CONCEPT_REF",  # Structured intent
                "fallback_used": True,  # Fallback was used
            }
        }
        
        result = score_benchmark_item(
            benchmark_item={"expected": {}},
            response=mock_response,
            http_status=200,
            request_payload={"question": "الكبر"},
            schema_valid=True,
            schema_issues=[],
            request_error=None,
        )
        
        # Should FAIL because fallback was used for structured intent
        assert result["verdict"] == "FAIL"
        assert "fallback_used_for_structured_intent" in result["reasons"]


class TestHybridRetrieverFailClosed:
    """Test that HybridEvidenceRetriever is fail-closed."""
    
    def test_retriever_returns_empty_on_no_match(self):
        """
        Verify the hybrid retriever returns empty results (not fabricated)
        when no matches are found.
        """
        try:
            from src.ml.hybrid_evidence_retriever import HybridEvidenceRetriever
            from pathlib import Path
            
            # Check if the index file exists
            index_file = Path("data/evidence/evidence_index_v2_chunked.jsonl")
            if not index_file.exists():
                pytest.skip("Evidence index not available")
            
            retriever = HybridEvidenceRetriever(use_bm25=True, use_dense=False)
            
            # Query with nonsense that should not match anything
            response = retriever.search("xyzzy gibberish nonsense query 12345")
            
            # Should return empty or very low results, NOT fabricated evidence
            # The key is that fallback_used should be False (no synthetic data)
            assert response.fallback_used == False
            
            # If results are returned, they should be from actual retrieval
            # not from a generic fallback
            for result in response.results:
                verse_key = result.verse_key
                assert verse_key not in GENERIC_DEFAULT_VERSES or response.deterministic_count > 0
                
        except ImportError:
            pytest.skip("HybridEvidenceRetriever not available")


class TestMandatoryProofSystemFailClosed:
    """Test that MandatoryProofSystem is fail-closed."""
    
    def test_proof_debug_has_fail_closed_fields(self):
        """Verify ProofDebug has the fail-closed tracking fields."""
        from src.ml.mandatory_proof_system import ProofDebug
        
        debug = ProofDebug()
        
        # Should have fallback tracking
        assert hasattr(debug, 'fallback_used')
        assert hasattr(debug, 'fallback_reasons')
        assert hasattr(debug, 'quran_fallback')
        assert hasattr(debug, 'graph_fallback')
        assert hasattr(debug, 'taxonomy_fallback')
        
        # Default should be no fallback
        assert debug.fallback_used == False
        assert debug.fallback_reasons == []
        assert debug.quran_fallback == False


class TestIntentClassifierRouting:
    """Test that intent classifier routes correctly."""
    
    def test_analytical_queries_not_free_text(self):
        """
        Analytical queries should NOT be routed to FREE_TEXT.
        They should be routed to their specific intent types.
        """
        from src.ml.intent_classifier import classify_intent, IntentType
        
        test_cases = [
            # (query, expected_intent_NOT_FREE_TEXT)
            ("trace all causal chains from الغفلة to الكفر", IntentType.GRAPH_CAUSAL),
            ("compare tafsir methodologies", IntentType.CROSS_TAFSIR_ANALYSIS),
            ("11-dimensional profile of الكبر", IntentType.PROFILE_11D),
            ("network centrality analysis", IntentType.GRAPH_METRICS),
            ("heart state transitions", IntentType.HEART_STATE),
            ("agent type analysis for المؤمن", IntentType.AGENT_ANALYSIS),
            ("temporal spatial context دنيا آخرة", IntentType.TEMPORAL_SPATIAL),
            ("consequence analysis العقوبة", IntentType.CONSEQUENCE_ANALYSIS),
        ]
        
        for query, expected_intent in test_cases:
            result = classify_intent(query)
            assert result.intent != IntentType.FREE_TEXT, \
                f"Query '{query[:30]}...' should not be FREE_TEXT, got {result.intent}"
            assert result.intent == expected_intent, \
                f"Query '{query[:30]}...' expected {expected_intent}, got {result.intent}"
    
    def test_surah_ref_detection(self):
        """Test SURAH_REF intent detection."""
        from src.ml.intent_classifier import classify_intent, IntentType
        
        queries = [
            "سورة البقرة",
            "سورة الفاتحة",
            "سورة الكهف",
        ]
        
        for query in queries:
            result = classify_intent(query)
            assert result.intent == IntentType.SURAH_REF, \
                f"Query '{query}' should be SURAH_REF, got {result.intent}"
    
    def test_ayah_ref_detection(self):
        """Test AYAH_REF intent detection."""
        from src.ml.intent_classifier import classify_intent, IntentType
        
        queries = [
            "2:255",
            "البقرة:255",
        ]
        
        for query in queries:
            result = classify_intent(query)
            assert result.intent == IntentType.AYAH_REF, \
                f"Query '{query}' should be AYAH_REF, got {result.intent}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
