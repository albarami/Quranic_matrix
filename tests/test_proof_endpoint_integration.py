"""
Test: Runtime Proof Endpoint Integration (Phase 6.3)

Runtime tests that call the actual proof system and assert evidence structure.
Uses deterministic test fixture for reproducible, fast integration tests.

Test Tiers:
- @pytest.mark.integration: Uses fixture index (runs in CI)
- @pytest.mark.e2e_gpu: Uses full corpus + GPU (manual/optional)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.mark.integration
class TestProofSystemRuntime:
    """Runtime integration tests for proof system using fixture index."""
    
    @pytest.fixture(scope="class")
    def proof_system(self):
        """Initialize the proof system with fixture index for testing."""
        from src.ml.full_power_system import FullPowerQBMSystem
        from src.ml.mandatory_proof_system import MandatoryProofSystem
        
        # Use fixture index for deterministic, fast tests
        system = FullPowerQBMSystem(use_fixture=True)
        return MandatoryProofSystem(system)
    
    def test_proof_ayah_ref_uses_hybrid(self, proof_system):
        """AYAH_REF queries must use hybrid retrieval mode."""
        result = proof_system.answer_with_full_proof("قارن تفسير البقرة:7 عند الخمسة")
        
        debug = result.get('debug', {})
        
        # Must use hybrid retrieval mode
        assert debug.get('retrieval_mode') == 'hybrid', \
            f"Expected retrieval_mode='hybrid', got '{debug.get('retrieval_mode')}'"
    
    def test_proof_ayah_ref_core_coverage(self, proof_system):
        """AYAH_REF queries must have good core source coverage."""
        result = proof_system.answer_with_full_proof("تفسير الآية 2:255")
        
        debug = result.get('debug', {})
        sources_covered = debug.get('sources_covered', [])
        core_count = debug.get('core_sources_count', 0)
        
        # Should have at least 3/5 core sources
        assert core_count >= 3, \
            f"Expected at least 3 core sources, got {core_count}: {sources_covered}"
    
    def test_proof_response_structure_valid(self, proof_system):
        """Proof response must have valid structure with all required components."""
        result = proof_system.answer_with_full_proof("تفسير سورة الفاتحة")
        
        # Check response has required top-level keys
        assert 'components' in result, "Missing 'components' in response"
        assert 'debug' in result, "Missing 'debug' in response"
        
        components = result.get('components', {})
        
        # Check all 5 core tafsir sources are present in structure
        for source in CORE_SOURCES:
            assert source in components, f"Missing tafsir source: {source}"
    
    def test_proof_debug_has_index_source(self, proof_system):
        """Debug must show index_source as 'fixture' when using fixture."""
        result = proof_system.answer_with_full_proof("تفسير الآية 1:1")
        
        debug = result.get('debug', {})
        
        # When using fixture, index_source should be 'fixture'
        assert debug.get('index_source') == 'fixture', \
            f"Expected index_source='fixture', got '{debug.get('index_source')}'"
    
    def test_proof_no_synthetic_evidence(self, proof_system):
        """Proof must not contain synthetic/fabricated evidence."""
        result = proof_system.answer_with_full_proof("ما هو الصبر في القرآن")
        
        components = result.get('components', {})
        
        # Check graph evidence is not fabricated
        graph = components.get('graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        paths = graph.get('paths', [])
        
        # Fabricated patterns to check for
        fabricated_patterns = [
            'سلوك_أ', 'سلوك_ب', 'سلوك_ج',  # Placeholder behaviors
            'سلوك1', 'سلوك2',  # Numbered placeholders
        ]
        
        for node in nodes:
            node_str = str(node)
            for pattern in fabricated_patterns:
                assert pattern not in node_str, f"Found fabricated pattern in node: {pattern}"
        
        for edge in edges:
            edge_str = str(edge)
            for pattern in fabricated_patterns:
                assert pattern not in edge_str, f"Found fabricated pattern in edge: {pattern}"
    
    def test_proof_hybrid_retriever_is_primary(self, proof_system):
        """Hybrid retriever must be the primary tafsir retrieval path."""
        result = proof_system.answer_with_full_proof("الصبر")
        
        debug = result.get('debug', {})
        
        # Retrieval mode must be 'hybrid' (not stratified or rag_only)
        assert debug.get('retrieval_mode') == 'hybrid', \
            f"Expected retrieval_mode='hybrid', got '{debug.get('retrieval_mode')}'"
    
    def test_proof_debug_fields_present(self, proof_system):
        """Debug fields must be present in response."""
        result = proof_system.answer_with_full_proof("2:255")
        
        debug = result.get('debug', {})
        
        # Required debug fields
        assert 'retrieval_mode' in debug, "Missing retrieval_mode in debug"
        assert 'fallback_used' in debug, "Missing fallback_used in debug"


@pytest.mark.unit
class TestQueryRouterIntegration:
    """Runtime tests for QueryRouter integration (no index needed)."""
    
    @pytest.fixture(scope="class")
    def router(self):
        from src.ml.query_router import get_query_router
        return get_query_router()
    
    def test_router_returns_entity_type_for_concepts(self, router):
        """Router must return entity_type for concept queries."""
        result = router.route("الصبر")
        
        assert result.entity_type is not None, "entity_type is None for concept query"
        assert result.entity_type == "BEHAVIOR"
    
    def test_router_returns_canonical_id(self, router):
        """Router must return canonical_id for known concepts."""
        result = router.route("الصبر")
        
        assert result.canonical_id is not None, "canonical_id is None"
        assert result.canonical_id == "BEH_EMO_PATIENCE"
    
    def test_router_distinguishes_entity_types(self, router):
        """Router must correctly distinguish entity types."""
        test_cases = [
            ("الصبر", "BEHAVIOR"),
            ("المؤمن", "AGENT"),
            ("القلب", "ORGAN"),
        ]
        
        for query, expected_type in test_cases:
            result = router.route(query)
            assert result.entity_type == expected_type, \
                f"Query '{query}': expected {expected_type}, got {result.entity_type}"


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    """Runtime tests for HybridEvidenceRetriever using fixture."""
    
    @pytest.fixture(scope="class")
    def retriever(self):
        from src.ml.hybrid_evidence_retriever import get_hybrid_retriever
        return get_hybrid_retriever(use_bm25=True, use_dense=False)
    
    def test_retriever_returns_chunk_ids(self, retriever):
        """Retriever must return chunk_ids in results."""
        response = retriever.search("2:255", top_k=10)
        
        for result in response.results:
            assert result.chunk_id is not None, "chunk_id is None"
            assert len(result.chunk_id) > 0, "chunk_id is empty"
    
    def test_retriever_returns_verse_keys(self, retriever):
        """Retriever must return verse_keys in results."""
        response = retriever.search("2:255", top_k=10)
        
        for result in response.results:
            assert result.verse_key is not None, "verse_key is None"
    
    def test_retriever_source_diversity(self, retriever):
        """Retriever must return results from multiple sources."""
        response = retriever.search("الصبر", top_k=20)
        
        sources = set(r.source for r in response.results)
        
        # Should have at least 3 different sources
        assert len(sources) >= 3, f"Only {len(sources)} sources: {sources}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
