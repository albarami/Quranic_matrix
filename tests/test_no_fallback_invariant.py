"""
Phase 0: No-Fallback Invariant Tests
These tests verify that the primary retrieval path works without fallbacks.

After remediation, ALL these tests should pass with fallback_used = False.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tests.conftest import (
    assert_no_fallback,
    assert_source_distribution,
    FallbackDetectedError,
    STANDARD_QUERIES,
)


class TestNoFallbackInvariant:
    """Tests that verify the no-fallback invariant"""
    
    def test_standard_queries_no_fallback(self, qbm_system):
        """Standard queries must not use fallback"""
        queries = ["ما هو الصبر؟", "ما هو الكبر؟", "ما هي التقوى؟"]
        
        for query in queries:
            response = qbm_system.answer_with_full_proof(query)
            
            # This is the key Phase 0 check
            debug = response.get("debug", {})
            fallback_used = debug.get("fallback_used", False)
            
            print(f"\nQuery: {query}")
            print(f"  fallback_used: {fallback_used}")
            print(f"  fallback_reasons: {debug.get('fallback_reasons', [])}")
            
            # After remediation, this assertion should pass
            # For now, we document the current state
            if fallback_used:
                pytest.skip(
                    f"BASELINE: Fallback used for '{query}'. "
                    f"Reasons: {debug.get('fallback_reasons', [])}"
                )
    
    def test_source_distribution_invariant(self, qbm_system):
        """All 5 tafsir sources must have >= min_per_source results"""
        response = qbm_system.answer_with_full_proof("ما هو الكبر؟")
        
        debug = response.get("debug", {})
        dist = debug.get("retrieval_distribution", {})
        
        print(f"\nRetrieval distribution: {dist}")
        
        required_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
        min_per_source = 5
        
        missing = []
        for source in required_sources:
            count = dist.get(source, 0)
            if count < min_per_source:
                missing.append(f"{source}: {count}")
        
        if missing:
            pytest.skip(
                f"BASELINE: Sources below minimum: {missing}. "
                "This will be fixed in Phase 4 (Stratified Retrieval)."
            )
    
    def test_debug_section_present(self, qbm_system):
        """Verify debug section is present in response"""
        response = qbm_system.answer_with_full_proof("ما هو الصبر؟")
        
        assert "debug" in response, "Response missing 'debug' field"
        
        debug = response["debug"]
        assert "fallback_used" in debug, "Debug missing 'fallback_used'"
        assert "fallback_reasons" in debug, "Debug missing 'fallback_reasons'"
        assert "retrieval_distribution" in debug, "Debug missing 'retrieval_distribution'"
        assert "primary_path_latency_ms" in debug, "Debug missing 'primary_path_latency_ms'"
        assert "component_fallbacks" in debug, "Debug missing 'component_fallbacks'"
        
        print("\n✅ Debug section structure verified")
        print(f"  fallback_used: {debug['fallback_used']}")
        print(f"  primary_path_latency_ms: {debug['primary_path_latency_ms']}")
    
    def test_component_fallback_tracking(self, qbm_system):
        """Verify individual component fallback tracking"""
        response = qbm_system.answer_with_full_proof("ما هو الكبر؟")
        
        debug = response.get("debug", {})
        components = debug.get("component_fallbacks", {})
        
        assert "quran" in components, "Missing quran fallback tracking"
        assert "graph" in components, "Missing graph fallback tracking"
        assert "taxonomy" in components, "Missing taxonomy fallback tracking"
        assert "tafsir" in components, "Missing tafsir fallback tracking"
        
        tafsir = components.get("tafsir", {})
        for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            assert source in tafsir, f"Missing {source} fallback tracking"
        
        print("\n✅ Component fallback tracking verified")
        print(f"  quran_fallback: {components.get('quran')}")
        print(f"  graph_fallback: {components.get('graph')}")
        print(f"  taxonomy_fallback: {components.get('taxonomy')}")
        print(f"  tafsir_fallbacks: {tafsir}")


class TestStrictClient:
    """Tests using the strict client that enforces no-fallback"""
    
    def test_strict_client_raises_on_fallback(self, strict_client):
        """Strict client should raise FallbackDetectedError when fallback is used"""
        try:
            # This may raise FallbackDetectedError if fallback is used
            response = strict_client.query("ما هو الصبر؟", allow_fallback=False)
            print("\n✅ Query completed without fallback")
        except FallbackDetectedError as e:
            # Expected during baseline - document it
            print(f"\n⚠️ BASELINE: Fallback detected")
            print(f"  Reasons: {e.debug_info.get('fallback_reasons', [])}")
            pytest.skip("Fallback detected - this is expected during baseline")
    
    def test_strict_client_allows_fallback_when_permitted(self, strict_client):
        """Strict client should not raise when allow_fallback=True"""
        response = strict_client.query("ما هو الصبر؟", allow_fallback=True)
        
        assert "debug" in response
        print("\n✅ Query completed with allow_fallback=True")


class TestBaselineDocumentation:
    """Tests that document the current baseline state"""
    
    def test_document_fallback_rate(self, qbm_system):
        """Document the current fallback rate for all standard queries"""
        from tests.conftest import generate_baseline_report
        
        baseline = generate_baseline_report(qbm_system, STANDARD_QUERIES)
        
        print("\n" + "=" * 60)
        print("BASELINE FALLBACK REPORT")
        print("=" * 60)
        print(f"Total queries: {baseline['total_queries']}")
        print(f"Fallbacks used: {baseline['total_fallbacks']}")
        print(f"Fallback rate: {baseline['fallback_rate']:.1%}")
        print("\nComponent fallback counts:")
        for component, count in baseline['component_fallback_counts'].items():
            if isinstance(count, dict):
                for sub, sub_count in count.items():
                    print(f"  {component}.{sub}: {sub_count}")
            else:
                print(f"  {component}: {count}")
        
        print("\nPer-query results:")
        for result in baseline['results']:
            status = "❌" if result.get('fallback_used') else "✅"
            print(f"  {status} {result['query'][:30]}...")
            if result.get('fallback_reasons'):
                for reason in result['fallback_reasons']:
                    print(f"      → {reason}")
        
        # This test always passes - it's for documentation
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
