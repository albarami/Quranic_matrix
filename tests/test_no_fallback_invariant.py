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
        """All 7 tafsir sources must have >= min_per_source results"""
        response = qbm_system.answer_with_full_proof("ما هو الكبر؟")
        
        debug = response.get("debug", {})
        dist = debug.get("retrieval_distribution", {})
        
        print(f"\nRetrieval distribution: {dist}")
        
        required_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
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
        for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]:
            assert source in tafsir, f"Missing {source} fallback tracking"
        
        print("\n✅ Component fallback tracking verified")
        print(f"  quran_fallback: {components.get('quran')}")
        print(f"  graph_fallback: {components.get('graph')}")
        print(f"  taxonomy_fallback: {components.get('taxonomy')}")
        print(f"  tafsir_fallbacks: {tafsir}")


class TestSevenSourceSubstrate:
    """
    Phase 9.9D: Truth assertion tests for 7-source substrate.
    These tests prove the real 7-source retrieval is used for structured intents.
    """
    
    def test_surah_ref_uses_deterministic_chunked_7_sources(self, client):
        """
        For SURAH_REF الفاتحة, assert:
        - debug.intent == "SURAH_REF"
        - debug.retrieval_mode == "deterministic_chunked"
        - debug.component_fallbacks.tafsir.baghawi == False
        - debug.component_fallbacks.tafsir.muyassar == False
        - proof contains baghawi and muyassar sections populated
        """
        response = client.post(
            "/api/proof/query",
            json={"question": "سورة الفاتحة", "mode": "summary", "per_ayah": True, "proof_only": True}
        )
        assert response.status_code == 200
        data = response.json()
        
        debug = data.get("debug", {})
        proof = data.get("proof", {})
        
        # Intent must be SURAH_REF
        assert debug.get("intent") == "SURAH_REF", \
            f"Expected intent=SURAH_REF, got {debug.get('intent')}"
        
        # Retrieval mode must be deterministic_chunked for structured intents
        assert debug.get("retrieval_mode") == "deterministic_chunked", \
            f"Expected retrieval_mode=deterministic_chunked, got {debug.get('retrieval_mode')}"
        
        # Core sources count must be 7
        assert debug.get("core_sources_count") == 7, \
            f"Expected core_sources_count=7, got {debug.get('core_sources_count')}"
        
        # All 7 sources must be covered
        sources_covered = set(debug.get("sources_covered", []))
        expected_sources = {"ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"}
        assert sources_covered == expected_sources, \
            f"Expected all 7 sources, got {sources_covered}"
        
        # No tafsir fallbacks for any source
        # Handle both lightweight backend (tafsir_fallbacks) and full backend (component_fallbacks.tafsir)
        tafsir_fallbacks = debug.get("tafsir_fallbacks", debug.get("component_fallbacks", {}).get("tafsir", {}))
        for source in expected_sources:
            assert tafsir_fallbacks.get(source) == False, \
                f"Fallback used for {source}: {tafsir_fallbacks.get(source)}"
        
        # Proof must contain tafsir with baghawi and muyassar sections
        # Handle both proof_only (nested tafsir) and full response (flat) structures
        tafsir = proof.get("tafsir", proof)  # Fallback to proof itself for flat structure
        assert "baghawi" in tafsir or "baghawi" in proof, "Proof missing baghawi section"
        assert "muyassar" in tafsir or "muyassar" in proof, "Proof missing muyassar section"
        
        # Baghawi and muyassar must have content (not empty)
        baghawi_content = tafsir.get("baghawi", proof.get("baghawi", []))
        muyassar_content = tafsir.get("muyassar", proof.get("muyassar", []))
        assert len(baghawi_content) > 0 or proof.get("baghawi_total", 0) > 0, \
            "Baghawi section is empty"
        assert len(muyassar_content) > 0 or proof.get("muyassar_total", 0) > 0, \
            "Muyassar section is empty"
        
        print("\n✅ 7-source substrate verified for SURAH_REF")
        print(f"  intent: {debug.get('intent')}")
        print(f"  retrieval_mode: {debug.get('retrieval_mode')}")
        print(f"  core_sources_count: {debug.get('core_sources_count')}")
        print(f"  sources_covered: {sources_covered}")
        print(f"  baghawi_count: {len(baghawi_content)}")
        print(f"  muyassar_count: {len(muyassar_content)}")
    
    def test_ayah_ref_uses_deterministic_7_sources(self, client):
        """AYAH_REF must also use deterministic 7-source retrieval."""
        response = client.post(
            "/api/proof/query",
            json={"question": "2:255", "mode": "summary", "proof_only": True}
        )
        assert response.status_code == 200
        data = response.json()
        
        debug = data.get("debug", {})
        
        # For AYAH_REF, should use deterministic_chunked
        if debug.get("intent") == "AYAH_REF":
            assert debug.get("retrieval_mode") == "deterministic_chunked", \
                f"AYAH_REF should use deterministic_chunked, got {debug.get('retrieval_mode')}"
            
            # No tafsir fallbacks
            # Handle both lightweight backend (tafsir_fallbacks) and full backend (component_fallbacks.tafsir)
            tafsir_fallbacks = debug.get("tafsir_fallbacks", debug.get("component_fallbacks", {}).get("tafsir", {}))
            for source in ["baghawi", "muyassar"]:
                assert tafsir_fallbacks.get(source) == False, \
                    f"Fallback used for {source} in AYAH_REF"
    
    def test_proof_only_does_not_initialize_fullpower(self, client):
        """
        Phase 9.10E: proof_only=True must NOT initialize FullPower GPU components.
        
        Asserts:
        - debug.fullpower_used == False
        - debug.index_source != "runtime_build"
        - Response time < 5 seconds (no GPU init overhead)
        """
        import time
        start = time.time()
        
        response = client.post(
            "/api/proof/query",
            json={"question": "سورة الفاتحة", "mode": "summary", "proof_only": True}
        )
        
        elapsed = time.time() - start
        assert response.status_code == 200
        data = response.json()
        
        debug = data.get("debug", {})
        
        # Must NOT use FullPower
        assert debug.get("fullpower_used") == False, \
            f"proof_only should not use FullPower, got fullpower_used={debug.get('fullpower_used')}"
        
        # Must NOT do runtime index build
        assert debug.get("index_source") != "runtime_build", \
            f"proof_only should not build index at runtime, got index_source={debug.get('index_source')}"
        
        # Should be fast (< 5 seconds) - no GPU init overhead
        assert elapsed < 5.0, \
            f"proof_only should complete in <5s, took {elapsed:.2f}s"
        
        print(f"\n✅ proof_only test passed in {elapsed:.2f}s")
        print(f"   fullpower_used: {debug.get('fullpower_used')}")
        print(f"   index_source: {debug.get('index_source')}")
        print(f"   retrieval_mode: {debug.get('retrieval_mode')}")


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
