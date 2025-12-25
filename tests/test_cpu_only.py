"""
Phase 1: CPU-Only Tests
These tests run without GPU and are suitable for CI environments.
Uses mock fixtures instead of full GPU-powered system.
"""

import pytest
from tests.conftest import (
    assert_no_fallback,
    assert_source_distribution,
    FallbackDetectedError,
)


class TestProofDebugSchema:
    """Test ProofDebug schema without GPU"""
    
    def test_mock_response_has_debug(self, mock_proof_response):
        """Verify mock response has debug section"""
        assert "debug" in mock_proof_response
        debug = mock_proof_response["debug"]
        assert "fallback_used" in debug
        assert "fallback_reasons" in debug
        assert "retrieval_distribution" in debug
    
    def test_assert_no_fallback_passes(self, mock_proof_response):
        """assert_no_fallback should pass for clean response"""
        # Should not raise
        assert_no_fallback(mock_proof_response)
    
    def test_assert_no_fallback_fails_on_fallback(self, mock_fallback_response):
        """assert_no_fallback should raise for fallback response"""
        with pytest.raises(FallbackDetectedError) as exc_info:
            assert_no_fallback(mock_fallback_response)
        
        assert "FALLBACK DETECTED" in str(exc_info.value)
        assert exc_info.value.debug_info["fallback_used"] is True


class TestMiniFixtures:
    """Test the mini-fixtures work correctly"""
    
    def test_qbm_system_mini_exists(self, qbm_system_mini):
        """Mini system fixture should be available"""
        assert qbm_system_mini is not None
        assert qbm_system_mini.index_source == "disk"
    
    def test_qbm_system_mini_search(self, qbm_system_mini):
        """Mini system should return mock search results"""
        results = qbm_system_mini.search("test query")
        assert len(results) > 0
        assert results[0]["metadata"]["type"] == "quran"
    
    def test_mock_proof_response_structure(self, mock_proof_response):
        """Mock proof response should have correct structure"""
        assert mock_proof_response["validation"]["score"] == 100.0
        assert mock_proof_response["debug"]["fallback_used"] is False
        
        dist = mock_proof_response["debug"]["retrieval_distribution"]
        assert "ibn_kathir" in dist
        assert "quran" in dist


class TestSourceDistribution:
    """Test source distribution assertions"""
    
    def test_source_distribution_passes(self, mock_proof_response):
        """Source distribution should pass for balanced response"""
        # Should not raise - all sources have 10+ results
        assert_source_distribution(mock_proof_response, min_per_source=5)
    
    def test_source_distribution_fails_on_missing(self, mock_fallback_response):
        """Source distribution should fail when sources are missing"""
        with pytest.raises(AssertionError) as exc_info:
            assert_source_distribution(mock_fallback_response, min_per_source=5)
        
        # Should mention missing sources
        assert "below minimum" in str(exc_info.value)


class TestFallbackDetection:
    """Test fallback detection logic"""
    
    def test_component_fallbacks_tracked(self, mock_fallback_response):
        """Component-level fallbacks should be tracked"""
        components = mock_fallback_response["debug"]["component_fallbacks"]
        
        assert components["quran"] is True
        assert components["graph"] is False
        assert components["tafsir"]["tabari"] is True
        assert components["tafsir"]["saadi"] is True
    
    def test_fallback_reasons_recorded(self, mock_fallback_response):
        """Fallback reasons should be recorded"""
        reasons = mock_fallback_response["debug"]["fallback_reasons"]
        
        assert len(reasons) > 0
        assert "quran" in reasons[0]


class TestIndexSource:
    """Test index_source tracking"""
    
    def test_disk_index_source(self, mock_proof_response):
        """Normal response should have disk index source"""
        assert mock_proof_response["debug"]["index_source"] == "disk"
    
    def test_runtime_build_index_source(self, mock_fallback_response):
        """Fallback response can have runtime_build index source"""
        assert mock_fallback_response["debug"]["index_source"] == "runtime_build"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
