#!/usr/bin/env python3
"""
Phase 4 Tests: Capability Engines

Tests for:
1. All 16 capability engines (A-J mapped)
2. Disallow patterns (no generic opening verses)
3. Provenance requirements
4. Deterministic execution

Run with: pytest tests/test_capabilities_phase4.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capabilities import get_engine, list_engines, CAPABILITY_MAP
from src.capabilities.base import CapabilityEngine, CapabilityResult
from src.capabilities.engines import GENERIC_OPENING_VERSES


# ============================================================================
# Engine Registry Tests
# ============================================================================

@pytest.mark.unit
class TestEngineRegistry:
    """Tests for capability engine registry."""
    
    def test_list_engines_returns_engines(self):
        """list_engines must return registered engines."""
        engines = list_engines()
        assert len(engines) >= 10, f"Expected >=10 engines, got {len(engines)}"
    
    def test_all_capabilities_have_engines(self):
        """All capabilities in CAPABILITY_MAP must have engines."""
        for cap_id in CAPABILITY_MAP.keys():
            engine = get_engine(cap_id)
            assert engine is not None, f"No engine for capability: {cap_id}"
    
    def test_get_engine_returns_correct_type(self):
        """get_engine must return CapabilityEngine instances."""
        engine = get_engine("GRAPH_CAUSAL")
        assert engine is not None
        assert isinstance(engine, CapabilityEngine)
    
    def test_capability_map_has_required_fields(self):
        """CAPABILITY_MAP entries must have required fields."""
        for cap_id, info in CAPABILITY_MAP.items():
            assert "name" in info, f"{cap_id} missing 'name'"
            assert "description" in info, f"{cap_id} missing 'description'"
            assert "sections" in info, f"{cap_id} missing 'sections'"


# ============================================================================
# Individual Engine Tests
# ============================================================================

@pytest.mark.unit
class TestGraphCausalEngine:
    """Tests for GRAPH_CAUSAL engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("GRAPH_CAUSAL")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_graph_data(self):
        """Engine must return graph statistics."""
        engine = get_engine("GRAPH_CAUSAL")
        result = engine.execute("test query")
        assert "total_causal_edges" in result.data
        assert "causal_types" in result.data
    
    def test_engine_has_provenance(self):
        """Engine must return provenance."""
        engine = get_engine("GRAPH_CAUSAL")
        result = engine.execute("test query")
        assert result.has_provenance()


@pytest.mark.unit
class TestMultihopEngine:
    """Tests for MULTIHOP engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("MULTIHOP")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success


@pytest.mark.unit
class TestTafsirMultiSourceEngine:
    """Tests for TAFSIR_MULTI_SOURCE engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("TAFSIR_MULTI_SOURCE")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_lists_required_sources(self):
        """Engine must list required tafsir sources."""
        engine = get_engine("TAFSIR_MULTI_SOURCE")
        result = engine.execute("test query")
        assert "required_sources" in result.data
        assert len(result.data["required_sources"]) >= 5


@pytest.mark.unit
class TestProvenanceEngine:
    """Tests for PROVENANCE engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("PROVENANCE")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_manifest_info(self):
        """Engine must return KB manifest info."""
        engine = get_engine("PROVENANCE")
        result = engine.execute("test query")
        assert "kb_version" in result.data
        assert "git_commit" in result.data


@pytest.mark.unit
class TestTaxonomyEngine:
    """Tests for TAXONOMY engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("TAXONOMY")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_categories(self):
        """Engine must return behavior categories."""
        engine = get_engine("TAXONOMY")
        result = engine.execute("test query")
        assert "categories" in result.data
        assert "total_behaviors" in result.data
        assert result.data["total_behaviors"] >= 70


@pytest.mark.unit
class TestHeartStateEngine:
    """Tests for HEART_STATE engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("HEART_STATE")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_heart_states(self):
        """Engine must return heart states."""
        engine = get_engine("HEART_STATE")
        result = engine.execute("test query")
        assert "heart_states" in result.data
        assert result.data["count"] >= 10


@pytest.mark.unit
class TestAgentModelEngine:
    """Tests for AGENT_MODEL engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("AGENT_MODEL")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_agents(self):
        """Engine must return agents."""
        engine = get_engine("AGENT_MODEL")
        result = engine.execute("test query")
        assert "agents" in result.data
        assert result.data["count"] >= 10


@pytest.mark.unit
class TestConsequenceModelEngine:
    """Tests for CONSEQUENCE_MODEL engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("CONSEQUENCE_MODEL")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_consequences(self):
        """Engine must return consequences."""
        engine = get_engine("CONSEQUENCE_MODEL")
        result = engine.execute("test query")
        assert "consequences" in result.data
        assert result.data["count"] >= 10


@pytest.mark.unit
class TestGraphMetricsEngine:
    """Tests for GRAPH_METRICS engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("GRAPH_METRICS")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
        assert result.success
    
    def test_engine_returns_metrics(self):
        """Engine must return graph metrics."""
        engine = get_engine("GRAPH_METRICS")
        result = engine.execute("test query")
        assert "node_count" in result.data
        assert "edge_count" in result.data


@pytest.mark.unit
class TestIntegrationE2EEngine:
    """Tests for INTEGRATION_E2E engine."""
    
    def test_engine_executes(self):
        """Engine must execute without error."""
        engine = get_engine("INTEGRATION_E2E")
        result = engine.execute("test query")
        assert isinstance(result, CapabilityResult)
    
    def test_engine_checks_components(self):
        """Engine must check all components."""
        engine = get_engine("INTEGRATION_E2E")
        result = engine.execute("test query")
        assert "components" in result.data
        assert "all_available" in result.data


# ============================================================================
# Disallow Pattern Tests
# ============================================================================

@pytest.mark.unit
class TestDisallowPatterns:
    """Tests for disallow patterns (no generic opening verses)."""
    
    def test_generic_opening_verses_defined(self):
        """GENERIC_OPENING_VERSES must be defined."""
        assert len(GENERIC_OPENING_VERSES) > 0
        assert "1:1" in GENERIC_OPENING_VERSES
        assert "2:1" in GENERIC_OPENING_VERSES
    
    def test_fatiha_verses_in_disallow(self):
        """All Fatiha verses must be in disallow list."""
        for i in range(1, 8):
            assert f"1:{i}" in GENERIC_OPENING_VERSES
    
    def test_early_baqarah_in_disallow(self):
        """Early Baqarah verses must be in disallow list."""
        for i in range(1, 21):
            assert f"2:{i}" in GENERIC_OPENING_VERSES
    
    def test_graph_causal_filters_generic_verses(self):
        """GRAPH_CAUSAL engine must filter generic opening verses."""
        engine = get_engine("GRAPH_CAUSAL")
        result = engine.execute("test", params={
            "from_behavior": "BEH_EMO_PATIENCE",
            "to_behavior": "BEH_EMO_GRATITUDE",
        })
        
        verse_keys = result.get_verse_keys()
        for vk in verse_keys:
            assert vk not in GENERIC_OPENING_VERSES, \
                f"Generic verse {vk} should be filtered"


# ============================================================================
# Provenance Requirement Tests
# ============================================================================

@pytest.mark.unit
class TestProvenanceRequirements:
    """Tests for provenance requirements."""
    
    def test_all_engines_return_provenance(self):
        """All engines must return provenance."""
        for cap_id in list_engines():
            engine = get_engine(cap_id)
            result = engine.execute("test query")
            assert result.has_provenance(), \
                f"Engine {cap_id} missing provenance"
    
    def test_provenance_has_source(self):
        """Provenance entries must have source field."""
        engine = get_engine("PROVENANCE")
        result = engine.execute("test query")
        
        for prov in result.provenance:
            assert "source" in prov or "type" in prov, \
                f"Provenance missing source/type: {prov}"


# ============================================================================
# Deterministic Execution Tests
# ============================================================================

@pytest.mark.unit
class TestDeterministicExecution:
    """Tests for deterministic execution."""
    
    def test_same_query_same_result(self):
        """Same query must produce same result."""
        engine = get_engine("TAXONOMY")
        
        result1 = engine.execute("test query")
        result2 = engine.execute("test query")
        
        assert result1.data == result2.data
    
    def test_graph_metrics_deterministic(self):
        """Graph metrics must be deterministic."""
        engine = get_engine("GRAPH_METRICS")
        
        result1 = engine.execute("test")
        result2 = engine.execute("test")
        
        assert result1.data["node_count"] == result2.data["node_count"]
        assert result1.data["edge_count"] == result2.data["edge_count"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
