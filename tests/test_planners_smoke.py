"""
Phase 3: Smoke Tests for All Planners

Tests that each planner can be instantiated and produces valid output structure.
One sample per planner - basic functionality verification.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Mock LegendaryPlanner for isolated testing
# ============================================================================

@pytest.fixture
def mock_legendary_planner():
    """Create a mock LegendaryPlanner with minimal required methods."""
    planner = MagicMock()

    # Mock canonical_entities
    planner.canonical_entities = {
        "behaviors": [
            {"id": "BEH_SPEECH_TRUTHFULNESS", "ar": "الصدق", "en": "Truthfulness"},
        ],
        "agents": [
            {"id": "AGT_BELIEVER", "ar": "مؤمن", "en": "Believer"},
        ],
        "consequences": [
            {"id": "CSQ_JANNAH", "ar": "الجنة", "en": "Paradise"},
        ],
        "heart_states": [
            {"id": "HRT_SEALED", "ar": "القلب المختوم", "en": "Sealed Heart"},
        ],
    }

    # Mock semantic_graph
    planner.semantic_graph = {
        "nodes": [
            {"id": "BEH_SPEECH_TRUTHFULNESS", "type": "BEHAVIOR"},
            {"id": "AGT_BELIEVER", "type": "AGENT"},
        ],
        "edges": [
            {
                "source": "AGT_BELIEVER",
                "target": "BEH_SPEECH_TRUTHFULNESS",
                "edge_type": "PERFORMS",
                "evidence_count": 5,
                "confidence": 0.8,
            },
        ],
    }

    # Mock methods
    planner.enumerate_canonical_inventory.return_value = {
        "status": "success",
        "items": [
            {"id": "BEH_SPEECH_TRUTHFULNESS", "ar": "الصدق", "en": "Truthfulness", "total_mentions": 10, "verse_count": 5},
        ],
    }

    planner.get_concept_evidence.return_value = {
        "status": "found",
        "total_mentions": 10,
        "verse_keys": ["2:42", "3:71"],
        "sources": {"tabari": 3, "kathir": 2},
    }

    planner.get_semantic_neighbors.return_value = [
        {
            "entity_id": "CSQ_JANNAH",
            "edge_type": "LEADS_TO",
            "evidence_count": 3,
            "confidence": 0.7,
            "direction": "outgoing",
        },
    ]

    planner.find_causal_paths.return_value = [
        [
            {"source": "A", "target": "B", "edge_type": "CAUSES", "evidence_count": 2},
            {"source": "B", "target": "C", "edge_type": "CAUSES", "evidence_count": 1},
        ],
    ]

    planner.resolve_entities.return_value = {
        "entities": [
            {"entity_id": "BEH_SPEECH_TRUTHFULNESS"},
            {"entity_id": "CSQ_JANNAH"},
        ],
    }

    planner.compute_global_causal_density.return_value = {
        "status": "computed",
        "top_nodes": [
            {"id": "BEH_SPEECH_TRUTHFULNESS", "out_degree": 5, "in_degree": 3},
        ],
    }

    planner.find_all_cycles.return_value = []

    planner.compute_chain_length_distribution.return_value = {
        "distribution": {2: 5, 3: 3, 4: 1},
        "total_paths": 9,
    }

    planner.compute_global_tafsir_coverage.return_value = {
        "tabari": {"concept_count": 50},
        "kathir": {"concept_count": 45},
    }

    return planner


# ============================================================================
# CausalChainPlanner Smoke Tests
# ============================================================================

class TestCausalChainPlannerSmoke:
    """Smoke tests for CausalChainPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that CausalChainPlanner can be instantiated."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(mock_legendary_planner)
        assert planner is not None

    def test_find_causal_chains_returns_valid_structure(self, mock_legendary_planner):
        """Test that find_causal_chains returns valid result structure."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(mock_legendary_planner)

        result = planner.find_causal_chains(
            query="من الصدق إلى الجنة",
            from_entity_id="BEH_SPEECH_TRUTHFULNESS",
            to_entity_id="CSQ_JANNAH",
        )

        assert hasattr(result, 'from_entity')
        assert hasattr(result, 'to_entity')
        assert hasattr(result, 'paths')
        assert hasattr(result, 'gaps')
        assert hasattr(result, 'to_dict')

    def test_to_dict_produces_serializable_output(self, mock_legendary_planner):
        """Test that to_dict produces JSON-serializable output."""
        import json
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(mock_legendary_planner)

        result = planner.find_causal_chains(
            query="test",
            from_entity_id="A",
            to_entity_id="B",
        )

        # Should not raise
        json.dumps(result.to_dict())


# ============================================================================
# CrossTafsirPlanner Smoke Tests
# ============================================================================

class TestCrossTafsirPlannerSmoke:
    """Smoke tests for CrossTafsirPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that CrossTafsirPlanner can be instantiated."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(mock_legendary_planner)
        assert planner is not None

    def test_compare_tafsir_for_entity_returns_valid_structure(self, mock_legendary_planner):
        """Test that compare_tafsir_for_entity returns valid result structure."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(mock_legendary_planner)

        result = planner.compare_tafsir_for_entity("BEH_SPEECH_TRUTHFULNESS")

        assert hasattr(result, 'entity_id')
        assert hasattr(result, 'source_evidence')
        assert hasattr(result, 'total_sources')
        assert hasattr(result, 'gaps')


# ============================================================================
# Profile11DPlanner Smoke Tests
# ============================================================================

class TestProfile11DPlannerSmoke:
    """Smoke tests for Profile11DPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that Profile11DPlanner can be instantiated."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(mock_legendary_planner)
        assert planner is not None

    def test_profile_behavior_returns_valid_structure(self, mock_legendary_planner):
        """Test that profile_behavior returns valid result structure."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(mock_legendary_planner)

        result = planner.profile_behavior("BEH_SPEECH_TRUTHFULNESS")

        assert hasattr(result, 'entity_id')
        assert hasattr(result, 'dimensions')
        assert hasattr(result, 'total_dimensions')
        assert hasattr(result, 'gaps')


# ============================================================================
# GraphMetricsPlanner Smoke Tests
# ============================================================================

class TestGraphMetricsPlannerSmoke:
    """Smoke tests for GraphMetricsPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that GraphMetricsPlanner can be instantiated."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(mock_legendary_planner)
        assert planner is not None

    def test_compute_graph_metrics_returns_valid_structure(self, mock_legendary_planner):
        """Test that compute_graph_metrics returns valid result structure."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(mock_legendary_planner)

        result = planner.compute_graph_metrics()

        assert hasattr(result, 'total_nodes')
        assert hasattr(result, 'total_edges')
        assert hasattr(result, 'top_by_degree')
        assert hasattr(result, 'gaps')


# ============================================================================
# HeartStatePlanner Smoke Tests
# ============================================================================

class TestHeartStatePlannerSmoke:
    """Smoke tests for HeartStatePlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that HeartStatePlanner can be instantiated."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(mock_legendary_planner)
        assert planner is not None

    def test_get_inventory_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_heart_state_inventory returns valid result structure."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(mock_legendary_planner)

        result = planner.get_heart_state_inventory()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'states')
        assert hasattr(result, 'transitions')
        assert hasattr(result, 'gaps')


# ============================================================================
# AgentPlanner Smoke Tests
# ============================================================================

class TestAgentPlannerSmoke:
    """Smoke tests for AgentPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that AgentPlanner can be instantiated."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(mock_legendary_planner)
        assert planner is not None

    def test_get_inventory_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_agent_inventory returns valid result structure."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(mock_legendary_planner)

        result = planner.get_agent_inventory()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'agents')
        assert hasattr(result, 'behavior_mappings')
        assert hasattr(result, 'gaps')


# ============================================================================
# TemporalSpatialPlanner Smoke Tests
# ============================================================================

class TestTemporalSpatialPlannerSmoke:
    """Smoke tests for TemporalSpatialPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that TemporalSpatialPlanner can be instantiated."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(mock_legendary_planner)
        assert planner is not None

    def test_get_temporal_mapping_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_temporal_mapping returns valid result structure."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(mock_legendary_planner)

        result = planner.get_temporal_mapping()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'temporal_axes')
        assert hasattr(result, 'spatial_axes')
        assert hasattr(result, 'gaps')

    def test_get_spatial_mapping_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_spatial_mapping returns valid result structure."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(mock_legendary_planner)

        result = planner.get_spatial_mapping()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'temporal_axes')
        assert hasattr(result, 'spatial_axes')


# ============================================================================
# ConsequencePlanner Smoke Tests
# ============================================================================

class TestConsequencePlannerSmoke:
    """Smoke tests for ConsequencePlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that ConsequencePlanner can be instantiated."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(mock_legendary_planner)
        assert planner is not None

    def test_get_inventory_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_consequence_inventory returns valid result structure."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(mock_legendary_planner)

        result = planner.get_consequence_inventory()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'consequences')
        assert hasattr(result, 'behavior_mappings')
        assert hasattr(result, 'positive_count')
        assert hasattr(result, 'negative_count')
        assert hasattr(result, 'gaps')


# ============================================================================
# EmbeddingsPlanner Smoke Tests
# ============================================================================

class TestEmbeddingsPlannerSmoke:
    """Smoke tests for EmbeddingsPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that EmbeddingsPlanner can be instantiated."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_legendary_planner)
        assert planner is not None

    def test_get_model_info_returns_valid_structure(self, mock_legendary_planner):
        """Test that get_model_info returns valid result structure."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_legendary_planner)

        result = planner.get_model_info()

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'model_info')
        assert hasattr(result, 'limitations')
        assert hasattr(result, 'gaps')

    def test_get_limitation_disclosure_returns_limitations(self, mock_legendary_planner):
        """Test that get_limitation_disclosure returns limitations."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_legendary_planner)

        result = planner.get_limitation_disclosure()

        assert result.query_type == "limitation_disclosure"
        # Should always have general limitations
        assert len(result.limitations) > 0


# ============================================================================
# IntegrationPlanner Smoke Tests
# ============================================================================

class TestIntegrationPlannerSmoke:
    """Smoke tests for IntegrationPlanner."""

    def test_instantiation(self, mock_legendary_planner):
        """Test that IntegrationPlanner can be instantiated."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(mock_legendary_planner)
        assert planner is not None

    def test_run_full_analysis_returns_valid_structure(self, mock_legendary_planner):
        """Test that run_full_analysis returns valid result structure."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(mock_legendary_planner)

        result = planner.run_full_analysis("BEH_SPEECH_TRUTHFULNESS")

        assert hasattr(result, 'query_type')
        assert hasattr(result, 'entity_id')
        assert hasattr(result, 'component_results')
        assert hasattr(result, 'consistency_checks')
        assert hasattr(result, 'total_gaps')
        assert hasattr(result, 'total_conflicts')


# ============================================================================
# All Planners Import Test
# ============================================================================

class TestAllPlannersImport:
    """Test that all planners can be imported."""

    def test_all_planners_importable(self):
        """Test that all planners can be imported from the module."""
        from src.ml.planners import (
            CausalChainPlanner,
            CrossTafsirPlanner,
            Profile11DPlanner,
            GraphMetricsPlanner,
            HeartStatePlanner,
            AgentPlanner,
            TemporalSpatialPlanner,
            ConsequencePlanner,
            EmbeddingsPlanner,
            IntegrationPlanner,
        )

        assert CausalChainPlanner is not None
        assert CrossTafsirPlanner is not None
        assert Profile11DPlanner is not None
        assert GraphMetricsPlanner is not None
        assert HeartStatePlanner is not None
        assert AgentPlanner is not None
        assert TemporalSpatialPlanner is not None
        assert ConsequencePlanner is not None
        assert EmbeddingsPlanner is not None
        assert IntegrationPlanner is not None
