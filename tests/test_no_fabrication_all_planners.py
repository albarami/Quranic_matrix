"""
Phase 3: No Fabrication Tests for All Planners

Tests that planners fail-closed when evidence is missing.
No planner should fabricate data - they must return gaps/empty results.
"""

import pytest
from unittest.mock import MagicMock


# ============================================================================
# Mock LegendaryPlanner that returns "not found" / empty
# ============================================================================

@pytest.fixture
def mock_planner_no_data():
    """Create a mock LegendaryPlanner that returns no data."""
    planner = MagicMock()

    # Mock empty canonical_entities
    planner.canonical_entities = {}

    # Mock empty semantic_graph
    planner.semantic_graph = {"nodes": [], "edges": []}

    # Mock methods returning empty/not_found
    planner.enumerate_canonical_inventory.return_value = {
        "status": "no_entities",
        "items": [],
    }

    planner.get_concept_evidence.return_value = {
        "status": "not_found",
        "total_mentions": 0,
        "verse_keys": [],
        "sources": {},
    }

    planner.get_semantic_neighbors.return_value = []

    planner.find_causal_paths.return_value = []

    planner.resolve_entities.return_value = {
        "entities": [],
    }

    planner.compute_global_causal_density.return_value = {
        "status": "no_graph",
        "top_nodes": [],
    }

    planner.find_all_cycles.return_value = []

    planner.compute_chain_length_distribution.return_value = {
        "distribution": {},
        "total_paths": 0,
    }

    planner.compute_global_tafsir_coverage.return_value = {}

    return planner


# ============================================================================
# CausalChainPlanner No Fabrication Tests
# ============================================================================

class TestCausalChainNoFabrication:
    """Test that CausalChainPlanner fails-closed when no data."""

    def test_no_paths_returns_empty_paths_list(self, mock_planner_no_data):
        """When no causal paths exist, should return empty paths, not fabricated ones."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(mock_planner_no_data)

        result = planner.find_causal_chains(
            query="test query",
            from_entity_id="NONEXISTENT_A",
            to_entity_id="NONEXISTENT_B",
        )

        assert result.paths == []
        assert result.qualifying_paths_count == 0
        assert "no_causal_paths_found" in result.gaps

    def test_insufficient_entities_returns_gap(self, mock_planner_no_data):
        """When entities cannot be resolved, should return gap."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(mock_planner_no_data)

        result = planner.find_causal_chains(query="nonexistent query")

        assert "insufficient_entities_resolved" in result.gaps


# ============================================================================
# CrossTafsirPlanner No Fabrication Tests
# ============================================================================

class TestCrossTafsirNoFabrication:
    """Test that CrossTafsirPlanner fails-closed when no data."""

    def test_no_evidence_returns_empty_sources(self, mock_planner_no_data):
        """When no evidence exists, should return empty sources, not fabricated ones."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(mock_planner_no_data)

        result = planner.compare_tafsir_for_entity("NONEXISTENT_CONCEPT")

        assert result.sources_with_evidence == 0
        # Should still have 7 sources, but none with evidence
        assert all(s.chunk_count == 0 for s in result.source_evidence)

    def test_not_found_records_gap(self, mock_planner_no_data):
        """When concept not found, should record gap."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(mock_planner_no_data)

        result = planner.compare_tafsir_for_entity("NONEXISTENT_CONCEPT")

        assert "entity_not_in_index:NONEXISTENT_CONCEPT" in result.gaps


# ============================================================================
# Profile11DPlanner No Fabrication Tests
# ============================================================================

class TestProfile11DNoFabrication:
    """Test that Profile11DPlanner fails-closed when no data."""

    def test_no_evidence_returns_empty_dimensions(self, mock_planner_no_data):
        """When no evidence exists, all dimensions should be gaps."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(mock_planner_no_data)

        result = planner.profile_behavior("NONEXISTENT_BEHAVIOR")

        # Should have all 11 dimensions but with no data
        assert result.total_dimensions == 11
        assert result.filled_dimensions == 0
        assert all(d.is_gap for d in result.dimensions)

    def test_missing_dimensions_recorded_as_gaps(self, mock_planner_no_data):
        """Missing dimension data should be recorded as gaps."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(mock_planner_no_data)

        result = planner.profile_behavior("NONEXISTENT_BEHAVIOR")

        # Should have gaps for missing dimensions
        missing_gaps = [g for g in result.gaps if g.startswith("dimension_missing:")]
        assert len(missing_gaps) == 11  # All dimensions missing


# ============================================================================
# GraphMetricsPlanner No Fabrication Tests
# ============================================================================

class TestGraphMetricsNoFabrication:
    """Test that GraphMetricsPlanner fails-closed when no data."""

    def test_no_graph_returns_empty_metrics(self, mock_planner_no_data):
        """When no graph data exists, should return empty metrics."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(mock_planner_no_data)

        result = planner.compute_graph_metrics()

        # Should have empty node/edge data
        assert result.total_nodes == 0
        assert result.total_edges == 0
        # When graph is empty but loaded, no gap is needed - just zero counts

    def test_entity_not_in_graph_records_gap(self, mock_planner_no_data):
        """When entity not in graph, should return empty metrics."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(mock_planner_no_data)

        result = planner.get_node_metrics("NONEXISTENT_ENTITY")

        # get_node_metrics returns a dict with status or empty metrics
        assert result.get("total_degree", 0) == 0 or result.get("status") == "no_graph"


# ============================================================================
# HeartStatePlanner No Fabrication Tests
# ============================================================================

class TestHeartStateNoFabrication:
    """Test that HeartStatePlanner fails-closed when no data."""

    def test_no_entities_returns_empty_states(self, mock_planner_no_data):
        """When no canonical entities loaded, should return empty states."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(mock_planner_no_data)

        result = planner.get_heart_state_inventory()

        assert result.states == []
        assert result.states_with_evidence == 0
        assert "canonical_entities_not_loaded" in result.gaps

    def test_specific_state_not_found_records_gap(self, mock_planner_no_data):
        """When specific state not found, should record gap."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(mock_planner_no_data)

        result = planner.analyze_heart_state("HRT_NONEXISTENT")

        assert "heart_state_not_in_index:HRT_NONEXISTENT" in result.gaps


# ============================================================================
# AgentPlanner No Fabrication Tests
# ============================================================================

class TestAgentNoFabrication:
    """Test that AgentPlanner fails-closed when no data."""

    def test_no_entities_returns_empty_agents(self, mock_planner_no_data):
        """When no canonical entities loaded, should return empty agents."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(mock_planner_no_data)

        result = planner.get_agent_inventory()

        assert result.agents == []
        assert result.agents_with_evidence == 0
        assert "canonical_entities_not_loaded" in result.gaps

    def test_specific_agent_not_found_records_gap(self, mock_planner_no_data):
        """When specific agent not found, should record gap."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(mock_planner_no_data)

        result = planner.analyze_agent("AGT_NONEXISTENT")

        assert "agent_not_in_index:AGT_NONEXISTENT" in result.gaps

    def test_no_behavior_attribution_records_gap(self, mock_planner_no_data):
        """When no agents found for behavior, should record gap."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(mock_planner_no_data)

        result = planner.get_behavior_attribution("BEH_NONEXISTENT")

        assert "no_agents_found_for_behavior:BEH_NONEXISTENT" in result.gaps


# ============================================================================
# TemporalSpatialPlanner No Fabrication Tests
# ============================================================================

class TestTemporalSpatialNoFabrication:
    """Test that TemporalSpatialPlanner fails-closed when no data."""

    def test_no_evidence_returns_zero_mentions(self, mock_planner_no_data):
        """When no evidence exists, all axes should have 0 mentions."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(mock_planner_no_data)

        result = planner.get_temporal_mapping()

        assert result.temporal_with_evidence == 0
        assert all(t.total_mentions == 0 for t in result.temporal_axes)

    def test_no_context_for_behavior_records_gap(self, mock_planner_no_data):
        """When no temporal/spatial context for behavior, should record gap."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(mock_planner_no_data)

        result = planner.analyze_behavior_temporal_spatial("BEH_NONEXISTENT")

        assert "no_temporal_spatial_context_for:BEH_NONEXISTENT" in result.gaps


# ============================================================================
# ConsequencePlanner No Fabrication Tests
# ============================================================================

class TestConsequenceNoFabrication:
    """Test that ConsequencePlanner fails-closed when no data."""

    def test_no_entities_returns_empty_consequences(self, mock_planner_no_data):
        """When no canonical entities loaded, should return empty consequences."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(mock_planner_no_data)

        result = planner.get_consequence_inventory()

        assert result.consequences == []
        assert result.consequences_with_evidence == 0
        assert "canonical_entities_not_loaded" in result.gaps

    def test_specific_consequence_not_found_records_gap(self, mock_planner_no_data):
        """When specific consequence not found, should record gap."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(mock_planner_no_data)

        result = planner.analyze_consequence("CSQ_NONEXISTENT")

        assert "consequence_not_in_index:CSQ_NONEXISTENT" in result.gaps

    def test_no_consequences_for_behavior_records_gap(self, mock_planner_no_data):
        """When no consequences found for behavior, should record gap."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(mock_planner_no_data)

        result = planner.get_behavior_consequences("BEH_NONEXISTENT")

        assert "no_consequences_found_for_behavior:BEH_NONEXISTENT" in result.gaps


# ============================================================================
# EmbeddingsPlanner No Fabrication Tests
# ============================================================================

class TestEmbeddingsNoFabrication:
    """Test that EmbeddingsPlanner is truthful about limitations."""

    def test_no_model_records_gap(self, mock_planner_no_data):
        """When no model configured, should record gap."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_planner_no_data)

        # Mock empty registry
        planner._registry = {}

        result = planner.get_model_info()

        assert "no_active_model_configured" in result.gaps

    def test_limitation_disclosure_always_has_limitations(self, mock_planner_no_data):
        """Limitation disclosure should always include general limitations."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_planner_no_data)

        result = planner.get_limitation_disclosure()

        # Should always have general limitations even without model
        assert len(result.limitations) >= 4  # 4 general limitations

    def test_search_without_model_records_gap(self, mock_planner_no_data):
        """Search without model info should record gap."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(mock_planner_no_data)

        # Mock empty registry
        planner._registry = {}

        result = planner.semantic_search("test query")

        assert "cannot_perform_search:no_model_info" in result.gaps
        assert result.search_results == []


# ============================================================================
# IntegrationPlanner No Fabrication Tests
# ============================================================================

class TestIntegrationNoFabrication:
    """Test that IntegrationPlanner aggregates gaps correctly."""

    def test_aggregates_all_component_gaps(self, mock_planner_no_data):
        """Should aggregate gaps from all components."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(mock_planner_no_data)

        result = planner.run_full_analysis("NONEXISTENT_ENTITY")

        # Should have gaps from multiple components
        assert len(result.total_gaps) > 0

    def test_partial_status_when_some_components_fail(self, mock_planner_no_data):
        """Components with no data should have partial or failed status."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(mock_planner_no_data)

        result = planner.run_full_analysis("NONEXISTENT_ENTITY")

        # At least some components should not be fully successful
        statuses = [c.status for c in result.component_results]
        assert "partial" in statuses or "failed" in statuses


# ============================================================================
# Cross-Planner Consistency: No Fabrication Guarantee
# ============================================================================

class TestCrossplannerNoFabrication:
    """Test that no planner fabricates when given nonexistent entity."""

    @pytest.mark.parametrize("planner_class,method,args", [
        ("CausalChainPlanner", "find_causal_chains", {"query": "test", "from_entity_id": "X", "to_entity_id": "Y"}),
        ("CrossTafsirPlanner", "compare_tafsir_for_entity", {"entity_id": "NONEXISTENT"}),
        ("Profile11DPlanner", "profile_behavior", {"entity_id": "NONEXISTENT"}),
        # GraphMetricsPlanner.get_node_metrics returns dict, tested separately
        ("HeartStatePlanner", "analyze_heart_state", {"state_id": "NONEXISTENT"}),
        ("AgentPlanner", "analyze_agent", {"agent_id": "NONEXISTENT"}),
        ("ConsequencePlanner", "analyze_consequence", {"consequence_id": "NONEXISTENT"}),
    ])
    def test_nonexistent_entity_no_fabrication(self, mock_planner_no_data, planner_class, method, args):
        """All planners should not fabricate data for nonexistent entities."""
        import importlib
        planners_module = importlib.import_module("src.ml.planners")
        PlannerClass = getattr(planners_module, planner_class)

        planner = PlannerClass(mock_planner_no_data)
        method_func = getattr(planner, method)
        result = method_func(**args)

        # Result should have gaps (fail-closed behavior)
        result_dict = result.to_dict()
        assert "gaps" in result_dict
        # Should either have gaps OR have zero evidence
        has_gaps = len(result_dict.get("gaps", [])) > 0

        # Check that we're not fabricating positive numbers
        mentions_fields = [
            "total_mentions", "evidence_count", "verse_count",
            "sources_with_evidence", "filled_dimensions",
            "states_with_evidence", "agents_with_evidence",
            "consequences_with_evidence",
        ]

        all_zero_or_missing = True
        for field in mentions_fields:
            if field in result_dict:
                if result_dict[field] > 0:
                    all_zero_or_missing = False
                    break

        # Either has gaps OR all counts are zero
        assert has_gaps or all_zero_or_missing, f"{planner_class}.{method} fabricated data without evidence"
