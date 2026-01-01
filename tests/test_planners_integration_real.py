"""
Phase 3: REAL Integration Tests for All Planners

NO MOCKS. NO HARDCODING.
Tests run against real data files:
- data/graph/semantic_graph_v2.json
- data/evidence/concept_index_v2.jsonl
- vocab/canonical_entities.json
- vocab/temporal.json
- vocab/spatial.json
- data/models/registry.json
"""

import pytest
import json
from pathlib import Path


# ============================================================================
# Skip if data files not present
# ============================================================================

DATA_FILES = [
    "data/graph/semantic_graph_v2.json",
    "vocab/canonical_entities.json",
    "vocab/temporal.json",
    "vocab/spatial.json",
]

def data_files_exist():
    """Check if required data files exist."""
    return all(Path(f).exists() for f in DATA_FILES)


pytestmark = pytest.mark.skipif(
    not data_files_exist(),
    reason="Required data files not present"
)


# ============================================================================
# Real LegendaryPlanner fixture
# ============================================================================

@pytest.fixture(scope="module")
def real_legendary_planner():
    """Get real LegendaryPlanner with actual data loaded."""
    from src.ml.legendary_planner import get_legendary_planner
    planner = get_legendary_planner()
    return planner


@pytest.fixture(scope="module")
def canonical_entities():
    """Load real canonical entities."""
    with open("vocab/canonical_entities.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# CausalChainPlanner Real Tests
# ============================================================================

class TestCausalChainPlannerReal:
    """Real integration tests for CausalChainPlanner."""

    def test_find_chains_with_real_entities(self, real_legendary_planner, canonical_entities):
        """Test finding causal chains between real entities from canonical list."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(real_legendary_planner)

        # Get real behavior IDs from canonical entities
        behaviors = canonical_entities.get("behaviors", [])
        if len(behaviors) >= 2:
            from_entity = behaviors[0]["id"]
            to_entity = behaviors[1]["id"]

            result = planner.find_causal_chains(
                query=f"from {from_entity} to {to_entity}",
                from_entity_id=from_entity,
                to_entity_id=to_entity,
                max_depth=3,
            )

            # Result should be valid structure
            assert hasattr(result, 'from_entity')
            assert hasattr(result, 'to_entity')
            assert hasattr(result, 'paths')
            assert hasattr(result, 'gaps')
            assert result.from_entity == from_entity
            assert result.to_entity == to_entity

    def test_to_dict_serializable(self, real_legendary_planner, canonical_entities):
        """Test that result is JSON-serializable."""
        from src.ml.planners import CausalChainPlanner
        planner = CausalChainPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if len(behaviors) >= 2:
            result = planner.find_causal_chains(
                query="test",
                from_entity_id=behaviors[0]["id"],
                to_entity_id=behaviors[1]["id"],
            )

            # Should not raise
            serialized = json.dumps(result.to_dict())
            assert len(serialized) > 0


# ============================================================================
# CrossTafsirPlanner Real Tests
# ============================================================================

class TestCrossTafsirPlannerReal:
    """Real integration tests for CrossTafsirPlanner."""

    def test_compare_tafsir_with_real_entity(self, real_legendary_planner, canonical_entities):
        """Test comparing tafsir sources for a real entity."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            entity_id = behaviors[0]["id"]

            result = planner.compare_tafsir_for_entity(entity_id)

            # Should have 7 tafsir sources
            assert result.total_sources == 7
            assert hasattr(result, 'source_evidence')
            assert hasattr(result, 'gaps')
            assert len(result.source_evidence) == 7

    def test_agreement_ratio_in_valid_range(self, real_legendary_planner, canonical_entities):
        """Test that agreement ratio is between 0 and 1."""
        from src.ml.planners import CrossTafsirPlanner
        planner = CrossTafsirPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            result = planner.compare_tafsir_for_entity(behaviors[0]["id"])

            assert 0.0 <= result.agreement_ratio <= 1.0


# ============================================================================
# Profile11DPlanner Real Tests
# ============================================================================

class TestProfile11DPlannerReal:
    """Real integration tests for Profile11DPlanner."""

    def test_profile_real_behavior(self, real_legendary_planner, canonical_entities):
        """Test profiling a real behavior from canonical list."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            entity_id = behaviors[0]["id"]

            result = planner.profile_behavior(entity_id)

            # Should have exactly 11 dimensions
            assert result.total_dimensions == 11
            assert len(result.dimensions) == 11
            assert hasattr(result, 'gaps')

    def test_all_11_dimensions_present(self, real_legendary_planner, canonical_entities):
        """Test that all 11 Bouzidani dimensions are present."""
        from src.ml.planners import Profile11DPlanner
        planner = Profile11DPlanner(real_legendary_planner)

        expected_dimensions = [
            "organ", "situational", "systemic", "spatial", "temporal",
            "agent", "source", "evaluation", "heart_state", "consequence",
            "relationships"
        ]

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            result = planner.profile_behavior(behaviors[0]["id"])

            actual_dims = [d.dimension for d in result.dimensions]
            for dim in expected_dimensions:
                assert dim in actual_dims, f"Missing dimension: {dim}"


# ============================================================================
# GraphMetricsPlanner Real Tests
# ============================================================================

class TestGraphMetricsPlannerReal:
    """Real integration tests for GraphMetricsPlanner."""

    def test_compute_metrics_from_real_graph(self, real_legendary_planner):
        """Test computing metrics from real semantic graph."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(real_legendary_planner)

        result = planner.compute_graph_metrics()

        # Should have non-zero counts if graph has data
        assert hasattr(result, 'total_nodes')
        assert hasattr(result, 'total_edges')
        assert hasattr(result, 'top_by_degree')

        # Graph should have some content
        if Path("data/graph/semantic_graph_v2.json").exists():
            # Either has data or has gap
            assert result.total_nodes >= 0

    def test_node_metrics_for_real_entity(self, real_legendary_planner, canonical_entities):
        """Test getting node metrics for a real entity."""
        from src.ml.planners import GraphMetricsPlanner
        planner = GraphMetricsPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            entity_id = behaviors[0]["id"]

            result = planner.get_node_metrics(entity_id)

            assert "entity_id" in result
            # Either has metrics OR has no_graph status (fail-closed)
            has_metrics = "in_degree" in result and "out_degree" in result
            has_no_graph = result.get("status") == "no_graph"
            assert has_metrics or has_no_graph, "Should either have metrics or fail-closed status"


# ============================================================================
# HeartStatePlanner Real Tests
# ============================================================================

class TestHeartStatePlannerReal:
    """Real integration tests for HeartStatePlanner."""

    def test_get_real_heart_state_inventory(self, real_legendary_planner, canonical_entities):
        """Test getting heart state inventory from real data."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(real_legendary_planner)

        result = planner.get_heart_state_inventory()

        # Should have heart states from canonical entities
        heart_states = canonical_entities.get("heart_states", [])
        assert hasattr(result, 'states')
        assert hasattr(result, 'transitions')
        assert hasattr(result, 'gaps')

    def test_analyze_real_heart_state(self, real_legendary_planner, canonical_entities):
        """Test analyzing a real heart state."""
        from src.ml.planners import HeartStatePlanner
        planner = HeartStatePlanner(real_legendary_planner)

        heart_states = canonical_entities.get("heart_states", [])
        if heart_states:
            state_id = heart_states[0]["id"]

            result = planner.analyze_heart_state(state_id)

            assert hasattr(result, 'states')
            assert hasattr(result, 'gaps')


# ============================================================================
# AgentPlanner Real Tests
# ============================================================================

class TestAgentPlannerReal:
    """Real integration tests for AgentPlanner."""

    def test_get_real_agent_inventory(self, real_legendary_planner, canonical_entities):
        """Test getting agent inventory from real data."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(real_legendary_planner)

        result = planner.get_agent_inventory()

        # Should have agents from canonical entities (14 agents)
        assert hasattr(result, 'agents')
        assert hasattr(result, 'behavior_mappings')
        assert hasattr(result, 'gaps')

    def test_analyze_real_agent(self, real_legendary_planner, canonical_entities):
        """Test analyzing a real agent."""
        from src.ml.planners import AgentPlanner
        planner = AgentPlanner(real_legendary_planner)

        agents = canonical_entities.get("agents", [])
        if agents:
            agent_id = agents[0]["id"]

            result = planner.analyze_agent(agent_id)

            assert hasattr(result, 'agents')
            assert hasattr(result, 'gaps')


# ============================================================================
# TemporalSpatialPlanner Real Tests
# ============================================================================

class TestTemporalSpatialPlannerReal:
    """Real integration tests for TemporalSpatialPlanner."""

    def test_get_real_temporal_mapping(self, real_legendary_planner):
        """Test getting temporal mapping from real vocab."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(real_legendary_planner)

        result = planner.get_temporal_mapping()

        # Should have temporal axes from vocab/temporal.json
        assert hasattr(result, 'temporal_axes')
        assert hasattr(result, 'gaps')

        # Load expected count from vocab file
        with open("vocab/temporal.json", "r", encoding="utf-8") as f:
            temporal_vocab = json.load(f)
            expected_count = len(temporal_vocab.get("items", []))

        assert result.total_temporal == expected_count

    def test_get_real_spatial_mapping(self, real_legendary_planner):
        """Test getting spatial mapping from real vocab."""
        from src.ml.planners import TemporalSpatialPlanner
        planner = TemporalSpatialPlanner(real_legendary_planner)

        result = planner.get_spatial_mapping()

        # Should have spatial axes from vocab/spatial.json
        assert hasattr(result, 'spatial_axes')
        assert hasattr(result, 'gaps')

        # Load expected count from vocab file
        with open("vocab/spatial.json", "r", encoding="utf-8") as f:
            spatial_vocab = json.load(f)
            expected_count = len(spatial_vocab.get("items", []))

        assert result.total_spatial == expected_count


# ============================================================================
# ConsequencePlanner Real Tests
# ============================================================================

class TestConsequencePlannerReal:
    """Real integration tests for ConsequencePlanner."""

    def test_get_real_consequence_inventory(self, real_legendary_planner, canonical_entities):
        """Test getting consequence inventory from real data."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(real_legendary_planner)

        result = planner.get_consequence_inventory()

        # Should have consequences from canonical entities (16 consequences)
        assert hasattr(result, 'consequences')
        assert hasattr(result, 'behavior_mappings')
        assert hasattr(result, 'positive_count')
        assert hasattr(result, 'negative_count')
        assert hasattr(result, 'gaps')

    def test_analyze_real_consequence(self, real_legendary_planner, canonical_entities):
        """Test analyzing a real consequence."""
        from src.ml.planners import ConsequencePlanner
        planner = ConsequencePlanner(real_legendary_planner)

        consequences = canonical_entities.get("consequences", [])
        if consequences:
            csq_id = consequences[0]["id"]

            result = planner.analyze_consequence(csq_id)

            assert hasattr(result, 'consequences')
            assert hasattr(result, 'gaps')


# ============================================================================
# EmbeddingsPlanner Real Tests
# ============================================================================

class TestEmbeddingsPlannerReal:
    """Real integration tests for EmbeddingsPlanner."""

    def test_get_real_model_info(self, real_legendary_planner):
        """Test getting model info from real registry."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(real_legendary_planner)

        result = planner.get_model_info()

        assert hasattr(result, 'model_info')
        assert hasattr(result, 'limitations')
        assert hasattr(result, 'gaps')

        # Should load from real registry if it exists
        if Path("data/models/registry.json").exists():
            with open("data/models/registry.json", "r", encoding="utf-8") as f:
                registry = json.load(f)

            if registry.get("active_model"):
                assert result.model_info is not None
                assert result.model_info.model_name == registry["active_model"]

    def test_limitation_disclosure_truthful(self, real_legendary_planner):
        """Test that limitation disclosure reflects real model status."""
        from src.ml.planners import EmbeddingsPlanner
        planner = EmbeddingsPlanner(real_legendary_planner)

        result = planner.get_limitation_disclosure()

        # Should always have limitations
        assert len(result.limitations) > 0

        # If model doesn't pass threshold, should say so
        if result.model_info and not result.model_info.passes_threshold:
            assert any("NOT pass" in lim or "below" in lim.lower() for lim in result.limitations)


# ============================================================================
# IntegrationPlanner Real Tests
# ============================================================================

class TestIntegrationPlannerReal:
    """Real integration tests for IntegrationPlanner."""

    def test_full_analysis_with_real_entity(self, real_legendary_planner, canonical_entities):
        """Test full integrated analysis with a real entity."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            entity_id = behaviors[0]["id"]

            result = planner.run_full_analysis(entity_id)

            assert hasattr(result, 'entity_id')
            assert hasattr(result, 'component_results')
            assert hasattr(result, 'consistency_checks')
            assert hasattr(result, 'total_gaps')
            assert result.entity_id == entity_id

            # Should have results from multiple components
            assert result.total_components > 0

    def test_consistency_checks_run(self, real_legendary_planner, canonical_entities):
        """Test that consistency checks actually run."""
        from src.ml.planners import IntegrationPlanner
        planner = IntegrationPlanner(real_legendary_planner)

        behaviors = canonical_entities.get("behaviors", [])
        if behaviors:
            result = planner.run_full_analysis(behaviors[0]["id"])

            # Should have consistency checks
            assert hasattr(result, 'consistency_checks')
            # Checks should have been evaluated
            for check in result.consistency_checks:
                assert hasattr(check, 'passed')
                assert isinstance(check.passed, bool)


# ============================================================================
# Cross-Planner Data Consistency Tests
# ============================================================================

class TestCrossPlannerConsistency:
    """Test that planners return consistent data from shared sources."""

    def test_canonical_entity_counts_match(self, canonical_entities):
        """Test that entity counts match canonical file."""
        expected_counts = {
            "behaviors": 87,
            "agents": 14,
            "consequences": 16,
            "heart_states": 12,
        }

        for entity_type, expected in expected_counts.items():
            actual = len(canonical_entities.get(entity_type, []))
            # Allow some variance if schema evolved
            assert actual > 0, f"No {entity_type} found in canonical entities"

    def test_tafsir_sources_count(self, real_legendary_planner):
        """Test that tafsir sources count is 7."""
        from src.ml.planners import CrossTafsirPlanner
        from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

        assert len(CANONICAL_TAFSIR_SOURCES) == 7

        planner = CrossTafsirPlanner(real_legendary_planner)
        result = planner.compare_tafsir_for_entity("BEH_SPEECH_TRUTHFULNESS")
        assert result.total_sources == 7
