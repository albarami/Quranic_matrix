"""
Test: Query Planner with Debug Traceability (Phase 7.1)

Ensures query planner correctly orchestrates Truth Layer components.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.query_planner import QueryPlanner, QueryType, QueryPlan, DebugTrace


@pytest.fixture(scope="module")
def planner():
    """Get a loaded query planner."""
    p = QueryPlanner(base_path=Path("."))
    p.load()
    return p


@pytest.mark.unit
class TestQueryPlannerLoading:
    """Tests for query planner initialization."""
    
    def test_planner_loads_canonical_entities(self, planner):
        """Planner must load canonical entities."""
        assert planner.canonical_entities is not None
        assert len(planner._term_to_entity) > 0
    
    def test_planner_loads_concept_index(self, planner):
        """Planner must load concept index."""
        assert planner.concept_index is not None
        assert len(planner.concept_index) > 0
    
    def test_planner_loads_graphs(self, planner):
        """Planner must load both graphs."""
        assert planner.cooccurrence_graph is not None
        assert planner.semantic_graph is not None
    
    def test_planner_loads_tafsir_sources(self, planner):
        """Planner must load tafsir sources."""
        assert planner.tafsir_sources is not None


@pytest.mark.unit
class TestEntityResolution:
    """Tests for entity resolution."""
    
    def test_resolve_envy_term(self, planner):
        """Should resolve الحسد to BEH_EMO_ENVY."""
        entity_id = planner.resolve_entity("الحسد")
        assert entity_id == "BEH_EMO_ENVY"
    
    def test_resolve_patience_term(self, planner):
        """Should resolve الصبر to BEH_EMO_PATIENCE."""
        entity_id = planner.resolve_entity("الصبر")
        assert entity_id == "BEH_EMO_PATIENCE"
    
    def test_resolve_unknown_term(self, planner):
        """Should return None for unknown terms."""
        entity_id = planner.resolve_entity("كلمة_غير_موجودة")
        assert entity_id is None


@pytest.mark.unit
class TestQueryTypeDetection:
    """Tests for query type detection."""
    
    def test_detect_behavior_profile(self, planner):
        """Should detect behavior profile query."""
        query_type = planner.detect_query_type("ملف الحسد الكامل")
        assert query_type == QueryType.BEHAVIOR_PROFILE
    
    def test_detect_causal_chain(self, planner):
        """Should detect causal chain query."""
        query_type = planner.detect_query_type("الغفلة يؤدي إلى الكفر")
        assert query_type == QueryType.CAUSAL_CHAIN
    
    def test_detect_verse_lookup(self, planner):
        """Should detect verse lookup query."""
        query_type = planner.detect_query_type("آية 2:255")
        assert query_type == QueryType.VERSE_LOOKUP
    
    def test_detect_concept_analysis(self, planner):
        """Should default to concept analysis."""
        query_type = planner.detect_query_type("ما هو الحسد؟")
        assert query_type == QueryType.CONCEPT_ANALYSIS


@pytest.mark.unit
class TestPlanCreation:
    """Tests for query plan creation."""
    
    def test_create_plan_has_steps(self, planner):
        """Plan must have execution steps."""
        plan = planner.create_plan("تحليل الحسد")
        assert len(plan.steps) > 0
    
    def test_plan_starts_with_entity_resolution(self, planner):
        """First step must be entity resolution."""
        plan = planner.create_plan("تحليل الحسد")
        assert plan.steps[0].action == "resolve_entities"
    
    def test_plan_ends_with_aggregation(self, planner):
        """Last step must be result aggregation."""
        plan = planner.create_plan("تحليل الحسد")
        assert plan.steps[-1].action == "aggregate_results"
    
    def test_behavior_profile_has_axes_step(self, planner):
        """Behavior profile query must include 11-axes computation."""
        plan = planner.create_plan("ملف الحسد الكامل")
        actions = [s.action for s in plan.steps]
        assert "compute_11_axes" in actions


@pytest.mark.integration
class TestPlanExecution:
    """Tests for query plan execution."""
    
    def test_execute_returns_debug_trace(self, planner):
        """Execution must return debug trace."""
        plan = planner.create_plan("تحليل الحسد")
        trace = planner.execute_plan(plan)
        
        assert isinstance(trace, DebugTrace)
        assert trace.query_id == plan.query_id
    
    def test_trace_has_entity_resolution(self, planner):
        """Trace must include entity resolution results."""
        plan = planner.create_plan("تحليل الحسد")
        trace = planner.execute_plan(plan)
        
        assert "resolved_terms" in trace.entity_resolution
    
    def test_trace_has_concept_lookups(self, planner):
        """Trace must include concept lookups."""
        plan = planner.create_plan("تحليل الحسد")
        trace = planner.execute_plan(plan)
        
        assert len(trace.concept_lookups) > 0
    
    def test_all_steps_complete(self, planner):
        """All steps must complete (or fail gracefully)."""
        plan = planner.create_plan("تحليل الحسد")
        trace = planner.execute_plan(plan)
        
        for step in plan.steps:
            assert step.status in ["completed", "failed", "skipped"]


@pytest.mark.integration
class TestQueryEndToEnd:
    """End-to-end query tests."""
    
    def test_query_envy_returns_results(self, planner):
        """Query for الحسد must return results."""
        result = planner.query("ما هو الحسد في القرآن؟")
        
        assert result["query_type"] == "concept_analysis"
        assert result["target_entity"] == "BEH_EMO_ENVY"
        assert "results" in result
        assert "debug_trace" in result
    
    def test_query_results_have_evidence(self, planner):
        """Query results must include concept evidence."""
        result = planner.query("تحليل الحسد")
        
        evidence = result["results"].get("concept_evidence", {})
        assert evidence.get("total_mentions", 0) > 0
    
    def test_debug_trace_serializable(self, planner):
        """Debug trace must be JSON serializable."""
        import json
        
        result = planner.query("تحليل الحسد")
        trace_dict = result["debug_trace"]
        
        # Should not raise
        json_str = json.dumps(trace_dict, ensure_ascii=False)
        assert len(json_str) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
