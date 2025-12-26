"""
Test: Legendary 25 Question Planner (Phase 9.1)

Tests for the 25-question-class planner with debug traceability.
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from src.ml.legendary_planner import (
    LegendaryPlanner,
    QuestionClass,
    get_legendary_planner,
)


@pytest.fixture(scope="module")
def planner():
    """Get the legendary planner."""
    p = get_legendary_planner()
    p.load()
    return p


@pytest.mark.unit
class TestQuestionClassDetection:
    """Tests for question class detection."""
    
    def test_detect_causal_chain(self, planner):
        """Should detect causal chain queries."""
        queries = [
            "سبب الغفلة يؤدي إلى الكفر",
            "What causes heedlessness to lead to disbelief?",
            "path to destruction",
        ]
        for q in queries:
            qclass = planner.detect_question_class(q)
            assert qclass == QuestionClass.CAUSAL_CHAIN, f"Failed for: {q}"
    
    def test_detect_behavior_profile(self, planner):
        """Should detect behavior profile queries."""
        queries = [
            "ملف كامل للحسد",
            "11-axis profile for envy",
            "complete profile of الحسد",
        ]
        for q in queries:
            qclass = planner.detect_question_class(q)
            assert qclass == QuestionClass.BEHAVIOR_PROFILE_11AXIS, f"Failed for: {q}"
    
    def test_detect_complete_analysis(self, planner):
        """Should detect complete analysis queries."""
        queries = [
            "تحليل كامل للتوبة",  # Use كامل instead of شامل
            "complete analysis of repentance",
            "comprehensive analysis",
        ]
        for q in queries:
            qclass = planner.detect_question_class(q)
            assert qclass == QuestionClass.COMPLETE_ANALYSIS, f"Failed for: {q}"
    
    def test_fallback_to_free_text(self, planner):
        """Unknown queries should fall back to FREE_TEXT."""
        qclass = planner.detect_question_class("random query without patterns")
        assert qclass == QuestionClass.FREE_TEXT


@pytest.mark.unit
class TestEntityResolution:
    """Tests for entity resolution."""
    
    def test_resolve_envy(self, planner):
        """Should resolve الحسد to BEH_EMO_ENVY."""
        resolution = planner.resolve_entities("ما هو الحسد؟")
        
        entity_ids = [e["entity_id"] for e in resolution["entities"]]
        assert "BEH_EMO_ENVY" in entity_ids
    
    def test_resolve_multiple_entities(self, planner):
        """Should resolve multiple entities in query."""
        resolution = planner.resolve_entities("العلاقة بين الحسد والكبر")
        
        assert len(resolution["entities"]) >= 2


@pytest.mark.unit
class TestPlanCreation:
    """Tests for plan creation."""
    
    def test_causal_chain_plan_has_path_finding(self, planner):
        """Causal chain plan should include path finding step."""
        plan = planner.create_plan(
            "طريق الغفلة إلى الكفر",
            QuestionClass.CAUSAL_CHAIN
        )
        
        actions = [s.action for s in plan]
        assert "find_causal_paths" in actions
        assert "resolve_entities" in actions
    
    def test_behavior_profile_plan_has_evidence_and_neighbors(self, planner):
        """Behavior profile plan should get evidence and neighbors."""
        plan = planner.create_plan(
            "ملف كامل للحسد",
            QuestionClass.BEHAVIOR_PROFILE_11AXIS
        )
        
        actions = [s.action for s in plan]
        assert "get_concept_evidence" in actions
        assert "get_semantic_neighbors" in actions
    
    def test_all_plans_have_provenance_bundling(self, planner):
        """All plans should end with provenance bundling."""
        for qclass in [QuestionClass.CAUSAL_CHAIN, QuestionClass.BEHAVIOR_PROFILE_11AXIS, QuestionClass.COMPLETE_ANALYSIS]:
            plan = planner.create_plan("test query", qclass)
            last_action = plan[-1].action
            assert last_action == "bundle_provenance"


@pytest.mark.unit
class TestDebugTrace:
    """Tests for debug trace."""
    
    def test_debug_trace_has_question_class(self, planner):
        """Debug trace should include question_class."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert debug.question_class is not None
        assert debug.question_class != ""
    
    def test_debug_trace_has_plan_steps(self, planner):
        """Debug trace should include plan_steps."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert len(debug.plan_steps) > 0
        for step in debug.plan_steps:
            assert step.action is not None
            assert step.component is not None
    
    def test_debug_trace_has_provenance_summary(self, planner):
        """Debug trace should include provenance_summary."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert debug.provenance_summary is not None
        assert debug.provenance_summary.no_fabrication == True
    
    def test_debug_trace_serializable(self, planner):
        """Debug trace should be JSON serializable."""
        results, debug = planner.query("ما هو الحسد؟")
        
        trace_dict = debug.to_dict()
        json_str = json.dumps(trace_dict, ensure_ascii=False)
        
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert "question_class" in parsed
        assert "plan_steps" in parsed


@pytest.mark.unit
class TestConceptEvidence:
    """Tests for concept evidence retrieval."""
    
    def test_get_concept_evidence_for_envy(self, planner):
        """Should get evidence for BEH_EMO_ENVY."""
        evidence = planner.get_concept_evidence("BEH_EMO_ENVY")
        
        assert evidence["status"] == "found"
        assert evidence["total_mentions"] > 0
        assert evidence["sources_count"] >= 3
    
    def test_get_concept_evidence_returns_no_evidence_honestly(self, planner):
        """Should return no_evidence for unknown concepts."""
        evidence = planner.get_concept_evidence("UNKNOWN_CONCEPT_XYZ")
        
        assert evidence["status"] in ["not_found", "no_evidence"]


@pytest.mark.unit
class TestSemanticNeighbors:
    """Tests for semantic graph neighbors."""
    
    def test_get_semantic_neighbors_for_envy(self, planner):
        """Should get semantic neighbors for BEH_EMO_ENVY."""
        neighbors = planner.get_semantic_neighbors("BEH_EMO_ENVY")
        
        assert len(neighbors) > 0
        for n in neighbors:
            assert "entity_id" in n
            assert "edge_type" in n
            assert "confidence" in n
    
    def test_semantic_neighbors_sorted_by_confidence(self, planner):
        """Neighbors should be sorted by confidence descending."""
        neighbors = planner.get_semantic_neighbors("BEH_EMO_ENVY")
        
        if len(neighbors) >= 2:
            for i in range(len(neighbors) - 1):
                assert neighbors[i]["confidence"] >= neighbors[i + 1]["confidence"]


@pytest.mark.unit
class TestEndToEndQuery:
    """Tests for end-to-end query execution."""
    
    def test_query_returns_results_and_debug(self, planner):
        """Query should return both results and debug trace."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert results is not None
        assert debug is not None
        assert isinstance(results, dict)
    
    def test_query_results_have_entities(self, planner):
        """Query results should include resolved entities."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert "entities" in results
        assert len(results["entities"]) > 0
    
    def test_query_results_have_evidence(self, planner):
        """Query results should include evidence."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert "evidence" in results
    
    def test_query_duration_tracked(self, planner):
        """Query should track total duration."""
        results, debug = planner.query("ما هو الحسد؟")
        
        assert debug.total_duration_ms > 0


@pytest.mark.unit
class TestAllQuestionClasses:
    """Tests that all 25 question classes are defined."""
    
    def test_all_25_classes_exist(self):
        """Should have all 25 question classes defined."""
        expected_classes = [
            "causal_chain", "shortest_path", "reinforcement_loop",
            "cross_tafsir_comparative", "makki_madani_analysis", "consensus_dispute",
            "behavior_profile_11axis", "organ_behavior_mapping", "state_transition",
            "agent_attribution", "agent_contrast_matrix", "prophetic_archetype",
            "network_centrality", "community_detection", "bridge_behaviors",
            "temporal_mapping", "spatial_mapping",
            "surah_fingerprints", "frequency_centrality", "makki_madani_shift",
            "semantic_landscape", "meaning_drift",
            "complete_analysis", "prescription_generator", "genome_artifact",
        ]
        
        actual_classes = [qc.value for qc in QuestionClass if qc != QuestionClass.FREE_TEXT]
        
        for expected in expected_classes:
            assert expected in actual_classes, f"Missing class: {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
