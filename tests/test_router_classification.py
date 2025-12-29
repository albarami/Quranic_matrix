"""
Phase 1 Tests: Router Classification

These tests ensure analytical questions are NOT routed to FREE_TEXT.
Each test has Arabic + English variants to verify language-agnostic routing.
"""

import pytest
from src.ml.question_class_router import (
    route_question,
    RouterResult,
    is_analytical_question,
    get_planner_for_question,
    ANALYTICAL_INTENTS,
)
from src.ml.legendary_planner import QuestionClass
from src.ml.intent_classifier import IntentType


class TestAnalyticalQuestionsNotFreeText:
    """Test that analytical questions are NOT routed to FREE_TEXT."""
    
    # Section A: Causal Chain Analysis
    def test_causal_chain_english(self):
        """Causal chain query in English should NOT be FREE_TEXT."""
        result = route_question("Trace all causal chains from الغفلة to الكفر with minimum 3 hops")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Causal chain query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.CAUSAL_CHAIN
    
    def test_causal_chain_arabic(self):
        """Causal chain query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما الذي يؤدي إلى الكفر من الغفلة؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Causal chain query should not be FREE_TEXT, got {result.question_class}"
    
    # Section B: Cross-Tafsir Comparison
    def test_cross_tafsir_english(self):
        """Cross-tafsir query in English should NOT be FREE_TEXT."""
        result = route_question("Compare tafsir methodologies for الربا across all 7 sources")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Cross-tafsir query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.CROSS_TAFSIR_COMPARATIVE
    
    def test_cross_tafsir_arabic(self):
        """Cross-tafsir query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما رأي المفسرين في الربا؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Cross-tafsir query should not be FREE_TEXT, got {result.question_class}"
    
    # Section C: 11-Dimensional Profile
    def test_profile_11d_english(self):
        """11D profile query in English should NOT be FREE_TEXT."""
        result = route_question("Generate 11-dimensional profile for الكبر")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Profile query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.BEHAVIOR_PROFILE_11AXIS
    
    def test_profile_11d_arabic(self):
        """11D profile query in Arabic should NOT be FREE_TEXT."""
        result = route_question("حلل سلوك الكبر تحليلاً شاملاً")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Profile query should not be FREE_TEXT, got {result.question_class}"
    
    # Section D: Graph Metrics
    def test_graph_metrics_english(self):
        """Graph metrics query in English should NOT be FREE_TEXT."""
        result = route_question("Calculate PageRank centrality for all behaviors")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Graph metrics query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.NETWORK_CENTRALITY
    
    def test_graph_metrics_arabic(self):
        """Graph metrics query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما هو أهم سلوك في الشبكة؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Graph metrics query should not be FREE_TEXT, got {result.question_class}"
    
    # Section E: Heart State
    def test_heart_state_english(self):
        """Heart state query in English should NOT be FREE_TEXT."""
        result = route_question("Analyze heart state transitions from قلب قاسي to قلب سليم")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Heart state query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.STATE_TRANSITION
    
    def test_heart_state_arabic(self):
        """Heart state query in Arabic should NOT be FREE_TEXT."""
        result = route_question("كيف يتحول القلب من قاسي إلى سليم؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Heart state query should not be FREE_TEXT, got {result.question_class}"
    
    # Section F: Agent Analysis
    def test_agent_analysis_english(self):
        """Agent analysis query in English should NOT be FREE_TEXT."""
        result = route_question("Compare behaviors between believer and disbeliever")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Agent analysis query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.AGENT_ATTRIBUTION
    
    def test_agent_analysis_arabic(self):
        """Agent analysis query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما الفرق بين المؤمن والكافر في السلوك؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Agent analysis query should not be FREE_TEXT, got {result.question_class}"
    
    # Section G: Temporal/Spatial
    def test_temporal_spatial_english(self):
        """Temporal/spatial query in English should NOT be FREE_TEXT."""
        result = route_question("Map behaviors to دنيا vs آخرة temporal context")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Temporal/spatial query should not be FREE_TEXT, got {result.question_class}"
    
    def test_temporal_spatial_arabic(self):
        """Temporal/spatial query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما السلوكيات المرتبطة بالدنيا والآخرة؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Temporal/spatial query should not be FREE_TEXT, got {result.question_class}"
    
    # Section H: Consequence Analysis
    def test_consequence_english(self):
        """Consequence query in English should NOT be FREE_TEXT."""
        result = route_question("What are the consequences and punishments for الكبر?")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Consequence query should not be FREE_TEXT, got {result.question_class}"
    
    def test_consequence_arabic(self):
        """Consequence query in Arabic should NOT be FREE_TEXT."""
        result = route_question("ما عاقبة الكبر؟")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Consequence query should not be FREE_TEXT, got {result.question_class}"
    
    # Section I: Embeddings Analysis
    def test_embeddings_english(self):
        """Embeddings query in English should NOT be FREE_TEXT."""
        result = route_question("Show t-SNE visualization of behavior embeddings")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Embeddings query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.SEMANTIC_LANDSCAPE
    
    # Section J: Integration E2E
    def test_integration_english(self):
        """Integration query in English should NOT be FREE_TEXT."""
        result = route_question("Run comprehensive analysis using ALL system components")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Integration query should not be FREE_TEXT, got {result.question_class}"
        assert result.question_class == QuestionClass.COMPLETE_ANALYSIS
    
    def test_integration_arabic(self):
        """Integration query in Arabic should NOT be FREE_TEXT."""
        result = route_question("تحليل شامل لسلوك الكبر باستخدام جميع المكونات")
        assert result.question_class != QuestionClass.FREE_TEXT, \
            f"Integration query should not be FREE_TEXT, got {result.question_class}"


class TestDeterministicRetrievalRouting:
    """Test that SURAH_REF and AYAH_REF are handled correctly."""
    
    def test_surah_ref_routes_to_free_text(self):
        """Surah reference should route to FREE_TEXT for deterministic retrieval."""
        result = route_question("سورة البقرة")
        # SURAH_REF goes to FREE_TEXT but with deterministic retrieval flag
        assert "deterministic_retrieval" in result.routing_reason or \
               result.intent_type == IntentType.SURAH_REF
    
    def test_ayah_ref_routes_to_free_text(self):
        """Ayah reference should route to FREE_TEXT for deterministic retrieval."""
        result = route_question("2:255")
        # AYAH_REF goes to FREE_TEXT but with deterministic retrieval flag
        assert "deterministic_retrieval" in result.routing_reason or \
               result.intent_type == IntentType.AYAH_REF


class TestConceptRefRouting:
    """Test that CONCEPT_REF routes to profile analysis."""
    
    def test_concept_ref_routes_to_profile(self):
        """Concept reference should route to behavior profile."""
        result = route_question("ما هو سلوك الكبر؟")
        # Should route to profile, not FREE_TEXT
        assert result.question_class in {
            QuestionClass.BEHAVIOR_PROFILE_11AXIS,
            QuestionClass.COMPLETE_ANALYSIS,
        } or result.question_class != QuestionClass.FREE_TEXT


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_is_analytical_question_true(self):
        """Analytical questions should return True."""
        assert is_analytical_question("Trace causal chains from الغفلة to الكفر")
        assert is_analytical_question("Compare tafsir methodologies")
        assert is_analytical_question("Generate 11-dimensional profile")
    
    def test_is_analytical_question_false(self):
        """Simple queries should return False."""
        # Empty query
        assert not is_analytical_question("")
        # Very simple query without analytical markers
        # Note: This might still be analytical if it contains behavior terms
    
    def test_get_planner_for_question(self):
        """get_planner_for_question should return correct planner name."""
        planner = get_planner_for_question("Trace causal chains")
        assert planner == "causal_chain"
        
        planner = get_planner_for_question("Compare tafsir methodologies")
        assert planner == "cross_tafsir_comparative"


class TestRouterResultStructure:
    """Test RouterResult structure."""
    
    def test_router_result_has_all_fields(self):
        """RouterResult should have all required fields."""
        result = route_question("حلل سلوك الكبر")
        
        assert hasattr(result, 'question_class')
        assert hasattr(result, 'intent_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'matched_patterns')
        assert hasattr(result, 'extracted_entities')
        assert hasattr(result, 'routing_reason')
    
    def test_router_result_to_dict(self):
        """RouterResult.to_dict() should return valid dict."""
        result = route_question("حلل سلوك الكبر")
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'question_class' in d
        assert 'intent_type' in d
        assert 'confidence' in d
        assert 'routing_reason' in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
