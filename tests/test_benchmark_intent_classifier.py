"""
Unit tests for the deterministic intent classifier.

Tests that benchmark questions from each section (A-J) are correctly
classified to their expected intent types, NOT FREE_TEXT.

Hard gate: 2 questions per section (20 tests total) must pass.
"""

import pytest
from src.ml.intent_classifier import (
    classify_intent,
    IntentType,
    IntentResult,
    get_section_for_intent,
    get_intent_for_section,
)


class TestSectionAGraphCausal:
    """Section A: Causal chain analysis questions."""
    
    def test_a01_complete_destruction_pathway(self):
        """A01: Trace ALL distinct causal chains."""
        question = (
            "Trace ALL distinct causal chains (minimum 3 hops) from الغفلة (heedlessness) "
            "to الكفر (disbelief). For each chain, provide: intermediate behaviors, "
            "verse evidence for each link, and tafsir from Ibn Kathir and Qurtubi "
            "confirming the causal relationship."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.GRAPH_CAUSAL, f"Expected GRAPH_CAUSAL, got {result.intent}"
    
    def test_a02_salvations_shortest_path(self):
        """A02: Minimum behavioral transformations."""
        question = (
            "What is the minimum number of behavioral transformations required to move "
            "from الكبر (arrogance) to الإخلاص (sincerity)? Show the optimal path with verse evidence."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.GRAPH_CAUSAL, f"Expected GRAPH_CAUSAL, got {result.intent}"


class TestSectionBCrossTafsir:
    """Section B: Cross-tafsir analysis questions."""
    
    def test_b01_methodological_fingerprinting(self):
        """B01: Tafsir source fingerprints."""
        question = (
            "For each of the 5 tafsir sources, identify their top 5 most frequently "
            "mentioned behaviors. Do these fingerprints align with known methodological emphases?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.CROSS_TAFSIR_ANALYSIS, f"Expected CROSS_TAFSIR_ANALYSIS, got {result.intent}"
    
    def test_b02_riba_divergence(self):
        """B02: Compare tafsir interpretations."""
        question = (
            "Compare all 5 tafsir interpretations of الربا across every verse where it appears. "
            "Create a divergence matrix showing where they agree and disagree."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.CROSS_TAFSIR_ANALYSIS, f"Expected CROSS_TAFSIR_ANALYSIS, got {result.intent}"


class TestSectionCProfile11D:
    """Section C: 11-dimensional profile questions."""
    
    def test_c01_complete_hasad_profile(self):
        """C01: Full 11-dimensional profile."""
        question = (
            "Generate the full 11-dimensional profile for الحسد: organic, situational, "
            "systemic, spatial, temporal, agent, source, evaluation, heart-type, "
            "consequence, and related behaviors."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.PROFILE_11D, f"Expected PROFILE_11D, got {result.intent}"
    
    def test_c02_organic_axis_dominance(self):
        """C02: Axis coverage analysis."""
        question = (
            "Which axis of the 11 dimensions has the most data coverage across all behaviors? "
            "Which is most sparse? Quantify per axis."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.PROFILE_11D, f"Expected PROFILE_11D, got {result.intent}"


class TestSectionDGraphMetrics:
    """Section D: Graph metrics questions."""
    
    def test_d01_global_network_statistics(self):
        """D01: Network statistics."""
        question = (
            "Report: node count, edge count, density, average degree, diameter, "
            "average clustering coefficient for the complete behavioral graph."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.GRAPH_METRICS, f"Expected GRAPH_METRICS, got {result.intent}"
    
    def test_d02_centrality_ranking(self):
        """D02: Degree centrality ranking."""
        question = (
            "Rank all 73 behaviors by degree centrality (total connections). "
            "What characterizes high-degree behaviors?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.GRAPH_METRICS, f"Expected GRAPH_METRICS, got {result.intent}"


class TestSectionEHeartState:
    """Section E: Heart state questions."""
    
    def test_e01_complete_heart_state_inventory(self):
        """E01: Heart states inventory."""
        question = (
            "List all heart states mentioned in Quran with their behavioral correlates: "
            "قلب سليم، قلب قاسٍ، قلب مريض، قلب ميت، قلب مطمئن، etc."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.HEART_STATE, f"Expected HEART_STATE, got {result.intent}"
    
    def test_e02_heart_state_transitions(self):
        """E02: Heart state transitions."""
        question = (
            "Map all possible transitions between heart states. Which behaviors cause "
            "قلب سليم to degrade to قلب قاسٍ? Which restore it?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.HEART_STATE, f"Expected HEART_STATE, got {result.intent}"


class TestSectionFAgentAnalysis:
    """Section F: Agent analysis questions."""
    
    def test_f01_agent_type_distribution(self):
        """F01: Agent types."""
        question = (
            "For each of the 73 behaviors, identify which agent types perform them: "
            "believer, disbeliever, munafiq, Prophet, Allah attribution."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.AGENT_ANALYSIS, f"Expected AGENT_ANALYSIS, got {result.intent}"
    
    def test_f02_munafiq_behaviors(self):
        """F02: Munafiq-specific behaviors."""
        question = (
            "Which behaviors are exclusively attributed to المنافقين in the Quran? "
            "Provide verse evidence for each."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.AGENT_ANALYSIS, f"Expected AGENT_ANALYSIS, got {result.intent}"


class TestSectionGTemporalSpatial:
    """Section G: Temporal/spatial context questions."""
    
    def test_g01_dunya_akhira_distribution(self):
        """G01: Dunya/Akhira distribution."""
        question = (
            "Classify all behaviors by their temporal context: دنيا only, آخرة only, "
            "or both. Which behaviors span both realms?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.TEMPORAL_SPATIAL, f"Expected TEMPORAL_SPATIAL, got {result.intent}"
    
    def test_g02_spatial_context(self):
        """G02: Spatial context analysis."""
        question = (
            "Map behaviors to their spatial contexts: مسجد، بيت، سوق، etc. "
            "Which behaviors are context-specific vs universal?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.TEMPORAL_SPATIAL, f"Expected TEMPORAL_SPATIAL, got {result.intent}"


class TestSectionHConsequenceAnalysis:
    """Section H: Consequence analysis questions."""
    
    def test_h01_consequence_severity(self):
        """H01: Consequence severity ranking."""
        question = (
            "Rank all negative behaviors by consequence severity based on Quranic punishment "
            "descriptions. Which lead to الخسران المبين?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.CONSEQUENCE_ANALYSIS, f"Expected CONSEQUENCE_ANALYSIS, got {result.intent}"
    
    def test_h02_reward_analysis(self):
        """H02: Reward analysis."""
        question = (
            "For positive behaviors, map the promised rewards: جنة، رضوان، مغفرة. "
            "Which behaviors have the most explicit reward promises?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.CONSEQUENCE_ANALYSIS, f"Expected CONSEQUENCE_ANALYSIS, got {result.intent}"


class TestSectionIEmbeddingsAnalysis:
    """Section I: Embeddings analysis questions."""
    
    def test_i01_tsne_visualization(self):
        """I01: t-SNE visualization."""
        question = (
            "Generate a t-SNE visualization of all 73 behaviors in embedding space. "
            "Which behaviors cluster together? Do clusters align with categories?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.EMBEDDINGS_ANALYSIS, f"Expected EMBEDDINGS_ANALYSIS, got {result.intent}"
    
    def test_i02_nearest_neighbors(self):
        """I02: Nearest neighbors analysis."""
        question = (
            "For each behavior, find its 5 nearest neighbors in embedding space. "
            "Do semantic neighbors match graph neighbors?"
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.EMBEDDINGS_ANALYSIS, f"Expected EMBEDDINGS_ANALYSIS, got {result.intent}"


class TestSectionJIntegrationE2E:
    """Section J: End-to-end integration questions."""
    
    def test_j01_comprehensive_analysis(self):
        """J01: Comprehensive analysis using ALL components."""
        question = (
            "Using ALL system components (graph, tafsir, embeddings, taxonomy), "
            "provide a comprehensive analysis of الكبر including causal chains, "
            "11D profile, cross-tafsir consensus, and embedding neighbors."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.INTEGRATION_E2E, f"Expected INTEGRATION_E2E, got {result.intent}"
    
    def test_j02_behavior_genome(self):
        """J02: Complete behavior genome."""
        question = (
            "Generate the complete 'genome' for الصبر: every dimension, every connection, "
            "every tafsir mention, every verse, every consequence. Full system integration."
        )
        result = classify_intent(question)
        assert result.intent != IntentType.FREE_TEXT, f"Expected non-FREE_TEXT, got {result.intent}"
        assert result.intent == IntentType.INTEGRATION_E2E, f"Expected INTEGRATION_E2E, got {result.intent}"


class TestHelperFunctions:
    """Test helper functions for section/intent mapping."""
    
    def test_section_to_intent_mapping(self):
        """Test section letter to intent mapping."""
        assert get_intent_for_section("A") == IntentType.GRAPH_CAUSAL
        assert get_intent_for_section("B") == IntentType.CROSS_TAFSIR_ANALYSIS
        assert get_intent_for_section("C") == IntentType.PROFILE_11D
        assert get_intent_for_section("D") == IntentType.GRAPH_METRICS
        assert get_intent_for_section("E") == IntentType.HEART_STATE
        assert get_intent_for_section("F") == IntentType.AGENT_ANALYSIS
        assert get_intent_for_section("G") == IntentType.TEMPORAL_SPATIAL
        assert get_intent_for_section("H") == IntentType.CONSEQUENCE_ANALYSIS
        assert get_intent_for_section("I") == IntentType.EMBEDDINGS_ANALYSIS
        assert get_intent_for_section("J") == IntentType.INTEGRATION_E2E
    
    def test_intent_to_section_mapping(self):
        """Test intent to section letter mapping."""
        assert get_section_for_intent(IntentType.GRAPH_CAUSAL) == "A"
        assert get_section_for_intent(IntentType.CROSS_TAFSIR_ANALYSIS) == "B"
        assert get_section_for_intent(IntentType.PROFILE_11D) == "C"
        assert get_section_for_intent(IntentType.GRAPH_METRICS) == "D"
        assert get_section_for_intent(IntentType.HEART_STATE) == "E"
        assert get_section_for_intent(IntentType.AGENT_ANALYSIS) == "F"
        assert get_section_for_intent(IntentType.TEMPORAL_SPATIAL) == "G"
        assert get_section_for_intent(IntentType.CONSEQUENCE_ANALYSIS) == "H"
        assert get_section_for_intent(IntentType.EMBEDDINGS_ANALYSIS) == "I"
        assert get_section_for_intent(IntentType.INTEGRATION_E2E) == "J"


class TestNonBenchmarkIntents:
    """Test that standard intents still work."""
    
    def test_surah_ref_arabic(self):
        """Test Arabic surah reference."""
        result = classify_intent("ما تفسير سورة الفاتحة؟")
        assert result.intent == IntentType.SURAH_REF
    
    def test_ayah_ref_numeric(self):
        """Test numeric ayah reference."""
        result = classify_intent("What is the meaning of 2:255?")
        assert result.intent == IntentType.AYAH_REF
    
    def test_free_text_fallback(self):
        """Test that generic questions fall back to FREE_TEXT."""
        result = classify_intent("Hello, how are you?")
        assert result.intent == IntentType.FREE_TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
