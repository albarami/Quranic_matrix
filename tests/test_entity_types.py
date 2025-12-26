"""
Test: Canonical Entity Types (Phase 6.0)

Ensures QueryRouter correctly classifies entities and doesn't conflate
agents/organs/states with behaviors.
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.query_router import QueryRouter, QueryIntent, get_query_router, TERM_TO_ENTITY_TYPE


class TestEntityTypeMapping:
    """Tests for canonical entity type mapping."""
    
    @pytest.fixture
    def router(self):
        return get_query_router()
    
    def test_entity_types_file_exists(self):
        """vocab/entity_types.json must exist."""
        entity_types_file = Path("vocab/entity_types.json")
        assert entity_types_file.exists(), "vocab/entity_types.json not found"
    
    def test_entity_types_has_required_types(self):
        """Entity types must include BEHAVIOR, AGENT, ORGAN, STATE."""
        entity_types_file = Path("vocab/entity_types.json")
        with open(entity_types_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        required_types = ["BEHAVIOR", "AGENT", "ORGAN", "STATE"]
        entity_types = data.get("entity_types", {})
        
        for t in required_types:
            assert t in entity_types, f"Missing entity type: {t}"
    
    def test_term_to_entity_type_loaded(self):
        """TERM_TO_ENTITY_TYPE must be loaded from vocab."""
        assert len(TERM_TO_ENTITY_TYPE) > 0, "TERM_TO_ENTITY_TYPE is empty"
    
    def test_entity_type_mapping_complete_for_core_vocab(self):
        """Core behavior/agent/organ terms must have entity type mappings."""
        core_behaviors = ["صبر", "شكر", "تقوى", "كبر", "ظلم"]
        core_agents = ["مؤمن", "كافر", "منافق"]
        core_organs = ["قلب", "لسان", "عين"]
        
        for term in core_behaviors:
            assert term in TERM_TO_ENTITY_TYPE, f"Missing behavior: {term}"
            assert TERM_TO_ENTITY_TYPE[term]["entity_type"] == "BEHAVIOR"
        
        for term in core_agents:
            assert term in TERM_TO_ENTITY_TYPE, f"Missing agent: {term}"
            assert TERM_TO_ENTITY_TYPE[term]["entity_type"] == "AGENT"
        
        for term in core_organs:
            assert term in TERM_TO_ENTITY_TYPE, f"Missing organ: {term}"
            assert TERM_TO_ENTITY_TYPE[term]["entity_type"] == "ORGAN"


class TestRouterEntityTyping:
    """Tests that router correctly assigns entity types."""
    
    @pytest.fixture
    def router(self):
        return get_query_router()
    
    def test_router_does_not_classify_agent_as_behavior(self, router):
        """Agent terms must be typed as AGENT, not BEHAVIOR."""
        result = router.route("المؤمن")
        
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.entity_type == "AGENT", f"Expected AGENT, got {result.entity_type}"
        assert result.canonical_id == "AGT_BELIEVER"
    
    def test_router_does_not_classify_organ_as_behavior(self, router):
        """Organ terms must be typed as ORGAN, not BEHAVIOR."""
        result = router.route("القلب")
        
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.entity_type == "ORGAN", f"Expected ORGAN, got {result.entity_type}"
        assert result.canonical_id == "ORG_HEART"
    
    def test_router_classifies_behavior_correctly(self, router):
        """Behavior terms must be typed as BEHAVIOR."""
        result = router.route("الصبر")
        
        assert result.intent == QueryIntent.CONCEPT_REF
        assert result.entity_type == "BEHAVIOR", f"Expected BEHAVIOR, got {result.entity_type}"
        assert result.canonical_id == "BEH_EMO_PATIENCE"
    
    def test_router_handles_concept_id_with_entity_type(self, router):
        """Concept IDs should return correct entity type."""
        test_cases = [
            ("BEH_EMO_PATIENCE", "BEHAVIOR"),
            ("AGT_BELIEVER", "AGENT"),
            ("ORG_HEART", "ORGAN"),
            ("STA_IMAN", "STATE"),
        ]
        
        for concept_id, expected_type in test_cases:
            result = router.route(concept_id)
            assert result.intent == QueryIntent.CONCEPT_REF
            assert result.entity_type == expected_type, f"{concept_id}: expected {expected_type}, got {result.entity_type}"
    
    def test_router_result_includes_entity_type_in_dict(self, router):
        """RouterResult.to_dict() must include entity_type."""
        result = router.route("الصبر")
        d = result.to_dict()
        
        assert "entity_type" in d
        assert "canonical_id" in d
        assert d["entity_type"] == "BEHAVIOR"


class TestRouterBackwardCompatibility:
    """Ensure router still works for existing query types."""
    
    @pytest.fixture
    def router(self):
        return get_query_router()
    
    def test_ayah_ref_still_works(self, router):
        """AYAH_REF detection must still work."""
        result = router.route("2:255")
        assert result.intent == QueryIntent.AYAH_REF
        assert result.extracted_ref == "2:255"
    
    def test_surah_ref_still_works(self, router):
        """SURAH_REF detection must still work."""
        result = router.route("سورة البقرة")
        assert result.intent == QueryIntent.SURAH_REF
    
    def test_free_text_still_works(self, router):
        """FREE_TEXT fallback must still work."""
        result = router.route("كيف أتعامل مع الابتلاء")
        assert result.intent == QueryIntent.FREE_TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
