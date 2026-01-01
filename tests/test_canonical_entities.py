"""
Test: Canonical Entity Registry (Phase 6.1)

Ensures entity types are not conflated and all IDs are properly mapped.
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VOCAB_DIR = Path(__file__).parent.parent / "vocab"
CANONICAL_FILE = VOCAB_DIR / "canonical_entities.json"


@pytest.mark.unit
class TestCanonicalEntityRegistry:
    """Tests for the canonical entity registry."""
    
    @pytest.fixture(scope="class")
    def registry(self):
        """Load the canonical entities registry."""
        with open(CANONICAL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_registry_file_exists(self):
        """Canonical entities file must exist."""
        assert CANONICAL_FILE.exists(), f"Missing: {CANONICAL_FILE}"
    
    def test_registry_has_required_sections(self, registry):
        """Registry must have all required sections."""
        required = ['entity_types', 'behaviors', 'agents', 'organs', 
                    'heart_states', 'consequences', 'axes_11', 'term_to_entity']
        for section in required:
            assert section in registry, f"Missing section: {section}"
    
    def test_behaviors_count(self, registry):
        """Should have 87 behaviors."""
        behaviors = registry['behaviors']
        assert len(behaviors) == 87, f"Expected 87 behaviors, got {len(behaviors)}"
    
    def test_behavior_ids_unique(self, registry):
        """All behavior IDs must be unique."""
        ids = [b['id'] for b in registry['behaviors']]
        assert len(ids) == len(set(ids)), "Duplicate behavior IDs found"
    
    def test_behavior_ids_have_prefix(self, registry):
        """All behavior IDs must start with BEH_."""
        for b in registry['behaviors']:
            assert b['id'].startswith('BEH_'), f"Invalid behavior ID: {b['id']}"
    
    def test_agent_ids_have_prefix(self, registry):
        """All agent IDs must start with AGT_."""
        for a in registry['agents']:
            assert a['id'].startswith('AGT_'), f"Invalid agent ID: {a['id']}"
    
    def test_organ_ids_have_prefix(self, registry):
        """All organ IDs must start with ORG_."""
        for o in registry['organs']:
            assert o['id'].startswith('ORG_'), f"Invalid organ ID: {o['id']}"
    
    def test_heart_state_ids_have_prefix(self, registry):
        """All heart state IDs must start with HRT_."""
        for h in registry['heart_states']:
            assert h['id'].startswith('HRT_'), f"Invalid heart state ID: {h['id']}"
    
    def test_consequence_ids_have_prefix(self, registry):
        """All consequence IDs must start with CSQ_."""
        for c in registry['consequences']:
            assert c['id'].startswith('CSQ_'), f"Invalid consequence ID: {c['id']}"


@pytest.mark.unit
class TestEntityTypeNoConflation:
    """Tests to ensure agents/organs/states are not classified as behaviors."""
    
    @pytest.fixture(scope="class")
    def registry(self):
        with open(CANONICAL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def term_to_entity(self, registry):
        return registry['term_to_entity']
    
    def test_agent_terms_not_behaviors(self, term_to_entity):
        """Agent terms must not be classified as BEHAVIOR."""
        agent_terms = ['مؤمن', 'كافر', 'منافق', 'نبي', 'رسول', 'ملك', 'شيطان']
        for term in agent_terms:
            if term in term_to_entity:
                assert term_to_entity[term]['entity_type'] == 'AGENT', \
                    f"'{term}' should be AGENT, not {term_to_entity[term]['entity_type']}"
    
    def test_organ_terms_not_behaviors(self, term_to_entity):
        """Organ terms must not be classified as BEHAVIOR."""
        organ_terms = ['قلب', 'لسان', 'عين', 'أذن', 'يد', 'نفس']
        for term in organ_terms:
            if term in term_to_entity:
                assert term_to_entity[term]['entity_type'] == 'ORGAN', \
                    f"'{term}' should be ORGAN, not {term_to_entity[term]['entity_type']}"
    
    def test_heart_state_terms_not_behaviors(self, term_to_entity):
        """Heart state terms must not be classified as BEHAVIOR."""
        heart_terms = ['قلب سليم', 'قلب قاسٍ', 'قلب مريض']
        for term in heart_terms:
            if term in term_to_entity:
                assert term_to_entity[term]['entity_type'] == 'HEART_STATE', \
                    f"'{term}' should be HEART_STATE, not {term_to_entity[term]['entity_type']}"
    
    def test_behavior_terms_are_behaviors(self, term_to_entity):
        """Behavior terms must be classified as BEHAVIOR."""
        behavior_terms = ['صبر', 'شكر', 'تقوى', 'توبة', 'كبر', 'ظلم', 'حسد', 'غفلة']
        for term in behavior_terms:
            if term in term_to_entity:
                assert term_to_entity[term]['entity_type'] == 'BEHAVIOR', \
                    f"'{term}' should be BEHAVIOR, not {term_to_entity[term]['entity_type']}"


@pytest.mark.unit
class TestCanonicalVocabCoverage:
    """Tests to ensure all referenced IDs exist in the registry."""
    
    @pytest.fixture(scope="class")
    def registry(self):
        with open(CANONICAL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_term_to_entity_ids_exist(self, registry):
        """All canonical_ids in term_to_entity must exist in the registry."""
        # Build set of all valid IDs
        valid_ids = set()
        valid_ids.update(b['id'] for b in registry['behaviors'])
        valid_ids.update(a['id'] for a in registry['agents'])
        valid_ids.update(o['id'] for o in registry['organs'])
        valid_ids.update(h['id'] for h in registry['heart_states'])
        valid_ids.update(c['id'] for c in registry['consequences'])
        
        # Check all term_to_entity mappings
        missing = []
        for term, mapping in registry['term_to_entity'].items():
            canonical_id = mapping.get('canonical_id')
            if canonical_id and canonical_id not in valid_ids:
                missing.append((term, canonical_id))
        
        assert len(missing) == 0, f"Missing canonical IDs: {missing}"
    
    def test_all_behaviors_have_arabic(self, registry):
        """All behaviors must have Arabic labels."""
        for b in registry['behaviors']:
            assert 'ar' in b and b['ar'], f"Behavior {b['id']} missing Arabic label"
    
    def test_all_behaviors_have_category(self, registry):
        """All behaviors must have a category."""
        valid_categories = {'speech', 'financial', 'emotional', 'spiritual', 
                           'social', 'cognitive', 'physical'}
        for b in registry['behaviors']:
            assert 'category' in b, f"Behavior {b['id']} missing category"
            assert b['category'] in valid_categories, \
                f"Behavior {b['id']} has invalid category: {b['category']}"


@pytest.mark.unit
class TestAxes11Coverage:
    """Tests for the 11-axis classification system."""
    
    @pytest.fixture(scope="class")
    def registry(self):
        with open(CANONICAL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_has_11_axes(self, registry):
        """Must have exactly 11 axes."""
        axes = registry['axes_11']
        assert len(axes) == 11, f"Expected 11 axes, got {len(axes)}"
    
    def test_axes_have_values(self, registry):
        """Each axis must have at least one value."""
        for axis_key, axis_data in registry['axes_11'].items():
            assert 'values' in axis_data, f"Axis {axis_key} missing values"
            assert len(axis_data['values']) > 0, f"Axis {axis_key} has no values"
    
    def test_axis_values_have_ids(self, registry):
        """All axis values must have IDs starting with AXV_."""
        for axis_key, axis_data in registry['axes_11'].items():
            for value in axis_data['values']:
                assert 'id' in value, f"Axis {axis_key} value missing ID"
                assert value['id'].startswith('AXV_'), \
                    f"Invalid axis value ID: {value['id']}"


@pytest.mark.integration
class TestQueryRouterEntityTyping:
    """Integration tests for QueryRouter entity typing."""
    
    @pytest.fixture(scope="class")
    def router(self):
        from src.ml.query_router import get_query_router
        return get_query_router()
    
    def test_router_returns_entity_type_for_behavior(self, router):
        """Router must return entity_type=BEHAVIOR for behavior queries."""
        result = router.route("الصبر")
        assert result.entity_type == "BEHAVIOR", \
            f"Expected BEHAVIOR, got {result.entity_type}"
    
    def test_router_returns_entity_type_for_agent(self, router):
        """Router must return entity_type=AGENT for agent queries."""
        result = router.route("المؤمن")
        assert result.entity_type == "AGENT", \
            f"Expected AGENT, got {result.entity_type}"
    
    def test_router_returns_entity_type_for_organ(self, router):
        """Router must return entity_type=ORGAN for organ queries."""
        result = router.route("القلب")
        assert result.entity_type == "ORGAN", \
            f"Expected ORGAN, got {result.entity_type}"
    
    def test_router_returns_canonical_id(self, router):
        """Router must return canonical_id for known terms."""
        result = router.route("الحسد")
        assert result.canonical_id == "BEH_EMO_ENVY", \
            f"Expected BEH_EMO_ENVY, got {result.canonical_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
