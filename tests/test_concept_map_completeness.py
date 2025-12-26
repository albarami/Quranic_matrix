"""
Phase 10.4: Deterministic concept map completeness tests.

Ensures all canonical entities have entries in the concept index
with proper verse evidence.
"""
import pytest
import json
from pathlib import Path

CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")


@pytest.fixture(scope="module")
def canonical_entities():
    """Load canonical entities vocabulary."""
    if not CANONICAL_ENTITIES_FILE.exists():
        pytest.skip("Canonical entities file not found")
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def concept_index():
    """Load concept index."""
    if not CONCEPT_INDEX_FILE.exists():
        pytest.skip("Concept index file not found")
    index = {}
    with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            index[entry["concept_id"]] = entry
    return index


@pytest.mark.phase10
class TestConceptMapCompleteness:
    """All canonical entities must have concept index entries."""
    
    def test_all_behaviors_in_concept_index(self, canonical_entities, concept_index):
        """All 73+ behaviors must have concept index entries."""
        behaviors = canonical_entities.get("behaviors", [])
        behavior_ids = {b.get("id") for b in behaviors}
        concept_ids = set(concept_index.keys())
        
        missing = behavior_ids - concept_ids
        assert len(missing) == 0, f"Behaviors missing from concept index: {missing}"
        assert len(behaviors) >= 73, f"Expected 73+ behaviors, got {len(behaviors)}"
    
    def test_all_agents_in_concept_index(self, canonical_entities, concept_index):
        """All agents must have concept index entries."""
        agents = canonical_entities.get("agents", [])
        agent_ids = {a.get("id") for a in agents}
        concept_ids = set(concept_index.keys())
        
        missing = agent_ids - concept_ids
        assert len(missing) == 0, f"Agents missing from concept index: {missing}"
    
    def test_all_organs_in_concept_index(self, canonical_entities, concept_index):
        """All organs must have concept index entries."""
        organs = canonical_entities.get("organs", [])
        organ_ids = {o.get("id") for o in organs}
        concept_ids = set(concept_index.keys())
        
        missing = organ_ids - concept_ids
        assert len(missing) == 0, f"Organs missing from concept index: {missing}"
    
    def test_all_heart_states_in_concept_index(self, canonical_entities, concept_index):
        """All heart states must have concept index entries."""
        heart_states = canonical_entities.get("heart_states", [])
        state_ids = {s.get("id") for s in heart_states}
        concept_ids = set(concept_index.keys())
        
        missing = state_ids - concept_ids
        assert len(missing) == 0, f"Heart states missing from concept index: {missing}"
    
    def test_all_consequences_in_concept_index(self, canonical_entities, concept_index):
        """All consequences must have concept index entries."""
        consequences = canonical_entities.get("consequences", [])
        consequence_ids = {c.get("id") for c in consequences}
        concept_ids = set(concept_index.keys())
        
        missing = consequence_ids - concept_ids
        assert len(missing) == 0, f"Consequences missing from concept index: {missing}"


@pytest.mark.phase10
class TestConceptIndexQuality:
    """Concept index entries must have proper evidence."""
    
    def test_concepts_have_verses(self, concept_index):
        """Each concept should have at least one verse."""
        concepts_without_verses = []
        
        for cid, entry in concept_index.items():
            verses = entry.get("verses", [])
            if len(verses) == 0:
                concepts_without_verses.append(cid)
        
        # Allow up to 10% without verses (some concepts may be abstract)
        total = len(concept_index)
        pct = len(concepts_without_verses) / total * 100 if total > 0 else 0
        
        assert pct <= 10, \
            f"{len(concepts_without_verses)}/{total} ({pct:.1f}%) concepts have no verses: {concepts_without_verses[:10]}"
    
    def test_concepts_have_terms(self, concept_index):
        """Each concept must have an Arabic term."""
        concepts_without_terms = []
        
        for cid, entry in concept_index.items():
            term = entry.get("term", "")
            if not term:
                concepts_without_terms.append(cid)
        
        assert len(concepts_without_terms) == 0, \
            f"Concepts without Arabic terms: {concepts_without_terms}"
    
    def test_verse_keys_are_valid_format(self, concept_index):
        """Verse keys should be in format 'surah:ayah'."""
        invalid_verse_keys = []
        
        for cid, entry in concept_index.items():
            for v in entry.get("verses", []):
                vk = v.get("verse_key", "")
                if vk and ":" not in vk:
                    invalid_verse_keys.append((cid, vk))
        
        assert len(invalid_verse_keys) == 0, \
            f"Invalid verse key formats: {invalid_verse_keys[:10]}"


@pytest.mark.phase10
class TestConceptIndexIntegrity:
    """Concept index integrity checks."""
    
    def test_no_duplicate_concept_ids(self, concept_index):
        """Concept IDs must be unique (already enforced by dict, but verify file)."""
        seen_ids = set()
        duplicates = []
        
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                cid = entry.get("concept_id")
                if cid in seen_ids:
                    duplicates.append(cid)
                seen_ids.add(cid)
        
        assert len(duplicates) == 0, f"Duplicate concept IDs: {duplicates}"
    
    def test_concept_ids_match_entity_type(self, concept_index):
        """Concept IDs should match their entity type prefix."""
        mismatched = []
        
        # Map entity types to their valid prefixes (some have legacy prefixes)
        prefix_map = {
            "BEHAVIOR": ["BEH_"],
            "AGENT": ["AGT_"],
            "ORGAN": ["ORG_"],
            "HEART_STATE": ["HRT_", "HST_"],  # HRT_ is the actual prefix used
            "CONSEQUENCE": ["CON_", "CSQ_"],  # CSQ_ is the actual prefix used
        }
        
        for cid, entry in concept_index.items():
            entity_type = entry.get("entity_type", "")
            valid_prefixes = prefix_map.get(entity_type, [])
            
            if valid_prefixes and not any(cid.startswith(p) for p in valid_prefixes):
                mismatched.append((cid, entity_type, valid_prefixes))
        
        assert len(mismatched) == 0, \
            f"Concept IDs not matching entity type: {mismatched[:10]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "phase10"])
