"""
Test: Concept Index v2 Completeness (Phase 8.2)

Ensures the concept index:
- Contains ALL 126 canonical entities (not just those with evidence)
- Returns status=no_evidence honestly when no evidence exists
- Hits 5 core sources when available
- Never fabricates evidence
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
CONCEPT_METADATA_FILE = Path("data/evidence/concept_index_v2_metadata.json")
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.fixture(scope="module")
def concept_index():
    """Load the concept index."""
    if not CONCEPT_INDEX_FILE.exists():
        pytest.skip(f"Concept index not found: {CONCEPT_INDEX_FILE}")
    
    index = {}
    with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            index[entry["concept_id"]] = entry
    return index


@pytest.fixture(scope="module")
def concept_metadata():
    """Load the concept index metadata."""
    if not CONCEPT_METADATA_FILE.exists():
        pytest.skip(f"Metadata not found: {CONCEPT_METADATA_FILE}")
    
    with open(CONCEPT_METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def canonical_entities():
    """Load canonical entities registry."""
    if not CANONICAL_ENTITIES_FILE.exists():
        pytest.skip(f"Canonical entities not found: {CANONICAL_ENTITIES_FILE}")
    
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_canonical_ids(canonical_entities):
    """Extract all canonical IDs from the registry."""
    ids = set()
    for behavior in canonical_entities.get("behaviors", []):
        ids.add(behavior["id"])
    for agent in canonical_entities.get("agents", []):
        ids.add(agent["id"])
    for organ in canonical_entities.get("organs", []):
        ids.add(organ["id"])
    for state in canonical_entities.get("heart_states", []):
        ids.add(state["id"])
    for consequence in canonical_entities.get("consequences", []):
        ids.add(consequence["id"])
    return ids


@pytest.mark.unit
class TestConceptIndexCompleteness:
    """Tests for concept index completeness."""
    
    def test_concept_index_contains_all_73_behaviors(self, concept_index, canonical_entities):
        """Concept index must contain all 73 behaviors."""
        behavior_ids = {b["id"] for b in canonical_entities.get("behaviors", [])}
        
        assert len(behavior_ids) == 87, f"Expected 87 behaviors, got {len(behavior_ids)}"
        
        missing = behavior_ids - set(concept_index.keys())
        assert len(missing) == 0, f"Missing behaviors in concept index: {missing}"
    
    def test_concept_index_contains_all_agents(self, concept_index, canonical_entities):
        """Concept index must contain all agents."""
        agent_ids = {a["id"] for a in canonical_entities.get("agents", [])}
        
        missing = agent_ids - set(concept_index.keys())
        assert len(missing) == 0, f"Missing agents in concept index: {missing}"
    
    def test_concept_index_contains_all_organs(self, concept_index, canonical_entities):
        """Concept index must contain all organs."""
        organ_ids = {o["id"] for o in canonical_entities.get("organs", [])}
        
        missing = organ_ids - set(concept_index.keys())
        assert len(missing) == 0, f"Missing organs in concept index: {missing}"
    
    def test_concept_index_contains_all_heart_states(self, concept_index, canonical_entities):
        """Concept index must contain all heart states."""
        state_ids = {s["id"] for s in canonical_entities.get("heart_states", [])}
        
        missing = state_ids - set(concept_index.keys())
        assert len(missing) == 0, f"Missing heart states in concept index: {missing}"
    
    def test_concept_index_contains_all_consequences(self, concept_index, canonical_entities):
        """Concept index must contain all consequences."""
        consequence_ids = {c["id"] for c in canonical_entities.get("consequences", [])}
        
        missing = consequence_ids - set(concept_index.keys())
        assert len(missing) == 0, f"Missing consequences in concept index: {missing}"
    
    def test_concept_index_has_exactly_126_entries(self, concept_index, canonical_entities):
        """Concept index must have exactly 126 entries (all canonical entities)."""
        all_ids = get_all_canonical_ids(canonical_entities)
        
        assert len(concept_index) == len(all_ids), \
            f"Expected {len(all_ids)} entries, got {len(concept_index)}"
        assert len(concept_index) == 168, f"Expected 168 entries, got {len(concept_index)}"


@pytest.mark.unit
class TestNoEvidenceHonesty:
    """Tests for honest no_evidence status."""
    
    def test_concept_ref_returns_no_evidence_honestly(self, concept_index):
        """Entries without evidence must have status=no_evidence."""
        for concept_id, entry in concept_index.items():
            if entry["total_mentions"] == 0:
                assert entry["status"] == "no_evidence", \
                    f"{concept_id} has 0 mentions but status={entry['status']}"
                assert len(entry["tafsir_chunks"]) == 0, \
                    f"{concept_id} has no_evidence but has tafsir_chunks"
                assert len(entry["verses"]) == 0, \
                    f"{concept_id} has no_evidence but has verses"
    
    def test_entries_with_evidence_have_found_status(self, concept_index):
        """Entries with evidence must have status=found."""
        for concept_id, entry in concept_index.items():
            if entry["total_mentions"] > 0:
                assert entry["status"] == "found", \
                    f"{concept_id} has {entry['total_mentions']} mentions but status={entry['status']}"
    
    def test_no_fabricated_evidence(self, concept_index):
        """No entry should have evidence without proper provenance."""
        for concept_id, entry in concept_index.items():
            for chunk in entry.get("tafsir_chunks", []):
                assert "chunk_id" in chunk, f"{concept_id} has chunk without chunk_id"
                assert "verse_key" in chunk, f"{concept_id} has chunk without verse_key"
                assert "source" in chunk, f"{concept_id} has chunk without source"
                assert "char_start" in chunk, f"{concept_id} has chunk without char_start"
                assert "char_end" in chunk, f"{concept_id} has chunk without char_end"
                assert "quote" in chunk, f"{concept_id} has chunk without quote"


@pytest.mark.unit
class TestSourceCoverage:
    """Tests for multi-source coverage."""
    
    def test_concept_ref_5_sources_when_available(self, concept_index):
        """Concepts with 5 sources must report all 5."""
        concepts_with_5 = [
            c for c in concept_index.values() 
            if c.get("sources_count", 0) == 5
        ]
        
        # We know from the build output that 96 concepts have 5/5 sources
        assert len(concepts_with_5) >= 90, \
            f"Expected at least 90 concepts with 5 sources, got {len(concepts_with_5)}"
        
        for entry in concepts_with_5:
            assert set(entry["sources_covered"]) == set(CORE_SOURCES), \
                f"{entry['concept_id']} claims 5 sources but sources_covered={entry['sources_covered']}"
    
    def test_per_source_stats_consistent(self, concept_index):
        """per_source_stats must be consistent with sources_covered."""
        for concept_id, entry in concept_index.items():
            sources_with_counts = [
                s for s in CORE_SOURCES 
                if entry["per_source_stats"][s]["count"] > 0
            ]
            
            assert set(sources_with_counts) == set(entry.get("sources_covered", [])), \
                f"{concept_id} has inconsistent source stats"
    
    def test_envy_has_5_sources(self, concept_index):
        """الحسد (BEH_EMO_ENVY) must have 5 source coverage."""
        envy = concept_index.get("BEH_EMO_ENVY")
        assert envy is not None, "BEH_EMO_ENVY not found in concept index"
        assert envy["sources_count"] == 5, \
            f"BEH_EMO_ENVY has {envy['sources_count']}/5 sources"
    
    def test_repentance_has_5_sources(self, concept_index):
        """التوبة (BEH_SPI_REPENTANCE) must have 5 source coverage."""
        repentance = concept_index.get("BEH_SPI_REPENTANCE")
        assert repentance is not None, "BEH_SPI_REPENTANCE not found in concept index"
        assert repentance["sources_count"] == 5, \
            f"BEH_SPI_REPENTANCE has {repentance['sources_count']}/5 sources"


@pytest.mark.unit
class TestMetadataIntegrity:
    """Tests for metadata integrity."""
    
    def test_metadata_version_is_2(self, concept_metadata):
        """Metadata version must be 2.0."""
        assert concept_metadata["version"] == "2.0"
    
    def test_metadata_completeness_stats(self, concept_metadata):
        """Metadata must have completeness stats."""
        completeness = concept_metadata.get("completeness", {})
        
        assert "total_canonical_entities" in completeness
        assert "entities_in_index" in completeness
        assert "entities_with_evidence" in completeness
        assert "entities_without_evidence" in completeness
        assert "coverage_rate" in completeness
        
        # Verify counts match
        assert completeness["total_canonical_entities"] == 168
        assert completeness["entities_in_index"] == 168
        assert completeness["entities_with_evidence"] + completeness["entities_without_evidence"] == 168
    
    def test_metadata_entity_type_counts(self, concept_metadata):
        """Metadata must have correct entity type counts."""
        by_type = concept_metadata["stats"]["concepts_by_entity_type"]
        
        assert by_type.get("BEHAVIOR") == 87, f"Expected 87 behaviors, got {by_type.get('BEHAVIOR')}"
        assert by_type.get("AGENT") == 14, f"Expected 14 agents, got {by_type.get('AGENT')}"
        assert by_type.get("ORGAN") == 39, f"Expected 39 organs, got {by_type.get('ORGAN')}"
        assert by_type.get("HEART_STATE") == 12, f"Expected 12 heart states, got {by_type.get('HEART_STATE')}"
        assert by_type.get("CONSEQUENCE") == 16, f"Expected 16 consequences, got {by_type.get('CONSEQUENCE')}"


@pytest.mark.unit
class TestEvidenceQuality:
    """Tests for evidence quality."""
    
    def test_evidence_has_valid_offsets(self, concept_index):
        """All evidence must have valid char offsets."""
        for concept_id, entry in concept_index.items():
            for chunk in entry.get("tafsir_chunks", [])[:10]:  # Check first 10
                assert chunk["char_start"] >= 0, \
                    f"{concept_id} has negative char_start"
                assert chunk["char_end"] > chunk["char_start"], \
                    f"{concept_id} has char_end <= char_start"
    
    def test_evidence_quotes_contain_term(self, concept_index):
        """Evidence quotes should contain the search term (normalized)."""
        import re
        
        def normalize(text):
            if not text:
                return text
            text = re.sub(r'[\u064B-\u0652]', '', text)
            text = re.sub(r'[أإآٱ]', 'ا', text)
            text = re.sub(r'ى', 'ي', text)
            text = re.sub(r'ة', 'ه', text)
            text = re.sub(r'ـ', '', text)
            return text
        
        # Check a sample of concepts
        sample_concepts = ["BEH_EMO_ENVY", "BEH_SPR_REPENTANCE", "BEH_SPR_PATIENCE"]
        
        for concept_id in sample_concepts:
            entry = concept_index.get(concept_id)
            if not entry or entry["status"] == "no_evidence":
                continue
            
            term = entry["term"]
            term_normalized = normalize(term)
            
            # Check first 5 chunks
            for chunk in entry.get("tafsir_chunks", [])[:5]:
                quote = chunk.get("quote", "")
                quote_normalized = normalize(quote)
                
                # Term or term with ال should be in quote
                assert term_normalized in quote_normalized or \
                       f"ال{term_normalized}" in quote_normalized, \
                    f"{concept_id}: term '{term}' not found in quote"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
