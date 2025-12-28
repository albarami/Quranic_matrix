"""
Test: Concept Evidence Index (Phase 6.1)

Ensures the deterministic concept index has valid data with offsets.
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v1.jsonl")
METADATA_FILE = Path("data/evidence/concept_index_v1_metadata.json")
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


def load_concept_index():
    """Load the concept index."""
    concepts = {}
    with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            concepts[entry['concept_id']] = entry
    return concepts


class TestConceptIndexExists:
    """Tests that concept index files exist."""
    
    def test_concept_index_file_exists(self):
        """concept_index_v1.jsonl must exist."""
        assert CONCEPT_INDEX_FILE.exists(), f"{CONCEPT_INDEX_FILE} not found"
    
    def test_metadata_file_exists(self):
        """concept_index_v1_metadata.json must exist."""
        assert METADATA_FILE.exists(), f"{METADATA_FILE} not found"


class TestConceptIndexContent:
    """Tests for concept index content."""
    
    @pytest.fixture(scope="class")
    def concept_index(self):
        return load_concept_index()
    
    def test_concept_index_nonempty_for_top_behaviors(self, concept_index):
        """Top behaviors must have entries in the index."""
        top_behaviors = [
            "BEH_EMO_PATIENCE",  # صبر
            "BEH_EMO_GRATITUDE",  # شكر
            "BEH_SPI_TAQWA",  # تقوى
            "BEH_COG_ARROGANCE",  # كبر
        ]
        
        for behavior_id in top_behaviors:
            assert behavior_id in concept_index, f"Missing behavior: {behavior_id}"
            entry = concept_index[behavior_id]
            assert entry['total_mentions'] > 0, f"{behavior_id} has no mentions"
    
    def test_concept_ref_returns_core_sources_when_available(self, concept_index):
        """Concepts should have evidence from multiple core sources."""
        # Check top concepts have good source coverage
        concepts_with_good_coverage = 0
        
        for concept_id, entry in concept_index.items():
            sources_covered = sum(
                1 for s in CORE_SOURCES 
                if entry['per_source_stats'].get(s, {}).get('count', 0) > 0
            )
            if sources_covered >= 3:  # At least 3/5 sources
                concepts_with_good_coverage += 1
        
        # At least 50% of concepts should have good coverage
        coverage_rate = concepts_with_good_coverage / len(concept_index)
        assert coverage_rate >= 0.5, f"Only {coverage_rate:.1%} concepts have 3+ sources"
    
    def test_offsets_valid_for_concept_mentions(self, concept_index):
        """Tafsir chunks must have valid char_start/char_end offsets."""
        for concept_id, entry in concept_index.items():
            for chunk in entry.get('tafsir_chunks', [])[:10]:  # Check first 10
                assert 'char_start' in chunk, f"{concept_id}: missing char_start"
                assert 'char_end' in chunk, f"{concept_id}: missing char_end"
                assert chunk['char_start'] >= 0, f"{concept_id}: invalid char_start"
                assert chunk['char_end'] > chunk['char_start'], f"{concept_id}: char_end <= char_start"
                assert 'quote' in chunk, f"{concept_id}: missing quote"
    
    def test_chunks_have_required_fields(self, concept_index):
        """Each tafsir chunk must have chunk_id, verse_key, source."""
        for concept_id, entry in concept_index.items():
            for chunk in entry.get('tafsir_chunks', [])[:10]:
                assert 'chunk_id' in chunk, f"{concept_id}: missing chunk_id"
                assert 'verse_key' in chunk, f"{concept_id}: missing verse_key"
                assert 'source' in chunk, f"{concept_id}: missing source"
    
    def test_entity_types_are_valid(self, concept_index):
        """All entries must have valid entity_type."""
        # Updated to include HEART_STATE and CONSEQUENCE per canonical_entities.json
        valid_types = {"BEHAVIOR", "AGENT", "ORGAN", "STATE", "HEART_STATE", "CONSEQUENCE", "AXIS_VALUE"}
        
        for concept_id, entry in concept_index.items():
            assert entry.get('entity_type') in valid_types, \
                f"{concept_id}: invalid entity_type {entry.get('entity_type')}"


class TestConceptIndexMetadata:
    """Tests for concept index metadata."""
    
    def test_metadata_has_version(self):
        """Metadata must include version."""
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert 'version' in metadata
        assert 'created_at' in metadata
        assert 'concept_count' in metadata
    
    def test_metadata_stats_present(self):
        """Metadata must include stats."""
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert 'stats' in metadata
        assert 'concepts_with_evidence' in metadata['stats']
        assert 'concepts_by_entity_type' in metadata['stats']


class TestConceptIndexCoverage:
    """Tests for source coverage in concept index."""
    
    @pytest.fixture(scope="class")
    def concept_index(self):
        return load_concept_index()
    
    def test_top_concepts_have_5_source_coverage(self, concept_index):
        """Top concepts by mentions should have 5/5 source coverage."""
        # Sort by mentions
        sorted_concepts = sorted(
            concept_index.values(), 
            key=lambda x: -x['total_mentions']
        )[:10]
        
        full_coverage_count = 0
        for entry in sorted_concepts:
            sources_covered = sum(
                1 for s in CORE_SOURCES 
                if entry['per_source_stats'].get(s, {}).get('count', 0) > 0
            )
            if sources_covered == 5:
                full_coverage_count += 1
        
        # Top 10 concepts should mostly have full coverage
        assert full_coverage_count >= 7, \
            f"Only {full_coverage_count}/10 top concepts have 5/5 coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
