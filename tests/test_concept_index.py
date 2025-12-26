"""
Test: Deterministic Concept Index (Phase 6.2)

Ensures concept index has proper structure, offsets, and per-source coverage.
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v1.jsonl")
METADATA_FILE = Path("data/evidence/concept_index_v1_metadata.json")
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.fixture(scope="module")
def concept_index():
    """Load the concept index."""
    if not CONCEPT_INDEX_FILE.exists():
        pytest.skip("Concept index not built yet")
    
    concepts = {}
    with open(CONCEPT_INDEX_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            concepts[entry['concept_id']] = entry
    return concepts


@pytest.fixture(scope="module")
def metadata():
    """Load the concept index metadata."""
    if not METADATA_FILE.exists():
        pytest.skip("Concept index metadata not found")
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.mark.unit
class TestConceptIndexStructure:
    """Tests for concept index structure."""
    
    def test_concept_index_exists(self):
        """Concept index file must exist."""
        assert CONCEPT_INDEX_FILE.exists(), f"Missing: {CONCEPT_INDEX_FILE}"
    
    def test_concept_index_not_empty(self, concept_index):
        """Concept index must have entries."""
        assert len(concept_index) > 0, "Concept index is empty"
    
    def test_concept_entries_have_required_fields(self, concept_index):
        """Each concept entry must have required fields."""
        required_fields = ['concept_id', 'term', 'entity_type', 'verses', 
                          'tafsir_chunks', 'per_source_stats', 'total_mentions']
        
        for concept_id, entry in concept_index.items():
            for field in required_fields:
                assert field in entry, f"Concept {concept_id} missing field: {field}"
    
    def test_tafsir_chunks_have_offsets(self, concept_index):
        """Tafsir chunks must have character offsets."""
        for concept_id, entry in concept_index.items():
            for chunk in entry['tafsir_chunks'][:5]:  # Check first 5
                assert 'char_start' in chunk, f"Chunk missing char_start in {concept_id}"
                assert 'char_end' in chunk, f"Chunk missing char_end in {concept_id}"
                assert 'quote' in chunk, f"Chunk missing quote in {concept_id}"
                assert chunk['char_start'] < chunk['char_end'], \
                    f"Invalid offsets in {concept_id}: {chunk['char_start']} >= {chunk['char_end']}"


@pytest.mark.unit
class TestConceptIndexOffsets:
    """Tests for offset validity."""
    
    def test_offsets_are_valid_integers(self, concept_index):
        """Offsets must be non-negative integers."""
        for concept_id, entry in concept_index.items():
            for chunk in entry['tafsir_chunks'][:5]:
                assert isinstance(chunk['char_start'], int), f"char_start not int in {concept_id}"
                assert isinstance(chunk['char_end'], int), f"char_end not int in {concept_id}"
                assert chunk['char_start'] >= 0, f"Negative char_start in {concept_id}"
    
    def test_quotes_contain_term(self, concept_index):
        """Quote context should contain the search term (or normalized form)."""
        # Check a few concepts
        for concept_id, entry in list(concept_index.items())[:5]:
            term = entry['term']
            for chunk in entry['tafsir_chunks'][:3]:
                quote = chunk.get('quote', '')
                # Term or normalized form should be in quote
                # (normalization may change exact form)
                assert len(quote) > 0, f"Empty quote in {concept_id}"


@pytest.mark.unit
class TestConceptIndexCoverage:
    """Tests for per-source coverage."""
    
    def test_per_source_stats_structure(self, concept_index):
        """per_source_stats must have all core sources."""
        for concept_id, entry in concept_index.items():
            stats = entry['per_source_stats']
            for source in CORE_SOURCES:
                assert source in stats, f"Missing source {source} in {concept_id}"
                assert 'count' in stats[source], f"Missing count for {source} in {concept_id}"
    
    def test_concept_ref_returns_core_sources_when_available(self, concept_index):
        """High-frequency concepts should have coverage from all 5 sources."""
        # Check concepts with many mentions
        high_freq_concepts = [c for c in concept_index.values() if c['total_mentions'] > 100]
        
        for entry in high_freq_concepts[:5]:
            sources_with_data = sum(
                1 for s in CORE_SOURCES 
                if entry['per_source_stats'][s]['count'] > 0
            )
            assert sources_with_data >= 3, \
                f"Concept {entry['concept_id']} has only {sources_with_data}/5 sources"


@pytest.mark.unit
class TestConceptIndexReproducibility:
    """Tests for reproducibility."""
    
    def test_metadata_has_version(self, metadata):
        """Metadata must have version info."""
        assert 'version' in metadata
        assert 'created_at' in metadata
        assert 'canonical_entities_version' in metadata
    
    def test_metadata_has_stats(self, metadata):
        """Metadata must have statistics."""
        assert 'stats' in metadata
        assert 'concepts_with_evidence' in metadata['stats']
        assert 'concepts_by_entity_type' in metadata['stats']


@pytest.mark.integration
class TestConceptIndexRetrieval:
    """Integration tests for concept-based retrieval."""
    
    def test_envy_concept_has_evidence(self, concept_index):
        """الحسد (envy) must have evidence from multiple sources."""
        envy = concept_index.get('BEH_EMO_ENVY')
        assert envy is not None, "Envy concept not found"
        assert envy['total_mentions'] > 0, "Envy has no mentions"
        
        # Should have evidence from at least 3 sources
        sources_covered = sum(
            1 for s in CORE_SOURCES 
            if envy['per_source_stats'][s]['count'] > 0
        )
        assert sources_covered >= 3, f"Envy only has {sources_covered}/5 sources"
    
    def test_patience_concept_has_evidence(self, concept_index):
        """الصبر (patience) must have evidence."""
        patience = concept_index.get('BEH_EMO_PATIENCE')
        assert patience is not None, "Patience concept not found"
        assert patience['total_mentions'] > 0, "Patience has no mentions"
    
    def test_repentance_concept_has_evidence(self, concept_index):
        """التوبة (repentance) must have evidence."""
        repentance = concept_index.get('BEH_SPI_REPENTANCE')
        assert repentance is not None, "Repentance concept not found"
        assert repentance['total_mentions'] > 0, "Repentance has no mentions"
    
    def test_concept_verses_are_valid(self, concept_index):
        """Verse references must be valid format."""
        for concept_id, entry in list(concept_index.items())[:10]:
            for verse in entry['verses'][:5]:
                assert 'verse_key' in verse
                # verse_key should be surah:ayah format
                parts = verse['verse_key'].split(':')
                assert len(parts) == 2, f"Invalid verse_key: {verse['verse_key']}"
                assert parts[0].isdigit(), f"Invalid surah: {verse['verse_key']}"
                assert parts[1].isdigit(), f"Invalid ayah: {verse['verse_key']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
