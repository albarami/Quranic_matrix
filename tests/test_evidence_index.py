"""
Test: Deterministic Evidence Index

Step 1 Gate: These tests MUST pass before proceeding to Step 2.
"""

import pytest
import json
from pathlib import Path

EVIDENCE_INDEX_FILE = Path("data/evidence/evidence_index_v1.jsonl")
METADATA_FILE = Path("data/evidence/evidence_index_v1_metadata.json")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@pytest.fixture
def evidence_entries():
    """Load all evidence index entries."""
    if not EVIDENCE_INDEX_FILE.exists():
        pytest.skip("Evidence index not built yet")
    
    entries = []
    with open(EVIDENCE_INDEX_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


@pytest.fixture
def metadata():
    """Load evidence index metadata."""
    if not METADATA_FILE.exists():
        pytest.skip("Evidence index metadata not found")
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_evidence_index_exists():
    """Evidence index file must exist."""
    assert EVIDENCE_INDEX_FILE.exists(), f"Evidence index not found: {EVIDENCE_INDEX_FILE}"
    assert METADATA_FILE.exists(), f"Metadata not found: {METADATA_FILE}"


def test_all_core_sources_present(evidence_entries):
    """All 5 core tafsir sources must be present in the index."""
    sources_found = set()
    for entry in evidence_entries:
        sources_found.add(entry['source'])
    
    for source in CORE_SOURCES:
        assert source in sources_found, f"Core source missing: {source}"


def test_minimum_entries_per_source(evidence_entries):
    """Each core source must have at least 6000 entries (covering most verses)."""
    source_counts = {}
    for entry in evidence_entries:
        source = entry['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source in CORE_SOURCES:
        count = source_counts.get(source, 0)
        assert count >= 6000, f"Source {source} has only {count} entries, expected >= 6000"


def test_required_fields_present(evidence_entries):
    """Each entry must have all required fields."""
    required_fields = [
        'verse_key', 'surah', 'ayah', 'source', 'chunk_id',
        'char_start', 'char_end', 'text_clean',
        'build_version', 'extractor_version', 'created_at'
    ]
    
    for i, entry in enumerate(evidence_entries[:100]):  # Check first 100
        for field in required_fields:
            assert field in entry, f"Entry {i} missing field: {field}"


def test_chunk_ids_are_stable(evidence_entries):
    """Chunk IDs must follow the deterministic format."""
    for entry in evidence_entries[:100]:
        chunk_id = entry['chunk_id']
        source = entry['source']
        surah = entry['surah']
        ayah = entry['ayah']
        chunk_index = entry.get('chunk_index', 0)
        
        expected_id = f"{source}_{surah:03d}_{ayah:03d}_chunk{chunk_index:02d}"
        assert chunk_id == expected_id, f"Chunk ID mismatch: {chunk_id} != {expected_id}"


def test_offsets_are_valid(evidence_entries):
    """char_start and char_end must be valid offsets."""
    for entry in evidence_entries[:100]:
        char_start = entry['char_start']
        char_end = entry['char_end']
        text_clean = entry['text_clean']
        
        assert char_start >= 0, f"char_start must be >= 0: {char_start}"
        assert char_end > char_start, f"char_end must be > char_start: {char_end} <= {char_start}"
        assert char_end == len(text_clean), f"char_end must equal text length: {char_end} != {len(text_clean)}"


def test_verse_key_format(evidence_entries):
    """verse_key must be in format 'surah:ayah'."""
    for entry in evidence_entries[:100]:
        verse_key = entry['verse_key']
        surah = entry['surah']
        ayah = entry['ayah']
        
        expected_key = f"{surah}:{ayah}"
        assert verse_key == expected_key, f"verse_key mismatch: {verse_key} != {expected_key}"


def test_text_clean_not_empty(evidence_entries):
    """text_clean must not be empty."""
    for entry in evidence_entries[:100]:
        text_clean = entry['text_clean']
        assert text_clean and len(text_clean) >= 10, f"text_clean too short: {len(text_clean)}"


def test_entries_are_sorted(evidence_entries):
    """Entries must be sorted by (surah, ayah, source, chunk_index)."""
    prev = None
    for entry in evidence_entries:
        current = (entry['surah'], entry['ayah'], entry['source'], entry.get('chunk_index', 0))
        if prev is not None:
            assert current >= prev, f"Entries not sorted: {prev} > {current}"
        prev = current


def test_metadata_has_required_fields(metadata):
    """Metadata must have required fields."""
    required = ['build_version', 'extractor_version', 'created_at', 'stats', 'core_sources']
    for field in required:
        assert field in metadata, f"Metadata missing field: {field}"


def test_verses_with_all_core_sources(evidence_entries):
    """Most verses should have all 5 core sources."""
    verse_sources = {}
    for entry in evidence_entries:
        vk = entry['verse_key']
        if vk not in verse_sources:
            verse_sources[vk] = set()
        verse_sources[vk].add(entry['source'])
    
    complete_count = 0
    for vk, sources in verse_sources.items():
        if all(s in sources for s in CORE_SOURCES):
            complete_count += 1
    
    total_verses = len(verse_sources)
    coverage = complete_count / total_verses if total_verses > 0 else 0
    
    assert coverage >= 0.99, f"Only {coverage:.1%} of verses have all 5 core sources"


def test_reproducibility():
    """Running the builder again should produce the same chunk IDs."""
    from scripts.build_evidence_index import generate_chunk_id
    
    # Test deterministic chunk ID generation
    id1 = generate_chunk_id("ibn_kathir", 2, 255, 0)
    id2 = generate_chunk_id("ibn_kathir", 2, 255, 0)
    
    assert id1 == id2, "Chunk ID generation is not deterministic"
    assert id1 == "ibn_kathir_002_255_chunk00", f"Unexpected chunk ID format: {id1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
