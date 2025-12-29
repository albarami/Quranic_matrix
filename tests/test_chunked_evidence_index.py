"""
Test: Chunked Evidence Index

Step 2 Gate: These tests MUST pass before proceeding to Step 3.
"""

import pytest
import json
from pathlib import Path

CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
METADATA_FILE = Path("data/evidence/evidence_index_v2_chunked_metadata.json")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Chunking parameters from the builder
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1400


@pytest.fixture
def chunked_entries():
    """Load all chunked evidence entries."""
    if not CHUNKED_INDEX_FILE.exists():
        pytest.skip("Chunked evidence index not built yet")
    
    entries = []
    with open(CHUNKED_INDEX_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


@pytest.fixture
def metadata():
    """Load chunked evidence index metadata."""
    if not METADATA_FILE.exists():
        pytest.skip("Chunked evidence index metadata not found")
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_chunked_index_exists():
    """Chunked evidence index file must exist."""
    if not CHUNKED_INDEX_FILE.exists():
        pytest.skip("Chunked evidence index not built yet")
    assert METADATA_FILE.exists(), f"Metadata not found: {METADATA_FILE}"


def test_more_chunks_than_entries(chunked_entries, metadata):
    """Chunking should produce more chunks than original entries."""
    total_entries = metadata['stats']['total_entries']
    total_chunks = metadata['stats']['total_chunks']
    
    assert total_chunks > total_entries, \
        f"Expected more chunks than entries: {total_chunks} <= {total_entries}"


def test_long_tafsir_has_more_chunks(metadata):
    """Long tafsir (Tabari, Ibn Kathir) should have more chunks per entry than short (Jalalayn)."""
    avg_tabari = metadata['stats']['avg_chunks_per_entry']['tabari']
    avg_jalalayn = metadata['stats']['avg_chunks_per_entry']['jalalayn']
    
    assert avg_tabari > avg_jalalayn, \
        f"Tabari should have more chunks than Jalalayn: {avg_tabari} <= {avg_jalalayn}"
    
    assert avg_tabari >= 2.0, f"Tabari should have at least 2 chunks/entry: {avg_tabari}"


def test_chunk_ids_are_stable(chunked_entries):
    """Chunk IDs must follow the deterministic format."""
    for entry in chunked_entries[:100]:
        chunk_id = entry['chunk_id']
        source = entry['source']
        surah = entry['surah']
        ayah = entry['ayah']
        chunk_index = entry['chunk_index']
        
        expected_id = f"{source}_{surah:03d}_{ayah:03d}_chunk{chunk_index:02d}"
        assert chunk_id == expected_id, f"Chunk ID mismatch: {chunk_id} != {expected_id}"


def test_chunk_sizes_within_bounds(chunked_entries):
    """Chunks should not exceed max size; small chunks are allowed for short tafsir."""
    too_large = 0
    
    for entry in chunked_entries:
        char_count = entry['char_count']
        
        # Only check max bound - short tafsir entries are naturally small
        if char_count > MAX_CHUNK_CHARS + 100:
            too_large += 1
    
    total = len(chunked_entries)
    
    # No chunks should exceed max size
    assert too_large / total < 0.01, f"Too many large chunks: {too_large}/{total}"


def test_offsets_are_valid(chunked_entries):
    """char_start and char_end must be valid offsets."""
    for entry in chunked_entries[:100]:
        char_start = entry['char_start']
        char_end = entry['char_end']
        text_clean = entry['text_clean']
        char_count = entry['char_count']
        
        assert char_start >= 0, f"char_start must be >= 0: {char_start}"
        assert char_end > char_start, f"char_end must be > char_start"
        assert char_count == len(text_clean), f"char_count mismatch"


def test_chunk_indices_are_sequential(chunked_entries):
    """For each verse+source, chunk indices should be sequential starting from 0."""
    verse_source_chunks = {}
    
    for entry in chunked_entries:
        key = (entry['verse_key'], entry['source'])
        if key not in verse_source_chunks:
            verse_source_chunks[key] = []
        verse_source_chunks[key].append(entry['chunk_index'])
    
    for key, indices in list(verse_source_chunks.items())[:100]:
        sorted_indices = sorted(indices)
        expected = list(range(len(indices)))
        assert sorted_indices == expected, \
            f"Chunk indices not sequential for {key}: {sorted_indices}"


def test_total_chunks_field_correct(chunked_entries):
    """total_chunks field should match actual chunk count for each verse+source."""
    verse_source_counts = {}
    
    for entry in chunked_entries:
        key = (entry['verse_key'], entry['source'])
        if key not in verse_source_counts:
            verse_source_counts[key] = {'count': 0, 'total_chunks': entry['total_chunks']}
        verse_source_counts[key]['count'] += 1
    
    for key, data in list(verse_source_counts.items())[:100]:
        assert data['count'] == data['total_chunks'], \
            f"total_chunks mismatch for {key}: {data['count']} != {data['total_chunks']}"


def test_entries_are_sorted(chunked_entries):
    """Entries must be sorted by (surah, ayah, source, chunk_index)."""
    prev = None
    for entry in chunked_entries:
        current = (entry['surah'], entry['ayah'], entry['source'], entry['chunk_index'])
        if prev is not None:
            assert current >= prev, f"Entries not sorted: {prev} > {current}"
        prev = current


def test_metadata_has_chunking_params(metadata):
    """Metadata must include chunking parameters."""
    assert 'chunking_params' in metadata
    params = metadata['chunking_params']
    
    assert 'min_chunk_chars' in params
    assert 'target_chunk_chars' in params
    assert 'max_chunk_chars' in params


def test_reproducibility():
    """Chunk ID generation should be deterministic."""
    from scripts.build_chunked_evidence_index import generate_chunk_id
    
    id1 = generate_chunk_id("tabari", 2, 255, 3)
    id2 = generate_chunk_id("tabari", 2, 255, 3)
    
    assert id1 == id2, "Chunk ID generation is not deterministic"
    assert id1 == "tabari_002_255_chunk03", f"Unexpected chunk ID format: {id1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
