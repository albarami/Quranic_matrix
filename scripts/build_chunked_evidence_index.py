"""
Build Chunked Evidence Index

Step 2: Deterministic chunking for tafsir entries.

Long tafsir passages (Ibn Kathir, Tabari, Qurtubi) are chunked into smaller
retrieval units to compete fairly with short tafsir (Jalalayn).

Chunking strategy:
- Target chunk size: 200-350 tokens (approx 800-1400 chars for Arabic)
- Prefer paragraph/sentence boundaries
- Stable chunk_id derivation
- Store offsets and cleaned text

CI Fixture Mode (QBM_USE_FIXTURE=1):
- Builds from data/test_fixtures/fixture_v1/tafsir_chunks.jsonl
- No dependency on large data/tafsir/*.ar.jsonl files

Output: data/evidence/evidence_index_v2_chunked.jsonl
"""

import json
import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

# Check for CI fixture mode
USE_FIXTURE = os.getenv("QBM_USE_FIXTURE", "0") == "1"
FIXTURE_DIR = Path("data/test_fixtures/fixture_v1")

# Version info
BUILD_VERSION = "2.0.0"
EXTRACTOR_VERSION = "2.0.0"
CHUNKER_VERSION = "1.0.0"

# Chunking parameters
MIN_CHUNK_CHARS = 200   # Minimum chunk size in characters
TARGET_CHUNK_CHARS = 800  # Target chunk size (~200 tokens for Arabic)
MAX_CHUNK_CHARS = 1400   # Maximum chunk size (~350 tokens for Arabic)

# Sentence/paragraph delimiters for Arabic
SENTENCE_DELIMITERS = re.compile(r'([.،؛:؟!]\s+)')
PARAGRAPH_DELIMITERS = re.compile(r'\n\s*\n')

# Core tafsir sources
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
ALL_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]

# Paths
TAFSIR_DIR = Path("data/tafsir")
OUTPUT_DIR = Path("data/evidence")
OUTPUT_FILE = OUTPUT_DIR / "evidence_index_v2_chunked.jsonl"


def clean_text(text: str) -> str:
    """Normalize Arabic text for consistent indexing."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_sentence_boundaries(text: str) -> List[int]:
    """Find sentence boundary positions in text."""
    boundaries = [0]
    for match in SENTENCE_DELIMITERS.finditer(text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    return sorted(set(boundaries))


def split_into_chunks(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks at sentence boundaries.
    
    Returns list of (char_start, char_end, chunk_text) tuples.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        # Short text - return as single chunk
        return [(0, len(text), text)]
    
    boundaries = find_sentence_boundaries(text)
    chunks = []
    
    chunk_start = 0
    current_end = 0
    
    for i, boundary in enumerate(boundaries[1:], 1):
        segment_len = boundary - chunk_start
        
        # If adding this segment would exceed max, finalize current chunk
        if segment_len > MAX_CHUNK_CHARS and current_end > chunk_start:
            chunk_text = text[chunk_start:current_end].strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append((chunk_start, current_end, chunk_text))
            chunk_start = current_end
        
        current_end = boundary
        
        # If we've reached target size and have a good boundary, finalize
        if segment_len >= TARGET_CHUNK_CHARS:
            chunk_text = text[chunk_start:current_end].strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append((chunk_start, current_end, chunk_text))
            chunk_start = current_end
    
    # Handle remaining text
    if chunk_start < len(text):
        remaining = text[chunk_start:].strip()
        if remaining:
            if chunks and len(remaining) < MIN_CHUNK_CHARS:
                # Merge with previous chunk if too short
                prev_start, prev_end, prev_text = chunks.pop()
                chunks.append((prev_start, len(text), prev_text + " " + remaining))
            else:
                chunks.append((chunk_start, len(text), remaining))
    
    return chunks if chunks else [(0, len(text), text)]


def generate_chunk_id(source: str, surah: int, ayah: int, chunk_index: int) -> str:
    """Generate a stable, deterministic chunk ID."""
    return f"{source}_{surah:03d}_{ayah:03d}_chunk{chunk_index:02d}"


def generate_content_hash(text: str) -> str:
    """Generate a hash of the content for integrity checking."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def load_tafsir_source(source: str) -> List[Dict[str, Any]]:
    """Load a tafsir source file."""
    filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
    
    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return []
    
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                ref = data.get('reference', {})
                if isinstance(ref, dict):
                    surah = ref.get('surah')
                    ayah = ref.get('ayah')
                    text = data.get('text_ar', '')
                    
                    if surah and ayah and text:
                        entries.append({
                            'surah': int(surah),
                            'ayah': int(ayah),
                            'text': text,
                            'line_num': line_num,
                        })
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error at line {line_num}: {e}")
    
    return entries


def build_from_fixture() -> Dict[str, Any]:
    """Build chunked evidence index from CI fixture (no large corpora needed)."""
    print("=" * 60)
    print("Building Chunked Evidence Index from CI Fixture")
    print("=" * 60)
    
    fixture_file = FIXTURE_DIR / "tafsir_chunks.jsonl"
    if not fixture_file.exists():
        print(f"ERROR: Fixture file not found: {fixture_file}")
        return {}
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    
    all_entries = []
    with open(fixture_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            # Convert fixture format to evidence index format
            entry = {
                'verse_key': chunk.get('verse_key', f"{chunk['surah']}:{chunk['ayah']}"),
                'surah': chunk['surah'],
                'ayah': chunk['ayah'],
                'source': chunk['source'],
                'chunk_id': chunk.get('chunk_id', f"{chunk['source']}_{chunk['surah']}_{chunk['ayah']}_0"),
                'chunk_index': 0,
                'total_chunks': 1,
                'char_start': chunk.get('char_start', 0),
                'char_end': chunk.get('char_end', len(chunk.get('text', ''))),
                'text_clean': chunk.get('text', ''),
                'char_count': len(chunk.get('text', '')),
                'content_hash': hashlib.md5(chunk.get('text', '').encode()).hexdigest()[:12],
                'build_version': BUILD_VERSION,
                'extractor_version': EXTRACTOR_VERSION,
                'chunker_version': CHUNKER_VERSION,
                'created_at': created_at,
            }
            all_entries.append(entry)
    
    # Sort for deterministic output
    all_entries.sort(key=lambda x: (x['surah'], x['ayah'], x['source'], x['chunk_index']))
    
    # Write output
    print(f"Writing {len(all_entries)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Built chunked evidence index from fixture ({len(all_entries)} entries)")
    return {'total_chunks': len(all_entries), 'source': 'fixture'}


def build_chunked_evidence_index() -> Dict[str, Any]:
    """Build the chunked evidence index."""
    
    # Use fixture mode if enabled (CI environment)
    if USE_FIXTURE:
        print("QBM_USE_FIXTURE=1 detected, building from fixture...")
        return build_from_fixture()
    
    print("=" * 60)
    print("Building Chunked Evidence Index v2")
    print("=" * 60)
    print(f"Chunk size: {MIN_CHUNK_CHARS}-{MAX_CHUNK_CHARS} chars (target: {TARGET_CHUNK_CHARS})")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    created_at = datetime.now(timezone.utc).isoformat()
    
    stats = {
        'total_entries': 0,
        'total_chunks': 0,
        'entries_per_source': {},
        'chunks_per_source': {},
        'avg_chunks_per_entry': {},
        'verses_covered': set(),
    }
    
    all_entries = []
    
    for source in ALL_SOURCES:
        print(f"\nProcessing {source}...")
        
        tafsir_entries = load_tafsir_source(source)
        print(f"  Loaded {len(tafsir_entries)} entries")
        
        source_entries = 0
        source_chunks = 0
        
        for entry in tafsir_entries:
            surah = entry['surah']
            ayah = entry['ayah']
            text = entry['text']
            
            # Clean text
            text_clean = clean_text(text)
            
            if not text_clean or len(text_clean) < 10:
                continue
            
            source_entries += 1
            verse_key = f"{surah}:{ayah}"
            stats['verses_covered'].add(verse_key)
            
            # Chunk the text
            chunks = split_into_chunks(text_clean)
            
            for chunk_index, (char_start, char_end, chunk_text) in enumerate(chunks):
                chunk_id = generate_chunk_id(source, surah, ayah, chunk_index)
                content_hash = generate_content_hash(chunk_text)
                
                evidence_entry = {
                    'verse_key': verse_key,
                    'surah': surah,
                    'ayah': ayah,
                    'source': source,
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'total_chunks': len(chunks),
                    'char_start': char_start,
                    'char_end': char_end,
                    'text_clean': chunk_text,
                    'char_count': len(chunk_text),
                    'content_hash': content_hash,
                    'build_version': BUILD_VERSION,
                    'extractor_version': EXTRACTOR_VERSION,
                    'chunker_version': CHUNKER_VERSION,
                    'created_at': created_at,
                }
                
                all_entries.append(evidence_entry)
                source_chunks += 1
        
        stats['entries_per_source'][source] = source_entries
        stats['chunks_per_source'][source] = source_chunks
        stats['avg_chunks_per_entry'][source] = round(source_chunks / max(source_entries, 1), 2)
        stats['total_entries'] += source_entries
        stats['total_chunks'] += source_chunks
        
        print(f"  Entries: {source_entries}, Chunks: {source_chunks}, Avg: {stats['avg_chunks_per_entry'][source]:.2f}")
    
    # Sort entries for deterministic output
    all_entries.sort(key=lambda x: (x['surah'], x['ayah'], x['source'], x['chunk_index']))
    
    # Write output
    print(f"\nWriting {len(all_entries)} chunks to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    stats['verses_covered'] = len(stats['verses_covered'])
    
    # Summary
    print("\n" + "=" * 60)
    print("Chunked Evidence Index Build Complete")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Verses covered: {stats['verses_covered']}")
    print(f"Overall avg chunks/entry: {stats['total_chunks'] / max(stats['total_entries'], 1):.2f}")
    print(f"\nChunks per source:")
    for source in ALL_SOURCES:
        marker = "✓" if source in CORE_SOURCES else " "
        chunks = stats['chunks_per_source'].get(source, 0)
        avg = stats['avg_chunks_per_entry'].get(source, 0)
        print(f"  {marker} {source}: {chunks} chunks (avg {avg:.2f}/entry)")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Build version: {BUILD_VERSION}")
    
    # Write metadata
    metadata = {
        'build_version': BUILD_VERSION,
        'extractor_version': EXTRACTOR_VERSION,
        'chunker_version': CHUNKER_VERSION,
        'created_at': created_at,
        'chunking_params': {
            'min_chunk_chars': MIN_CHUNK_CHARS,
            'target_chunk_chars': TARGET_CHUNK_CHARS,
            'max_chunk_chars': MAX_CHUNK_CHARS,
        },
        'stats': stats,
        'core_sources': CORE_SOURCES,
        'all_sources': ALL_SOURCES,
    }
    
    metadata_file = OUTPUT_DIR / "evidence_index_v2_chunked_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata: {metadata_file}")
    
    return stats


if __name__ == "__main__":
    build_chunked_evidence_index()
