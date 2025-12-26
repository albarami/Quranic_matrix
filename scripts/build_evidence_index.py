"""
Build Deterministic Evidence Index

Step 1: Create a versioned, deterministic evidence index mapping verses to tafsir chunks.

Output: data/evidence/evidence_index_v1.jsonl

Each record contains:
- verse_key (e.g., "5:38")
- source (tafsir name)
- chunk_id (stable, deterministic)
- char_start, char_end
- text_clean
- build_version, extractor_version, created_at
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Version info
BUILD_VERSION = "1.0.0"
EXTRACTOR_VERSION = "1.0.0"

# Core tafsir sources (5 required + 2 optional)
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
ALL_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]

# Paths
TAFSIR_DIR = Path("data/tafsir")
OUTPUT_DIR = Path("data/evidence")
OUTPUT_FILE = OUTPUT_DIR / "evidence_index_v1.jsonl"


def clean_text(text: str) -> str:
    """Normalize Arabic text for consistent indexing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize some Arabic characters (optional, for consistency)
    # Keep diacritics for now as they may be important for tafsir
    
    return text


def generate_chunk_id(source: str, surah: int, ayah: int, chunk_index: int) -> str:
    """Generate a stable, deterministic chunk ID."""
    # Format: source_surah_ayah_chunkN
    # This is reproducible given the same inputs
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


def build_evidence_index() -> Dict[str, Any]:
    """Build the complete evidence index."""
    
    print("=" * 60)
    print("Building Deterministic Evidence Index v1")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    created_at = datetime.now(timezone.utc).isoformat()
    
    stats = {
        'total_entries': 0,
        'entries_per_source': {},
        'verses_covered': set(),
        'core_sources_complete': 0,
    }
    
    all_entries = []
    
    for source in ALL_SOURCES:
        print(f"\nProcessing {source}...")
        
        tafsir_entries = load_tafsir_source(source)
        print(f"  Loaded {len(tafsir_entries)} entries")
        
        source_count = 0
        
        for entry in tafsir_entries:
            surah = entry['surah']
            ayah = entry['ayah']
            text = entry['text']
            
            # Clean text
            text_clean = clean_text(text)
            
            if not text_clean or len(text_clean) < 10:
                continue
            
            # For now, treat entire tafsir entry as one chunk (Step 2 will add chunking)
            chunk_index = 0
            chunk_id = generate_chunk_id(source, surah, ayah, chunk_index)
            content_hash = generate_content_hash(text_clean)
            
            verse_key = f"{surah}:{ayah}"
            
            evidence_entry = {
                'verse_key': verse_key,
                'surah': surah,
                'ayah': ayah,
                'source': source,
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'char_start': 0,
                'char_end': len(text_clean),
                'text_clean': text_clean,
                'content_hash': content_hash,
                'build_version': BUILD_VERSION,
                'extractor_version': EXTRACTOR_VERSION,
                'created_at': created_at,
            }
            
            all_entries.append(evidence_entry)
            source_count += 1
            stats['verses_covered'].add(verse_key)
        
        stats['entries_per_source'][source] = source_count
        stats['total_entries'] += source_count
        print(f"  Indexed {source_count} entries")
    
    # Sort entries for deterministic output
    all_entries.sort(key=lambda x: (x['surah'], x['ayah'], x['source'], x['chunk_index']))
    
    # Write output
    print(f"\nWriting {len(all_entries)} entries to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Calculate core sources coverage
    verses_with_all_core = 0
    verse_source_map = {}
    for entry in all_entries:
        vk = entry['verse_key']
        if vk not in verse_source_map:
            verse_source_map[vk] = set()
        verse_source_map[vk].add(entry['source'])
    
    for vk, sources in verse_source_map.items():
        if all(s in sources for s in CORE_SOURCES):
            verses_with_all_core += 1
    
    stats['verses_covered'] = len(stats['verses_covered'])
    stats['verses_with_all_core_sources'] = verses_with_all_core
    
    # Summary
    print("\n" + "=" * 60)
    print("Evidence Index Build Complete")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Verses covered: {stats['verses_covered']}")
    print(f"Verses with all 5 core sources: {verses_with_all_core}")
    print(f"\nEntries per source:")
    for source, count in stats['entries_per_source'].items():
        marker = "✓" if source in CORE_SOURCES else " "
        print(f"  {marker} {source}: {count}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Build version: {BUILD_VERSION}")
    print(f"Created at: {created_at}")
    
    # Write metadata
    metadata = {
        'build_version': BUILD_VERSION,
        'extractor_version': EXTRACTOR_VERSION,
        'created_at': created_at,
        'stats': {k: v for k, v in stats.items() if k != 'verses_covered'},
        'core_sources': CORE_SOURCES,
        'all_sources': ALL_SOURCES,
    }
    
    metadata_file = OUTPUT_DIR / "evidence_index_v1_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata: {metadata_file}")
    
    return stats


if __name__ == "__main__":
    build_evidence_index()
