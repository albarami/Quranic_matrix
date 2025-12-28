"""
Build Deterministic Test Fixture Index (Enterprise Testing Infrastructure)

Creates a small, deterministic fixture dataset for integration tests:
- 50 verses from selected surahs
- Tafsir from all 5 core sources for those verses
- Chunked evidence index
- Vector index (CPU-safe, deterministic)

Output: data/test_fixtures/fixture_v1/
"""

import json
import os
import sys
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any

# Ensure reproducibility
SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
FIXTURE_DIR = DATA_DIR / "test_fixtures" / "fixture_v1"
TAFSIR_DIR = DATA_DIR / "tafsir"

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Selected verses for fixture (deterministic selection)
FIXTURE_VERSES = [
    # Al-Fatiha (complete)
    "1:1", "1:2", "1:3", "1:4", "1:5", "1:6", "1:7",
    # Al-Baqarah (selected)
    "2:1", "2:2", "2:3", "2:255", "2:256", "2:285", "2:286",
    # Al-Imran (selected)
    "3:1", "3:2", "3:18", "3:19",
    # An-Nisa (selected)
    "4:1", "4:36",
    # Al-Maidah (selected)
    "5:1", "5:3",
    # Al-Ikhlas (complete)
    "112:1", "112:2", "112:3", "112:4",
    # Al-Falaq (complete)
    "113:1", "113:2", "113:3", "113:4", "113:5",
    # An-Nas (complete)
    "114:1", "114:2", "114:3", "114:4", "114:5", "114:6",
    # Additional verses for behavior coverage
    "2:45", "2:153", "2:177", "2:183",  # Patience, prayer, righteousness, fasting
    "3:134", "3:159",  # Forgiveness, consultation
    "4:135",  # Justice
    "49:12",  # Backbiting
    "31:18", "31:19",  # Arrogance, humility
]


def load_quran_verses() -> Dict[str, str]:
    """Load Quran verses from XML."""
    import xml.etree.ElementTree as ET
    
    quran_path = DATA_DIR / "quran" / "quran-uthmani.xml"
    verses = {}
    
    tree = ET.parse(quran_path)
    root = tree.getroot()
    
    for sura in root.findall('sura'):
        sura_idx = int(sura.get('index'))
        for aya in sura.findall('aya'):
            aya_idx = int(aya.get('index'))
            verse_key = f"{sura_idx}:{aya_idx}"
            if verse_key in FIXTURE_VERSES:
                verses[verse_key] = aya.get('text', '')
    
    return verses


def load_tafsir_for_verses(verse_keys: List[str]) -> Dict[str, Dict[str, str]]:
    """Load tafsir for specified verses from all core sources."""
    tafsir_data = {source: {} for source in CORE_SOURCES}
    
    for source in CORE_SOURCES:
        filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue
        
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    ref = entry.get("reference", {})
                    key = f"{ref.get('surah')}:{ref.get('ayah')}"
                    if key in verse_keys:
                        tafsir_data[source][key] = entry.get("text_ar", "")
    
    return tafsir_data


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def generate_chunk_id(source: str, verse_key: str, chunk_idx: int) -> str:
    """Generate deterministic chunk ID."""
    content = f"{source}:{verse_key}:{chunk_idx}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def build_fixture_evidence_index(
    verses: Dict[str, str],
    tafsir: Dict[str, Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Build chunked evidence index for fixture."""
    chunks = []
    
    for verse_key, verse_text in verses.items():
        parts = verse_key.split(":")
        surah = int(parts[0])
        ayah = int(parts[1])
        
        for source in CORE_SOURCES:
            tafsir_text = tafsir.get(source, {}).get(verse_key, "")
            if not tafsir_text:
                continue
            
            text_chunks = chunk_text(tafsir_text)
            for idx, chunk_text_content in enumerate(text_chunks):
                chunk_id = generate_chunk_id(source, verse_key, idx)
                chunks.append({
                    'chunk_id': chunk_id,
                    'verse_key': verse_key,
                    'source': source,
                    'surah': surah,
                    'ayah': ayah,
                    'chunk_index': idx,
                    'total_chunks': len(text_chunks),
                    'text_clean': chunk_text_content,
                    'char_count': len(chunk_text_content),
                })
    
    return chunks


def build_fixture_vector_index(chunks: List[Dict[str, Any]]) -> tuple:
    """Build vector index for fixture (CPU-safe, deterministic)."""
    import numpy as np
    
    # Use deterministic pseudo-embeddings for testing
    # In production, we'd use real embeddings, but for fixture tests
    # we need deterministic, fast, CPU-safe embeddings
    
    np.random.seed(SEED)
    
    embedding_dim = 768
    embeddings = []
    metadata = []
    
    for chunk in chunks:
        # Generate deterministic embedding based on chunk content
        # This is NOT for semantic search quality - it's for testing infrastructure
        content_hash = hashlib.md5(chunk['text_clean'].encode()).digest()
        seed_from_hash = int.from_bytes(content_hash[:4], 'little')
        np.random.seed(seed_from_hash)
        
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        embeddings.append(embedding)
        metadata.append({
            'type': 'tafsir',
            'source': chunk['source'],
            'verse': chunk['verse_key'],
            'surah': chunk['surah'],
            'ayah': chunk['ayah'],
            'text': chunk['text_clean'][:300],
            'chunk_id': chunk['chunk_id'],
        })
    
    # Add Quran verses to index
    for verse_key, verse_text in load_quran_verses().items():
        parts = verse_key.split(":")
        surah = int(parts[0])
        ayah = int(parts[1])
        
        content_hash = hashlib.md5(verse_text.encode()).digest()
        seed_from_hash = int.from_bytes(content_hash[:4], 'little')
        np.random.seed(seed_from_hash)
        
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        embeddings.append(embedding)
        metadata.append({
            'type': 'quran',
            'source': 'quran',
            'verse': verse_key,
            'surah': surah,
            'ayah': ayah,
            'text': verse_text,
        })
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    return embeddings_array, metadata


def main():
    """Build the complete test fixture."""
    print("=" * 60)
    print("Building Deterministic Test Fixture")
    print("=" * 60)
    
    # Create fixture directory
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Quran verses
    print(f"\n1. Loading {len(FIXTURE_VERSES)} Quran verses...")
    verses = load_quran_verses()
    print(f"   Loaded {len(verses)} verses")
    
    # 2. Load tafsir
    print("\n2. Loading tafsir for fixture verses...")
    tafsir = load_tafsir_for_verses(FIXTURE_VERSES)
    for source in CORE_SOURCES:
        print(f"   {source}: {len(tafsir[source])} entries")
    
    # 3. Build chunked evidence index
    print("\n3. Building chunked evidence index...")
    chunks = build_fixture_evidence_index(verses, tafsir)
    print(f"   Created {len(chunks)} chunks")
    
    # Save evidence index
    evidence_path = FIXTURE_DIR / "fixture_evidence_index.jsonl"
    with open(evidence_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"   Saved to {evidence_path}")
    
    # 4. Build vector index
    print("\n4. Building vector index (deterministic, CPU-safe)...")
    embeddings, metadata = build_fixture_vector_index(chunks)
    print(f"   Created {len(embeddings)} embeddings")
    
    # Save vector index
    import numpy as np
    index_path = FIXTURE_DIR / "fixture_index.npy"
    np.save(str(index_path), embeddings)
    print(f"   Saved embeddings to {index_path}")
    
    # Save metadata
    metadata_path = FIXTURE_DIR / "fixture_index.npy.metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"   Saved metadata to {metadata_path}")
    
    # 5. Save fixture manifest
    manifest = {
        'version': '1.0',
        'seed': SEED,
        'verse_count': len(verses),
        'chunk_count': len(chunks),
        'embedding_count': len(embeddings),
        'embedding_dim': 768,
        'sources': CORE_SOURCES,
        'fixture_verses': FIXTURE_VERSES,
    }
    manifest_path = FIXTURE_DIR / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n5. Saved manifest to {manifest_path}")
    
    print("\n" + "=" * 60)
    print("Fixture build complete!")
    print(f"  Verses: {len(verses)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Location: {FIXTURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
