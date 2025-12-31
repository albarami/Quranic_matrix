#!/usr/bin/env python3
"""
CI Bootstrap: Build tafsir indexes from committed fixture data.

This script is run in CI before tests to ensure deterministic indexes exist.
It uses the minimal fixture data committed to the repo, not large corpora.

CRITICAL: Output format must match what StratifiedTafsirRetriever expects:
- Files: data/indexes/tafsir/{source}.json (NOT *_index.jsonl)
- Shape: {"documents": [{"surah": N, "ayah": N, "text": "...", "source": "..."}, ...]}
- All 7 sources must have >= 1 document

Usage:
    python scripts/ci_bootstrap_fixture.py --fixture data/test_fixtures/fixture_v1 --out data/indexes/tafsir
"""
import argparse
import json
import os
import sys
from pathlib import Path

# All 7 tafsir sources required by StratifiedTafsirRetriever
REQUIRED_SOURCES = [
    "ibn_kathir", "tabari", "qurtubi", "saadi", 
    "jalalayn", "baghawi", "muyassar"
]

def load_fixture_verses(fixture_dir: Path) -> dict:
    """Load verses from fixture JSONL."""
    verses = {}
    verses_file = fixture_dir / "quran_verses.jsonl"
    if verses_file.exists():
        with open(verses_file, 'r', encoding='utf-8') as f:
            for line in f:
                v = json.loads(line)
                verses[v["key"]] = v
    return verses

def load_fixture_tafsir(fixture_dir: Path) -> list:
    """Load tafsir chunks from fixture JSONL."""
    chunks = []
    tafsir_file = fixture_dir / "tafsir_chunks.jsonl"
    if tafsir_file.exists():
        with open(tafsir_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks

def build_tafsir_index(chunks: list, output_dir: Path):
    """
    Build tafsir indexes in the EXACT format StratifiedTafsirRetriever expects.
    
    Output: {source}.json with {"documents": [...]} structure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by source
    by_source = {source: [] for source in REQUIRED_SOURCES}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        if source in by_source:
            by_source[source].append({
                "surah": chunk.get("surah"),
                "ayah": chunk.get("ayah"),
                "text": chunk.get("text", ""),
                "source": source,
                "verse_key": chunk.get("verse_key", f"{chunk.get('surah')}:{chunk.get('ayah')}")
            })
    
    # Write per-source indexes in EXACT format expected by StratifiedTafsirRetriever
    for source in REQUIRED_SOURCES:
        documents = by_source[source]
        if not documents:
            print(f"  WARNING: No documents for {source}, creating placeholder")
            # Create at least one placeholder document to satisfy _validate_indexes()
            documents = [{
                "surah": 1,
                "ayah": 1,
                "text": f"[CI fixture placeholder for {source}]",
                "source": source,
                "verse_key": "1:1"
            }]
        
        # Write as {source}.json with {"documents": [...]} structure
        index_file = output_dir / f"{source}.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({"documents": documents}, f, ensure_ascii=False)
        print(f"  Wrote {len(documents)} documents to {index_file.name}")
    
    # Write manifest
    manifest = {
        "type": "ci_fixture_index",
        "sources": REQUIRED_SOURCES,
        "total_chunks": sum(len(by_source[s]) for s in REQUIRED_SOURCES),
        "chunks_by_source": {s: len(by_source[s]) for s in REQUIRED_SOURCES}
    }
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def build_verse_index(verses: dict, output_dir: Path):
    """Build verse lookup index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write verse index
    verse_index_file = output_dir / "verse_index.json"
    with open(verse_index_file, 'w', encoding='utf-8') as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(verses)} verses to verse_index.json")

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI fixture indexes")
    parser.add_argument("--fixture", required=True, help="Path to fixture directory")
    parser.add_argument("--out", required=True, help="Output directory for indexes")
    args = parser.parse_args()
    
    fixture_dir = Path(args.fixture)
    output_dir = Path(args.out)
    
    if not fixture_dir.exists():
        print(f"ERROR: Fixture directory not found: {fixture_dir}")
        sys.exit(1)
    
    print(f"CI Bootstrap: Building indexes from {fixture_dir}")
    
    # Load fixture data
    verses = load_fixture_verses(fixture_dir)
    print(f"Loaded {len(verses)} verses from fixture")
    
    chunks = load_fixture_tafsir(fixture_dir)
    print(f"Loaded {len(chunks)} tafsir chunks from fixture")
    
    if not verses and not chunks:
        print("ERROR: No fixture data found")
        sys.exit(1)
    
    # Build indexes
    print("\nBuilding tafsir indexes...")
    build_tafsir_index(chunks, output_dir)
    
    print("\nBuilding verse index...")
    build_verse_index(verses, output_dir)
    
    # Set environment marker
    env_file = output_dir / ".ci_fixture_mode"
    env_file.write_text("1")
    
    print(f"\n✓ CI fixture indexes built successfully in {output_dir}")
    print("  Set QBM_USE_FIXTURE=1 to use these indexes")

if __name__ == "__main__":
    main()
