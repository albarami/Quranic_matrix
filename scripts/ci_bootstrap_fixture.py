#!/usr/bin/env python3
"""
CI Bootstrap: Build tafsir indexes from committed fixture data.

This script is run in CI before tests to ensure deterministic indexes exist.
It uses the minimal fixture data committed to the repo, not large corpora.

Usage:
    python scripts/ci_bootstrap_fixture.py --fixture data/test_fixtures/fixture_v1 --out data/indexes/tafsir
"""
import argparse
import json
import os
import sys
from pathlib import Path

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
    """Build simple tafsir index from chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by source
    by_source = {}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(chunk)
    
    # Write per-source indexes
    for source, source_chunks in by_source.items():
        index_file = output_dir / f"{source}_index.jsonl"
        with open(index_file, 'w', encoding='utf-8') as f:
            for chunk in source_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        print(f"  Wrote {len(source_chunks)} chunks to {index_file.name}")
    
    # Write combined index
    combined_file = output_dir / "combined_index.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(chunks)} chunks to combined_index.jsonl")
    
    # Write manifest
    manifest = {
        "type": "ci_fixture_index",
        "sources": list(by_source.keys()),
        "total_chunks": len(chunks),
        "chunks_by_source": {s: len(c) for s, c in by_source.items()}
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
