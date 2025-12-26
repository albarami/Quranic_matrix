"""
Build Deterministic Concept Evidence Index (Phase 6.1)

Maps each canonical concept to:
- Verses where it is mentioned (direct Quran signals)
- Tafsir chunks where it is mentioned (with offsets)
- Per-source coverage stats
- Related concepts (for graph expansion)

Output: data/evidence/concept_index_v1.jsonl
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
ENTITY_TYPES_FILE = Path("vocab/entity_types.json")
BEHAVIOR_CONCEPTS_FILE = Path("vocab/behavior_concepts.json")
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
OUTPUT_FILE = Path("data/evidence/concept_index_v1.jsonl")
METADATA_FILE = Path("data/evidence/concept_index_v1_metadata.json")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


def load_entity_types() -> Dict[str, Any]:
    """Load canonical entity types."""
    with open(ENTITY_TYPES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_behavior_concepts() -> Dict[str, Any]:
    """Load behavior concepts vocabulary."""
    with open(BEHAVIOR_CONCEPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunked_index() -> List[Dict[str, Any]]:
    """Load the chunked tafsir evidence index."""
    chunks = []
    with open(CHUNKED_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for matching."""
    if not text:
        return text
    
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Normalize Alef forms
    text = re.sub(r'[أإآٱ]', 'ا', text)
    
    # Normalize Yaa
    text = re.sub(r'ى', 'ي', text)
    
    # Normalize Taa Marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # Remove tatweel
    text = re.sub(r'ـ', '', text)
    
    return text


def find_term_in_text(text: str, term: str) -> List[Dict[str, int]]:
    """
    Find all occurrences of a term in text with character offsets.
    
    Returns list of {char_start, char_end, quote} for each match.
    """
    matches = []
    text_normalized = normalize_arabic(text)
    term_normalized = normalize_arabic(term)
    
    # Also try with ال prefix
    terms_to_search = [term_normalized]
    if not term_normalized.startswith('ال'):
        terms_to_search.append('ال' + term_normalized)
    
    for search_term in terms_to_search:
        start = 0
        while True:
            idx = text_normalized.find(search_term, start)
            if idx == -1:
                break
            
            # Extract quote context (50 chars before and after)
            quote_start = max(0, idx - 50)
            quote_end = min(len(text), idx + len(search_term) + 50)
            quote = text[quote_start:quote_end]
            
            matches.append({
                'char_start': idx,
                'char_end': idx + len(search_term),
                'quote': quote.strip(),
            })
            start = idx + 1
    
    return matches


def build_concept_index():
    """Build the deterministic concept evidence index."""
    logger.info("Loading data...")
    
    entity_types = load_entity_types()
    behavior_concepts = load_behavior_concepts()
    chunks = load_chunked_index()
    
    term_to_entity = entity_types.get("term_to_entity_type", {})
    
    logger.info(f"Loaded {len(chunks)} chunks")
    logger.info(f"Loaded {len(term_to_entity)} term mappings")
    
    # Build concept index
    concept_index = {}
    
    # Process each concept from entity types
    for term, info in term_to_entity.items():
        entity_type = info.get("entity_type")
        canonical_id = info.get("canonical_id")
        
        if not canonical_id:
            continue
        
        concept_entry = {
            'concept_id': canonical_id,
            'term': term,
            'entity_type': entity_type,
            'verses': [],
            'tafsir_chunks': [],
            'per_source_stats': {s: {'count': 0, 'chunks': []} for s in CORE_SOURCES},
            'total_mentions': 0,
        }
        
        # Search for term in all chunks
        for chunk in chunks:
            text = chunk.get('text_clean', '')
            if not text:
                continue
            
            matches = find_term_in_text(text, term)
            
            if matches:
                source = chunk.get('source', '')
                verse_key = chunk.get('verse_key', '')
                
                for match in matches:
                    chunk_evidence = {
                        'chunk_id': chunk.get('chunk_id'),
                        'verse_key': verse_key,
                        'source': source,
                        'surah': chunk.get('surah'),
                        'ayah': chunk.get('ayah'),
                        'char_start': match['char_start'],
                        'char_end': match['char_end'],
                        'quote': match['quote'],
                    }
                    
                    concept_entry['tafsir_chunks'].append(chunk_evidence)
                    concept_entry['total_mentions'] += 1
                    
                    # Track per-source stats
                    if source in CORE_SOURCES:
                        concept_entry['per_source_stats'][source]['count'] += 1
                        if len(concept_entry['per_source_stats'][source]['chunks']) < 10:
                            concept_entry['per_source_stats'][source]['chunks'].append(chunk_evidence)
                    
                    # Track unique verses
                    if verse_key and verse_key not in [v['verse_key'] for v in concept_entry['verses']]:
                        concept_entry['verses'].append({
                            'verse_key': verse_key,
                            'surah': chunk.get('surah'),
                            'ayah': chunk.get('ayah'),
                        })
        
        # Only include concepts with evidence
        if concept_entry['total_mentions'] > 0:
            # Limit tafsir_chunks to top 50 for storage
            concept_entry['tafsir_chunks'] = concept_entry['tafsir_chunks'][:50]
            concept_index[canonical_id] = concept_entry
    
    logger.info(f"Built index for {len(concept_index)} concepts")
    
    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for concept_id, entry in concept_index.items():
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Wrote concept index to {OUTPUT_FILE}")
    
    # Write metadata
    metadata = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'concept_count': len(concept_index),
        'total_chunks_searched': len(chunks),
        'entity_types_version': entity_types.get('version', '1.0'),
        'vocab_version': behavior_concepts.get('version', '1.0'),
        'core_sources': CORE_SOURCES,
        'stats': {
            'concepts_with_evidence': len(concept_index),
            'avg_mentions_per_concept': sum(c['total_mentions'] for c in concept_index.values()) / len(concept_index) if concept_index else 0,
            'concepts_by_entity_type': defaultdict(int),
        }
    }
    
    for entry in concept_index.values():
        metadata['stats']['concepts_by_entity_type'][entry['entity_type']] += 1
    
    metadata['stats']['concepts_by_entity_type'] = dict(metadata['stats']['concepts_by_entity_type'])
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote metadata to {METADATA_FILE}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Concept Evidence Index Summary")
    print("=" * 60)
    print(f"Total concepts indexed: {len(concept_index)}")
    print(f"Concepts by entity type:")
    for et, count in metadata['stats']['concepts_by_entity_type'].items():
        print(f"  {et}: {count}")
    
    # Show top concepts by mentions
    top_concepts = sorted(concept_index.values(), key=lambda x: -x['total_mentions'])[:10]
    print(f"\nTop 10 concepts by mentions:")
    for c in top_concepts:
        sources_covered = sum(1 for s in CORE_SOURCES if c['per_source_stats'][s]['count'] > 0)
        print(f"  {c['term']} ({c['concept_id']}): {c['total_mentions']} mentions, {sources_covered}/5 sources")
    
    return concept_index


if __name__ == "__main__":
    build_concept_index()
