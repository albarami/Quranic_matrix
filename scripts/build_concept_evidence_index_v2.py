"""
Build Complete Concept Evidence Index v2 (Phase 8.2)

Maps EVERY canonical entity to evidence:
- If evidence exists: include verse_keys + tafsir_mentions with offsets
- If evidence does not exist: keep entry with empty evidence and status=no_evidence

This ensures the concept index is COMPLETE for all 126 canonical entities.

Output: data/evidence/concept_index_v2.jsonl
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
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
OUTPUT_FILE = Path("data/evidence/concept_index_v2.jsonl")
METADATA_FILE = Path("data/evidence/concept_index_v2_metadata.json")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


def load_canonical_entities() -> Dict[str, Any]:
    """Load canonical entities registry."""
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
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
    
    if not term_normalized:
        return matches
    
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


def build_all_entities_list(canonical_entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a complete list of ALL canonical entities.
    
    Returns list of {id, ar, en, entity_type} for every entity.
    """
    entities = []
    
    # Add all behaviors (73)
    for behavior in canonical_entities.get("behaviors", []):
        entities.append({
            "id": behavior["id"],
            "ar": behavior.get("ar", ""),
            "en": behavior.get("en", ""),
            "entity_type": "BEHAVIOR",
        })
    
    # Add all agents
    for agent in canonical_entities.get("agents", []):
        entities.append({
            "id": agent["id"],
            "ar": agent.get("ar", ""),
            "en": agent.get("en", ""),
            "entity_type": "AGENT",
        })
    
    # Add all organs
    for organ in canonical_entities.get("organs", []):
        entities.append({
            "id": organ["id"],
            "ar": organ.get("ar", ""),
            "en": organ.get("en", ""),
            "entity_type": "ORGAN",
        })
    
    # Add all heart states
    for state in canonical_entities.get("heart_states", []):
        entities.append({
            "id": state["id"],
            "ar": state.get("ar", ""),
            "en": state.get("en", ""),
            "entity_type": "HEART_STATE",
        })
    
    # Add all consequences
    for consequence in canonical_entities.get("consequences", []):
        entities.append({
            "id": consequence["id"],
            "ar": consequence.get("ar", ""),
            "en": consequence.get("en", ""),
            "entity_type": "CONSEQUENCE",
        })
    
    return entities


def build_concept_index_v2():
    """
    Build the COMPLETE concept evidence index.
    
    Every canonical entity gets an entry, even if no evidence is found.
    """
    logger.info("Loading data...")
    
    canonical_entities = load_canonical_entities()
    chunks = load_chunked_index()
    
    # Build complete entity list
    all_entities = build_all_entities_list(canonical_entities)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    logger.info(f"Found {len(all_entities)} canonical entities")
    
    # Build concept index for EVERY entity
    concept_index = {}
    entities_with_evidence = 0
    entities_without_evidence = 0
    
    for entity in all_entities:
        entity_id = entity["id"]
        term = entity["ar"]
        entity_type = entity["entity_type"]
        
        concept_entry = {
            'concept_id': entity_id,
            'term': term,
            'term_en': entity.get("en", ""),
            'entity_type': entity_type,
            'status': 'no_evidence',  # Default to no_evidence
            'verses': [],
            'tafsir_chunks': [],
            'per_source_stats': {s: {'count': 0, 'chunks': []} for s in CORE_SOURCES},
            'total_mentions': 0,
        }
        
        # Skip if no Arabic term to search
        if not term:
            concept_index[entity_id] = concept_entry
            entities_without_evidence += 1
            continue
        
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
        
        # Update status based on evidence
        if concept_entry['total_mentions'] > 0:
            concept_entry['status'] = 'found'
            entities_with_evidence += 1
            # Limit tafsir_chunks to top 50 for storage
            concept_entry['tafsir_chunks'] = concept_entry['tafsir_chunks'][:50]
        else:
            entities_without_evidence += 1
        
        # Calculate sources covered
        concept_entry['sources_covered'] = [
            s for s in CORE_SOURCES 
            if concept_entry['per_source_stats'][s]['count'] > 0
        ]
        concept_entry['sources_count'] = len(concept_entry['sources_covered'])
        
        concept_index[entity_id] = concept_entry
    
    logger.info(f"Built index for {len(concept_index)} concepts")
    logger.info(f"  With evidence: {entities_with_evidence}")
    logger.info(f"  Without evidence: {entities_without_evidence}")
    
    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for concept_id, entry in concept_index.items():
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Wrote concept index to {OUTPUT_FILE}")
    
    # Write metadata
    metadata = {
        'version': '2.0',
        'created_at': datetime.now().isoformat(),
        'concept_count': len(concept_index),
        'total_chunks_searched': len(chunks),
        'canonical_entities_version': canonical_entities.get('version', '1.0'),
        'core_sources': CORE_SOURCES,
        'completeness': {
            'total_canonical_entities': len(all_entities),
            'entities_in_index': len(concept_index),
            'entities_with_evidence': entities_with_evidence,
            'entities_without_evidence': entities_without_evidence,
            'coverage_rate': entities_with_evidence / len(all_entities) if all_entities else 0,
        },
        'stats': {
            'concepts_with_evidence': entities_with_evidence,
            'concepts_without_evidence': entities_without_evidence,
            'avg_mentions_per_concept': sum(c['total_mentions'] for c in concept_index.values()) / len(concept_index) if concept_index else 0,
            'concepts_by_entity_type': defaultdict(int),
            'concepts_with_5_sources': 0,
            'concepts_with_3plus_sources': 0,
        }
    }
    
    for entry in concept_index.values():
        metadata['stats']['concepts_by_entity_type'][entry['entity_type']] += 1
        if entry['sources_count'] >= 5:
            metadata['stats']['concepts_with_5_sources'] += 1
        if entry['sources_count'] >= 3:
            metadata['stats']['concepts_with_3plus_sources'] += 1
    
    metadata['stats']['concepts_by_entity_type'] = dict(metadata['stats']['concepts_by_entity_type'])
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote metadata to {METADATA_FILE}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Concept Evidence Index v2 Summary (COMPLETE)")
    print("=" * 60)
    print(f"Total canonical entities: {len(all_entities)}")
    print(f"Total concepts indexed: {len(concept_index)}")
    print(f"  With evidence: {entities_with_evidence}")
    print(f"  Without evidence (status=no_evidence): {entities_without_evidence}")
    print(f"\nConcepts by entity type:")
    for et, count in metadata['stats']['concepts_by_entity_type'].items():
        print(f"  {et}: {count}")
    
    print(f"\nSource coverage:")
    print(f"  Concepts with 5/5 sources: {metadata['stats']['concepts_with_5_sources']}")
    print(f"  Concepts with 3+/5 sources: {metadata['stats']['concepts_with_3plus_sources']}")
    
    # Show entities without evidence
    no_evidence = [e for e in concept_index.values() if e['status'] == 'no_evidence']
    if no_evidence:
        print(f"\nEntities without evidence ({len(no_evidence)}):")
        for e in no_evidence[:10]:
            print(f"  {e['concept_id']}: {e['term']} ({e['term_en']})")
        if len(no_evidence) > 10:
            print(f"  ... and {len(no_evidence) - 10} more")
    
    # Show top concepts by mentions
    top_concepts = sorted(
        [c for c in concept_index.values() if c['total_mentions'] > 0], 
        key=lambda x: -x['total_mentions']
    )[:10]
    print(f"\nTop 10 concepts by mentions:")
    for c in top_concepts:
        print(f"  {c['term']} ({c['concept_id']}): {c['total_mentions']} mentions, {c['sources_count']}/5 sources")
    
    return concept_index


if __name__ == "__main__":
    build_concept_index_v2()
