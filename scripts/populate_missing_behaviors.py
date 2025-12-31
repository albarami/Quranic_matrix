#!/usr/bin/env python3
"""
Populate missing 14 behaviors in concept_index_v3.jsonl

This script:
1. Loads the 14 missing behaviors from canonical_entities.json
2. Searches Quran text for their roots and synonyms
3. Builds proper verse mappings with evidence
4. Appends to concept_index_v3.jsonl
"""

import json
import re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
VOCAB_DIR = Path("vocab")

# Load canonical entities
ent = json.load(open(VOCAB_DIR / "canonical_entities.json", encoding="utf-8"))
canonical_map = {b["id"]: b for b in ent.get("behaviors", [])}

# Load existing concept index
existing_ids = set()
with open(DATA_DIR / "evidence" / "concept_index_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        existing_ids.add(entry.get("concept_id", ""))

missing_ids = set(canonical_map.keys()) - existing_ids
missing_behaviors = [canonical_map[m] for m in missing_ids if m.startswith("BEH_")]

print(f"Missing behaviors to populate: {len(missing_behaviors)}")

# Load Quran verses
quran_path = DATA_DIR / "quran" / "uthmani_hafs_v1.tok_v1.json"
quran_verses = {}
if quran_path.exists():
    quran_data = json.load(open(quran_path, encoding="utf-8"))
    for surah in quran_data.get("surahs", []):
        surah_num = surah.get("surah", 0)
        for ayah in surah.get("ayat", []):
            ayah_num = ayah.get("ayah", 0)
            text = ayah.get("text", "")
            if surah_num and ayah_num:
                verse_key = f"{surah_num}:{ayah_num}"
                quran_verses[verse_key] = text
    print(f"Loaded {len(quran_verses)} Quran verses")
else:
    print(f"ERROR: Quran source not found at {quran_path}")
    exit(1)

# Build root patterns for searching
def build_root_pattern(root):
    """Build regex pattern from Arabic root like 'ص-د-ق'."""
    letters = root.replace("-", "")
    if len(letters) < 2:
        return None
    # Build pattern that matches any form of the root
    # Allow any characters between root letters
    pattern = ".*".join(letters)
    return pattern

def search_verses_for_behavior(behavior):
    """Search Quran verses for a behavior's roots and synonyms."""
    matches = []
    
    roots = behavior.get("roots", [])
    synonyms = behavior.get("synonyms", [])
    ar_term = behavior.get("ar", "")
    
    # Build search terms
    search_terms = set()
    if ar_term:
        search_terms.add(ar_term)
        # Remove ال prefix
        if ar_term.startswith("ال"):
            search_terms.add(ar_term[2:])
    
    for syn in synonyms:
        if syn:
            search_terms.add(syn)
            if syn.startswith("ال"):
                search_terms.add(syn[2:])
    
    # Search each verse
    for verse_key, text in quran_verses.items():
        matched_tokens = []
        
        # Check synonyms and terms
        for term in search_terms:
            if term and term in text:
                matched_tokens.append(term)
        
        # Check root patterns
        for root in roots:
            pattern = build_root_pattern(root)
            if pattern:
                # Find words matching the root pattern
                words = text.split()
                for word in words:
                    if re.search(pattern, word):
                        if word not in matched_tokens:
                            matched_tokens.append(word)
        
        if matched_tokens:
            parts = verse_key.split(":")
            surah = int(parts[0])
            ayah = int(parts[1])
            
            matches.append({
                "verse_key": verse_key,
                "surah": surah,
                "ayah": ayah,
                "text_uthmani": text,
                "evidence": [{
                    "type": "lexical",
                    "matched_token": matched_tokens[0],
                    "pattern": "|".join(search_terms)
                }],
                "directness": "direct" if any(t in text for t in [ar_term] + synonyms[:3]) else "inferred",
                "provenance": "lexeme_index_v1"
            })
    
    return matches

# Process each missing behavior
new_entries = []
for beh in missing_behaviors:
    beh_id = beh["id"]
    print(f"\nProcessing {beh_id}: {beh.get('ar')} ({beh.get('en')})")
    
    verses = search_verses_for_behavior(beh)
    print(f"  Found {len(verses)} verses")
    
    if verses:
        entry = {
            "concept_id": beh_id,
            "term": beh.get("ar", ""),
            "term_en": beh.get("en", ""),
            "entity_type": "BEHAVIOR",
            "status": "active",
            "verses": verses,
            "tafsir_chunks": []  # Will be populated by tafsir indexer
        }
        new_entries.append(entry)

print(f"\n{'='*60}")
print(f"Total new entries: {len(new_entries)}")
print(f"Total verses mapped: {sum(len(e['verses']) for e in new_entries)}")

# Append to concept_index_v3.jsonl
output_path = DATA_DIR / "evidence" / "concept_index_v3.jsonl"
with open(output_path, "a", encoding="utf-8") as f:
    for entry in new_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nAppended {len(new_entries)} entries to {output_path}")

# Verify
with open(output_path, "r", encoding="utf-8") as f:
    total = sum(1 for _ in f)
print(f"Total entries in concept_index_v3.jsonl: {total}")
