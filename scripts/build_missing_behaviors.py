#!/usr/bin/env python3
"""
Build missing 14 behavior verse mappings.

This script:
1. Identifies the 14 behaviors missing from concept_index_v3.jsonl
2. Searches the Quran text for their Arabic roots
3. Builds proper verse mappings with evidence
4. Outputs updated concept_index entries
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
ci_ids = set()
with open(DATA_DIR / "evidence" / "concept_index_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("concept_id", "").startswith("BEH_"):
            ci_ids.add(entry["concept_id"])

missing = set(canonical_map.keys()) - ci_ids

print(f"Missing behaviors: {len(missing)}")
print("=" * 70)

for m in sorted(missing):
    beh = canonical_map[m]
    print(f"\n{m}:")
    print(f"  Arabic: {beh.get('ar')}")
    print(f"  English: {beh.get('en')}")
    print(f"  Category: {beh.get('category')}")
    print(f"  Roots: {beh.get('roots', [])}")
    print(f"  Synonyms: {beh.get('synonyms', [])}")
