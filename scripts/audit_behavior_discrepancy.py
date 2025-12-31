#!/usr/bin/env python3
"""Audit behavior discrepancy between canonical_entities.json and concept_index_v3.jsonl"""

import json

# Load canonical entities
ent = json.load(open('vocab/canonical_entities.json', encoding='utf-8'))
canonical_ids = {b['id'] for b in ent.get('behaviors', [])}
canonical_map = {b['id']: b for b in ent.get('behaviors', [])}

# Load concept index
ci_ids = set()
with open('data/evidence/concept_index_v3.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('concept_id', '').startswith('BEH_'):
            ci_ids.add(entry['concept_id'])

# Find missing
missing = canonical_ids - ci_ids
extra = ci_ids - canonical_ids

print(f'Canonical entities: {len(canonical_ids)} behaviors')
print(f'Concept index v3: {len(ci_ids)} behaviors')
print(f'Missing from concept_index: {len(missing)}')
print(f'Extra in concept_index: {len(extra)}')

print('\nMissing behaviors (in canonical but not in concept_index):')
for m in sorted(missing):
    beh = canonical_map.get(m, {})
    ar = beh.get('ar', 'N/A')
    en = beh.get('en', 'N/A')
    print(f'  {m}: {ar} ({en})')

if extra:
    print('\nExtra behaviors (in concept_index but not in canonical):')
    for e in sorted(extra):
        print(f'  {e}')
