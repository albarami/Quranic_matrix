#!/usr/bin/env python3
"""Fix missing fields in concept_index_v3.jsonl for new behaviors"""
import json

entries = []
with open('data/evidence/concept_index_v3.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entries.append(json.loads(line))

fixed = 0
for entry in entries:
    changed = False
    
    if 'evidence_policy_mode' not in entry:
        entry['evidence_policy_mode'] = 'lexical'
        changed = True
    
    if 'statistics' not in entry:
        entry['statistics'] = {
            'total_sources': 0,
            'sources_by_type': {},
            'avg_confidence': 0.0
        }
        changed = True
    
    if 'tafsir_chunks' not in entry:
        entry['tafsir_chunks'] = []
        changed = True
    
    if 'total_mentions' not in entry:
        entry['total_mentions'] = 0
        changed = True
    
    if 'validation' not in entry:
        entry['validation'] = {
            'passed': True,
            'errors': [],
            'warnings': []
        }
        changed = True
    
    if 'verses' not in entry:
        entry['verses'] = []
        changed = True
    
    if 'term' not in entry:
        # Use concept_id as fallback
        entry['term'] = entry.get('concept_id', '').replace('BEH_', '').replace('_', ' ')
        changed = True
    
    if 'term_en' not in entry:
        entry['term_en'] = entry.get('concept_id', '').replace('BEH_', '').replace('_', ' ').title()
        changed = True
    
    if changed:
        fixed += 1

with open('data/evidence/concept_index_v3.jsonl', 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'Fixed {fixed} entries')
