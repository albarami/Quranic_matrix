#!/usr/bin/env python3
"""Fix missing evidence_policy_mode field in concept_index_v3.jsonl"""
import json

entries = []
with open('data/evidence/concept_index_v3.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entries.append(json.loads(line))

fixed = 0
for entry in entries:
    if 'evidence_policy_mode' not in entry:
        entry['evidence_policy_mode'] = 'lexical'
        fixed += 1

with open('data/evidence/concept_index_v3.jsonl', 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'Fixed {fixed} entries')
