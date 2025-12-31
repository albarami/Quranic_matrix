#!/usr/bin/env python3
"""Count canonical entities."""
import json

with open('vocab/canonical_entities.json', 'r', encoding='utf-8') as f:
    e = json.load(f)

print(f"behaviors: {len(e.get('behaviors', []))}")
print(f"agents: {len(e.get('agents', []))}")
print(f"organs: {len(e.get('organs', []))}")
print(f"heart_states: {len(e.get('heart_states', []))}")
print(f"consequences: {len(e.get('consequences', []))}")

total = (len(e.get('behaviors', [])) + 
         len(e.get('agents', [])) + 
         len(e.get('organs', [])) + 
         len(e.get('heart_states', [])) + 
         len(e.get('consequences', [])))
print(f"TOTAL: {total}")
