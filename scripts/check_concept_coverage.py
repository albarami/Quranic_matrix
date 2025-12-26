"""
Phase 10.4: Check concept map completeness.
Verify all canonical entities have entries in concept index.
"""
import json
from pathlib import Path

CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")

def main():
    # Load canonical entities
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    # Load concept index
    concepts = []
    with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            concepts.append(json.loads(line))
    
    concept_ids = set(c.get("concept_id") for c in concepts)
    
    print("=== CONCEPT MAP COMPLETENESS CHECK ===\n")
    print(f"Concept index entries: {len(concepts)}")
    
    total_missing = 0
    for key in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
        items = entities.get(key, [])
        ids = set(i.get("id") for i in items)
        missing = ids - concept_ids
        total_missing += len(missing)
        
        status = "OK" if len(missing) == 0 else "MISSING"
        print(f"\n{key}: {len(ids)} total, {len(missing)} missing [{status}]")
        
        if missing:
            for m in list(missing)[:5]:
                # Find the term for this ID
                term = next((i.get("term_ar", i.get("name", m)) for i in items if i.get("id") == m), m)
                print(f"  - {m}: {term}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total missing: {total_missing}")
    
    if total_missing == 0:
        print("All canonical entities have concept index entries!")
        return 0
    else:
        print(f"WARNING: {total_missing} entities missing from concept index")
        return 1

if __name__ == "__main__":
    exit(main())
