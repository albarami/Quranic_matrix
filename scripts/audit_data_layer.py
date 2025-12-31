#!/usr/bin/env python3
"""
Audit the QBM data layer to verify academic-grade data is available.

This script checks:
1. Evidence index has actual tafsir text
2. Semantic graph has causal edges with evidence
3. Behavior-to-behavior causal paths exist
4. Tafsir sources are complete
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
VOCAB_DIR = Path("vocab")


def audit_evidence_index():
    """Audit the evidence index for tafsir chunks."""
    print("\n" + "=" * 70)
    print("1. EVIDENCE INDEX AUDIT")
    print("=" * 70)
    
    ev_path = DATA_DIR / "evidence" / "evidence_index_v2_chunked.jsonl"
    if not ev_path.exists():
        print(f"  ERROR: {ev_path} not found")
        return
    
    sources = defaultdict(int)
    total = 0
    sample_text = None
    
    with open(ev_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sources[entry.get("source", "unknown")] += 1
            total += 1
            if sample_text is None and entry.get("text_clean"):
                sample_text = entry.get("text_clean")[:200]
    
    print(f"  Total chunks: {total:,}")
    print(f"  Sources: {dict(sources)}")
    print(f"  Sample text: {sample_text}...")
    

def audit_semantic_graph():
    """Audit the semantic graph for causal edges."""
    print("\n" + "=" * 70)
    print("2. SEMANTIC GRAPH AUDIT")
    print("=" * 70)
    
    sg_path = DATA_DIR / "graph" / "semantic_graph_v2.json"
    if not sg_path.exists():
        print(f"  ERROR: {sg_path} not found")
        return
    
    with open(sg_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    
    # Count edge types
    edge_types = defaultdict(int)
    for e in edges:
        edge_types[e.get("edge_type", "unknown")] += 1
    print(f"  Edge types: {dict(edge_types)}")
    
    # Count edges with evidence
    edges_with_ev = [e for e in edges if e.get("evidence")]
    print(f"  Edges with evidence: {len(edges_with_ev)}")
    
    # Count behavior-to-behavior causal edges
    causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS", "PREVENTS"}
    beh_causal = [
        e for e in edges 
        if e.get("edge_type") in causal_types
        and e.get("source", "").startswith("BEH_")
        and e.get("target", "").startswith("BEH_")
    ]
    print(f"  Behavior-to-behavior causal edges: {len(beh_causal)}")
    
    if beh_causal:
        print("  Sample behavior causal edges:")
        for e in beh_causal[:5]:
            print(f"    {e['source']} --{e['edge_type']}--> {e['target']}")
            if e.get("evidence"):
                ev = e["evidence"][0]
                print(f"      Evidence: {ev.get('verse_key')} from {ev.get('source')}")
                print(f"      Quote: {ev.get('quote', '')[:100]}...")


def audit_concept_index():
    """Audit the concept index for behavior-verse mappings."""
    print("\n" + "=" * 70)
    print("3. CONCEPT INDEX AUDIT")
    print("=" * 70)
    
    # Try v3 first, then v2
    ci_path = DATA_DIR / "evidence" / "concept_index_v3.jsonl"
    if not ci_path.exists():
        ci_path = DATA_DIR / "evidence" / "concept_index_v2.jsonl"
    
    if not ci_path.exists():
        print(f"  ERROR: concept index not found")
        return
    
    print(f"  Using: {ci_path.name}")
    
    behaviors = []
    total_verses = 0
    total_tafsir = 0
    
    with open(ci_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("concept_id", "").startswith("BEH_"):
                behaviors.append(entry)
                total_verses += len(entry.get("verses", []))
                total_tafsir += len(entry.get("tafsir_chunks", []))
    
    print(f"  Behaviors: {len(behaviors)}")
    print(f"  Total verse links: {total_verses}")
    print(f"  Total tafsir chunks: {total_tafsir}")
    
    if behaviors:
        sample = behaviors[0]
        print(f"  Sample behavior: {sample.get('concept_id')}")
        print(f"    Term: {sample.get('term')}")
        print(f"    Verses: {len(sample.get('verses', []))}")
        if sample.get("verses"):
            print(f"    First verse: {sample['verses'][0]}")


def audit_tafsir_files():
    """Audit raw tafsir files."""
    print("\n" + "=" * 70)
    print("4. TAFSIR FILES AUDIT")
    print("=" * 70)
    
    tafsir_dir = DATA_DIR / "tafsir"
    if not tafsir_dir.exists():
        print(f"  ERROR: {tafsir_dir} not found")
        return
    
    tafsir_files = list(tafsir_dir.glob("*.jsonl"))
    print(f"  Tafsir files: {len(tafsir_files)}")
    
    for tf in sorted(tafsir_files):
        with open(tf, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"    {tf.name}: {count:,} entries")


def audit_canonical_entities():
    """Audit canonical entities vocabulary."""
    print("\n" + "=" * 70)
    print("5. CANONICAL ENTITIES AUDIT")
    print("=" * 70)
    
    ce_path = VOCAB_DIR / "canonical_entities.json"
    if not ce_path.exists():
        print(f"  ERROR: {ce_path} not found")
        return
    
    with open(ce_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    for entity_type in ["behaviors", "agents", "organs", "heart_states", "consequences"]:
        items = entities.get(entity_type, [])
        print(f"  {entity_type}: {len(items)}")
        if items:
            sample = items[0]
            print(f"    Sample: {sample.get('id')} - {sample.get('ar')} ({sample.get('en')})")


def main():
    print("=" * 70)
    print("QBM DATA LAYER AUDIT - ACADEMIC GRADE VERIFICATION")
    print("=" * 70)
    
    audit_evidence_index()
    audit_semantic_graph()
    audit_concept_index()
    audit_tafsir_files()
    audit_canonical_entities()
    
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
