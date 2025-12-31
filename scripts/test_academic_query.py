#!/usr/bin/env python3
"""
Test academic-grade query output from LightweightProofBackend.

This script tests Question A01 from the benchmark:
"Trace ALL distinct causal chains (minimum 3 hops) from الغفلة (heedlessness) 
to الكفر (disbelief). For each chain, provide: intermediate behaviors, 
verse evidence for each link, and tafsir from Ibn Kathir and Qurtubi."
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from src.ml.proof_only_backend import LightweightProofBackend


def test_causal_chain_query():
    """Test finding causal chains with full evidence."""
    print("=" * 70)
    print("ACADEMIC QUERY TEST: Causal Chain from الغفلة to الكفر")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    
    # Resolve behavior terms
    source_term = "الغفلة"
    target_term = "الكفر"
    
    source_id = backend._resolve_behavior_term(source_term)
    target_id = backend._resolve_behavior_term(target_term)
    
    print(f"\nSource: {source_term} -> {source_id}")
    print(f"Target: {target_term} -> {target_id}")
    
    if not source_id or not target_id:
        print("ERROR: Could not resolve behavior terms")
        return
    
    # Find causal paths
    paths = backend._find_causal_paths(source_id, target_id, min_hops=2, max_hops=5)
    print(f"\nCausal paths found: {len(paths)}")
    
    if not paths:
        print("No paths found. Checking graph connectivity...")
        # Debug: show what edges exist from source
        graph = backend._load_semantic_graph()
        causal_types = {"CAUSES", "LEADS_TO", "STRENGTHENS"}
        outgoing = [e for e in graph["edges"] 
                   if e.get("source") == source_id 
                   and e.get("edge_type") in causal_types]
        print(f"Outgoing causal edges from {source_id}: {len(outgoing)}")
        for e in outgoing[:5]:
            print(f"  -> {e['target']} ({e['edge_type']})")
        return
    
    # Display paths with full evidence
    for i, path in enumerate(paths[:3]):
        print(f"\n{'='*60}")
        print(f"PATH {i+1}: {path['hops']} hops")
        print("=" * 60)
        
        # Show node chain
        nodes = path["nodes"]
        print(f"Chain: {' -> '.join(nodes)}")
        
        # Show each edge with evidence
        for j, edge in enumerate(path["edges"]):
            print(f"\n  LINK {j+1}: {edge['source']} --{edge.get('edge_type')}--> {edge['target']}")
            print(f"  Confidence: {edge.get('confidence', 'N/A')}")
            
            evidence = edge.get("evidence", [])
            print(f"  Evidence items: {len(evidence)}")
            
            # Show tafsir quotes from Ibn Kathir and Qurtubi
            for ev in evidence[:3]:
                source = ev.get("source", "")
                verse_key = ev.get("verse_key", "")
                quote = ev.get("quote", "")
                
                if source in ["ibn_kathir", "qurtubi"]:
                    print(f"\n    [{source.upper()}] Verse {verse_key}:")
                    print(f"    \"{quote[:200]}...\"")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def test_behavior_dossier():
    """Test getting full behavior dossier with tafsir."""
    print("\n" + "=" * 70)
    print("ACADEMIC QUERY TEST: Behavior Dossier for الصبر (Patience)")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    
    # Load concept index
    concept_index = backend._load_concept_index()
    
    # Find patience behavior
    patience_id = backend._resolve_behavior_term("الصبر")
    print(f"\nBehavior ID: {patience_id}")
    
    if patience_id and patience_id in concept_index:
        concept = concept_index[patience_id]
        verses = concept.get("verses", [])
        
        print(f"Verses linked: {len(verses)}")
        
        # Get tafsir for first 3 verses
        verse_keys = [v["verse_key"] for v in verses[:3]]
        print(f"\nSample verses: {verse_keys}")
        
        tafsir = backend._get_tafsir_for_verses(verse_keys)
        
        for source, chunks in tafsir.items():
            if chunks:
                print(f"\n[{source.upper()}] {len(chunks)} chunks")
                for chunk in chunks[:1]:
                    print(f"  Verse {chunk['verse_key']}:")
                    print(f"  \"{chunk['text'][:200]}...\"")
    else:
        print("Behavior not found in concept index")


if __name__ == "__main__":
    test_causal_chain_query()
    test_behavior_dossier()
