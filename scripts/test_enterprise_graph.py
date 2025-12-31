#!/usr/bin/env python3
"""
Test the enterprise graph with a real benchmark question.

This tests:
1. Causal path finding from behavior A to behavior B
2. Tafsir evidence retrieval for the path
3. Academic-grade output with provenance
"""

import sys
sys.path.insert(0, ".")

from src.ml.proof_only_backend import LightweightProofBackend

def test_causal_chain():
    """Test causal chain from heedlessness to disbelief."""
    print("=" * 70)
    print("TEST: Causal Chain - Heedlessness to Disbelief")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    
    # Test causal path finding directly
    paths = backend._find_causal_paths(
        source_id="BEH_COG_HEEDLESSNESS",
        target_id="BEH_SPI_DISBELIEF",
        min_hops=1,
        max_hops=5
    )
    
    print(f"\nCausal paths from BEH_COG_HEEDLESSNESS to BEH_SPI_DISBELIEF:")
    print(f"Paths found: {len(paths)}")
    
    for i, path in enumerate(paths[:5]):
        print(f"\n  Path {i+1}:")
        nodes = path.get('nodes', [])
        edges = path.get('edges', [])
        print(f"    Nodes: {' -> '.join(nodes)}")
        edge_types = [e.get('edge_type', 'N/A') for e in edges]
        print(f"    Edge types: {edge_types}")
        print(f"    Hops: {path.get('hops', 'N/A')}")
        if edges:
            print(f"    Edge evidence:")
            for e in edges[:2]:
                ev = e.get('evidence', {})
                desc = ev.get('description', '') if isinstance(ev, dict) else ''
                print(f"      - {e.get('edge_type')}: {desc[:60]}...")
    
    return paths


def test_faith_chain():
    """Test causal chain from faith to good deeds."""
    print("\n" + "=" * 70)
    print("TEST: Causal Chain - Faith to Charity")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    
    # Test causal path finding directly
    paths = backend._find_causal_paths(
        source_id="BEH_SPI_FAITH",
        target_id="BEH_FIN_CHARITY",
        min_hops=1,
        max_hops=3
    )
    
    print(f"\nCausal paths from BEH_SPI_FAITH to BEH_FIN_CHARITY:")
    print(f"Paths found: {len(paths)}")
    
    for i, path in enumerate(paths[:5]):
        print(f"\n  Path {i+1}:")
        nodes = path.get('nodes', [])
        edges = path.get('edges', [])
        print(f"    Nodes: {' -> '.join(nodes)}")
        edge_types = [e.get('edge_type', 'N/A') for e in edges]
        print(f"    Edge types: {edge_types}")
        if edges:
            for e in edges[:2]:
                ev = e.get('evidence', {})
                desc = ev.get('description', '') if isinstance(ev, dict) else ''
                print(f"      - {e.get('edge_type')}: {desc[:60]}...")
    
    return paths


def test_behavior_dossier():
    """Test behavior dossier with tafsir evidence."""
    print("\n" + "=" * 70)
    print("TEST: Behavior Dossier - Patience (الصبر)")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    
    # Load concept index for patience
    concept_index = backend._load_concept_index()
    patience_data = concept_index.get("BEH_EMO_PATIENCE", {})
    
    print(f"\nBehavior: BEH_EMO_PATIENCE")
    print(f"Term: {patience_data.get('term')} ({patience_data.get('term_en')})")
    
    verses = patience_data.get("verses", [])
    print(f"\nVerses linked: {len(verses)}")
    
    # Get verse keys
    verse_keys = [v.get("verse_key") for v in verses[:10]]
    print(f"Sample verse keys: {verse_keys[:5]}")
    
    # Get tafsir for these verses
    tafsir = backend._get_tafsir_for_verses(verse_keys[:5], max_per_source=2)
    
    print(f"\nTafsir sources retrieved:")
    for source, chunks in tafsir.items():
        if chunks:
            print(f"  {source}: {len(chunks)} chunks")
            for chunk in chunks[:1]:
                text = chunk.get("text", "")[:150]
                print(f"    {chunk.get('verse_key')}: {text}...")
    
    return {"behavior": patience_data, "tafsir": tafsir}


def test_graph_statistics():
    """Test graph statistics."""
    print("\n" + "=" * 70)
    print("TEST: Graph Statistics")
    print("=" * 70)
    
    backend = LightweightProofBackend()
    graph = backend._load_semantic_graph()
    
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    print(f"\nGraph version: {graph.get('version')}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    # Count edge types
    from collections import Counter
    edge_types = Counter(e.get("edge_type") for e in edges)
    print(f"\nEdge types:")
    for et, count in edge_types.most_common():
        print(f"  {et}: {count}")
    
    # Count behavior nodes
    beh_nodes = [n for n in nodes if n.get("type") == "BEHAVIOR"]
    print(f"\nBehavior nodes: {len(beh_nodes)}")
    
    # Count behavior-to-behavior causal edges
    beh_ids = {n["id"] for n in beh_nodes}
    causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
    beh_causal = [e for e in edges 
                  if e["source"] in beh_ids and e["target"] in beh_ids 
                  and e.get("edge_type") in causal_types]
    print(f"Behavior-to-behavior causal edges: {len(beh_causal)}")
    
    return graph


if __name__ == "__main__":
    print("QBM Enterprise Graph Test Suite")
    print("=" * 70)
    
    # Test graph statistics first
    test_graph_statistics()
    
    # Test causal chains
    test_causal_chain()
    test_faith_chain()
    
    # Test behavior dossier
    test_behavior_dossier()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
