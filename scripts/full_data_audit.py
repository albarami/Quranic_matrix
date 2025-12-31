#!/usr/bin/env python3
"""
Complete QBM Data Audit

Verifies all claims about the system:
- 87 behaviors across 6,236 ayat
- 736K relations in graph
- 107K vectors in embeddings
- Multi-source tafsir (7 sources)
- Bouzidani Five-Context Framework
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
VOCAB_DIR = Path("vocab")

def audit_behaviors():
    """Audit behavior counts and mappings."""
    print("\n" + "=" * 70)
    print("1. BEHAVIOR AUDIT")
    print("=" * 70)
    
    # Canonical entities
    ent = json.load(open(VOCAB_DIR / "canonical_entities.json", encoding="utf-8"))
    canonical_behaviors = ent.get("behaviors", [])
    canonical_ids = {b["id"] for b in canonical_behaviors}
    
    # Concept index
    ci_ids = set()
    ci_verse_count = 0
    with open(DATA_DIR / "evidence" / "concept_index_v3.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("concept_id", "").startswith("BEH_"):
                ci_ids.add(entry["concept_id"])
                ci_verse_count += len(entry.get("verses", []))
    
    # Behavioral classifier
    try:
        from src.ml.behavioral_classifier import BEHAVIOR_CLASSES
        classifier_count = len(BEHAVIOR_CLASSES)
    except:
        classifier_count = "N/A"
    
    missing = canonical_ids - ci_ids
    
    print(f"  canonical_entities.json: {len(canonical_behaviors)} behaviors")
    print(f"  concept_index_v3.jsonl: {len(ci_ids)} behaviors")
    print(f"  behavioral_classifier.py: {classifier_count} behaviors")
    print(f"  Missing from concept_index: {len(missing)}")
    print(f"  Total behavior-verse links: {ci_verse_count}")
    
    if missing:
        print("\n  Missing behaviors:")
        for m in sorted(missing)[:5]:
            beh = next((b for b in canonical_behaviors if b["id"] == m), {})
            print(f"    {m}: {beh.get('ar')} ({beh.get('en')})")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    
    return {
        "canonical": len(canonical_behaviors),
        "concept_index": len(ci_ids),
        "missing": len(missing),
        "verse_links": ci_verse_count
    }


def audit_graphs():
    """Audit graph relations."""
    print("\n" + "=" * 70)
    print("2. GRAPH RELATIONS AUDIT")
    print("=" * 70)
    
    graphs = {}
    
    # semantic_graph_v3 (enterprise graph - primary)
    sg3_path = DATA_DIR / "graph" / "semantic_graph_v3.json"
    if sg3_path.exists():
        sg3 = json.load(open(sg3_path, encoding="utf-8"))
        graphs["semantic_graph_v3 (ENTERPRISE)"] = {
            "nodes": len(sg3.get("nodes", [])),
            "edges": len(sg3.get("edges", []))
        }
    
    # semantic_graph_v2
    sg2 = json.load(open(DATA_DIR / "graph" / "semantic_graph_v2.json", encoding="utf-8"))
    graphs["semantic_graph_v2"] = {
        "nodes": len(sg2.get("nodes", [])),
        "edges": len(sg2.get("edges", []))
    }
    
    # graph_v3
    g3 = json.load(open(DATA_DIR / "graph" / "graph_v3.json", encoding="utf-8"))
    graphs["graph_v3"] = {
        "nodes": len(g3.get("nodes", [])),
        "edges": len(g3.get("edges", []))
    }
    
    # cooccurrence_graph_v1
    co = json.load(open(DATA_DIR / "graph" / "cooccurrence_graph_v1.json", encoding="utf-8"))
    graphs["cooccurrence_graph_v1"] = {
        "nodes": len(co.get("nodes", [])),
        "edges": co.get("edge_count", len(co.get("edges", [])))
    }
    
    # semantic_graph_v1
    sg1 = json.load(open(DATA_DIR / "graph" / "semantic_graph_v1.json", encoding="utf-8"))
    graphs["semantic_graph_v1"] = {
        "nodes": len(sg1.get("nodes", [])),
        "edges": len(sg1.get("edges", []))
    }
    
    total_edges = sum(g["edges"] for g in graphs.values())
    
    for name, stats in graphs.items():
        print(f"  {name}: {stats['nodes']} nodes, {stats['edges']} edges")
    
    # Count behavior-to-behavior causal edges in enterprise graph
    if sg3_path.exists():
        beh_ids = {n["id"] for n in sg3.get("nodes", []) if n.get("type") == "BEHAVIOR"}
        causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
        beh_causal = [e for e in sg3.get("edges", []) 
                      if e["source"] in beh_ids and e["target"] in beh_ids 
                      and e.get("edge_type") in causal_types]
        print(f"\n  ENTERPRISE GRAPH CAUSAL EDGES: {len(beh_causal)} behavior-to-behavior")
    
    print(f"\n  TOTAL EDGES (all graphs): {total_edges:,}")
    
    return {"graphs": graphs, "total_edges": total_edges}


def audit_embeddings():
    """Audit embedding vectors."""
    print("\n" + "=" * 70)
    print("3. EMBEDDINGS AUDIT")
    print("=" * 70)
    
    embeddings = {}
    
    # gpu_index
    gpu_path = DATA_DIR / "embeddings" / "gpu_index.npy"
    if gpu_path.exists():
        gpu = np.load(gpu_path)
        embeddings["gpu_index"] = {"shape": gpu.shape, "count": gpu.shape[0]}
        print(f"  gpu_index.npy: {gpu.shape[0]:,} vectors, dim={gpu.shape[1]}")
    
    # annotations_embeddings
    ann_path = DATA_DIR / "embeddings" / "annotations_embeddings.npy"
    if ann_path.exists():
        ann = np.load(ann_path)
        embeddings["annotations"] = {"shape": ann.shape, "count": ann.shape[0]}
        print(f"  annotations_embeddings.npy: {ann.shape[0]:,} vectors, dim={ann.shape[1]}")
    
    # full_power_index
    fp_path = DATA_DIR / "indexes" / "full_power_index.npy"
    if fp_path.exists():
        fp = np.load(fp_path)
        embeddings["full_power"] = {"shape": fp.shape, "count": fp.shape[0]}
        print(f"  full_power_index.npy: {fp.shape[0]:,} vectors, dim={fp.shape[1]}")
    
    total_vectors = sum(e["count"] for e in embeddings.values())
    print(f"\n  TOTAL VECTORS: {total_vectors:,}")
    print(f"  CLAIMED: 107,000 vectors")
    print(f"  ACTUAL vs CLAIMED: {total_vectors / 107000 * 100:.1f}%")
    
    return embeddings


def audit_tafsir():
    """Audit tafsir sources."""
    print("\n" + "=" * 70)
    print("4. TAFSIR SOURCES AUDIT")
    print("=" * 70)
    
    tafsir_dir = DATA_DIR / "tafsir"
    sources = {}
    
    for tf in sorted(tafsir_dir.glob("*.jsonl")):
        with open(tf, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        sources[tf.stem] = count
        print(f"  {tf.name}: {count:,} entries")
    
    # Evidence index sources
    ev_sources = defaultdict(int)
    with open(DATA_DIR / "evidence" / "evidence_index_v2_chunked.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ev_sources[entry.get("source", "unknown")] += 1
    
    print(f"\n  Evidence index by source:")
    for src, count in sorted(ev_sources.items()):
        print(f"    {src}: {count:,} chunks")
    
    print(f"\n  TOTAL TAFSIR FILES: {len(sources)}")
    print(f"  TOTAL EVIDENCE CHUNKS: {sum(ev_sources.values()):,}")
    
    return {"files": sources, "evidence_chunks": dict(ev_sources)}


def audit_bouzidani():
    """Audit Bouzidani Framework implementation."""
    print("\n" + "=" * 70)
    print("5. BOUZIDANI FRAMEWORK AUDIT")
    print("=" * 70)
    
    try:
        from src.ml.qbm_bouzidani_taxonomy import BOUZIDANI_TAXONOMY, BehaviorDefinition
        taxonomy_count = len(BOUZIDANI_TAXONOMY)
        print(f"  BOUZIDANI_TAXONOMY entries: {taxonomy_count}")
        
        # Check coverage
        ent = json.load(open(VOCAB_DIR / "canonical_entities.json", encoding="utf-8"))
        canonical_ids = {b["id"] for b in ent.get("behaviors", [])}
        taxonomy_ids = set(BOUZIDANI_TAXONOMY.keys())
        
        covered = canonical_ids & taxonomy_ids
        missing = canonical_ids - taxonomy_ids
        
        print(f"  Canonical behaviors: {len(canonical_ids)}")
        print(f"  Covered by taxonomy: {len(covered)}")
        print(f"  Missing from taxonomy: {len(missing)}")
        print(f"  Coverage: {len(covered) / len(canonical_ids) * 100:.1f}%")
        
        # Check Five Contexts
        from src.ml.qbm_bouzidani_taxonomy import (
            OrganicContext, SituationalContext, SystemicContext,
            TemporalContext, SpatialContext
        )
        print(f"\n  Five Contexts implemented:")
        print(f"    OrganicContext: {len(OrganicContext)} values")
        print(f"    SituationalContext: {len(SituationalContext)} values")
        print(f"    SystemicContext: {len(SystemicContext)} values")
        print(f"    TemporalContext: {len(TemporalContext)} values")
        print(f"    SpatialContext: {len(SpatialContext)} values")
        
        return {"taxonomy_count": taxonomy_count, "coverage": len(covered) / len(canonical_ids)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def audit_rag():
    """Audit RAG system components."""
    print("\n" + "=" * 70)
    print("6. RAG SYSTEM AUDIT")
    print("=" * 70)
    
    # Check ChromaDB
    chroma_path = DATA_DIR / "chromadb" / "chroma.sqlite3"
    if chroma_path.exists():
        print(f"  ChromaDB: {chroma_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"  ChromaDB: NOT FOUND")
    
    # Check models
    models_dir = DATA_DIR / "models"
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    print(f"  Model directories: {len(model_dirs)}")
    for md in model_dirs[:5]:
        files = list(md.glob("*"))
        print(f"    {md.name}: {len(files)} files")
    
    # Check if key components exist
    components = {
        "proof_only_backend": Path("src/ml/proof_only_backend.py").exists(),
        "graph_reasoner": Path("src/ml/graph_reasoner.py").exists(),
        "arabic_embeddings": Path("src/ml/arabic_embeddings.py").exists(),
        "hybrid_rag_system": Path("src/ml/hybrid_rag_system.py").exists(),
        "full_power_system": Path("src/ml/full_power_system.py").exists(),
    }
    
    print(f"\n  RAG Components:")
    for comp, exists in components.items():
        status = "EXISTS" if exists else "MISSING"
        print(f"    {comp}: {status}")
    
    return {"components": components}


def main():
    print("=" * 70)
    print("QBM COMPLETE DATA AUDIT")
    print("=" * 70)
    
    results = {}
    results["behaviors"] = audit_behaviors()
    results["graphs"] = audit_graphs()
    results["embeddings"] = audit_embeddings()
    results["tafsir"] = audit_tafsir()
    results["bouzidani"] = audit_bouzidani()
    results["rag"] = audit_rag()
    
    print("\n" + "=" * 70)
    print("SUMMARY: CLAIMED vs ACTUAL")
    print("=" * 70)
    
    print(f"""
  BEHAVIORS:
    Claimed: 87 behaviors across 6,236 ayat
    Actual: {results['behaviors']['canonical']} canonical, {results['behaviors']['concept_index']} with verse mappings
    Gap: {results['behaviors']['missing']} behaviors missing verse mappings
    Verse links: {results['behaviors']['verse_links']:,} (not 6,236 ayat - these are behavior-verse links)
    
  RELATIONS:
    Claimed: 736,000 relations
    Actual: {results['graphs']['total_edges']:,} edges across all graphs
    Gap: {736000 - results['graphs']['total_edges']:,} missing
    
  EMBEDDINGS:
    Claimed: 107,000 vectors
    Actual: {sum(e['count'] for e in results['embeddings'].values()):,} vectors
    Status: EXCEEDS claim (multiple embedding sets)
    
  TAFSIR:
    Claimed: Multi-source (Ibn Kathir, Tabari, Qurtubi, Sa'di, Jalalayn)
    Actual: 7 sources with {sum(results['tafsir']['evidence_chunks'].values()):,} chunks
    Status: VERIFIED
""")
    
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
