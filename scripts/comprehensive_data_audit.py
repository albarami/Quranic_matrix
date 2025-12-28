"""
Comprehensive Data Audit for QBM System
========================================
Audits all data sources: Quran, Tafsir, Annotations, Behaviors, 
Graph, Vocabs, Embeddings, and their interconnections.

As a Data Engineer and Arabic/Quranic Expert perspective.
"""

import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def audit_quran_data() -> Dict[str, Any]:
    """Audit Quran verse data."""
    print("\n" + "=" * 60)
    print("1. QURAN DATA AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "stats": {}}
    
    quran_file = DATA_DIR / "quran" / "quran_full.json"
    if not quran_file.exists():
        results["status"] = "ERROR"
        results["issues"].append("quran_full.json not found")
        return results
    
    with open(quran_file, 'r', encoding='utf-8') as f:
        quran = json.load(f)
    
    total_verses = 0
    surahs = len(quran)
    empty_verses = 0
    
    for surah_num, surah_data in quran.items():
        verses = surah_data.get("verses", {})
        for ayah_num, text in verses.items():
            total_verses += 1
            if not text or len(text.strip()) < 5:
                empty_verses += 1
    
    results["stats"] = {
        "surahs": surahs,
        "total_verses": total_verses,
        "empty_verses": empty_verses,
        "expected_verses": 6236,
    }
    
    if total_verses != 6236:
        results["issues"].append(f"Expected 6236 verses, found {total_verses}")
    if empty_verses > 0:
        results["issues"].append(f"{empty_verses} empty verses")
    
    print(f"  Surahs: {surahs}")
    print(f"  Total verses: {total_verses} (expected: 6236)")
    print(f"  Empty verses: {empty_verses}")
    print(f"  Status: {results['status']}")
    
    return results


def audit_tafsir_data() -> Dict[str, Any]:
    """Audit all 7 tafsir sources."""
    print("\n" + "=" * 60)
    print("2. TAFSIR DATA AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "sources": {}}
    
    tafsir_dir = DATA_DIR / "tafsir"
    expected_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "muyassar", "baghawi"]
    
    for source in expected_sources:
        filepath = tafsir_dir / f"{source}.ar.jsonl"
        source_stats = {"records": 0, "html_contaminated": 0, "empty": 0, "avg_length": 0}
        
        if not filepath.exists():
            results["issues"].append(f"Missing: {source}.ar.jsonl")
            results["sources"][source] = {"status": "MISSING"}
            continue
        
        total_length = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    text = record.get("text_ar", "")
                    source_stats["records"] += 1
                    total_length += len(text)
                    
                    if "<" in text and ">" in text:
                        source_stats["html_contaminated"] += 1
                    if len(text) < 10:
                        source_stats["empty"] += 1
        
        source_stats["avg_length"] = total_length // max(source_stats["records"], 1)
        results["sources"][source] = source_stats
        
        status = "✅" if source_stats["html_contaminated"] == 0 else "❌ HTML"
        completeness = source_stats["records"] / 6236 * 100
        print(f"  {source}: {source_stats['records']} records ({completeness:.0f}%) {status}")
        
        if source_stats["html_contaminated"] > 0:
            results["issues"].append(f"{source}: {source_stats['html_contaminated']} HTML contaminated")
    
    return results


def audit_behavioral_annotations() -> Dict[str, Any]:
    """Audit behavioral annotations."""
    print("\n" + "=" * 60)
    print("3. BEHAVIORAL ANNOTATIONS AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "files": {}}
    
    ann_dir = DATA_DIR / "annotations"
    files_to_check = [
        "tafsir_annotations.jsonl",
        "tafsir_behavioral_annotations.jsonl",
        "tafsir_behavioral_5axis.jsonl",
    ]
    
    for filename in files_to_check:
        filepath = ann_dir / filename
        if not filepath.exists():
            results["issues"].append(f"Missing: {filename}")
            continue
        
        stats = {"records": 0, "html": 0, "behaviors": Counter(), "sources": Counter()}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    stats["records"] += 1
                    
                    context = record.get("context", "")
                    if "<" in context and ">" in context:
                        stats["html"] += 1
                    
                    beh = record.get("behavior_id") or record.get("behavior_ar", "")
                    if beh:
                        stats["behaviors"][beh] += 1
                    
                    source = record.get("source", "")
                    if source:
                        stats["sources"][source] += 1
        
        html_pct = stats["html"] / max(stats["records"], 1) * 100
        status = "✅ CLEAN" if stats["html"] == 0 else f"❌ {html_pct:.1f}% HTML"
        
        results["files"][filename] = {
            "records": stats["records"],
            "html_contaminated": stats["html"],
            "unique_behaviors": len(stats["behaviors"]),
            "sources": dict(stats["sources"]),
        }
        
        print(f"  {filename}:")
        print(f"    Records: {stats['records']}")
        print(f"    HTML: {stats['html']} ({html_pct:.1f}%) {status}")
        print(f"    Unique behaviors: {len(stats['behaviors'])}")
        print(f"    Sources: {dict(stats['sources'])}")
    
    return results


def audit_behavior_vocabulary() -> Dict[str, Any]:
    """Audit behavior vocabulary and schema."""
    print("\n" + "=" * 60)
    print("4. BEHAVIOR VOCABULARY AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "vocabs": {}}
    
    # Check 5-axis schema
    try:
        from src.ml.qbm_5axis_schema import ARABIC_TO_ID, is_true_behavior
        
        results["vocabs"]["5axis_schema"] = {
            "behaviors": len(ARABIC_TO_ID),
            "sample": list(ARABIC_TO_ID.items())[:5],
        }
        print(f"  5-axis schema: {len(ARABIC_TO_ID)} behaviors defined")
        
        # Verify key behaviors exist
        key_behaviors = ["الصبر", "الشكر", "التوبة", "الإيمان", "الكفر", "الذكر"]
        for beh in key_behaviors:
            if beh in ARABIC_TO_ID:
                print(f"    ✅ {beh} -> {ARABIC_TO_ID[beh]}")
            else:
                print(f"    ❌ Missing: {beh}")
                results["issues"].append(f"Missing key behavior: {beh}")
    except ImportError as e:
        results["issues"].append(f"Cannot import 5-axis schema: {e}")
    
    # Check behavior keywords
    try:
        from src.ml.embedding_pipeline import BEHAVIOR_KEYWORDS
        results["vocabs"]["behavior_keywords"] = len(BEHAVIOR_KEYWORDS)
        print(f"  Behavior keywords: {len(BEHAVIOR_KEYWORDS)} defined")
    except ImportError:
        pass
    
    return results


def audit_graph_data() -> Dict[str, Any]:
    """Audit behavior graph and relationships."""
    print("\n" + "=" * 60)
    print("5. GRAPH DATA AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "graphs": {}}
    
    # Check behavior graph
    graph_files = [
        "behavior_graph_v2.json",
        "annotations/tafsir_relationships.json",
    ]
    
    for filename in graph_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            results["issues"].append(f"Missing: {filename}")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        
        if isinstance(graph, dict):
            if "nodes" in graph:
                # Node-edge format
                nodes = len(graph.get("nodes", []))
                edges = len(graph.get("edges", []))
                results["graphs"][filename] = {"nodes": nodes, "edges": edges}
                print(f"  {filename}: {nodes} nodes, {edges} edges")
            else:
                # Relationship format
                total_rels = sum(len(v) if isinstance(v, list) else len(v) for v in graph.values())
                results["graphs"][filename] = {"relationship_types": len(graph), "total_relations": total_rels}
                print(f"  {filename}: {len(graph)} relationship types, {total_rels} relations")
    
    return results


def audit_bm25_indexes() -> Dict[str, Any]:
    """Audit BM25 indexes."""
    print("\n" + "=" * 60)
    print("6. BM25 INDEX AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "indexes": {}}
    
    index_dir = DATA_DIR / "indexes" / "tafsir"
    if not index_dir.exists():
        results["status"] = "ERROR"
        results["issues"].append("Index directory not found")
        return results
    
    expected_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "muyassar", "baghawi"]
    total_docs = 0
    
    for source in expected_sources:
        index_file = index_dir / f"{source}.json"
        if not index_file.exists():
            results["issues"].append(f"Missing index: {source}")
            continue
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        docs = len(index.get("documents", []))
        results["indexes"][source] = docs
        total_docs += docs
        print(f"  {source}: {docs} documents")
    
    print(f"  Total: {total_docs} documents")
    results["total_documents"] = total_docs
    
    return results


def audit_embeddings() -> Dict[str, Any]:
    """Audit embedding models and data."""
    print("\n" + "=" * 60)
    print("7. EMBEDDINGS AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "models": {}}
    
    # Check model registry
    registry_file = DATA_DIR / "models" / "registry.json"
    if registry_file.exists():
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        active = registry.get("active_model", "none")
        models = registry.get("models", {})
        
        print(f"  Active model: {active}")
        print(f"  Registered models: {len(models)}")
        
        for name, info in models.items():
            acc = info.get("accuracy", 0)
            status = "✅" if acc >= 0.75 else "⚠️"
            print(f"    {status} {name}: {acc*100:.1f}% accuracy")
        
        results["models"] = {
            "active": active,
            "count": len(models),
            "best_accuracy": max((m.get("accuracy", 0) for m in models.values()), default=0),
        }
    else:
        results["issues"].append("Model registry not found")
    
    # Check evaluation benchmark
    benchmark_file = DATA_DIR / "evaluation" / "semantic_similarity_gold.jsonl"
    if benchmark_file.exists():
        count = sum(1 for line in open(benchmark_file, 'r', encoding='utf-8') if line.strip())
        print(f"  Evaluation benchmark: {count} pairs")
        results["benchmark_pairs"] = count
    
    return results


def audit_data_connections() -> Dict[str, Any]:
    """Audit connections between data sources."""
    print("\n" + "=" * 60)
    print("8. DATA CONNECTIONS AUDIT")
    print("=" * 60)
    
    results = {"status": "OK", "issues": [], "connections": {}}
    
    # Check Quran -> Tafsir alignment
    print("  Checking Quran <-> Tafsir alignment...")
    
    quran_file = DATA_DIR / "quran" / "quran_full.json"
    if quran_file.exists():
        with open(quran_file, 'r', encoding='utf-8') as f:
            quran = json.load(f)
        
        quran_refs = set()
        for surah_num, surah_data in quran.items():
            for ayah_num in surah_data.get("verses", {}).keys():
                quran_refs.add(f"{surah_num}:{ayah_num}")
        
        # Check tafsir coverage
        tafsir_dir = DATA_DIR / "tafsir"
        for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
            filepath = tafsir_dir / f"{source}.ar.jsonl"
            if filepath.exists():
                tafsir_refs = set()
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            ref = record.get("reference", {})
                            tafsir_refs.add(f"{ref.get('surah')}:{ref.get('ayah')}")
                
                coverage = len(tafsir_refs & quran_refs) / len(quran_refs) * 100
                print(f"    {source}: {coverage:.1f}% Quran coverage")
    
    # Check Behavior -> Tafsir alignment
    print("  Checking Behavior <-> Tafsir alignment...")
    
    ann_file = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"
    if ann_file.exists():
        behavior_sources = Counter()
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    behavior_sources[record.get("source", "unknown")] += 1
        
        for source, count in behavior_sources.most_common():
            print(f"    {source}: {count} behavioral annotations")
    
    return results


def run_full_audit():
    """Run complete data audit."""
    print("=" * 60)
    print("QBM COMPREHENSIVE DATA AUDIT")
    print("=" * 60)
    print("Perspective: Data Engineer + Arabic/Quranic Expert")
    
    all_results = {}
    all_issues = []
    
    # Run all audits
    all_results["quran"] = audit_quran_data()
    all_results["tafsir"] = audit_tafsir_data()
    all_results["annotations"] = audit_behavioral_annotations()
    all_results["vocabulary"] = audit_behavior_vocabulary()
    all_results["graph"] = audit_graph_data()
    all_results["bm25"] = audit_bm25_indexes()
    all_results["embeddings"] = audit_embeddings()
    all_results["connections"] = audit_data_connections()
    
    # Collect all issues
    for category, results in all_results.items():
        for issue in results.get("issues", []):
            all_issues.append(f"[{category}] {issue}")
    
    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    
    if all_issues:
        print(f"\n⚠️ Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All data sources are clean and properly connected!")
    
    # Save audit report
    report_file = DATA_DIR / "audit_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to: {report_file}")
    
    return all_results, all_issues


if __name__ == "__main__":
    run_full_audit()
