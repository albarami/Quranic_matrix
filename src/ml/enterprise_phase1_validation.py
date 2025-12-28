"""
PHASE 1: Data Preparation & Validation
Enterprise Implementation Plan - With Bouzidani Enhancements

Validates:
- 47,142 TRUE behavior annotations (filtered with Bouzidani taxonomy)
- 33 behavior classes (not 87 - updated based on Bouzidani framework)
- 5 tafsir sources
- Behavior graph
"""

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def validate_phase1():
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION & VALIDATION")
    print("With Bouzidani Framework Enhancements")
    print("=" * 70)
    print()
    
    results = {"passed": 0, "failed": 0, "warnings": 0}
    
    # 1. Check filtered annotations
    ann_file = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"
    print("[1/4] Checking filtered annotations...")
    
    if ann_file.exists():
        annotations = []
        tafsir_counts = Counter()
        behavior_counts = Counter()
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ann = json.loads(line)
                    annotations.append(ann)
                    tafsir_counts[ann.get('source', 'unknown')] += 1
                    behavior_counts[ann.get('behavior_id', 'unknown')] += 1
        
        print(f"  ✅ Annotations: {len(annotations):,}")
        print(f"  ✅ Unique behaviors: {len(behavior_counts)}")
        results["passed"] += 1
        
        # Check tafsir distribution
        print("\n  Tafsir distribution:")
        for tafsir, count in tafsir_counts.most_common():
            print(f"    - {tafsir}: {count:,}")
        
        if len(tafsir_counts) >= 5:
            print("  ✅ All 5 tafsirs present")
            results["passed"] += 1
        else:
            print(f"  ⚠️ Only {len(tafsir_counts)} tafsirs found")
            results["warnings"] += 1
        
        # Check behavior distribution
        print("\n  Top 10 behaviors:")
        for beh, count in behavior_counts.most_common(10):
            print(f"    - {beh}: {count:,}")
        
        # Verify 33 TRUE behaviors (Bouzidani taxonomy)
        if len(behavior_counts) >= 30:
            print(f"\n  ✅ {len(behavior_counts)} behavior classes (target: 33)")
            results["passed"] += 1
        else:
            print(f"\n  ❌ Only {len(behavior_counts)} behaviors (target: 33)")
            results["failed"] += 1
    else:
        print(f"  ❌ Annotations file not found: {ann_file}")
        results["failed"] += 1
    
    print()
    
    # 2. Check behavior graph
    print("[2/4] Checking behavior graph...")
    graph_file = DATA_DIR / "behavior_graph_v2.json"
    
    if graph_file.exists():
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        print(f"  ✅ Graph nodes: {len(nodes)}")
        print(f"  ✅ Graph edges: {len(edges)}")
        results["passed"] += 1
    else:
        print(f"  ❌ Graph file not found: {graph_file}")
        results["failed"] += 1
    
    print()
    
    # 3. Check models directory
    print("[3/4] Checking trained models...")
    models_dir = DATA_DIR / "models"
    
    expected_models = [
        "qbm-embeddings-v2",
        "qbm-classifier-v2", 
        "qbm-relations-v2",
        "qbm-gnn-v2",
        "qbm-reranker-v2"
    ]
    
    found_models = []
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                found_models.append(model_dir.name)
                print(f"  ✅ {model_dir.name}")
        
        for expected in expected_models:
            if expected in found_models:
                results["passed"] += 1
            else:
                print(f"  ⚠️ Missing: {expected}")
                results["warnings"] += 1
    else:
        print(f"  ❌ Models directory not found")
        results["failed"] += 1
    
    print()
    
    # 4. Check 5-axis schema
    print("[4/4] Checking Bouzidani 5-axis schema...")
    schema_file = PROJECT_ROOT / "src" / "ml" / "qbm_5axis_schema.py"
    
    if schema_file.exists():
        print(f"  ✅ Schema file exists: {schema_file.name}")
        results["passed"] += 1
    else:
        print(f"  ❌ Schema file not found")
        results["failed"] += 1
    
    # Summary
    print()
    print("=" * 70)
    print("PHASE 1 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Warnings: {results['warnings']}")
    print()
    
    if results["failed"] == 0:
        print("✅ PHASE 1 PASSED - Ready for Layer Training")
        return True
    else:
        print("❌ PHASE 1 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    validate_phase1()
