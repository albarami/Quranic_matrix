"""
Test RAG System with Real Queries

Purpose: Evaluate actual output quality, not just embedding metrics.
The real test is whether the system produces useful answers.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

# Load the trained embeddings
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "annotations" / "tafsir_behavioral_5axis.jsonl"


def load_embeddings():
    """Load the trained QBM embeddings."""
    model_path = MODELS_DIR / "qbm-embeddings-enterprise"
    if model_path.exists():
        return SentenceTransformer(str(model_path))
    else:
        print(f"Model not found at {model_path}")
        return None


def load_annotations():
    """Load behavioral annotations."""
    annotations = []
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def semantic_search(model, query: str, annotations: list, top_k: int = 10):
    """Search annotations using semantic similarity."""
    query_emb = model.encode(query)
    
    results = []
    for ann in annotations:
        text = ann.get("context", "")
        if len(text) < 20:
            continue
        
        text_emb = model.encode(text)
        sim = np.dot(query_emb, text_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(text_emb))
        
        results.append({
            "behavior_id": ann.get("behavior_id", ""),
            "behavior_name": ann.get("behavior_name_ar", ""),
            "text": text[:200],
            "surah": ann.get("surah", ""),
            "ayah": ann.get("ayah", ""),
            "similarity": float(sim),
        })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def test_query(model, annotations, query: str):
    """Test a single query and display results."""
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    
    results = semantic_search(model, query, annotations, top_k=10)
    
    print("\nTop 10 Results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['behavior_id']}] {r['behavior_name']} (sim: {r['similarity']:.3f})")
        print(f"   Surah {r['surah']}:{r['ayah']}")
        print(f"   {r['text'][:150]}...")
    
    # Check if results are relevant
    print("\n" + "-" * 40)
    print("RELEVANCE CHECK:")
    
    # For الكبر query, check if الكبر results appear
    if "الكبر" in query:
        kibr_count = sum(1 for r in results if "KIBR" in r["behavior_id"] or "كبر" in r["behavior_name"])
        print(f"  الكبر-related results in top 10: {kibr_count}")
        if kibr_count >= 3:
            print("  ✅ Good - الكبر behavior found")
        else:
            print("  ⚠️ May need improvement")
    
    # Check for أكبر false positives
    akbar_false_positives = sum(1 for r in results if "أكبر" in r["text"] and "KIBR" not in r["behavior_id"])
    if akbar_false_positives > 0:
        print(f"  أكبر false positives: {akbar_false_positives}")
    else:
        print("  ✅ No أكبر false positives")
    
    return results


def main():
    print("=" * 70)
    print("REAL QUERY TEST - Evaluating Output Quality")
    print("=" * 70)
    
    print("\nLoading embeddings...")
    model = load_embeddings()
    if model is None:
        return
    
    print("Loading annotations...")
    annotations = load_annotations()
    print(f"Loaded {len(annotations):,} annotations")
    
    # Test queries
    queries = [
        "ما علاقة الكبر بقسوة القلب؟",
        "كيف يؤدي النفاق إلى الكفر؟",
        "ما هي ثمرات الصبر في القرآن؟",
    ]
    
    for query in queries:
        test_query(model, annotations, query)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
