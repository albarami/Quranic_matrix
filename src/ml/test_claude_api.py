"""
Test the full RAG system with Azure OpenAI API.
This is the REAL test - does the system produce a good answer?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml"))

from hybrid_rag_system import HybridRAGSystem

def test_query(system, query):
    """Test a single query."""
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    
    result = system.answer(query, use_api=True)
    
    print("\nANSWER:")
    print("-" * 70)
    print(result["answer"][:2000] + "..." if len(result["answer"]) > 2000 else result["answer"])
    
    print("\n" + "-" * 70)
    print(f"Behaviors: {result['evidence']['behaviors_detected']}")
    print(f"Candidates: {result['evidence']['candidates_retrieved']}")
    print(f"Time: {result['processing_time_ms']}ms")
    
    return result


def test_robustness():
    print("=" * 70)
    print("ROBUSTNESS TEST - 3 DIVERSE QUERIES")
    print("=" * 70)
    
    print("\nInitializing system...")
    system = HybridRAGSystem()
    
    queries = [
        "ما هي علامات النفاق في القرآن؟",
        "كيف يحقق المؤمن التوكل على الله؟",
    ]
    
    results = []
    for query in queries:
        result = test_query(system, query)
        results.append(result)
    
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    test_robustness()
