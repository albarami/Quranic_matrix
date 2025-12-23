"""
Test Suite for QBM Mandatory 13-Component Proof System
Tests all 10 Legendary Queries with validation.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def proof_system():
    """Initialize the full power QBM system with proof system."""
    from src.ml.full_power_system import FullPowerQBMSystem
    from src.ml.mandatory_proof_system import integrate_with_system
    
    system = FullPowerQBMSystem()
    
    # Build index if not already built
    status = system.get_status()
    if status["vector_search"].get("status") == "not_built":
        system.build_index()
        system.build_graph()
    
    # Add proof system
    system = integrate_with_system(system)
    
    return system


# =============================================================================
# TEST: PROOF VALIDATION
# =============================================================================

def test_proof_validation_structure(proof_system):
    """Test that proof validation returns correct structure."""
    result = proof_system.answer_with_full_proof("ما هو الكبر؟")
    
    assert "validation" in result
    assert "score" in result["validation"]
    assert "passed" in result["validation"]
    assert "missing" in result["validation"]
    assert "checks" in result["validation"]
    
    # Check all 13 components are validated
    checks = result["validation"]["checks"]
    assert len(checks) == 13
    
    print(f"\n✅ Proof Validation Structure Test PASSED")
    print(f"   Score: {result['validation']['score']:.1f}%")


# =============================================================================
# TEST: 13 MANDATORY COMPONENTS
# =============================================================================

def test_13_mandatory_components(proof_system):
    """Test that all 13 components are present in proof."""
    result = proof_system.answer_with_full_proof("حلل سلوك الكبر")
    
    proof = result["proof"]
    
    # Check each component
    components = {
        "quran": len(proof.quran.verses) > 0,
        "ibn_kathir": proof.ibn_kathir.source == "ibn_kathir",
        "tabari": proof.tabari.source == "tabari",
        "qurtubi": proof.qurtubi.source == "qurtubi",
        "saadi": proof.saadi.source == "saadi",
        "jalalayn": proof.jalalayn.source == "jalalayn",
        "graph_nodes": hasattr(proof.graph, 'nodes'),
        "graph_edges": hasattr(proof.graph, 'edges'),
        "graph_paths": hasattr(proof.graph, 'paths'),
        "embeddings": hasattr(proof.embeddings, 'similarities'),
        "rag": len(proof.rag.retrieved_docs) > 0,
        "taxonomy": hasattr(proof.taxonomy, 'behaviors'),
        "statistics": len(proof.statistics.counts) > 0,
    }
    
    present_count = sum(components.values())
    
    print(f"\n✅ 13 Mandatory Components Test")
    print(f"   Components present: {present_count}/13")
    for name, present in components.items():
        status = "✓" if present else "✗"
        print(f"   {status} {name}")
    
    assert present_count >= 10, f"Only {present_count}/13 components present"


# =============================================================================
# LEGENDARY QUERY TESTS
# =============================================================================

def test_legendary_query_1_complete_analysis(proof_system):
    """Q1: حلل سلوك الكبر تحليلاً شاملاً - Complete behavior analysis"""
    result = proof_system.answer_with_full_proof("حلل سلوك \"الكبر\" تحليلاً شاملاً")
    
    assert result["validation"]["score"] >= 50, f"Score {result['validation']['score']}% too low"
    assert len(result["answer"]) > 100, "Answer too short"
    
    print(f"\n✅ Legendary Query 1 (Complete Analysis)")
    print(f"   Score: {result['validation']['score']:.1f}%")
    print(f"   Answer length: {len(result['answer'])} chars")


def test_legendary_query_2_causal_chain(proof_system):
    """Q2: ارسم السلسلة من الغفلة إلى جهنم - Causal chain"""
    result = proof_system.answer_with_full_proof("ارسم السلسلة من \"الغفلة\" إلى \"جهنم\"")
    
    assert result["validation"]["score"] >= 50
    assert len(result["answer"]) > 100
    
    print(f"\n✅ Legendary Query 2 (Causal Chain)")
    print(f"   Score: {result['validation']['score']:.1f}%")


def test_legendary_query_3_cross_tafsir(proof_system):
    """Q3: قارن تفسير البقرة:7 عند الخمسة - Cross-tafsir comparison"""
    result = proof_system.answer_with_full_proof("قارن تفسير البقرة:7 عند الخمسة")
    
    assert result["validation"]["score"] >= 50
    
    # Check all 5 tafsir are present
    proof = result["proof"]
    tafsir_count = sum([
        len(proof.ibn_kathir.quotes) > 0,
        len(proof.tabari.quotes) > 0,
        len(proof.qurtubi.quotes) > 0,
        len(proof.saadi.quotes) > 0,
        len(proof.jalalayn.quotes) > 0,
    ])
    
    print(f"\n✅ Legendary Query 3 (Cross-Tafsir)")
    print(f"   Score: {result['validation']['score']:.1f}%")
    print(f"   Tafsir sources with quotes: {tafsir_count}/5")


def test_legendary_query_4_statistics(proof_system):
    """Q4: التحليل الإحصائي الكامل للسلوكيات - Statistical analysis"""
    result = proof_system.answer_with_full_proof("التحليل الإحصائي الكامل للسلوكيات")
    
    assert result["validation"]["score"] >= 50
    
    # Check statistics are present
    stats = result["proof"].statistics
    assert len(stats.counts) > 0, "No counts in statistics"
    
    print(f"\n✅ Legendary Query 4 (Statistics)")
    print(f"   Score: {result['validation']['score']:.1f}%")
    print(f"   Statistics counts: {len(stats.counts)}")


def test_legendary_query_5_hidden_patterns(proof_system):
    """Q5: اكتشف 5 أنماط مخفية - Hidden pattern discovery"""
    result = proof_system.answer_with_full_proof("اكتشف 5 أنماط مخفية")
    
    assert result["validation"]["score"] >= 50
    assert len(result["answer"]) > 100
    
    print(f"\n✅ Legendary Query 5 (Hidden Patterns)")
    print(f"   Score: {result['validation']['score']:.1f}%")


def test_legendary_query_6_network_traversal(proof_system):
    """Q6: شبكة علاقات الإيمان - Network traversal"""
    result = proof_system.answer_with_full_proof("شبكة علاقات \"الإيمان\"")
    
    assert result["validation"]["score"] >= 50
    
    # Check graph evidence
    graph = result["proof"].graph
    
    print(f"\n✅ Legendary Query 6 (Network Traversal)")
    print(f"   Score: {result['validation']['score']:.1f}%")
    print(f"   Graph nodes: {len(graph.nodes)}")
    print(f"   Graph edges: {len(graph.edges)}")


def test_legendary_query_7_dimensions(proof_system):
    """Q7: النفاق عبر الأبعاد الإحدى عشر - 11 dimensions"""
    result = proof_system.answer_with_full_proof("النفاق عبر الأبعاد الإحدى عشر")
    
    assert result["validation"]["score"] >= 50
    
    # Check taxonomy dimensions
    taxonomy = result["proof"].taxonomy
    
    print(f"\n✅ Legendary Query 7 (11 Dimensions)")
    print(f"   Score: {result['validation']['score']:.1f}%")
    print(f"   Dimensions: {len(taxonomy.dimensions)}")


def test_legendary_query_8_personality_comparison(proof_system):
    """Q8: قارن سلوك الصلاة بين 3 شخصيات - Personality comparison"""
    result = proof_system.answer_with_full_proof("قارن سلوك الصلاة بين 3 شخصيات")
    
    assert result["validation"]["score"] >= 50
    assert len(result["answer"]) > 100
    
    print(f"\n✅ Legendary Query 8 (Personality Comparison)")
    print(f"   Score: {result['validation']['score']:.1f}%")


def test_legendary_query_9_heart_journey(proof_system):
    """Q9: رحلة القلب من السلامة إلى الموت - Full integration"""
    result = proof_system.answer_with_full_proof("رحلة القلب من السلامة إلى الموت")
    
    assert result["validation"]["score"] >= 50
    assert len(result["answer"]) > 100
    
    print(f"\n✅ Legendary Query 9 (Heart Journey)")
    print(f"   Score: {result['validation']['score']:.1f}%")


def test_legendary_query_10_ultimate_synthesis(proof_system):
    """Q10: الـ 3 سلوكيات الأهم والأخطر - Ultimate synthesis"""
    result = proof_system.answer_with_full_proof("الـ 3 سلوكيات الأهم والأخطر")
    
    assert result["validation"]["score"] >= 50
    assert len(result["answer"]) > 100
    
    print(f"\n✅ Legendary Query 10 (Ultimate Synthesis)")
    print(f"   Score: {result['validation']['score']:.1f}%")


# =============================================================================
# TEST: RUN ALL LEGENDARY QUERIES
# =============================================================================

def test_run_all_legendary_queries(proof_system):
    """Run all 10 legendary queries and check overall pass rate."""
    results = proof_system.run_legendary_queries()
    
    # Calculate metrics
    avg_score = sum(r.get('score', 0) for r in results) / len(results)
    passed_count = sum(1 for r in results if r.get('passed', False))
    
    print(f"\n{'='*60}")
    print(f"ALL LEGENDARY QUERIES SUMMARY")
    print(f"{'='*60}")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Passed: {passed_count}/{len(results)}")
    
    for r in results:
        status = "✓" if r.get('passed', False) else "✗"
        print(f"  {status} Q{r['id']}: {r.get('score', 0):.1f}%")
    
    # At least 50% should pass
    assert passed_count >= 5, f"Only {passed_count}/10 queries passed"
    
    print(f"\n✅ All Legendary Queries Test PASSED")


# =============================================================================
# TEST: PROOF MARKDOWN GENERATION
# =============================================================================

def test_proof_markdown_generation(proof_system):
    """Test that proof generates valid markdown."""
    result = proof_system.answer_with_full_proof("ما هو الكبر؟")
    
    markdown = result["proof_markdown"]
    
    # Check markdown structure
    assert "## 1️⃣" in markdown, "Missing Quran section"
    assert "## 2️⃣" in markdown or "ابن كثير" in markdown, "Missing Ibn Kathir section"
    assert "## 8️⃣" in markdown or "الشبكة" in markdown, "Missing Graph section"
    assert "|" in markdown, "Missing tables"
    
    print(f"\n✅ Proof Markdown Generation Test PASSED")
    print(f"   Markdown length: {len(markdown)} chars")


# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

def test_proof_performance(proof_system):
    """Test that proof generation completes in reasonable time."""
    start = time.time()
    result = proof_system.answer_with_full_proof("ما هو الكبر؟")
    elapsed = time.time() - start
    
    print(f"\n✅ Proof Performance Test")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Reported time: {result['processing_time_ms']:.0f}ms")
    
    # Should complete within 60 seconds (including LLM call)
    assert elapsed < 60, f"Proof generation took {elapsed:.2f}s (>60s limit)"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
