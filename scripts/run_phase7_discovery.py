"""
Phase 7: Discovery System

Run this script to test the discovery system components:
1. Semantic search
2. Pattern discovery
3. Cross-reference finder
4. Thematic clustering
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.discovery import (
    SemanticSearchEngine,
    PatternDiscovery,
    CrossReferenceFinder,
    ThematicClustering,
)


def test_semantic_search():
    """Test semantic search engine."""
    print("\n" + "=" * 60)
    print("Testing Semantic Search Engine")
    print("=" * 60)

    engine = SemanticSearchEngine(use_reranker=False)  # Skip reranker for speed

    # Test Arabic query
    query = "الصبر على البلاء"  # Patience in adversity
    print(f"\nQuery: {query}")
    
    results = engine.search(query, top_k=5)
    print(f"\nTop 5 results:")
    for r in results:
        print(f"  {r['rank']}. [{r['source']} {r['surah']}:{r['ayah']}] Score: {r['score']:.4f}")
        print(f"     Type: {r['type']}, Behavior: {r['behavior']}")

    # Test behavior search
    print("\n--- Behavior Search ---")
    behavior_results = engine.search_by_behavior("التوبة", top_k=5)
    print(f"Results for behavior 'التوبة' (repentance):")
    for r in behavior_results[:3]:
        print(f"  [{r['source']} {r['surah']}:{r['ayah']}] Score: {r['score']:.4f}")

    print(f"\nStats: {engine.get_stats()}")


def test_pattern_discovery():
    """Test pattern discovery."""
    print("\n" + "=" * 60)
    print("Testing Pattern Discovery")
    print("=" * 60)

    discovery = PatternDiscovery()

    # Co-occurring behaviors
    print("\n--- Co-occurring Behaviors (same ayah) ---")
    cooccur = discovery.find_cooccurring_behaviors(min_cooccurrence=10, same_ayah=True)
    print(f"Found {len(cooccur)} co-occurring pairs")
    for pair in cooccur[:5]:
        print(f"  {pair['behavior_1']} + {pair['behavior_2']}: {pair['cooccurrence_count']} times")

    # Surah themes
    print("\n--- Surah Themes (top 3 surahs) ---")
    themes = discovery.find_surah_themes(top_n=3)
    for surah in [2, 3, 4]:  # Al-Baqarah, Al-Imran, An-Nisa
        if surah in themes:
            print(f"  Surah {surah}:")
            for t in themes[surah]:
                print(f"    - {t['behavior']}: {t['count']} ({t['percentage']}%)")

    # Cause-effect patterns
    print("\n--- Cause-Effect Patterns ---")
    cause_effect = discovery.find_cause_effect_patterns()
    print(f"Found {len(cause_effect)} cause-effect patterns")
    for pattern in cause_effect[:3]:
        print(f"  {pattern['cause']} -> {pattern['effect']}: {pattern['evidence_count']} evidence")

    print(f"\nStats: {discovery.get_stats()}")


def test_cross_reference():
    """Test cross-reference finder."""
    print("\n" + "=" * 60)
    print("Testing Cross-Reference Finder")
    print("=" * 60)

    finder = CrossReferenceFinder()

    # Find ayat by behavior
    print("\n--- Ayat by Behavior ---")
    behavior = "الإيمان"  # Faith
    ayat = finder.find_ayat_by_behavior(behavior, include_similar=False)
    print(f"Found {len(ayat)} ayat for '{behavior}'")
    for a in ayat[:5]:
        print(f"  {a['surah']}:{a['ayah']} - {a['all_behaviors']}")

    # Find similar ayat
    print("\n--- Similar Ayat ---")
    surah, ayah = 2, 255  # Ayat al-Kursi
    similar = finder.find_similar_ayat(surah, ayah, top_k=5)
    print(f"Ayat similar to {surah}:{ayah}:")
    for s in similar:
        print(f"  {s['surah']}:{s['ayah']} - Similarity: {s['similarity']:.4f}")

    # Cross-references
    print("\n--- Cross-References ---")
    refs = finder.find_cross_references(2, 255, method="both", top_k=3)
    print(f"Semantic refs: {len(refs.get('semantic', []))}")
    print(f"Behavior refs: {len(refs.get('behavior', []))}")

    # Behavior network
    print("\n--- Behavior Network ---")
    network = finder.get_behavior_network()
    print(f"Nodes: {network['stats']['behavior_count']}")
    print(f"Edges: {network['stats']['edge_count']}")

    print(f"\nStats: {finder.get_stats()}")


def test_thematic_clustering():
    """Test thematic clustering."""
    print("\n" + "=" * 60)
    print("Testing Thematic Clustering")
    print("=" * 60)

    clustering = ThematicClustering(n_clusters=15)

    # Run K-means
    print("\n--- K-means Clustering ---")
    result = clustering.cluster_kmeans(n_clusters=15)
    print(f"Silhouette score: {result['silhouette_score']:.4f}")
    print(f"Cluster sizes: {sorted(result['cluster_sizes'].values(), reverse=True)[:5]}...")

    # Get themes
    print("\n--- Cluster Themes ---")
    themes = clustering.get_cluster_themes(top_n=3)
    for cluster_id in list(themes.keys())[:5]:
        theme = themes[cluster_id]
        print(f"  Cluster {cluster_id} ({theme['size']} items): {theme['label']}")
        if theme['top_behaviors']:
            print(f"    Top behaviors: {[b[0] for b in theme['top_behaviors'][:2]]}")

    # Get samples from cluster 0
    print("\n--- Cluster 0 Samples ---")
    samples = clustering.get_cluster_samples(0, n_samples=3, method="centroid")
    for s in samples:
        print(f"  [{s.get('source', '')} {s.get('surah', '')}:{s.get('ayah', '')}] {s.get('type', '')}")

    print(f"\nStats: {clustering.get_stats()}")


def main():
    print("=" * 60)
    print("Phase 7: Discovery System")
    print("=" * 60)

    try:
        test_semantic_search()
    except Exception as e:
        print(f"Semantic search error: {e}")

    try:
        test_pattern_discovery()
    except Exception as e:
        print(f"Pattern discovery error: {e}")

    try:
        test_cross_reference()
    except Exception as e:
        print(f"Cross-reference error: {e}")

    try:
        test_thematic_clustering()
    except Exception as e:
        print(f"Thematic clustering error: {e}")

    print("\n" + "=" * 60)
    print("Phase 7 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
