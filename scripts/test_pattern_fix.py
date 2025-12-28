"""Test pattern discovery fix."""
import sys
sys.path.insert(0, ".")

from src.ai.discovery import PatternDiscovery

p = PatternDiscovery()

# Test co-occurrence
cooc = p.find_cooccurring_behaviors(min_cooccurrence=10)
print(f"Co-occurring pairs (>=10): {len(cooc)}")
print("Top 5:")
for c in cooc[:5]:
    print(f"  {c['behavior_1']} + {c['behavior_2']}: {c['cooccurrence_count']}")

# Test surah themes
themes = p.find_surah_themes(top_n=3)
print(f"\nSurah 2 themes:")
for t in themes.get(2, []):
    print(f"  {t['behavior']}: {t['count']} ({t['percentage']}%)")

# Test behavior distribution
dist = p.get_behavior_distribution()
print(f"\nTop 5 behaviors:")
for b, count in list(dist.items())[:5]:
    print(f"  {b}: {count}")
