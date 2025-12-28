"""Analyze coverage per surah to understand why some show different percentages."""
import json
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "exports" / "qbm_silver_20251221.json"

# Known ayah counts per surah (from Quran)
SURAH_AYAT_COUNTS = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

spans = data["spans"]

# Count unique ayat per surah in our data
surah_ayat_covered = defaultdict(set)
surah_span_count = defaultdict(int)

for s in spans:
    ref = s.get("reference", {})
    surah = ref.get("surah")
    ayah = ref.get("ayah")
    if surah and ayah:
        surah_ayat_covered[surah].add(ayah)
        surah_span_count[surah] += 1

print("SURAH COVERAGE ANALYSIS")
print("=" * 80)
print(f"{'Surah':<6} {'Name':<20} {'Covered':<10} {'Total':<10} {'Coverage':<10} {'Spans':<10}")
print("-" * 80)

low_coverage = []
full_coverage = []

for surah_num in range(1, 115):
    total_ayat = SURAH_AYAT_COUNTS.get(surah_num, 0)
    covered_ayat = len(surah_ayat_covered.get(surah_num, set()))
    span_count = surah_span_count.get(surah_num, 0)
    coverage = (covered_ayat / total_ayat * 100) if total_ayat > 0 else 0
    
    # Get surah name from first span
    surah_name = ""
    for s in spans:
        if s.get("reference", {}).get("surah") == surah_num:
            surah_name = s.get("reference", {}).get("surah_name", "")
            break
    
    print(f"{surah_num:<6} {surah_name:<20} {covered_ayat:<10} {total_ayat:<10} {coverage:.1f}%{'':<5} {span_count:<10}")
    
    if coverage < 100:
        low_coverage.append((surah_num, surah_name, coverage, covered_ayat, total_ayat))
    else:
        full_coverage.append((surah_num, surah_name))

print()
print("=" * 80)
print(f"SUMMARY:")
print(f"  Surahs with 100% coverage: {len(full_coverage)}")
print(f"  Surahs with <100% coverage: {len(low_coverage)}")
print()

if low_coverage:
    print("SURAHS WITH INCOMPLETE COVERAGE:")
    for surah_num, name, cov, covered, total in sorted(low_coverage, key=lambda x: x[2]):
        missing = total - covered
        print(f"  Surah {surah_num} ({name}): {cov:.1f}% - missing {missing} ayat")
