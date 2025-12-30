#!/usr/bin/env python3
"""
Phase 6: Audit Tafsir Coverage for Concept Index Verses

Ensures tafsir coverage is correct and auditable.

For each verse in the concept index:
1. Check which tafsir sources have entries
2. Report coverage percentage
3. Flag missing entries

Outputs:
- artifacts/tafsir_coverage_report.json: Coverage report

Exit codes:
- 0: Success
- 1: Failure
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configured tafsir sources
CONFIGURED_SOURCES = [
    "ibn_kathir",
    "tabari",
    "qurtubi",
    "saadi",
    "jalalayn",
    "muyassar",
    "baghawi"
]


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tafsir_index() -> Dict[str, Set[str]]:
    """
    Load tafsir index showing which verses have tafsir for each source.

    Returns:
        {source: {verse_key, ...}, ...}
    """
    tafsir_dir = Path("data/tafsir")
    index = {}

    for source in CONFIGURED_SOURCES:
        filepath = tafsir_dir / f"{source}.ar.jsonl"
        if not filepath.exists():
            print(f"  Warning: {filepath} not found")
            continue

        index[source] = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        ref = record.get("reference", {})
                        surah = ref.get("surah")
                        ayah = ref.get("ayah")
                        if surah and ayah:
                            index[source].add(f"{surah}:{ayah}")
                    except json.JSONDecodeError:
                        continue

    return index


def get_concept_index_verses() -> Set[str]:
    """Get all unique verse_keys from concept_index_v3."""
    verse_keys = set()
    index_path = Path("data/evidence/concept_index_v3.jsonl")

    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        return verse_keys

    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            for verse in entry.get("verses", []):
                verse_keys.add(verse.get("verse_key", ""))

    return verse_keys


def audit_tafsir_for_verse(verse_key: str, tafsir_index: Dict[str, Set[str]]) -> Dict[str, Any]:
    """
    Check tafsir availability for a verse.

    Returns:
        {
            "verse_key": "2:45",
            "available_sources": ["ibn_kathir", "tabari", ...],
            "missing_sources": ["qurtubi"],
            "coverage": 0.857
        }
    """
    available = []
    missing = []

    for source in CONFIGURED_SOURCES:
        if source in tafsir_index and verse_key in tafsir_index[source]:
            available.append(source)
        else:
            missing.append(source)

    coverage = len(available) / len(CONFIGURED_SOURCES) if CONFIGURED_SOURCES else 0

    return {
        "verse_key": verse_key,
        "available_sources": available,
        "missing_sources": missing,
        "coverage": round(coverage, 3)
    }


def main() -> int:
    """Audit tafsir coverage."""
    print("=" * 60)
    print("Phase 6: Tafsir Coverage Audit")
    print("=" * 60)

    try:
        # Load tafsir index
        print("\nLoading tafsir sources...")
        tafsir_index = load_tafsir_index()

        available_sources = list(tafsir_index.keys())
        print(f"  Available sources: {available_sources}")
        for source, verses in tafsir_index.items():
            print(f"    {source}: {len(verses)} verses")

        # Get concept index verses
        print("\nLoading concept index verses...")
        concept_verses = get_concept_index_verses()
        print(f"  Found {len(concept_verses)} unique verses")

        # Audit each verse
        print("\nAuditing tafsir coverage...")
        results = []
        total_coverage = 0

        for vk in sorted(concept_verses, key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1]))):
            result = audit_tafsir_for_verse(vk, tafsir_index)
            results.append(result)
            total_coverage += result["coverage"]

        # Calculate statistics
        avg_coverage = total_coverage / len(results) if results else 0
        full_coverage = len([r for r in results if r["coverage"] == 1.0])
        partial_coverage = len([r for r in results if 0 < r["coverage"] < 1.0])
        no_coverage = len([r for r in results if r["coverage"] == 0])

        # Get missing sources breakdown
        source_missing_count = {source: 0 for source in CONFIGURED_SOURCES}
        for result in results:
            for missing in result["missing_sources"]:
                source_missing_count[missing] = source_missing_count.get(missing, 0) + 1

        # Check key patience verses
        patience_verses = ["2:45", "2:153", "3:200", "39:10", "103:3"]
        patience_coverage = {}
        for pv in patience_verses:
            result = audit_tafsir_for_verse(pv, tafsir_index)
            patience_coverage[pv] = {
                "coverage": result["coverage"],
                "available": len(result["available_sources"]),
                "missing": result["missing_sources"]
            }

        # Generate report
        report = {
            "phase": 6,
            "description": "Tafsir Coverage Audit Report",
            "configured_sources": CONFIGURED_SOURCES,
            "available_sources": available_sources,
            "statistics": {
                "total_verses_audited": len(results),
                "average_coverage": round(avg_coverage, 3),
                "full_coverage_count": full_coverage,
                "partial_coverage_count": partial_coverage,
                "no_coverage_count": no_coverage,
                "full_coverage_rate": f"{(full_coverage / len(results) * 100):.1f}%"
            },
            "source_coverage": {
                source: {
                    "verses_available": len(tafsir_index.get(source, set())),
                    "verses_missing_in_index": source_missing_count[source],
                    "coverage_rate": f"{((len(results) - source_missing_count[source]) / len(results) * 100):.1f}%"
                }
                for source in CONFIGURED_SOURCES
            },
            "patience_verses": patience_coverage,
            "sample_results": results[:50]
        }

        # Save report
        report_path = Path("artifacts/tafsir_coverage_report.json")
        save_json(report_path, report)
        print(f"\nSaved report to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("TAFSIR COVERAGE SUMMARY")
        print("=" * 60)
        print(f"Total verses audited: {len(results)}")
        print(f"Average coverage: {avg_coverage:.1%}")
        print(f"Full coverage: {full_coverage} ({full_coverage / len(results) * 100:.1f}%)")
        print(f"Partial coverage: {partial_coverage}")
        print(f"No coverage: {no_coverage}")

        print(f"\nSource breakdown:")
        for source, stats in report["source_coverage"].items():
            print(f"  {source}: {stats['coverage_rate']} coverage")

        print(f"\nPatience verses coverage:")
        for pv, stats in patience_coverage.items():
            status = "FULL" if stats["coverage"] == 1.0 else f"{stats['coverage']:.0%}"
            print(f"  {pv}: {status} ({stats['available']}/{len(CONFIGURED_SOURCES)} sources)")

        print("\n" + "=" * 60)

        # Success - we're just auditing, not failing on low coverage
        # (some verses may legitimately not have tafsir in all sources)
        if avg_coverage >= 0.5:
            print("Phase 6: TAFSIR AUDIT COMPLETE")
            print("=" * 60)
            return 0
        else:
            print(f"Phase 6: WARNING - Low average coverage ({avg_coverage:.1%})")
            print("=" * 60)
            return 0  # Still return 0, this is an audit not a gate

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
