#!/usr/bin/env python3
"""
Phase 4: Rebuild Concept Index v3 with Validation

Replaces corrupt concept_index_v2 with validated v3.

For each behavior:
1. If lexical mode: retrieve verses via lexeme index
2. If annotation mode: retrieve verses via annotation tables (future)
3. Produce unified verse set with evidence provenance
4. Validate 100% of verses match policy

Outputs:
- data/evidence/concept_index_v3.jsonl: New validated index
- artifacts/concept_index_v3_report.json: Build statistics

Exit codes:
- 0: Success
- 1: Failure
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.quran_store import get_quran_store
from src.data.lexeme_search import search_token_pattern, load_lexeme_index
from src.data.behavior_registry import get_behavior_registry
from src.models.evidence_policy import EvidenceMode


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_lexical_pattern(behavior) -> str:
    """
    Build regex pattern from behavior's lexical spec.

    Combines forms and synonyms into a single pattern.
    """
    lexical_spec = behavior.evidence_policy.lexical_spec
    if not lexical_spec:
        return ""

    all_patterns = []

    # Add forms
    for form in lexical_spec.forms:
        if form:
            all_patterns.append(re.escape(form))

    # Add synonyms
    for syn in lexical_spec.synonyms:
        if syn:
            all_patterns.append(re.escape(syn))

    # If no patterns, use the Arabic label as fallback
    if not all_patterns:
        label = behavior.label_ar
        if label:
            all_patterns.append(re.escape(label))

    return "|".join(all_patterns) if all_patterns else ""


def find_lexical_evidence(behavior, store) -> List[Dict[str, Any]]:
    """
    Find all verses with lexical evidence for a behavior.

    Returns list of verse entries with evidence details.
    """
    pattern = build_lexical_pattern(behavior)
    if not pattern:
        return []

    # Search the lexeme index
    try:
        results = search_token_pattern(pattern)
    except Exception as e:
        print(f"  Warning: Search error for {behavior.behavior_id}: {e}")
        return []

    # Group by verse
    by_verse: Dict[str, Dict[str, Any]] = {}

    for match in results:
        vk = match["verse_key"]

        if vk not in by_verse:
            verse = store.get_verse(vk)
            if not verse:
                continue

            surah, ayah = vk.split(":")
            by_verse[vk] = {
                "verse_key": vk,
                "surah": int(surah),
                "ayah": int(ayah),
                "text_uthmani": verse.text_uthmani,
                "evidence": [],
                "directness": "direct",  # Lexical matches are direct
                "provenance": "lexeme_index_v1"
            }

        # Add evidence entry
        by_verse[vk]["evidence"].append({
            "type": "lexical",
            "matched_token": match.get("matched_token", ""),
            "token_index": match.get("token_index", -1),
            "pattern": pattern[:50]  # Truncate for readability
        })

    return list(by_verse.values())


def validate_verse_against_pattern(verse_key: str, pattern: str, store) -> bool:
    """
    Validate that a verse actually matches the lexical pattern.
    """
    verse = store.get_verse(verse_key)
    if not verse:
        return False

    regex = re.compile(pattern)

    # Check normalized tokens
    for token in verse.tokens_norm:
        if regex.search(token):
            return True

    # Check full normalized text
    if regex.search(verse.text_norm):
        return True

    return False


def build_concept_entry(behavior, store) -> Dict[str, Any]:
    """
    Build complete concept index entry for a behavior.

    Returns entry with verses, statistics, and validation info.
    """
    behavior_id = behavior.behavior_id
    policy = behavior.evidence_policy

    verses = []
    validation_errors = []

    # Handle based on evidence mode
    if policy.mode in [EvidenceMode.LEXICAL, EvidenceMode.HYBRID]:
        lexical_verses = find_lexical_evidence(behavior, store)

        # Validate each verse if lexical is required
        if policy.lexical_required:
            pattern = build_lexical_pattern(behavior)
            if pattern:
                for verse in lexical_verses:
                    is_valid = validate_verse_against_pattern(
                        verse["verse_key"], pattern, store
                    )
                    if not is_valid:
                        validation_errors.append({
                            "verse_key": verse["verse_key"],
                            "reason": "no_pattern_match"
                        })

        verses.extend(lexical_verses)

    if policy.mode in [EvidenceMode.ANNOTATION, EvidenceMode.HYBRID]:
        # TODO: Implement annotation lookup when annotation store is ready
        pass

    # Sort verses by surah:ayah
    verses.sort(key=lambda v: (v["surah"], v["ayah"]))

    # Calculate statistics
    lexical_count = len([v for v in verses if any(
        e.get("type") == "lexical" for e in v.get("evidence", [])
    )])
    annotation_count = len([v for v in verses if any(
        e.get("type") == "annotation" for e in v.get("evidence", [])
    )])
    direct_count = len([v for v in verses if v.get("directness") == "direct"])
    indirect_count = len([v for v in verses if v.get("directness") == "indirect"])

    return {
        "concept_id": behavior_id,
        "term": behavior.label_ar,
        "term_en": behavior.label_en,
        "category": behavior.category,
        "entity_type": "BEHAVIOR",
        "evidence_policy_mode": policy.mode.value,
        "lexical_required": policy.lexical_required,
        "verses": verses,
        "statistics": {
            "total_verses": len(verses),
            "lexical_mentions": lexical_count,
            "annotation_mentions": annotation_count,
            "direct_count": direct_count,
            "indirect_count": indirect_count,
            "validation_errors": len(validation_errors)
        },
        "validation": {
            "passed": len(validation_errors) == 0,
            "errors": validation_errors[:10]  # Sample errors
        }
    }


def main() -> int:
    """Build concept index v3."""
    print("=" * 60)
    print("Phase 4: Rebuild Concept Index v3")
    print("=" * 60)

    try:
        # Load dependencies
        print("\nLoading Quran store...")
        store = get_quran_store()
        print(f"  Loaded {store.total_verses} verses")

        print("\nLoading behavior registry...")
        registry = get_behavior_registry()
        behaviors = registry.get_all()
        print(f"  Loaded {len(behaviors)} behaviors")

        print("\nLoading lexeme index...")
        lexeme_index = load_lexeme_index()
        print(f"  Loaded {lexeme_index['total_unique_tokens']} unique tokens")

        # Build index
        print("\nBuilding concept index v3...")
        entries = []
        total_verses = 0
        total_errors = 0
        by_mode = {"lexical": 0, "annotation": 0, "hybrid": 0}

        for i, behavior in enumerate(behaviors):
            if (i + 1) % 10 == 0:
                print(f"  Processing behavior {i + 1}/{len(behaviors)}...")

            entry = build_concept_entry(behavior, store)
            entries.append(entry)

            total_verses += entry["statistics"]["total_verses"]
            total_errors += entry["statistics"]["validation_errors"]
            mode = entry["evidence_policy_mode"]
            by_mode[mode] = by_mode.get(mode, 0) + 1

        # Sort entries by concept_id
        entries.sort(key=lambda e: e["concept_id"])

        # Write JSONL
        output_path = Path("data/evidence/concept_index_v3.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Generate report
        report = {
            "phase": 4,
            "description": "Concept Index v3 Build Report",
            "statistics": {
                "total_behaviors": len(entries),
                "total_verses_indexed": total_verses,
                "total_validation_errors": total_errors,
                "by_mode": by_mode,
                "average_verses_per_behavior": round(total_verses / len(entries), 1) if entries else 0
            },
            "validation": {
                "all_passed": total_errors == 0,
                "behaviors_with_errors": [
                    e["concept_id"] for e in entries
                    if not e["validation"]["passed"]
                ][:20]  # Sample
            },
            "sample_entries": [
                {
                    "concept_id": e["concept_id"],
                    "term": e["term"],
                    "total_verses": e["statistics"]["total_verses"],
                    "validation_passed": e["validation"]["passed"]
                }
                for e in entries[:10]
            ]
        }

        report_path = Path("artifacts/concept_index_v3_report.json")
        save_json(report_path, report)
        print(f"Saved report to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("CONCEPT INDEX V3 BUILD SUMMARY")
        print("=" * 60)
        print(f"Total behaviors: {len(entries)}")
        print(f"Total verses indexed: {total_verses}")
        print(f"Average verses per behavior: {report['statistics']['average_verses_per_behavior']}")
        print(f"By mode: {by_mode}")
        print(f"Validation errors: {total_errors}")

        # Highlight patience behavior
        patience_entry = next(
            (e for e in entries if "PATIENCE" in e["concept_id"].upper() or e["term"] == "الصبر"),
            None
        )
        if patience_entry:
            print(f"\nPatience (sabr) stats:")
            print(f"  Total verses: {patience_entry['statistics']['total_verses']}")
            print(f"  Lexical mentions: {patience_entry['statistics']['lexical_mentions']}")
            print(f"  Validation passed: {patience_entry['validation']['passed']}")

        print("\n" + "=" * 60)
        if total_errors == 0:
            print("Phase 4: CONCEPT INDEX V3 BUILT SUCCESSFULLY")
            print("=" * 60)
            return 0
        else:
            print(f"Phase 4: WARNING - {total_errors} validation errors")
            print("=" * 60)
            return 0  # Don't fail, validation is in Phase 5

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
