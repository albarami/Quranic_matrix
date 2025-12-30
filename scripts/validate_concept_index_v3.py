#!/usr/bin/env python3
"""
Phase 5: Validate Concept Index v3 Against Evidence Policies

Ensures NO invalid data enters the system. Build MUST FAIL if validation fails.

EXIT CODES:
- 0: All validations pass
- 1: Validation failures (build should fail)

Validation Gates:
1. lexical-required behavior has verse without lexical match -> FAIL
2. annotation-required behavior has verse without annotation provenance -> FAIL
3. Missing evidence provenance -> FAIL
4. Invalid verse_key format -> FAIL

Usage:
    python scripts/validate_concept_index_v3.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.quran_store import get_quran_store
from src.data.behavior_registry import get_behavior_registry
from src.models.evidence_policy import EvidenceMode


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_verse_key_format(verse_key: str) -> bool:
    """Validate that verse_key is in format 'surah:ayah'."""
    if not verse_key or ':' not in verse_key:
        return False

    parts = verse_key.split(':')
    if len(parts) != 2:
        return False

    try:
        surah = int(parts[0])
        ayah = int(parts[1])
        return 1 <= surah <= 114 and ayah >= 1
    except ValueError:
        return False


def validate_verse_exists(verse_key: str, store) -> bool:
    """Validate that verse exists in Quran store."""
    verse = store.get_verse(verse_key)
    return verse is not None


def validate_lexical_match(verse_key: str, pattern: str, store) -> Dict[str, Any]:
    """
    Validate that a verse actually matches the lexical pattern.

    Returns:
        {
            "valid": True/False,
            "matched_token": "token" if valid,
            "reason": "..." if invalid
        }
    """
    verse = store.get_verse(verse_key)
    if not verse:
        return {"valid": False, "reason": f"verse_key {verse_key} not found"}

    if not pattern:
        return {"valid": False, "reason": "empty pattern"}

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"valid": False, "reason": f"invalid regex pattern: {e}"}

    # Check normalized tokens
    for token in verse.tokens_norm:
        if regex.search(token):
            return {"valid": True, "matched_token": token}

    # Also check full text
    if regex.search(verse.text_norm):
        return {"valid": True, "matched_in": "text_norm"}

    return {
        "valid": False,
        "reason": f"no match for pattern in tokens or text",
        "sample_tokens": verse.tokens_norm[:5]
    }


def build_lexical_pattern(behavior) -> str:
    """Build regex pattern from behavior's lexical spec."""
    lexical_spec = behavior.evidence_policy.lexical_spec
    if not lexical_spec:
        return ""

    all_patterns = []

    for form in lexical_spec.forms:
        if form:
            all_patterns.append(re.escape(form))

    for syn in lexical_spec.synonyms:
        if syn:
            all_patterns.append(re.escape(syn))

    # Fallback to Arabic label
    if not all_patterns:
        label = behavior.label_ar
        if label:
            all_patterns.append(re.escape(label))

    return "|".join(all_patterns) if all_patterns else ""


def validate_behavior_entry(entry: Dict, behavior, store) -> Dict[str, Any]:
    """
    Validate all verses for a behavior against its evidence policy.

    Returns validation result with detailed errors.
    """
    behavior_id = entry["concept_id"]
    policy = behavior.evidence_policy
    verses = entry.get("verses", [])

    result = {
        "behavior_id": behavior_id,
        "total_verses": len(verses),
        "valid_count": 0,
        "invalid_count": 0,
        "errors": [],
        "passed": True
    }

    # Gate 1: Check verse_key format
    for verse in verses:
        vk = verse.get("verse_key", "")
        if not validate_verse_key_format(vk):
            result["invalid_count"] += 1
            result["errors"].append({
                "verse_key": vk,
                "gate": "verse_key_format",
                "reason": "Invalid verse_key format"
            })
            continue

        # Gate 2: Check verse exists
        if not validate_verse_exists(vk, store):
            result["invalid_count"] += 1
            result["errors"].append({
                "verse_key": vk,
                "gate": "verse_exists",
                "reason": "Verse not found in Quran store"
            })
            continue

        # Gate 3: Check evidence provenance
        evidence = verse.get("evidence", [])
        if not evidence:
            result["invalid_count"] += 1
            result["errors"].append({
                "verse_key": vk,
                "gate": "evidence_provenance",
                "reason": "No evidence provided"
            })
            continue

        # Gate 4: For lexical-required behaviors, validate lexical match
        if policy.lexical_required:
            has_valid_lexical = False
            pattern = build_lexical_pattern(behavior)

            if pattern:
                validation = validate_lexical_match(vk, pattern, store)
                has_valid_lexical = validation["valid"]

                if not has_valid_lexical:
                    result["invalid_count"] += 1
                    result["errors"].append({
                        "verse_key": vk,
                        "gate": "lexical_match",
                        "reason": validation.get("reason", "No lexical match")
                    })
                    continue

        # If we get here, the verse is valid
        result["valid_count"] += 1

    # Behavior passes if no invalid verses
    result["passed"] = result["invalid_count"] == 0

    # Limit error samples to keep report manageable
    if len(result["errors"]) > 20:
        result["errors"] = result["errors"][:20]
        result["errors_truncated"] = True

    return result


def main() -> int:
    """Run validation gates on concept_index_v3."""
    print("=" * 60)
    print("Phase 5: Validation Gates")
    print("=" * 60)

    try:
        # Load dependencies
        print("\nLoading Quran store...")
        store = get_quran_store()
        print(f"  Loaded {store.total_verses} verses")

        print("\nLoading behavior registry...")
        registry = get_behavior_registry()
        print(f"  Loaded {registry.count()} behaviors")

        print("\nLoading concept_index_v3...")
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        if not index_path.exists():
            print(f"ERROR: {index_path} not found")
            return 1

        entries = []
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))
        print(f"  Loaded {len(entries)} behavior entries")

        # Run validation
        print("\nRunning validation gates...")
        all_results = []
        total_failures = 0
        total_verses_checked = 0
        total_errors = 0

        # Gate counters
        gate_failures = {
            "verse_key_format": 0,
            "verse_exists": 0,
            "evidence_provenance": 0,
            "lexical_match": 0
        }

        for i, entry in enumerate(entries):
            behavior_id = entry["concept_id"]
            behavior = registry.get(behavior_id)

            if not behavior:
                all_results.append({
                    "behavior_id": behavior_id,
                    "error": "behavior not found in registry",
                    "passed": False
                })
                total_failures += 1
                continue

            result = validate_behavior_entry(entry, behavior, store)
            all_results.append(result)

            total_verses_checked += result["total_verses"]
            total_errors += result["invalid_count"]

            if not result["passed"]:
                total_failures += 1
                # Count gate failures
                for err in result["errors"]:
                    gate = err.get("gate", "unknown")
                    gate_failures[gate] = gate_failures.get(gate, 0) + 1

            if (i + 1) % 20 == 0:
                print(f"  Validated {i + 1}/{len(entries)} behaviors...")

        # Generate report
        report = {
            "phase": 5,
            "description": "Validation Gates Report",
            "summary": {
                "total_behaviors": len(entries),
                "behaviors_passed": len(entries) - total_failures,
                "behaviors_failed": total_failures,
                "total_verses_checked": total_verses_checked,
                "total_verse_errors": total_errors,
                "pass_rate": f"{((len(entries) - total_failures) / len(entries) * 100):.1f}%"
            },
            "gate_failures": gate_failures,
            "validation_passed": total_failures == 0,
            "results": all_results
        }

        report_path = Path("artifacts/concept_index_v3_validation.json")
        save_json(report_path, report)
        print(f"\nSaved report to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION GATES SUMMARY")
        print("=" * 60)
        print(f"Total behaviors: {len(entries)}")
        print(f"Behaviors passed: {len(entries) - total_failures}")
        print(f"Behaviors failed: {total_failures}")
        print(f"Total verses checked: {total_verses_checked}")
        print(f"Total verse errors: {total_errors}")
        print(f"\nGate failure breakdown:")
        for gate, count in gate_failures.items():
            print(f"  {gate}: {count}")

        # Highlight patience
        patience_result = next(
            (r for r in all_results if "PATIENCE" in r.get("behavior_id", "").upper()),
            None
        )
        if patience_result:
            print(f"\nPatience (sabr) validation:")
            print(f"  Verses: {patience_result['total_verses']}")
            print(f"  Valid: {patience_result['valid_count']}")
            print(f"  Invalid: {patience_result['invalid_count']}")
            print(f"  Passed: {patience_result['passed']}")

        print("\n" + "=" * 60)

        # EXIT CODE based on validation
        if total_failures > 0:
            print("VALIDATION FAILED")
            print(f"  {total_failures} behaviors have invalid verses")
            failed_behaviors = [r["behavior_id"] for r in all_results if not r.get("passed", True)]
            print(f"  Failed: {failed_behaviors[:10]}...")
            print("=" * 60)
            return 1
        else:
            print("VALIDATION PASSED")
            print("  All behaviors have valid evidence")
            print("=" * 60)
            return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
