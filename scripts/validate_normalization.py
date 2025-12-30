#!/usr/bin/env python3
"""
Phase 1: Validate Arabic Normalization

This script validates that:
1. Normalization produces expected outputs for known test cases
2. QuranStore loads correctly with 6236 verses
3. Key verses (patience) can be found via normalized search
4. صبر root is findable in expected verses

Exit codes:
- 0: All validations pass
- 1: Validation failures (build should fail)

Outputs:
- artifacts/normalization_validation.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text.ar_normalize import normalize_ar, validate_normalization
from src.data.quran_store import get_quran_store, clear_store


def save_json(path: Path, data: dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def test_known_normalizations() -> dict:
    """
    Test normalization against known inputs/outputs.

    These test cases cover:
    - Diacritics removal
    - Qur'anic marks removal
    - Alif normalization
    - Tatweel removal
    """
    test_cases = [
        # (uthmani, expected_normalized)
        ("بِٱلصَّبْرِ", "بالصبر"),
        ("ٱلصَّـٰبِرِينَ", "الصابرين"),
        ("وَٱسْتَعِينُوا۟", "واستعينوا"),
        ("ٱصْبِرُوا۟", "اصبروا"),
        ("صَابِرُوا۟", "صابروا"),
        ("ٱلصَّلَوٰةِ", "الصلواة"),
        ("ٱلْخَـٰشِعِينَ", "الخاشعين"),
        ("إِلَّا", "الا"),
        ("أَن", "ان"),
        ("ءَامَنُوا۟", "ءامنوا"),
    ]

    results = []
    for uthmani, expected_norm in test_cases:
        result = validate_normalization(uthmani, expected_norm)
        results.append(result)

    passed_count = sum(1 for r in results if r["passed"])

    return {
        "test_count": len(test_cases),
        "passed_count": passed_count,
        "failed_count": len(test_cases) - passed_count,
        "all_passed": passed_count == len(test_cases),
        "results": results
    }


def test_quran_store_load() -> dict:
    """Test that QuranStore loads correctly."""
    try:
        clear_store()  # Clear any cached store
        store = get_quran_store()
        stats = store.get_statistics()

        checks = {
            "loaded": True,
            "total_surahs": stats["total_surahs"],
            "total_verses": stats["total_verses"],
            "total_tokens": stats["total_tokens"],
            "surahs_valid": stats["total_surahs"] == 114,
            "verses_valid": stats["total_verses"] == 6236,
            "tokens_valid": stats["total_tokens"] > 70000
        }

        checks["all_valid"] = all([
            checks["surahs_valid"],
            checks["verses_valid"],
            checks["tokens_valid"]
        ])

        return checks

    except Exception as e:
        return {
            "loaded": False,
            "error": str(e),
            "all_valid": False
        }


def test_sabr_detection() -> dict:
    """
    Test that صبر (patience) forms are findable after normalization.

    Known patience verses to verify:
    - 2:45: بِٱلصَّبْرِ -> بالصبر
    - 2:153: بِٱلصَّبْرِ... ٱلصَّـٰبِرِينَ -> بالصبر... الصابرين
    - 3:200: ٱصْبِرُوا۟ وَصَابِرُوا۟ -> اصبروا وصابروا
    - 39:10: ٱلصَّـٰبِرُونَ -> الصابرون
    - 103:3: بِٱلصَّبْرِ -> بالصبر
    """
    import re

    store = get_quran_store()

    # Key patience verses that MUST be found
    required_verses = [
        "2:45",    # واستعينوا بالصبر والصلاة
        "2:153",   # استعينوا بالصبر... إن الله مع الصابرين
        "3:200",   # اصبروا وصابروا
        "39:10",   # يوفى الصابرون أجرهم
        "103:3"    # وتواصوا بالصبر
    ]

    results = []

    # Root family patterns for ص-ب-ر (sabr)
    # This matches: صبر, صابر, اصبر, تصبر, يصبر, نصبر, etc.
    sabr_patterns = [
        r"صبر",      # Base root: sabr, sabar
        r"صابر",     # Active participle: saabir
        r"اصبر",     # Imperative: isbir
        r"تصبر",     # Present: tasbir
        r"يصبر",     # Present: yasbir
        r"نصبر",     # Present: nasbir
    ]
    sabr_regex = re.compile("|".join(sabr_patterns))

    for verse_key in required_verses:
        verse = store.get_verse(verse_key)
        if verse is None:
            results.append({
                "verse_key": verse_key,
                "found_verse": False,
                "found_sabr": False
            })
            continue

        # Check if any token matches root family
        found_in_token = any(sabr_regex.search(t) for t in verse.tokens_norm)

        # Also check full text
        found_in_text = bool(sabr_regex.search(verse.text_norm))

        results.append({
            "verse_key": verse_key,
            "found_verse": True,
            "found_sabr": found_in_token or found_in_text,
            "tokens_norm": verse.tokens_norm[:5],  # Sample
            "text_norm_preview": verse.text_norm[:60]
        })

    found_count = sum(1 for r in results if r["found_sabr"])

    return {
        "required_verses": len(required_verses),
        "found_count": found_count,
        "all_found": found_count == len(required_verses),
        "results": results
    }


def test_sabr_total_count() -> dict:
    """
    Count total verses containing ص-ب-ر (sabr) root family.

    Expected: ~100 verses (varies by pattern strictness)
    """
    store = get_quran_store()

    # Comprehensive pattern for ص-ب-ر root family
    # Matches: صبر, صابر, اصبر, فاصبر, يصبر, تصبر, نصبر, etc.
    sabr_pattern = r"صبر|صابر|اصبر|فاصبر|يصبر|تصبر|نصبر"

    # Search for sabr patterns in normalized tokens
    matches = store.search_tokens_norm(sabr_pattern)

    # Get unique verses
    verse_keys = set(m[0] for m in matches)

    # Also search full text for any missed
    text_matches = store.search_text_norm(sabr_pattern)
    for vk, _ in text_matches:
        verse_keys.add(vk)

    return {
        "token_matches": len(matches),
        "unique_verses": len(verse_keys),
        "expected_min": 80,
        "expected_max": 120,
        "in_range": 80 <= len(verse_keys) <= 120,
        "sample_verses": sorted(list(verse_keys))[:15]
    }


def main() -> int:
    """Run all normalization validations."""
    print("=" * 60)
    print("Phase 1: Arabic Normalization Validation")
    print("=" * 60)

    all_passed = True
    report = {
        "phase": 1,
        "description": "Arabic Normalization + QuranStore",
        "tests": {}
    }

    # Test 1: Known normalizations
    print("\n1. Testing known normalizations...")
    norm_results = test_known_normalizations()
    report["tests"]["known_normalizations"] = norm_results

    if norm_results["all_passed"]:
        print(f"   [PASS] All {norm_results['test_count']} normalization tests passed")
    else:
        print(f"   [FAIL] {norm_results['failed_count']}/{norm_results['test_count']} tests failed:")
        for r in norm_results["results"]:
            if not r["passed"]:
                print(f"      {r['uthmani']} -> {r['actual']} (expected {r['expected']})")
        all_passed = False

    # Test 2: QuranStore load
    print("\n2. Testing QuranStore load...")
    load_results = test_quran_store_load()
    report["tests"]["quran_store_load"] = load_results

    if load_results.get("all_valid"):
        print(f"   [PASS] Loaded {load_results['total_verses']} verses, {load_results['total_tokens']} tokens")
    else:
        print(f"   [FAIL] QuranStore validation failed:")
        if not load_results.get("loaded"):
            print(f"      Error: {load_results.get('error')}")
        else:
            if not load_results.get("surahs_valid"):
                print(f"      Surahs: {load_results['total_surahs']} (expected 114)")
            if not load_results.get("verses_valid"):
                print(f"      Verses: {load_results['total_verses']} (expected 6236)")
        all_passed = False

    # Test 3: Sabr detection in required verses
    print("\n3. Testing sabr detection in required verses...")
    sabr_results = test_sabr_detection()
    report["tests"]["sabr_detection"] = sabr_results

    if sabr_results["all_found"]:
        print(f"   [PASS] Found sabr in all {sabr_results['required_verses']} required verses")
    else:
        print(f"   [FAIL] Only found sabr in {sabr_results['found_count']}/{sabr_results['required_verses']} verses:")
        for r in sabr_results["results"]:
            if not r["found_sabr"]:
                print(f"      Missing: {r['verse_key']}")
        all_passed = False

    # Test 4: Total sabr count
    print("\n4. Testing total sabr verse count...")
    count_results = test_sabr_total_count()
    report["tests"]["sabr_total_count"] = count_results

    if count_results["in_range"]:
        print(f"   [PASS] Found {count_results['unique_verses']} unique verses with sabr")
    else:
        print(f"   [WARN] Found {count_results['unique_verses']} verses (expected {count_results['expected_min']}-{count_results['expected_max']})")
        # This is a warning, not a failure - exact count depends on pattern

    # Save report
    report["all_passed"] = all_passed
    save_json(Path("artifacts/normalization_validation.json"), report)
    print(f"\nSaved: artifacts/normalization_validation.json")

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("Phase 1 Validation: ALL PASSED")
        print("Ready to proceed to Phase 2 (Lexeme Index)")
        print("=" * 60)
        return 0
    else:
        print("Phase 1 Validation: FAILED")
        print("DO NOT proceed until all tests pass")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
