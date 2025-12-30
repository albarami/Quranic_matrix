#!/usr/bin/env python3
"""
Phase 0 Tests: Quran Schema Validation

These tests validate that the Quran JSON schema is correct and that
the baseline artifacts have been created.

Exit code 0 = all tests pass (proceed to Phase 1)
Exit code 1 = tests failed (DO NOT proceed)
"""

import json
import sys
from pathlib import Path

# Expected values
EXPECTED_SURAHS = 114
EXPECTED_VERSES = 6236
MIN_TOKENS = 70000  # At least 70k tokens expected


def test_artifacts_exist() -> bool:
    """Test that required artifacts exist."""
    artifacts = [
        Path("artifacts/quran_schema_report.json"),
        Path("artifacts/quran_counts.json"),
    ]

    all_exist = True
    for artifact in artifacts:
        if not artifact.exists():
            print(f"  [FAIL] Missing artifact: {artifact}")
            all_exist = False
        else:
            print(f"  [PASS] Found: {artifact}")

    return all_exist


def test_quran_counts() -> bool:
    """Test that Quran counts match expected values."""
    counts_path = Path("artifacts/quran_counts.json")

    if not counts_path.exists():
        print("  [FAIL] quran_counts.json not found")
        return False

    with open(counts_path, 'r', encoding='utf-8') as f:
        counts = json.load(f)

    all_pass = True

    # Check schema_valid
    if not counts.get("schema_valid"):
        print(f"  [FAIL] schema_valid is False")
        all_pass = False
    else:
        print(f"  [PASS] schema_valid: True")

    # Check surahs
    actual_surahs = counts.get("total_surahs", 0)
    if actual_surahs != EXPECTED_SURAHS:
        print(f"  [FAIL] total_surahs: {actual_surahs} (expected {EXPECTED_SURAHS})")
        all_pass = False
    else:
        print(f"  [PASS] total_surahs: {actual_surahs}")

    # Check verses
    actual_verses = counts.get("total_verses", 0)
    if actual_verses != EXPECTED_VERSES:
        print(f"  [FAIL] total_verses: {actual_verses} (expected {EXPECTED_VERSES})")
        all_pass = False
    else:
        print(f"  [PASS] total_verses: {actual_verses}")

    # Check tokens (minimum)
    actual_tokens = counts.get("total_tokens", 0)
    if actual_tokens < MIN_TOKENS:
        print(f"  [FAIL] total_tokens: {actual_tokens} (expected >= {MIN_TOKENS})")
        all_pass = False
    else:
        print(f"  [PASS] total_tokens: {actual_tokens}")

    # Check error count
    error_count = counts.get("error_count", -1)
    if error_count != 0:
        print(f"  [FAIL] error_count: {error_count} (expected 0)")
        all_pass = False
    else:
        print(f"  [PASS] error_count: 0")

    return all_pass


def test_schema_report_structure() -> bool:
    """Test that schema report has required fields."""
    report_path = Path("artifacts/quran_schema_report.json")

    if not report_path.exists():
        print("  [FAIL] quran_schema_report.json not found")
        return False

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    required_fields = [
        "schema_valid",
        "total_surahs",
        "total_verses",
        "total_tokens",
        "surah_counts",
        "sample_verses"
    ]

    all_pass = True
    for field in required_fields:
        if field not in report:
            print(f"  [FAIL] Missing field: {field}")
            all_pass = False
        else:
            print(f"  [PASS] Has field: {field}")

    # Validate surah_counts structure
    if "surah_counts" in report and len(report["surah_counts"]) > 0:
        first_surah = report["surah_counts"][0]
        surah_fields = ["surah", "verse_count"]
        for field in surah_fields:
            if field not in first_surah:
                print(f"  [FAIL] surah_counts[0] missing: {field}")
                all_pass = False

    return all_pass


def test_documentation_exists() -> bool:
    """Test that required documentation exists."""
    docs = [
        Path("docs/REVERTED_PATCHES.md"),
    ]

    all_exist = True
    for doc in docs:
        if not doc.exists():
            print(f"  [FAIL] Missing doc: {doc}")
            all_exist = False
        else:
            print(f"  [PASS] Found: {doc}")

    return all_exist


def main() -> int:
    """Run all Phase 0 tests."""
    print("=" * 60)
    print("Phase 0 Tests: Quran Schema Validation")
    print("=" * 60)

    all_passed = True

    print("\n1. Testing artifacts exist...")
    if not test_artifacts_exist():
        all_passed = False

    print("\n2. Testing Quran counts...")
    if not test_quran_counts():
        all_passed = False

    print("\n3. Testing schema report structure...")
    if not test_schema_report_structure():
        all_passed = False

    print("\n4. Testing documentation exists...")
    if not test_documentation_exists():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("Phase 0 Tests: ALL PASSED")
        print("Ready to proceed to Phase 1")
        print("=" * 60)
        return 0
    else:
        print("Phase 0 Tests: FAILED")
        print("DO NOT proceed until all tests pass")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
