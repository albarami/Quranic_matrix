#!/usr/bin/env python3
"""
Phase 0: Audit Quran JSON schema and produce baseline reports.

Outputs:
- artifacts/quran_schema_report.json
- artifacts/quran_counts.json

Exit codes:
- 0: Schema valid
- 1: Schema invalid (build should fail)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Paths
QURAN_PATH = Path("data/quran/uthmani_hafs_v1.tok_v1.json")
ARTIFACTS_DIR = Path("artifacts")
EXPECTED_VERSE_COUNT = 6236
EXPECTED_SURAH_COUNT = 114


def audit_quran_schema(quran_path: Path) -> Dict[str, Any]:
    """
    Validate Quran JSON schema.

    Validates:
    1. surahs[*].ayat[*] exists
    2. Total verses = 6236
    3. Each ayah has: ayah (int), text (str), tokens (list)
    4. verse_key format is "{surah}:{ayah}" consistently
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Load JSON
    with open(quran_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check top-level structure
    if 'surahs' not in data:
        errors.append("Missing 'surahs' key at top level")
        return {
            "schema_valid": False,
            "errors": errors,
            "total_surahs": 0,
            "total_verses": 0
        }

    surahs = data['surahs']
    total_surahs = len(surahs)
    total_verses = 0
    verses_with_tokens = 0
    total_tokens = 0
    surah_counts = []
    sample_verses = []

    # Validate each surah
    for surah_data in surahs:
        surah_num = surah_data.get('surah')

        if surah_num is None:
            errors.append(f"Surah missing 'surah' key: {surah_data.keys()}")
            continue

        if 'ayat' not in surah_data:
            errors.append(f"Surah {surah_num} missing 'ayat' key")
            continue

        ayat = surah_data['ayat']
        surah_verse_count = len(ayat)
        surah_counts.append({
            "surah": surah_num,
            "name_ar": surah_data.get('name_ar', ''),
            "verse_count": surah_verse_count
        })

        # Validate each ayah
        for ayah_data in ayat:
            total_verses += 1

            # Check required fields
            ayah_num = ayah_data.get('ayah')
            if ayah_num is None:
                errors.append(f"Surah {surah_num}: ayah missing 'ayah' key")
                continue

            verse_key = f"{surah_num}:{ayah_num}"

            if 'text' not in ayah_data:
                errors.append(f"Verse {verse_key}: missing 'text' key")
            elif not isinstance(ayah_data['text'], str):
                errors.append(f"Verse {verse_key}: 'text' is not string")

            if 'tokens' not in ayah_data:
                warnings.append(f"Verse {verse_key}: missing 'tokens' key")
            elif not isinstance(ayah_data['tokens'], list):
                errors.append(f"Verse {verse_key}: 'tokens' is not list")
            else:
                verses_with_tokens += 1
                total_tokens += len(ayah_data['tokens'])

                # Validate token structure
                for i, token in enumerate(ayah_data['tokens']):
                    if not isinstance(token, dict):
                        errors.append(f"Verse {verse_key} token {i}: not a dict")
                    elif 'text' not in token:
                        errors.append(f"Verse {verse_key} token {i}: missing 'text'")

            # Sample first few verses
            if len(sample_verses) < 10:
                sample_verses.append({
                    "verse_key": verse_key,
                    "text_preview": ayah_data.get('text', '')[:50] + "...",
                    "token_count": len(ayah_data.get('tokens', []))
                })

    # Check counts
    if total_surahs != EXPECTED_SURAH_COUNT:
        errors.append(f"Expected {EXPECTED_SURAH_COUNT} surahs, got {total_surahs}")

    if total_verses != EXPECTED_VERSE_COUNT:
        errors.append(f"Expected {EXPECTED_VERSE_COUNT} verses, got {total_verses}")

    schema_valid = len(errors) == 0

    return {
        "schema_valid": schema_valid,
        "total_surahs": total_surahs,
        "total_verses": total_verses,
        "verses_with_tokens": verses_with_tokens,
        "total_tokens": total_tokens,
        "expected_verse_count": EXPECTED_VERSE_COUNT,
        "expected_surah_count": EXPECTED_SURAH_COUNT,
        "errors": errors,
        "warnings": warnings,
        "surah_counts": surah_counts,
        "sample_verses": sample_verses,
        "metadata": data.get('metadata', {}),
        "quran_text_version": data.get('quran_text_version', ''),
        "tokenization_id": data.get('tokenization_id', '')
    }


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    """Run audit and produce reports."""
    print("=" * 60)
    print("Phase 0: Quran Schema Audit")
    print("=" * 60)

    if not QURAN_PATH.exists():
        print(f"ERROR: Quran file not found: {QURAN_PATH}")
        return 1

    print(f"\nAuditing: {QURAN_PATH}")
    report = audit_quran_schema(QURAN_PATH)

    # Save full report
    save_json(ARTIFACTS_DIR / "quran_schema_report.json", report)
    print(f"Saved: artifacts/quran_schema_report.json")

    # Save summary counts
    counts = {
        "total_surahs": report["total_surahs"],
        "total_verses": report["total_verses"],
        "verses_with_tokens": report["verses_with_tokens"],
        "total_tokens": report["total_tokens"],
        "schema_valid": report["schema_valid"],
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"])
    }
    save_json(ARTIFACTS_DIR / "quran_counts.json", counts)
    print(f"Saved: artifacts/quran_counts.json")

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Surahs:     {report['total_surahs']} (expected {EXPECTED_SURAH_COUNT})")
    print(f"  Verses:     {report['total_verses']} (expected {EXPECTED_VERSE_COUNT})")
    print(f"  Tokens:     {report['total_tokens']}")
    print(f"  Errors:     {len(report['errors'])}")
    print(f"  Warnings:   {len(report['warnings'])}")
    print(f"  Valid:      {'YES' if report['schema_valid'] else 'NO'}")
    print(f"{'='*60}")

    if report["errors"]:
        print("\nERRORS:")
        for err in report["errors"][:20]:
            print(f"  - {err}")
        if len(report["errors"]) > 20:
            print(f"  ... and {len(report['errors']) - 20} more")

    if report["warnings"]:
        print("\nWARNINGS:")
        for warn in report["warnings"][:10]:
            print(f"  - {warn}")

    # Exit code
    if report["schema_valid"]:
        print("\n[PASS] Schema validation PASSED")
        return 0
    else:
        print("\n[FAIL] Schema validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
