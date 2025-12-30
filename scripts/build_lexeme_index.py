#!/usr/bin/env python3
"""
Phase 2: Build Universal Lexeme Index

Creates an inverted index: normalized_token -> [(verse_key, token_index), ...]
This enables O(1) lookup for any token pattern without scanning all verses.

Outputs:
- data/index/lexeme_index.json: The inverted index
- artifacts/lexeme_index_report.json: Summary statistics

Exit codes:
- 0: Success
- 1: Failure
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.quran_store import get_quran_store


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_lexeme_index() -> Dict[str, Any]:
    """
    Build inverted index from normalized tokens.

    Structure:
    {
        "version": "1.0",
        "total_unique_tokens": int,
        "total_postings": int,
        "index": {
            "token_norm": [
                {"verse_key": "2:45", "token_index": 1},
                ...
            ]
        }
    }
    """
    print("Loading QuranStore...")
    store = get_quran_store()
    print(f"Loaded {store.total_verses} verses, {store.total_tokens} tokens")

    print("Building inverted index...")
    index: Dict[str, List[Dict]] = defaultdict(list)

    # Track statistics
    empty_tokens = 0
    processed_tokens = 0

    for verse_key, verse in store.verses.items():
        for idx, token_norm in enumerate(verse.tokens_norm):
            # Skip empty tokens (can happen from punctuation-only tokens)
            if not token_norm.strip():
                empty_tokens += 1
                continue

            index[token_norm].append({
                "verse_key": verse_key,
                "token_index": idx
            })
            processed_tokens += 1

    # Convert defaultdict to regular dict
    index_dict = dict(index)

    # Calculate statistics
    total_unique = len(index_dict)
    total_postings = sum(len(postings) for postings in index_dict.values())

    print(f"Built index: {total_unique} unique tokens, {total_postings} postings")
    if empty_tokens > 0:
        print(f"Skipped {empty_tokens} empty tokens")

    return {
        "version": "1.0",
        "total_unique_tokens": total_unique,
        "total_postings": total_postings,
        "processed_tokens": processed_tokens,
        "empty_tokens_skipped": empty_tokens,
        "index": index_dict
    }


def validate_index(index_data: Dict) -> Dict[str, Any]:
    """Validate the built index."""
    results = {
        "valid": True,
        "checks": {}
    }

    # Check 1: Has required keys
    required_keys = ["version", "total_unique_tokens", "total_postings", "index"]
    for key in required_keys:
        if key not in index_data:
            results["valid"] = False
            results["checks"][f"has_{key}"] = False
        else:
            results["checks"][f"has_{key}"] = True

    # Check 2: Unique tokens count > 10000 (Quran has many unique words)
    unique_count = index_data.get("total_unique_tokens", 0)
    results["checks"]["unique_tokens_reasonable"] = unique_count > 10000
    if unique_count <= 10000:
        results["valid"] = False

    # Check 3: Can find known tokens
    index = index_data.get("index", {})

    # Test for صبر (patience)
    sabr_found = any("صبر" in token for token in index.keys())
    results["checks"]["sabr_findable"] = sabr_found
    if not sabr_found:
        results["valid"] = False

    # Test for الله (Allah)
    allah_found = "الله" in index
    results["checks"]["allah_findable"] = allah_found
    if not allah_found:
        results["valid"] = False

    # Check 4: Postings have required structure
    sample_valid = True
    for token, postings in list(index.items())[:10]:
        for posting in postings:
            if "verse_key" not in posting or "token_index" not in posting:
                sample_valid = False
                break
        if not sample_valid:
            break
    results["checks"]["posting_structure_valid"] = sample_valid
    if not sample_valid:
        results["valid"] = False

    return results


def generate_report(index_data: Dict, validation: Dict) -> Dict:
    """Generate summary report."""
    index = index_data.get("index", {})

    # Find most frequent tokens
    token_freqs = [(token, len(postings)) for token, postings in index.items()]
    token_freqs.sort(key=lambda x: -x[1])
    top_tokens = token_freqs[:20]

    # Find tokens containing صبر
    sabr_tokens = [
        (token, len(index[token]))
        for token in index.keys()
        if "صبر" in token or "صابر" in token
    ]

    return {
        "phase": 2,
        "description": "Universal Lexeme Index",
        "statistics": {
            "total_unique_tokens": index_data.get("total_unique_tokens", 0),
            "total_postings": index_data.get("total_postings", 0),
            "processed_tokens": index_data.get("processed_tokens", 0),
            "empty_tokens_skipped": index_data.get("empty_tokens_skipped", 0),
        },
        "validation": validation,
        "top_20_tokens": [
            {"token": t, "frequency": f} for t, f in top_tokens
        ],
        "sabr_family_tokens": [
            {"token": t, "frequency": f} for t, f in sabr_tokens
        ],
        "sample_postings": {
            token: index.get(token, [])[:3]
            for token in ["الله", "الصبر", "والصبر"] if token in index
        }
    }


def main() -> int:
    """Build lexeme index and save outputs."""
    print("=" * 60)
    print("Phase 2: Build Universal Lexeme Index")
    print("=" * 60)

    try:
        # Build index
        index_data = build_lexeme_index()

        # Validate
        print("\nValidating index...")
        validation = validate_index(index_data)

        for check, passed in validation["checks"].items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        # Save index
        index_path = Path("data/index/lexeme_index.json")
        print(f"\nSaving index to {index_path}...")
        save_json(index_path, index_data)

        # Generate and save report
        report = generate_report(index_data, validation)
        report_path = Path("artifacts/lexeme_index_report.json")
        save_json(report_path, report)
        print(f"Saved report to {report_path}")

        # Final result
        print("\n" + "=" * 60)
        if validation["valid"]:
            print("Phase 2: LEXEME INDEX BUILT SUCCESSFULLY")
            print(f"  Unique tokens: {index_data['total_unique_tokens']}")
            print(f"  Total postings: {index_data['total_postings']}")
            print("=" * 60)
            return 0
        else:
            print("Phase 2: VALIDATION FAILED")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
