#!/usr/bin/env python3
"""
Fast Lexeme Search using Pre-built Index

This module provides fast search functions over the lexeme index.
All searches are O(1) for exact matches and O(n) for pattern matching
where n is the number of unique tokens.

Usage:
    from src.data.lexeme_search import search_token_pattern, get_verse_keys_for_pattern

    # Find all occurrences of صبر patterns
    results = search_token_pattern(r"صبر|صابر")
    verse_keys = get_verse_keys_for_pattern(r"صبر")

Version: 1.0.0
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

__version__ = "1.0.0"

# Default path to lexeme index
DEFAULT_INDEX_PATH = Path("data/index/lexeme_index.json")


@lru_cache(maxsize=1)
def load_lexeme_index(path: Optional[str] = None) -> Dict:
    """
    Load lexeme index (cached).

    The index is loaded once and cached for all subsequent calls.

    Args:
        path: Optional path to index file. Uses default if not specified.

    Returns:
        Index dictionary with structure:
        {
            "version": "1.0",
            "total_unique_tokens": int,
            "total_postings": int,
            "index": {token: [{verse_key, token_index}, ...], ...}
        }
    """
    if path is None:
        path = DEFAULT_INDEX_PATH
    else:
        path = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_index() -> Dict[str, List[Dict]]:
    """Get the index dictionary directly."""
    return load_lexeme_index()["index"]


def search_exact_token(token_norm: str) -> List[Dict]:
    """
    Exact match on normalized token.

    Args:
        token_norm: Normalized token to search for

    Returns:
        List of postings: [{"verse_key": "2:45", "token_index": 1}, ...]
    """
    index = get_index()
    return index.get(token_norm, [])


def search_token_pattern(pattern: str) -> List[Dict]:
    """
    Regex search over all tokens.

    Returns postings for all tokens matching the pattern.
    Each posting includes the matched token.

    Args:
        pattern: Regex pattern to match against tokens

    Returns:
        List of postings with matched_token:
        [{"verse_key": "2:45", "token_index": 1, "matched_token": "بالصبر"}, ...]
    """
    index = get_index()
    regex = re.compile(pattern)
    results = []

    for token, postings in index.items():
        if regex.search(token):
            for posting in postings:
                # Create copy with matched token
                result = posting.copy()
                result["matched_token"] = token
                results.append(result)

    return results


def search_root_family(root_forms: List[str]) -> List[Dict]:
    """
    Search for a family of root forms.

    Creates a combined pattern from all forms and searches.

    Args:
        root_forms: List of normalized forms to search for
                   e.g., ["صبر", "صابر", "اصبر"]

    Returns:
        List of postings with matched tokens
    """
    if not root_forms:
        return []

    # Escape special regex chars and join with OR
    pattern = "|".join(re.escape(f) for f in root_forms)
    return search_token_pattern(pattern)


def get_verse_keys_for_pattern(pattern: str) -> Set[str]:
    """
    Get unique verse_keys matching pattern.

    Useful when you only need to know which verses contain a pattern,
    not the specific token positions.

    Args:
        pattern: Regex pattern to match

    Returns:
        Set of verse_keys
    """
    results = search_token_pattern(pattern)
    return {r["verse_key"] for r in results}


def get_verse_keys_for_token(token_norm: str) -> Set[str]:
    """
    Get unique verse_keys for exact token match.

    Args:
        token_norm: Normalized token

    Returns:
        Set of verse_keys
    """
    postings = search_exact_token(token_norm)
    return {p["verse_key"] for p in postings}


def count_token_occurrences(pattern: str) -> Dict[str, int]:
    """
    Count occurrences of each matching token.

    Args:
        pattern: Regex pattern to match

    Returns:
        Dict mapping token to occurrence count
    """
    index = get_index()
    regex = re.compile(pattern)
    counts = {}

    for token, postings in index.items():
        if regex.search(token):
            counts[token] = len(postings)

    return counts


def get_token_frequency(token_norm: str) -> int:
    """
    Get frequency (number of occurrences) of exact token.

    Args:
        token_norm: Normalized token

    Returns:
        Number of occurrences across all verses
    """
    postings = search_exact_token(token_norm)
    return len(postings)


def get_statistics() -> Dict[str, Any]:
    """
    Get index statistics.

    Returns:
        Dict with statistics from the loaded index
    """
    data = load_lexeme_index()
    return {
        "version": data.get("version"),
        "total_unique_tokens": data.get("total_unique_tokens"),
        "total_postings": data.get("total_postings")
    }


def clear_cache():
    """Clear the cached index (useful for testing)."""
    load_lexeme_index.cache_clear()


# ============================================================================
# Convenience Functions for Common Searches
# ============================================================================

def find_sabr_verses() -> Set[str]:
    """
    Find all verses containing صبر (patience) root family.

    Returns:
        Set of verse_keys
    """
    sabr_pattern = r"صبر|صابر|اصبر|فاصبر|يصبر|تصبر|نصبر"
    return get_verse_keys_for_pattern(sabr_pattern)


def find_allah_verses() -> Set[str]:
    """
    Find all verses containing الله (Allah).

    Returns:
        Set of verse_keys
    """
    return get_verse_keys_for_token("الله")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Lexeme Search Self-Test")
    print("=" * 50)

    try:
        stats = get_statistics()
        print(f"Index loaded: {stats['total_unique_tokens']} tokens, {stats['total_postings']} postings")

        # Test صبر search
        sabr_verses = find_sabr_verses()
        print(f"\nSabr (patience) verses: {len(sabr_verses)}")
        print(f"Sample: {sorted(list(sabr_verses))[:5]}")

        # Test Allah search
        allah_verses = find_allah_verses()
        print(f"\nAllah verses: {len(allah_verses)}")

        # Test pattern search
        results = search_token_pattern(r"صبر")
        print(f"\nTokens containing 'صبر': {len(results)} occurrences")

        # Show token counts
        counts = count_token_occurrences(r"صبر|صابر")
        print(f"Token frequencies:")
        for token, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {token}: {count}")

        print("\n" + "=" * 50)
        print("Self-test PASSED")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
