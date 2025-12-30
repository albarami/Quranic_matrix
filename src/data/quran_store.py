#!/usr/bin/env python3
"""
Canonical Qur'an Store - Single Source of Truth (SSOT)

This module provides the authoritative access point for all Qur'anic text.
It loads Uthmani text and provides both original and normalized views.

Features:
- Load from JSON (uthmani_hafs_v1.tok_v1.json)
- Original Uthmani text preservation
- Normalized text for searching
- Token-level access (original and normalized)
- Singleton pattern for efficient memory usage

Version: 1.0.0
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text.ar_normalize import normalize_ar, normalize_tokens

__version__ = "1.0.0"

# Default path to Quran JSON
DEFAULT_QURAN_PATH = Path("data/quran/uthmani_hafs_v1.tok_v1.json")


@dataclass
class Verse:
    """
    Represents a single Qur'anic verse with both original and normalized text.

    Attributes:
        verse_key: Canonical key format "{surah}:{ayah}" (e.g., "2:45")
        surah: Surah number (1-114)
        ayah: Ayah number within surah
        text_uthmani: Original Uthmani text with diacritics
        text_norm: Normalized text (no diacritics, unified alifs)
        tokens_uthmani: List of tokens from original text
        tokens_norm: List of normalized tokens
    """
    verse_key: str
    surah: int
    ayah: int
    text_uthmani: str
    text_norm: str
    tokens_uthmani: List[str]
    tokens_norm: List[str]

    def __post_init__(self):
        """Validate verse data."""
        assert self.surah >= 1 and self.surah <= 114, f"Invalid surah: {self.surah}"
        assert self.ayah >= 1, f"Invalid ayah: {self.ayah}"
        assert self.verse_key == f"{self.surah}:{self.ayah}", "verse_key mismatch"


@dataclass
class QuranStore:
    """
    Single Source of Truth (SSOT) for Qur'an text.

    Loads Uthmani text from JSON and provides:
    - Original text
    - Normalized text
    - Token-level access (original and normalized)
    - Search capabilities

    Usage:
        store = get_quran_store()
        verse = store.get_verse("2:45")
        print(verse.text_norm)  # Normalized text for searching
    """
    verses: Dict[str, Verse] = field(default_factory=dict)
    by_surah: Dict[int, List[str]] = field(default_factory=dict)
    total_verses: int = 0
    total_tokens: int = 0
    source_path: str = ""
    quran_version: str = ""
    tokenization_id: str = ""

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'QuranStore':
        """
        Load Qur'an from JSON and build normalized store.

        Args:
            path: Path to Quran JSON file. If None, uses default path.

        Returns:
            Populated QuranStore instance

        Raises:
            FileNotFoundError: If Quran JSON file not found
            ValueError: If JSON structure is invalid
        """
        if path is None:
            path = DEFAULT_QURAN_PATH

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Quran file not found: {path}")

        store = cls()
        store.source_path = str(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract metadata
        store.quran_version = data.get('quran_text_version', '')
        store.tokenization_id = data.get('tokenization_id', '')

        # Validate structure
        if 'surahs' not in data:
            raise ValueError("Quran JSON missing 'surahs' key")

        # Process each surah
        for surah_data in data.get('surahs', []):
            surah_num = surah_data.get('surah')
            if surah_num is None:
                continue

            store.by_surah[surah_num] = []

            # Process each ayah
            for ayah_data in surah_data.get('ayat', []):
                ayah_num = ayah_data.get('ayah')
                if ayah_num is None:
                    continue

                verse_key = f"{surah_num}:{ayah_num}"
                text_uthmani = ayah_data.get('text', '')

                # Extract tokens
                tokens_uthmani = [
                    t.get('text', '')
                    for t in ayah_data.get('tokens', [])
                ]

                # Create verse with normalized text
                verse = Verse(
                    verse_key=verse_key,
                    surah=surah_num,
                    ayah=ayah_num,
                    text_uthmani=text_uthmani,
                    text_norm=normalize_ar(text_uthmani),
                    tokens_uthmani=tokens_uthmani,
                    tokens_norm=normalize_tokens(tokens_uthmani)
                )

                store.verses[verse_key] = verse
                store.by_surah[surah_num].append(verse_key)
                store.total_tokens += len(tokens_uthmani)

        store.total_verses = len(store.verses)
        return store

    def get_verse(self, verse_key: str) -> Optional[Verse]:
        """
        Get verse by key.

        Args:
            verse_key: Format "{surah}:{ayah}" (e.g., "2:45")

        Returns:
            Verse object or None if not found
        """
        return self.verses.get(verse_key)

    def get_verses(self, verse_keys: List[str]) -> List[Verse]:
        """
        Get multiple verses by keys.

        Args:
            verse_keys: List of verse keys

        Returns:
            List of Verse objects (excludes missing keys)
        """
        return [
            self.verses[vk]
            for vk in verse_keys
            if vk in self.verses
        ]

    def get_surah_verses(self, surah: int) -> List[Verse]:
        """
        Get all verses in a surah.

        Args:
            surah: Surah number (1-114)

        Returns:
            List of Verse objects in order
        """
        verse_keys = self.by_surah.get(surah, [])
        return [self.verses[vk] for vk in verse_keys]

    def search_tokens_norm(self, pattern: str) -> List[Tuple[str, int, str]]:
        """
        Search normalized tokens for pattern.

        Args:
            pattern: Regex pattern to match

        Returns:
            List of tuples: (verse_key, token_index, matched_token)
        """
        import re
        regex = re.compile(pattern)
        results = []

        for verse_key, verse in self.verses.items():
            for idx, token in enumerate(verse.tokens_norm):
                if regex.search(token):
                    results.append((verse_key, idx, token))

        return results

    def search_text_norm(self, pattern: str) -> List[Tuple[str, str]]:
        """
        Search normalized text for pattern.

        Args:
            pattern: Regex pattern to match

        Returns:
            List of tuples: (verse_key, matched_portion)
        """
        import re
        regex = re.compile(pattern)
        results = []

        for verse_key, verse in self.verses.items():
            match = regex.search(verse.text_norm)
            if match:
                results.append((verse_key, match.group()))

        return results

    def find_verses_containing(self, substring: str) -> List[str]:
        """
        Find all verse keys where normalized text contains substring.

        Args:
            substring: Text to search for (will be normalized)

        Returns:
            List of verse_keys
        """
        search_norm = normalize_ar(substring)
        return [
            vk for vk, verse in self.verses.items()
            if search_norm in verse.text_norm
        ]

    def get_statistics(self) -> Dict:
        """Get statistics about the loaded Quran data."""
        return {
            "total_surahs": len(self.by_surah),
            "total_verses": self.total_verses,
            "total_tokens": self.total_tokens,
            "source_path": self.source_path,
            "quran_version": self.quran_version,
            "tokenization_id": self.tokenization_id
        }


# ============================================================================
# Singleton Pattern
# ============================================================================

_store: Optional[QuranStore] = None


def get_quran_store(path: Optional[str] = None, force_reload: bool = False) -> QuranStore:
    """
    Get or create singleton QuranStore.

    Args:
        path: Optional path to Quran JSON (only used on first load)
        force_reload: If True, reload even if already loaded

    Returns:
        QuranStore singleton instance
    """
    global _store

    if _store is None or force_reload:
        _store = QuranStore.load(path)

    return _store


def clear_store():
    """Clear the singleton store (useful for testing)."""
    global _store
    _store = None


# ============================================================================
# Testing / Validation
# ============================================================================

if __name__ == "__main__":
    print("QuranStore Self-Test")
    print("=" * 60)

    try:
        store = get_quran_store()
        stats = store.get_statistics()

        print(f"Loaded: {stats['source_path']}")
        print(f"Surahs: {stats['total_surahs']}")
        print(f"Verses: {stats['total_verses']}")
        print(f"Tokens: {stats['total_tokens']}")

        # Test verse lookup
        print("\nVerse 2:45 (patience verse):")
        verse = store.get_verse("2:45")
        if verse:
            print(f"  Original: {verse.text_uthmani[:50]}...")
            print(f"  Normalized: {verse.text_norm[:50]}...")
            print(f"  Tokens (norm): {verse.tokens_norm[:3]}...")

            # Check for sabr
            has_sabr = any("صبر" in t for t in verse.tokens_norm)
            print(f"  Contains 'sabr': {has_sabr}")
        else:
            print("  NOT FOUND!")

        # Test search
        print("\nSearch for 'sabr' pattern:")
        results = store.search_tokens_norm(r"صبر")
        print(f"  Found in {len(results)} tokens")
        if results:
            sample = results[:5]
            for vk, idx, token in sample:
                print(f"    {vk}: token[{idx}] = {token}")

        print("\n" + "=" * 60)
        print("Self-test PASSED")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
