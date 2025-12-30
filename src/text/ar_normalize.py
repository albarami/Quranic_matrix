#!/usr/bin/env python3
"""
Arabic Text Normalization for Qur'anic Text

Deterministic normalization pipeline for Uthmani script.
This module is the SINGLE SOURCE OF TRUTH for Arabic normalization.

NORMALIZATION RULES (documented and versioned):
1. Remove tashkil (diacritics): fatha, damma, kasra, shadda, sukun, etc.
2. Remove Qur'anic marks: waqf signs, sajda marks, etc.
3. Normalize alif variants: [إأآٱ] -> ا
4. Normalize hamza-on-waw: ؤ -> و
5. Normalize hamza-on-ya: ئ -> ي
6. Normalize ta marbuta: ة -> ه (optional, disabled by default)
7. Remove tatweel (kashida): ـ
8. Preserve Arabic letters and spaces only

Version: 1.0.0
"""

import re
from typing import List

__version__ = "1.0.0"

# ============================================================================
# Unicode Patterns for Arabic Text
# ============================================================================

# Tashkil (diacritics) - U+064B to U+0652
# fatha ً, damma ٌ, kasra ٍ, fatha َ, damma ُ, kasra ِ, shadda ّ, sukun ْ
# Note: superscript alef U+0670 is converted to alif, not stripped
TASHKEEL_PATTERN = re.compile(r'[\u064B-\u0652]')

# Superscript alef (ٰ U+0670) - indicates long 'a' sound, convert to regular alif
SUPERSCRIPT_ALEF = '\u0670'

# Qur'anic annotation marks - U+06D6 to U+06ED
# Includes: small high seen, rounded high stop, etc.
QURANIC_MARKS_PATTERN = re.compile(r'[\u06D6-\u06ED]')

# Small waw/ya/high meem used in Uthmani
SMALL_LETTERS_PATTERN = re.compile(r'[\u06E5-\u06E6]')

# Alif variants to normalize to bare alif (ا)
# إ (alif with hamza below), أ (alif with hamza above), آ (alif madda), ٱ (alif wasla)
ALIF_VARIANTS = re.compile(r'[إأآٱ]')

# Tatweel (kashida) - used for text stretching
TATWEEL = '\u0640'

# Arabic letter range for validation
ARABIC_LETTER_RANGE = re.compile(r'[\u0621-\u064A]')


# ============================================================================
# Individual Normalization Functions
# ============================================================================

def convert_superscript_alef(text: str) -> str:
    """
    Convert superscript alef (ٰ U+0670) to regular alif (ا).

    The superscript alef indicates a long 'a' sound and should be
    converted to a regular alif for normalization, not stripped.

    Example:
        ٱلصَّـٰبِرِينَ -> ٱلصَّـابِرِينَ
    """
    return text.replace(SUPERSCRIPT_ALEF, 'ا')


def strip_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel).

    Removes: fatha, damma, kasra, shadda, sukun, tanwin

    Example:
        بِٱلصَّبْرِ -> بٱلصبر
    """
    return TASHKEEL_PATTERN.sub('', text)


def strip_quranic_marks(text: str) -> str:
    """
    Remove Qur'anic annotation marks (waqf, sajda, etc.).

    Removes marks in range U+06D6 to U+06ED used for recitation guidance.

    Example:
        وَٱسْتَعِينُوا۟ -> وَٱسْتَعِينُوا
    """
    text = QURANIC_MARKS_PATTERN.sub('', text)
    text = SMALL_LETTERS_PATTERN.sub('', text)
    return text


def normalize_alifs(text: str) -> str:
    """
    Normalize all alif variants to bare alif (ا).

    Normalizes: إ أ آ ٱ -> ا

    Example:
        ٱلصَّبْرِ -> الصبر (after diacritics removed)
    """
    return ALIF_VARIANTS.sub('ا', text)


def normalize_hamza(text: str) -> str:
    """
    Normalize hamza-on-carrier forms.

    Normalizes:
        ؤ (hamza on waw) -> و
        ئ (hamza on ya) -> ي

    Note: Standalone hamza (ء) is preserved.
    """
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ي')
    return text


def normalize_ta_marbuta(text: str) -> str:
    """
    Normalize ta marbuta to ha.

    ة -> ه

    WARNING: This can change meaning. Disabled by default in normalize_ar().
    Use only when needed for specific matching scenarios.
    """
    return text.replace('ة', 'ه')


def remove_tatweel(text: str) -> str:
    """
    Remove kashida/tatweel character (ـ).

    Tatweel is used for text stretching and has no semantic meaning.
    """
    return text.replace(TATWEEL, '')


def normalize_alif_maqsura(text: str) -> str:
    """
    Normalize alif maqsura to ya.

    ى -> ي

    This is common in Arabic text normalization.
    """
    return text.replace('ى', 'ي')


# ============================================================================
# Main Normalization Pipeline
# ============================================================================

def normalize_ar(text: str, normalize_ta: bool = False) -> str:
    """
    Full Arabic normalization pipeline.

    Order matters - each step depends on previous:
    1. Strip Qur'anic marks first (they can interfere with other processing)
    2. Convert superscript alef to regular alif (preserves long 'a' sound)
    3. Strip diacritics (tashkeel)
    4. Normalize alifs (all variants to bare alif)
    5. Normalize hamza (on carriers)
    6. Remove tatweel
    7. Normalize alif maqsura
    8. Optionally normalize ta marbuta

    Args:
        text: Arabic text (typically Uthmani script)
        normalize_ta: If True, convert ة to ه (default False)

    Returns:
        Normalized Arabic text

    Example:
        >>> normalize_ar("بِٱلصَّبْرِ")
        'بالصبر'
        >>> normalize_ar("ٱلصَّـٰبِرِينَ")
        'الصابرين'
    """
    # Step 1: Remove Qur'anic marks (waqf, sajda, etc.)
    text = strip_quranic_marks(text)

    # Step 2: Convert superscript alef to regular alif (before stripping diacritics)
    text = convert_superscript_alef(text)

    # Step 3: Remove diacritics
    text = strip_diacritics(text)

    # Step 4: Normalize alif variants
    text = normalize_alifs(text)

    # Step 5: Normalize hamza on carriers
    text = normalize_hamza(text)

    # Step 6: Remove tatweel
    text = remove_tatweel(text)

    # Step 7: Normalize alif maqsura
    text = normalize_alif_maqsura(text)

    # Step 8: Optionally normalize ta marbuta
    if normalize_ta:
        text = normalize_ta_marbuta(text)

    return text


def normalize_tokens(tokens: List[str], normalize_ta: bool = False) -> List[str]:
    """
    Normalize a list of tokens.

    Args:
        tokens: List of Arabic tokens
        normalize_ta: If True, convert ة to ه

    Returns:
        List of normalized tokens
    """
    return [normalize_ar(t, normalize_ta) for t in tokens]


# ============================================================================
# Utility Functions
# ============================================================================

def contains_arabic(text: str) -> bool:
    """Check if text contains any Arabic letters."""
    return bool(ARABIC_LETTER_RANGE.search(text))


def extract_arabic_only(text: str) -> str:
    """Extract only Arabic letters and spaces from text."""
    # Keep Arabic letters (U+0621-U+064A) and space
    return re.sub(r'[^\u0621-\u064A\s]', '', text)


def get_root_pattern(root: str) -> str:
    """
    Create a regex pattern to match Arabic root in normalized text.

    This handles common morphological variations by matching the root
    letters in sequence with optional letters between them.

    Args:
        root: Arabic root (e.g., "صبر")

    Returns:
        Regex pattern string

    Example:
        >>> get_root_pattern("صبر")
        'ص[^\\s]*ب[^\\s]*ر'
    """
    if not root:
        return ""

    # Normalize the root first
    root_norm = normalize_ar(root)

    # Create pattern with optional chars between root letters
    letters = list(root_norm)
    pattern_parts = [re.escape(letters[0])]
    for letter in letters[1:]:
        pattern_parts.append(r'[^\s]*')
        pattern_parts.append(re.escape(letter))

    return ''.join(pattern_parts)


# ============================================================================
# Testing / Validation
# ============================================================================

def validate_normalization(uthmani: str, expected: str) -> dict:
    """
    Validate normalization against expected output.

    Args:
        uthmani: Original Uthmani text
        expected: Expected normalized output

    Returns:
        dict with validation results
    """
    actual = normalize_ar(uthmani)
    return {
        "uthmani": uthmani,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected
    }


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        ("بِٱلصَّبْرِ", "بالصبر"),
        ("ٱلصَّـٰبِرِينَ", "الصابرين"),
        ("وَٱسْتَعِينُوا۟", "واستعينوا"),
        ("ٱصْبِرُوا۟", "اصبروا"),
        ("صَابِرُوا۟", "صابروا"),
    ]

    print("Arabic Normalization Self-Test")
    print("=" * 50)

    all_pass = True
    for uthmani, expected in test_cases:
        result = validate_normalization(uthmani, expected)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {uthmani} -> {result['actual']} (expected: {expected})")
        if not result["passed"]:
            all_pass = False

    print("=" * 50)
    print(f"Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
