"""
Compatibility wrapper for Arabic normalization.

Enterprise plan references src/utils/arabic_normalize.py; the SSOT lives in
src/text/ar_normalize.py. This module re-exports the SSOT functions.
"""

from src.text.ar_normalize import (
    NORMALIZATION_PROFILES,
    normalize_ar,
    normalize_with_profile,
    normalize_strict,
    normalize_loose,
    normalize_tokens,
)

__all__ = [
    "NORMALIZATION_PROFILES",
    "normalize_ar",
    "normalize_with_profile",
    "normalize_strict",
    "normalize_loose",
    "normalize_tokens",
]
