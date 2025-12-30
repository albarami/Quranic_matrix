#!/usr/bin/env python3
"""
Phase 1 Tests: Arabic Normalization + QuranStore

These tests validate that:
1. Arabic normalization produces expected outputs
2. QuranStore loads 6236 verses correctly
3. Normalized text enables sabr root family detection

Run with: pytest tests/phase1/ -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text.ar_normalize import normalize_ar, normalize_tokens, convert_superscript_alef
from src.data.quran_store import get_quran_store, clear_store


# ============================================================================
# Normalization Tests
# ============================================================================

class TestArabicNormalization:
    """Test Arabic text normalization."""

    @pytest.mark.parametrize("uthmani,expected", [
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
    ])
    def test_known_normalizations(self, uthmani: str, expected: str):
        """Test normalization against known inputs/outputs."""
        actual = normalize_ar(uthmani)
        assert actual == expected, f"normalize_ar('{uthmani}') = '{actual}', expected '{expected}'"

    def test_superscript_alef_conversion(self):
        """Test that superscript alef is converted to regular alif."""
        # U+0670 (superscript alef) should become ا
        result = convert_superscript_alef("ـٰ")
        assert "ا" in result

    def test_normalize_tokens(self):
        """Test token list normalization."""
        tokens = ["بِٱلصَّبْرِ", "وَٱلصَّلَوٰةِ"]
        result = normalize_tokens(tokens)
        assert result == ["بالصبر", "والصلواة"]

    def test_normalize_ar_idempotent(self):
        """Test that normalizing already normalized text is idempotent."""
        text = "بالصبر"
        result = normalize_ar(text)
        assert result == text


# ============================================================================
# QuranStore Tests
# ============================================================================

class TestQuranStore:
    """Test QuranStore loading and searching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear and reload store for each test."""
        clear_store()
        self.store = get_quran_store()

    def test_total_verses(self):
        """Test that store has exactly 6236 verses."""
        assert self.store.total_verses == 6236

    def test_total_surahs(self):
        """Test that store has 114 surahs."""
        assert len(self.store.by_surah) == 114

    def test_verse_lookup(self):
        """Test verse lookup by key."""
        verse = self.store.get_verse("2:45")
        assert verse is not None
        assert verse.surah == 2
        assert verse.ayah == 45
        assert verse.verse_key == "2:45"

    def test_verse_has_normalized_text(self):
        """Test that verses have normalized text."""
        verse = self.store.get_verse("2:45")
        assert verse is not None
        assert verse.text_norm != verse.text_uthmani
        assert len(verse.text_norm) > 0

    def test_verse_has_normalized_tokens(self):
        """Test that verses have normalized tokens."""
        verse = self.store.get_verse("2:45")
        assert verse is not None
        assert len(verse.tokens_norm) > 0
        assert len(verse.tokens_norm) == len(verse.tokens_uthmani)


# ============================================================================
# Sabr Detection Tests
# ============================================================================

class TestSabrDetection:
    """Test detection of صبر (patience) root family in normalized text."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup store and patterns."""
        clear_store()
        self.store = get_quran_store()
        # Sabr root family patterns
        self.sabr_patterns = ["صبر", "صابر", "اصبر", "تصبر", "يصبر", "نصبر"]

    def _contains_sabr(self, text: str) -> bool:
        """Check if text contains any sabr pattern."""
        return any(p in text for p in self.sabr_patterns)

    @pytest.mark.parametrize("verse_key", [
        "2:45",    # بالصبر والصلاة
        "2:153",   # استعينوا بالصبر... الصابرين
        "3:200",   # اصبروا وصابروا
        "39:10",   # يوفى الصابرون
        "103:3",   # وتواصوا بالصبر
    ])
    def test_required_patience_verses(self, verse_key: str):
        """Test that required patience verses contain sabr pattern."""
        verse = self.store.get_verse(verse_key)
        assert verse is not None, f"Verse {verse_key} not found"

        # Check tokens or text
        found_in_tokens = any(self._contains_sabr(t) for t in verse.tokens_norm)
        found_in_text = self._contains_sabr(verse.text_norm)

        assert found_in_tokens or found_in_text, \
            f"Verse {verse_key} does not contain sabr pattern. " \
            f"Tokens: {verse.tokens_norm[:5]}... Text: {verse.text_norm[:50]}..."

    def test_sabr_verse_count_in_range(self):
        """Test that total sabr verses is in expected range."""
        import re
        sabr_regex = re.compile("|".join(self.sabr_patterns))

        verse_keys = set()
        for vk, verse in self.store.verses.items():
            if sabr_regex.search(verse.text_norm):
                verse_keys.add(vk)

        # Expected: 80-120 verses with sabr
        assert len(verse_keys) >= 80, f"Too few sabr verses: {len(verse_keys)}"
        assert len(verse_keys) <= 120, f"Too many sabr verses: {len(verse_keys)}"


# ============================================================================
# Artifact Tests
# ============================================================================

class TestArtifacts:
    """Test that Phase 1 artifacts exist and are valid."""

    def test_normalization_validation_artifact(self):
        """Test that normalization_validation.json exists and is valid."""
        import json
        artifact_path = Path("artifacts/normalization_validation.json")
        assert artifact_path.exists(), "normalization_validation.json not found"

        with open(artifact_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data.get("all_passed") == True, "Validation did not pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
