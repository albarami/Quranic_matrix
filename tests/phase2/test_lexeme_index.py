#!/usr/bin/env python3
"""
Phase 2 Tests: Universal Lexeme Index

These tests validate that:
1. Lexeme index exists and has correct structure
2. Search functions work correctly
3. Sabr root family is findable with correct counts

Run with: pytest tests/phase2/ -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.lexeme_search import (
    load_lexeme_index,
    search_exact_token,
    search_token_pattern,
    search_root_family,
    get_verse_keys_for_pattern,
    get_verse_keys_for_token,
    count_token_occurrences,
    get_statistics,
    find_sabr_verses,
    find_allah_verses,
    clear_cache,
)


# ============================================================================
# Index Structure Tests
# ============================================================================

class TestLexemeIndexStructure:
    """Test lexeme index structure and metadata."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear cache before each test."""
        clear_cache()

    def test_index_file_exists(self):
        """Test that index file exists."""
        index_path = Path("data/index/lexeme_index.json")
        assert index_path.exists(), "lexeme_index.json not found"

    def test_index_has_required_keys(self):
        """Test that index has required top-level keys."""
        data = load_lexeme_index()
        required = ["version", "total_unique_tokens", "total_postings", "index"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_index_version(self):
        """Test index version."""
        data = load_lexeme_index()
        assert data["version"] == "1.0"

    def test_unique_token_count_reasonable(self):
        """Test that unique token count is reasonable."""
        stats = get_statistics()
        # Quran has many unique words - should be > 10000
        assert stats["total_unique_tokens"] > 10000
        # But not unreasonably many
        assert stats["total_unique_tokens"] < 50000

    def test_posting_count_reasonable(self):
        """Test that posting count is reasonable."""
        stats = get_statistics()
        # Total postings should be close to total tokens in Quran
        assert stats["total_postings"] > 70000
        assert stats["total_postings"] < 90000


# ============================================================================
# Exact Token Search Tests
# ============================================================================

class TestExactTokenSearch:
    """Test exact token search functionality."""

    def test_search_allah(self):
        """Test searching for الله."""
        postings = search_exact_token("الله")
        assert len(postings) > 0
        # الله appears many times in Quran
        assert len(postings) > 2000

    def test_search_nonexistent_token(self):
        """Test searching for token that doesn't exist."""
        postings = search_exact_token("xyz123")
        assert len(postings) == 0

    def test_posting_structure(self):
        """Test that postings have correct structure."""
        postings = search_exact_token("الله")
        assert len(postings) > 0
        for posting in postings[:10]:
            assert "verse_key" in posting
            assert "token_index" in posting
            # Verify verse_key format
            parts = posting["verse_key"].split(":")
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()


# ============================================================================
# Pattern Search Tests
# ============================================================================

class TestPatternSearch:
    """Test pattern-based search functionality."""

    def test_search_sabr_pattern(self):
        """Test searching for صبر pattern."""
        results = search_token_pattern(r"صبر")
        assert len(results) > 0
        # All results should have matched_token
        for r in results[:10]:
            assert "matched_token" in r
            assert "صبر" in r["matched_token"]

    def test_search_combined_pattern(self):
        """Test searching with OR pattern."""
        results = search_token_pattern(r"صبر|صابر")
        # Should find more than just صبر alone
        sabr_only = search_token_pattern(r"^صبر")
        assert len(results) >= len(sabr_only)

    def test_get_verse_keys_for_pattern(self):
        """Test getting unique verse keys."""
        verse_keys = get_verse_keys_for_pattern(r"صبر|صابر")
        assert isinstance(verse_keys, set)
        assert len(verse_keys) > 50


# ============================================================================
# Root Family Search Tests
# ============================================================================

class TestRootFamilySearch:
    """Test root family search functionality."""

    def test_search_root_family(self):
        """Test searching for root family."""
        forms = ["صبر", "صابر", "اصبر"]
        results = search_root_family(forms)
        assert len(results) > 0

    def test_search_empty_forms(self):
        """Test searching with empty forms list."""
        results = search_root_family([])
        assert results == []


# ============================================================================
# Sabr Validation Tests
# ============================================================================

class TestSabrValidation:
    """Test sabr (patience) specific functionality."""

    def test_find_sabr_verses_count(self):
        """Test that sabr verse count is in expected range."""
        verses = find_sabr_verses()
        # Should find 80-120 verses based on Phase 1 validation
        assert len(verses) >= 80, f"Too few sabr verses: {len(verses)}"
        assert len(verses) <= 120, f"Too many sabr verses: {len(verses)}"

    def test_find_sabr_verses_includes_required(self):
        """Test that required sabr verses are found."""
        verses = find_sabr_verses()
        required = ["2:45", "2:153", "3:200", "39:10", "103:3"]
        for verse_key in required:
            assert verse_key in verses, f"Required verse {verse_key} not found"

    def test_count_sabr_tokens(self):
        """Test counting sabr tokens."""
        counts = count_token_occurrences(r"صبر|صابر")
        assert len(counts) > 0
        # Total occurrences should match postings
        total = sum(counts.values())
        assert total > 50


# ============================================================================
# Allah Validation Tests
# ============================================================================

class TestAllahValidation:
    """Test الله specific functionality."""

    def test_find_allah_verses_count(self):
        """Test that الله verse count is significant."""
        verses = find_allah_verses()
        # الله appears in many verses (actual: ~1500+)
        assert len(verses) > 1500

    def test_allah_frequency(self):
        """Test الله frequency."""
        postings = search_exact_token("الله")
        # الله appears many times (actual: ~2100+)
        assert len(postings) > 2000


# ============================================================================
# Artifact Tests
# ============================================================================

class TestArtifacts:
    """Test that Phase 2 artifacts exist and are valid."""

    def test_lexeme_index_report_exists(self):
        """Test that lexeme_index_report.json exists."""
        import json
        report_path = Path("artifacts/lexeme_index_report.json")
        assert report_path.exists(), "lexeme_index_report.json not found"

        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data.get("validation", {}).get("valid") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
