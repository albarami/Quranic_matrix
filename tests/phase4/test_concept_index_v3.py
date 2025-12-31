#!/usr/bin/env python3
"""
Phase 4 Tests: Concept Index v3

These tests validate that:
1. Concept index v3 file exists
2. All behaviors are indexed
3. Patience behavior has correct verses
4. All verses have evidence provenance
5. Validation passes for lexical behaviors

Run with: pytest tests/phase4/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.quran_store import get_quran_store
from src.data.behavior_registry import get_behavior_registry, clear_registry


# ============================================================================
# Concept Index File Tests
# ============================================================================

class TestConceptIndexFile:
    """Test concept index v3 file exists and has correct structure."""

    def test_concept_index_file_exists(self):
        """Test that concept_index_v3.jsonl exists."""
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        assert index_path.exists(), "concept_index_v3.jsonl not found"

    def test_concept_index_is_valid_jsonl(self):
        """Test that file is valid JSONL."""
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        entries = []
        with open(index_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON at line {line_num}: {e}")

        assert len(entries) > 0, "concept_index_v3.jsonl is empty"

    def test_concept_index_has_expected_behavior_count(self):
        """Test that index has expected number of behaviors."""
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        entries = []
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))

        # Should have 87 behaviors (matches current registry)
        assert len(entries) == 87, f"Expected 87 behaviors, got {len(entries)}"

    def test_all_entries_have_required_fields(self):
        """Test that all entries have required fields."""
        required_fields = [
            "concept_id", "term", "term_en", "entity_type",
            "evidence_policy_mode", "verses", "statistics", "validation"
        ]

        index_path = Path("data/evidence/concept_index_v3.jsonl")
        with open(index_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                entry = json.loads(line)
                for field in required_fields:
                    assert field in entry, f"Entry at line {line_num} missing field: {field}"


# ============================================================================
# Patience Behavior Tests
# ============================================================================

class TestPatienceBehavior:
    """Test patience (الصبر) behavior specifically."""

    @pytest.fixture
    def patience_entry(self):
        """Load patience entry from index."""
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry["concept_id"] == "BEH_EMO_PATIENCE":
                    return entry
        pytest.fail("BEH_EMO_PATIENCE not found in concept index")

    def test_patience_exists_in_index(self, patience_entry):
        """Test that patience behavior exists."""
        assert patience_entry is not None

    def test_patience_has_reasonable_verse_count(self, patience_entry):
        """Test that patience has reasonable number of verses (not 291 with 79% invalid)."""
        verse_count = patience_entry["statistics"]["total_verses"]
        # Should be between 60-120 (actual lexical matches)
        assert verse_count >= 60, f"Too few patience verses: {verse_count}"
        assert verse_count <= 120, f"Too many patience verses (might be corrupted): {verse_count}"

    def test_patience_required_verses_present(self, patience_entry):
        """Test that required patience verses are included."""
        required_verses = ["2:45", "2:153", "3:200", "39:10", "103:3"]
        verse_keys = {v["verse_key"] for v in patience_entry["verses"]}

        for req in required_verses:
            assert req in verse_keys, f"Required patience verse {req} not found"

    def test_patience_validation_passed(self, patience_entry):
        """Test that patience validation passed."""
        assert patience_entry["validation"]["passed"] == True

    def test_patience_verses_have_evidence(self, patience_entry):
        """Test that all patience verses have evidence."""
        for verse in patience_entry["verses"]:
            assert "evidence" in verse, f"Verse {verse['verse_key']} has no evidence"
            assert len(verse["evidence"]) > 0, f"Verse {verse['verse_key']} has empty evidence"

    def test_patience_evidence_is_lexical(self, patience_entry):
        """Test that patience evidence is lexical (since it's lexical-required)."""
        for verse in patience_entry["verses"]:
            has_lexical = any(e["type"] == "lexical" for e in verse["evidence"])
            assert has_lexical, f"Verse {verse['verse_key']} has no lexical evidence"


# ============================================================================
# Verse Evidence Tests
# ============================================================================

class TestVerseEvidence:
    """Test that verses have proper evidence provenance."""

    @pytest.fixture
    def all_entries(self):
        """Load all entries from index."""
        index_path = Path("data/evidence/concept_index_v3.jsonl")
        entries = []
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))
        return entries

    def test_all_verses_have_evidence(self, all_entries):
        """Test that all verses have evidence field."""
        for entry in all_entries:
            for verse in entry.get("verses", []):
                assert "evidence" in verse, \
                    f"{entry['concept_id']} verse {verse['verse_key']} has no evidence"

    def test_all_verses_have_provenance(self, all_entries):
        """Test that all verses have provenance field."""
        for entry in all_entries:
            for verse in entry.get("verses", []):
                assert "provenance" in verse, \
                    f"{entry['concept_id']} verse {verse['verse_key']} has no provenance"

    def test_all_verses_have_directness(self, all_entries):
        """Test that all verses have directness field."""
        for entry in all_entries:
            for verse in entry.get("verses", []):
                assert "directness" in verse, \
                    f"{entry['concept_id']} verse {verse['verse_key']} has no directness"

    def test_evidence_has_type(self, all_entries):
        """Test that all evidence entries have type field."""
        for entry in all_entries:
            for verse in entry.get("verses", []):
                for ev in verse.get("evidence", []):
                    assert "type" in ev, \
                        f"{entry['concept_id']} verse {verse['verse_key']} evidence has no type"


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Test concept index statistics."""

    @pytest.fixture
    def report(self):
        """Load concept index report."""
        report_path = Path("artifacts/concept_index_v3_report.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_report_exists(self, report):
        """Test that report exists and loads."""
        assert report is not None

    def test_total_behaviors_matches(self, report):
        """Test that total behaviors matches expected."""
        assert report["statistics"]["total_behaviors"] == 87

    def test_validation_passed(self, report):
        """Test that validation passed for all behaviors."""
        assert report["validation"]["all_passed"] == True

    def test_no_validation_errors(self, report):
        """Test that there are no validation errors."""
        assert report["statistics"]["total_validation_errors"] == 0


# ============================================================================
# Registry Alignment Tests
# ============================================================================

class TestRegistryAlignment:
    """Test that concept index aligns with behavior registry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry before each test."""
        clear_registry()

    def test_all_registry_behaviors_in_index(self):
        """Test that all behaviors from registry are in index."""
        registry = get_behavior_registry()
        registry_ids = {b.behavior_id for b in registry.get_all()}

        index_path = Path("data/evidence/concept_index_v3.jsonl")
        index_ids = set()
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                index_ids.add(entry["concept_id"])

        missing = registry_ids - index_ids
        assert len(missing) == 0, f"Behaviors missing from index: {missing}"

    def test_all_index_behaviors_in_registry(self):
        """Test that all behaviors in index are in registry."""
        registry = get_behavior_registry()
        registry_ids = {b.behavior_id for b in registry.get_all()}

        index_path = Path("data/evidence/concept_index_v3.jsonl")
        index_ids = set()
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                index_ids.add(entry["concept_id"])

        extra = index_ids - registry_ids
        assert len(extra) == 0, f"Extra behaviors in index: {extra}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
