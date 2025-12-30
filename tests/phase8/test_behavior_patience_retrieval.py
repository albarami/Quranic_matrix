#!/usr/bin/env python3
"""
Phase 8 Tests: Patience (الصبر) Retrieval End-to-End

These tests validate the complete retrieval pipeline for patience:
1. Concept index has correct verses
2. Graph has correct edges
3. All verses are lexically validated
4. Tafsir is available for key verses

Run with: pytest tests/phase8/test_behavior_patience_retrieval.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.quran_store import get_quran_store
from src.data.behavior_registry import get_behavior_registry, clear_registry


# ============================================================================
# Required Verses and Thresholds
# ============================================================================

REQUIRED_PATIENCE_VERSES = [
    "2:45",    # بالصبر والصلاة
    "2:153",   # استعينوا بالصبر... مع الصابرين
    "3:200",   # اصبروا وصابروا
    "39:10",   # يوفى الصابرون أجرهم
    "103:3"    # وتواصوا بالصبر
]

MIN_PATIENCE_VERSES = 60
MAX_PATIENCE_VERSES = 120

SABR_PATTERN = re.compile(r"صبر|صابر|اصبر|نصبر|يصبر|تصبر|فاصبر")


# ============================================================================
# Concept Index Tests
# ============================================================================

class TestPatienceConceptIndex:
    """Test patience retrieval from concept index."""

    @pytest.fixture
    def patience_entry(self):
        """Load patience entry from concept index."""
        with open("data/evidence/concept_index_v3.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry["concept_id"] == "BEH_EMO_PATIENCE":
                    return entry
        pytest.fail("BEH_EMO_PATIENCE not found")

    def test_patience_verse_count(self, patience_entry):
        """Test that patience has reasonable verse count."""
        verse_count = len(patience_entry["verses"])
        assert verse_count >= MIN_PATIENCE_VERSES, \
            f"Too few: {verse_count} < {MIN_PATIENCE_VERSES}"
        assert verse_count <= MAX_PATIENCE_VERSES, \
            f"Too many: {verse_count} > {MAX_PATIENCE_VERSES}"

    def test_required_verses_present(self, patience_entry):
        """Test that required verses are present."""
        verse_keys = {v["verse_key"] for v in patience_entry["verses"]}
        for req in REQUIRED_PATIENCE_VERSES:
            assert req in verse_keys, f"Missing required verse: {req}"

    def test_all_verses_have_sabr(self, patience_entry):
        """Test that all verses actually contain صبر root."""
        store = get_quran_store()
        invalid = []

        for v in patience_entry["verses"]:
            verse = store.get_verse(v["verse_key"])
            if not verse:
                invalid.append(v["verse_key"])
                continue

            if not SABR_PATTERN.search(verse.text_norm):
                invalid.append(v["verse_key"])

        assert len(invalid) == 0, \
            f"Found {len(invalid)} verses without sabr: {invalid[:10]}"


# ============================================================================
# Graph Tests
# ============================================================================

class TestPatienceGraph:
    """Test patience in knowledge graph."""

    @pytest.fixture
    def graph(self):
        """Load graph."""
        with open("data/graph/graph_v3.json", 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_patience_node_exists(self, graph):
        """Test patience node exists with correct label."""
        node = next(
            (n for n in graph["nodes"] if n["id"] == "BEH_EMO_PATIENCE"),
            None
        )
        assert node is not None
        assert node["labelAr"] == "الصبر"
        assert node["labelEn"] == "Patience"

    def test_patience_edges_count(self, graph):
        """Test patience has correct number of edges."""
        edges = [e for e in graph["edges"] if e["source"] == "BEH_EMO_PATIENCE"]
        assert len(edges) >= MIN_PATIENCE_VERSES
        assert len(edges) <= MAX_PATIENCE_VERSES

    def test_patience_edges_are_lexical(self, graph):
        """Test all patience edges are lexical type."""
        edges = [e for e in graph["edges"] if e["source"] == "BEH_EMO_PATIENCE"]
        non_lexical = [e for e in edges if e["evidenceType"] != "lexical"]
        assert len(non_lexical) == 0


# ============================================================================
# Validation Tests
# ============================================================================

class TestPatienceValidation:
    """Test patience validation results."""

    @pytest.fixture
    def validation_report(self):
        """Load validation report."""
        with open("artifacts/concept_index_v3_validation.json", 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_patience_passed_validation(self, validation_report):
        """Test patience passed all validation gates."""
        result = next(
            (r for r in validation_report["results"]
             if r["behavior_id"] == "BEH_EMO_PATIENCE"),
            None
        )
        assert result is not None
        assert result["passed"] == True
        assert result["invalid_count"] == 0


# ============================================================================
# Tafsir Tests
# ============================================================================

class TestPatienceTafsir:
    """Test tafsir coverage for patience verses."""

    @pytest.fixture
    def tafsir_report(self):
        """Load tafsir coverage report."""
        with open("artifacts/tafsir_coverage_report.json", 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_key_verses_have_full_coverage(self, tafsir_report):
        """Test key patience verses have 7/7 tafsir sources."""
        for verse_key in ["2:45", "2:153", "3:200"]:
            coverage = tafsir_report["patience_verses"][verse_key]["coverage"]
            assert coverage == 1.0, \
                f"Verse {verse_key} has only {coverage:.0%} tafsir coverage"


# ============================================================================
# Registry Tests
# ============================================================================

class TestPatienceRegistry:
    """Test patience in behavior registry."""

    @pytest.fixture
    def patience(self):
        """Get patience behavior from registry."""
        clear_registry()
        registry = get_behavior_registry()
        return registry.get_by_label_ar("الصبر")

    def test_patience_exists(self, patience):
        """Test patience exists in registry."""
        assert patience is not None
        assert patience.behavior_id == "BEH_EMO_PATIENCE"

    def test_patience_is_lexical_required(self, patience):
        """Test patience has lexical_required=True."""
        assert patience.evidence_policy.lexical_required == True

    def test_patience_has_lexical_spec(self, patience):
        """Test patience has lexical specification."""
        spec = patience.evidence_policy.lexical_spec
        assert spec is not None
        assert len(spec.forms) > 0 or len(spec.synonyms) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
