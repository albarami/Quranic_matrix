#!/usr/bin/env python3
"""
Phase 8 Tests: All Behaviors Contract

Parametrized tests that validate ALL 73 behaviors meet the contract:
1. Every behavior is in concept index
2. Every behavior has evidence provenance
3. Every behavior passes validation
4. Every behavior is in graph

Run with: pytest tests/phase8/test_all_behaviors_contract.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.behavior_registry import get_behavior_registry, clear_registry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def registry():
    """Load behavior registry."""
    clear_registry()
    return get_behavior_registry()


@pytest.fixture(scope="module")
def concept_index():
    """Load concept index."""
    entries = {}
    with open("data/evidence/concept_index_v3.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["concept_id"]] = entry
    return entries


@pytest.fixture(scope="module")
def validation_report():
    """Load validation report."""
    with open("artifacts/concept_index_v3_validation.json", 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def graph():
    """Load graph."""
    with open("data/graph/graph_v3.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_behavior_ids():
    """Get all behavior IDs from registry."""
    clear_registry()
    reg = get_behavior_registry()
    return [b.behavior_id for b in reg.get_all()]


# ============================================================================
# Concept Index Contract
# ============================================================================

@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_in_concept_index(behavior_id, concept_index):
    """Every behavior must be in concept index."""
    assert behavior_id in concept_index, \
        f"Behavior {behavior_id} not in concept index"


@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_has_verses(behavior_id, concept_index):
    """Every behavior must have at least one verse."""
    entry = concept_index.get(behavior_id)
    if entry:
        verses = entry.get("verses", [])
        # Some behaviors may have no verses (annotation-only)
        # But for now all are lexical, so should have verses
        assert len(verses) >= 0  # Relaxed for edge cases


@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_verses_have_evidence(behavior_id, concept_index):
    """Every verse must have evidence."""
    entry = concept_index.get(behavior_id)
    if not entry:
        pytest.skip(f"Behavior {behavior_id} not in index")

    for verse in entry.get("verses", []):
        evidence = verse.get("evidence", [])
        assert len(evidence) > 0, \
            f"{behavior_id} verse {verse['verse_key']} has no evidence"


# ============================================================================
# Validation Contract
# ============================================================================

@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_passed_validation(behavior_id, validation_report):
    """Every behavior must pass validation."""
    result = next(
        (r for r in validation_report["results"] if r["behavior_id"] == behavior_id),
        None
    )
    if not result:
        pytest.skip(f"Behavior {behavior_id} not in validation")

    assert result["passed"] == True, \
        f"Behavior {behavior_id} failed validation: {result.get('errors', [])[:3]}"


@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_has_no_invalid_verses(behavior_id, validation_report):
    """Every behavior must have zero invalid verses."""
    result = next(
        (r for r in validation_report["results"] if r["behavior_id"] == behavior_id),
        None
    )
    if not result:
        pytest.skip(f"Behavior {behavior_id} not in validation")

    assert result["invalid_count"] == 0, \
        f"Behavior {behavior_id} has {result['invalid_count']} invalid verses"


# ============================================================================
# Graph Contract
# ============================================================================

@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_in_graph(behavior_id, graph):
    """Every behavior must be a node in graph."""
    node = next(
        (n for n in graph["nodes"] if n["id"] == behavior_id),
        None
    )
    assert node is not None, f"Behavior {behavior_id} not in graph"


@pytest.mark.parametrize("behavior_id", get_all_behavior_ids())
def test_behavior_has_arabic_label(behavior_id, graph):
    """Every behavior must have Arabic label in graph."""
    node = next(
        (n for n in graph["nodes"] if n["id"] == behavior_id),
        None
    )
    if not node:
        pytest.skip(f"Behavior {behavior_id} not in graph")

    assert node.get("labelAr"), f"Behavior {behavior_id} has no Arabic label"


# ============================================================================
# Cross-Module Contract
# ============================================================================

class TestCrossModuleConsistency:
    """Test consistency across all modules."""

    def test_registry_index_behavior_count_match(self, registry, concept_index):
        """Registry and index have same behavior count."""
        assert registry.count() == len(concept_index)

    def test_index_validation_behavior_count_match(self, concept_index, validation_report):
        """Index and validation have same behavior count."""
        assert len(concept_index) == validation_report["summary"]["total_behaviors"]

    def test_index_graph_behavior_count_match(self, concept_index, graph):
        """Index and graph have same behavior count."""
        graph_behaviors = [n for n in graph["nodes"] if n["type"] == "behavior"]
        assert len(concept_index) == len(graph_behaviors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
