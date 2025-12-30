#!/usr/bin/env python3
"""
Phase 3 Tests: Behavior Registry with Evidence Policy

These tests validate that:
1. Registry file exists and has correct structure
2. All behaviors have evidence policies
3. Patience behavior has correct lexical specification
4. Registry loader works correctly

Run with: pytest tests/phase3/ -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.evidence_policy import (
    EvidenceMode,
    EvidencePolicy,
    LexicalSpec,
    BehaviorDefinition,
    create_lexical_behavior
)
from src.data.behavior_registry import (
    get_behavior_registry,
    clear_registry,
    BehaviorRegistry
)


# ============================================================================
# Evidence Policy Schema Tests
# ============================================================================

class TestEvidencePolicySchema:
    """Test evidence policy schema classes."""

    def test_lexical_spec_creation(self):
        """Test LexicalSpec creation."""
        spec = LexicalSpec(
            roots=["صبر"],
            forms=["صبر", "صابر"],
            synonyms=["اصبر"],
            exclude_patterns=[]
        )
        assert spec.roots == ["صبر"]
        assert len(spec.get_all_patterns()) == 3

    def test_evidence_policy_creation(self):
        """Test EvidencePolicy creation."""
        policy = EvidencePolicy(
            mode=EvidenceMode.LEXICAL,
            lexical_required=True
        )
        assert policy.mode == EvidenceMode.LEXICAL
        assert policy.lexical_required == True

    def test_behavior_definition_creation(self):
        """Test BehaviorDefinition creation."""
        behavior = create_lexical_behavior(
            behavior_id="TEST_BEH",
            label_ar="اختبار",
            label_en="Test",
            category="test",
            roots=["خ-ب-ر"],
            forms=["اختبار"]
        )
        assert behavior.behavior_id == "TEST_BEH"
        assert behavior.is_lexical() == True

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        spec = LexicalSpec(roots=["صبر"], forms=["صبر"])
        d = spec.to_dict()
        spec2 = LexicalSpec.from_dict(d)
        assert spec.roots == spec2.roots
        assert spec.forms == spec2.forms


# ============================================================================
# Registry File Tests
# ============================================================================

class TestRegistryFile:
    """Test behavior registry file."""

    def test_registry_file_exists(self):
        """Test that registry file exists."""
        registry_path = Path("data/behaviors/behavior_registry.json")
        assert registry_path.exists(), "behavior_registry.json not found"

    def test_registry_has_required_keys(self):
        """Test that registry has required keys."""
        import json
        with open("data/behaviors/behavior_registry.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        required = ["version", "total_behaviors", "behaviors"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_registry_has_behaviors(self):
        """Test that registry has behaviors."""
        import json
        with open("data/behaviors/behavior_registry.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert len(data["behaviors"]) > 50, "Expected at least 50 behaviors"

    def test_behaviors_have_evidence_policy(self):
        """Test that all behaviors have evidence_policy."""
        import json
        with open("data/behaviors/behavior_registry.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        for b in data["behaviors"]:
            assert "evidence_policy" in b, f"Behavior {b['behavior_id']} missing evidence_policy"
            assert "mode" in b["evidence_policy"], f"Behavior {b['behavior_id']} missing mode"


# ============================================================================
# Registry Loader Tests
# ============================================================================

class TestRegistryLoader:
    """Test BehaviorRegistry loader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry before each test."""
        clear_registry()

    def test_load_registry(self):
        """Test loading registry."""
        registry = get_behavior_registry()
        assert registry is not None
        assert registry.count() > 50

    def test_get_behavior_by_id(self):
        """Test getting behavior by ID."""
        registry = get_behavior_registry()
        # Find a behavior ID
        behavior = registry.get("BEH_EMO_PATIENCE")
        # If that ID doesn't exist, try getting by label
        if behavior is None:
            behavior = registry.get_by_label_ar("الصبر")
        assert behavior is not None

    def test_get_behavior_by_arabic_label(self):
        """Test getting behavior by Arabic label."""
        registry = get_behavior_registry()
        patience = registry.get_by_label_ar("الصبر")
        assert patience is not None
        assert patience.label_en.lower() == "patience"

    def test_get_lexical_behaviors(self):
        """Test getting lexical behaviors."""
        registry = get_behavior_registry()
        lexical = registry.get_lexical_behaviors()
        assert len(lexical) > 0
        for b in lexical:
            assert b.is_lexical()

    def test_get_statistics(self):
        """Test getting statistics."""
        registry = get_behavior_registry()
        stats = registry.get_statistics()
        assert "total_behaviors" in stats
        assert "by_mode" in stats
        assert stats["total_behaviors"] > 50


# ============================================================================
# Patience Behavior Tests
# ============================================================================

class TestPatienceBehavior:
    """Test patience (الصبر) behavior specifically."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup registry and get patience behavior."""
        clear_registry()
        self.registry = get_behavior_registry()
        self.patience = self.registry.get_by_label_ar("الصبر")

    def test_patience_exists(self):
        """Test that patience behavior exists."""
        assert self.patience is not None

    def test_patience_is_lexical(self):
        """Test that patience uses lexical evidence."""
        assert self.patience.evidence_policy.mode == EvidenceMode.LEXICAL

    def test_patience_lexical_required(self):
        """Test that patience requires lexical match."""
        assert self.patience.evidence_policy.lexical_required == True

    def test_patience_has_lexical_spec(self):
        """Test that patience has lexical specification."""
        spec = self.patience.evidence_policy.lexical_spec
        assert spec is not None
        assert len(spec.forms) > 0 or len(spec.synonyms) > 0

    def test_patience_patterns_include_sabr(self):
        """Test that patience patterns include صبر forms."""
        spec = self.patience.evidence_policy.lexical_spec
        all_patterns = spec.forms + spec.synonyms
        # At least one pattern should contain صبر
        has_sabr = any("صبر" in p or "صابر" in p for p in all_patterns)
        assert has_sabr, f"No sabr pattern found in {all_patterns}"


# ============================================================================
# Artifact Tests
# ============================================================================

class TestArtifacts:
    """Test that Phase 3 artifacts exist and are valid."""

    def test_behavior_registry_report_exists(self):
        """Test that behavior_registry_report.json exists."""
        import json
        report_path = Path("artifacts/behavior_registry_report.json")
        assert report_path.exists(), "behavior_registry_report.json not found"

        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data.get("validation", {}).get("valid") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
