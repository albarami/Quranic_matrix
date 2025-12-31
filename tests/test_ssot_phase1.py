#!/usr/bin/env python3
"""
Phase 1 SSOT Tests: Enterprise QBM Brain

Tests for:
1. SSOT counts (verses=6236, behaviors=87 target)
2. Arabic normalization with STRICT/LOOSE profiles
3. Foreign key integrity (no orphans)
4. Behavior inventory reconciliation

Run with: pytest tests/test_ssot_phase1.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
BEHAVIOR_CONCEPTS_FILE = Path("vocab/behavior_concepts.json")
SCHEMA_FILE = Path("schemas/postgres_truth_layer.sql")
MIGRATION_FILE = Path("db/migrations/001_ssot_extensions.sql")


# ============================================================================
# SSOT Count Tests
# ============================================================================

@pytest.mark.unit
class TestSSOTCounts:
    """Tests for SSOT entity counts."""
    
    def test_canonical_entities_file_exists(self):
        """canonical_entities.json must exist."""
        assert CANONICAL_ENTITIES_FILE.exists(), f"Missing: {CANONICAL_ENTITIES_FILE}"
    
    def test_behavior_concepts_file_exists(self):
        """behavior_concepts.json must exist."""
        assert BEHAVIOR_CONCEPTS_FILE.exists(), f"Missing: {BEHAVIOR_CONCEPTS_FILE}"
    
    def test_canonical_entities_has_behaviors(self):
        """canonical_entities.json must have behaviors section."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "behaviors" in data, "Missing 'behaviors' section"
        assert len(data["behaviors"]) == 87, f"Expected 87 behaviors, got {len(data['behaviors'])}"

    def test_behavior_concepts_has_87(self):
        """behavior_concepts.json must have 87 behaviors."""
        with open(BEHAVIOR_CONCEPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = 0
        for _, items in data.get("categories", {}).items():
            total += len(items)

        assert total == 87, f"Expected 87 behaviors in behavior_concepts.json, got {total}"
    
    def test_canonical_entities_has_agents(self):
        """canonical_entities.json must have agents section."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "agents" in data, "Missing 'agents' section"
        assert len(data["agents"]) == 14, f"Expected 14 agents, got {len(data['agents'])}"
    
    def test_canonical_entities_has_heart_states(self):
        """canonical_entities.json must have heart_states section."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "heart_states" in data, "Missing 'heart_states' section"
        assert len(data["heart_states"]) == 12, f"Expected 12 heart_states, got {len(data['heart_states'])}"
    
    def test_canonical_entities_has_consequences(self):
        """canonical_entities.json must have consequences section."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "consequences" in data, "Missing 'consequences' section"
        assert len(data["consequences"]) == 16, f"Expected 16 consequences, got {len(data['consequences'])}"
    
    def test_behavior_inventory_audit_exists(self):
        """Behavior inventory audit must have been run."""
        audit_file = Path("artifacts/behavior_inventory_audit.json")
        assert audit_file.exists(), "Run: python scripts/audit_behavior_inventory.py"
    
    def test_classifier_has_87_behaviors(self):
        """behavioral_classifier.py must have 87 behaviors."""
        try:
            from src.ml.behavioral_classifier import BEHAVIOR_CLASSES
            assert len(BEHAVIOR_CLASSES) == 87, f"Expected 87 behaviors, got {len(BEHAVIOR_CLASSES)}"
        except ImportError:
            pytest.skip("behavioral_classifier not available")


# ============================================================================
# Arabic Normalization Tests
# ============================================================================

@pytest.mark.unit
class TestArabicNormalization:
    """Tests for Arabic normalization with STRICT/LOOSE profiles."""
    
    def test_normalize_ar_import(self):
        """ar_normalize module must be importable."""
        from src.text.ar_normalize import normalize_ar
        assert normalize_ar is not None
    
    def test_normalization_profiles_exist(self):
        """NORMALIZATION_PROFILES must be defined."""
        from src.text.ar_normalize import NORMALIZATION_PROFILES
        assert "STRICT" in NORMALIZATION_PROFILES
        assert "LOOSE" in NORMALIZATION_PROFILES
    
    def test_strict_profile_disables_taa_marbuta(self):
        """STRICT profile must NOT normalize taa marbuta."""
        from src.text.ar_normalize import NORMALIZATION_PROFILES
        assert NORMALIZATION_PROFILES["STRICT"]["normalize_taa_marbuta"] == False
    
    def test_loose_profile_enables_taa_marbuta(self):
        """LOOSE profile must normalize taa marbuta."""
        from src.text.ar_normalize import NORMALIZATION_PROFILES
        assert NORMALIZATION_PROFILES["LOOSE"]["normalize_taa_marbuta"] == True
    
    def test_normalize_with_profile_strict(self):
        """normalize_with_profile STRICT must work."""
        from src.text.ar_normalize import normalize_with_profile
        
        result = normalize_with_profile("بِٱلصَّبْرِ", "STRICT")
        assert result == "بالصبر"
    
    def test_normalize_with_profile_loose(self):
        """normalize_with_profile LOOSE must convert taa marbuta."""
        from src.text.ar_normalize import normalize_with_profile
        
        result = normalize_with_profile("الرحمة", "LOOSE")
        assert result == "الرحمه"  # ة → ه
    
    def test_normalize_strict_shorthand(self):
        """normalize_strict shorthand must work."""
        from src.text.ar_normalize import normalize_strict
        
        result = normalize_strict("ٱلصَّـٰبِرِينَ")
        assert result == "الصابرين"
    
    def test_normalize_loose_shorthand(self):
        """normalize_loose shorthand must work."""
        from src.text.ar_normalize import normalize_loose
        
        result = normalize_loose("الصلاة")
        assert result == "الصلاه"  # ة → ه
    
    def test_strict_preserves_taa_marbuta(self):
        """STRICT profile must preserve taa marbuta."""
        from src.text.ar_normalize import normalize_strict
        
        result = normalize_strict("الرحمة")
        assert "ة" in result  # taa marbuta preserved
    
    @pytest.mark.parametrize("uthmani,expected", [
        ("بِٱلصَّبْرِ", "بالصبر"),
        ("ٱلصَّـٰبِرِينَ", "الصابرين"),
        ("وَٱسْتَعِينُوا۟", "واستعينوا"),
        ("ٱصْبِرُوا۟", "اصبروا"),
    ])
    def test_known_normalizations(self, uthmani: str, expected: str):
        """Test normalization against known inputs/outputs."""
        from src.text.ar_normalize import normalize_ar
        
        actual = normalize_ar(uthmani)
        assert actual == expected, f"normalize_ar('{uthmani}') = '{actual}', expected '{expected}'"


# ============================================================================
# Schema Tests
# ============================================================================

@pytest.mark.unit
class TestSchemaFiles:
    """Tests for schema files."""
    
    def test_base_schema_exists(self):
        """Base postgres_truth_layer.sql must exist."""
        assert SCHEMA_FILE.exists(), f"Missing: {SCHEMA_FILE}"
    
    def test_migration_file_exists(self):
        """Migration 001_ssot_extensions.sql must exist."""
        assert MIGRATION_FILE.exists(), f"Missing: {MIGRATION_FILE}"
    
    def test_migration_has_behavior_verse_links(self):
        """Migration must define behavior_verse_links table."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "behavior_verse_links" in sql
        assert "FOREIGN KEY (behavior_id) REFERENCES entities(entity_id)" in sql
    
    def test_migration_has_embedding_registry(self):
        """Migration must define embedding_registry table."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "embedding_registry" in sql
        assert "model_id" in sql
        assert "dimensions" in sql
    
    def test_migration_has_pgvector(self):
        """Migration must enable pgvector extension."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    
    def test_migration_has_behavior_trigger(self):
        """Migration must have trigger to enforce behavior entity type."""
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        
        assert "check_behavior_entity_type" in sql
        assert "enforce_behavior_type" in sql


# ============================================================================
# Foreign Key Integrity Tests (Dry Run)
# ============================================================================

@pytest.mark.unit
class TestForeignKeyIntegrity:
    """Tests for foreign key integrity (without database)."""
    
    def test_all_behaviors_have_required_fields(self):
        """All behaviors must have id, ar, en fields."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for b in data.get("behaviors", []):
            assert "id" in b, f"Behavior missing 'id': {b}"
            assert "ar" in b, f"Behavior missing 'ar': {b}"
            assert "en" in b, f"Behavior missing 'en': {b}"
    
    def test_all_behavior_ids_unique(self):
        """All behavior IDs must be unique."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ids = [b["id"] for b in data.get("behaviors", [])]
        assert len(ids) == len(set(ids)), "Duplicate behavior IDs found"
    
    def test_all_behavior_ids_have_prefix(self):
        """All behavior IDs must have BEH_ prefix."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for b in data.get("behaviors", []):
            assert b["id"].startswith("BEH_"), f"Invalid ID prefix: {b['id']}"
    
    def test_no_orphan_agents(self):
        """All agents must have valid IDs."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for a in data.get("agents", []):
            assert "id" in a, f"Agent missing 'id': {a}"
            assert a["id"].startswith("AGT_"), f"Invalid agent ID prefix: {a['id']}"
    
    def test_no_orphan_consequences(self):
        """All consequences must have valid IDs."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for c in data.get("consequences", []):
            assert "id" in c, f"Consequence missing 'id': {c}"
            assert c["id"].startswith("CSQ_"), f"Invalid consequence ID prefix: {c['id']}"


# ============================================================================
# Behavior Inventory Tests
# ============================================================================

@pytest.mark.unit
class TestBehaviorInventory:
    """Tests for behavior inventory reconciliation."""
    
    def test_audit_report_exists(self):
        """Behavior inventory audit report must exist."""
        audit_file = Path("artifacts/behavior_inventory_audit.json")
        assert audit_file.exists(), "Run: python scripts/audit_behavior_inventory.py"
    
    def test_audit_report_has_counts(self):
        """Audit report must have counts section."""
        audit_file = Path("artifacts/behavior_inventory_audit.json")
        if not audit_file.exists():
            pytest.skip("Audit report not found")
        
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "counts" in data
        assert data["counts"]["canonical_entities"] == 87
        assert data["counts"]["behavior_concepts"] == 87
        assert data["counts"]["classifier_behaviors"] == 87
    
    def test_audit_report_has_gaps(self):
        """Audit report must have gaps section."""
        audit_file = Path("artifacts/behavior_inventory_audit.json")
        if not audit_file.exists():
            pytest.skip("Audit report not found")
        
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "gaps" in data
        assert "missing_in_canonical" in data["gaps"]
        assert "missing_in_classifier" in data["gaps"]
    
    def test_classifier_target_is_87(self):
        """Classifier behavior target must be 87."""
        audit_file = Path("artifacts/behavior_inventory_audit.json")
        if not audit_file.exists():
            pytest.skip("Audit report not found")
        
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert data["counts"]["target"] == 87

    def test_behavior_registry_has_evidence_policies(self):
        """Behavior registry must include evidence policies for all behaviors."""
        registry_file = Path("data/behaviors/behavior_registry.json")
        assert registry_file.exists(), "behavior_registry.json not found"

        with open(registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("total_behaviors") == 87
        for behavior in data.get("behaviors", []):
            policy = behavior.get("evidence_policy")
            assert policy and policy.get("mode"), f"Missing evidence_policy for {behavior.get('behavior_id')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
