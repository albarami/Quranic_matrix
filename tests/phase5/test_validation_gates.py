#!/usr/bin/env python3
"""
Phase 5 Tests: Validation Gates

These tests validate that:
1. Validation script exits with correct codes
2. All behaviors pass validation
3. Patience behavior has zero invalid verses
4. Gate failure counts are zero
5. Validation report has correct structure

Run with: pytest tests/phase5/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Validation Report Tests
# ============================================================================

class TestValidationReport:
    """Test validation report exists and has correct structure."""

    @pytest.fixture
    def report(self):
        """Load validation report."""
        report_path = Path("artifacts/concept_index_v3_validation.json")
        assert report_path.exists(), "Validation report not found"
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_report_exists(self, report):
        """Test that report exists and loads."""
        assert report is not None

    def test_report_has_required_fields(self, report):
        """Test that report has required fields."""
        required = ["phase", "summary", "gate_failures", "validation_passed", "results"]
        for field in required:
            assert field in report, f"Missing field: {field}"

    def test_validation_passed_overall(self, report):
        """Test that overall validation passed."""
        assert report["validation_passed"] == True, \
            f"Validation failed: {report['summary']}"

    def test_all_behaviors_passed(self, report):
        """Test that all behaviors passed validation."""
        summary = report["summary"]
        assert summary["behaviors_failed"] == 0, \
            f"Failed behaviors: {summary['behaviors_failed']}"
        assert summary["behaviors_passed"] == 73

    def test_no_verse_errors(self, report):
        """Test that there are no verse errors."""
        assert report["summary"]["total_verse_errors"] == 0


# ============================================================================
# Gate Failure Tests
# ============================================================================

class TestGateFailures:
    """Test that all validation gates pass."""

    @pytest.fixture
    def report(self):
        """Load validation report."""
        report_path = Path("artifacts/concept_index_v3_validation.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_verse_key_format_gate(self, report):
        """Test that no verses fail verse_key_format gate."""
        assert report["gate_failures"].get("verse_key_format", 0) == 0

    def test_verse_exists_gate(self, report):
        """Test that no verses fail verse_exists gate."""
        assert report["gate_failures"].get("verse_exists", 0) == 0

    def test_evidence_provenance_gate(self, report):
        """Test that no verses fail evidence_provenance gate."""
        assert report["gate_failures"].get("evidence_provenance", 0) == 0

    def test_lexical_match_gate(self, report):
        """Test that no verses fail lexical_match gate."""
        assert report["gate_failures"].get("lexical_match", 0) == 0


# ============================================================================
# Patience Validation Tests
# ============================================================================

class TestPatienceValidation:
    """Test patience (الصبر) behavior validation specifically."""

    @pytest.fixture
    def patience_result(self):
        """Load patience validation result."""
        report_path = Path("artifacts/concept_index_v3_validation.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        for result in report["results"]:
            if result["behavior_id"] == "BEH_EMO_PATIENCE":
                return result

        pytest.fail("BEH_EMO_PATIENCE not found in validation results")

    def test_patience_exists_in_results(self, patience_result):
        """Test that patience is in validation results."""
        assert patience_result is not None

    def test_patience_validation_passed(self, patience_result):
        """Test that patience passed validation."""
        assert patience_result["passed"] == True

    def test_patience_no_invalid_verses(self, patience_result):
        """Test that patience has zero invalid verses."""
        assert patience_result["invalid_count"] == 0

    def test_patience_all_verses_valid(self, patience_result):
        """Test that all patience verses are valid."""
        assert patience_result["valid_count"] == patience_result["total_verses"]

    def test_patience_verse_count_reasonable(self, patience_result):
        """Test that patience has reasonable verse count."""
        verse_count = patience_result["total_verses"]
        # Should be between 60-120 validated verses
        assert verse_count >= 60, f"Too few patience verses: {verse_count}"
        assert verse_count <= 120, f"Too many patience verses: {verse_count}"


# ============================================================================
# Individual Behavior Tests
# ============================================================================

class TestBehaviorResults:
    """Test individual behavior validation results."""

    @pytest.fixture
    def all_results(self):
        """Load all validation results."""
        report_path = Path("artifacts/concept_index_v3_validation.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report["results"]

    def test_all_behaviors_have_results(self, all_results):
        """Test that all behaviors have validation results."""
        assert len(all_results) == 73

    def test_all_behaviors_passed(self, all_results):
        """Test that all behaviors passed validation."""
        failed = [r for r in all_results if not r.get("passed", True)]
        assert len(failed) == 0, f"Failed behaviors: {[r['behavior_id'] for r in failed]}"

    def test_all_behaviors_have_no_errors(self, all_results):
        """Test that no behaviors have errors."""
        with_errors = [r for r in all_results if r.get("invalid_count", 0) > 0]
        assert len(with_errors) == 0, \
            f"Behaviors with errors: {[r['behavior_id'] for r in with_errors]}"

    def test_result_structure(self, all_results):
        """Test that all results have correct structure."""
        required_fields = ["behavior_id", "total_verses", "valid_count", "invalid_count", "passed"]

        for result in all_results:
            for field in required_fields:
                assert field in result, \
                    f"{result.get('behavior_id', 'unknown')} missing field: {field}"


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestCrossValidation:
    """Test validation against concept index."""

    def test_validation_matches_concept_index(self):
        """Test that validation results match concept index entries."""
        # Load validation report
        val_path = Path("artifacts/concept_index_v3_validation.json")
        with open(val_path, 'r', encoding='utf-8') as f:
            val_report = json.load(f)

        # Load concept index
        idx_path = Path("data/evidence/concept_index_v3.jsonl")
        index_entries = {}
        with open(idx_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                index_entries[entry["concept_id"]] = entry

        # Verify counts match
        for result in val_report["results"]:
            behavior_id = result["behavior_id"]
            if behavior_id in index_entries:
                idx_count = len(index_entries[behavior_id].get("verses", []))
                val_count = result["total_verses"]
                assert idx_count == val_count, \
                    f"{behavior_id}: index has {idx_count} verses, validation checked {val_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
