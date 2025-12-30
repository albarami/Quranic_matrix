#!/usr/bin/env python3
"""
Phase 6 Tests: Tafsir Coverage Audit

These tests validate that:
1. Tafsir coverage report exists
2. Average coverage meets threshold
3. Patience verses have full coverage
4. All configured sources are available

Run with: pytest tests/phase6/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Tafsir Report Tests
# ============================================================================

class TestTafsirReport:
    """Test tafsir coverage report exists and has correct structure."""

    @pytest.fixture
    def report(self):
        """Load tafsir coverage report."""
        report_path = Path("artifacts/tafsir_coverage_report.json")
        assert report_path.exists(), "Tafsir coverage report not found"
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_report_exists(self, report):
        """Test that report exists and loads."""
        assert report is not None

    def test_report_has_required_fields(self, report):
        """Test that report has required fields."""
        required = [
            "phase", "configured_sources", "available_sources",
            "statistics", "source_coverage", "patience_verses"
        ]
        for field in required:
            assert field in report, f"Missing field: {field}"

    def test_statistics_structure(self, report):
        """Test that statistics has correct structure."""
        stats = report["statistics"]
        required = [
            "total_verses_audited", "average_coverage",
            "full_coverage_count", "partial_coverage_count", "no_coverage_count"
        ]
        for field in required:
            assert field in stats, f"Missing statistic: {field}"


# ============================================================================
# Coverage Threshold Tests
# ============================================================================

class TestCoverageThresholds:
    """Test that coverage meets required thresholds."""

    @pytest.fixture
    def report(self):
        """Load tafsir coverage report."""
        report_path = Path("artifacts/tafsir_coverage_report.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_average_coverage_meets_threshold(self, report):
        """Test that average coverage >= 95%."""
        avg_coverage = report["statistics"]["average_coverage"]
        assert avg_coverage >= 0.95, f"Average coverage {avg_coverage:.1%} below 95%"

    def test_verses_audited_reasonable(self, report):
        """Test that a reasonable number of verses were audited."""
        total = report["statistics"]["total_verses_audited"]
        # Should have at least 1000 unique verses in concept index
        assert total >= 1000, f"Only {total} verses audited, expected >= 1000"

    def test_no_coverage_verses_minimal(self, report):
        """Test that very few verses have no coverage."""
        no_coverage = report["statistics"]["no_coverage_count"]
        total = report["statistics"]["total_verses_audited"]
        # Allow up to 5% with no coverage
        assert no_coverage <= total * 0.05, \
            f"{no_coverage} verses have no coverage, exceeds 5% threshold"


# ============================================================================
# Patience Verses Tests
# ============================================================================

class TestPatienceVersesCoverage:
    """Test patience (الصبر) verses tafsir coverage specifically."""

    @pytest.fixture
    def report(self):
        """Load tafsir coverage report."""
        report_path = Path("artifacts/tafsir_coverage_report.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_patience_verses_present(self, report):
        """Test that patience verses are in report."""
        patience_verses = ["2:45", "2:153", "3:200", "39:10", "103:3"]
        for pv in patience_verses:
            assert pv in report["patience_verses"], \
                f"Patience verse {pv} not in report"

    def test_patience_verses_full_coverage(self, report):
        """Test that key patience verses have full (7/7) coverage."""
        key_verses = ["2:45", "2:153", "3:200"]
        for pv in key_verses:
            coverage = report["patience_verses"][pv]["coverage"]
            assert coverage == 1.0, \
                f"Patience verse {pv} has only {coverage:.0%} coverage, expected 100%"

    def test_patience_verses_high_coverage(self, report):
        """Test that all patience verses have >= 90% coverage."""
        for pv, stats in report["patience_verses"].items():
            coverage = stats["coverage"]
            assert coverage >= 0.9, \
                f"Patience verse {pv} has only {coverage:.0%} coverage"


# ============================================================================
# Source Coverage Tests
# ============================================================================

class TestSourceCoverage:
    """Test individual tafsir source coverage."""

    @pytest.fixture
    def report(self):
        """Load tafsir coverage report."""
        report_path = Path("artifacts/tafsir_coverage_report.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_all_configured_sources_available(self, report):
        """Test that all configured sources are available."""
        configured = set(report["configured_sources"])
        available = set(report["available_sources"])
        missing = configured - available
        assert len(missing) == 0, f"Configured sources not available: {missing}"

    def test_primary_sources_present(self, report):
        """Test that primary tafsir sources are present."""
        primary = ["ibn_kathir", "tabari", "qurtubi"]
        available = report["available_sources"]
        for source in primary:
            assert source in available, f"Primary source {source} not available"

    def test_sources_have_reasonable_coverage(self, report):
        """Test that each source has >= 80% coverage."""
        for source, stats in report["source_coverage"].items():
            coverage_str = stats["coverage_rate"]
            coverage = float(coverage_str.rstrip('%')) / 100
            assert coverage >= 0.80, \
                f"Source {source} has only {coverage:.0%} coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
