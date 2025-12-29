"""
Tests for Truth Metrics v1 artifact.

Validates:
- Checksum exists and is valid
- Percentages sum to ~100 for each distribution
- Agent types come from real data (not all "unknown")
- Tafsir sources count == 7
"""

import json
import pytest
from pathlib import Path

TRUTH_METRICS_FILE = Path("data/metrics/truth_metrics_v1.json")


@pytest.fixture
def truth_metrics():
    """Load truth metrics artifact."""
    if not TRUTH_METRICS_FILE.exists():
        pytest.skip(f"Truth metrics file not found: {TRUTH_METRICS_FILE}")
    
    with open(TRUTH_METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
class TestTruthMetricsSchema:
    """Tests for truth metrics schema."""
    
    def test_schema_version_exists(self, truth_metrics):
        """Schema version must be present."""
        assert "schema_version" in truth_metrics
        assert truth_metrics["schema_version"] == "metrics_v1"
    
    def test_generated_at_exists(self, truth_metrics):
        """Generated timestamp must be present."""
        assert "generated_at" in truth_metrics
        assert truth_metrics["generated_at"].endswith("Z")
    
    def test_build_version_exists(self, truth_metrics):
        """Build version (git SHA) must be present."""
        assert "build_version" in truth_metrics
        assert len(truth_metrics["build_version"]) > 0
    
    def test_source_files_exists(self, truth_metrics):
        """Source files list must be present."""
        assert "source_files" in truth_metrics
        assert len(truth_metrics["source_files"]) > 0
    
    def test_checksum_exists(self, truth_metrics):
        """Checksum must be present and valid format."""
        assert "checksum" in truth_metrics
        checksum = truth_metrics["checksum"]
        assert len(checksum) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in checksum)
    
    def test_status_is_ready(self, truth_metrics):
        """Status must be 'ready'."""
        assert "status" in truth_metrics
        assert truth_metrics["status"] == "ready"
    
    def test_metrics_object_exists(self, truth_metrics):
        """Metrics object must be present."""
        assert "metrics" in truth_metrics
        assert isinstance(truth_metrics["metrics"], dict)


@pytest.mark.unit
class TestTruthMetricsTotals:
    """Tests for totals in truth metrics."""
    
    def test_totals_exists(self, truth_metrics):
        """Totals section must exist."""
        assert "totals" in truth_metrics["metrics"]
    
    def test_spans_count_positive(self, truth_metrics):
        """Spans count must be positive."""
        totals = truth_metrics["metrics"]["totals"]
        assert "spans" in totals
        assert totals["spans"] > 0
        # Should be 6236 based on canonical dataset
        assert totals["spans"] == 6236
    
    def test_unique_verse_keys_positive(self, truth_metrics):
        """Unique verse keys must be positive."""
        totals = truth_metrics["metrics"]["totals"]
        assert "unique_verse_keys" in totals
        assert totals["unique_verse_keys"] > 0
    
    def test_tafsir_sources_count_is_seven(self, truth_metrics):
        """Tafsir sources count must be exactly 7."""
        totals = truth_metrics["metrics"]["totals"]
        assert "tafsir_sources_count" in totals
        assert totals["tafsir_sources_count"] == 7


@pytest.mark.unit
class TestAgentDistribution:
    """Tests for agent distribution."""
    
    def test_agent_distribution_exists(self, truth_metrics):
        """Agent distribution must exist."""
        assert "agent_distribution" in truth_metrics["metrics"]
    
    def test_agent_distribution_not_all_unknown(self, truth_metrics):
        """Agent distribution must not be all 'unknown'."""
        agent_dist = truth_metrics["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        
        # Must have more than one type
        assert len(items) > 1
        
        # First item must not be 'unknown' with 100%
        assert not (items[0]["key"] == "unknown" and items[0]["percentage"] > 99)
    
    def test_agent_distribution_has_real_types(self, truth_metrics):
        """Agent distribution must have real agent types."""
        agent_dist = truth_metrics["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        keys = [item["key"] for item in items]
        
        # Must have AGT_ALLAH (the most common)
        assert "AGT_ALLAH" in keys
        
        # Must have AGT_DISBELIEVER
        assert "AGT_DISBELIEVER" in keys
        
        # Must have AGT_BELIEVER
        assert "AGT_BELIEVER" in keys
    
    def test_agent_distribution_percentages_sum_to_100(self, truth_metrics):
        """Agent distribution percentages must sum to ~100."""
        agent_dist = truth_metrics["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        
        total_pct = sum(item["percentage"] for item in items)
        assert 99.5 <= total_pct <= 100.5, f"Percentages sum to {total_pct}, expected ~100"
    
    def test_agent_distribution_has_arabic_labels(self, truth_metrics):
        """Agent distribution items must have Arabic labels."""
        agent_dist = truth_metrics["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        
        for item in items:
            assert "label_ar" in item
            assert len(item["label_ar"]) > 0
    
    def test_agt_allah_is_largest(self, truth_metrics):
        """AGT_ALLAH must be the largest agent type."""
        agent_dist = truth_metrics["metrics"]["agent_distribution"]
        items = agent_dist["items"]
        
        # First item (sorted by count desc) should be AGT_ALLAH
        assert items[0]["key"] == "AGT_ALLAH"
        assert items[0]["percentage"] > 40  # Should be ~45.78%


@pytest.mark.unit
class TestBehaviorForms:
    """Tests for behavior forms distribution."""
    
    def test_behavior_forms_exists(self, truth_metrics):
        """Behavior forms must exist."""
        assert "behavior_forms" in truth_metrics["metrics"]
    
    def test_behavior_forms_percentages_sum_to_100(self, truth_metrics):
        """Behavior forms percentages must sum to ~100."""
        forms = truth_metrics["metrics"]["behavior_forms"]
        items = forms["items"]
        
        total_pct = sum(item["percentage"] for item in items)
        assert 99.5 <= total_pct <= 100.5, f"Percentages sum to {total_pct}, expected ~100"
    
    def test_inner_state_is_largest(self, truth_metrics):
        """inner_state must be the largest behavior form."""
        forms = truth_metrics["metrics"]["behavior_forms"]
        items = forms["items"]
        
        assert items[0]["key"] == "inner_state"
        assert items[0]["percentage"] > 50  # Should be ~51%


@pytest.mark.unit
class TestEvaluations:
    """Tests for evaluations distribution."""
    
    def test_evaluations_exists(self, truth_metrics):
        """Evaluations must exist."""
        assert "evaluations" in truth_metrics["metrics"]
    
    def test_evaluations_percentages_sum_to_100(self, truth_metrics):
        """Evaluations percentages must sum to ~100."""
        evals = truth_metrics["metrics"]["evaluations"]
        items = evals["items"]
        
        total_pct = sum(item["percentage"] for item in items)
        assert 99.5 <= total_pct <= 100.5, f"Percentages sum to {total_pct}, expected ~100"
    
    def test_neutral_is_largest(self, truth_metrics):
        """neutral must be the largest evaluation."""
        evals = truth_metrics["metrics"]["evaluations"]
        items = evals["items"]
        
        assert items[0]["key"] == "neutral"
        assert items[0]["percentage"] > 70  # Should be ~77%


@pytest.mark.unit
class TestChecksumIntegrity:
    """Tests for checksum integrity."""
    
    def test_checksum_matches_recomputation(self, truth_metrics):
        """Checksum must match when recomputed."""
        import hashlib
        
        metrics = truth_metrics["metrics"]
        payload_str = json.dumps(metrics, sort_keys=True, ensure_ascii=False)
        recomputed = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        
        assert truth_metrics["checksum"] == recomputed, "Checksum mismatch - data may be corrupted"
