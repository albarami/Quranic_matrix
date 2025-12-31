"""
Unit tests for the Dataset Mode Contract.

This test file validates the dataset mode contract logic without any IO.
It ensures the two-lane pipeline expectations are correctly defined.

Gate: pytest -q tests/test_dataset_mode_contract.py
"""

import os
import pytest


class TestDatasetModeContract:
    """Test the dataset mode contract logic (no IO)."""
    
    def setup_method(self):
        """Save original environment state."""
        self._original_env = {
            "QBM_DATASET_MODE": os.environ.get("QBM_DATASET_MODE"),
            "QBM_DATA_PROFILE": os.environ.get("QBM_DATA_PROFILE"),
            "QBM_USE_FIXTURE": os.environ.get("QBM_USE_FIXTURE"),
        }
    
    def teardown_method(self):
        """Restore original environment state."""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def test_dataset_mode_enum_values(self):
        """Test DatasetMode enum has correct values."""
        from src.core.data_profile import DatasetMode
        
        assert DatasetMode.FIXTURE.value == "fixture"
        assert DatasetMode.FULL.value == "full"
    
    def test_get_dataset_mode_fixture_via_primary_env(self):
        """Test QBM_DATASET_MODE=fixture is detected."""
        from src.core.data_profile import get_dataset_mode, DatasetMode
        
        os.environ["QBM_DATASET_MODE"] = "fixture"
        os.environ.pop("QBM_DATA_PROFILE", None)
        os.environ.pop("QBM_USE_FIXTURE", None)
        
        assert get_dataset_mode() == DatasetMode.FIXTURE
    
    def test_get_dataset_mode_full_via_primary_env(self):
        """Test QBM_DATASET_MODE=full is detected."""
        from src.core.data_profile import get_dataset_mode, DatasetMode
        
        os.environ["QBM_DATASET_MODE"] = "full"
        os.environ.pop("QBM_DATA_PROFILE", None)
        os.environ.pop("QBM_USE_FIXTURE", None)
        
        assert get_dataset_mode() == DatasetMode.FULL
    
    def test_get_dataset_mode_fixture_via_legacy_flag(self):
        """Test QBM_USE_FIXTURE=1 maps to fixture mode."""
        from src.core.data_profile import get_dataset_mode, DatasetMode
        
        os.environ.pop("QBM_DATASET_MODE", None)
        os.environ.pop("QBM_DATA_PROFILE", None)
        os.environ["QBM_USE_FIXTURE"] = "1"
        
        assert get_dataset_mode() == DatasetMode.FIXTURE
    
    def test_get_dataset_mode_default_is_full(self):
        """Test default mode is FULL (safe for local dev)."""
        from src.core.data_profile import get_dataset_mode, DatasetMode
        
        os.environ.pop("QBM_DATASET_MODE", None)
        os.environ.pop("QBM_DATA_PROFILE", None)
        os.environ.pop("QBM_USE_FIXTURE", None)
        
        assert get_dataset_mode() == DatasetMode.FULL
    
    def test_is_fixture_mode_helper(self):
        """Test is_fixture_mode() helper."""
        from src.core.data_profile import is_fixture_mode, set_fixture_mode
        
        set_fixture_mode(True)
        assert is_fixture_mode() is True
        
        set_fixture_mode(False)
        # After clearing, default is FULL
        assert is_fixture_mode() is False
    
    def test_is_full_mode_helper(self):
        """Test is_full_mode() helper."""
        from src.core.data_profile import is_full_mode, set_fixture_mode, set_full_mode
        
        set_full_mode()
        assert is_full_mode() is True
        
        set_fixture_mode(True)
        assert is_full_mode() is False
    
    def test_expected_behavior_count_fixture(self):
        """Test expected behavior count for FIXTURE mode."""
        from src.core.data_profile import (
            expected_behavior_count, DatasetMode,
            FIXTURE_MIN_BEHAVIORS
        )
        
        count = expected_behavior_count(DatasetMode.FIXTURE)
        assert count == FIXTURE_MIN_BEHAVIORS
        assert count == 20  # Explicit check
    
    def test_expected_behavior_count_full(self):
        """Test expected behavior count for FULL mode."""
        from src.core.data_profile import (
            expected_behavior_count, DatasetMode,
            FULL_BEHAVIOR_COUNT
        )
        
        count = expected_behavior_count(DatasetMode.FULL)
        assert count == FULL_BEHAVIOR_COUNT
        assert count == 87  # Explicit check
    
    def test_expected_benchmark_questions_fixture(self):
        """Test expected benchmark questions for FIXTURE mode."""
        from src.core.data_profile import (
            expected_benchmark_questions, DatasetMode,
            FIXTURE_MIN_BENCHMARK_QUESTIONS
        )
        
        count = expected_benchmark_questions(DatasetMode.FIXTURE)
        assert count == FIXTURE_MIN_BENCHMARK_QUESTIONS
        assert count == 20  # Explicit check
    
    def test_expected_benchmark_questions_full(self):
        """Test expected benchmark questions for FULL mode."""
        from src.core.data_profile import (
            expected_benchmark_questions, DatasetMode,
            FULL_BENCHMARK_QUESTIONS
        )
        
        count = expected_benchmark_questions(DatasetMode.FULL)
        assert count == FULL_BENCHMARK_QUESTIONS
        assert count == 200  # Explicit check
    
    def test_requires_full_ssot_fixture(self):
        """Test FIXTURE mode does not require full SSOT."""
        from src.core.data_profile import requires_full_ssot, DatasetMode
        
        assert requires_full_ssot(DatasetMode.FIXTURE) is False
    
    def test_requires_full_ssot_full(self):
        """Test FULL mode requires full SSOT."""
        from src.core.data_profile import requires_full_ssot, DatasetMode
        
        assert requires_full_ssot(DatasetMode.FULL) is True
    
    def test_get_mode_expectations_fixture(self):
        """Test get_mode_expectations for FIXTURE mode."""
        from src.core.data_profile import get_mode_expectations, DatasetMode
        
        exp = get_mode_expectations(DatasetMode.FIXTURE)
        
        assert exp["mode"] == "fixture"
        assert exp["behaviors"] == 20
        assert exp["benchmark_questions"] == 20
        assert exp["requires_full_ssot"] is False
        assert exp["is_fixture"] is True
        assert exp["is_full"] is False
        # Canonical counts should still be available for reference
        assert exp["canonical_behaviors"] == 87
        assert exp["canonical_benchmark"] == 200
    
    def test_get_mode_expectations_full(self):
        """Test get_mode_expectations for FULL mode."""
        from src.core.data_profile import get_mode_expectations, DatasetMode
        
        exp = get_mode_expectations(DatasetMode.FULL)
        
        assert exp["mode"] == "full"
        assert exp["behaviors"] == 87
        assert exp["benchmark_questions"] == 200
        assert exp["requires_full_ssot"] is True
        assert exp["is_fixture"] is False
        assert exp["is_full"] is True
    
    def test_canonical_counts_match_full_mode(self):
        """Test that canonical counts match FULL mode expectations."""
        from src.core.data_profile import (
            FULL_BEHAVIOR_COUNT, FULL_ORGAN_COUNT, FULL_AGENT_COUNT,
            FULL_HEART_STATE_COUNT, FULL_CONSEQUENCE_COUNT, FULL_ENTITY_COUNT
        )
        
        # These are the canonical counts from vocab/canonical_entities.json
        assert FULL_BEHAVIOR_COUNT == 87
        assert FULL_ORGAN_COUNT == 40
        assert FULL_AGENT_COUNT == 14
        assert FULL_HEART_STATE_COUNT == 12
        assert FULL_CONSEQUENCE_COUNT == 16
        
        # Total should be sum of above
        expected_total = (
            FULL_BEHAVIOR_COUNT + FULL_ORGAN_COUNT + FULL_AGENT_COUNT +
            FULL_HEART_STATE_COUNT + FULL_CONSEQUENCE_COUNT
        )
        assert FULL_ENTITY_COUNT == expected_total
        assert FULL_ENTITY_COUNT == 169
    
    def test_fixture_thresholds_are_reasonable(self):
        """Test fixture thresholds are reasonable minimums."""
        from src.core.data_profile import (
            FIXTURE_MIN_BEHAVIORS, FIXTURE_MIN_BENCHMARK_QUESTIONS,
            FULL_BEHAVIOR_COUNT, FULL_BENCHMARK_QUESTIONS
        )
        
        # Fixture minimums should be less than full counts
        assert FIXTURE_MIN_BEHAVIORS < FULL_BEHAVIOR_COUNT
        assert FIXTURE_MIN_BENCHMARK_QUESTIONS < FULL_BENCHMARK_QUESTIONS
        
        # But not too small (at least 10% coverage)
        assert FIXTURE_MIN_BEHAVIORS >= FULL_BEHAVIOR_COUNT * 0.1
        assert FIXTURE_MIN_BENCHMARK_QUESTIONS >= FULL_BENCHMARK_QUESTIONS * 0.1


class TestDatasetModeContractIntegration:
    """Integration tests that verify contract with actual data."""
    
    def test_canonical_entities_json_matches_constants(self):
        """Verify canonical_entities.json matches our constants."""
        from src.utils.canonical_counts import get_canonical_counts
        from src.core.data_profile import (
            FULL_BEHAVIOR_COUNT, FULL_ORGAN_COUNT, FULL_AGENT_COUNT,
            FULL_HEART_STATE_COUNT, FULL_CONSEQUENCE_COUNT
        )
        
        counts = get_canonical_counts()
        
        assert counts["behaviors"] == FULL_BEHAVIOR_COUNT
        assert counts["organs"] == FULL_ORGAN_COUNT
        assert counts["agents"] == FULL_AGENT_COUNT
        assert counts["heart_states"] == FULL_HEART_STATE_COUNT
        assert counts["consequences"] == FULL_CONSEQUENCE_COUNT
