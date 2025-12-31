"""
Dataset Mode Test Helpers.

This module provides pytest decorators and helpers for writing mode-aware tests.
Tests can use these to properly handle FIXTURE vs FULL mode expectations.

Usage:
    from tests.helpers.dataset_mode import skip_if_not_full, expected_counts
    
    @skip_if_not_full("Requires all 87 behaviors")
    def test_full_behavior_coverage():
        ...
    
    def test_behavior_count():
        counts = expected_counts()
        assert actual_count >= counts["min_behaviors"]
"""

import functools
import pytest
from typing import Dict, Any, Callable

from src.core.data_profile import (
    get_dataset_mode,
    DatasetMode,
    is_fixture_mode,
    is_full_mode,
    expected_behavior_count,
    expected_entity_count,
    expected_benchmark_questions,
    requires_full_ssot,
    get_mode_expectations,
    FULL_BEHAVIOR_COUNT,
    FULL_ENTITY_COUNT,
    FULL_BENCHMARK_QUESTIONS,
    FIXTURE_MIN_BEHAVIORS,
    FIXTURE_MIN_BENCHMARK_QUESTIONS,
)


def skip_if_not_full(reason: str = "Requires FULL dataset mode"):
    """
    Decorator to skip test if not running in FULL mode.
    
    Use this for tests that require:
    - All 87 behaviors present
    - Full SSOT (Quran + 7 tafsir files)
    - 200-question benchmark
    - Strict audit pack validation
    
    Args:
        reason: Explanation of why FULL mode is required
        
    Example:
        @skip_if_not_full("Requires all 87 behaviors in graph")
        def test_full_graph_coverage():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_fixture_mode():
                pytest.skip(f"FIXTURE mode: {reason}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_not_fixture(reason: str = "Requires FIXTURE dataset mode"):
    """
    Decorator to skip test if not running in FIXTURE mode.
    
    Use this for tests that specifically test fixture behavior.
    
    Args:
        reason: Explanation of why FIXTURE mode is required
        
    Example:
        @skip_if_not_fixture("Tests fixture-specific behavior")
        def test_fixture_bootstrap():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_full_mode():
                pytest.skip(f"FULL mode: {reason}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pytest markers for mode-aware tests
require_full_mode = pytest.mark.skipif(
    is_fixture_mode(),
    reason="Test requires FULL dataset mode (all 87 behaviors, full SSOT)"
)

require_fixture_mode = pytest.mark.skipif(
    is_full_mode(),
    reason="Test requires FIXTURE dataset mode"
)


def expected_counts() -> Dict[str, Any]:
    """
    Get expected counts for the current dataset mode.
    
    Returns a dictionary with:
    - mode: "fixture" or "full"
    - min_behaviors: Minimum expected behavior count
    - max_behaviors: Maximum expected behavior count (always 87)
    - min_benchmark_questions: Minimum expected benchmark questions
    - max_benchmark_questions: Maximum expected benchmark questions (always 200)
    - requires_full_ssot: Whether full SSOT is required
    
    Example:
        counts = expected_counts()
        assert actual_behaviors >= counts["min_behaviors"]
        assert actual_behaviors <= counts["max_behaviors"]
    """
    mode = get_dataset_mode()
    
    if mode == DatasetMode.FULL:
        return {
            "mode": "full",
            "min_behaviors": FULL_BEHAVIOR_COUNT,
            "max_behaviors": FULL_BEHAVIOR_COUNT,
            "exact_behaviors": FULL_BEHAVIOR_COUNT,
            "min_entities": FULL_ENTITY_COUNT,
            "max_entities": FULL_ENTITY_COUNT,
            "min_benchmark_questions": FULL_BENCHMARK_QUESTIONS,
            "max_benchmark_questions": FULL_BENCHMARK_QUESTIONS,
            "exact_benchmark_questions": FULL_BENCHMARK_QUESTIONS,
            "requires_full_ssot": True,
            "is_fixture": False,
            "is_full": True,
        }
    else:
        return {
            "mode": "fixture",
            "min_behaviors": FIXTURE_MIN_BEHAVIORS,
            "max_behaviors": FULL_BEHAVIOR_COUNT,  # Can have up to 87
            "exact_behaviors": None,  # No exact count in fixture mode
            "min_entities": FIXTURE_MIN_BEHAVIORS + 10,
            "max_entities": FULL_ENTITY_COUNT,
            "min_benchmark_questions": FIXTURE_MIN_BENCHMARK_QUESTIONS,
            "max_benchmark_questions": FULL_BENCHMARK_QUESTIONS,
            "exact_benchmark_questions": FIXTURE_MIN_BENCHMARK_QUESTIONS,
            "requires_full_ssot": False,
            "is_fixture": True,
            "is_full": False,
        }


def assert_behavior_count_valid(actual_count: int, context: str = ""):
    """
    Assert behavior count is valid for current mode.
    
    In FULL mode: must equal 87
    In FIXTURE mode: must be >= 20 and <= 87
    
    Args:
        actual_count: The actual behavior count to validate
        context: Optional context for error message
    """
    counts = expected_counts()
    prefix = f"{context}: " if context else ""
    
    if counts["is_full"]:
        assert actual_count == counts["exact_behaviors"], (
            f"{prefix}FULL mode requires exactly {counts['exact_behaviors']} behaviors, "
            f"got {actual_count}"
        )
    else:
        assert actual_count >= counts["min_behaviors"], (
            f"{prefix}FIXTURE mode requires at least {counts['min_behaviors']} behaviors, "
            f"got {actual_count}"
        )
        assert actual_count <= counts["max_behaviors"], (
            f"{prefix}Behavior count {actual_count} exceeds canonical maximum "
            f"{counts['max_behaviors']}"
        )


def assert_benchmark_questions_valid(actual_count: int, context: str = ""):
    """
    Assert benchmark question count is valid for current mode.
    
    In FULL mode: must equal 200
    In FIXTURE mode: must equal 20
    
    Args:
        actual_count: The actual question count to validate
        context: Optional context for error message
    """
    counts = expected_counts()
    prefix = f"{context}: " if context else ""
    
    expected = counts["exact_benchmark_questions"]
    if counts["is_full"]:
        expected = FULL_BENCHMARK_QUESTIONS
    else:
        expected = FIXTURE_MIN_BENCHMARK_QUESTIONS
    
    assert actual_count == expected, (
        f"{prefix}{counts['mode'].upper()} mode expects {expected} benchmark questions, "
        f"got {actual_count}"
    )


def get_benchmark_dataset_path() -> str:
    """
    Get the appropriate benchmark dataset path for current mode.
    
    Returns:
        Path to benchmark JSONL file (fixture or full)
    """
    if is_fixture_mode():
        return "data/benchmarks/benchmark_fixture.jsonl"
    else:
        return "data/benchmarks/qbm_legendary_200.v1.jsonl"
