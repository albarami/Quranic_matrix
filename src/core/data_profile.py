"""
Data Profile: Unified switch for fixture vs full data mode.

This module provides a single source of truth for determining whether
the system is running in fixture mode (CI) or full data mode (release).

Usage:
    from src.core.data_profile import is_fixture_mode, get_data_profile
    
    if is_fixture_mode():
        # Use fixture data
    else:
        # Use full data
"""

import os
from enum import Enum
from typing import Optional


class DataProfile(Enum):
    """Data profile modes."""
    FIXTURE = "fixture"  # CI mode - minimal fixture data
    FULL = "full"        # Release mode - full data pack


def get_data_profile() -> DataProfile:
    """
    Get the current data profile.
    
    Resolution order:
    1. QBM_DATA_PROFILE env var (explicit)
    2. QBM_USE_FIXTURE env var (legacy, maps to fixture)
    3. Default to FULL
    """
    # Check explicit profile
    profile = os.getenv("QBM_DATA_PROFILE", "").lower()
    if profile == "fixture":
        return DataProfile.FIXTURE
    elif profile == "full":
        return DataProfile.FULL
    
    # Check legacy fixture flag
    if os.getenv("QBM_USE_FIXTURE", "0") == "1":
        return DataProfile.FIXTURE
    
    # Default to full
    return DataProfile.FULL


def is_fixture_mode() -> bool:
    """Check if running in fixture mode (CI)."""
    return get_data_profile() == DataProfile.FIXTURE


def is_full_mode() -> bool:
    """Check if running in full data mode (release)."""
    return get_data_profile() == DataProfile.FULL


def set_fixture_mode(enabled: bool = True) -> None:
    """Set fixture mode (for testing)."""
    if enabled:
        os.environ["QBM_USE_FIXTURE"] = "1"
        os.environ["QBM_DATA_PROFILE"] = "fixture"
    else:
        os.environ.pop("QBM_USE_FIXTURE", None)
        os.environ.pop("QBM_DATA_PROFILE", None)


# Canonical entity counts (single source of truth)
CANONICAL_COUNTS = {
    "behaviors": 87,
    "organs": 39,
    "agents": 14,
    "heart_states": 12,
    "consequences": 16,
    "total": 168
}


def get_canonical_counts() -> dict:
    """Get canonical entity counts."""
    return CANONICAL_COUNTS.copy()
