"""
Data Profile: Unified switch for fixture vs full data mode.

This module provides a single source of truth for determining whether
the system is running in fixture mode (CI) or full data mode (release).

Environment Variables:
    QBM_DATASET_MODE: Primary switch - "fixture" or "full"
    QBM_DATA_PROFILE: Alias for QBM_DATASET_MODE
    QBM_USE_FIXTURE: Legacy flag - "1" maps to fixture mode

Usage:
    from src.core.data_profile import is_fixture_mode, get_dataset_mode
    
    if is_fixture_mode():
        # Use fixture data (CI on main)
    else:
        # Use full data (tags/releases)

Rules:
    - CI on main always runs fixture mode
    - Release workflows (tags) run full mode
    - Audit pack --strict only runs in full mode
"""

import os
from enum import Enum
from typing import Optional


class DatasetMode(Enum):
    """Dataset mode for CI vs Release builds."""
    FIXTURE = "fixture"  # CI mode - minimal fixture data, fast, deterministic
    FULL = "full"        # Release mode - full SSOT data pack required


# Alias for backwards compatibility
DataProfile = DatasetMode


def get_dataset_mode() -> DatasetMode:
    """
    Get the current dataset mode.
    
    Resolution order:
    1. QBM_DATASET_MODE env var (primary)
    2. QBM_DATA_PROFILE env var (alias)
    3. QBM_USE_FIXTURE env var (legacy, maps to fixture)
    4. Default to FULL (safe default for local dev)
    
    Returns:
        DatasetMode.FIXTURE or DatasetMode.FULL
    """
    # Check primary dataset mode
    mode = os.getenv("QBM_DATASET_MODE", "").lower()
    if mode == "fixture":
        return DatasetMode.FIXTURE
    elif mode == "full":
        return DatasetMode.FULL
    
    # Check alias
    profile = os.getenv("QBM_DATA_PROFILE", "").lower()
    if profile == "fixture":
        return DatasetMode.FIXTURE
    elif profile == "full":
        return DatasetMode.FULL
    
    # Check legacy fixture flag
    if os.getenv("QBM_USE_FIXTURE", "0") == "1":
        return DatasetMode.FIXTURE
    
    # Default to full (safe for local development)
    return DatasetMode.FULL


# Alias for backwards compatibility
def get_data_profile() -> DatasetMode:
    """Alias for get_dataset_mode() - backwards compatibility."""
    return get_dataset_mode()


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


# Import canonical counts from the authoritative source
def get_canonical_counts() -> dict:
    """
    Get canonical entity counts from vocab/canonical_entities.json.
    
    This is the single source of truth for all entity counts.
    """
    from src.utils.canonical_counts import get_canonical_counts as _get_counts
    return _get_counts()


# For backwards compatibility - lazy loaded
def _get_canonical_counts_dict() -> dict:
    return get_canonical_counts()


# Expose as module-level constant (computed on first access)
class _CanonicalCountsProxy:
    """Lazy proxy for canonical counts."""
    _cache = None
    
    def __getitem__(self, key):
        if self._cache is None:
            self._cache = get_canonical_counts()
        return self._cache[key]
    
    def get(self, key, default=None):
        if self._cache is None:
            self._cache = get_canonical_counts()
        return self._cache.get(key, default)
    
    def copy(self):
        if self._cache is None:
            self._cache = get_canonical_counts()
        return self._cache.copy()
    
    def __repr__(self):
        return repr(get_canonical_counts())


CANONICAL_COUNTS = _CanonicalCountsProxy()
