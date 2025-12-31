"""
Canonical Counts: Single source of truth for entity counts.

This module loads counts directly from vocab/canonical_entities.json
to ensure all tests, reports, and builders use consistent values.

Usage:
    from src.utils.canonical_counts import get_canonical_counts, get_all_behavior_ids
    
    counts = get_canonical_counts()
    # {'behaviors': 87, 'organs': 40, 'agents': 14, ...}
    
    behavior_ids = get_all_behavior_ids()
    # ['BEH_SPEECH_TRUTHFULNESS', 'BEH_COG_ARROGANCE', ...]
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any


# Path to canonical entities file
CANONICAL_ENTITIES_PATH = Path(__file__).parent.parent.parent / "vocab" / "canonical_entities.json"


@lru_cache(maxsize=1)
def _load_canonical_entities() -> Dict[str, Any]:
    """Load canonical entities from JSON file (cached)."""
    if not CANONICAL_ENTITIES_PATH.exists():
        raise FileNotFoundError(
            f"Canonical entities file not found: {CANONICAL_ENTITIES_PATH}\n"
            "This file is required for consistent entity counts."
        )
    
    with open(CANONICAL_ENTITIES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_canonical_counts() -> Dict[str, int]:
    """
    Get canonical entity counts from the authoritative source.
    
    Returns:
        Dictionary with counts for each entity type:
        - behaviors: 87
        - organs: 40
        - agents: 14
        - heart_states: 12
        - consequences: 16
        - total: sum of above (excluding axis_values)
    """
    data = _load_canonical_entities()
    entity_types = data.get("entity_types", {})
    
    counts = {
        "behaviors": entity_types.get("BEHAVIOR", {}).get("count", 0),
        "organs": entity_types.get("ORGAN", {}).get("count", 0),
        "agents": entity_types.get("AGENT", {}).get("count", 0),
        "heart_states": entity_types.get("HEART_STATE", {}).get("count", 0),
        "consequences": entity_types.get("CONSEQUENCE", {}).get("count", 0),
    }
    
    # Total excludes axis_values (they're classification metadata, not entities)
    counts["total"] = sum(counts.values())
    
    return counts


def get_all_behavior_ids() -> List[str]:
    """
    Get all canonical behavior IDs.
    
    Returns:
        List of behavior IDs (e.g., ['BEH_SPEECH_TRUTHFULNESS', ...])
    """
    data = _load_canonical_entities()
    behaviors = data.get("behaviors", [])
    return [b["id"] for b in behaviors if "id" in b]


def get_all_organ_ids() -> List[str]:
    """Get all canonical organ IDs."""
    data = _load_canonical_entities()
    organs = data.get("organs", [])
    return [o["id"] for o in organs if "id" in o]


def get_all_agent_ids() -> List[str]:
    """Get all canonical agent IDs."""
    data = _load_canonical_entities()
    agents = data.get("agents", [])
    return [a["id"] for a in agents if "id" in a]


def get_all_heart_state_ids() -> List[str]:
    """Get all canonical heart state IDs."""
    data = _load_canonical_entities()
    heart_states = data.get("heart_states", [])
    return [h["id"] for h in heart_states if "id" in h]


def get_all_consequence_ids() -> List[str]:
    """Get all canonical consequence IDs."""
    data = _load_canonical_entities()
    consequences = data.get("consequences", [])
    return [c["id"] for c in consequences if "id" in c]


def get_behavior_by_id(behavior_id: str) -> Dict[str, Any]:
    """Get a specific behavior by ID."""
    data = _load_canonical_entities()
    for b in data.get("behaviors", []):
        if b.get("id") == behavior_id:
            return b
    return {}


def validate_behavior_count(actual_count: int) -> bool:
    """Validate that actual count matches canonical count."""
    expected = get_canonical_counts()["behaviors"]
    return actual_count == expected


def get_missing_behaviors(actual_ids: List[str]) -> List[str]:
    """
    Get list of canonical behaviors missing from actual list.
    
    Args:
        actual_ids: List of behavior IDs present in the system
        
    Returns:
        List of missing behavior IDs
    """
    canonical_ids = set(get_all_behavior_ids())
    actual_set = set(actual_ids)
    return sorted(canonical_ids - actual_set)


def get_extra_behaviors(actual_ids: List[str]) -> List[str]:
    """
    Get list of behaviors in actual list but not in canonical.
    
    Args:
        actual_ids: List of behavior IDs present in the system
        
    Returns:
        List of extra behavior IDs (not in canonical)
    """
    canonical_ids = set(get_all_behavior_ids())
    actual_set = set(actual_ids)
    return sorted(actual_set - canonical_ids)


# For backwards compatibility with existing code
CANONICAL_COUNTS = get_canonical_counts()
