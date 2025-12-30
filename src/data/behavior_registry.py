#!/usr/bin/env python3
"""
Behavior Registry Loader

Loads and provides access to the behavior registry with evidence policies.

Usage:
    from src.data.behavior_registry import get_behavior_registry

    registry = get_behavior_registry()
    behavior = registry.get("BEH_EMO_PATIENCE")
    lexical_behaviors = registry.get_lexical_behaviors()

Version: 1.0.0
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.evidence_policy import (
    BehaviorDefinition,
    EvidencePolicy,
    EvidenceMode,
    LexicalSpec,
    AnnotationSpec
)

__version__ = "1.0.0"

# Default path to behavior registry
DEFAULT_REGISTRY_PATH = Path("data/behaviors/behavior_registry.json")


class BehaviorRegistry:
    """
    Registry of all behaviors with their evidence policies.

    Provides access to behavior definitions and filtering by various criteria.
    """

    def __init__(self, behaviors: Dict[str, BehaviorDefinition], metadata: Dict[str, Any] = None):
        """
        Initialize registry with behavior definitions.

        Args:
            behaviors: Dictionary mapping behavior_id to BehaviorDefinition
            metadata: Optional metadata about the registry
        """
        self.behaviors = behaviors
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'BehaviorRegistry':
        """
        Load registry from JSON file.

        Args:
            path: Path to registry JSON. Uses default if not specified.

        Returns:
            BehaviorRegistry instance
        """
        if path is None:
            path = DEFAULT_REGISTRY_PATH
        else:
            path = Path(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        behaviors = {}
        for b in data.get("behaviors", []):
            behavior_id = b["behavior_id"]
            behaviors[behavior_id] = cls._parse_behavior(b)

        metadata = {
            "version": data.get("version"),
            "total_behaviors": data.get("total_behaviors"),
            "lexical_behaviors": data.get("lexical_behaviors"),
            "annotation_behaviors": data.get("annotation_behaviors")
        }

        return cls(behaviors, metadata)

    @staticmethod
    def _parse_behavior(data: Dict[str, Any]) -> BehaviorDefinition:
        """Parse JSON to BehaviorDefinition."""
        # Parse evidence policy
        ep_data = data.get("evidence_policy", {})

        # Parse lexical spec if present
        lexical_spec = None
        if "lexical_spec" in ep_data:
            ls_data = ep_data["lexical_spec"]
            lexical_spec = LexicalSpec(
                roots=ls_data.get("roots", []),
                forms=ls_data.get("forms", []),
                synonyms=ls_data.get("synonyms", []),
                exclude_patterns=ls_data.get("exclude_patterns", [])
            )

        # Parse annotation spec if present
        annotation_spec = None
        if "annotation_spec" in ep_data:
            as_data = ep_data["annotation_spec"]
            annotation_spec = AnnotationSpec.from_dict(as_data)

        # Create evidence policy
        evidence_policy = EvidencePolicy(
            mode=EvidenceMode(ep_data.get("mode", "lexical")),
            lexical_required=ep_data.get("lexical_required", False),
            min_sources=ep_data.get("min_sources", ["quran_tokens_norm"]),
            lexical_spec=lexical_spec,
            annotation_spec=annotation_spec
        )

        return BehaviorDefinition(
            behavior_id=data["behavior_id"],
            label_ar=data["label_ar"],
            label_en=data["label_en"],
            category=data.get("category", "general"),
            evidence_policy=evidence_policy,
            bouzidani_axes=data.get("bouzidani_axes", {}),
            description_ar=data.get("description_ar", ""),
            description_en=data.get("description_en", "")
        )

    def get(self, behavior_id: str) -> Optional[BehaviorDefinition]:
        """
        Get behavior by ID.

        Args:
            behavior_id: Behavior identifier

        Returns:
            BehaviorDefinition or None if not found
        """
        return self.behaviors.get(behavior_id)

    def get_by_label_ar(self, label_ar: str) -> Optional[BehaviorDefinition]:
        """
        Get behavior by Arabic label.

        Args:
            label_ar: Arabic label

        Returns:
            BehaviorDefinition or None if not found
        """
        for b in self.behaviors.values():
            if b.label_ar == label_ar:
                return b
        return None

    def get_all(self) -> List[BehaviorDefinition]:
        """Get all behaviors."""
        return list(self.behaviors.values())

    def get_by_category(self, category: str) -> List[BehaviorDefinition]:
        """
        Get behaviors by category.

        Args:
            category: Category name

        Returns:
            List of matching behaviors
        """
        return [b for b in self.behaviors.values() if b.category == category]

    def get_lexical_behaviors(self) -> List[BehaviorDefinition]:
        """
        Get behaviors that use lexical evidence.

        Returns:
            List of behaviors with lexical or hybrid mode
        """
        return [b for b in self.behaviors.values() if b.is_lexical()]

    def get_annotation_behaviors(self) -> List[BehaviorDefinition]:
        """
        Get behaviors that use annotation evidence.

        Returns:
            List of behaviors with annotation or hybrid mode
        """
        return [b for b in self.behaviors.values() if b.is_annotation()]

    def get_lexical_required_behaviors(self) -> List[BehaviorDefinition]:
        """
        Get behaviors where lexical match is required.

        Returns:
            List of behaviors with lexical_required=True
        """
        return [
            b for b in self.behaviors.values()
            if b.evidence_policy.lexical_required
        ]

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return list(set(b.category for b in self.behaviors.values()))

    def count(self) -> int:
        """Get total number of behaviors."""
        return len(self.behaviors)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_mode = {}
        by_category = {}

        for b in self.behaviors.values():
            mode = b.evidence_policy.mode.value
            by_mode[mode] = by_mode.get(mode, 0) + 1

            by_category[b.category] = by_category.get(b.category, 0) + 1

        return {
            "total_behaviors": self.count(),
            "by_mode": by_mode,
            "by_category": by_category,
            "metadata": self.metadata
        }


# ============================================================================
# Singleton Pattern
# ============================================================================

_registry: Optional[BehaviorRegistry] = None


def get_behavior_registry(path: Optional[str] = None, force_reload: bool = False) -> BehaviorRegistry:
    """
    Get or create singleton BehaviorRegistry.

    Args:
        path: Optional path to registry JSON (only used on first load)
        force_reload: If True, reload even if already loaded

    Returns:
        BehaviorRegistry singleton instance
    """
    global _registry

    if _registry is None or force_reload:
        _registry = BehaviorRegistry.load(path)

    return _registry


def clear_registry():
    """Clear the singleton registry (useful for testing)."""
    global _registry
    _registry = None


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("BehaviorRegistry Self-Test")
    print("=" * 50)

    try:
        registry = get_behavior_registry()
        stats = registry.get_statistics()

        print(f"Total behaviors: {stats['total_behaviors']}")
        print(f"By mode: {stats['by_mode']}")
        print(f"Categories: {registry.get_categories()}")

        # Test patience lookup
        patience = registry.get_by_label_ar("الصبر")
        if patience:
            print(f"\nPatience behavior found: {patience.behavior_id}")
            print(f"  Mode: {patience.evidence_policy.mode.value}")
            print(f"  Lexical required: {patience.evidence_policy.lexical_required}")
            if patience.evidence_policy.lexical_spec:
                print(f"  Forms: {patience.evidence_policy.lexical_spec.forms[:3]}...")
        else:
            print("\nWARNING: Patience behavior not found!")

        # Test lexical behaviors
        lexical = registry.get_lexical_behaviors()
        print(f"\nLexical behaviors: {len(lexical)}")

        print("\n" + "=" * 50)
        print("Self-test PASSED")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
