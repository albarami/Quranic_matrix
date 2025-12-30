#!/usr/bin/env python3
"""
Evidence Policy Schema for Behaviors

Every behavior MUST declare how its evidence is collected.
This ensures transparency and prevents mixing of evidence types.

Evidence Modes:
- LEXICAL: Found via token/root matching in normalized Quran text
- ANNOTATION: Found via human annotation (scholarly tagging)
- HYBRID: Both lexical AND annotation evidence required/allowed

Directness Levels:
- DIRECT: Explicit mention of the behavior term
- INDIRECT: Implied/contextual reference
- DERIVED: Scholarly inference from related concepts

Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

__version__ = "1.0.0"


class EvidenceMode(str, Enum):
    """How evidence is collected for a behavior."""
    LEXICAL = "lexical"         # Token/root matching only
    ANNOTATION = "annotation"   # Human annotation only
    HYBRID = "hybrid"          # Both methods


class DirectnessLevel(str, Enum):
    """How directly the behavior is referenced."""
    DIRECT = "direct"          # Explicit mention
    INDIRECT = "indirect"      # Implied/contextual
    DERIVED = "derived"        # Scholarly inference


@dataclass
class LexicalSpec:
    """
    Specification for lexical evidence collection.

    Defines exactly which patterns will match for this behavior.
    All patterns work on NORMALIZED Arabic text.
    """
    # Arabic roots (3-letter base forms)
    roots: List[str] = field(default_factory=list)

    # Specific normalized word forms to match
    forms: List[str] = field(default_factory=list)

    # Synonyms that also indicate this behavior
    synonyms: List[str] = field(default_factory=list)

    # Patterns to exclude (false positives)
    exclude_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "roots": self.roots,
            "forms": self.forms,
            "synonyms": self.synonyms,
            "exclude_patterns": self.exclude_patterns
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LexicalSpec':
        """Create from dictionary."""
        return cls(
            roots=data.get("roots", []),
            forms=data.get("forms", []),
            synonyms=data.get("synonyms", []),
            exclude_patterns=data.get("exclude_patterns", [])
        )

    def get_all_patterns(self) -> List[str]:
        """Get all patterns for matching (forms + synonyms)."""
        return self.forms + self.synonyms


@dataclass
class AnnotationSpec:
    """
    Specification for annotation-based evidence.

    Defines requirements for human-annotated evidence.
    """
    # What directness levels are allowed
    allowed_types: List[DirectnessLevel] = field(
        default_factory=lambda: [DirectnessLevel.DIRECT]
    )

    # Minimum confidence score (0.0 to 1.0)
    min_confidence: float = 0.0

    # Required annotators (if any)
    required_annotators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed_types": [t.value for t in self.allowed_types],
            "min_confidence": self.min_confidence,
            "required_annotators": self.required_annotators
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationSpec':
        """Create from dictionary."""
        return cls(
            allowed_types=[
                DirectnessLevel(t) for t in data.get("allowed_types", ["direct"])
            ],
            min_confidence=data.get("min_confidence", 0.0),
            required_annotators=data.get("required_annotators", [])
        )


@dataclass
class EvidencePolicy:
    """
    Complete evidence policy for a behavior.

    This determines HOW evidence is collected and validated.
    """
    # Primary mode of evidence collection
    mode: EvidenceMode

    # If True, lexical match is REQUIRED (even in hybrid mode)
    lexical_required: bool = False

    # Sources that must be consulted
    min_sources: List[str] = field(
        default_factory=lambda: ["quran_tokens_norm"]
    )

    # Lexical specification (if mode is lexical or hybrid)
    lexical_spec: Optional[LexicalSpec] = None

    # Annotation specification (if mode is annotation or hybrid)
    annotation_spec: Optional[AnnotationSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mode": self.mode.value,
            "lexical_required": self.lexical_required,
            "min_sources": self.min_sources
        }
        if self.lexical_spec:
            result["lexical_spec"] = self.lexical_spec.to_dict()
        if self.annotation_spec:
            result["annotation_spec"] = self.annotation_spec.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidencePolicy':
        """Create from dictionary."""
        lexical_spec = None
        if "lexical_spec" in data:
            lexical_spec = LexicalSpec.from_dict(data["lexical_spec"])

        annotation_spec = None
        if "annotation_spec" in data:
            annotation_spec = AnnotationSpec.from_dict(data["annotation_spec"])

        return cls(
            mode=EvidenceMode(data.get("mode", "lexical")),
            lexical_required=data.get("lexical_required", False),
            min_sources=data.get("min_sources", ["quran_tokens_norm"]),
            lexical_spec=lexical_spec,
            annotation_spec=annotation_spec
        )


@dataclass
class BehaviorDefinition:
    """
    Complete definition of a behavior.

    This is the authoritative record for each behavior in the system.
    """
    # Unique identifier (e.g., "BEH_EMO_PATIENCE")
    behavior_id: str

    # Arabic label
    label_ar: str

    # English label
    label_en: str

    # Category (e.g., "emotional", "social", "worship")
    category: str

    # Evidence policy (how to collect evidence)
    evidence_policy: EvidencePolicy

    # Bouzidani 11-axis classification (optional)
    bouzidani_axes: Dict[str, Any] = field(default_factory=dict)

    # Arabic description (optional)
    description_ar: str = ""

    # English description (optional)
    description_en: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "behavior_id": self.behavior_id,
            "label_ar": self.label_ar,
            "label_en": self.label_en,
            "category": self.category,
            "evidence_policy": self.evidence_policy.to_dict(),
            "bouzidani_axes": self.bouzidani_axes,
            "description_ar": self.description_ar,
            "description_en": self.description_en
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehaviorDefinition':
        """Create from dictionary."""
        return cls(
            behavior_id=data["behavior_id"],
            label_ar=data["label_ar"],
            label_en=data["label_en"],
            category=data.get("category", "general"),
            evidence_policy=EvidencePolicy.from_dict(data["evidence_policy"]),
            bouzidani_axes=data.get("bouzidani_axes", {}),
            description_ar=data.get("description_ar", ""),
            description_en=data.get("description_en", "")
        )

    def get_lexical_patterns(self) -> List[str]:
        """Get all lexical patterns for this behavior."""
        if self.evidence_policy.lexical_spec:
            return self.evidence_policy.lexical_spec.get_all_patterns()
        return []

    def is_lexical(self) -> bool:
        """Check if this behavior uses lexical evidence."""
        return self.evidence_policy.mode in [EvidenceMode.LEXICAL, EvidenceMode.HYBRID]

    def is_annotation(self) -> bool:
        """Check if this behavior uses annotation evidence."""
        return self.evidence_policy.mode in [EvidenceMode.ANNOTATION, EvidenceMode.HYBRID]


# ============================================================================
# Factory Functions
# ============================================================================

def create_lexical_behavior(
    behavior_id: str,
    label_ar: str,
    label_en: str,
    category: str,
    roots: List[str],
    forms: List[str],
    synonyms: List[str] = None,
    exclude: List[str] = None
) -> BehaviorDefinition:
    """
    Create a behavior with lexical evidence policy.

    Convenience function for creating lexical-only behaviors.
    """
    return BehaviorDefinition(
        behavior_id=behavior_id,
        label_ar=label_ar,
        label_en=label_en,
        category=category,
        evidence_policy=EvidencePolicy(
            mode=EvidenceMode.LEXICAL,
            lexical_required=True,
            lexical_spec=LexicalSpec(
                roots=roots,
                forms=forms,
                synonyms=synonyms or [],
                exclude_patterns=exclude or []
            )
        )
    )


def create_annotation_behavior(
    behavior_id: str,
    label_ar: str,
    label_en: str,
    category: str,
    allowed_types: List[DirectnessLevel] = None
) -> BehaviorDefinition:
    """
    Create a behavior with annotation-only evidence policy.

    For behaviors that cannot be detected lexically.
    """
    return BehaviorDefinition(
        behavior_id=behavior_id,
        label_ar=label_ar,
        label_en=label_en,
        category=category,
        evidence_policy=EvidencePolicy(
            mode=EvidenceMode.ANNOTATION,
            lexical_required=False,
            annotation_spec=AnnotationSpec(
                allowed_types=allowed_types or [DirectnessLevel.DIRECT]
            )
        )
    )
