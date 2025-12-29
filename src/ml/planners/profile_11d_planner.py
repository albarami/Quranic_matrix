"""
Phase 3: PROFILE_11D Planner (3.3)

Thin wrapper around LegendaryPlanner for 11-dimensional behavior profiling.
Uses Bouzidani taxonomy for axis classification.

REUSES:
- LegendaryPlanner.resolve_entities()
- LegendaryPlanner.get_concept_evidence()
- qbm_bouzidani_taxonomy.py for 11-axis classification

ADDS:
- Gap labeling when axis data missing (fail-closed, no invention)
- Structured 11-dimension output
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 11 dimensions from Bouzidani taxonomy
ELEVEN_DIMENSIONS = [
    "organ",           # العضوي - which organ (قلب، لسان، يد، etc.)
    "situational",     # الموقفي - context (داخلي، خارجي)
    "systemic",        # النظامي - scope (فردي، جماعي)
    "spatial",         # المكاني - location (مسجد، بيت، سوق)
    "temporal",        # الزماني - time (دنيا، آخرة، كلاهما)
    "agent",           # الفاعل - who does it (مؤمن، كافر، منافق)
    "source",          # المصدر - origin (نفس، شيطان، قرين)
    "evaluation",      # التقييم - assessment (محمود، مذموم، محايد)
    "heart_state",     # حالة القلب - heart state (سليم، قاسي، مريض)
    "consequence",     # العاقبة - outcome (جنة، نار، تائب)
    "relationships",   # العلاقات - relations (سببي، تعزيزي، تضاد)
]

DIMENSION_LABELS_AR = {
    "organ": "العضوي",
    "situational": "الموقفي",
    "systemic": "النظامي",
    "spatial": "المكاني",
    "temporal": "الزماني",
    "agent": "الفاعل",
    "source": "المصدر",
    "evaluation": "التقييم",
    "heart_state": "حالة القلب",
    "consequence": "العاقبة",
    "relationships": "العلاقات",
}


@dataclass
class DimensionValue:
    """Value for a single dimension."""
    dimension: str
    dimension_ar: str
    value: str
    value_ar: str = ""
    evidence_count: int = 0
    verse_keys: List[str] = field(default_factory=list)
    confidence: float = 1.0
    is_gap: bool = False


@dataclass
class Profile11DResult:
    """Result of 11-dimensional profile analysis."""
    entity_id: str
    entity_label_ar: str
    entity_label_en: str
    dimensions: List[DimensionValue]
    total_dimensions: int
    filled_dimensions: int
    completeness_ratio: float
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_label_ar": self.entity_label_ar,
            "entity_label_en": self.entity_label_en,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "dimension_ar": d.dimension_ar,
                    "value": d.value,
                    "value_ar": d.value_ar,
                    "evidence_count": d.evidence_count,
                    "verse_keys": d.verse_keys[:3],
                    "confidence": d.confidence,
                    "is_gap": d.is_gap,
                }
                for d in self.dimensions
            ],
            "total_dimensions": self.total_dimensions,
            "filled_dimensions": self.filled_dimensions,
            "completeness_ratio": self.completeness_ratio,
            "gaps": self.gaps,
        }


class Profile11DPlanner:
    """
    Planner for PROFILE_11D (behavior_profile_11axis) question class.

    Wraps LegendaryPlanner to provide:
    - 11-dimensional behavior profiling
    - Gap detection for missing dimensions
    - Evidence-backed dimension values
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner
        self._taxonomy_data = None

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def _load_taxonomy(self):
        """Load taxonomy data if available."""
        if self._taxonomy_data is not None:
            return

        taxonomy_file = Path("data/taxonomy/qbm_bouzidani_taxonomy.json")
        if taxonomy_file.exists():
            with open(taxonomy_file, "r", encoding="utf-8") as f:
                self._taxonomy_data = json.load(f)
        else:
            self._taxonomy_data = {}

    def profile_behavior(
        self,
        entity_id: str,
        entity_label_ar: str = "",
        entity_label_en: str = "",
    ) -> Profile11DResult:
        """
        Create 11-dimensional profile for a behavior.

        Args:
            entity_id: The canonical entity ID
            entity_label_ar: Arabic label
            entity_label_en: English label

        Returns:
            Profile11DResult with all 11 dimensions
        """
        self._ensure_planner()
        self._load_taxonomy()

        gaps = []
        dimensions = []

        # Get evidence from concept index
        evidence = self._planner.get_concept_evidence(entity_id)

        if evidence.get("status") == "not_found":
            gaps.append(f"entity_not_in_index:{entity_id}")

        # Get neighbors to infer dimension values
        neighbors = self._planner.get_semantic_neighbors(entity_id)

        # Extract dimension values from taxonomy or infer from neighbors
        taxonomy_entry = self._taxonomy_data.get(entity_id, {})

        for dim in ELEVEN_DIMENSIONS:
            dim_ar = DIMENSION_LABELS_AR.get(dim, dim)

            # Try to get from taxonomy
            value = taxonomy_entry.get(dim, "")
            value_ar = taxonomy_entry.get(f"{dim}_ar", "")
            evidence_count = 0
            verse_keys = []
            confidence = 1.0
            is_gap = False

            if not value:
                # Try to infer from neighbors
                inferred = self._infer_dimension_from_neighbors(dim, neighbors, entity_id)
                if inferred:
                    value = inferred.get("value", "")
                    value_ar = inferred.get("value_ar", "")
                    evidence_count = inferred.get("evidence_count", 0)
                    confidence = inferred.get("confidence", 0.5)

            if not value:
                # Mark as gap - fail-closed, no invention
                is_gap = True
                value = "-"
                value_ar = "-"
                gaps.append(f"dimension_missing:{dim}")

            # Get verse keys from evidence if available
            if evidence.get("verse_keys"):
                verse_keys = evidence["verse_keys"][:3]

            dimensions.append(DimensionValue(
                dimension=dim,
                dimension_ar=dim_ar,
                value=value,
                value_ar=value_ar,
                evidence_count=evidence_count,
                verse_keys=verse_keys,
                confidence=confidence,
                is_gap=is_gap,
            ))

        filled = sum(1 for d in dimensions if not d.is_gap)

        return Profile11DResult(
            entity_id=entity_id,
            entity_label_ar=entity_label_ar,
            entity_label_en=entity_label_en,
            dimensions=dimensions,
            total_dimensions=len(ELEVEN_DIMENSIONS),
            filled_dimensions=filled,
            completeness_ratio=filled / len(ELEVEN_DIMENSIONS),
            gaps=gaps,
        )

    def _infer_dimension_from_neighbors(
        self,
        dimension: str,
        neighbors: List[Dict[str, Any]],
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Infer dimension value from semantic graph neighbors.

        Returns None if cannot infer with confidence.
        """
        # Dimension-specific inference rules
        inference_rules = {
            "organ": ["HAS_ORGAN", "MANIFESTS_IN"],
            "agent": ["EXCLUSIVE_TO", "PERFORMED_BY"],
            "heart_state": ["RESULTS_IN", "INDICATES"],
            "consequence": ["LEADS_TO", "RESULTS_IN"],
            "evaluation": ["CLASSIFIED_AS"],
        }

        edge_types = inference_rules.get(dimension, [])
        if not edge_types:
            return None

        for neighbor in neighbors:
            if neighbor.get("edge_type") in edge_types:
                return {
                    "value": neighbor.get("entity_id", ""),
                    "value_ar": "",
                    "evidence_count": neighbor.get("evidence_count", 0),
                    "confidence": neighbor.get("confidence", 0.5),
                }

        return None

    def profile_from_query(self, query: str) -> List[Profile11DResult]:
        """
        Profile all behaviors mentioned in a query.

        Args:
            query: The query text

        Returns:
            List of Profile11DResult for each resolved entity
        """
        self._ensure_planner()

        resolution = self._planner.resolve_entities(query)
        results = []

        for entity in resolution.get("entities", []):
            if entity.get("entity_type") == "behavior":
                result = self.profile_behavior(
                    entity_id=entity.get("entity_id", ""),
                    entity_label_ar=entity.get("term", ""),
                )
                results.append(result)

        return results
