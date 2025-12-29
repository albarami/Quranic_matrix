"""
Phase 3: TEMPORAL_SPATIAL Planner (3.7)

Thin wrapper around LegendaryPlanner for temporal and spatial analysis.
Handles temporal axes (9 items) and spatial axes (8 items).

REUSES:
- vocab/temporal.json
- vocab/spatial.json
- LegendaryPlanner.get_concept_evidence()

ADDS:
- Mapping aggregation for payload output
- Combined temporal-spatial analysis
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TemporalAxis:
    """A temporal axis value with evidence."""
    axis_id: str
    label_ar: str
    label_en: str
    quranic_terms: List[str]
    total_mentions: int
    verse_count: int
    sample_verses: List[str] = field(default_factory=list)


@dataclass
class SpatialAxis:
    """A spatial axis value with evidence."""
    axis_id: str
    label_ar: str
    label_en: str
    quranic_terms: List[str]
    total_mentions: int
    verse_count: int
    sample_verses: List[str] = field(default_factory=list)


@dataclass
class TemporalSpatialResult:
    """Result of temporal-spatial analysis."""
    query_type: str  # "temporal", "spatial", "combined"
    temporal_axes: List[TemporalAxis]
    spatial_axes: List[SpatialAxis]
    total_temporal: int
    total_spatial: int
    temporal_with_evidence: int
    spatial_with_evidence: int
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "temporal_axes": [
                {
                    "axis_id": t.axis_id,
                    "label_ar": t.label_ar,
                    "label_en": t.label_en,
                    "quranic_terms": t.quranic_terms,
                    "total_mentions": t.total_mentions,
                    "verse_count": t.verse_count,
                    "sample_verses": t.sample_verses[:3],
                }
                for t in self.temporal_axes
            ],
            "spatial_axes": [
                {
                    "axis_id": s.axis_id,
                    "label_ar": s.label_ar,
                    "label_en": s.label_en,
                    "quranic_terms": s.quranic_terms,
                    "total_mentions": s.total_mentions,
                    "verse_count": s.verse_count,
                    "sample_verses": s.sample_verses[:3],
                }
                for s in self.spatial_axes
            ],
            "total_temporal": self.total_temporal,
            "total_spatial": self.total_spatial,
            "temporal_with_evidence": self.temporal_with_evidence,
            "spatial_with_evidence": self.spatial_with_evidence,
            "gaps": self.gaps,
        }


class TemporalSpatialPlanner:
    """
    Planner for TEMPORAL_MAPPING and SPATIAL questions.

    Wraps LegendaryPlanner to provide:
    - Temporal axis enumeration with evidence
    - Spatial axis enumeration with evidence
    - Combined temporal-spatial analysis
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner
        self._temporal_vocab = None
        self._spatial_vocab = None

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def _load_temporal_vocab(self) -> List[Dict]:
        """Load temporal vocabulary from file."""
        if self._temporal_vocab is not None:
            return self._temporal_vocab

        vocab_path = Path("vocab/temporal.json")
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._temporal_vocab = data.get("items", [])
        else:
            self._temporal_vocab = []

        return self._temporal_vocab

    def _load_spatial_vocab(self) -> List[Dict]:
        """Load spatial vocabulary from file."""
        if self._spatial_vocab is not None:
            return self._spatial_vocab

        vocab_path = Path("vocab/spatial.json")
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._spatial_vocab = data.get("items", [])
        else:
            self._spatial_vocab = []

        return self._spatial_vocab

    def get_temporal_mapping(self) -> TemporalSpatialResult:
        """
        Get inventory of all temporal axes with evidence.

        Returns:
            TemporalSpatialResult with temporal axes
        """
        self._ensure_planner()

        gaps = []
        temporal_axes = []

        vocab = self._load_temporal_vocab()
        if not vocab:
            gaps.append("temporal_vocab_not_loaded")

        for item in vocab:
            axis_id = item.get("id", "")

            # Get evidence for this temporal axis
            evidence = self._planner.get_concept_evidence(axis_id)

            temporal_axes.append(TemporalAxis(
                axis_id=axis_id,
                label_ar=item.get("ar", ""),
                label_en=item.get("en", ""),
                quranic_terms=item.get("quranic_terms", []),
                total_mentions=evidence.get("total_mentions", 0),
                verse_count=len(evidence.get("verse_keys", [])),
                sample_verses=evidence.get("verse_keys", [])[:5],
            ))

        temporal_with_evidence = sum(1 for t in temporal_axes if t.total_mentions > 0)

        return TemporalSpatialResult(
            query_type="temporal",
            temporal_axes=temporal_axes,
            spatial_axes=[],
            total_temporal=len(temporal_axes),
            total_spatial=0,
            temporal_with_evidence=temporal_with_evidence,
            spatial_with_evidence=0,
            gaps=gaps,
        )

    def get_spatial_mapping(self) -> TemporalSpatialResult:
        """
        Get inventory of all spatial axes with evidence.

        Returns:
            TemporalSpatialResult with spatial axes
        """
        self._ensure_planner()

        gaps = []
        spatial_axes = []

        vocab = self._load_spatial_vocab()
        if not vocab:
            gaps.append("spatial_vocab_not_loaded")

        for item in vocab:
            axis_id = item.get("id", "")

            # Get evidence for this spatial axis
            evidence = self._planner.get_concept_evidence(axis_id)

            spatial_axes.append(SpatialAxis(
                axis_id=axis_id,
                label_ar=item.get("ar", ""),
                label_en=item.get("en", ""),
                quranic_terms=item.get("quranic_terms", []),
                total_mentions=evidence.get("total_mentions", 0),
                verse_count=len(evidence.get("verse_keys", [])),
                sample_verses=evidence.get("verse_keys", [])[:5],
            ))

        spatial_with_evidence = sum(1 for s in spatial_axes if s.total_mentions > 0)

        return TemporalSpatialResult(
            query_type="spatial",
            temporal_axes=[],
            spatial_axes=spatial_axes,
            total_temporal=0,
            total_spatial=len(spatial_axes),
            temporal_with_evidence=0,
            spatial_with_evidence=spatial_with_evidence,
            gaps=gaps,
        )

    def get_combined_mapping(self) -> TemporalSpatialResult:
        """
        Get combined temporal and spatial analysis.

        Returns:
            TemporalSpatialResult with both temporal and spatial axes
        """
        temporal_result = self.get_temporal_mapping()
        spatial_result = self.get_spatial_mapping()

        combined_gaps = temporal_result.gaps + spatial_result.gaps

        return TemporalSpatialResult(
            query_type="combined",
            temporal_axes=temporal_result.temporal_axes,
            spatial_axes=spatial_result.spatial_axes,
            total_temporal=temporal_result.total_temporal,
            total_spatial=spatial_result.total_spatial,
            temporal_with_evidence=temporal_result.temporal_with_evidence,
            spatial_with_evidence=spatial_result.spatial_with_evidence,
            gaps=combined_gaps,
        )

    def analyze_behavior_temporal_spatial(self, behavior_id: str) -> TemporalSpatialResult:
        """
        Analyze temporal and spatial contexts for a specific behavior.

        Args:
            behavior_id: The behavior entity ID

        Returns:
            TemporalSpatialResult with temporal/spatial contexts for the behavior
        """
        self._ensure_planner()

        gaps = []
        temporal_axes = []
        spatial_axes = []

        # Get neighbors of the behavior to find temporal/spatial contexts
        neighbors = self._planner.get_semantic_neighbors(behavior_id)

        temporal_vocab = {item.get("id"): item for item in self._load_temporal_vocab()}
        spatial_vocab = {item.get("id"): item for item in self._load_spatial_vocab()}

        for neighbor in neighbors:
            entity_id = neighbor.get("entity_id", "")

            if entity_id in temporal_vocab:
                item = temporal_vocab[entity_id]
                evidence = self._planner.get_concept_evidence(entity_id)

                temporal_axes.append(TemporalAxis(
                    axis_id=entity_id,
                    label_ar=item.get("ar", ""),
                    label_en=item.get("en", ""),
                    quranic_terms=item.get("quranic_terms", []),
                    total_mentions=evidence.get("total_mentions", 0),
                    verse_count=len(evidence.get("verse_keys", [])),
                    sample_verses=evidence.get("verse_keys", [])[:3],
                ))

            if entity_id in spatial_vocab:
                item = spatial_vocab[entity_id]
                evidence = self._planner.get_concept_evidence(entity_id)

                spatial_axes.append(SpatialAxis(
                    axis_id=entity_id,
                    label_ar=item.get("ar", ""),
                    label_en=item.get("en", ""),
                    quranic_terms=item.get("quranic_terms", []),
                    total_mentions=evidence.get("total_mentions", 0),
                    verse_count=len(evidence.get("verse_keys", [])),
                    sample_verses=evidence.get("verse_keys", [])[:3],
                ))

        if not temporal_axes and not spatial_axes:
            gaps.append(f"no_temporal_spatial_context_for:{behavior_id}")

        return TemporalSpatialResult(
            query_type="combined",
            temporal_axes=temporal_axes,
            spatial_axes=spatial_axes,
            total_temporal=len(temporal_axes),
            total_spatial=len(spatial_axes),
            temporal_with_evidence=sum(1 for t in temporal_axes if t.total_mentions > 0),
            spatial_with_evidence=sum(1 for s in spatial_axes if s.total_mentions > 0),
            gaps=gaps,
        )
