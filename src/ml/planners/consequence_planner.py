"""
Phase 3: CONSEQUENCE Planner (3.8)

Thin wrapper around LegendaryPlanner for consequence analysis.
Handles 16 canonical consequences and behavior→consequence mappings.

REUSES:
- LegendaryPlanner.canonical_entities (consequences section)
- LegendaryPlanner.get_semantic_neighbors() with edge_types=["RESULTS_IN", "LEADS_TO"]

ADDS:
- Behavior→consequence mapping from graph edges
- Consequence inventory with evidence counts
- Polarity and temporal classification
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Consequence:
    """A canonical consequence with evidence."""
    consequence_id: str
    label_ar: str
    label_en: str
    temporal: str  # "dunya", "akhira", "both"
    polarity: str  # "positive", "negative", "neutral"
    total_mentions: int
    verse_count: int
    sample_verses: List[str] = field(default_factory=list)


@dataclass
class BehaviorConsequenceMapping:
    """Mapping between a behavior and a consequence."""
    behavior_id: str
    consequence_id: str
    edge_type: str
    evidence_count: int
    confidence: float
    verse_keys: List[str] = field(default_factory=list)


@dataclass
class ConsequenceResult:
    """Result of consequence analysis."""
    query_type: str  # "inventory", "mapping", "specific"
    consequences: List[Consequence]
    behavior_mappings: List[BehaviorConsequenceMapping]
    total_consequences: int
    consequences_with_evidence: int
    positive_count: int
    negative_count: int
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "consequences": [
                {
                    "consequence_id": c.consequence_id,
                    "label_ar": c.label_ar,
                    "label_en": c.label_en,
                    "temporal": c.temporal,
                    "polarity": c.polarity,
                    "total_mentions": c.total_mentions,
                    "verse_count": c.verse_count,
                    "sample_verses": c.sample_verses[:3],
                }
                for c in self.consequences
            ],
            "behavior_mappings": [
                {
                    "behavior_id": m.behavior_id,
                    "consequence_id": m.consequence_id,
                    "edge_type": m.edge_type,
                    "evidence_count": m.evidence_count,
                    "confidence": m.confidence,
                }
                for m in self.behavior_mappings
            ],
            "total_consequences": self.total_consequences,
            "consequences_with_evidence": self.consequences_with_evidence,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "gaps": self.gaps,
        }


class ConsequencePlanner:
    """
    Planner for CONSEQUENCE question class.

    Wraps LegendaryPlanner to provide:
    - Consequence inventory with evidence
    - Behavior→consequence mapping aggregation
    - Polarity and temporal analysis
    """

    # 16 canonical consequences
    CANONICAL_CONSEQUENCES = [
        "CSQ_JANNAH", "CSQ_NAAR", "CSQ_RIDWAN", "CSQ_GHADAB",
        "CSQ_MAGHFIRA", "CSQ_ADHAB", "CSQ_HIDAYA", "CSQ_DALAL",
        "CSQ_BARAKAH", "CSQ_BALAA", "CSQ_RIZQ", "CSQ_FAQR",
        "CSQ_IZZ", "CSQ_DHULL", "CSQ_SAKINA", "CSQ_QALAQ"
    ]

    # Consequence metadata
    CONSEQUENCE_META = {
        "CSQ_JANNAH": {"ar": "الجنة", "en": "Paradise", "temporal": "akhira", "polarity": "positive"},
        "CSQ_NAAR": {"ar": "النار", "en": "Hellfire", "temporal": "akhira", "polarity": "negative"},
        "CSQ_RIDWAN": {"ar": "رضوان الله", "en": "Allah's Pleasure", "temporal": "both", "polarity": "positive"},
        "CSQ_GHADAB": {"ar": "غضب الله", "en": "Allah's Wrath", "temporal": "both", "polarity": "negative"},
        "CSQ_MAGHFIRA": {"ar": "المغفرة", "en": "Forgiveness", "temporal": "both", "polarity": "positive"},
        "CSQ_ADHAB": {"ar": "العذاب", "en": "Punishment", "temporal": "both", "polarity": "negative"},
        "CSQ_HIDAYA": {"ar": "الهداية", "en": "Guidance", "temporal": "dunya", "polarity": "positive"},
        "CSQ_DALAL": {"ar": "الضلال", "en": "Misguidance", "temporal": "dunya", "polarity": "negative"},
        "CSQ_BARAKAH": {"ar": "البركة", "en": "Blessing", "temporal": "dunya", "polarity": "positive"},
        "CSQ_BALAA": {"ar": "البلاء", "en": "Trial/Affliction", "temporal": "dunya", "polarity": "neutral"},
        "CSQ_RIZQ": {"ar": "الرزق", "en": "Provision", "temporal": "dunya", "polarity": "positive"},
        "CSQ_FAQR": {"ar": "الفقر", "en": "Poverty", "temporal": "dunya", "polarity": "negative"},
        "CSQ_IZZ": {"ar": "العزة", "en": "Honor", "temporal": "both", "polarity": "positive"},
        "CSQ_DHULL": {"ar": "الذل", "en": "Humiliation", "temporal": "both", "polarity": "negative"},
        "CSQ_SAKINA": {"ar": "السكينة", "en": "Tranquility", "temporal": "dunya", "polarity": "positive"},
        "CSQ_QALAQ": {"ar": "القلق", "en": "Anxiety", "temporal": "dunya", "polarity": "negative"},
    }

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def get_consequence_inventory(self) -> ConsequenceResult:
        """
        Get inventory of all canonical consequences with evidence.

        Returns:
            ConsequenceResult with all consequences
        """
        self._ensure_planner()

        gaps = []
        consequences = []

        # Get consequences from canonical entities
        inventory = self._planner.enumerate_canonical_inventory("consequences")

        if inventory.get("status") == "no_entities":
            gaps.append("canonical_entities_not_loaded")

        for item in inventory.get("items", []):
            csq_id = item.get("id", "")
            meta = self.CONSEQUENCE_META.get(csq_id, {})

            consequences.append(Consequence(
                consequence_id=csq_id,
                label_ar=item.get("ar", meta.get("ar", "")),
                label_en=item.get("en", meta.get("en", "")),
                temporal=meta.get("temporal", "both"),
                polarity=meta.get("polarity", "neutral"),
                total_mentions=item.get("total_mentions", 0),
                verse_count=item.get("verse_count", 0),
                sample_verses=item.get("sample_verses", []),
            ))

        # Get behavior→consequence mappings
        behavior_mappings = self._get_all_behavior_consequences()

        consequences_with_evidence = sum(1 for c in consequences if c.total_mentions > 0)
        positive_count = sum(1 for c in consequences if c.polarity == "positive")
        negative_count = sum(1 for c in consequences if c.polarity == "negative")

        return ConsequenceResult(
            query_type="inventory",
            consequences=consequences,
            behavior_mappings=behavior_mappings,
            total_consequences=len(consequences),
            consequences_with_evidence=consequences_with_evidence,
            positive_count=positive_count,
            negative_count=negative_count,
            gaps=gaps,
        )

    def _get_all_behavior_consequences(self) -> List[BehaviorConsequenceMapping]:
        """Get all behavior→consequence mappings from semantic graph."""
        mappings = []

        if not self._planner.semantic_graph:
            return mappings

        # Get all consequence IDs
        consequence_ids = set(self.CANONICAL_CONSEQUENCES)

        # Find edges that lead to consequences (RESULTS_IN, LEADS_TO, CAUSES)
        for edge in self._planner.semantic_graph.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            edge_type = edge.get("edge_type", "")

            # Behavior → Consequence edges
            if tgt in consequence_ids and edge_type in ["RESULTS_IN", "LEADS_TO", "CAUSES"]:
                mappings.append(BehaviorConsequenceMapping(
                    behavior_id=src,
                    consequence_id=tgt,
                    edge_type=edge_type,
                    evidence_count=edge.get("evidence_count", 0),
                    confidence=edge.get("confidence", 0.0),
                ))

        return mappings

    def analyze_consequence(self, consequence_id: str) -> ConsequenceResult:
        """
        Analyze a specific consequence.

        Args:
            consequence_id: The consequence entity ID

        Returns:
            ConsequenceResult focused on one consequence
        """
        self._ensure_planner()

        gaps = []
        consequences = []
        mappings = []

        meta = self.CONSEQUENCE_META.get(consequence_id, {})

        # Get evidence for the consequence
        evidence = self._planner.get_concept_evidence(consequence_id)

        if evidence.get("status") == "not_found":
            gaps.append(f"consequence_not_in_index:{consequence_id}")
        else:
            consequences.append(Consequence(
                consequence_id=consequence_id,
                label_ar=meta.get("ar", ""),
                label_en=meta.get("en", ""),
                temporal=meta.get("temporal", "both"),
                polarity=meta.get("polarity", "neutral"),
                total_mentions=evidence.get("total_mentions", 0),
                verse_count=len(evidence.get("verse_keys", [])),
                sample_verses=evidence.get("verse_keys", [])[:5],
            ))

        # Get behaviors leading to this consequence
        neighbors = self._planner.get_semantic_neighbors(consequence_id)

        for neighbor in neighbors:
            edge_type = neighbor.get("edge_type", "")
            direction = neighbor.get("direction", "")

            # Look for incoming edges (behaviors leading to this consequence)
            if direction == "incoming" and edge_type in ["RESULTS_IN", "LEADS_TO", "CAUSES"]:
                mappings.append(BehaviorConsequenceMapping(
                    behavior_id=neighbor.get("entity_id", ""),
                    consequence_id=consequence_id,
                    edge_type=edge_type,
                    evidence_count=neighbor.get("evidence_count", 0),
                    confidence=neighbor.get("confidence", 0.0),
                ))

        positive_count = sum(1 for c in consequences if c.polarity == "positive")
        negative_count = sum(1 for c in consequences if c.polarity == "negative")

        return ConsequenceResult(
            query_type="specific",
            consequences=consequences,
            behavior_mappings=mappings,
            total_consequences=len(consequences),
            consequences_with_evidence=sum(1 for c in consequences if c.total_mentions > 0),
            positive_count=positive_count,
            negative_count=negative_count,
            gaps=gaps,
        )

    def get_behavior_consequences(self, behavior_id: str) -> ConsequenceResult:
        """
        Get all consequences of a specific behavior.

        Args:
            behavior_id: The behavior entity ID

        Returns:
            ConsequenceResult with consequences of this behavior
        """
        self._ensure_planner()

        gaps = []
        consequences = []
        mappings = []

        # Get neighbors of the behavior to find consequences
        neighbors = self._planner.get_semantic_neighbors(behavior_id)

        for neighbor in neighbors:
            entity_id = neighbor.get("entity_id", "")
            edge_type = neighbor.get("edge_type", "")

            # Check if this neighbor is a consequence
            if entity_id in self.CANONICAL_CONSEQUENCES or entity_id.startswith("CSQ_"):
                meta = self.CONSEQUENCE_META.get(entity_id, {})
                evidence = self._planner.get_concept_evidence(entity_id)

                consequences.append(Consequence(
                    consequence_id=entity_id,
                    label_ar=meta.get("ar", ""),
                    label_en=meta.get("en", ""),
                    temporal=meta.get("temporal", "both"),
                    polarity=meta.get("polarity", "neutral"),
                    total_mentions=evidence.get("total_mentions", 0),
                    verse_count=len(evidence.get("verse_keys", [])),
                    sample_verses=evidence.get("verse_keys", [])[:3],
                ))

                mappings.append(BehaviorConsequenceMapping(
                    behavior_id=behavior_id,
                    consequence_id=entity_id,
                    edge_type=edge_type,
                    evidence_count=neighbor.get("evidence_count", 0),
                    confidence=neighbor.get("confidence", 0.0),
                ))

        if not consequences:
            gaps.append(f"no_consequences_found_for_behavior:{behavior_id}")

        positive_count = sum(1 for c in consequences if c.polarity == "positive")
        negative_count = sum(1 for c in consequences if c.polarity == "negative")

        return ConsequenceResult(
            query_type="mapping",
            consequences=consequences,
            behavior_mappings=mappings,
            total_consequences=len(consequences),
            consequences_with_evidence=sum(1 for c in consequences if c.total_mentions > 0),
            positive_count=positive_count,
            negative_count=negative_count,
            gaps=gaps,
        )
