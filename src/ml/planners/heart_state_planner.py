"""
Phase 3: HEART_STATE Planner (3.5)

Thin wrapper around LegendaryPlanner for heart state analysis.
Handles 12 canonical heart states and transitions.

REUSES:
- LegendaryPlanner.canonical_entities (heart_states section)
- LegendaryPlanner.get_concept_evidence()
- LegendaryPlanner.get_semantic_neighbors()

ADDS:
- Transition graph from semantic graph edges (evidence-backed only)
- Heart state inventory with evidence counts
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class HeartState:
    """A canonical heart state with evidence."""
    state_id: str
    label_ar: str
    label_en: str
    category: str
    total_mentions: int
    verse_count: int
    sample_verses: List[str] = field(default_factory=list)


@dataclass
class HeartTransition:
    """A transition between heart states."""
    from_state: str
    to_state: str
    edge_type: str
    evidence_count: int
    confidence: float
    verse_keys: List[str] = field(default_factory=list)


@dataclass
class HeartStateResult:
    """Result of heart state analysis."""
    query_type: str  # "inventory", "transition", "specific"
    states: List[HeartState]
    transitions: List[HeartTransition]
    total_states: int
    states_with_evidence: int
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "states": [
                {
                    "state_id": s.state_id,
                    "label_ar": s.label_ar,
                    "label_en": s.label_en,
                    "category": s.category,
                    "total_mentions": s.total_mentions,
                    "verse_count": s.verse_count,
                    "sample_verses": s.sample_verses[:3],
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "edge_type": t.edge_type,
                    "evidence_count": t.evidence_count,
                    "confidence": t.confidence,
                }
                for t in self.transitions
            ],
            "total_states": self.total_states,
            "states_with_evidence": self.states_with_evidence,
            "gaps": self.gaps,
        }


class HeartStatePlanner:
    """
    Planner for HEART_STATE (state_transition) question class.

    Wraps LegendaryPlanner to provide:
    - Heart state inventory with evidence
    - State transition analysis
    - Evidence-backed transitions only
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def get_heart_state_inventory(self) -> HeartStateResult:
        """
        Get inventory of all canonical heart states with evidence.

        Returns:
            HeartStateResult with all heart states
        """
        self._ensure_planner()

        gaps = []
        states = []

        # Get heart states from canonical entities
        inventory = self._planner.enumerate_canonical_inventory("heart_states")

        if inventory.get("status") == "no_entities":
            gaps.append("canonical_entities_not_loaded")

        for item in inventory.get("items", []):
            states.append(HeartState(
                state_id=item.get("id", ""),
                label_ar=item.get("ar", ""),
                label_en=item.get("en", ""),
                category=item.get("category", ""),
                total_mentions=item.get("total_mentions", 0),
                verse_count=item.get("verse_count", 0),
                sample_verses=item.get("sample_verses", []),
            ))

        # Get transitions between heart states
        transitions = self._get_heart_transitions()

        states_with_evidence = sum(1 for s in states if s.total_mentions > 0)

        return HeartStateResult(
            query_type="inventory",
            states=states,
            transitions=transitions,
            total_states=len(states),
            states_with_evidence=states_with_evidence,
            gaps=gaps,
        )

    def _get_heart_transitions(self) -> List[HeartTransition]:
        """Get transitions between heart states from semantic graph."""
        transitions = []

        if not self._planner.semantic_graph:
            return transitions

        # Get all heart state IDs
        heart_state_ids = set()
        if self._planner.canonical_entities:
            for item in self._planner.canonical_entities.get("heart_states", []):
                heart_state_ids.add(item.get("id", ""))

        # Find edges between heart states
        for edge in self._planner.semantic_graph.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")

            if src in heart_state_ids and tgt in heart_state_ids:
                transitions.append(HeartTransition(
                    from_state=src,
                    to_state=tgt,
                    edge_type=edge.get("edge_type", ""),
                    evidence_count=edge.get("evidence_count", 0),
                    confidence=edge.get("confidence", 0.0),
                ))

        return transitions

    def analyze_heart_state(self, state_id: str) -> HeartStateResult:
        """
        Analyze a specific heart state.

        Args:
            state_id: The heart state entity ID

        Returns:
            HeartStateResult focused on one state
        """
        self._ensure_planner()

        gaps = []
        states = []
        transitions = []

        # Get evidence for the state
        evidence = self._planner.get_concept_evidence(state_id)

        if evidence.get("status") == "not_found":
            gaps.append(f"heart_state_not_in_index:{state_id}")
        else:
            states.append(HeartState(
                state_id=state_id,
                label_ar="",
                label_en="",
                category="",
                total_mentions=evidence.get("total_mentions", 0),
                verse_count=len(evidence.get("verse_keys", [])),
                sample_verses=evidence.get("verse_keys", [])[:5],
            ))

        # Get transitions involving this state
        neighbors = self._planner.get_semantic_neighbors(state_id)

        for neighbor in neighbors:
            transitions.append(HeartTransition(
                from_state=state_id if neighbor.get("direction") == "outgoing" else neighbor.get("entity_id", ""),
                to_state=neighbor.get("entity_id", "") if neighbor.get("direction") == "outgoing" else state_id,
                edge_type=neighbor.get("edge_type", ""),
                evidence_count=neighbor.get("evidence_count", 0),
                confidence=neighbor.get("confidence", 0.0),
            ))

        return HeartStateResult(
            query_type="specific",
            states=states,
            transitions=transitions,
            total_states=len(states),
            states_with_evidence=sum(1 for s in states if s.total_mentions > 0),
            gaps=gaps,
        )
