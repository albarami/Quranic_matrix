"""
Phase 3: AGENT Planner (3.6)

Thin wrapper around LegendaryPlanner for agent attribution analysis.
Handles 14 canonical agents and their behavior mappings.

REUSES:
- LegendaryPlanner.canonical_entities (agents section)
- LegendaryPlanner.get_concept_evidence()
- LegendaryPlanner.get_semantic_neighbors()

ADDS:
- Agent→behavior mapping aggregation
- Agent inventory with evidence counts
- Behavior attribution analysis
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """A canonical agent with evidence."""
    agent_id: str
    label_ar: str
    label_en: str
    total_mentions: int
    verse_count: int
    sample_verses: List[str] = field(default_factory=list)


@dataclass
class AgentBehaviorMapping:
    """Mapping between an agent and a behavior."""
    agent_id: str
    behavior_id: str
    edge_type: str
    evidence_count: int
    confidence: float
    verse_keys: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of agent attribution analysis."""
    query_type: str  # "inventory", "attribution", "specific"
    agents: List[Agent]
    behavior_mappings: List[AgentBehaviorMapping]
    total_agents: int
    agents_with_evidence: int
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "label_ar": a.label_ar,
                    "label_en": a.label_en,
                    "total_mentions": a.total_mentions,
                    "verse_count": a.verse_count,
                    "sample_verses": a.sample_verses[:3],
                }
                for a in self.agents
            ],
            "behavior_mappings": [
                {
                    "agent_id": m.agent_id,
                    "behavior_id": m.behavior_id,
                    "edge_type": m.edge_type,
                    "evidence_count": m.evidence_count,
                    "confidence": m.confidence,
                }
                for m in self.behavior_mappings
            ],
            "total_agents": self.total_agents,
            "agents_with_evidence": self.agents_with_evidence,
            "gaps": self.gaps,
        }


class AgentPlanner:
    """
    Planner for AGENT_ATTRIBUTION question class.

    Wraps LegendaryPlanner to provide:
    - Agent inventory with evidence
    - Agent→behavior mapping aggregation
    - Behavior attribution analysis
    """

    # 14 canonical agents
    CANONICAL_AGENTS = [
        "AGT_BELIEVER", "AGT_DISBELIEVER", "AGT_HYPOCRITE", "AGT_POLYTHEIST",
        "AGT_HUMAN_GENERAL", "AGT_PROPHET_MESSENGER", "AGT_ANGEL", "AGT_JINN",
        "AGT_SHAYTAN", "AGT_ANIMAL", "AGT_PEOPLE_OF_BOOK", "AGT_RIGHTEOUS",
        "AGT_WRONGDOER", "AGT_ALLAH"
    ]

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def get_agent_inventory(self) -> AgentResult:
        """
        Get inventory of all canonical agents with evidence.

        Returns:
            AgentResult with all agents
        """
        self._ensure_planner()

        gaps = []
        agents = []

        # Get agents from canonical entities
        inventory = self._planner.enumerate_canonical_inventory("agents")

        if inventory.get("status") == "no_entities":
            gaps.append("canonical_entities_not_loaded")

        for item in inventory.get("items", []):
            agents.append(Agent(
                agent_id=item.get("id", ""),
                label_ar=item.get("ar", ""),
                label_en=item.get("en", ""),
                total_mentions=item.get("total_mentions", 0),
                verse_count=item.get("verse_count", 0),
                sample_verses=item.get("sample_verses", []),
            ))

        # Get behavior mappings for all agents
        behavior_mappings = self._get_all_agent_behaviors()

        agents_with_evidence = sum(1 for a in agents if a.total_mentions > 0)

        return AgentResult(
            query_type="inventory",
            agents=agents,
            behavior_mappings=behavior_mappings,
            total_agents=len(agents),
            agents_with_evidence=agents_with_evidence,
            gaps=gaps,
        )

    def _get_all_agent_behaviors(self) -> List[AgentBehaviorMapping]:
        """Get all agent→behavior mappings from semantic graph."""
        mappings = []

        if not self._planner.semantic_graph:
            return mappings

        # Get all agent IDs
        agent_ids = set(self.CANONICAL_AGENTS)

        # Find edges from agents to behaviors (PERFORMS, EXHIBITS)
        for edge in self._planner.semantic_graph.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            edge_type = edge.get("edge_type", "")

            # Agent → Behavior edges
            if src in agent_ids and edge_type in ["PERFORMS", "EXHIBITS", "CAUSES"]:
                mappings.append(AgentBehaviorMapping(
                    agent_id=src,
                    behavior_id=tgt,
                    edge_type=edge_type,
                    evidence_count=edge.get("evidence_count", 0),
                    confidence=edge.get("confidence", 0.0),
                ))

        return mappings

    def analyze_agent(self, agent_id: str) -> AgentResult:
        """
        Analyze a specific agent.

        Args:
            agent_id: The agent entity ID

        Returns:
            AgentResult focused on one agent
        """
        self._ensure_planner()

        gaps = []
        agents = []
        mappings = []

        # Get evidence for the agent
        evidence = self._planner.get_concept_evidence(agent_id)

        if evidence.get("status") == "not_found":
            gaps.append(f"agent_not_in_index:{agent_id}")
        else:
            agents.append(Agent(
                agent_id=agent_id,
                label_ar="",
                label_en="",
                total_mentions=evidence.get("total_mentions", 0),
                verse_count=len(evidence.get("verse_keys", [])),
                sample_verses=evidence.get("verse_keys", [])[:5],
            ))

        # Get behaviors attributed to this agent
        neighbors = self._planner.get_semantic_neighbors(agent_id)

        for neighbor in neighbors:
            edge_type = neighbor.get("edge_type", "")
            if edge_type in ["PERFORMS", "EXHIBITS", "CAUSES"]:
                mappings.append(AgentBehaviorMapping(
                    agent_id=agent_id,
                    behavior_id=neighbor.get("entity_id", ""),
                    edge_type=edge_type,
                    evidence_count=neighbor.get("evidence_count", 0),
                    confidence=neighbor.get("confidence", 0.0),
                ))

        return AgentResult(
            query_type="specific",
            agents=agents,
            behavior_mappings=mappings,
            total_agents=len(agents),
            agents_with_evidence=sum(1 for a in agents if a.total_mentions > 0),
            gaps=gaps,
        )

    def get_behavior_attribution(self, behavior_id: str) -> AgentResult:
        """
        Get all agents that perform a specific behavior.

        Args:
            behavior_id: The behavior entity ID

        Returns:
            AgentResult with agents attributed to this behavior
        """
        self._ensure_planner()

        gaps = []
        agents = []
        mappings = []

        # Get neighbors of the behavior to find agents
        neighbors = self._planner.get_semantic_neighbors(behavior_id)

        for neighbor in neighbors:
            entity_id = neighbor.get("entity_id", "")
            edge_type = neighbor.get("edge_type", "")

            # Check if this neighbor is an agent performing the behavior
            if entity_id.startswith("AGT_") or entity_id in self.CANONICAL_AGENTS:
                # Get evidence for this agent
                evidence = self._planner.get_concept_evidence(entity_id)

                agents.append(Agent(
                    agent_id=entity_id,
                    label_ar="",
                    label_en="",
                    total_mentions=evidence.get("total_mentions", 0),
                    verse_count=len(evidence.get("verse_keys", [])),
                    sample_verses=evidence.get("verse_keys", [])[:3],
                ))

                mappings.append(AgentBehaviorMapping(
                    agent_id=entity_id,
                    behavior_id=behavior_id,
                    edge_type=edge_type,
                    evidence_count=neighbor.get("evidence_count", 0),
                    confidence=neighbor.get("confidence", 0.0),
                ))

        if not agents:
            gaps.append(f"no_agents_found_for_behavior:{behavior_id}")

        return AgentResult(
            query_type="attribution",
            agents=agents,
            behavior_mappings=mappings,
            total_agents=len(agents),
            agents_with_evidence=sum(1 for a in agents if a.total_mentions > 0),
            gaps=gaps,
        )
