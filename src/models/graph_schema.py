#!/usr/bin/env python3
"""
Graph Schema for QBM Knowledge Graph

Defines node and edge types for the behavior knowledge graph.
All nodes and edges are derived from validated sources (SSOT).

Node Types:
- BEHAVIOR: A Quranic behavior (e.g., patience, gratitude)
- VERSE: A verse from the Quran
- TAFSIR_CHUNK: A tafsir commentary on a verse

Edge Types:
- MENTIONED_IN: Behavior is mentioned in a verse (with evidence type)
- EXPLAINED_BY: Verse is explained by tafsir
- RELATES_TO: Behavior relates to another behavior

Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

__version__ = "1.0.0"


class NodeType(str, Enum):
    """Types of nodes in the graph."""
    BEHAVIOR = "behavior"
    VERSE = "verse"
    TAFSIR_CHUNK = "tafsir_chunk"
    LEXEME = "lexeme"
    AGENT = "agent"
    ORGAN = "organ"
    CONSEQUENCE = "consequence"
    HEART_STATE = "heart_state"


class EdgeType(str, Enum):
    """Types of edges in the graph."""
    MENTIONED_IN = "MENTIONED_IN"       # Behavior -> Verse
    EXPLAINED_BY = "EXPLAINED_BY"       # Verse -> Tafsir
    HAS_LEXEME = "HAS_LEXEME"          # Verse -> Lexeme
    RELATES_TO = "RELATES_TO"          # Behavior <-> Behavior
    COMPLEMENTS = "COMPLEMENTS"         # Behavior <-> Behavior
    OPPOSES = "OPPOSES"                # Behavior <-> Behavior
    LEADS_TO = "LEADS_TO"              # Behavior -> Behavior


class EvidenceType(str, Enum):
    """Types of evidence for edges."""
    LEXICAL = "lexical"
    ANNOTATION = "annotation"
    SEMANTIC = "semantic"
    INFERRED = "inferred"


@dataclass
class GraphNode:
    """
    A node in the knowledge graph.

    Attributes:
        id: Unique identifier (e.g., "BEH_EMO_PATIENCE", "VERSE_2_45")
        type: Node type
        label_ar: Arabic label
        label_en: English label
        metadata: Additional metadata
    """
    id: str
    type: NodeType
    label_ar: str
    label_en: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "labelAr": self.label_ar,
            "labelEn": self.label_en,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            label_ar=data.get("labelAr", ""),
            label_en=data.get("labelEn", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class GraphEdge:
    """
    An edge in the knowledge graph.

    Attributes:
        source: Source node ID
        target: Target node ID
        type: Edge type
        evidence_type: Type of evidence supporting this edge
        weight: Edge weight (0-1)
        provenance: Source of this edge
    """
    source: str
    target: str
    type: EdgeType
    evidence_type: EvidenceType = EvidenceType.LEXICAL
    weight: float = 1.0
    provenance: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "evidenceType": self.evidence_type.value,
            "weight": self.weight,
            "provenance": self.provenance,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            type=EdgeType(data["type"]),
            evidence_type=EvidenceType(data.get("evidenceType", "lexical")),
            weight=data.get("weight", 1.0),
            provenance=data.get("provenance", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class KnowledgeGraph:
    """
    The complete knowledge graph.

    Attributes:
        version: Graph version
        nodes: All nodes
        edges: All edges
        metadata: Graph metadata
    """
    version: str = "3.0"
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges to a node."""
        return [e for e in self.edges if e.target == node_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "metadata": self.metadata,
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "nodes_by_type": self._count_nodes_by_type(),
                "edges_by_type": self._count_edges_by_type()
            },
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges]
        }

    def _count_nodes_by_type(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in self.nodes:
            type_name = node.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _count_edges_by_type(self) -> Dict[str, int]:
        """Count edges by type."""
        counts = {}
        for edge in self.edges:
            type_name = edge.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """Create from dictionary."""
        graph = cls(
            version=data.get("version", "3.0"),
            metadata=data.get("metadata", {})
        )

        for node_data in data.get("nodes", []):
            graph.add_node(GraphNode.from_dict(node_data))

        for edge_data in data.get("edges", []):
            graph.add_edge(GraphEdge.from_dict(edge_data))

        return graph


# ============================================================================
# Factory Functions
# ============================================================================

def create_behavior_node(
    behavior_id: str,
    label_ar: str,
    label_en: str,
    category: str = ""
) -> GraphNode:
    """Create a behavior node."""
    return GraphNode(
        id=behavior_id,
        type=NodeType.BEHAVIOR,
        label_ar=label_ar,
        label_en=label_en,
        metadata={"category": category}
    )


def create_verse_node(
    verse_key: str,
    text_uthmani: str = ""
) -> GraphNode:
    """Create a verse node."""
    surah, ayah = verse_key.split(":")
    return GraphNode(
        id=f"VERSE_{surah}_{ayah}",
        type=NodeType.VERSE,
        label_ar=verse_key,
        label_en=f"Verse {verse_key}",
        metadata={
            "verse_key": verse_key,
            "surah": int(surah),
            "ayah": int(ayah),
            "text_uthmani": text_uthmani[:100] if text_uthmani else ""
        }
    )


def create_mentioned_in_edge(
    behavior_id: str,
    verse_key: str,
    evidence_type: EvidenceType = EvidenceType.LEXICAL,
    weight: float = 1.0
) -> GraphEdge:
    """Create a MENTIONED_IN edge from behavior to verse."""
    surah, ayah = verse_key.split(":")
    verse_id = f"VERSE_{surah}_{ayah}"

    return GraphEdge(
        source=behavior_id,
        target=verse_id,
        type=EdgeType.MENTIONED_IN,
        evidence_type=evidence_type,
        weight=weight,
        provenance="concept_index_v3"
    )


if __name__ == "__main__":
    # Test the schema
    print("Graph Schema Test")
    print("=" * 50)

    graph = KnowledgeGraph()

    # Add behavior node
    patience = create_behavior_node(
        "BEH_EMO_PATIENCE",
        "الصبر",
        "Patience",
        "emotional"
    )
    graph.add_node(patience)

    # Add verse node
    verse_2_45 = create_verse_node("2:45")
    graph.add_node(verse_2_45)

    # Add edge
    edge = create_mentioned_in_edge("BEH_EMO_PATIENCE", "2:45")
    graph.add_edge(edge)

    # Print summary
    data = graph.to_dict()
    print(f"Nodes: {data['statistics']['total_nodes']}")
    print(f"Edges: {data['statistics']['total_edges']}")
    print(f"Sample node: {graph.nodes[0].to_dict()}")
    print(f"Sample edge: {graph.edges[0].to_dict()}")

    print("\nTest PASSED")
