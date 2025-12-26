"""
QBM Knowledge Graph using NetworkX with SQLite persistence.

This module implements the behavioral knowledge graph for the Quranic Behavior Matrix,
storing relationships between behaviors, ayat, agents, organs, and other entities.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


class QBMKnowledgeGraph:
    """Knowledge graph using NetworkX with SQLite persistence."""

    # Node types in the graph
    NODE_TYPES = {
        "Surah",
        "Ayah",
        "Span",
        "Behavior",
        "BehaviorCategory",
        "Agent",
        "Organ",
        "Source",
        "Context",
        "HeartType",
        "Tafsir",
        "Scholar",
        "Consequence",
    }

    # Edge types (relationships) in the graph
    EDGE_TYPES = {
        # Structural
        "CONTAINS",
        "HAS_SPAN",
        "ANNOTATED_AS",
        # Behavioral Relationships
        "CAUSES",
        "RESULTS_IN",
        "OPPOSITE_OF",
        "SIMILAR_TO",
        "LEADS_TO",
        # Contextual
        "PERFORMED_BY",
        "INVOLVES_ORGAN",
        "SOURCED_FROM",
        "EVALUATED_AS",
        # Heart-Personality-Behavior
        "CHARACTERISTIC_OF",
        "MANIFESTS_AS",
        # Tafsir
        "EXPLAINED_BY",
        "AUTHORED_BY",
        # Co-occurrence
        "CO_OCCURS_WITH",
        "CONTRASTED_WITH",
        # Generic
        "RELATED",
    }

    def __init__(self, db_path: str = "data/qbm_graph.db"):
        """
        Initialize the knowledge graph.

        Args:
            db_path: Path to SQLite database for persistence.
        """
        self.G = nx.MultiDiGraph()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables for persistence."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    properties JSON NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    properties JSON,
                    PRIMARY KEY (source, target, edge_type)
                );
                CREATE INDEX IF NOT EXISTS idx_node_type ON nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type);
                CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source);
                CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target);
            """
            )

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def add_node(
        self, node_id: str, node_type: str, **properties: Any
    ) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of node (must be in NODE_TYPES).
            **properties: Additional properties for the node.
        """
        if node_type not in self.NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {self.NODE_TYPES}")
        self.G.add_node(node_id, node_type=node_type, **properties)

    def add_behavior(
        self,
        code: str,
        name_ar: str,
        name_en: str,
        category: str,
        **properties: Any,
    ) -> None:
        """
        Add a behavior node.

        Args:
            code: Behavior code (e.g., BEH_COG_ARROGANCE).
            name_ar: Arabic name.
            name_en: English name.
            category: Behavior category.
            **properties: Additional properties.
        """
        self.G.add_node(
            code,
            node_type="Behavior",
            name_ar=name_ar,
            name_en=name_en,
            category=category,
            **properties,
        )

    def add_ayah(
        self,
        surah: int,
        ayah: int,
        text_uthmani: str,
        text_simple: Optional[str] = None,
        **properties: Any,
    ) -> None:
        """
        Add an ayah node.

        Args:
            surah: Surah number (1-114).
            ayah: Ayah number within the surah.
            text_uthmani: Uthmani text of the ayah.
            text_simple: Simplified text (optional).
            **properties: Additional properties.
        """
        ayah_id = f"{surah}:{ayah}"
        self.G.add_node(
            ayah_id,
            node_type="Ayah",
            surah=surah,
            ayah=ayah,
            text_uthmani=text_uthmani,
            text_simple=text_simple or text_uthmani,
            **properties,
        )

    def add_agent(self, agent_id: str, name_ar: str, agent_type: str, **properties: Any) -> None:
        """Add an agent node (e.g., مؤمن، كافر، منافق)."""
        self.G.add_node(
            agent_id,
            node_type="Agent",
            name_ar=name_ar,
            agent_type=agent_type,
            **properties,
        )

    def add_organ(self, organ_id: str, name_ar: str, name_en: str, **properties: Any) -> None:
        """Add an organ node (e.g., قلب، لسان، عين)."""
        self.G.add_node(
            organ_id,
            node_type="Organ",
            name_ar=name_ar,
            name_en=name_en,
            **properties,
        )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes by ID."""
        if node_id in self.G.nodes:
            return dict(self.G.nodes[node_id])
        return None

    def get_nodes_by_type(self, node_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all nodes of a specific type."""
        return [
            (node_id, dict(attrs))
            for node_id, attrs in self.G.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        **properties: Any,
    ) -> None:
        """
        Add a typed edge between nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            rel_type: Relationship type (must be in EDGE_TYPES).
            **properties: Additional edge properties.
        """
        if rel_type not in self.EDGE_TYPES:
            raise ValueError(f"Invalid edge type: {rel_type}. Must be one of {self.EDGE_TYPES}")
        self.G.add_edge(source, target, edge_type=rel_type, **properties)

    def add_causal_relationship(
        self, cause: str, effect: str, confidence: float = 1.0, **properties: Any
    ) -> None:
        """Add a CAUSES relationship between behaviors."""
        self.add_relationship(cause, effect, "CAUSES", confidence=confidence, **properties)

    def add_opposite_relationship(self, behavior1: str, behavior2: str, **properties: Any) -> None:
        """Add bidirectional OPPOSITE_OF relationship."""
        self.add_relationship(behavior1, behavior2, "OPPOSITE_OF", **properties)
        self.add_relationship(behavior2, behavior1, "OPPOSITE_OF", **properties)

    def get_relationships(
        self, node_id: str, rel_type: Optional[str] = None, direction: str = "both"
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get relationships for a node.

        Args:
            node_id: Node ID to get relationships for.
            rel_type: Filter by relationship type (optional).
            direction: 'in', 'out', or 'both'.

        Returns:
            List of (source, target, edge_data) tuples.
        """
        edges = []

        if direction in ("out", "both"):
            for _, target, data in self.G.out_edges(node_id, data=True):
                if rel_type is None or data.get("edge_type") == rel_type:
                    edges.append((node_id, target, dict(data)))

        if direction in ("in", "both"):
            for source, _, data in self.G.in_edges(node_id, data=True):
                if rel_type is None or data.get("edge_type") == rel_type:
                    edges.append((source, node_id, dict(data)))

        return edges

    # -------------------------------------------------------------------------
    # Path Finding & Analysis
    # -------------------------------------------------------------------------

    def find_causal_chain(
        self, start: str, end: str, max_depth: int = 5
    ) -> List[List[str]]:
        """
        Find all causal paths between two behaviors.

        Args:
            start: Starting behavior ID.
            end: Ending behavior ID.
            max_depth: Maximum path length.

        Returns:
            List of paths, where each path is a list of node IDs.
        """
        if start not in self.G or end not in self.G:
            return []

        paths = []
        try:
            for path in nx.all_simple_paths(self.G, start, end, cutoff=max_depth):
                if self._is_causal_path(path):
                    paths.append(path)
        except nx.NetworkXError:
            pass
        return paths

    def _is_causal_path(self, path: List[str]) -> bool:
        """Check if a path consists only of causal edges (ALL edges must be causal)."""
        causal_types = {"CAUSES", "RESULTS_IN", "LEADS_TO"}
        for i in range(len(path) - 1):
            edge_data = self.G.get_edge_data(path[i], path[i + 1])
            if edge_data is None:
                return False
            # MultiDiGraph returns dict of dicts - check if ANY edge between nodes is causal
            has_causal_edge = False
            for key, data in edge_data.items():
                if data.get("edge_type") in causal_types:
                    has_causal_edge = True
                    break
            if not has_causal_edge:
                return False
        return True

    def find_all_paths(
        self, start: str, end: str, max_depth: int = 5
    ) -> List[List[str]]:
        """Find all paths between two nodes."""
        if start not in self.G or end not in self.G:
            return []
        try:
            return list(nx.all_simple_paths(self.G, start, end, cutoff=max_depth))
        except nx.NetworkXError:
            return []

    # -------------------------------------------------------------------------
    # Graph Analytics
    # -------------------------------------------------------------------------

    def get_hub_behaviors(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most connected behaviors using betweenness centrality.

        Args:
            top_n: Number of top behaviors to return.

        Returns:
            List of (behavior_id, centrality_score) tuples.
        """
        # Filter to behavior nodes only
        behavior_nodes = [
            n for n, d in self.G.nodes(data=True) if d.get("node_type") == "Behavior"
        ]

        if len(behavior_nodes) < 2:
            return [(n, 0.0) for n in behavior_nodes]

        # Create subgraph of behaviors
        subgraph = self.G.subgraph(behavior_nodes)

        if subgraph.number_of_edges() == 0:
            return [(n, 0.0) for n in behavior_nodes[:top_n]]

        centrality = nx.betweenness_centrality(subgraph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def find_communities(self) -> List[Set[str]]:
        """
        Detect behavioral clusters using Louvain algorithm.

        Returns:
            List of sets, where each set contains node IDs in a community.
        """
        if self.G.number_of_nodes() == 0:
            return []

        # Convert to undirected for community detection
        undirected = self.G.to_undirected()

        if undirected.number_of_edges() == 0:
            # Return each node as its own community
            return [{n} for n in undirected.nodes()]

        try:
            return list(nx.community.louvain_communities(undirected))
        except Exception:
            # Fallback if louvain fails
            return [{n} for n in undirected.nodes()]

    def get_behavior_statistics(self) -> Dict[str, Any]:
        """Get statistics about behaviors in the graph."""
        behaviors = self.get_nodes_by_type("Behavior")

        stats = {
            "total_behaviors": len(behaviors),
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "categories": {},
            "edge_types": {},
        }

        # Count by category
        for _, attrs in behaviors:
            cat = attrs.get("category", "unknown")
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

        # Count edge types
        for _, _, data in self.G.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

        return stats

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Persist graph to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing data
            conn.execute("DELETE FROM nodes")
            conn.execute("DELETE FROM edges")

            # Save nodes
            for node_id, attrs in self.G.nodes(data=True):
                conn.execute(
                    "INSERT INTO nodes (id, node_type, properties) VALUES (?, ?, ?)",
                    (
                        str(node_id),
                        attrs.get("node_type", "Unknown"),
                        json.dumps(attrs, ensure_ascii=False),
                    ),
                )

            # Save edges
            for source, target, attrs in self.G.edges(data=True):
                conn.execute(
                    "INSERT OR REPLACE INTO edges (source, target, edge_type, properties) VALUES (?, ?, ?, ?)",
                    (
                        str(source),
                        str(target),
                        attrs.get("edge_type", "RELATED"),
                        json.dumps(attrs, ensure_ascii=False),
                    ),
                )

            conn.commit()

    def load(self) -> None:
        """Load graph from SQLite."""
        if not self.db_path.exists():
            return

        self.G.clear()

        with sqlite3.connect(self.db_path) as conn:
            # Load nodes
            cursor = conn.execute("SELECT id, properties FROM nodes")
            for row in cursor:
                node_id = row[0]
                props = json.loads(row[1])
                self.G.add_node(node_id, **props)

            # Load edges
            cursor = conn.execute("SELECT source, target, properties FROM edges")
            for row in cursor:
                source, target = row[0], row[1]
                props = json.loads(row[2]) if row[2] else {}
                self.G.add_edge(source, target, **props)

    def export_graphml(self, path: str) -> None:
        """Export graph to GraphML format for visualization."""
        # Convert all attributes to strings for GraphML compatibility
        G_export = nx.DiGraph()
        for node_id, attrs in self.G.nodes(data=True):
            G_export.add_node(node_id, **{k: str(v) for k, v in attrs.items()})
        for source, target, attrs in self.G.edges(data=True):
            G_export.add_edge(source, target, **{k: str(v) for k, v in attrs.items()})
        nx.write_graphml(G_export, path)

    # -------------------------------------------------------------------------
    # Bulk Loading
    # -------------------------------------------------------------------------

    def load_behaviors_from_vocab(self, vocab_path: str) -> int:
        """
        Load behaviors from vocabulary JSON file.

        Args:
            vocab_path: Path to behavior_concepts.json.

        Returns:
            Number of behaviors loaded.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        count = 0
        for category, behaviors in vocab.get("categories", {}).items():
            # Add category node
            cat_id = f"CAT_{category.upper()}"
            self.add_node(cat_id, "BehaviorCategory", name_en=category)

            for behavior in behaviors:
                self.add_behavior(
                    code=behavior["id"],
                    name_ar=behavior.get("ar", ""),
                    name_en=behavior.get("en", ""),
                    category=category,
                    quranic_roots=behavior.get("quranic_roots", []),
                )
                # Link to category
                self.add_relationship(behavior["id"], cat_id, "RELATED")
                count += 1

        return count

    def load_agents_from_vocab(self, vocab_path: str) -> int:
        """Load agents from vocabulary JSON file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        count = 0
        # Vocab uses 'items' key, not 'agents'
        for agent in vocab.get("items", []):
            self.add_agent(
                agent_id=agent["id"],
                name_ar=agent.get("ar", ""),
                agent_type=agent.get("en", "unknown"),
            )
            count += 1

        return count

    def load_organs_from_vocab(self, vocab_path: str) -> int:
        """Load organs from vocabulary JSON file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        count = 0
        # Vocab uses 'items' key, not 'organs'
        for organ in vocab.get("items", []):
            self.add_organ(
                organ_id=organ["id"],
                name_ar=organ.get("ar", ""),
                name_en=organ.get("id", "").replace("ORG_", "").replace("_", " ").title(),
            )
            count += 1

        return count
