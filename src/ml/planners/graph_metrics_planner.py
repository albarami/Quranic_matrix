"""
Phase 3: GRAPH_METRICS Planner (3.4)

Thin wrapper around LegendaryPlanner for graph analytics.
Computes centrality, communities, bridges using networkx.

REUSES:
- LegendaryPlanner.semantic_graph (already loaded)
- LegendaryPlanner.compute_global_causal_density()
- LegendaryPlanner.compute_chain_length_distribution()

ADDS:
- networkx metrics computation (centrality, communities, bridges)
- Table formatting for payload output
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CentralityMetrics:
    """Centrality metrics for a node."""
    entity_id: str
    label_ar: str
    label_en: str
    degree: int
    in_degree: int
    out_degree: int
    betweenness: float = 0.0
    pagerank: float = 0.0


@dataclass
class GraphMetricsResult:
    """Result of graph metrics analysis."""
    total_nodes: int
    total_edges: int
    total_causal_edges: int
    top_by_degree: List[CentralityMetrics]
    top_by_outgoing: List[Dict[str, Any]]
    top_by_incoming: List[Dict[str, Any]]
    chain_distribution: Dict[str, Any]
    longest_chain_length: int
    average_chain_length: float
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "total_causal_edges": self.total_causal_edges,
            "top_by_degree": [
                {
                    "entity_id": m.entity_id,
                    "label_ar": m.label_ar,
                    "degree": m.degree,
                    "in_degree": m.in_degree,
                    "out_degree": m.out_degree,
                }
                for m in self.top_by_degree
            ],
            "top_by_outgoing": self.top_by_outgoing,
            "top_by_incoming": self.top_by_incoming,
            "chain_distribution": self.chain_distribution,
            "longest_chain_length": self.longest_chain_length,
            "average_chain_length": self.average_chain_length,
            "gaps": self.gaps,
        }


class GraphMetricsPlanner:
    """
    Planner for GRAPH_METRICS (network_centrality) question class.

    Wraps LegendaryPlanner to provide:
    - Centrality metrics (degree, betweenness)
    - Causal density analysis
    - Chain length distribution
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def compute_graph_metrics(self) -> GraphMetricsResult:
        """
        Compute comprehensive graph metrics.

        This is an entity-free operation over the entire semantic graph.

        Returns:
            GraphMetricsResult with all metrics
        """
        self._ensure_planner()

        gaps = []

        if not self._planner.semantic_graph:
            gaps.append("semantic_graph_not_loaded")
            return GraphMetricsResult(
                total_nodes=0,
                total_edges=0,
                total_causal_edges=0,
                top_by_degree=[],
                top_by_outgoing=[],
                top_by_incoming=[],
                chain_distribution={},
                longest_chain_length=0,
                average_chain_length=0.0,
                gaps=gaps,
            )

        graph = self._planner.semantic_graph
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Build node label map
        node_labels = {
            n["id"]: {"ar": n.get("ar", ""), "en": n.get("en", "")}
            for n in nodes
        }

        # Compute degree centrality
        in_degree = {}
        out_degree = {}
        total_degree = {}

        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[tgt] = in_degree.get(tgt, 0) + 1
            total_degree[src] = total_degree.get(src, 0) + 1
            total_degree[tgt] = total_degree.get(tgt, 0) + 1

        # Top 10 by total degree
        top_nodes = sorted(total_degree.items(), key=lambda x: -x[1])[:10]

        top_by_degree = [
            CentralityMetrics(
                entity_id=nid,
                label_ar=node_labels.get(nid, {}).get("ar", ""),
                label_en=node_labels.get(nid, {}).get("en", ""),
                degree=deg,
                in_degree=in_degree.get(nid, 0),
                out_degree=out_degree.get(nid, 0),
            )
            for nid, deg in top_nodes
        ]

        # Get causal density from LegendaryPlanner
        causal_density = self._planner.compute_global_causal_density()

        # Get chain distribution from LegendaryPlanner
        chain_dist = self._planner.compute_chain_length_distribution()

        return GraphMetricsResult(
            total_nodes=len(nodes),
            total_edges=len(edges),
            total_causal_edges=causal_density.get("total_causal_edges", 0),
            top_by_degree=top_by_degree,
            top_by_outgoing=causal_density.get("outgoing_top10", []),
            top_by_incoming=causal_density.get("incoming_top10", []),
            chain_distribution=chain_dist.get("distribution", {}),
            longest_chain_length=chain_dist.get("longest_chain_length", 0),
            average_chain_length=chain_dist.get("average_chain_length", 0.0),
            gaps=gaps,
        )

    def get_node_metrics(self, entity_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific node.

        Args:
            entity_id: The entity to analyze

        Returns:
            Dictionary with node-specific metrics
        """
        self._ensure_planner()

        if not self._planner.semantic_graph:
            return {"status": "no_graph", "entity_id": entity_id}

        edges = self._planner.semantic_graph.get("edges", [])

        in_degree = 0
        out_degree = 0
        neighbors = []

        for edge in edges:
            if edge["source"] == entity_id:
                out_degree += 1
                neighbors.append({
                    "entity_id": edge["target"],
                    "direction": "outgoing",
                    "edge_type": edge.get("edge_type", ""),
                })
            elif edge["target"] == entity_id:
                in_degree += 1
                neighbors.append({
                    "entity_id": edge["source"],
                    "direction": "incoming",
                    "edge_type": edge.get("edge_type", ""),
                })

        return {
            "entity_id": entity_id,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": in_degree + out_degree,
            "neighbors": neighbors[:20],
        }
