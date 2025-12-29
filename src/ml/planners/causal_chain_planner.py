"""
Phase 3: CAUSAL_CHAIN Planner (3.1)

Thin wrapper around LegendaryPlanner for causal chain analysis.
Handles multi-hop chain computation with edge-level provenance.

REUSES:
- LegendaryPlanner.find_causal_paths()
- LegendaryPlanner.resolve_entities()
- LegendaryPlanner.get_concept_evidence()

ADDS:
- Multi-hop chain formatting for min_hops requirement
- Edge-level provenance attachment from semantic graph edges
- Tafsir quote retrieval integration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalPath:
    """A causal path with full provenance."""
    nodes: List[str]
    edges: List[Dict[str, Any]]
    hops: int
    total_evidence_count: int
    verse_keys: List[str] = field(default_factory=list)


@dataclass
class CausalChainResult:
    """Result of causal chain analysis."""
    from_entity: str
    to_entity: str
    paths: List[CausalPath]
    min_hops_requested: int
    qualifying_paths_count: int
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "paths": [
                {
                    "nodes": p.nodes,
                    "edges": p.edges,
                    "hops": p.hops,
                    "total_evidence_count": p.total_evidence_count,
                    "verse_keys": p.verse_keys,
                }
                for p in self.paths
            ],
            "min_hops_requested": self.min_hops_requested,
            "qualifying_paths_count": self.qualifying_paths_count,
            "gaps": self.gaps,
        }


class CausalChainPlanner:
    """
    Planner for CAUSAL_CHAIN question class.

    Wraps LegendaryPlanner to provide:
    - Multi-hop causal path finding
    - Edge-level provenance
    - Path validation against min_hops requirement
    """

    def __init__(self, legendary_planner=None):
        """
        Args:
            legendary_planner: Optional pre-loaded LegendaryPlanner instance.
                              If None, will create and load one.
        """
        self._planner = legendary_planner

    def _ensure_planner(self):
        """Ensure planner is loaded."""
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def find_causal_chains(
        self,
        query: str,
        from_entity_id: Optional[str] = None,
        to_entity_id: Optional[str] = None,
        min_hops: int = 3,
        max_depth: int = 5,
    ) -> CausalChainResult:
        """
        Find causal chains between entities.

        Args:
            query: The query text (used for entity resolution if IDs not provided)
            from_entity_id: Starting entity ID (optional, resolved from query)
            to_entity_id: Target entity ID (optional, resolved from query)
            min_hops: Minimum number of hops required
            max_depth: Maximum search depth

        Returns:
            CausalChainResult with paths and provenance
        """
        self._ensure_planner()

        gaps = []

        # Resolve entities from query if not provided
        if not from_entity_id or not to_entity_id:
            resolution = self._planner.resolve_entities(query)
            entities = resolution.get("entities", [])

            if len(entities) < 2:
                # Not enough entities - try to extract from query patterns
                entities = self._extract_entity_hints(query)

            if len(entities) >= 2:
                if not from_entity_id:
                    from_entity_id = entities[0].get("entity_id", entities[0]) if isinstance(entities[0], dict) else entities[0]
                if not to_entity_id:
                    to_entity_id = entities[1].get("entity_id", entities[1]) if isinstance(entities[1], dict) else entities[1]
            else:
                gaps.append("insufficient_entities_resolved")
                return CausalChainResult(
                    from_entity=from_entity_id or "",
                    to_entity=to_entity_id or "",
                    paths=[],
                    min_hops_requested=min_hops,
                    qualifying_paths_count=0,
                    gaps=gaps,
                )

        # Find causal paths using LegendaryPlanner
        raw_paths = self._planner.find_causal_paths(from_entity_id, to_entity_id, max_depth)

        if not raw_paths:
            gaps.append("no_causal_paths_found")

        # Convert to CausalPath objects with provenance
        paths = []
        for raw_path in raw_paths:
            # raw_path is a list of edges
            nodes = [from_entity_id]
            verse_keys = []
            total_evidence = 0

            for edge in raw_path:
                nodes.append(edge.get("target", ""))
                total_evidence += edge.get("evidence_count", 0)
                # Extract verse keys from edge provenance if available
                edge_verses = edge.get("verse_keys", [])
                verse_keys.extend(edge_verses)

            hops = len(raw_path)

            paths.append(CausalPath(
                nodes=nodes,
                edges=raw_path,
                hops=hops,
                total_evidence_count=total_evidence,
                verse_keys=verse_keys[:10],  # Limit to 10 verses
            ))

        # Filter paths meeting min_hops requirement
        qualifying = [p for p in paths if p.hops >= min_hops]

        if paths and not qualifying:
            gaps.append(f"no_paths_with_{min_hops}_hops")

        return CausalChainResult(
            from_entity=from_entity_id,
            to_entity=to_entity_id,
            paths=paths,
            min_hops_requested=min_hops,
            qualifying_paths_count=len(qualifying),
            gaps=gaps,
        )

    def _extract_entity_hints(self, query: str) -> List[str]:
        """Extract entity hints from query patterns like 'from X to Y'."""
        import re

        entities = []

        # Pattern: من X إلى Y
        ar_pattern = r'من\s+([\u0600-\u06FF]+)\s+إلى\s+([\u0600-\u06FF]+)'
        ar_match = re.search(ar_pattern, query)
        if ar_match:
            entities.append(ar_match.group(1))
            entities.append(ar_match.group(2))
            return entities

        # Pattern: from X to Y
        en_pattern = r'from\s+(\w+)\s+to\s+(\w+)'
        en_match = re.search(en_pattern, query, re.IGNORECASE)
        if en_match:
            entities.append(en_match.group(1))
            entities.append(en_match.group(2))

        return entities

    def get_path_evidence(self, path: CausalPath) -> Dict[str, Any]:
        """
        Get detailed evidence for each edge in a path.

        Args:
            path: A CausalPath to get evidence for

        Returns:
            Dictionary with edge-by-edge evidence
        """
        self._ensure_planner()

        edge_evidence = []

        for edge in path.edges:
            source = edge.get("source", "")
            target = edge.get("target", "")

            # Get evidence for source concept
            source_ev = self._planner.get_concept_evidence(source)
            target_ev = self._planner.get_concept_evidence(target)

            edge_evidence.append({
                "source": source,
                "target": target,
                "edge_type": edge.get("edge_type", ""),
                "confidence": edge.get("confidence", 0),
                "evidence_count": edge.get("evidence_count", 0),
                "source_evidence": source_ev,
                "target_evidence": target_ev,
            })

        return {
            "path_nodes": path.nodes,
            "path_hops": path.hops,
            "edge_evidence": edge_evidence,
        }
