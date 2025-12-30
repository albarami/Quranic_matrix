#!/usr/bin/env python3
"""
Phase 7: Build Graph from Validated Evidence

The graph is a PROJECTION of validated data sources:
- Behavior nodes from behavior_registry.json
- Verse nodes from concept_index_v3.jsonl
- MENTIONED_IN edges from concept_index_v3.jsonl evidence

If graph and SSOT disagree → SSOT wins → graph is regenerated.

Outputs:
- data/graph/graph_v3.json: Complete knowledge graph
- artifacts/graph_build_report.json: Build statistics

Exit codes:
- 0: Success
- 1: Failure
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.quran_store import get_quran_store
from src.data.behavior_registry import get_behavior_registry
from src.models.graph_schema import (
    KnowledgeGraph, GraphNode, GraphEdge,
    NodeType, EdgeType, EvidenceType,
    create_behavior_node, create_verse_node, create_mentioned_in_edge
)


def save_json(path: Path, data: Dict) -> None:
    """Save JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_concept_index() -> List[Dict]:
    """Load concept_index_v3.jsonl."""
    entries = []
    index_path = Path("data/evidence/concept_index_v3.jsonl")

    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    return entries


def build_behavior_nodes(registry) -> List[GraphNode]:
    """Create nodes for all behaviors from registry."""
    nodes = []

    for behavior in registry.get_all():
        node = create_behavior_node(
            behavior.behavior_id,
            behavior.label_ar,
            behavior.label_en,
            behavior.category
        )
        nodes.append(node)

    return nodes


def build_verse_nodes(concept_index: List[Dict], store) -> List[GraphNode]:
    """Create nodes for all unique verses in concept index."""
    verse_keys: Set[str] = set()
    nodes = []

    # Collect unique verse keys
    for entry in concept_index:
        for verse in entry.get("verses", []):
            verse_keys.add(verse.get("verse_key", ""))

    # Create nodes
    for vk in sorted(verse_keys, key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1]))):
        verse = store.get_verse(vk)
        text = verse.text_uthmani if verse else ""
        node = create_verse_node(vk, text)
        nodes.append(node)

    return nodes


def build_mentioned_in_edges(concept_index: List[Dict]) -> List[GraphEdge]:
    """
    Create MENTIONED_IN edges from concept index.

    Each behavior -> verse connection with evidence type from index.
    """
    edges = []

    for entry in concept_index:
        behavior_id = entry["concept_id"]

        for verse in entry.get("verses", []):
            vk = verse.get("verse_key", "")
            if not vk:
                continue

            # Determine evidence type from verse evidence
            evidence = verse.get("evidence", [])
            has_lexical = any(e.get("type") == "lexical" for e in evidence)
            has_annotation = any(e.get("type") == "annotation" for e in evidence)

            if has_lexical and has_annotation:
                ev_type = EvidenceType.LEXICAL  # Prefer lexical
            elif has_lexical:
                ev_type = EvidenceType.LEXICAL
            elif has_annotation:
                ev_type = EvidenceType.ANNOTATION
            else:
                ev_type = EvidenceType.INFERRED

            edge = create_mentioned_in_edge(
                behavior_id,
                vk,
                evidence_type=ev_type,
                weight=1.0
            )
            edges.append(edge)

    return edges


def main() -> int:
    """Build graph from validated evidence."""
    print("=" * 60)
    print("Phase 7: Build Graph from Evidence")
    print("=" * 60)

    try:
        # Load dependencies
        print("\nLoading Quran store...")
        store = get_quran_store()
        print(f"  Loaded {store.total_verses} verses")

        print("\nLoading behavior registry...")
        registry = get_behavior_registry()
        print(f"  Loaded {registry.count()} behaviors")

        print("\nLoading concept_index_v3...")
        concept_index = load_concept_index()
        print(f"  Loaded {len(concept_index)} behavior entries")

        # Build graph
        print("\nBuilding graph...")

        graph = KnowledgeGraph(
            version="3.0",
            metadata={
                "description": "QBM Knowledge Graph - SSOT Projection",
                "sources": [
                    "behavior_registry.json",
                    "concept_index_v3.jsonl"
                ]
            }
        )

        # Add behavior nodes
        print("  Creating behavior nodes...")
        behavior_nodes = build_behavior_nodes(registry)
        for node in behavior_nodes:
            graph.add_node(node)
        print(f"    Added {len(behavior_nodes)} behavior nodes")

        # Add verse nodes
        print("  Creating verse nodes...")
        verse_nodes = build_verse_nodes(concept_index, store)
        for node in verse_nodes:
            graph.add_node(node)
        print(f"    Added {len(verse_nodes)} verse nodes")

        # Add MENTIONED_IN edges
        print("  Creating MENTIONED_IN edges...")
        edges = build_mentioned_in_edges(concept_index)
        for edge in edges:
            graph.add_edge(edge)
        print(f"    Added {len(edges)} edges")

        # Save graph
        graph_path = Path("data/graph/graph_v3.json")
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving graph to {graph_path}...")
        graph_data = graph.to_dict()
        save_json(graph_path, graph_data)

        # Generate report
        report = {
            "phase": 7,
            "description": "Graph Build Report",
            "statistics": graph_data["statistics"],
            "sources": {
                "behavior_registry": registry.count(),
                "concept_index_entries": len(concept_index),
                "unique_verses": len(verse_nodes)
            },
            "sample_nodes": [n.to_dict() for n in graph.nodes[:5]],
            "sample_edges": [e.to_dict() for e in graph.edges[:5]]
        }

        report_path = Path("artifacts/graph_build_report.json")
        save_json(report_path, report)
        print(f"Saved report to {report_path}")

        # Check patience behavior
        patience_edges = graph.get_edges_from("BEH_EMO_PATIENCE")

        # Print summary
        print("\n" + "=" * 60)
        print("GRAPH BUILD SUMMARY")
        print("=" * 60)
        print(f"Total nodes: {graph_data['statistics']['total_nodes']}")
        print(f"  Behavior nodes: {graph_data['statistics']['nodes_by_type'].get('behavior', 0)}")
        print(f"  Verse nodes: {graph_data['statistics']['nodes_by_type'].get('verse', 0)}")
        print(f"Total edges: {graph_data['statistics']['total_edges']}")
        print(f"  MENTIONED_IN edges: {graph_data['statistics']['edges_by_type'].get('MENTIONED_IN', 0)}")

        print(f"\nPatience (sabr) edges: {len(patience_edges)}")
        if patience_edges:
            sample_targets = [e.target for e in patience_edges[:5]]
            print(f"  Sample targets: {sample_targets}")

        print("\n" + "=" * 60)
        print("Phase 7: GRAPH BUILT SUCCESSFULLY")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
