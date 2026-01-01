"""
Graph Normalization for QBM.

Normalizes any graph file into GraphV3 schema:
- Exactly 87 BEHAVIOR nodes (from canonical registry)
- Uppercase node_type and edge_type
- Full provenance on edges (EvidenceRef with chunk_id + offsets)
- Deterministic ordering and stable IDs
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Paths
CANONICAL_ENTITIES_PATH = Path("vocab/canonical_entities.json")
GRAPH_V2_PATH = Path("data/graph/semantic_graph_v2.json")
GRAPH_V3_PATH = Path("data/graph/semantic_graph_v3.json")


@dataclass
class EvidenceRef:
    """Full provenance for edge evidence - auditable and traceable."""
    verse_key: str
    source: str  # quran, ibn_kathir, tabari, etc.
    chunk_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    relevance: str = "direct"  # direct, indirect, supporting
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class NodeV3:
    """Normalized node in GraphV3 schema."""
    id: str
    node_type: str  # BEHAVIOR, AGENT, ORGAN, STATE (uppercase)
    label_ar: str
    label_en: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "label_ar": self.label_ar,
            "label_en": self.label_en,
            "attributes": self.attributes,
        }


@dataclass
class EdgeV3:
    """Normalized edge in GraphV3 schema with full provenance."""
    source: str
    target: str
    edge_type: str  # CAUSES, LEADS_TO, PREVENTS, STRENGTHENS, OPPOSITE
    evidence_count: int
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "evidence_count": self.evidence_count,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence": self.confidence,
        }


@dataclass
class GraphMetadata:
    """Metadata for normalized graph."""
    schema_version: str = "3.0.0"
    normalized_at: str = ""
    source_file: str = ""
    source_hash: str = ""
    behavior_count: int = 0
    total_node_count: int = 0
    edge_count: int = 0
    canonical_registry_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphV3:
    """Normalized graph in GraphV3 schema."""
    nodes: List[NodeV3]
    edges: List[EdgeV3]
    metadata: GraphMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata.to_dict(),
        }
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of graph content."""
        content = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _normalize_node_type(raw_type: str) -> str:
    """Normalize node type to uppercase."""
    type_map = {
        "behavior": "BEHAVIOR",
        "Behavior": "BEHAVIOR",
        "BEHAVIOR": "BEHAVIOR",
        "agent": "AGENT",
        "Agent": "AGENT",
        "AGENT": "AGENT",
        "organ": "ORGAN",
        "Organ": "ORGAN",
        "ORGAN": "ORGAN",
        "state": "STATE",
        "State": "STATE",
        "STATE": "STATE",
        "consequence": "CONSEQUENCE",
        "Consequence": "CONSEQUENCE",
        "CONSEQUENCE": "CONSEQUENCE",
        "context": "CONTEXT",
        "Context": "CONTEXT",
        "CONTEXT": "CONTEXT",
    }
    return type_map.get(raw_type, raw_type.upper())


def _normalize_edge_type(raw_type: str) -> str:
    """Normalize edge type to uppercase."""
    type_map = {
        "causes": "CAUSES",
        "Causes": "CAUSES",
        "CAUSES": "CAUSES",
        "leads_to": "LEADS_TO",
        "leadsTo": "LEADS_TO",
        "LEADS_TO": "LEADS_TO",
        "prevents": "PREVENTS",
        "Prevents": "PREVENTS",
        "PREVENTS": "PREVENTS",
        "strengthens": "STRENGTHENS",
        "Strengthens": "STRENGTHENS",
        "STRENGTHENS": "STRENGTHENS",
        "weakens": "WEAKENS",
        "Weakens": "WEAKENS",
        "WEAKENS": "WEAKENS",
        "opposite": "OPPOSITE",
        "Opposite": "OPPOSITE",
        "OPPOSITE": "OPPOSITE",
        "mentioned_in": "MENTIONED_IN",
        "MENTIONED_IN": "MENTIONED_IN",
    }
    return type_map.get(raw_type, raw_type.upper())


def _load_canonical_behaviors() -> Dict[str, Dict[str, Any]]:
    """Load canonical behaviors from registry."""
    if not CANONICAL_ENTITIES_PATH.exists():
        logger.warning(f"Canonical entities file not found: {CANONICAL_ENTITIES_PATH}")
        return {}
    
    with open(CANONICAL_ENTITIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    behaviors = {}
    for b in data.get("behaviors", []):
        behaviors[b["id"]] = b
    
    return behaviors


def _convert_evidence_to_refs(edge_data: Dict[str, Any]) -> List[EvidenceRef]:
    """Convert legacy evidence format to EvidenceRef list."""
    refs = []
    
    # Check for existing evidence_refs
    if "evidence_refs" in edge_data:
        for ref in edge_data["evidence_refs"]:
            if isinstance(ref, dict):
                refs.append(EvidenceRef(
                    verse_key=ref.get("verse_key", ""),
                    source=ref.get("source", "quran"),
                    chunk_id=ref.get("chunk_id"),
                    char_start=ref.get("char_start"),
                    char_end=ref.get("char_end"),
                    relevance=ref.get("relevance", "direct"),
                ))
        return refs
    
    # Convert from legacy evidence format
    evidence = edge_data.get("evidence", [])
    if isinstance(evidence, list):
        for ev in evidence:
            if isinstance(ev, dict):
                verse_key = ev.get("verse_key", ev.get("verse", ""))
                source = ev.get("source", "quran")
                refs.append(EvidenceRef(
                    verse_key=verse_key,
                    source=source,
                    chunk_id=ev.get("chunk_id"),
                    char_start=ev.get("char_start"),
                    char_end=ev.get("char_end"),
                    relevance=ev.get("relevance", "direct"),
                ))
            elif isinstance(ev, str):
                # Simple verse key string
                refs.append(EvidenceRef(verse_key=ev, source="quran"))
    
    # Also check sources field for tafsir provenance
    sources = edge_data.get("sources", [])
    if isinstance(sources, list):
        for src in sources:
            if isinstance(src, dict) and src.get("source") != "quran":
                refs.append(EvidenceRef(
                    verse_key=src.get("verse_key", ""),
                    source=src.get("source", ""),
                    chunk_id=src.get("chunk_id"),
                    char_start=src.get("char_start"),
                    char_end=src.get("char_end"),
                    relevance="supporting",
                ))
    
    return refs


def normalize_graph(
    raw_graph: Dict[str, Any],
    canonical_behaviors: Optional[Dict[str, Dict[str, Any]]] = None,
    source_file: str = "",
) -> GraphV3:
    """
    Normalize a raw graph to GraphV3 schema.
    
    Args:
        raw_graph: Raw graph dictionary with nodes and edges
        canonical_behaviors: Optional dict of canonical behaviors (loaded if not provided)
        source_file: Source file path for metadata
        
    Returns:
        Normalized GraphV3 object
    """
    if canonical_behaviors is None:
        canonical_behaviors = _load_canonical_behaviors()
    
    # Track existing node IDs
    existing_node_ids: Set[str] = set()
    normalized_nodes: List[NodeV3] = []
    
    # Normalize existing nodes
    for raw_node in raw_graph.get("nodes", []):
        node_id = raw_node.get("id", "")
        if not node_id:
            continue
        
        existing_node_ids.add(node_id)
        
        # Determine node type
        raw_type = raw_node.get("node_type", raw_node.get("type", ""))
        node_type = _normalize_node_type(raw_type)
        
        # Get labels
        label_ar = raw_node.get("label_ar", raw_node.get("ar", ""))
        label_en = raw_node.get("label_en", raw_node.get("en", ""))
        
        # Collect attributes
        attributes = {}
        for key in ["category", "roots", "stems", "synonyms", "antonyms"]:
            if key in raw_node:
                attributes[key] = raw_node[key]
        
        normalized_nodes.append(NodeV3(
            id=node_id,
            node_type=node_type,
            label_ar=label_ar,
            label_en=label_en,
            attributes=attributes,
        ))
    
    # Add missing canonical behaviors
    for beh_id, beh_data in canonical_behaviors.items():
        if beh_id not in existing_node_ids:
            logger.info(f"Adding missing canonical behavior: {beh_id}")
            normalized_nodes.append(NodeV3(
                id=beh_id,
                node_type="BEHAVIOR",
                label_ar=beh_data.get("ar", ""),
                label_en=beh_data.get("en", ""),
                attributes={
                    "category": beh_data.get("category", ""),
                    "roots": beh_data.get("roots", []),
                    "synonyms": beh_data.get("synonyms", []),
                    "added_from_canonical": True,
                },
            ))
            existing_node_ids.add(beh_id)
    
    # Sort nodes by ID for deterministic ordering
    normalized_nodes.sort(key=lambda n: n.id)
    
    # Normalize edges
    normalized_edges: List[EdgeV3] = []
    for raw_edge in raw_graph.get("edges", []):
        source = raw_edge.get("source", "")
        target = raw_edge.get("target", "")
        
        # Skip edges with missing endpoints
        if not source or not target:
            continue
        if source not in existing_node_ids or target not in existing_node_ids:
            logger.warning(f"Skipping edge with missing endpoint: {source} -> {target}")
            continue
        
        edge_type = _normalize_edge_type(raw_edge.get("edge_type", ""))
        evidence_count = raw_edge.get("evidence_count", 0)
        confidence = raw_edge.get("confidence", 0.0)
        
        # Convert evidence to EvidenceRef format
        evidence_refs = _convert_evidence_to_refs(raw_edge)
        
        normalized_edges.append(EdgeV3(
            source=source,
            target=target,
            edge_type=edge_type,
            evidence_count=evidence_count,
            evidence_refs=evidence_refs,
            confidence=confidence,
        ))
    
    # Sort edges for deterministic ordering
    normalized_edges.sort(key=lambda e: (e.source, e.target, e.edge_type))
    
    # Count behaviors
    behavior_count = sum(1 for n in normalized_nodes if n.node_type == "BEHAVIOR")
    
    # Compute canonical registry hash
    canonical_hash = ""
    if canonical_behaviors:
        canonical_content = json.dumps(
            sorted(canonical_behaviors.keys()),
            sort_keys=True
        )
        canonical_hash = hashlib.sha256(canonical_content.encode()).hexdigest()[:16]
    
    # Compute source hash
    source_hash = ""
    if raw_graph:
        source_content = json.dumps(raw_graph, sort_keys=True, ensure_ascii=False)
        source_hash = hashlib.sha256(source_content.encode()).hexdigest()[:16]
    
    metadata = GraphMetadata(
        schema_version="3.0.0",
        normalized_at=datetime.utcnow().isoformat() + "Z",
        source_file=source_file,
        source_hash=source_hash,
        behavior_count=behavior_count,
        total_node_count=len(normalized_nodes),
        edge_count=len(normalized_edges),
        canonical_registry_hash=canonical_hash,
    )
    
    return GraphV3(
        nodes=normalized_nodes,
        edges=normalized_edges,
        metadata=metadata,
    )


def load_and_normalize(
    graph_path: Optional[Path] = None,
    canonical_path: Optional[Path] = None,
) -> GraphV3:
    """
    Load a graph file and normalize it to GraphV3.
    
    Args:
        graph_path: Path to graph JSON file (defaults to semantic_graph_v2.json)
        canonical_path: Path to canonical entities file
        
    Returns:
        Normalized GraphV3 object
    """
    if graph_path is None:
        graph_path = GRAPH_V2_PATH
    
    if canonical_path is None:
        canonical_path = CANONICAL_ENTITIES_PATH
    
    # Load canonical behaviors
    canonical_behaviors = {}
    if canonical_path.exists():
        with open(canonical_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for b in data.get("behaviors", []):
            canonical_behaviors[b["id"]] = b
    
    # Load raw graph
    with open(graph_path, "r", encoding="utf-8") as f:
        raw_graph = json.load(f)
    
    return normalize_graph(
        raw_graph=raw_graph,
        canonical_behaviors=canonical_behaviors,
        source_file=str(graph_path),
    )


def save_normalized_graph(graph: GraphV3, output_path: Path) -> str:
    """
    Save normalized graph to JSON file.
    
    Args:
        graph: Normalized GraphV3 object
        output_path: Output file path
        
    Returns:
        Hash of saved content
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = graph.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    
    return graph.compute_hash()


# Singleton for normalized graph
_normalized_graph: Optional[GraphV3] = None


def get_normalized_graph(force_reload: bool = False) -> GraphV3:
    """
    Get the singleton normalized graph instance.
    
    All planners and engines should use this function to ensure
    they're using the same normalized graph.
    """
    global _normalized_graph
    
    if _normalized_graph is None or force_reload:
        _normalized_graph = load_and_normalize()
        logger.info(
            f"Loaded normalized graph: {_normalized_graph.metadata.behavior_count} behaviors, "
            f"{_normalized_graph.metadata.total_node_count} total nodes, "
            f"{_normalized_graph.metadata.edge_count} edges"
        )
    
    return _normalized_graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test normalization
    graph = load_and_normalize()
    print(f"Behavior count: {graph.metadata.behavior_count}")
    print(f"Total nodes: {graph.metadata.total_node_count}")
    print(f"Edges: {graph.metadata.edge_count}")
    print(f"Hash: {graph.compute_hash()}")
    
    # Validate
    from .contract import validate_graph_contract
    nodes_dict = [n.to_dict() for n in graph.nodes]
    edges_dict = [e.to_dict() for e in graph.edges]
    is_valid, violations = validate_graph_contract(nodes_dict, edges_dict)
    print(f"Valid: {is_valid}")
    if violations:
        for v in violations[:5]:
            print(f"  - {v.message}")
