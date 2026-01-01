"""
Graph Normalization Layer for QBM.

Phase 1 of Behavior Mastery Plan:
- Unified GraphV3 schema contract
- Normalization of all graph files
- Full provenance on edges (EvidenceRef with chunk_id + offsets)
- Exactly 87 behavior nodes enforced
"""

from .normalize import (
    normalize_graph,
    load_and_normalize,
    GraphV3,
    NodeV3,
    EdgeV3,
    EvidenceRef,
    GraphMetadata,
)
from .contract import (
    validate_graph_contract,
    VALID_NODE_TYPES,
    VALID_EDGE_TYPES,
)

__all__ = [
    "normalize_graph",
    "load_and_normalize",
    "GraphV3",
    "NodeV3",
    "EdgeV3",
    "EvidenceRef",
    "GraphMetadata",
    "validate_graph_contract",
    "VALID_NODE_TYPES",
    "VALID_EDGE_TYPES",
]
