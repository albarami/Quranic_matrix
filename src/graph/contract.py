"""
Graph Contract Validation for QBM.

Defines valid node types, edge types, and validation rules.
All graphs must pass contract validation before use.
"""

from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass

# Valid node types (uppercase only)
VALID_NODE_TYPES: Set[str] = {
    "BEHAVIOR",
    "AGENT",
    "ORGAN",
    "STATE",
    "AXIS_VALUE",
    "CONSEQUENCE",
    "CONTEXT",
    "HEART_STATE",  # Heart states (HRT_*)
    "EMOTION",
    "VIRTUE",
    "VICE",
}

# Valid edge types (uppercase only)
VALID_EDGE_TYPES: Set[str] = {
    "CAUSES",
    "LEADS_TO",
    "PREVENTS",
    "STRENGTHENS",
    "WEAKENS",
    "OPPOSITE",
    "OPPOSITE_OF",
    "MENTIONED_IN",
    "ASSOCIATED_WITH",
    "REQUIRES",
    "ENABLES",
    "COMPLEMENTS",
    "MANIFESTS_AS",
    "RESULTS_IN",
    "CORRELATES_WITH",
    "CONDITIONAL_ON",
    "INFLUENCES",
    "SUPPORTS",
}

# Required fields for nodes
REQUIRED_NODE_FIELDS: Set[str] = {"id", "node_type", "label_ar", "label_en"}

# Required fields for edges
REQUIRED_EDGE_FIELDS: Set[str] = {"source", "target", "edge_type", "evidence_count"}


@dataclass
class ContractViolation:
    """A single contract violation."""
    violation_type: str
    message: str
    context: Dict[str, Any]


def validate_graph_contract(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    require_exactly_87_behaviors: bool = True,
) -> Tuple[bool, List[ContractViolation]]:
    """
    Validate graph against GraphV3 contract.
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        require_exactly_87_behaviors: If True, require exactly 87 BEHAVIOR nodes
        
    Returns:
        Tuple of (is_valid, list of violations)
    """
    violations: List[ContractViolation] = []
    node_ids: Set[str] = set()
    behavior_count = 0
    
    # Validate nodes
    for i, node in enumerate(nodes):
        node_id = node.get("id", f"<missing_id_{i}>")
        node_ids.add(node_id)
        
        # Check required fields
        for field in REQUIRED_NODE_FIELDS:
            if field not in node:
                violations.append(ContractViolation(
                    violation_type="missing_node_field",
                    message=f"Node {node_id} missing required field: {field}",
                    context={"node_id": node_id, "field": field}
                ))
        
        # Check node_type is valid and uppercase
        node_type = node.get("node_type", "")
        if node_type not in VALID_NODE_TYPES:
            violations.append(ContractViolation(
                violation_type="invalid_node_type",
                message=f"Node {node_id} has invalid node_type: {node_type}",
                context={"node_id": node_id, "node_type": node_type, "valid_types": list(VALID_NODE_TYPES)}
            ))
        
        if node_type == "BEHAVIOR":
            behavior_count += 1
    
    # Check behavior count
    if require_exactly_87_behaviors and behavior_count != 87:
        violations.append(ContractViolation(
            violation_type="behavior_count_mismatch",
            message=f"Expected exactly 87 BEHAVIOR nodes, found {behavior_count}",
            context={"expected": 87, "actual": behavior_count}
        ))
    
    # Validate edges
    for i, edge in enumerate(edges):
        edge_id = f"{edge.get('source', '?')}->{edge.get('target', '?')}"
        
        # Check required fields
        for field in REQUIRED_EDGE_FIELDS:
            if field not in edge:
                violations.append(ContractViolation(
                    violation_type="missing_edge_field",
                    message=f"Edge {edge_id} missing required field: {field}",
                    context={"edge_id": edge_id, "field": field}
                ))
        
        # Check edge_type is valid and uppercase
        edge_type = edge.get("edge_type", "")
        if edge_type not in VALID_EDGE_TYPES:
            violations.append(ContractViolation(
                violation_type="invalid_edge_type",
                message=f"Edge {edge_id} has invalid edge_type: {edge_type}",
                context={"edge_id": edge_id, "edge_type": edge_type, "valid_types": list(VALID_EDGE_TYPES)}
            ))
        
        # Check source and target exist
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source and source not in node_ids:
            violations.append(ContractViolation(
                violation_type="dangling_edge_source",
                message=f"Edge source {source} not found in nodes",
                context={"edge_id": edge_id, "source": source}
            ))
        if target and target not in node_ids:
            violations.append(ContractViolation(
                violation_type="dangling_edge_target",
                message=f"Edge target {target} not found in nodes",
                context={"edge_id": edge_id, "target": target}
            ))
        
        # Check evidence_refs have provenance (if present)
        evidence_refs = edge.get("evidence_refs", [])
        for j, ref in enumerate(evidence_refs):
            if not isinstance(ref, dict):
                continue
            if not ref.get("verse_key"):
                violations.append(ContractViolation(
                    violation_type="evidence_ref_missing_verse_key",
                    message=f"Edge {edge_id} evidence_ref[{j}] missing verse_key",
                    context={"edge_id": edge_id, "ref_index": j}
                ))
            if not ref.get("source"):
                violations.append(ContractViolation(
                    violation_type="evidence_ref_missing_source",
                    message=f"Edge {edge_id} evidence_ref[{j}] missing source",
                    context={"edge_id": edge_id, "ref_index": j}
                ))
    
    is_valid = len(violations) == 0
    return is_valid, violations
