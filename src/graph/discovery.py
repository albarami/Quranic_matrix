"""
Discovery Engine for QBM.

Phase 3 of Behavior Mastery Plan:
- Multi-hop path enumeration with evidence
- Bridge behavior detection (betweenness centrality)
- Community detection and labeling
- Motif discovery (repeated subgraph patterns)
- Link prediction (sandboxed as hypothesis-only)

All algorithms are deterministic and evidence-backed.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)

# Output paths
DISCOVERY_OUTPUT_DIR = Path("artifacts/discovery")
DISCOVERY_REPORT_PATH = DISCOVERY_OUTPUT_DIR / "discovery_report.json"
BRIDGES_PATH = DISCOVERY_OUTPUT_DIR / "bridges.json"
COMMUNITIES_PATH = DISCOVERY_OUTPUT_DIR / "communities.json"
MOTIFS_PATH = DISCOVERY_OUTPUT_DIR / "motifs.json"


@dataclass
class EvidenceBundle:
    """Evidence supporting a hypothesis."""
    verse_keys: List[str] = field(default_factory=list)
    tafsir_refs: List[Dict[str, Any]] = field(default_factory=list)
    edge_evidence_count: int = 0
    source_diversity: int = 0  # Number of distinct sources
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FalsificationResult:
    """Result of falsification check."""
    counter_evidence_found: bool = False
    counter_evidence: List[str] = field(default_factory=list)
    search_scope: str = "full_corpus"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    """
    A discovery hypothesis with evidence and falsification check.
    
    Link predictions are sandboxed:
    - is_confirmed=False until evidence confirms
    - promotion_blocked=True (cannot influence operational graph)
    """
    hypothesis_id: str
    hypothesis_type: str  # path, bridge, motif, community, link_prediction
    involved_behaviors: List[str]
    evidence_bundle: EvidenceBundle
    confidence_score: float
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    falsification_check: FalsificationResult = field(default_factory=FalsificationResult)
    is_confirmed: bool = True  # False for link_prediction
    promotion_blocked: bool = False  # True for link_prediction
    description: str = ""
    generated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type,
            "involved_behaviors": self.involved_behaviors,
            "evidence_bundle": self.evidence_bundle.to_dict(),
            "confidence_score": self.confidence_score,
            "confidence_factors": self.confidence_factors,
            "falsification_check": self.falsification_check.to_dict(),
            "is_confirmed": self.is_confirmed,
            "promotion_blocked": self.promotion_blocked,
            "description": self.description,
            "generated_at": self.generated_at,
        }


@dataclass
class CausalPath:
    """A multi-hop causal path between behaviors."""
    path_id: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    total_hops: int
    total_evidence: int
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BridgeBehavior:
    """A behavior that bridges clusters."""
    behavior_id: str
    label_ar: str
    label_en: str
    betweenness_centrality: float
    connected_clusters: List[str]
    evidence_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorCommunity:
    """A cluster of related behaviors."""
    community_id: str
    label: str
    behaviors: List[str]
    internal_edges: int
    external_edges: int
    cohesion_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphMotif:
    """A repeated subgraph pattern."""
    motif_id: str
    pattern_type: str  # triad, chain, cycle, star
    pattern_description: str
    instances: List[List[str]]  # List of behavior ID lists
    support_count: int
    evidence_per_instance: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiscoveryEngine:
    """
    Deterministic discovery engine for graph intelligence.
    
    All algorithms produce reproducible, evidence-backed results.
    """
    
    def __init__(self):
        self.graph = None
        self.node_lookup: Dict[str, Any] = {}
        self.adjacency: Dict[str, List[Tuple[str, str, int]]] = {}  # node -> [(target, edge_type, evidence)]
        self.reverse_adjacency: Dict[str, List[Tuple[str, str, int]]] = {}
        
    def load_graph(self):
        """Load normalized graph."""
        from src.graph.normalize import get_normalized_graph
        self.graph = get_normalized_graph()
        
        # Build lookups
        self.node_lookup = {n.id: n for n in self.graph.nodes}
        
        # Build adjacency lists
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        
        for edge in self.graph.edges:
            self.adjacency[edge.source].append((
                edge.target, edge.edge_type, edge.evidence_count
            ))
            self.reverse_adjacency[edge.target].append((
                edge.source, edge.edge_type, edge.evidence_count
            ))
        
        logger.info(f"Loaded graph: {len(self.node_lookup)} nodes, {len(self.graph.edges)} edges")
    
    def _get_behavior_ids(self) -> List[str]:
        """Get all behavior node IDs."""
        return sorted([
            n.id for n in self.graph.nodes
            if n.node_type == "BEHAVIOR"
        ])
    
    def find_multihop_paths(
        self,
        min_hops: int = 2,
        max_hops: int = 6,
        top_k: int = 100,
    ) -> List[CausalPath]:
        """
        Enumerate top-k multi-hop paths between behaviors with evidence.
        
        Uses BFS with evidence scoring to find highest-evidence paths.
        """
        behavior_ids = self._get_behavior_ids()
        all_paths: List[CausalPath] = []
        
        for start in behavior_ids:
            # BFS to find paths
            queue = [(start, [start], [], 0)]  # (current, path, edges, evidence)
            visited_at_depth: Dict[str, int] = {start: 0}
            
            while queue:
                current, path, edges, total_evidence = queue.pop(0)
                depth = len(path) - 1
                
                if depth >= max_hops:
                    continue
                
                for target, edge_type, evidence in self.adjacency.get(current, []):
                    # Only follow paths to behaviors
                    if target not in self.node_lookup:
                        continue
                    target_node = self.node_lookup[target]
                    if target_node.node_type != "BEHAVIOR":
                        continue
                    
                    # Avoid cycles in path
                    if target in path:
                        continue
                    
                    new_path = path + [target]
                    new_edges = edges + [{
                        "source": current,
                        "target": target,
                        "edge_type": edge_type,
                        "evidence_count": evidence,
                    }]
                    new_evidence = total_evidence + evidence
                    new_depth = len(new_path) - 1
                    
                    # Record path if meets min_hops
                    if new_depth >= min_hops:
                        path_id = f"PATH_{start}_{target}_{new_depth}h"
                        confidence = min(1.0, new_evidence / (new_depth * 10))
                        
                        all_paths.append(CausalPath(
                            path_id=path_id,
                            nodes=new_path,
                            edges=new_edges,
                            total_hops=new_depth,
                            total_evidence=new_evidence,
                            confidence=confidence,
                        ))
                    
                    # Continue BFS if not at max depth
                    if new_depth < max_hops:
                        # Only visit if we haven't seen this node at a shorter depth
                        if target not in visited_at_depth or visited_at_depth[target] > new_depth:
                            visited_at_depth[target] = new_depth
                            queue.append((target, new_path, new_edges, new_evidence))
        
        # Sort by evidence and return top_k
        all_paths.sort(key=lambda p: (-p.total_evidence, -p.total_hops))
        return all_paths[:top_k]
    
    def find_bridge_behaviors(self, top_k: int = 20) -> List[BridgeBehavior]:
        """
        Compute betweenness centrality to identify bridge behaviors.
        
        Bridge behaviors connect different clusters and are intervention points.
        """
        behavior_ids = self._get_behavior_ids()
        
        # Compute betweenness centrality (simplified)
        centrality: Dict[str, float] = defaultdict(float)
        
        for source in behavior_ids:
            # BFS from source
            distances: Dict[str, int] = {source: 0}
            predecessors: Dict[str, List[str]] = defaultdict(list)
            queue = [source]
            
            while queue:
                current = queue.pop(0)
                current_dist = distances[current]
                
                for target, _, _ in self.adjacency.get(current, []):
                    if target not in self.node_lookup:
                        continue
                    if self.node_lookup[target].node_type != "BEHAVIOR":
                        continue
                    
                    if target not in distances:
                        distances[target] = current_dist + 1
                        queue.append(target)
                        predecessors[target].append(current)
                    elif distances[target] == current_dist + 1:
                        predecessors[target].append(current)
            
            # Accumulate centrality
            dependency: Dict[str, float] = defaultdict(float)
            nodes_by_distance = sorted(distances.keys(), key=lambda x: -distances[x])
            
            for node in nodes_by_distance:
                if node == source:
                    continue
                for pred in predecessors[node]:
                    dependency[pred] += (1 + dependency[node]) / len(predecessors[node])
                if node != source:
                    centrality[node] += dependency[node]
        
        # Normalize
        n = len(behavior_ids)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            for node in centrality:
                centrality[node] *= norm
        
        # Get evidence counts
        evidence_counts: Dict[str, int] = defaultdict(int)
        for edge in self.graph.edges:
            evidence_counts[edge.source] += edge.evidence_count
            evidence_counts[edge.target] += edge.evidence_count
        
        # Build bridge list
        bridges = []
        for beh_id in sorted(centrality.keys(), key=lambda x: -centrality[x])[:top_k]:
            node = self.node_lookup.get(beh_id)
            if node:
                bridges.append(BridgeBehavior(
                    behavior_id=beh_id,
                    label_ar=node.label_ar,
                    label_en=node.label_en,
                    betweenness_centrality=round(centrality[beh_id], 6),
                    connected_clusters=[],  # Populated after community detection
                    evidence_count=evidence_counts[beh_id],
                ))
        
        return bridges
    
    def detect_communities(self) -> List[BehaviorCommunity]:
        """
        Cluster behaviors using label propagation.
        
        Labels clusters using most common edge types within cluster.
        """
        behavior_ids = self._get_behavior_ids()
        
        # Initialize each node with its own label
        labels: Dict[str, str] = {b: b for b in behavior_ids}
        
        # Label propagation iterations
        for _ in range(10):
            changed = False
            for node in behavior_ids:
                # Count neighbor labels
                neighbor_labels: Dict[str, int] = defaultdict(int)
                
                for target, _, evidence in self.adjacency.get(node, []):
                    if target in labels:
                        neighbor_labels[labels[target]] += evidence
                
                for source, _, evidence in self.reverse_adjacency.get(node, []):
                    if source in labels:
                        neighbor_labels[labels[source]] += evidence
                
                if neighbor_labels:
                    # Pick most common label
                    best_label = max(neighbor_labels.keys(), key=lambda x: neighbor_labels[x])
                    if labels[node] != best_label:
                        labels[node] = best_label
                        changed = True
            
            if not changed:
                break
        
        # Group by label
        communities_dict: Dict[str, List[str]] = defaultdict(list)
        for node, label in labels.items():
            communities_dict[label].append(node)
        
        # Build community objects
        communities = []
        for i, (label, members) in enumerate(sorted(communities_dict.items(), key=lambda x: -len(x[1]))):
            # Count internal/external edges
            member_set = set(members)
            internal = 0
            external = 0
            
            for member in members:
                for target, _, _ in self.adjacency.get(member, []):
                    if target in member_set:
                        internal += 1
                    elif target in self.node_lookup and self.node_lookup[target].node_type == "BEHAVIOR":
                        external += 1
            
            cohesion = internal / (internal + external + 1)
            
            # Generate label from common category
            categories = [self.node_lookup[m].attributes.get("category", "") for m in members if m in self.node_lookup]
            common_category = max(set(categories), key=categories.count) if categories else "mixed"
            
            communities.append(BehaviorCommunity(
                community_id=f"COMM_{i:03d}",
                label=f"{common_category}_cluster",
                behaviors=sorted(members),
                internal_edges=internal,
                external_edges=external,
                cohesion_score=round(cohesion, 3),
            ))
        
        return communities
    
    def find_motifs(self, min_support: int = 3) -> List[GraphMotif]:
        """
        Find repeated subgraph patterns (triads, chains, cycles).
        """
        behavior_ids = set(self._get_behavior_ids())
        motifs: List[GraphMotif] = []
        
        # Find triads: A -> B -> C where A also connects to C
        triad_instances: List[List[str]] = []
        triad_evidence: List[int] = []
        
        for a in behavior_ids:
            for b, _, ev_ab in self.adjacency.get(a, []):
                if b not in behavior_ids:
                    continue
                for c, _, ev_bc in self.adjacency.get(b, []):
                    if c not in behavior_ids or c == a:
                        continue
                    # Check if A -> C exists
                    for target, _, ev_ac in self.adjacency.get(a, []):
                        if target == c:
                            triad_instances.append([a, b, c])
                            triad_evidence.append(ev_ab + ev_bc + ev_ac)
                            break
        
        if len(triad_instances) >= min_support:
            motifs.append(GraphMotif(
                motif_id="MOTIF_TRIAD_001",
                pattern_type="triad",
                pattern_description="A -> B -> C with A -> C (transitive closure)",
                instances=triad_instances[:50],  # Limit instances
                support_count=len(triad_instances),
                evidence_per_instance=triad_evidence[:50],
            ))
        
        # Find chains: A -> B -> C -> D (linear 3-hop)
        chain_instances: List[List[str]] = []
        chain_evidence: List[int] = []
        
        for a in behavior_ids:
            for b, _, ev_ab in self.adjacency.get(a, []):
                if b not in behavior_ids:
                    continue
                for c, _, ev_bc in self.adjacency.get(b, []):
                    if c not in behavior_ids or c == a:
                        continue
                    for d, _, ev_cd in self.adjacency.get(c, []):
                        if d not in behavior_ids or d in [a, b]:
                            continue
                        chain_instances.append([a, b, c, d])
                        chain_evidence.append(ev_ab + ev_bc + ev_cd)
        
        if len(chain_instances) >= min_support:
            motifs.append(GraphMotif(
                motif_id="MOTIF_CHAIN_001",
                pattern_type="chain",
                pattern_description="A -> B -> C -> D (linear causal chain)",
                instances=chain_instances[:50],
                support_count=len(chain_instances),
                evidence_per_instance=chain_evidence[:50],
            ))
        
        # Find reinforcement cycles: A -> B -> A
        cycle_instances: List[List[str]] = []
        cycle_evidence: List[int] = []
        
        for a in behavior_ids:
            for b, _, ev_ab in self.adjacency.get(a, []):
                if b not in behavior_ids:
                    continue
                for target, _, ev_ba in self.adjacency.get(b, []):
                    if target == a:
                        if [b, a] not in cycle_instances:  # Avoid duplicates
                            cycle_instances.append([a, b])
                            cycle_evidence.append(ev_ab + ev_ba)
                        break
        
        if len(cycle_instances) >= min_support:
            motifs.append(GraphMotif(
                motif_id="MOTIF_CYCLE_001",
                pattern_type="cycle",
                pattern_description="A <-> B (reinforcement cycle)",
                instances=cycle_instances[:50],
                support_count=len(cycle_instances),
                evidence_per_instance=cycle_evidence[:50],
            ))
        
        return motifs
    
    def predict_links(self, min_confidence: float = 0.7) -> List[Hypothesis]:
        """
        Propose candidate edges - SANDBOXED AS HYPOTHESIS-ONLY.
        
        CRITICAL RULES:
        - Output stored under hypothesis_type="link_prediction"
        - NEVER treated as fact
        - NEVER influences operational graph
        - Can only be promoted to fact after evidence extraction confirms
        """
        behavior_ids = set(self._get_behavior_ids())
        predictions: List[Hypothesis] = []
        
        # Use common neighbors for prediction
        for a in behavior_ids:
            a_neighbors = set()
            for target, _, _ in self.adjacency.get(a, []):
                if target in behavior_ids:
                    a_neighbors.add(target)
            
            for b in behavior_ids:
                if b == a or b in a_neighbors:
                    continue
                
                b_neighbors = set()
                for target, _, _ in self.adjacency.get(b, []):
                    if target in behavior_ids:
                        b_neighbors.add(target)
                
                # Common neighbors
                common = a_neighbors & b_neighbors
                if len(common) >= 2:
                    # Calculate confidence based on common neighbors
                    confidence = len(common) / (len(a_neighbors | b_neighbors) + 1)
                    
                    if confidence >= min_confidence:
                        predictions.append(Hypothesis(
                            hypothesis_id=f"LINK_PRED_{a}_{b}",
                            hypothesis_type="link_prediction",
                            involved_behaviors=[a, b],
                            evidence_bundle=EvidenceBundle(
                                verse_keys=[],
                                edge_evidence_count=0,
                                source_diversity=0,
                            ),
                            confidence_score=round(confidence, 3),
                            confidence_factors={
                                "common_neighbors": len(common),
                                "jaccard_similarity": confidence,
                            },
                            falsification_check=FalsificationResult(
                                counter_evidence_found=False,
                                search_scope="not_searched",
                            ),
                            is_confirmed=False,  # NEVER confirmed until evidence
                            promotion_blocked=True,  # CANNOT influence operational graph
                            description=f"Predicted link between {a} and {b} based on {len(common)} common neighbors",
                            generated_at=datetime.utcnow().isoformat() + "Z",
                        ))
        
        # Sort by confidence
        predictions.sort(key=lambda p: -p.confidence_score)
        return predictions[:50]  # Limit to top 50
    
    def generate_discovery_report(self) -> Dict[str, Any]:
        """Generate complete discovery report."""
        logger.info("Generating discovery report...")
        
        # Find all discoveries
        paths = self.find_multihop_paths(min_hops=2, max_hops=5, top_k=100)
        bridges = self.find_bridge_behaviors(top_k=20)
        communities = self.detect_communities()
        motifs = self.find_motifs(min_support=3)
        link_predictions = self.predict_links(min_confidence=0.5)
        
        # Update bridge behaviors with community info
        behavior_to_community = {}
        for comm in communities:
            for beh in comm.behaviors:
                behavior_to_community[beh] = comm.community_id
        
        for bridge in bridges:
            connected = set()
            for target, _, _ in self.adjacency.get(bridge.behavior_id, []):
                if target in behavior_to_community:
                    connected.add(behavior_to_community[target])
            bridge.connected_clusters = sorted(connected)
        
        # Build report
        report = {
            "schema_version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "statistics": {
                "total_paths": len(paths),
                "total_bridges": len(bridges),
                "total_communities": len(communities),
                "total_motifs": len(motifs),
                "total_link_predictions": len(link_predictions),
                "behaviors_analyzed": len(self._get_behavior_ids()),
            },
            "top_paths": [p.to_dict() for p in paths[:20]],
            "bridges": [b.to_dict() for b in bridges],
            "communities": [c.to_dict() for c in communities],
            "motifs": [m.to_dict() for m in motifs],
            "link_predictions": [lp.to_dict() for lp in link_predictions],
        }
        
        return report
    
    def save_discovery_artifacts(self) -> str:
        """Save all discovery artifacts."""
        DISCOVERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_discovery_report()
        
        # Save main report
        with open(DISCOVERY_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved discovery report -> {DISCOVERY_REPORT_PATH}")
        
        # Save bridges
        with open(BRIDGES_PATH, "w", encoding="utf-8") as f:
            json.dump(report["bridges"], f, ensure_ascii=False, indent=2)
        
        # Save communities
        with open(COMMUNITIES_PATH, "w", encoding="utf-8") as f:
            json.dump(report["communities"], f, ensure_ascii=False, indent=2)
        
        # Save motifs
        with open(MOTIFS_PATH, "w", encoding="utf-8") as f:
            json.dump(report["motifs"], f, ensure_ascii=False, indent=2)
        
        # Compute report hash
        report_content = json.dumps(report, sort_keys=True, ensure_ascii=False)
        report_hash = hashlib.sha256(report_content.encode()).hexdigest()[:16]
        
        logger.info(f"Discovery artifacts saved. Hash: {report_hash}")
        return report_hash


def build_discovery_report() -> Dict[str, Any]:
    """Build and save discovery report."""
    engine = DiscoveryEngine()
    engine.load_graph()
    engine.save_discovery_artifacts()
    return engine.generate_discovery_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = build_discovery_report()
    print(f"Generated report with {report['statistics']['total_paths']} paths")
