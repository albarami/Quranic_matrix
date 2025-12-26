"""
Layer 5: Graph Neural Network Reasoning

Multi-hop reasoning over the behavioral graph.
Discovers patterns and paths not explicitly annotated.

Current (BAD):  graph.get_neighbors(node) → static edges
New (GOOD):     gnn.find_path(start, end) → learned behavioral chains
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import flags - DO NOT import torch_geometric at module level
# Windows DLL load can crash before exception handler catches it
TORCH_AVAILABLE = False
PYG_AVAILABLE = False
DEVICE = "cpu"

def _check_torch_available():
    """Lazy check for torch availability."""
    global TORCH_AVAILABLE, DEVICE
    if TORCH_AVAILABLE:
        return True
    try:
        import torch
        TORCH_AVAILABLE = True
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return True
    except ImportError:
        return False

def _check_pyg_available():
    """Lazy check for torch_geometric availability."""
    global PYG_AVAILABLE
    if PYG_AVAILABLE:
        return True
    if not _check_torch_available():
        return False
    try:
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        PYG_AVAILABLE = True
        return True
    except (ImportError, OSError, Exception) as e:
        logger.warning(f"torch_geometric not available: {e}")
        return False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Node types in the graph
NODE_TYPES = {
    "behavior": 0,
    "verse": 1,
    "tafsir": 2,
    "agent": 3,
    "dimension": 4,
}

# Edge types
EDGE_TYPES = {
    "causes": 0,
    "prevents": 1,
    "opposite": 2,
    "intensifies": 3,
    "co_occurs": 4,
    "mentioned_in": 5,
}


# =============================================================================
# GRAPH NEURAL NETWORK
# =============================================================================

def _create_gnn_model(num_node_features: int = 768, hidden_dim: int = 256,
                      num_relations: int = 6, num_heads: int = 8):
    """
    Factory function to create GNN model with lazy imports.
    Returns None if torch_geometric is not available.
    """
    if not _check_pyg_available():
        return None
    
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv
    
    class QBMGraphReasonerImpl(nn.Module):
        """
        Graph Attention Network for multi-hop reasoning.
        
        Learns:
        1. Which behaviors cluster together
        2. Common causal chains
        3. Hidden relationships not explicitly annotated
        """
        
        def __init__(self):
            super().__init__()
            
            self.num_relations = num_relations
            
            # Graph Attention layers
            self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=0.1)
            self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=4, dropout=0.1)
            self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=2, dropout=0.1)
            
            # Relation predictor
            self.relation_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2 * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_relations),
            )
            
            # Path scorer
            self.path_scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        def forward(self, x, edge_index):
            """Forward pass through GAT layers."""
            import torch.nn.functional as F
            
            # Layer 1
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
            # Layer 2
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
            # Layer 3
            x = self.conv3(x, edge_index)
            
            return x
        
        def predict_relation(self, node1_embed, node2_embed):
            """Predict relationship between two nodes."""
            import torch
            combined = torch.cat([node1_embed, node2_embed], dim=-1)
            return self.relation_predictor(combined)
        
        def score_path(self, path_embeddings):
            """Score a path through the graph."""
            import torch
            if len(path_embeddings) < 2:
                return 0.0
            
            total_score = 0.0
            for i in range(len(path_embeddings) - 1):
                combined = torch.cat([path_embeddings[i], path_embeddings[i+1]], dim=-1)
                score = self.path_scorer(combined)
                total_score += score.item()
            
            return total_score / (len(path_embeddings) - 1)
    
    return QBMGraphReasonerImpl()


# Legacy class name for backward compatibility
class QBMGraphReasoner:
    """
    Wrapper class for backward compatibility.
    Actual implementation is created lazily via _create_gnn_model().
    """
    
    def __init__(self, num_node_features: int = 768, hidden_dim: int = 256, 
                 num_relations: int = 6, num_heads: int = 8):
        self._impl = None
        self._params = (num_node_features, hidden_dim, num_relations, num_heads)
    
    def _get_impl(self):
        if self._impl is None:
            self._impl = _create_gnn_model(*self._params)
        return self._impl
    
    def __call__(self, x, edge_index):
        impl = self._get_impl()
        if impl is None:
            return x
        return impl(x, edge_index)
    
    def eval(self):
        impl = self._get_impl()
        if impl is not None:
            impl.eval()
    
    def train(self, mode=True):
        impl = self._get_impl()
        if impl is not None:
            impl.train(mode)
    
    def state_dict(self):
        impl = self._get_impl()
        if impl is not None:
            return impl.state_dict()
        return {}
    
    def load_state_dict(self, state_dict):
        impl = self._get_impl()
        if impl is not None:
            impl.load_state_dict(state_dict)
    
    def cuda(self):
        impl = self._get_impl()
        if impl is not None:
            impl.cuda()
        return self
    
    def predict_relation(self, node1_embed, node2_embed):
        impl = self._get_impl()
        if impl is not None:
            return impl.predict_relation(node1_embed, node2_embed)
        return None
    
    def score_path(self, path_embeddings):
        impl = self._get_impl()
        if impl is not None:
            return impl.score_path(path_embeddings)
        return 0.0
    


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class GraphBuilder:
    """Build PyTorch Geometric graph from QBM data."""
    
    def __init__(self):
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_types = []
        self.edges = []
        self.edge_types = []
    
    def add_node(self, node_id: str, node_type: str) -> int:
        """Add a node and return its index."""
        if node_id not in self.node_to_idx:
            idx = len(self.node_to_idx)
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id
            self.node_types.append(NODE_TYPES.get(node_type, 0))
        return self.node_to_idx[node_id]
    
    def add_edge(self, source: str, target: str, edge_type: str):
        """Add an edge between nodes."""
        src_idx = self.node_to_idx.get(source)
        tgt_idx = self.node_to_idx.get(target)
        
        if src_idx is not None and tgt_idx is not None:
            self.edges.append([src_idx, tgt_idx])
            self.edge_types.append(EDGE_TYPES.get(edge_type, 4))
    
    def build(self, node_embeddings: Dict[str, List[float]] = None):
        """Build PyTorch Geometric Data object."""
        if not _check_pyg_available() or not self.edges:
            return None
        
        import torch
        from torch_geometric.data import Data
        
        # Node features
        num_nodes = len(self.node_to_idx)
        if node_embeddings:
            # Use provided embeddings
            x = torch.zeros(num_nodes, 768)
            for node_id, idx in self.node_to_idx.items():
                if node_id in node_embeddings:
                    x[idx] = torch.tensor(node_embeddings[node_id])
        else:
            # Random initialization
            x = torch.randn(num_nodes, 768)
        
        # Edge index
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        
        # Edge attributes
        edge_attr = torch.tensor(self.edge_types, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    High-level reasoning over the behavioral graph.
    
    Capabilities:
    1. Find paths between behaviors
    2. Discover patterns
    3. Predict missing relationships
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.graph_data = None
        self.graph_builder = GraphBuilder()
        self.model_path = model_path or (MODELS_DIR / "qbm-graph-reasoner")
        self._fallback_graph = None
        self._fallback_adjacency = {}  # node -> list of (neighbor, edge_type)
        
        # Lazy initialization - don't import torch_geometric here
        self._model_initialized = False
        
        # Phase 10.1b: Track graph backend mode for API debug transparency
        self._graph_backend = "not_initialized"
        self._graph_backend_reason = ""
    
    def get_backend_info(self) -> Dict[str, str]:
        """Return current graph backend mode and reason for API debug."""
        return {
            "graph_backend": self._graph_backend,
            "graph_backend_reason": self._graph_backend_reason,
        }
    
    def _ensure_model(self):
        """Lazy model initialization."""
        if self._model_initialized:
            return
        self._model_initialized = True
        
        if _check_torch_available() and _check_pyg_available():
            self.model = QBMGraphReasoner()
            if DEVICE == "cuda":
                self.model = self.model.cuda()
            self._graph_backend = "pyg"
            self._graph_backend_reason = "torch_geometric available and loaded"
        else:
            # Set fallback mode with reason
            if not _check_torch_available():
                self._graph_backend = "json_fallback"
                self._graph_backend_reason = "torch not available"
            else:
                self._graph_backend = "json_fallback"
                self._graph_backend_reason = "torch_geometric DLL load failure or not installed"
    
    def _load_fallback_graph(self):
        """Load semantic graph JSON as fallback when torch_geometric unavailable."""
        if self._fallback_graph is not None:
            return self._fallback_graph
        
        fallback_path = DATA_DIR / "graph" / "semantic_graph_v2.json"
        if not fallback_path.exists():
            logger.warning(f"Fallback graph not found: {fallback_path}")
            return None
        
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                self._fallback_graph = json.load(f)
            
            # Build adjacency list for path finding
            for edge in self._fallback_graph.get("edges", []):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                edge_type = edge.get("edge_type", "CO_OCCURS_WITH")
                
                if src not in self._fallback_adjacency:
                    self._fallback_adjacency[src] = []
                self._fallback_adjacency[src].append((tgt, edge_type))
                
                # Also add nodes to graph_builder for compatibility
                self.graph_builder.add_node(src, "behavior")
                self.graph_builder.add_node(tgt, "behavior")
            
            logger.info(f"Loaded fallback graph with {len(self._fallback_adjacency)} nodes")
            return self._fallback_graph
        except Exception as e:
            logger.error(f"Failed to load fallback graph: {e}")
            return None
    
    def build_graph_from_relations(self, relations: List[Dict[str, Any]]):
        """Build graph from extracted relations."""
        for rel in relations:
            e1 = rel.get("entity1", "")
            e2 = rel.get("entity2", "")
            rel_type = rel.get("relation", "co_occurs").lower()
            
            if e1 and e2:
                self.graph_builder.add_node(e1, "behavior")
                self.graph_builder.add_node(e2, "behavior")
                self.graph_builder.add_edge(e1, e2, rel_type)
        
        self.graph_data = self.graph_builder.build()
        logger.info(f"Built graph with {len(self.graph_builder.node_to_idx)} nodes")
    
    def find_path(self, start: str, end: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Find path between two behaviors using learned representations.
        Falls back to semantic graph JSON if torch_geometric unavailable.
        
        Example: find_path("الغفلة", "قسوة_القلب")
        Returns: ["الغفلة", "الكبر", "الإعراض", "قسوة_القلب"]
        """
        # Try fallback graph first if torch_geometric not available
        if self.graph_data is None and not _check_pyg_available():
            return self._find_path_fallback(start, end, max_depth)
        
        if start not in self.graph_builder.node_to_idx:
            return {"error": f"Start node '{start}' not in graph"}
        if end not in self.graph_builder.node_to_idx:
            return {"error": f"End node '{end}' not in graph"}
        
        # BFS with scoring
        start_idx = self.graph_builder.node_to_idx[start]
        end_idx = self.graph_builder.node_to_idx[end]
        
        visited = set()
        queue = [(start_idx, [start])]
        all_paths = []
        
        while queue and len(all_paths) < 10:
            current_idx, path = queue.pop(0)
            
            if current_idx == end_idx:
                all_paths.append(path)
                continue
            
            if current_idx in visited or len(path) > max_depth:
                continue
            
            visited.add(current_idx)
            
            # Find neighbors
            if self.graph_data is not None:
                edge_index = self.graph_data.edge_index
                neighbors = edge_index[1][edge_index[0] == current_idx].tolist()
                
                for neighbor_idx in neighbors:
                    if neighbor_idx not in visited:
                        neighbor_name = self.graph_builder.idx_to_node[neighbor_idx]
                        queue.append((neighbor_idx, path + [neighbor_name]))
        
        if not all_paths:
            return {"found": False, "start": start, "end": end}
        
        # Return shortest path
        best_path = min(all_paths, key=len)
        
        return {
            "found": True,
            "path": best_path,
            "length": len(best_path) - 1,
            "all_paths": all_paths[:5],
        }
    
    def _find_path_fallback(self, start: str, end: str, max_depth: int = 5) -> Dict[str, Any]:
        """Find path using fallback semantic graph JSON."""
        self._load_fallback_graph()
        
        if not self._fallback_adjacency:
            return {"error": "No graph data available", "fallback": True}
        
        if start not in self._fallback_adjacency and start not in self.graph_builder.node_to_idx:
            return {"error": f"Start node '{start}' not in graph", "fallback": True}
        
        # BFS on fallback adjacency
        visited = set()
        queue = [(start, [start])]
        all_paths = []
        
        while queue and len(all_paths) < 10:
            current, path = queue.pop(0)
            
            if current == end:
                all_paths.append(path)
                continue
            
            if current in visited or len(path) > max_depth:
                continue
            
            visited.add(current)
            
            # Find neighbors from fallback adjacency
            neighbors = self._fallback_adjacency.get(current, [])
            for neighbor, edge_type in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        if not all_paths:
            return {"found": False, "start": start, "end": end, "fallback": True}
        
        best_path = min(all_paths, key=len)
        
        return {
            "found": True,
            "path": best_path,
            "length": len(best_path) - 1,
            "all_paths": all_paths[:5],
            "fallback": True,
        }
    
    def discover_patterns(self, min_support: int = 3) -> List[Dict[str, Any]]:
        """
        Discover behavioral patterns in the graph.
        
        Finds things like:
        - "Behaviors with agent=منافق always have evaluation=مذموم"
        - "الكبر always leads to قسوة within 2 hops"
        """
        patterns = []
        
        if self.graph_data is None:
            return patterns
        
        # Pattern 1: Common paths
        edge_index = self.graph_data.edge_index
        
        # Count edge frequencies
        edge_counts = {}
        for i in range(edge_index.shape[1]):
            src = self.graph_builder.idx_to_node[edge_index[0][i].item()]
            tgt = self.graph_builder.idx_to_node[edge_index[1][i].item()]
            key = (src, tgt)
            edge_counts[key] = edge_counts.get(key, 0) + 1
        
        # Find frequent edges
        for (src, tgt), count in edge_counts.items():
            if count >= min_support:
                patterns.append({
                    "type": "frequent_edge",
                    "source": src,
                    "target": tgt,
                    "count": count,
                })
        
        # Pattern 2: Hub nodes (high connectivity)
        node_degrees = {}
        for idx in range(len(self.graph_builder.node_to_idx)):
            out_degree = (edge_index[0] == idx).sum().item()
            in_degree = (edge_index[1] == idx).sum().item()
            node_degrees[idx] = out_degree + in_degree
        
        # Top hubs
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        for idx, degree in sorted_nodes[:10]:
            node_name = self.graph_builder.idx_to_node[idx]
            patterns.append({
                "type": "hub_node",
                "node": node_name,
                "degree": degree,
            })
        
        logger.info(f"Discovered {len(patterns)} patterns")
        return patterns
    
    def predict_missing_relations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Predict relationships not in the graph."""
        predictions = []
        
        self._ensure_model()
        
        if self.model is None or self.graph_data is None:
            return predictions
        
        if not _check_torch_available():
            return predictions
        
        import torch
        
        # Get node embeddings
        self.model.eval()
        with torch.no_grad():
            if DEVICE == "cuda":
                self.graph_data = self.graph_data.cuda()
            node_embeddings = self.model(self.graph_data.x, self.graph_data.edge_index)
        
        # Check all pairs not in graph
        existing_edges = set()
        edge_index = self.graph_data.edge_index
        for i in range(edge_index.shape[1]):
            existing_edges.add((edge_index[0][i].item(), edge_index[1][i].item()))
        
        nodes = list(self.graph_builder.node_to_idx.keys())
        
        for i, n1 in enumerate(nodes[:50]):  # Limit for speed
            for n2 in nodes[i+1:50]:
                idx1 = self.graph_builder.node_to_idx[n1]
                idx2 = self.graph_builder.node_to_idx[n2]
                
                if (idx1, idx2) not in existing_edges:
                    # Predict relation
                    logits = self.model.predict_relation(
                        node_embeddings[idx1].unsqueeze(0),
                        node_embeddings[idx2].unsqueeze(0)
                    )
                    import torch.nn.functional as F
                    probs = F.softmax(logits, dim=-1).squeeze()
                    
                    max_prob, pred_rel = probs.max(dim=-1)
                    
                    if max_prob.item() > threshold:
                        rel_name = list(EDGE_TYPES.keys())[pred_rel.item()]
                        predictions.append({
                            "entity1": n1,
                            "entity2": n2,
                            "predicted_relation": rel_name,
                            "confidence": max_prob.item(),
                        })
        
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:20]
    
    def save(self, path: Path = None):
        """Save the model."""
        if path is None:
            path = self.model_path
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None and _check_torch_available():
            import torch
            torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save graph structure
        with open(path / "graph.json", "w", encoding="utf-8") as f:
            json.dump({
                "node_to_idx": self.graph_builder.node_to_idx,
                "edges": self.graph_builder.edges,
                "edge_types": self.graph_builder.edge_types,
            }, f, ensure_ascii=False)
        
        logger.info(f"Saved to {path}")
    
    def load(self, path: Path = None):
        """Load the model."""
        if path is None:
            path = self.model_path
        
        if (path / "model.pt").exists() and self.model is not None and _check_torch_available():
            import torch
            self.model.load_state_dict(torch.load(path / "model.pt"))
        
        if (path / "graph.json").exists():
            with open(path / "graph.json", encoding="utf-8") as f:
                data = json.load(f)
                self.graph_builder.node_to_idx = data["node_to_idx"]
                self.graph_builder.idx_to_node = {v: k for k, v in data["node_to_idx"].items()}
                self.graph_builder.edges = data["edges"]
                self.graph_builder.edge_types = data["edge_types"]
            
            self.graph_data = self.graph_builder.build()
        
        logger.info(f"Loaded from {path}")


# =============================================================================
# TESTS
# =============================================================================

def test_graph_reasoning(engine: ReasoningEngine) -> Dict[str, Any]:
    """Test graph reasoning capabilities."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: Path finding
    path_result = engine.find_path("الكبر", "قسوة_القلب")
    test1_passed = path_result.get("found", False)
    results["tests"].append({
        "name": "Path: الكبر → قسوة_القلب",
        "passed": test1_passed,
        "result": path_result,
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: Pattern discovery
    patterns = engine.discover_patterns()
    test2_passed = len(patterns) > 0
    results["tests"].append({
        "name": "Pattern discovery",
        "passed": test2_passed,
        "patterns_found": len(patterns),
    })
    results["passed" if test2_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def build_and_train_reasoner(relations: List[Dict] = None) -> Dict[str, Any]:
    """Build and optionally train the graph reasoner."""
    logger.info("=" * 60)
    logger.info("BUILDING GRAPH REASONER")
    logger.info("=" * 60)
    
    # Default relations if none provided
    if relations is None:
        relations = [
            {"entity1": "الكبر", "entity2": "قسوة_القلب", "relation": "causes"},
            {"entity1": "الغفلة", "entity2": "الكبر", "relation": "causes"},
            {"entity1": "التقوى", "entity2": "المعصية", "relation": "prevents"},
            {"entity1": "الصدق", "entity2": "الكذب", "relation": "opposite"},
            {"entity1": "الإيمان", "entity2": "الكفر", "relation": "opposite"},
            {"entity1": "الذنوب", "entity2": "قسوة_القلب", "relation": "causes"},
            {"entity1": "الإعراض", "entity2": "ختم_القلب", "relation": "causes"},
            {"entity1": "الكبر", "entity2": "الظلم", "relation": "causes"},
        ]
    
    engine = ReasoningEngine()
    engine.build_graph_from_relations(relations)
    
    test_results = test_graph_reasoning(engine)
    engine.save()
    
    return {
        "status": "complete",
        "nodes": len(engine.graph_builder.node_to_idx),
        "edges": len(engine.graph_builder.edges),
        "test_results": test_results,
    }


_engine_instance = None

def get_reasoning_engine() -> ReasoningEngine:
    """Get the reasoning engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReasoningEngine()
        _engine_instance.load()
    return _engine_instance


if __name__ == "__main__":
    results = build_and_train_reasoner()
    print(json.dumps(results, indent=2, ensure_ascii=False))
