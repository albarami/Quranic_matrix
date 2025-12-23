"""
QBM FULL POWER SYSTEM - Utilizing ALL 8x A100 GPUs

Uses EXISTING infrastructure from src/ai/gpu/:
- TorchGPUVectorSearch: Pure PyTorch GPU vector search (Windows-compatible)
- GPUEmbeddingPipeline: Multi-GPU embedding with DataParallel
- CrossEncoderReranker: GPU-accelerated reranking

NO FAISS (not compatible with Windows).
Uses PyTorch tensors on GPU for everything.

Hardware: 8x A100-SXM4-80GB (640GB total VRAM)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TAFSIR_DIR = DATA_DIR / "tafsir"
INDEX_DIR = DATA_DIR / "indexes"


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

def get_gpu_config() -> Dict[str, Any]:
    """Get full GPU configuration."""
    try:
        import torch
        
        num_gpus = torch.cuda.device_count()
        gpus = []
        total_memory = 0
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            total_memory += memory_gb
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_gb": round(memory_gb, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        
        return {
            "num_gpus": num_gpus,
            "total_memory_gb": round(total_memory, 1),
            "gpus": gpus,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# GPU-ACCELERATED GRAPH (PyTorch-based, no external deps)
# =============================================================================

class GPUGraph:
    """
    Graph stored in GPU memory for fast traversal.
    
    Uses PyTorch tensors on GPU for:
    - Fast neighbor lookup
    - Parallel BFS/DFS
    - GPU-accelerated operations
    """
    
    def __init__(self):
        import torch
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.edge_index = None
        self.edge_types = None
        self.num_nodes = 0
        self.num_edges = 0
        
    def build_from_data(self, nodes: List[str], edges: List[Tuple[str, str, str]]):
        """Build graph on GPU from node and edge lists."""
        import torch
        
        # Create node mapping
        for i, node in enumerate(nodes):
            self.node_to_idx[node] = i
            self.idx_to_node[i] = node
        
        self.num_nodes = len(nodes)
        
        # Create edge tensors
        src_indices = []
        dst_indices = []
        edge_type_list = []
        edge_type_map = {}
        
        for src, dst, etype in edges:
            if src in self.node_to_idx and dst in self.node_to_idx:
                src_indices.append(self.node_to_idx[src])
                dst_indices.append(self.node_to_idx[dst])
                
                if etype not in edge_type_map:
                    edge_type_map[etype] = len(edge_type_map)
                edge_type_list.append(edge_type_map[etype])
        
        self.num_edges = len(src_indices)
        
        if self.num_edges > 0:
            self.edge_index = torch.tensor([src_indices, dst_indices], 
                                            dtype=torch.long, device=self.device)
            self.edge_types = torch.tensor(edge_type_list, 
                                            dtype=torch.long, device=self.device)
        
        logger.info(f"GPU Graph: {self.num_nodes} nodes, {self.num_edges} edges on {self.device}")
    
    def get_neighbors(self, node: str) -> List[str]:
        """Get neighbors of a node."""
        if node not in self.node_to_idx or self.edge_index is None:
            return []
        
        node_idx = self.node_to_idx[node]
        mask = self.edge_index[0] == node_idx
        neighbor_indices = self.edge_index[1][mask].cpu().tolist()
        
        return [self.idx_to_node[i] for i in neighbor_indices]
    
    def find_path(self, start: str, end: str, max_depth: int = 5) -> List[str]:
        """Find path using BFS."""
        import torch
        
        if start not in self.node_to_idx or end not in self.node_to_idx:
            return []
        
        if self.edge_index is None:
            return []
        
        start_idx = self.node_to_idx[start]
        end_idx = self.node_to_idx[end]
        
        visited = set([start_idx])
        queue = [(start_idx, [start])]
        
        for _ in range(max_depth):
            if not queue:
                break
            
            next_queue = []
            for node_idx, path in queue:
                mask = self.edge_index[0] == node_idx
                neighbors = self.edge_index[1][mask].cpu().tolist()
                
                for neighbor in neighbors:
                    if neighbor == end_idx:
                        return path + [self.idx_to_node[neighbor]]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append((neighbor, path + [self.idx_to_node[neighbor]]))
            
            queue = next_queue
        
        return []


# =============================================================================
# FULL POWER UNIFIED SYSTEM
# =============================================================================

class FullPowerQBMSystem:
    """
    The COMPLETE QBM system utilizing ALL 8x A100 GPUs.
    
    Uses EXISTING infrastructure:
    - GPUEmbeddingPipeline: Multi-GPU embeddings with DataParallel
    - TorchGPUVectorSearch: PyTorch GPU vector search (Windows-compatible)
    - CrossEncoderReranker: GPU-accelerated reranking
    
    Components:
    - 8x A100 GPUs (640GB VRAM total)
    - All 5 tafsir sources (31,180 entries)
    - 76,597 behavioral annotations
    - Claude/GPT-5 API for reasoning
    """
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info("INITIALIZING FULL POWER QBM SYSTEM")
        logger.info("=" * 70)
        
        # Get GPU config
        self.gpu_config = get_gpu_config()
        logger.info(f"GPUs: {self.gpu_config.get('num_gpus', 0)}x A100")
        logger.info(f"Total VRAM: {self.gpu_config.get('total_memory_gb', 0)} GB")
        
        # Initialize components
        self.embedder = None
        self.vector_search = None
        self.reranker = None
        self.graph = None
        self.tafsir_data = {}
        self.behavioral_data = []
        self.all_texts = []
        self.all_metadata = []
        
        # LLM clients
        self.claude_client = None
        self.openai_client = None
        
        # GNN Reasoning Engine
        self.gnn_reasoner = None
        
        self._initialize_all()
    
    def _initialize_all(self):
        """Initialize all components using EXISTING infrastructure."""
        # Import existing GPU infrastructure
        try:
            from src.ai.gpu import GPUEmbeddingPipeline, TorchGPUVectorSearch, CrossEncoderReranker
            
            # 1. Multi-GPU Embeddings (uses DataParallel across all GPUs)
            logger.info("Initializing GPUEmbeddingPipeline...")
            self.embedder = GPUEmbeddingPipeline(
                model_name="aubmindlab/bert-base-arabertv2",
                batch_size=64,
                use_multi_gpu=True,
            )
            
            # 2. PyTorch GPU Vector Search (Windows-compatible, no FAISS)
            logger.info("Initializing TorchGPUVectorSearch...")
            self.vector_search = TorchGPUVectorSearch(
                embedding_dim=768,
                use_gpu=True,
            )
            
            # 3. Cross-Encoder Reranker
            logger.info("Initializing CrossEncoderReranker...")
            self.reranker = CrossEncoderReranker(
                model_name="amberoad/bert-multilingual-passage-reranking-msmarco",
                batch_size=32,
            )
            
        except ImportError as e:
            logger.error(f"Failed to import GPU infrastructure: {e}")
            logger.info("Falling back to basic implementation...")
        except Exception as e:
            logger.error(f"GPU component init failed: {e}")
        
        # 4. GPU Graph
        try:
            self.graph = GPUGraph()
        except Exception as e:
            logger.error(f"Graph init failed: {e}")
        
        # 5. GNN Reasoning Engine (for multi-hop reasoning)
        try:
            from src.ml.graph_reasoner import ReasoningEngine, PYG_AVAILABLE
            if PYG_AVAILABLE:
                self.gnn_reasoner = ReasoningEngine()
                logger.info("GNN ReasoningEngine initialized (multi-hop reasoning enabled)")
            else:
                logger.warning("torch_geometric not available - GNN reasoning disabled")
        except Exception as e:
            logger.warning(f"GNN ReasoningEngine init failed: {e}")
        
        # 5. Load Tafsir Data
        self._load_tafsir()
        
        # 6. Load Behavioral Annotations
        self._load_behavioral_data()
        
        # 7. Initialize LLM clients
        self._init_llm_clients()
        
        logger.info("=" * 70)
        logger.info("FULL POWER SYSTEM READY")
        logger.info("=" * 70)
    
    def _load_tafsir(self):
        """Load all 5 tafsir sources."""
        sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
        
        for source in sources:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if filepath.exists():
                self.tafsir_data[source] = {}
                with open(filepath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ref = entry.get("reference", {})
                            key = f"{ref.get('surah')}:{ref.get('ayah')}"
                            self.tafsir_data[source][key] = entry.get("text_ar", "")
                logger.info(f"Loaded tafsir: {source} ({len(self.tafsir_data[source])} entries)")
    
    def _load_behavioral_data(self):
        """Load behavioral annotations."""
        filepath = DATA_DIR / "annotations" / "tafsir_behavioral_annotations.jsonl"
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.behavioral_data.append(json.loads(line))
            logger.info(f"Loaded {len(self.behavioral_data)} behavioral annotations")
    
    def _init_llm_clients(self):
        """Initialize LLM API clients (Azure OpenAI GPT-5)."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Azure OpenAI (GPT-5) - Primary
        try:
            from openai import AzureOpenAI
            
            self.azure_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
            logger.info(f"Azure OpenAI client ready (deployment: {self.azure_deployment})")
        except Exception as e:
            logger.warning(f"Azure OpenAI init failed: {e}")
            self.azure_client = None
        
        # Claude (via C1 frontend) - Fallback
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic()
            logger.info("Claude API client ready")
        except:
            self.claude_client = None
        
        # Standard OpenAI - Fallback
        try:
            import openai
            self.openai_client = openai.OpenAI()
            logger.info("OpenAI API client ready")
        except:
            self.openai_client = None
    
    def build_index(self) -> Dict[str, Any]:
        """
        Build the complete vector index from all data.
        
        Uses GPUEmbeddingPipeline with DataParallel across all 8 GPUs.
        Uses TorchGPUVectorSearch for PyTorch-based GPU vector search.
        """
        logger.info("Building vector index using all GPUs...")
        start_time = time.time()
        
        # Collect all texts to embed
        self.all_texts = []
        self.all_metadata = []
        
        # Add tafsir entries
        for source, entries in self.tafsir_data.items():
            for verse_key, text in entries.items():
                if text and len(text) > 20:
                    self.all_texts.append(text[:512])
                    self.all_metadata.append({
                        "type": "tafsir",
                        "source": source,
                        "verse": verse_key,
                        "text": text[:300],
                    })
        
        # Add behavioral annotations
        for ann in self.behavioral_data:
            text = ann.get("context", "")
            if text and len(text) > 20:
                self.all_texts.append(text[:512])
                self.all_metadata.append({
                    "type": "behavior",
                    "behavior": ann.get("behavior_ar", ""),
                    "source": ann.get("source", ""),
                    "verse": f"{ann.get('surah')}:{ann.get('ayah')}",
                    "text": text[:300],
                })
        
        logger.info(f"Embedding {len(self.all_texts)} texts using GPUEmbeddingPipeline...")
        
        # Generate embeddings using existing multi-GPU pipeline
        if self.embedder:
            embeddings = self.embedder.embed_texts(self.all_texts, show_progress=True)
        else:
            logger.error("Embedder not available")
            return {"error": "Embedder not initialized"}
        
        # Build PyTorch GPU index
        if self.vector_search:
            self.vector_search.build_index(embeddings, self.all_metadata, show_progress=True)
        else:
            logger.error("Vector search not available")
            return {"error": "Vector search not initialized"}
        
        elapsed = time.time() - start_time
        rate = len(self.all_texts) / elapsed if elapsed > 0 else 0
        
        logger.info(f"Index built in {elapsed:.1f}s ({rate:.0f} texts/sec)")
        
        # Save index
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_path = INDEX_DIR / "full_power_index.npy"
        self.vector_search.save_index(str(index_path))
        
        return {
            "texts_indexed": len(self.all_texts),
            "time_seconds": round(elapsed, 2),
            "rate_per_second": round(rate, 0),
            "index_path": str(index_path),
        }
    
    def build_graph(self) -> Dict[str, Any]:
        """Build the behavioral graph on GPU and GNN reasoning graph."""
        logger.info("Building GPU graph...")
        
        nodes = set()
        edges = []
        gnn_relations = []  # For GNN multi-hop reasoning
        
        # Add behaviors as nodes
        for ann in self.behavioral_data:
            behavior = ann.get("behavior_ar", "")
            if behavior:
                nodes.add(f"behavior:{behavior}")
        
        # Add verses as nodes
        for source, entries in self.tafsir_data.items():
            for verse_key in entries.keys():
                nodes.add(f"verse:{verse_key}")
        
        # Add edges: behavior mentioned in verse
        # Also build co-occurrence edges for GNN
        verse_behaviors = {}  # verse -> list of behaviors
        
        for ann in self.behavioral_data:
            behavior = ann.get("behavior_ar", "")
            verse = f"{ann.get('surah')}:{ann.get('ayah')}"
            if behavior and verse:
                edges.append((f"verse:{verse}", f"behavior:{behavior}", "mentions"))
                edges.append((f"behavior:{behavior}", f"verse:{verse}", "mentioned_in"))
                
                # Track co-occurrences for GNN
                if verse not in verse_behaviors:
                    verse_behaviors[verse] = []
                verse_behaviors[verse].append(behavior)
        
        # Build co-occurrence edges for GNN (behaviors in same verse)
        for verse, behaviors in verse_behaviors.items():
            for i, b1 in enumerate(behaviors):
                for b2 in behaviors[i+1:]:
                    gnn_relations.append({
                        "entity1": b1,
                        "entity2": b2,
                        "relation": "co_occurs",
                        "verse": verse,
                    })
        
        if self.graph:
            self.graph.build_from_data(list(nodes), edges)
        
        # Build GNN graph for multi-hop reasoning
        if self.gnn_reasoner and gnn_relations:
            self.gnn_reasoner.build_graph_from_relations(gnn_relations)
            logger.info(f"GNN graph built with {len(gnn_relations)} behavioral relations")
        
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "gnn_relations": len(gnn_relations),
        }
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Full-power search using all GPU components.
        
        1. Embed query using GPUEmbeddingPipeline
        2. Vector search using TorchGPUVectorSearch
        3. Rerank using CrossEncoderReranker
        """
        if not self.embedder or not self.vector_search:
            return []
        
        # 1. Embed query
        query_embedding = self.embedder.embed_texts([query], show_progress=False)
        
        # 2. Vector search on GPU
        distances, indices, metadata_results = self.vector_search.search(
            query_embedding, k=top_k * 3
        )
        
        # 3. Prepare candidates for reranking
        candidates = []
        for i, (dist, meta) in enumerate(zip(distances[0], metadata_results[0])):
            text = meta.get("text", "")
            candidates.append({
                "text": text,
                "score": float(dist),
                "metadata": meta,
            })
        
        # 4. Rerank if available
        if self.reranker and candidates:
            candidate_texts = [c["text"] for c in candidates]
            candidate_meta = [c["metadata"] for c in candidates]
            
            reranked = self.reranker.rerank(
                query, candidate_texts, candidate_meta, top_k=top_k
            )
            
            return [
                {
                    "text": r["document"],
                    "score": r["score"],
                    "metadata": r["metadata"],
                }
                for r in reranked
            ]
        
        return candidates[:top_k]
    
    def _normalize_behavior(self, behavior: str) -> str:
        """
        Normalize Arabic behavior name to match graph vocabulary.
        Removes definite article 'ال' and common prefixes.
        """
        # Remove definite article
        if behavior.startswith("ال"):
            behavior = behavior[2:]
        # Remove common prefixes
        if behavior.startswith("ا"):
            behavior = behavior[1:]
        return behavior
    
    def find_behavioral_chain(self, behavior1: str, behavior2: str) -> Dict[str, Any]:
        """
        Find multi-hop path between two behaviors using GNN.
        
        Example: find_behavioral_chain("الكبر", "قسوة القلب")
        Returns: {"path": ["الكبر", "الإعراض", "قسوة القلب"], "length": 2}
        """
        if not self.gnn_reasoner:
            return {"error": "GNN reasoning not available"}
        
        # Normalize behavior names to match graph vocabulary (remove ال prefix)
        b1_normalized = self._normalize_behavior(behavior1)
        b2_normalized = self._normalize_behavior(behavior2)
        
        # Try multiple combinations to find a match
        combinations = [
            (b1_normalized, b2_normalized),  # Both normalized
            (behavior1, behavior2),           # Both original
            (b1_normalized, behavior2),       # First normalized
            (behavior1, b2_normalized),       # Second normalized
        ]
        
        for b1, b2 in combinations:
            result = self.gnn_reasoner.find_path(b1, b2)
            if result.get("found"):
                return result
        
        # Return last result (will contain error info)
        return result
    
    def discover_patterns(self) -> List[Dict[str, Any]]:
        """Discover behavioral patterns using GNN."""
        if not self.gnn_reasoner:
            return []
        return self.gnn_reasoner.discover_patterns()
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Complete RAG pipeline with full hardware utilization + GNN multi-hop reasoning."""
        start_time = time.time()
        
        # 1. Search for relevant content
        search_results = self.search(question, top_k=30)
        
        # 2. Extract behaviors from question for GNN reasoning
        gnn_context = ""
        if self.gnn_reasoner:
            # Common Arabic behavior keywords (with and without definite article)
            # Graph vocabulary uses: كبر, غفلة, كفر, etc. (without ال)
            behavior_keywords_with_al = ["الكبر", "القسوة", "الغفلة", "التوبة", "الإيمان", "الكفر", "النفاق", 
                                        "الصبر", "الشكر", "الحسد", "الرياء", "الإخلاص", "التواضع"]
            behavior_keywords_without_al = ["كبر", "قسوة", "غفلة", "توبة", "إيمان", "كفر", "نفاق",
                                           "صبر", "شكر", "حسد", "رياء", "إخلاص", "تواضع", "قاسي", "ظلم"]
            
            # Find behaviors in question (check both forms)
            found_behaviors = []
            for b in behavior_keywords_with_al + behavior_keywords_without_al:
                if b in question and b not in found_behaviors:
                    found_behaviors.append(b)
            
            if len(found_behaviors) >= 2:
                # Find path between first two behaviors (normalization happens in find_behavioral_chain)
                path_result = self.find_behavioral_chain(found_behaviors[0], found_behaviors[1])
                if path_result.get("found"):
                    gnn_context = f"\n\n[سلسلة سلوكية من GNN]: {' → '.join(path_result['path'])}"
                    gnn_context += f"\n(عدد الخطوات: {path_result['length']})"
            elif len(found_behaviors) == 1:
                # Single behavior - show its connections
                gnn_context = f"\n\n[سلوك مكتشف في الشبكة]: {found_behaviors[0]}"
        
        # 3. Build context
        context_parts = []
        for result in search_results[:15]:
            meta = result.get("metadata", {})
            text = result.get("text", "")
            if meta.get("type") == "tafsir":
                context_parts.append(f"[{meta.get('source')} - {meta.get('verse')}]: {text}")
            else:
                context_parts.append(f"[سلوك: {meta.get('behavior', '')}]: {text}")
        
        context = "\n\n".join(context_parts) + gnn_context
        
        # 4. Call LLM
        answer = self._call_llm(question, context)
        
        elapsed = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "sources": len(search_results),
            "processing_time_ms": round(elapsed * 1000, 2),
        }
    
    def _call_llm(self, question: str, context: str) -> str:
        """Call Azure OpenAI GPT-5 (primary) or Claude/OpenAI (fallback)."""
        system_prompt = """أنت عالم متخصص في تحليل السلوك القرآني وفق منهجية البوزيداني.
استخدم السياق المقدم للإجابة. استشهد بالآيات والتفاسير.
أجب باللغة العربية الفصحى."""
        
        user_message = f"السؤال: {question}\n\nالسياق:\n{context}"
        
        # 1. Try Azure OpenAI GPT-5 (Primary)
        if hasattr(self, 'azure_client') and self.azure_client:
            try:
                response = self.azure_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=4096,
                    temperature=0.7,
                )
                logger.info("Response from Azure OpenAI GPT-5")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Azure OpenAI error: {e}")
        
        # 2. Try Claude (Fallback)
        if self.claude_client:
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                )
                logger.info("Response from Claude")
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Claude error: {e}")
        
        # 3. Try Standard OpenAI (Fallback)
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
                logger.info("Response from OpenAI")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI error: {e}")
        
        return "[LLM not available - check Azure OpenAI configuration]"
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        vector_stats = self.vector_search.get_stats() if self.vector_search else {}
        
        # GNN status
        gnn_status = "not available"
        if self.gnn_reasoner:
            gnn_nodes = len(self.gnn_reasoner.graph_builder.node_to_idx) if self.gnn_reasoner.graph_builder else 0
            gnn_status = f"ready ({gnn_nodes} nodes)" if gnn_nodes > 0 else "ready (not built)"
        
        return {
            "gpu_config": self.gpu_config,
            "embedder": "ready" if self.embedder else "not loaded",
            "vector_search": vector_stats,
            "reranker": "ready" if self.reranker else "not loaded",
            "graph": {
                "nodes": self.graph.num_nodes if self.graph else 0,
                "edges": self.graph.num_edges if self.graph else 0,
            },
            "gnn_reasoner": gnn_status,
            "tafsir_sources": len(self.tafsir_data),
            "behavioral_annotations": len(self.behavioral_data),
            "azure_openai": f"ready ({self.azure_deployment})" if hasattr(self, 'azure_client') and self.azure_client else "not configured",
            "claude_api": "ready" if self.claude_client else "not configured",
            "openai_api": "ready" if self.openai_client else "not configured",
        }


# =============================================================================
# SINGLETON
# =============================================================================

_system = None

def get_full_power_system() -> FullPowerQBMSystem:
    """Get the full power system singleton."""
    global _system
    if _system is None:
        _system = FullPowerQBMSystem()
    return _system


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Initialize system
    system = FullPowerQBMSystem()
    
    # Print status
    print("\n" + "=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)
    status = system.get_status()
    print(json.dumps(status, indent=2, default=str))
    
    # Build index
    print("\n" + "=" * 70)
    print("BUILDING INDEX")
    print("=" * 70)
    index_stats = system.build_index()
    print(json.dumps(index_stats, indent=2))
    
    # Build graph
    print("\n" + "=" * 70)
    print("BUILDING GRAPH")
    print("=" * 70)
    graph_stats = system.build_graph()
    print(json.dumps(graph_stats, indent=2))
    
    # Test search
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)
    results = system.search("ما هو الكبر؟", top_k=5)
    for i, r in enumerate(results):
        print(f"\n{i+1}. Score: {r['score']:.4f}")
        print(f"   Type: {r['metadata'].get('type')}")
        print(f"   Text: {r['text'][:100]}...")
    
    # Test answer
    print("\n" + "=" * 70)
    print("TESTING RAG ANSWER")
    print("=" * 70)
    response = system.answer("ما علاقة الكبر بقسوة القلب؟")
    print(f"Question: {response['question']}")
    print(f"Processing time: {response['processing_time_ms']}ms")
    print(f"Sources used: {response['sources']}")
    print(f"\nAnswer:\n{response['answer'][:500]}...")
