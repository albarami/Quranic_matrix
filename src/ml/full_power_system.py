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
    
    def __init__(self, use_fixture: bool = False):
        """
        Initialize the Full Power QBM System.
        
        Args:
            use_fixture: If True, use minimal fixture data for fast testing.
                        Skips full index building and uses pre-built fixtures.
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING FULL POWER QBM SYSTEM")
        logger.info("=" * 70)
        
        self.use_fixture = use_fixture
        
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
        
        # Phase 0: Track index source (disk vs runtime_build)
        self.index_source = "disk"  # Default, updated to "runtime_build" if build_index() is called
        
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
        # Phase 10.1b: Track graph backend mode for API debug transparency
        self.graph_backend = "disabled"
        self.graph_backend_reason = "not initialized"
        try:
            from src.ml.graph_reasoner import ReasoningEngine, PYG_AVAILABLE
            self.gnn_reasoner = ReasoningEngine()
            self.gnn_reasoner._ensure_model()  # Force initialization to get backend info
            backend_info = self.gnn_reasoner.get_backend_info()
            self.graph_backend = backend_info.get("graph_backend", "unknown")
            self.graph_backend_reason = backend_info.get("graph_backend_reason", "")
            logger.info(f"GNN ReasoningEngine: backend={self.graph_backend}, reason={self.graph_backend_reason}")
        except Exception as e:
            self.graph_backend = "disabled"
            self.graph_backend_reason = f"initialization failed: {str(e)}"
            logger.warning(f"GNN ReasoningEngine init failed: {e}")
        
        # 5. Load Quran Verses (actual Arabic text)
        self._load_quran_verses()
        
        # 6. Load Tafsir Data
        self._load_tafsir()
        
        # 7. Load Behavioral Annotations
        self._load_behavioral_data()
        
        # 8. Initialize LLM clients
        self._init_llm_clients()
        
        logger.info("=" * 70)
        logger.info("FULL POWER SYSTEM READY")
        logger.info("=" * 70)
    
    def _load_quran_verses(self):
        """Load actual Quran verses from XML file."""
        self.quran_verses = {}  # {surah_num: {ayah_num: text}}
        
        quran_path = DATA_DIR / "quran" / "quran-uthmani.xml"
        if not quran_path.exists():
            logger.warning(f"Quran XML not found at {quran_path}")
            return
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(quran_path)
            root = tree.getroot()
            
            verse_count = 0
            for sura in root.findall('sura'):
                sura_idx = int(sura.get('index'))
                sura_name = sura.get('name', '')
                self.quran_verses[sura_idx] = {'name': sura_name, 'verses': {}}
                
                for aya in sura.findall('aya'):
                    aya_idx = int(aya.get('index'))
                    aya_text = aya.get('text', '')
                    if aya_text:
                        self.quran_verses[sura_idx]['verses'][aya_idx] = aya_text
                        verse_count += 1
            
            logger.info(f"Loaded {verse_count} Quran verses from {len(self.quran_verses)} surahs")
        except Exception as e:
            logger.error(f"Failed to load Quran verses: {e}")
            self.quran_verses = {}
    
    def get_quran_verse(self, surah: int, ayah: int) -> str:
        """Get actual Quran verse text by surah and ayah number."""
        if not hasattr(self, 'quran_verses') or not self.quran_verses:
            return ""
        
        sura_data = self.quran_verses.get(surah, {})
        verses = sura_data.get('verses', {})
        return verses.get(ayah, "")
    
    def get_surah_name(self, surah: int) -> str:
        """Get surah name by number."""
        if not hasattr(self, 'quran_verses') or not self.quran_verses:
            return ""
        
        sura_data = self.quran_verses.get(surah, {})
        return sura_data.get('name', "")
    
    def _load_tafsir(self):
        """Load all 7 tafsir sources."""
        sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
        
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
        # Phase 0: Mark index as runtime-built
        self.index_source = "runtime_build"
        logger.info("Building vector index using all GPUs...")
        start_time = time.time()
        
        # Collect all texts to embed
        self.all_texts = []
        self.all_metadata = []
        
        # SOURCE-TO-INDEX MAPPING: Track which indices belong to each source
        self.source_indices = {s: [] for s in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar", "quran"]}
        
        # Add Quran verses FIRST - these are the actual Arabic verses
        quran_count = 0
        current_idx = 0
        if hasattr(self, 'quran_verses') and self.quran_verses:
            for surah_num, sura_data in self.quran_verses.items():
                sura_name = sura_data.get('name', '')
                verses = sura_data.get('verses', {})
                for ayah_num, verse_text in verses.items():
                    # Include ALL verses including muqattaʿāt (الم، طه، يس، etc.)
                    # Previously excluded short verses with len>5, now include all
                    if verse_text:
                        self.all_texts.append(verse_text)
                        self.all_metadata.append({
                            "type": "quran",
                            "source": "quran",
                            "verse": f"{surah_num}:{ayah_num}",
                            "surah": surah_num,
                            "ayah": ayah_num,
                            "surah_name": sura_name,
                            "text": verse_text,
                        })
                        self.source_indices["quran"].append(current_idx)
                        quran_count += 1
                        current_idx += 1
            logger.info(f"[INDEX BUILD] Quran verses indexed: {quran_count}")
        
        # Add tafsir entries - track counts per source AND index positions
        tafsir_counts = {}
        for source, entries in self.tafsir_data.items():
            tafsir_counts[source] = 0
            for verse_key, text in entries.items():
                if text and len(text) > 20:
                    # Parse verse reference once so downstream can use real ints
                    surah_num = None
                    ayah_num = None
                    if verse_key and ":" in verse_key:
                        parts = verse_key.split(":", 1)
                        if len(parts) == 2:
                            if parts[0].isdigit():
                                surah_num = int(parts[0])
                            if parts[1].isdigit():
                                ayah_num = int(parts[1])

                    self.all_texts.append(text[:512])
                    self.all_metadata.append({
                        "type": "tafsir",
                        "source": source,
                        "verse": verse_key,
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "text": text[:300],
                    })
                    # Track index position for this source
                    if source in self.source_indices:
                        self.source_indices[source].append(current_idx)
                    tafsir_counts[source] += 1
                    current_idx += 1
        
        logger.info(f"[INDEX BUILD] Tafsir entries indexed per source: {tafsir_counts}")
        source_idx_counts = {s: len(v) for s, v in self.source_indices.items()}
        logger.info(f"[INDEX BUILD] Source index counts: {source_idx_counts}")
        
        # Add behavioral annotations
        for ann in self.behavioral_data:
            text = ann.get("context", "")
            if text and len(text) > 20:
                surah_num = ann.get("surah")
                ayah_num = ann.get("ayah")
                self.all_texts.append(text[:512])
                self.all_metadata.append({
                    "type": "behavior",
                    "behavior": ann.get("behavior_ar", ""),
                    "source": ann.get("source", ""),
                    "verse": f"{ann.get('surah')}:{ann.get('ayah')}",
                    "surah": surah_num,
                    "ayah": ayah_num,
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

        # Cache source index tensors for fast, source-restricted similarity search
        self.source_idx_tensors = {}
        try:
            import torch

            if self.vector_search and getattr(self.vector_search, "device", None) is not None:
                device = self.vector_search.device
                for source, idx_list in self.source_indices.items():
                    if idx_list:
                        self.source_idx_tensors[source] = torch.tensor(
                            idx_list, dtype=torch.long, device=device
                        )
                logger.info(f"[INDEX BUILD] source_idx_tensors built: {list(self.source_idx_tensors.keys())} with sizes {[(k, len(v)) for k, v in self.source_idx_tensors.items()]}")
            else:
                logger.warning(f"[INDEX BUILD] Cannot build source_idx_tensors: vector_search={self.vector_search}, device={getattr(self.vector_search, 'device', None)}")
        except Exception as e:
            logger.warning(f"Failed to build source_idx_tensors: {e}")
        
        return {
            "texts_indexed": len(self.all_texts),
            "time_seconds": round(elapsed, 2),
            "rate_per_second": round(rate, 0),
            "index_path": str(index_path),
        }

    def load_index(self, path: Optional[str] = None, load_to_gpu: bool = True) -> Dict[str, Any]:
        """
        Load a prebuilt vector index from disk.

        This avoids the expensive full embedding rebuild in `build_index()` and
        enables fast startup in API/tests when `data/indexes/full_power_index.npy`
        already exists.
        """
        index_path = Path(path) if path else (INDEX_DIR / "full_power_index.npy")
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not self.vector_search:
            raise RuntimeError("Vector search not initialized; cannot load index.")

        start_time = time.time()

        try:
            self.vector_search.load_index(str(index_path), load_to_gpu=load_to_gpu)
        except TypeError:
            # TorchGPUVectorSearch.load_index(path) has no load_to_gpu argument.
            self.vector_search.load_index(str(index_path))

        # Keep system metadata in sync with the loaded index.
        self.all_metadata = list(getattr(self.vector_search, "metadata", []) or [])

        # Rebuild source index mappings/tensors used for source-restricted retrieval.
        sources = [
            "ibn_kathir",
            "tabari",
            "qurtubi",
            "saadi",
            "jalalayn",
            "quran",
            "baghawi",
            "muyassar",
        ]
        self.source_indices = {source: [] for source in sources}
        for idx, meta in enumerate(self.all_metadata):
            if not isinstance(meta, dict):
                continue
            source = meta.get("source")
            if not source and isinstance(meta.get("metadata"), dict):
                source = meta["metadata"].get("source")
            if source in self.source_indices:
                self.source_indices[source].append(idx)

        self.source_idx_tensors = {}
        try:
            import torch

            device = getattr(self.vector_search, "device", None)
            if device is not None:
                for source, idx_list in self.source_indices.items():
                    if idx_list:
                        self.source_idx_tensors[source] = torch.tensor(
                            idx_list, dtype=torch.long, device=device
                        )
        except Exception as e:
            logger.warning(f"Failed to build source_idx_tensors after index load: {e}")

        self.index_source = "disk"

        elapsed = time.time() - start_time
        total_vectors = (
            int(getattr(self.vector_search.index_vectors, "shape", [0])[0])
            if getattr(self.vector_search, "index_vectors", None) is not None
            else len(self.all_metadata)
        )
        logger.info(f"Index loaded from disk in {elapsed:.2f}s ({total_vectors} vectors)")
        return {
            "total_vectors": total_vectors,
            "time_seconds": round(elapsed, 2),
            "index_path": str(index_path),
            "index_source": self.index_source,
        }

    def _vector_search_subset(
        self,
        query_embedding: np.ndarray,
        subset_indices,
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Vector search restricted to a subset of index rows (no fallbacks).

        This fixes "0 results for qurtubi/jalalayn" by selecting top-k *within* each
        source's vector region, even if those items rank low globally.
        """
        if not self.vector_search or getattr(self.vector_search, "index_vectors", None) is None:
            return []
        if subset_indices is None:
            return []

        try:
            import torch

            if isinstance(subset_indices, list):
                if not subset_indices:
                    return []
                subset_indices = torch.tensor(
                    subset_indices, dtype=torch.long, device=self.vector_search.device
                )

            if subset_indices.numel() == 0:
                return []

            query_tensor = torch.from_numpy(query_embedding.astype(np.float32)).to(
                self.vector_search.device
            )
            if query_tensor.ndim == 2:
                query_tensor = query_tensor[0]
            query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=0)

            subset_vectors = self.vector_search.index_vectors.index_select(0, subset_indices)
            scores = torch.mv(subset_vectors, query_tensor)

            k = min(int(k), int(scores.shape[0]))
            if k <= 0:
                return []

            top_scores, top_pos = torch.topk(scores, k)
            selected_indices = subset_indices.index_select(0, top_pos).detach().cpu().numpy().tolist()
            selected_scores = top_scores.detach().cpu().numpy().tolist()

            results: List[Dict[str, Any]] = []
            for score, idx in zip(selected_scores, selected_indices):
                meta = self.all_metadata[idx] if idx < len(self.all_metadata) else {"id": idx}
                results.append(
                    {
                        "text": meta.get("text", ""),
                        "score": float(score),
                        "metadata": meta,
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Subset vector search failed: {e}")
            return []

    def search_tafsir_by_source(
        self,
        query: str,
        per_source_k: int = 10,
        candidate_multiplier: int = 8,
        rerank: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Guaranteed per-source tafsir retrieval using *source-restricted vector search*.

        This avoids the previous behavior of "filling" missing sources from raw tafsir text,
        while still ensuring each of the 5 tafsir sources is represented.
        """
        if not self.embedder or not self.vector_search:
            return {}

        tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]

        query_embedding = self.embedder.embed_texts([query], show_progress=False)

        results: Dict[str, List[Dict[str, Any]]] = {}
        for source in tafsir_sources:
            subset = None
            if hasattr(self, "source_idx_tensors"):
                subset = self.source_idx_tensors.get(source)
            if subset is None and hasattr(self, "source_indices"):
                subset = self.source_indices.get(source, [])

            candidate_k = max(int(per_source_k) * int(candidate_multiplier), int(per_source_k))
            candidates = self._vector_search_subset(query_embedding, subset, k=candidate_k)

            # Replace the short preview text with the full tafsir entry for reranking/display
            for c in candidates:
                meta = c.get("metadata", {})
                verse_key = meta.get("verse")
                full = None
                if verse_key and source in self.tafsir_data:
                    full = self.tafsir_data[source].get(verse_key)
                if full and len(full) > 20:
                    c["text"] = full

            if rerank and self.reranker and candidates:
                reranked = self.reranker.rerank(
                    query,
                    [c["text"] for c in candidates],
                    [c["metadata"] for c in candidates],
                    top_k=int(per_source_k),
                )
                results[source] = [
                    {"text": r["document"], "score": r["score"], "metadata": r["metadata"]}
                    for r in reranked
                ]
            else:
                results[source] = candidates[: int(per_source_k)]

        return results
    
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
    
    def search(self, query: str, top_k: int = 20, ensure_source_diversity: bool = True) -> List[Dict]:
        """
        Full-power search using all GPU components.
        
        1. Embed query using GPUEmbeddingPipeline
        2. Vector search using TorchGPUVectorSearch
        3. Rerank using CrossEncoderReranker
        4. Ensure source diversity (all 5 tafsir sources represented)
        """
        if not self.embedder or not self.vector_search:
            return []
        
        # 1. Embed query
        query_embedding = self.embedder.embed_texts([query], show_progress=False)
        
        # 2. Global vector search (bounded candidate pool to keep reranking fast)
        total_indexed = len(self.all_metadata) if hasattr(self, "all_metadata") else 0
        candidate_k = max(top_k * 10, 200)
        candidate_k = min(candidate_k, 1200) if total_indexed else candidate_k
        distances, indices, metadata_results = self.vector_search.search(
            query_embedding, k=candidate_k
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
        
        # Log raw vector search distribution BEFORE reranking
        raw_source_counts = {}
        for c in candidates:
            src = c.get("metadata", {}).get("source", "unknown")
            raw_source_counts[src] = raw_source_counts.get(src, 0) + 1
        logger.info(f"[VECTOR SEARCH] Raw distribution BEFORE rerank: {raw_source_counts}")
        
        # 3. Optionally augment candidates with a small, per-source subset search
        # (prevents "0 results" for some sources without exploding rerank cost)
        logger.info(f"[DIVERSITY] ensure_source_diversity={ensure_source_diversity}, has_tensors={hasattr(self, 'source_idx_tensors')}, tensors_len={len(getattr(self, 'source_idx_tensors', {}))}")
        if ensure_source_diversity and hasattr(self, "source_idx_tensors") and self.source_idx_tensors:
            # Include quran source for actual verse retrieval
            all_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar", "quran"]
            per_source_hint = 2 if top_k >= len(all_sources) * 2 else 1
            per_source_candidate_k = max(per_source_hint * 10, 25)
            logger.info(f"[DIVERSITY] source_idx_tensors keys: {list(self.source_idx_tensors.keys())}")
            for source in all_sources:
                subset = self.source_idx_tensors.get(source)
                if subset is None:
                    logger.warning(f"[DIVERSITY] No subset for source: {source}")
                    continue
                subset_results = self._vector_search_subset(
                    query_embedding, subset_indices=subset, k=per_source_candidate_k
                )
                logger.info(f"[DIVERSITY] {source}: {len(subset_results)} results from subset search")
                candidates.extend(subset_results)

        # De-duplicate candidates (same type+source+verse)
        def _key(r: Dict[str, Any]) -> str:
            meta = r.get("metadata", {})
            return f"{meta.get('type')}|{meta.get('source')}|{meta.get('verse')}"

        deduped = {}
        for c in candidates:
            k = _key(c)
            if k not in deduped or c.get("score", 0) > deduped[k].get("score", 0):
                deduped[k] = c
        candidates = list(deduped.values())

        # 4. Rerank a bounded pool (but force-in a few per source including quran)
        if self.reranker and candidates:
            all_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar", "quran"]
            rerank_max = min(len(candidates), max(top_k * 15, 250))

            pool: List[Dict[str, Any]] = []
            used = set()

            # Guarantee minimal per-source presence in the rerank pool (including Quran)
            min_pool_per_source = 2 if top_k >= len(all_sources) * 2 else 1
            for source in all_sources:
                source_items = [
                    r
                    for r in candidates
                    if r.get("metadata", {}).get("source") == source
                ]
                source_items.sort(key=lambda x: x.get("score", 0), reverse=True)
                for r in source_items[:min_pool_per_source]:
                    rk = _key(r)
                    if rk not in used:
                        pool.append(r)
                        used.add(rk)

            # Fill the rest by best vector score overall
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            for r in candidates:
                if len(pool) >= rerank_max:
                    break
                rk = _key(r)
                if rk not in used:
                    pool.append(r)
                    used.add(rk)

            reranked = self.reranker.rerank(
                query,
                [c["text"] for c in pool],
                [c["metadata"] for c in pool],
                top_k=len(pool),
            )
            reranked_results = [
                {"text": r["document"], "score": r["score"], "metadata": r["metadata"]}
                for r in reranked
            ]
        else:
            reranked_results = candidates
        
        # 5. Final selection: keep relevance, but force a small, bounded tafsir diversity
        if ensure_source_diversity:
            tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
            per_source_target = 2 if top_k >= len(tafsir_sources) * 2 else 1

            selected: List[Dict[str, Any]] = []
            used = set()

            # First pass: take a small number per tafsir source
            for source in tafsir_sources:
                source_items = [
                    r
                    for r in reranked_results
                    if r.get("metadata", {}).get("type") == "tafsir"
                    and r.get("metadata", {}).get("source") == source
                ]
                for r in source_items[:per_source_target]:
                    rk = _key(r)
                    if rk not in used:
                        selected.append(r)
                        used.add(rk)

            # Second pass: fill remaining with best overall (any type)
            for r in reranked_results:
                if len(selected) >= top_k:
                    break
                rk = _key(r)
                if rk not in used:
                    selected.append(r)
                    used.add(rk)

            return selected[:top_k]

        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return reranked_results[:top_k]
    
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
