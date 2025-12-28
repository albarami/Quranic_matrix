"""
GPU-Accelerated Vector Search.

Provides fast similarity search with CUDA support for large-scale retrieval.
Supports both FAISS (if available) and pure PyTorch GPU search (Windows-compatible).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TorchGPUVectorSearch:
    """
    Pure PyTorch GPU-accelerated vector search (Windows-compatible).
    
    Uses all available GPUs for fast similarity search without FAISS.
    Works on Windows with 8x A100 GPUs using DataParallel.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        use_gpu: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize PyTorch GPU vector search.

        Args:
            embedding_dim: Dimension of embeddings.
            use_gpu: Use GPU acceleration if available.
            device: Specific device ('cuda', 'cuda:0', 'cpu', or None for auto).
        """
        self.embedding_dim = embedding_dim
        
        if device is None:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.index_vectors: Optional[torch.Tensor] = None
        self.metadata: List[Dict] = []
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"PyTorch GPU Vector Search initialized:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Device: {self.device}")
        print(f"  Available GPUs: {self.n_gpus}")

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Build index from embeddings (load to GPU).

        Args:
            embeddings: NumPy array of shape (n, embedding_dim).
            metadata: Optional list of metadata dicts for each embedding.
            show_progress: Show progress during indexing.
        """
        if show_progress:
            print(f"Loading {len(embeddings)} vectors to GPU...")

        # Convert to tensor and move to GPU
        self.index_vectors = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        
        # Normalize for cosine similarity
        self.index_vectors = torch.nn.functional.normalize(self.index_vectors, p=2, dim=1)

        self.metadata = metadata or [{"id": i} for i in range(len(embeddings))]

        if show_progress:
            print(f"Index built: {self.index_vectors.shape[0]} vectors on {self.device}")
            if self.device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated() / (1024**3)
                print(f"GPU memory used: {mem_gb:.2f} GB")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Search for similar vectors using GPU-accelerated cosine similarity.

        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim).
            k: Number of results per query.

        Returns:
            Tuple of (distances, indices, metadata_results).
        """
        if self.index_vectors is None:
            raise ValueError("Index not built. Call build_index first.")

        # Convert query to tensor
        query_tensor = torch.from_numpy(query_embeddings.astype(np.float32)).to(self.device)
        
        # Handle single query
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0)

        # Normalize query
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        # Shape: (n_queries, n_index)
        similarities = torch.mm(query_tensor, self.index_vectors.t())

        # Get top-k
        k = min(k, self.index_vectors.shape[0])
        distances, indices = torch.topk(similarities, k, dim=1)

        # Convert to numpy
        distances = distances.cpu().numpy()
        indices = indices.cpu().numpy()

        # Get metadata for results
        metadata_results = []
        for query_indices in indices:
            query_metadata = []
            for idx in query_indices:
                if 0 <= idx < len(self.metadata):
                    query_metadata.append(self.metadata[idx])
                else:
                    query_metadata.append({"id": int(idx)})
            metadata_results.append(query_metadata)

        return distances, indices, metadata_results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Batch search for large query sets.

        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim).
            k: Number of results per query.
            batch_size: Number of queries per batch.
            show_progress: Show progress.

        Returns:
            Tuple of (distances, indices, metadata_results).
        """
        n_queries = len(query_embeddings)
        all_distances = []
        all_indices = []
        all_metadata = []

        for i in range(0, n_queries, batch_size):
            batch = query_embeddings[i:i + batch_size]
            distances, indices, metadata = self.search(batch, k)
            all_distances.append(distances)
            all_indices.append(indices)
            all_metadata.extend(metadata)

            if show_progress and (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + batch_size, n_queries)}/{n_queries} queries")

        return np.vstack(all_distances), np.vstack(all_indices), all_metadata

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        if self.index_vectors is None:
            raise ValueError("No index to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save vectors
        np.save(str(path), self.index_vectors.cpu().numpy())

        # Save metadata
        metadata_path = str(path) + ".metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        print(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        path = Path(path)

        embeddings = np.load(str(path))
        self.index_vectors = torch.from_numpy(embeddings).to(self.device)

        # Load metadata
        metadata_path = str(path) + ".metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"Index loaded: {self.index_vectors.shape[0]} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index_vectors is None:
            return {"status": "not_built"}

        stats = {
            "status": "ready",
            "total_vectors": self.index_vectors.shape[0],
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "n_gpus": self.n_gpus,
        }

        if self.device.type == "cuda":
            stats["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)

        return stats


class GPUVectorSearch:
    """
    GPU-accelerated vector similarity search using FAISS (Linux) or PyTorch (Windows).
    
    Automatically selects the best backend:
    - FAISS-GPU on Linux (if available)
    - PyTorch GPU on Windows (always works with CUDA)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        """
        Initialize GPU vector search.

        Args:
            embedding_dim: Dimension of embeddings.
            index_type: FAISS index type ('Flat', 'IVFFlat', 'IVFPQ', 'HNSW').
            nlist: Number of clusters for IVF indexes.
            use_gpu: Use GPU acceleration if available.
            gpu_id: GPU device ID to use.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.gpu_id = gpu_id
        
        # Check if FAISS-GPU is available
        self.use_faiss = FAISS_AVAILABLE and faiss.get_num_gpus() > 0
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_faiss:
            print(f"Using FAISS-GPU backend ({faiss.get_num_gpus()} GPUs)")
            self._torch_search = None
        else:
            print(f"Using PyTorch GPU backend ({torch.cuda.device_count()} GPUs)")
            self._torch_search = TorchGPUVectorSearch(embedding_dim, use_gpu)

        self.index = None
        self.metadata: List[Dict] = []
        self.is_trained = False

    def _create_index(self, embeddings: np.ndarray):
        """Create FAISS index based on configuration."""
        n_vectors = len(embeddings)

        if self.index_type == "Flat":
            # Exact search - best accuracy, slower for large datasets
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors

        elif self.index_type == "IVFFlat":
            # Inverted file with flat storage - good balance
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(self.nlist, n_vectors // 10)  # Adjust nlist based on data size
            nlist = max(nlist, 1)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

        elif self.index_type == "IVFPQ":
            # Inverted file with product quantization - memory efficient
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(self.nlist, n_vectors // 10)
            nlist = max(nlist, 1)
            m = 8  # Number of subquantizers
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8)

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World - fast approximate search
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Build index from embeddings.

        Args:
            embeddings: NumPy array of shape (n, embedding_dim).
            metadata: Optional list of metadata dicts for each embedding.
            show_progress: Show progress during indexing.
        """
        # Use PyTorch backend if FAISS-GPU not available
        if self._torch_search is not None:
            self._torch_search.build_index(embeddings, metadata, show_progress)
            self.metadata = self._torch_search.metadata
            return

        # FAISS-GPU path
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        if show_progress:
            print(f"Building FAISS index for {len(embeddings)} vectors...")

        self.index = self._create_index(embeddings)

        if hasattr(self.index, 'train') and not self.index.is_trained:
            if show_progress:
                print("Training index...")
            self.index.train(embeddings)
            self.is_trained = True

        if show_progress:
            print("Adding vectors to index...")
        self.index.add(embeddings)

        if self.use_gpu:
            if show_progress:
                print(f"Moving index to GPU {self.gpu_id}...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)

        self.metadata = metadata or [{"id": i} for i in range(len(embeddings))]

        if show_progress:
            print(f"Index built: {self.index.ntotal} vectors")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Search for similar vectors.

        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim).
            k: Number of results per query.
            nprobe: Number of clusters to search (for IVF indexes).

        Returns:
            Tuple of (distances, indices, metadata_results).
        """
        # Use PyTorch backend if FAISS-GPU not available
        if self._torch_search is not None:
            return self._torch_search.search(query_embeddings, k)

        # FAISS-GPU path
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query_embeddings, k)

        metadata_results = []
        for query_indices in indices:
            query_metadata = []
            for idx in query_indices:
                if 0 <= idx < len(self.metadata):
                    query_metadata.append(self.metadata[idx])
                else:
                    query_metadata.append({"id": int(idx)})
            metadata_results.append(query_metadata)

        return distances, indices, metadata_results

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        # Use PyTorch backend if FAISS-GPU not available
        if self._torch_search is not None:
            self._torch_search.save_index(path)
            return

        if self.index is None:
            raise ValueError("No index to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, str(path))
        else:
            faiss.write_index(self.index, str(path))

        metadata_path = str(path) + ".metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        print(f"Index saved to {path}")

    def load_index(self, path: str, load_to_gpu: bool = True) -> None:
        """Load index from disk."""
        # Use PyTorch backend if FAISS-GPU not available
        if self._torch_search is not None:
            self._torch_search.load_index(path)
            self.metadata = self._torch_search.metadata
            return

        path = Path(path)
        self.index = faiss.read_index(str(path))

        if load_to_gpu and self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)

        metadata_path = str(path) + ".metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"Index loaded: {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        # Use PyTorch backend if FAISS-GPU not available
        if self._torch_search is not None:
            return self._torch_search.get_stats()

        if self.index is None:
            return {"status": "not_built"}

        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "on_gpu": self.use_gpu,
        }
