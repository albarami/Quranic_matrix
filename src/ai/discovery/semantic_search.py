"""
Semantic Search Engine for QBM Discovery System.

Provides GPU-accelerated semantic search across all Quranic annotations,
with support for Arabic queries, reranking, and filtering.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch

from ..gpu import GPUEmbeddingPipeline, GPUVectorSearch, CrossEncoderReranker


class SemanticSearchEngine:
    """
    Unified semantic search across all QBM data.
    
    Features:
    - GPU-accelerated embedding and search
    - Cross-encoder reranking for top results
    - Filtering by source, type, surah, behavior
    - Arabic and English query support
    """

    def __init__(
        self,
        embeddings_path: str = "data/embeddings/annotations_embeddings.npy",
        index_path: str = "data/embeddings/gpu_index.npy",
        use_reranker: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize semantic search engine.

        Args:
            embeddings_path: Path to precomputed embeddings.
            index_path: Path to GPU vector index.
            use_reranker: Whether to use cross-encoder reranking.
            device: Device for inference ('cuda', 'cpu', or None for auto).
        """
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)
        self.use_reranker = use_reranker

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize components
        print("Initializing Semantic Search Engine...")
        
        self.embedding_pipeline = GPUEmbeddingPipeline(
            model_name="aubmindlab/bert-base-arabertv2",
            batch_size=32,
            use_multi_gpu=True,
        )

        self.vector_search = GPUVectorSearch(
            embedding_dim=768,
            use_gpu=(self.device == "cuda"),
        )

        # Load index if exists
        if self.index_path.exists():
            self.vector_search.load_index(str(self.index_path))
            print(f"Loaded index: {self.vector_search.get_stats()}")
        else:
            print(f"Warning: Index not found at {self.index_path}")

        # Initialize reranker if requested
        if use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name="amberoad/bert-multilingual-passage-reranking-msmarco",
                batch_size=32,
            )
        else:
            self.reranker = None

        print("Semantic Search Engine ready.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant annotations.

        Args:
            query: Search query (Arabic or English).
            top_k: Number of results to return.
            rerank_top_n: Number of candidates to rerank (if reranker enabled).
            filters: Optional filters (source, type, surah, behavior).

        Returns:
            List of search results with scores and metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_texts([query], show_progress=False)

        # Initial retrieval
        retrieve_k = rerank_top_n if self.reranker else top_k
        distances, indices, metadata_results = self.vector_search.search(
            query_embedding, k=retrieve_k
        )

        # Build results
        results = []
        for i, (dist, meta) in enumerate(zip(distances[0], metadata_results[0])):
            result = {
                "rank": i + 1,
                "score": float(dist),
                "source": meta.get("source", ""),
                "surah": meta.get("surah", 0),
                "ayah": meta.get("ayah", 0),
                "type": meta.get("type", ""),
                "text": meta.get("text", meta.get("context", "")),
                "behavior": meta.get("behavior", ""),
                "metadata": meta,
            }
            results.append(result)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Rerank if enabled
        if self.reranker and len(results) > 0:
            documents = [r["text"] for r in results]
            metadata = [r["metadata"] for r in results]
            
            reranked = self.reranker.rerank(
                query=query,
                documents=documents,
                metadata=metadata,
                top_k=top_k,
            )

            # Update results with reranked scores
            results = []
            for i, item in enumerate(reranked):
                result = {
                    "rank": i + 1,
                    "score": item["score"],
                    "rerank_score": item["score"],
                    "original_rank": item["original_rank"] + 1,
                    "source": item["metadata"].get("source", ""),
                    "surah": item["metadata"].get("surah", 0),
                    "ayah": item["metadata"].get("ayah", 0),
                    "type": item["metadata"].get("type", ""),
                    "text": item["document"],
                    "behavior": item["metadata"].get("behavior", ""),
                    "metadata": item["metadata"],
                }
                results.append(result)
        else:
            results = results[:top_k]

        return results

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply filters to search results."""
        filtered = []

        for result in results:
            include = True

            if "source" in filters and filters["source"]:
                if result["source"] != filters["source"]:
                    include = False

            if "type" in filters and filters["type"]:
                if result["type"] != filters["type"]:
                    include = False

            if "surah" in filters and filters["surah"]:
                if result["surah"] != filters["surah"]:
                    include = False

            if "behavior" in filters and filters["behavior"]:
                if filters["behavior"].lower() not in result["behavior"].lower():
                    include = False

            if include:
                filtered.append(result)

        return filtered

    def search_by_behavior(
        self,
        behavior: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for annotations related to a specific behavior.

        Args:
            behavior: Behavior name (Arabic or English).
            top_k: Number of results.

        Returns:
            List of relevant annotations.
        """
        return self.search(
            query=behavior,
            top_k=top_k,
            filters={"type": "behavior"},
        )

    def search_by_surah(
        self,
        surah: int,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific surah.

        Args:
            surah: Surah number (1-114).
            query: Search query.
            top_k: Number of results.

        Returns:
            List of relevant annotations from the surah.
        """
        return self.search(
            query=query,
            top_k=top_k,
            filters={"surah": surah},
        )

    def find_similar(
        self,
        text: str,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find annotations similar to given text.

        Args:
            text: Text to find similar annotations for.
            top_k: Number of results.
            exclude_self: Exclude exact matches.

        Returns:
            List of similar annotations.
        """
        results = self.search(text, top_k=top_k + 1 if exclude_self else top_k)

        if exclude_self and len(results) > 0:
            # Remove exact match if present
            results = [r for r in results if r["text"] != text][:top_k]

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple queries.

        Args:
            queries: List of search queries.
            top_k: Number of results per query.
            show_progress: Show progress.

        Returns:
            List of result lists, one per query.
        """
        all_results = []

        for i, query in enumerate(queries):
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing query {i + 1}/{len(queries)}")

            results = self.search(query, top_k=top_k, rerank_top_n=top_k)
            all_results.append(results)

        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "index_stats": self.vector_search.get_stats(),
            "device": self.device,
            "reranker_enabled": self.reranker is not None,
            "embedding_model": "aubmindlab/bert-base-arabertv2",
        }
