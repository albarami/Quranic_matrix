"""
Cross-Encoder Reranker for Higher Accuracy Retrieval.

Uses a cross-encoder model to rerank initial retrieval results,
significantly improving precision for top-k results.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improved retrieval accuracy.
    
    Cross-encoders process query-document pairs together, allowing
    for deeper semantic understanding compared to bi-encoders.
    
    Workflow:
    1. Initial retrieval returns top-N candidates (fast, approximate)
    2. Reranker scores each (query, candidate) pair (slower, accurate)
    3. Results reordered by reranker scores
    """

    def __init__(
        self,
        model_name: str = "amberoad/bert-multilingual-passage-reranking-msmarco",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model for reranking.
                Options:
                - "amberoad/bert-multilingual-passage-reranking-msmarco" (multilingual)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (English, fast)
                - "BAAI/bge-reranker-large" (high accuracy)
            device: Device to use ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for reranking.
            max_length: Maximum token length for query + document.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer (use safetensors for torch < 2.6 compatibility)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            # Try safetensors first (safer for torch < 2.6)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, use_safetensors=True
            )
        except Exception:
            # Fallback to pytorch_model.bin with trust_remote_code
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Cross-Encoder Reranker initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents for a query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            metadata: Optional metadata for each document.
            top_k: Return only top-k results (None for all).

        Returns:
            List of dicts with 'document', 'score', 'rank', and 'metadata'.
        """
        if not documents:
            return []

        metadata = metadata or [{}] * len(documents)
        scores = []

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]

                # Create query-document pairs
                pairs = [[query, doc] for doc in batch_docs]

                # Tokenize
                encoded = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Get scores
                outputs = self.model(**encoded)
                logits = outputs.logits

                # Handle both regression (shape [N,1] or [N]) and
                # classification (shape [N,2+]) outputs.
                if logits.ndim == 2 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1)
                    batch_scores = probs[:, 1]
                else:
                    batch_scores = logits.squeeze(-1)

                batch_scores = batch_scores.detach().cpu().numpy()

                # Handle single output
                if batch_scores.ndim == 0:
                    batch_scores = [float(batch_scores)]
                else:
                    batch_scores = batch_scores.tolist()

                scores.extend(batch_scores)

        # Create results with scores
        results = []
        for idx, (doc, score, meta) in enumerate(zip(documents, scores, metadata)):
            results.append({
                "document": doc,
                "score": float(score),
                "original_rank": idx,
                "metadata": meta,
            })

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Add new rank
        for new_rank, result in enumerate(results):
            result["rank"] = new_rank

        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        metadata_list: Optional[List[List[Dict]]] = None,
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank documents for multiple queries.

        Args:
            queries: List of search queries.
            documents_list: List of document lists (one per query).
            metadata_list: Optional metadata for each document list.
            top_k: Return only top-k results per query.

        Returns:
            List of reranked results for each query.
        """
        if metadata_list is None:
            metadata_list = [None] * len(queries)

        results = []
        for query, documents, metadata in zip(queries, documents_list, metadata_list):
            query_results = self.rerank(query, documents, metadata, top_k)
            results.append(query_results)

        return results


class HybridRetriever:
    """
    Hybrid retriever combining vector search with cross-encoder reranking.
    
    Two-stage retrieval:
    1. Fast vector search retrieves top-N candidates
    2. Cross-encoder reranks to get accurate top-K
    """

    def __init__(
        self,
        vector_search,  # GPUVectorSearch instance
        embedding_pipeline,  # GPUEmbeddingPipeline instance
        reranker: Optional[CrossEncoderReranker] = None,
        initial_k: int = 100,
        final_k: int = 10,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_search: GPUVectorSearch instance for initial retrieval.
            embedding_pipeline: GPUEmbeddingPipeline for query embedding.
            reranker: CrossEncoderReranker for reranking (optional).
            initial_k: Number of candidates from vector search.
            final_k: Number of final results after reranking.
        """
        self.vector_search = vector_search
        self.embedding_pipeline = embedding_pipeline
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

        # Store document texts for reranking
        self.documents: List[str] = []

    def set_documents(self, documents: List[str]) -> None:
        """Set document texts for reranking."""
        self.documents = documents

    def search(
        self,
        query: str,
        use_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search with optional reranking.

        Args:
            query: Search query.
            use_reranking: Whether to apply cross-encoder reranking.

        Returns:
            List of search results with scores and metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_texts([query], show_progress=False)

        # Initial vector search
        k = self.initial_k if use_reranking and self.reranker else self.final_k
        distances, indices, metadata_results = self.vector_search.search(query_embedding, k=k)

        # Get results
        results = []
        for idx, (dist, meta) in enumerate(zip(distances[0], metadata_results[0])):
            result = {
                "score": float(dist),
                "rank": idx,
                "metadata": meta,
            }
            # Add document text if available
            if indices[0][idx] < len(self.documents):
                result["document"] = self.documents[indices[0][idx]]
            results.append(result)

        # Apply reranking if enabled
        if use_reranking and self.reranker and self.documents:
            # Get document texts for candidates
            candidate_docs = [r.get("document", "") for r in results]
            candidate_meta = [r.get("metadata", {}) for r in results]

            # Rerank
            reranked = self.reranker.rerank(
                query,
                candidate_docs,
                candidate_meta,
                top_k=self.final_k,
            )

            # Update results with reranked scores
            results = []
            for r in reranked:
                results.append({
                    "document": r["document"],
                    "score": r["score"],
                    "rank": r["rank"],
                    "original_rank": r["original_rank"],
                    "metadata": r["metadata"],
                })

        return results[:self.final_k]
