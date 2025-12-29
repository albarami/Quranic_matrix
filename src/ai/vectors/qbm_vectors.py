"""
QBM Vector Store using ChromaDB with Arabic embeddings.

This module provides semantic search capabilities for ayat, behaviors, and tafsir
using Arabic-optimized embeddings and ChromaDB for persistence.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class QBMVectorStore:
    """Vector store using ChromaDB with Arabic embeddings."""

    # Arabic embedding models (in order of preference)
    EMBEDDING_MODELS = [
        "aubmindlab/bert-base-arabertv2",  # Best for Arabic
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",  # Alternative
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Fallback
    ]

    def __init__(
        self,
        persist_dir: str = "data/chromadb",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory for ChromaDB persistence.
            embedding_model: HuggingFace model name for embeddings.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Initialize embedder
        self.embedder = None
        self.embedding_dim = 768  # Default for BERT models
        self._init_embedder(embedding_model)

        # Create collections
        self._init_collections()

    def _init_embedder(self, model_name: Optional[str] = None) -> None:
        """Initialize the sentence transformer embedder."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Using mock embeddings.")
            return

        # CI/offline mode: avoid downloading large embedding models.
        if os.getenv("CI") or os.getenv("QBM_OFFLINE") == "1":
            print("CI/offline mode: skipping embedding model load. Using mock embeddings.")
            return

        models_to_try = [model_name] if model_name else self.EMBEDDING_MODELS

        for model in models_to_try:
            if model is None:
                continue
            try:
                self.embedder = SentenceTransformer(model)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
                print(f"Loaded embedding model: {model} (dim={self.embedding_dim})")
                return
            except Exception as e:
                print(f"Failed to load {model}: {e}")
                continue

        print("Warning: No embedding model loaded. Using mock embeddings.")

    def _init_collections(self) -> None:
        """Initialize ChromaDB collections."""
        # Collection for Quranic ayat
        self.ayat = self.client.get_or_create_collection(
            name="qbm_ayat",
            metadata={"hnsw:space": "cosine", "description": "Quranic verses"}
        )

        # Collection for behaviors
        self.behaviors = self.client.get_or_create_collection(
            name="qbm_behaviors",
            metadata={"hnsw:space": "cosine", "description": "Behavioral concepts"}
        )

        # Collection for tafsir
        self.tafsir = self.client.get_or_create_collection(
            name="qbm_tafsir",
            metadata={"hnsw:space": "cosine", "description": "Tafsir explanations"}
        )

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for Arabic text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding.
        """
        if self.embedder is None:
            # Return mock embedding for testing
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(self.embedding_dim)]

        return self.embedder.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        if self.embedder is None:
            return [self.embed(t) for t in texts]

        return self.embedder.encode(texts).tolist()

    # -------------------------------------------------------------------------
    # Ayat Operations
    # -------------------------------------------------------------------------

    def add_ayah(
        self,
        ayah_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an ayah to the vector store.

        Args:
            ayah_id: Unique ID (e.g., "2:7" for Surah 2, Ayah 7).
            text: Arabic text of the ayah.
            metadata: Additional metadata (surah, ayah number, behaviors, etc.).
        """
        meta = metadata or {}
        self.ayat.add(
            ids=[ayah_id],
            embeddings=[self.embed(text)],
            metadatas=[meta],
            documents=[text],
        )

    def add_ayat_batch(
        self,
        ayat: List[Dict[str, Any]],
    ) -> int:
        """
        Add multiple ayat efficiently.

        Args:
            ayat: List of dicts with keys: id, text, metadata.

        Returns:
            Number of ayat added.
        """
        if not ayat:
            return 0

        ids = [a["id"] for a in ayat]
        texts = [a["text"] for a in ayat]
        metadatas = [a.get("metadata", {}) for a in ayat]
        embeddings = self.embed_batch(texts)

        self.ayat.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        return len(ayat)

    # -------------------------------------------------------------------------
    # Behavior Operations
    # -------------------------------------------------------------------------

    def add_behavior(
        self,
        behavior_id: str,
        name_ar: str,
        name_en: str,
        definition: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a behavior to the vector store.

        Args:
            behavior_id: Behavior code (e.g., "BEH_COG_ARROGANCE").
            name_ar: Arabic name.
            name_en: English name.
            definition: Definition/description for embedding.
            metadata: Additional metadata.
        """
        # Create rich text for embedding
        text = f"{name_ar} {name_en} {definition}".strip()
        meta = metadata or {}
        meta.update({"name_ar": name_ar, "name_en": name_en})

        self.behaviors.add(
            ids=[behavior_id],
            embeddings=[self.embed(text)],
            metadatas=[meta],
            documents=[text],
        )

    def add_behaviors_from_graph(self, graph) -> int:
        """
        Add all behaviors from a QBMKnowledgeGraph.

        Args:
            graph: QBMKnowledgeGraph instance.

        Returns:
            Number of behaviors added.
        """
        behaviors = graph.get_nodes_by_type("Behavior")
        count = 0

        for behavior_id, attrs in behaviors:
            self.add_behavior(
                behavior_id=behavior_id,
                name_ar=attrs.get("name_ar", ""),
                name_en=attrs.get("name_en", ""),
                definition=attrs.get("definition", ""),
                metadata={"category": attrs.get("category", "")},
            )
            count += 1

        return count

    # -------------------------------------------------------------------------
    # Tafsir Operations
    # -------------------------------------------------------------------------

    def add_tafsir(
        self,
        tafsir_id: str,
        text: str,
        surah: int,
        ayah: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a tafsir entry to the vector store.

        Args:
            tafsir_id: Unique ID (e.g., "ibn_kathir_2_7").
            text: Tafsir text.
            surah: Surah number.
            ayah: Ayah number.
            source: Tafsir source (e.g., "ibn_kathir").
            metadata: Additional metadata.
        """
        meta = metadata or {}
        meta.update({"surah": surah, "ayah": ayah, "source": source})

        self.tafsir.add(
            ids=[tafsir_id],
            embeddings=[self.embed(text)],
            metadatas=[meta],
            documents=[text],
        )

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search_similar(
        self,
        query: str,
        collection: str = "ayat",
        n: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Semantic search across a collection.

        Args:
            query: Search query in Arabic or English.
            collection: Collection to search ("ayat", "behaviors", "tafsir").
            n: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Dict with ids, documents, metadatas, distances.
        """
        coll = getattr(self, collection, self.ayat)

        query_embedding = self.embed(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n,
        }
        if where:
            kwargs["where"] = where

        return coll.query(**kwargs)

    def search_ayat(
        self,
        query: str,
        n: int = 10,
        surah: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search for similar ayat."""
        where = {"surah": surah} if surah else None
        return self.search_similar(query, "ayat", n, where)

    def search_behaviors(
        self,
        query: str,
        n: int = 10,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for similar behaviors."""
        where = {"category": category} if category else None
        return self.search_similar(query, "behaviors", n, where)

    def search_tafsir(
        self,
        query: str,
        n: int = 10,
        source: Optional[str] = None,
        surah: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search for similar tafsir entries."""
        where = {}
        if source:
            where["source"] = source
        if surah:
            where["surah"] = surah
        return self.search_similar(query, "tafsir", n, where or None)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, int]:
        """Get counts for all collections."""
        return {
            "ayat": self.ayat.count(),
            "behaviors": self.behaviors.count(),
            "tafsir": self.tafsir.count(),
        }

    def clear_collection(self, collection: str) -> None:
        """Clear all items from a collection."""
        coll = getattr(self, collection, None)
        if coll:
            # Get all IDs and delete
            results = coll.get()
            if results["ids"]:
                coll.delete(ids=results["ids"])

    def delete_all(self) -> None:
        """Delete all collections and recreate them."""
        self.client.delete_collection("qbm_ayat")
        self.client.delete_collection("qbm_behaviors")
        self.client.delete_collection("qbm_tafsir")
        self._init_collections()
