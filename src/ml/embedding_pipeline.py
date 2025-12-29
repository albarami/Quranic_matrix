"""
QBM Embedding Pipeline - GPU-Trained Unified System

This module provides:
1. Vocabulary management with Label Studio export format
2. GPU-accelerated embeddings using sentence-transformers (Arabic models)
3. Batch processing for all tafsir and annotations
4. FAISS vector database for fast similarity search
5. Cross-encoder reranking for result quality
6. Unified training pipeline connecting all components

ALL COMPONENTS CONNECTED - No silos.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

# ML imports (with fallbacks)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[Warning] NumPy not available")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if TORCH_AVAILABLE else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("[Warning] PyTorch not available, using CPU fallback")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[Warning] sentence-transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[Warning] FAISS not available, using brute-force search")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"
MODELS_DIR = DATA_DIR / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Arabic embedding model (multilingual)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Batch sizes
EMBEDDING_BATCH_SIZE = 64 if TORCH_AVAILABLE else 16
TAFSIR_SOURCES = CANONICAL_TAFSIR_SOURCES


# =============================================================================
# VOCABULARY & LABEL SCHEMA
# =============================================================================

@dataclass
class LabelSchema:
    """Label Studio compatible annotation schema for QBM."""
    
    # Behavioral dimensions (11 total)
    dimensions: Dict[str, List[str]] = field(default_factory=lambda: {
        "organic": ["قلب", "لسان", "عين", "أذن", "يد", "رجل", "وجه", "بطن", "فرج"],
        "situational": ["داخلي", "قولي", "علائقي", "جسدي", "سمة"],
        "systemic": ["عبادي", "أسري", "مجتمعي", "مالي", "قضائي", "سياسي"],
        "spatial": ["مسجد", "بيت", "سوق", "خلوة", "ملأ", "سفر", "حرب"],
        "temporal": ["دنيا", "موت", "برزخ", "قيامة", "آخرة"],
        "agent": ["الله", "مؤمن", "كافر", "منافق", "نبي", "ملائكة", "شيطان", "إنسان"],
        "source": ["وحي", "فطرة", "نفس", "شيطان", "بيئة", "قلب"],
        "evaluation": ["ممدوح", "مذموم", "محايد", "تحذير"],
        "heart_type": ["سليم", "مريض", "ميت", "قاسي", "منيب", "مختوم"],
        "consequence": ["دنيوية", "أخروية", "فردية", "مجتمعية"],
        "relationships": ["سبب", "نتيجة", "نقيض", "مشابه"],
    })
    
    # Behavioral vocabulary (key behaviors)
    behaviors: List[str] = field(default_factory=lambda: [
        # Positive
        "إيمان", "صبر", "شكر", "توبة", "تقوى", "إحسان", "صدق", "أمانة",
        "عدل", "رحمة", "تواضع", "خشوع", "ذكر", "دعاء", "توكل", "رضا",
        "حياء", "زهد", "ورع", "إخلاص", "يقين", "خوف", "رجاء", "محبة",
        # Negative
        "كفر", "نفاق", "كبر", "حسد", "غيبة", "كذب", "ظلم", "فسق",
        "رياء", "غضب", "بخل", "غفلة", "شرك", "فجور", "خيانة", "جهل",
        "عجب", "حقد", "سخرية", "لعن", "سرقة", "زنا", "قتل", "ربا",
    ])
    
    def to_label_studio_config(self) -> str:
        """Export as Label Studio XML config."""
        config = ['<View>']
        config.append('  <Text name="text" value="$text"/>')
        
        for dim_name, values in self.dimensions.items():
            config.append(f'  <Choices name="{dim_name}" toName="text" choice="multiple">')
            for val in values:
                config.append(f'    <Choice value="{val}"/>')
            config.append('  </Choices>')
        
        config.append('  <Choices name="behavior" toName="text" choice="multiple">')
        for behavior in self.behaviors:
            config.append(f'    <Choice value="{behavior}"/>')
        config.append('  </Choices>')
        
        config.append('</View>')
        return '\n'.join(config)
    
    def get_vocab_size(self) -> int:
        """Get total vocabulary size."""
        total = len(self.behaviors)
        for values in self.dimensions.values():
            total += len(values)
        return total
    
    def get_all_labels(self) -> List[str]:
        """Get all labels as flat list."""
        labels = list(self.behaviors)
        for values in self.dimensions.values():
            labels.extend(values)
        return labels


# =============================================================================
# EMBEDDING ENGINE (GPU-Accelerated)
# =============================================================================

class EmbeddingEngine:
    """
    GPU-accelerated embedding engine using sentence-transformers.
    Supports Arabic text through multilingual models.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # MiniLM dimension
        
        if ST_AVAILABLE:
            print(f"[EmbeddingEngine] Loading model: {model_name}")
            print(f"[EmbeddingEngine] Device: {DEVICE}")
            self.model = SentenceTransformer(model_name, device=DEVICE)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"[EmbeddingEngine] Embedding dimension: {self.dimension}")
        else:
            print("[EmbeddingEngine] Using fallback TF-IDF embeddings")
    
    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE, 
               show_progress: bool = True) -> Any:
        """
        Encode texts to embeddings.
        Uses GPU if available, falls back to TF-IDF otherwise.
        """
        if not texts:
            return np.array([]) if NUMPY_AVAILABLE else []
        
        if self.model is not None:
            # GPU-accelerated encoding
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embeddings
        else:
            # Fallback: Simple TF-IDF-like embeddings
            return self._fallback_encode(texts)
    
    def _fallback_encode(self, texts: List[str]) -> Any:
        """Fallback encoding using TF-IDF-like approach."""
        if not NUMPY_AVAILABLE:
            return [[0.0] * 100 for _ in texts]
        
        # Build vocabulary
        vocab = {}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        # Create sparse embeddings
        embeddings = np.zeros((len(texts), min(len(vocab), 1000)))
        for i, text in enumerate(texts):
            for word in text.split():
                if word in vocab and vocab[word] < 1000:
                    embeddings[i, vocab[word]] = 1.0
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings
    
    def encode_single(self, text: str) -> Any:
        """Encode a single text."""
        result = self.encode([text], show_progress=False)
        return result[0] if len(result) > 0 else None


# =============================================================================
# VECTOR DATABASE (FAISS)
# =============================================================================

class VectorDatabase:
    """
    FAISS-based vector database for fast similarity search.
    Falls back to brute-force if FAISS not available.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.id_map = []  # Maps index position to document ID
        self.metadata = {}  # Stores metadata for each document
        
        if FAISS_AVAILABLE:
            # Use IVF index for large datasets
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
            print(f"[VectorDB] FAISS index created (dim={dimension})")
        else:
            self.embeddings = []
            print(f"[VectorDB] Using brute-force search (dim={dimension})")
    
    def add(self, doc_id: str, embedding: Any, metadata: Dict[str, Any] = None):
        """Add a document to the index."""
        if NUMPY_AVAILABLE:
            embedding = np.array(embedding).astype('float32').reshape(1, -1)
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding) if FAISS_AVAILABLE else None
        
        if FAISS_AVAILABLE:
            self.index.add(embedding)
        else:
            self.embeddings.append(embedding.flatten() if NUMPY_AVAILABLE else embedding)
        
        self.id_map.append(doc_id)
        if metadata:
            self.metadata[doc_id] = metadata
    
    def add_batch(self, doc_ids: List[str], embeddings: Any, 
                  metadata_list: List[Dict[str, Any]] = None):
        """Add multiple documents in batch."""
        if NUMPY_AVAILABLE:
            embeddings = np.array(embeddings).astype('float32')
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embeddings)
        
        if FAISS_AVAILABLE:
            self.index.add(embeddings)
        else:
            for emb in embeddings:
                self.embeddings.append(emb)
        
        self.id_map.extend(doc_ids)
        
        if metadata_list:
            for doc_id, meta in zip(doc_ids, metadata_list):
                self.metadata[doc_id] = meta
    
    def search(self, query_embedding: Any, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents."""
        if NUMPY_AVAILABLE:
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(query_embedding)
        
        if FAISS_AVAILABLE:
            scores, indices = self.index.search(query_embedding, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.id_map):
                    doc_id = self.id_map[idx]
                    results.append((doc_id, float(score), self.metadata.get(doc_id, {})))
            return results
        else:
            # Brute-force search
            return self._brute_force_search(query_embedding, top_k)
    
    def _brute_force_search(self, query: Any, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Fallback brute-force search."""
        if not self.embeddings:
            return []
        
        if NUMPY_AVAILABLE:
            query = np.array(query).flatten()
            scores = []
            for emb in self.embeddings:
                emb = np.array(emb).flatten()
                # Cosine similarity
                score = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
                scores.append(score)
            
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                doc_id = self.id_map[idx]
                results.append((doc_id, float(scores[idx]), self.metadata.get(doc_id, {})))
            return results
        
        return []
    
    def save(self, path: Path):
        """Save the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, str(path / "faiss.index"))
        
        with open(path / "id_map.json", "w") as f:
            json.dump(self.id_map, f)
        
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        
        print(f"[VectorDB] Saved to {path}")
    
    def load(self, path: Path):
        """Load the index from disk."""
        if FAISS_AVAILABLE and (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))
        
        if (path / "id_map.json").exists():
            with open(path / "id_map.json") as f:
                self.id_map = json.load(f)
        
        if (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                self.metadata = json.load(f)
        
        print(f"[VectorDB] Loaded from {path}")
    
    def size(self) -> int:
        """Get number of documents in index."""
        return len(self.id_map)


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================

class Reranker:
    """
    Cross-encoder reranker for improving search result quality.
    """
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        
        if ST_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name, device=DEVICE)
                print(f"[Reranker] Loaded model: {model_name}")
            except Exception as e:
                print(f"[Reranker] Could not load model: {e}")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        Returns list of (original_index, score) tuples.
        """
        if not documents:
            return []
        
        if self.model is not None:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)
            
            # Sort by score
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            return indexed_scores[:top_k]
        else:
            # Fallback: simple keyword matching
            return self._fallback_rerank(query, documents, top_k)
    
    def _fallback_rerank(self, query: str, documents: List[str], 
                         top_k: int) -> List[Tuple[int, float]]:
        """Fallback reranking using keyword overlap."""
        query_terms = set(query.lower().split())
        scores = []
        
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / (len(query_terms) + 1)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# UNIFIED TRAINING PIPELINE
# =============================================================================

class UnifiedPipeline:
    """
    Complete unified pipeline that:
    1. Loads all data (spans, tafsir from 5 sources)
    2. Creates embeddings using GPU
    3. Builds FAISS index
    4. Provides search with reranking
    
    ALL CONNECTED - No silos.
    """
    
    def __init__(self):
        self.schema = LabelSchema()
        self.embedding_engine = EmbeddingEngine()
        self.vector_db = VectorDatabase(dimension=self.embedding_engine.dimension)
        self.reranker = Reranker()
        
        self.stats = {
            "spans_indexed": 0,
            "tafsir_indexed": 0,
            "total_documents": 0,
        }
    
    def index_spans(self, spans: List[Dict[str, Any]], batch_size: int = EMBEDDING_BATCH_SIZE):
        """Index all behavioral annotations."""
        print(f"[Pipeline] Indexing {len(spans)} spans...")
        
        texts = []
        doc_ids = []
        metadata_list = []
        
        for i, span in enumerate(spans):
            text = span.get("text_ar", "")
            if not text:
                continue
            
            texts.append(text)
            doc_ids.append(f"span:{i}")
            
            ref = span.get("reference", {})
            metadata_list.append({
                "type": "span",
                "surah": ref.get("surah"),
                "ayah": ref.get("ayah"),
                "agent": span.get("agent", {}).get("type"),
                "evaluation": span.get("normative", {}).get("evaluation"),
                "text_preview": text[:100],
            })
        
        # Batch encode
        if texts:
            print(f"[Pipeline] Encoding {len(texts)} span texts...")
            embeddings = self.embedding_engine.encode(texts, batch_size=batch_size)
            
            # Add to vector DB
            self.vector_db.add_batch(doc_ids, embeddings, metadata_list)
            self.stats["spans_indexed"] = len(texts)
        
        print(f"[Pipeline] Indexed {self.stats['spans_indexed']} spans")
    
    def index_tafsir(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """Index all 5 tafsir sources."""
        print(f"[Pipeline] Indexing tafsir from {len(TAFSIR_SOURCES)} sources...")
        
        for source in TAFSIR_SOURCES:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if not filepath.exists():
                print(f"[Pipeline] Missing: {source}")
                continue
            
            texts = []
            doc_ids = []
            metadata_list = []
            
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        ref = entry.get("reference", {})
                        text = entry.get("text_ar", "")
                        
                        if text and ref.get("surah") and ref.get("ayah"):
                            texts.append(text[:512])  # Truncate long texts
                            doc_ids.append(f"tafsir:{source}:{ref['surah']}:{ref['ayah']}")
                            metadata_list.append({
                                "type": "tafsir",
                                "source": source,
                                "surah": ref["surah"],
                                "ayah": ref["ayah"],
                                "text_preview": text[:100],
                            })
            
            if texts:
                print(f"[Pipeline] Encoding {len(texts)} {source} entries...")
                embeddings = self.embedding_engine.encode(texts, batch_size=batch_size)
                self.vector_db.add_batch(doc_ids, embeddings, metadata_list)
                self.stats["tafsir_indexed"] += len(texts)
            
            print(f"[Pipeline] Indexed {source}: {len(texts)} entries")
        
        print(f"[Pipeline] Total tafsir indexed: {self.stats['tafsir_indexed']}")
    
    def build_full_index(self, spans: List[Dict[str, Any]]):
        """Build complete index with all data."""
        start_time = time.time()
        
        print("=" * 60)
        print("BUILDING UNIFIED INDEX")
        print("=" * 60)
        
        # Index spans
        self.index_spans(spans)
        
        # Index all tafsir
        self.index_tafsir()
        
        self.stats["total_documents"] = self.vector_db.size()
        
        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"INDEX COMPLETE in {elapsed:.2f}s")
        print(f"Total documents: {self.stats['total_documents']}")
        print("=" * 60)
        
        return self.stats
    
    def search(self, query: str, top_k: int = 20, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Search across ALL indexed data (spans + all tafsir).
        Optionally reranks results for better quality.
        """
        # Encode query
        query_embedding = self.embedding_engine.encode_single(query)
        if query_embedding is None:
            return []
        
        # Vector search
        results = self.vector_db.search(query_embedding, top_k=top_k * 2 if rerank else top_k)
        
        if not results:
            return []
        
        # Rerank if enabled
        if rerank and self.reranker.model is not None:
            documents = [r[2].get("text_preview", "") for r in results]
            reranked = self.reranker.rerank(query, documents, top_k=top_k)
            
            final_results = []
            for orig_idx, score in reranked:
                doc_id, vec_score, metadata = results[orig_idx]
                final_results.append({
                    "doc_id": doc_id,
                    "vector_score": vec_score,
                    "rerank_score": score,
                    **metadata,
                })
            return final_results
        else:
            return [
                {"doc_id": doc_id, "score": score, **metadata}
                for doc_id, score, metadata in results[:top_k]
            ]
    
    def save(self, path: Path = None):
        """Save the pipeline state."""
        if path is None:
            path = EMBEDDINGS_DIR / "unified_index"
        
        self.vector_db.save(path)
        
        with open(path / "stats.json", "w") as f:
            json.dump(self.stats, f)
        
        print(f"[Pipeline] Saved to {path}")
    
    def load(self, path: Path = None):
        """Load the pipeline state."""
        if path is None:
            path = EMBEDDINGS_DIR / "unified_index"
        
        if path.exists():
            self.vector_db.load(path)
            
            if (path / "stats.json").exists():
                with open(path / "stats.json") as f:
                    self.stats = json.load(f)
            
            print(f"[Pipeline] Loaded from {path}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "vocab_size": self.schema.get_vocab_size(),
            "embedding_model": self.embedding_engine.model_name,
            "device": DEVICE,
            "faiss_available": FAISS_AVAILABLE,
            "gpu_available": TORCH_AVAILABLE,
        }


# =============================================================================
# SINGLETON AND INITIALIZATION
# =============================================================================

_pipeline_instance = None

def get_pipeline() -> UnifiedPipeline:
    """Get or create the unified pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = UnifiedPipeline()
    return _pipeline_instance


def build_and_save_index(spans: List[Dict[str, Any]]):
    """Build and save the complete index."""
    pipeline = get_pipeline()
    pipeline.build_full_index(spans)
    pipeline.save()
    return pipeline.get_stats()


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Unified Pipeline...")
    
    pipeline = UnifiedPipeline()
    print(f"Stats: {pipeline.get_stats()}")
    
    # Test encoding
    test_texts = ["الصبر من أعظم الأخلاق", "الكبر مذموم في القرآن"]
    embeddings = pipeline.embedding_engine.encode(test_texts)
    print(f"Encoded {len(test_texts)} texts, shape: {embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings)}")
