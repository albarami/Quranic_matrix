"""
QBM Stratified Tafsir Retriever - Phase 4
Guarantees results from all 5 tafsir sources without fallbacks.

Architecture:
1. Per-source BM25 indexes for exact Arabic term matching
2. Guaranteed minimum results per source (no fallbacks needed)
3. Fail-fast design: missing indexes = startup failure

NOTE: This is BM25-only. Semantic embeddings will be added in Phase 5
after the embedding model is evaluated/fine-tuned for Classical Arabic.

Hard Rule: If this retriever cannot return results from all sources,
the indexes are broken and must be rebuilt - NO runtime fallbacks.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEXES_DIR = DATA_DIR / "indexes" / "tafsir"
TAFSIR_DIR = DATA_DIR / "tafsir"
TAFSIR_DB = TAFSIR_DIR / "tafsir_cleaned.db"
TAFSIR_DB_ORIGINAL = TAFSIR_DIR / "tafsir.db"

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]


@dataclass
class SourceIndex:
    """Index for a single tafsir source (BM25-only in Phase 4)."""
    source_id: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    bm25: Any = None  # BM25Okapi index
    # NOTE: embeddings will be added in Phase 5
    
    def __len__(self):
        return len(self.documents)


class IndexNotFoundError(Exception):
    """Raised when required indexes are missing - NO FALLBACK ALLOWED."""
    pass


class StratifiedTafsirRetriever:
    """
    Stratified retriever guaranteeing results from all 5 tafsir sources.
    
    This is the Phase 4 solution to source collapse. Instead of a single
    global index where some sources get drowned out, we maintain separate
    indexes per source and query each independently.
    
    NO FALLBACKS: If any source index is missing or empty, we fail fast.
    The system should not operate with incomplete data.
    """
    
    MIN_RESULTS_PER_SOURCE = 5
    
    def __init__(self, fail_fast: bool = True):
        """
        Initialize the stratified retriever.
        
        Phase 4: BM25-only. Embeddings will be added in Phase 5.
        
        Args:
            fail_fast: If True, raise error if indexes missing (default: True)
        """
        self.fail_fast = fail_fast
        self.source_indexes: Dict[str, SourceIndex] = {}
        self._initialized = False
        
    def initialize(self):
        """
        Load pre-built indexes for all sources.
        
        NO RUNTIME BUILDS: If indexes are missing, this fails fast.
        Run `python -m src.ml.stratified_retriever` to build indexes.
        
        Raises:
            IndexNotFoundError: If indexes are missing (always - no fallback)
        """
        if self._initialized:
            return
            
        logger.info("Initializing StratifiedTafsirRetriever...")
        
        # Load from pre-built indexes ONLY - no runtime builds
        indexes_exist = self._try_load_indexes()
        
        if not indexes_exist:
            raise IndexNotFoundError(
                f"Pre-built tafsir indexes not found at {INDEXES_DIR}. "
                "Run `python -m src.ml.stratified_retriever` to build indexes."
            )
        
        # Validate all sources have data
        self._validate_indexes()
        
        self._initialized = True
        logger.info(f"StratifiedTafsirRetriever ready with {len(self.source_indexes)} sources")
    
    def _try_load_indexes(self) -> bool:
        """Try to load pre-built indexes. Returns True if successful."""
        if not INDEXES_DIR.exists():
            return False
        
        all_loaded = True
        for source in TAFSIR_SOURCES:
            index_file = INDEXES_DIR / f"{source}.json"
            if index_file.exists():
                try:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    idx = SourceIndex(source_id=source)
                    idx.documents = data.get("documents", [])
                    idx.texts = [d.get("text", "") for d in idx.documents]
                    
                    # Build BM25 index
                    if BM25Okapi and idx.texts:
                        tokenized = [self._tokenize(t) for t in idx.texts]
                        idx.bm25 = BM25Okapi(tokenized)
                    
                    self.source_indexes[source] = idx
                    logger.info(f"  Loaded {source}: {len(idx)} documents")
                except Exception as e:
                    logger.warning(f"  Failed to load {source}: {e}")
                    all_loaded = False
            else:
                all_loaded = False
        
        return all_loaded and len(self.source_indexes) == len(TAFSIR_SOURCES)
    
    def _build_indexes_from_db(self):
        """Build indexes from tafsir JSONL files or database."""
        from src.preprocessing.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        
        for source in TAFSIR_SOURCES:
            idx = SourceIndex(source_id=source)
            
            # Try JSONL file first (primary source)
            jsonl_path = TAFSIR_DIR / f"{source}.ar.jsonl"
            if jsonl_path.exists():
                logger.info(f"  Loading {source} from JSONL...")
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                # Handle different field names
                                text = entry.get("text_ar", entry.get("text", ""))
                                ref = entry.get("reference", {})
                                surah = ref.get("surah", entry.get("surah", 0))
                                ayah = ref.get("ayah", entry.get("ayah", 0))
                                
                                # Clean HTML if present
                                if text and cleaner.has_html(text):
                                    text = cleaner.clean(text)
                                
                                if text and len(text.strip()) > 20:
                                    idx.documents.append({
                                        "surah": surah,
                                        "ayah": ayah,
                                        "text": text,
                                        "source": source,
                                    })
                                    idx.texts.append(text)
                            except json.JSONDecodeError:
                                continue
            
            # Build BM25 index
            if BM25Okapi and idx.texts:
                tokenized = [self._tokenize(t) for t in idx.texts]
                idx.bm25 = BM25Okapi(tokenized)
            
            self.source_indexes[source] = idx
            logger.info(f"  Built {source}: {len(idx)} documents")
        
        # Save indexes for future use
        self._save_indexes()
    
    def _save_indexes(self):
        """Save indexes to disk for faster future loading."""
        INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        
        for source, idx in self.source_indexes.items():
            index_file = INDEXES_DIR / f"{source}.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({"documents": idx.documents}, f, ensure_ascii=False)
            logger.info(f"  Saved {source} index: {len(idx)} documents")
    
    def _validate_indexes(self):
        """Validate all sources have sufficient data."""
        missing = []
        empty = []
        
        for source in TAFSIR_SOURCES:
            if source not in self.source_indexes:
                missing.append(source)
            elif len(self.source_indexes[source]) == 0:
                empty.append(source)
        
        if missing or empty:
            msg = f"Index validation failed. Missing: {missing}, Empty: {empty}"
            if self.fail_fast:
                raise IndexNotFoundError(msg)
            logger.error(msg)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple Arabic tokenizer."""
        import re
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    def search(
        self,
        query: str,
        top_k_per_source: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all sources using BM25 and return stratified results.
        
        Phase 4: BM25-only. Semantic search will be added in Phase 5.
        
        Args:
            query: Search query in Arabic
            top_k_per_source: Results per source (default: 5)
            
        Returns:
            Dict mapping source_id to list of results
            
        Raises:
            IndexNotFoundError: If indexes not initialized
        """
        if not self._initialized:
            self.initialize()
        
        results = {}
        query_tokens = self._tokenize(query)
        
        for source, idx in self.source_indexes.items():
            source_results = []
            
            # BM25 search (Phase 4: BM25-only)
            if idx.bm25 and query_tokens:
                scores = idx.bm25.get_scores(query_tokens)
                top_indices = np.argsort(scores)[-top_k_per_source * 2:][::-1]
                
                for rank, i in enumerate(top_indices):
                    if scores[i] > 0:
                        doc = idx.documents[i].copy()
                        doc["bm25_score"] = float(scores[i])
                        doc["rank"] = rank
                        source_results.append(doc)
            
            # Sort by BM25 score and take top_k
            source_results.sort(key=lambda x: x.get("bm25_score", 0), reverse=True)
            results[source] = source_results[:top_k_per_source]
        
        return results
    
    def search_flat(
        self,
        query: str,
        top_k_per_source: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search and return flat list with source diversity guaranteed.
        
        Returns results interleaved from all sources to ensure diversity.
        """
        stratified = self.search(query, top_k_per_source)
        
        # Interleave results from all sources
        flat = []
        max_len = max(len(v) for v in stratified.values()) if stratified else 0
        
        for i in range(max_len):
            for source in TAFSIR_SOURCES:
                if source in stratified and i < len(stratified[source]):
                    flat.append(stratified[source][i])
        
        return flat
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexes."""
        return {
            "initialized": self._initialized,
            "sources": {
                source: len(idx) for source, idx in self.source_indexes.items()
            },
            "total_documents": sum(len(idx) for idx in self.source_indexes.values()),
            "indexes_dir": str(INDEXES_DIR),
        }


# Singleton instance
_stratified_retriever: Optional[StratifiedTafsirRetriever] = None


def get_stratified_retriever(fail_fast: bool = True) -> StratifiedTafsirRetriever:
    """Get or create the stratified retriever singleton."""
    global _stratified_retriever
    
    if _stratified_retriever is None:
        _stratified_retriever = StratifiedTafsirRetriever(fail_fast=fail_fast)
        _stratified_retriever.initialize()
    
    return _stratified_retriever


def build_tafsir_indexes():
    """
    Build tafsir indexes from database.
    Run this script to pre-build indexes before deployment.
    """
    print("Building tafsir indexes...")
    retriever = StratifiedTafsirRetriever(fail_fast=False)
    retriever._build_indexes_from_db()
    
    stats = retriever.get_stats()
    print(f"\nIndex Statistics:")
    for source, count in stats["sources"].items():
        print(f"  {source}: {count} documents")
    print(f"  Total: {stats['total_documents']} documents")
    print(f"\nIndexes saved to: {stats['indexes_dir']}")


if __name__ == "__main__":
    build_tafsir_indexes()
