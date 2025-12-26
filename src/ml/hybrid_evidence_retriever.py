"""
Hybrid Evidence Retriever

Step 3: Implements hybrid retrieval over chunked tafsir corpus.

Retrieval pipeline:
1. Deterministic: Direct lookup by verse reference
2. BM25: Lexical search over chunks
3. Dense: Embedding-based search (LaBSE/BGE-M3)
4. Fusion: RRF (Reciprocal Rank Fusion)
5. Rerank: Optional cross-encoder reranking

No synthetic evidence - returns empty if no results found.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

# Paths
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")

# Core sources
CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    chunk_id: str
    verse_key: str
    source: str
    text: str
    score: float
    retrieval_method: str
    surah: int = 0
    ayah: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'verse_key': self.verse_key,
            'source': self.source,
            'text': self.text[:500],
            'score': round(self.score, 4),
            'retrieval_method': self.retrieval_method,
            'surah': self.surah,
            'ayah': self.ayah,
        }


@dataclass
class SurahCoverage:
    """Coverage summary for SURAH_REF responses."""
    surah_num: int
    verse_count: int
    verses_covered: List[str]
    sources_per_verse: Dict[str, List[str]]
    total_chunks: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'surah_num': self.surah_num,
            'verse_count': self.verse_count,
            'verses_covered': self.verses_covered,
            'sources_per_verse': self.sources_per_verse,
            'total_chunks': self.total_chunks,
        }


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    deterministic_count: int = 0
    bm25_count: int = 0
    dense_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    
    # SURAH_REF specific fields
    intent: str = "FREE_TEXT"  # SURAH_REF, AYAH_REF, CONCEPT_REF, FREE_TEXT
    surah_coverage: Optional[SurahCoverage] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'query': self.query,
            'intent': self.intent,
            'results': [r.to_dict() for r in self.results],
            'counts': {
                'deterministic': self.deterministic_count,
                'bm25': self.bm25_count,
                'dense': self.dense_count,
                'total': len(self.results),
            },
            'fallback_used': self.fallback_used,
            'fallback_reason': self.fallback_reason,
        }
        if self.surah_coverage:
            result['surah_coverage'] = self.surah_coverage.to_dict()
        return result
    
    def get_summary_view(self, top_per_verse_source: int = 1) -> Dict[str, Any]:
        """
        Get a summarized view: top N chunks per verse per source.
        
        For SURAH_REF, this provides a readable summary while preserving
        full data access via the complete results list.
        """
        if self.intent != "SURAH_REF":
            return self.to_dict()
        
        # Group by verse_key -> source -> chunks
        by_verse_source: Dict[str, Dict[str, List[RetrievalResult]]] = {}
        for r in self.results:
            if r.verse_key not in by_verse_source:
                by_verse_source[r.verse_key] = {}
            if r.source not in by_verse_source[r.verse_key]:
                by_verse_source[r.verse_key][r.source] = []
            by_verse_source[r.verse_key][r.source].append(r)
        
        # Build summary: top N per verse per source
        summary_results = []
        for verse_key in sorted(by_verse_source.keys(), key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1]))):
            for source in CORE_SOURCES:
                chunks = by_verse_source.get(verse_key, {}).get(source, [])
                for chunk in chunks[:top_per_verse_source]:
                    summary_results.append(chunk.to_dict())
        
        return {
            'query': self.query,
            'intent': self.intent,
            'view': 'summary',
            'top_per_verse_source': top_per_verse_source,
            'summary_results': summary_results,
            'total_available': len(self.results),
            'surah_coverage': self.surah_coverage.to_dict() if self.surah_coverage else None,
        }


class ChunkedEvidenceIndex:
    """In-memory index of chunked evidence."""
    
    def __init__(self, index_file: Path = CHUNKED_INDEX_FILE):
        self.index_file = index_file
        self.chunks: List[Dict[str, Any]] = []
        self.by_verse: Dict[str, List[int]] = defaultdict(list)
        self.by_source: Dict[str, List[int]] = defaultdict(list)
        self.by_chunk_id: Dict[str, int] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load the chunked index into memory."""
        if self._loaded:
            return
        
        if not self.index_file.exists():
            raise FileNotFoundError(f"Chunked index not found: {self.index_file}")
        
        logger.info(f"Loading chunked index from {self.index_file}...")
        
        with open(self.index_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                chunk = json.loads(line)
                self.chunks.append(chunk)
                
                verse_key = chunk['verse_key']
                source = chunk['source']
                chunk_id = chunk['chunk_id']
                
                self.by_verse[verse_key].append(i)
                self.by_source[source].append(i)
                self.by_chunk_id[chunk_id] = i
        
        self._loaded = True
        logger.info(f"Loaded {len(self.chunks)} chunks")
    
    def get_by_verse(self, verse_key: str) -> List[Dict[str, Any]]:
        """Get all chunks for a verse."""
        self.load()
        indices = self.by_verse.get(verse_key, [])
        return [self.chunks[i] for i in indices]
    
    def get_by_verse_and_source(self, verse_key: str, source: str) -> List[Dict[str, Any]]:
        """Get chunks for a specific verse and source."""
        self.load()
        verse_indices = set(self.by_verse.get(verse_key, []))
        source_indices = set(self.by_source.get(source, []))
        common = verse_indices & source_indices
        return [self.chunks[i] for i in common]
    
    def get_all_texts(self) -> List[str]:
        """Get all chunk texts for BM25 indexing."""
        self.load()
        return [c['text_clean'] for c in self.chunks]


def normalize_arabic_query(text: str) -> str:
    """
    Normalize Arabic text for better BM25 matching.
    
    Phase 5.5.3: Query normalization + lexical expansion.
    """
    if not text:
        return text
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Normalize Alef forms (أ إ آ ا → ا)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    
    # Normalize Yaa forms (ى → ي)
    text = re.sub(r'ى', 'ي', text)
    
    # Normalize Taa Marbuta (ة → ه)
    text = re.sub(r'ة', 'ه', text)
    
    # Remove tatweel (ـ)
    text = re.sub(r'ـ', '', text)
    
    # Normalize Hamza forms
    text = re.sub(r'[ؤئء]', 'ء', text)
    
    return text


class BM25Retriever:
    """BM25 lexical retrieval over chunks."""
    
    def __init__(self, index: ChunkedEvidenceIndex):
        self.index = index
        self.bm25 = None
        self._built = False
    
    def build(self) -> None:
        """Build BM25 index."""
        if self._built:
            return
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 retrieval disabled")
            return
        
        self.index.load()
        
        logger.info("Building BM25 index...")
        
        # Tokenize Arabic text with normalization
        texts = self.index.get_all_texts()
        tokenized = [self._tokenize(t) for t in texts]
        
        self.bm25 = BM25Okapi(tokenized)
        self._built = True
        
        logger.info(f"BM25 index built with {len(texts)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Arabic tokenization with normalization."""
        # Normalize text first
        text = normalize_arabic_query(text)
        # Split on whitespace and punctuation
        tokens = re.findall(r'[\u0600-\u06FF]+', text)
        return tokens
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search and return (index, score) pairs."""
        if not self._built or self.bm25 is None:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class DenseRetriever:
    """Dense embedding-based retrieval."""
    
    def __init__(self, index: ChunkedEvidenceIndex, model_name: str = "sentence-transformers/LaBSE"):
        self.index = index
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self._built = False
    
    def build(self) -> None:
        """Build dense embeddings for all chunks."""
        if self._built:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence_transformers not installed, dense retrieval disabled")
            return
        
        self.index.load()
        
        logger.info(f"Loading model {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        logger.info("Building dense embeddings (this may take a while)...")
        texts = self.index.get_all_texts()
        
        # Encode in batches
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        self._built = True
        logger.info(f"Dense index built with {len(texts)} embeddings")
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search and return (index, score) pairs."""
        if not self._built or self.model is None:
            return []
        
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        
        # Compute cosine similarities
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(i), float(scores[i])) for i in top_indices]


class HybridEvidenceRetriever:
    """
    Hybrid retrieval combining deterministic, BM25, and dense methods.
    
    No synthetic evidence - returns empty results if nothing found.
    """
    
    def __init__(
        self,
        index_file: Path = CHUNKED_INDEX_FILE,
        dense_model: str = "sentence-transformers/LaBSE",
        use_bm25: bool = True,
        use_dense: bool = True,
    ):
        self.index = ChunkedEvidenceIndex(index_file)
        self.bm25 = BM25Retriever(self.index) if use_bm25 else None
        self.dense = DenseRetriever(self.index, dense_model) if use_dense else None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all retrieval components."""
        if self._initialized:
            return
        
        self.index.load()
        
        if self.bm25:
            self.bm25.build()
        
        if self.dense:
            self.dense.build()
        
        self._initialized = True
    
    # Surah verse counts for SURAH_REF deterministic retrieval
    SURAH_VERSE_COUNTS = {
        1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
        11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
        21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
        31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
        41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
        51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
        61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
        71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
        81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
        91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
        101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
        111: 5, 112: 4, 113: 5, 114: 6,
    }
    
    # Arabic surah names for parsing
    SURAH_NAMES = {
        'الفاتحة': 1, 'البقرة': 2, 'آل عمران': 3, 'النساء': 4, 'المائدة': 5,
        'الأنعام': 6, 'الأعراف': 7, 'الأنفال': 8, 'التوبة': 9, 'يونس': 10,
        'هود': 11, 'يوسف': 12, 'الرعد': 13, 'إبراهيم': 14, 'الحجر': 15,
        'النحل': 16, 'الإسراء': 17, 'الكهف': 18, 'مريم': 19, 'طه': 20,
        'الأنبياء': 21, 'الحج': 22, 'المؤمنون': 23, 'النور': 24, 'الفرقان': 25,
        'الشعراء': 26, 'النمل': 27, 'القصص': 28, 'العنكبوت': 29, 'الروم': 30,
        'لقمان': 31, 'السجدة': 32, 'الأحزاب': 33, 'سبأ': 34, 'فاطر': 35,
        'يس': 36, 'الصافات': 37, 'ص': 38, 'الزمر': 39, 'غافر': 40,
        'فصلت': 41, 'الشورى': 42, 'الزخرف': 43, 'الدخان': 44, 'الجاثية': 45,
        'الأحقاف': 46, 'محمد': 47, 'الفتح': 48, 'الحجرات': 49, 'ق': 50,
        'الذاريات': 51, 'الطور': 52, 'النجم': 53, 'القمر': 54, 'الرحمن': 55,
        'الواقعة': 56, 'الحديد': 57, 'المجادلة': 58, 'الحشر': 59, 'الممتحنة': 60,
        'الصف': 61, 'الجمعة': 62, 'المنافقون': 63, 'التغابن': 64, 'الطلاق': 65,
        'التحريم': 66, 'الملك': 67, 'القلم': 68, 'الحاقة': 69, 'المعارج': 70,
        'نوح': 71, 'الجن': 72, 'المزمل': 73, 'المدثر': 74, 'القيامة': 75,
        'الإنسان': 76, 'المرسلات': 77, 'النبأ': 78, 'النازعات': 79, 'عبس': 80,
        'التكوير': 81, 'الانفطار': 82, 'المطففين': 83, 'الانشقاق': 84, 'البروج': 85,
        'الطارق': 86, 'الأعلى': 87, 'الغاشية': 88, 'الفجر': 89, 'البلد': 90,
        'الشمس': 91, 'الليل': 92, 'الضحى': 93, 'الشرح': 94, 'التين': 95,
        'العلق': 96, 'القدر': 97, 'البينة': 98, 'الزلزلة': 99, 'العاديات': 100,
        'القارعة': 101, 'التكاثر': 102, 'العصر': 103, 'الهمزة': 104, 'الفيل': 105,
        'قريش': 106, 'الماعون': 107, 'الكوثر': 108, 'الكافرون': 109, 'النصر': 110,
        'المسد': 111, 'الإخلاص': 112, 'الفلق': 113, 'الناس': 114,
    }
    
    def _parse_surah_reference(self, query: str) -> Optional[int]:
        """Extract surah number from query (SURAH_REF intent)."""
        # Check for "سورة X" pattern (highest priority)
        match = re.search(r'سورة\s+([\u0600-\u06FF\s]+)', query)
        if match:
            surah_name = match.group(1).strip()
            # Sort by name length descending to match longer names first
            # This prevents "ص" from matching before "الإخلاص"
            sorted_names = sorted(self.SURAH_NAMES.items(), key=lambda x: len(x[0]), reverse=True)
            for name, num in sorted_names:
                if name == surah_name or name in surah_name:
                    return num
        
        # Check for word-boundary surah name match (not substring)
        # Sort by length to match longer names first
        sorted_names = sorted(self.SURAH_NAMES.items(), key=lambda x: len(x[0]), reverse=True)
        for name, num in sorted_names:
            # Use word boundary check - name must be a complete word
            # For single-char names like "ص" or "ق", require "سورة" prefix
            if len(name) <= 2:
                if f"سورة {name}" in query:
                    return num
            else:
                # For longer names, check if it appears as a word
                pattern = rf'\b{re.escape(name)}\b|{re.escape(name)}'
                if re.search(pattern, query):
                    return num
        
        return None
    
    def _parse_verse_reference(self, query: str) -> Optional[str]:
        """Extract verse reference from query (e.g., '2:255' or 'البقرة:255')."""
        # Numeric format: 2:255
        match = re.search(r'(\d+):(\d+)', query)
        if match:
            return f"{match.group(1)}:{match.group(2)}"
        
        for name, num in self.SURAH_NAMES.items():
            if name in query:
                ayah_match = re.search(r'(\d+)', query)
                if ayah_match:
                    return f"{num}:{ayah_match.group(1)}"
        
        return None
    
    def _deterministic_retrieval(self, query: str) -> List[RetrievalResult]:
        """Direct lookup by verse reference (AYAH_REF)."""
        verse_key = self._parse_verse_reference(query)
        if not verse_key:
            return []
        
        results = []
        chunks = self.index.get_by_verse(verse_key)
        
        for chunk in chunks:
            results.append(RetrievalResult(
                chunk_id=chunk['chunk_id'],
                verse_key=chunk['verse_key'],
                source=chunk['source'],
                text=chunk['text_clean'],
                score=1.0,  # Perfect match
                retrieval_method='deterministic',
                surah=chunk['surah'],
                ayah=chunk['ayah'],
            ))
        
        return results
    
    def _surah_deterministic_retrieval(self, surah_num: int) -> List[RetrievalResult]:
        """
        Deterministic retrieval for SURAH_REF intent.
        
        Fetches tafsir chunks ONLY for verses in the specified surah.
        No BM25 pollution - this is the primary evidence set.
        """
        results = []
        verse_count = self.SURAH_VERSE_COUNTS.get(surah_num, 0)
        
        if verse_count == 0:
            logger.warning(f"Unknown surah number: {surah_num}")
            return []
        
        logger.info(f"[SURAH_REF] Fetching tafsir for surah {surah_num} ({verse_count} verses)")
        
        # Fetch chunks for each verse in the surah
        for ayah in range(1, verse_count + 1):
            verse_key = f"{surah_num}:{ayah}"
            chunks = self.index.get_by_verse(verse_key)
            
            for chunk in chunks:
                if chunk['source'] in CORE_SOURCES:
                    results.append(RetrievalResult(
                        chunk_id=chunk['chunk_id'],
                        verse_key=chunk['verse_key'],
                        source=chunk['source'],
                        text=chunk['text_clean'],
                        score=1.0,  # Perfect match - deterministic
                        retrieval_method='surah_deterministic',
                        surah=chunk['surah'],
                        ayah=chunk['ayah'],
                    ))
        
        logger.info(f"[SURAH_REF] Retrieved {len(results)} chunks for surah {surah_num}")
        return results
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion of BM25 and dense results."""
        scores = defaultdict(float)
        
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += 1.0 / (k + rank + 1)
        
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] += 1.0 / (k + rank + 1)
        
        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: -x[1])
        return fused
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_per_source: int = 3,
        surah_num: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Hybrid search with source-aware bucketed selection.
        
        Phase 5.5: Ensures all 5 core sources are represented in top-K.
        Phase 6.3: SURAH_REF intent uses deterministic verse filtering.
        Returns empty results if nothing found - NO SYNTHETIC EVIDENCE.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            min_per_source: Minimum results per core source
            surah_num: If provided, enforces SURAH_REF deterministic retrieval
        """
        self.initialize()
        
        response = RetrievalResponse(query=query)
        seen_chunk_ids = set()
        
        # 0. SURAH_REF: If surah_num provided, use deterministic surah retrieval
        if surah_num is None:
            # Auto-detect SURAH_REF from query
            surah_num = self._parse_surah_reference(query)
        
        if surah_num is not None:
            # SURAH_REF intent: Deterministic retrieval for ALL verses in surah
            response.intent = "SURAH_REF"
            surah_results = self._surah_deterministic_retrieval(surah_num)
            for r in surah_results:
                if r.chunk_id not in seen_chunk_ids:
                    response.results.append(r)
                    seen_chunk_ids.add(r.chunk_id)
                    response.deterministic_count += 1
            
            # For SURAH_REF: Return ALL results, no top_k limit
            # This ensures complete coverage of all verses in the surah
            if response.deterministic_count > 0:
                # Build coverage summary
                verses_covered = set()
                sources_per_verse: Dict[str, List[str]] = {}
                for r in response.results:
                    verses_covered.add(r.verse_key)
                    if r.verse_key not in sources_per_verse:
                        sources_per_verse[r.verse_key] = []
                    if r.source not in sources_per_verse[r.verse_key]:
                        sources_per_verse[r.verse_key].append(r.source)
                
                response.surah_coverage = SurahCoverage(
                    surah_num=surah_num,
                    verse_count=self.SURAH_VERSE_COUNTS.get(surah_num, 0),
                    verses_covered=sorted(verses_covered, key=lambda x: int(x.split(':')[1])),
                    sources_per_verse=sources_per_verse,
                    total_chunks=len(response.results),
                )
                
                logger.info(f"[SURAH_REF] Returning {len(response.results)} results for surah {surah_num} "
                           f"({len(verses_covered)} verses covered)")
                return response
        
        # 1. Deterministic retrieval (verse reference) - AYAH_REF
        det_results = self._deterministic_retrieval(query)
        for r in det_results:
            if r.chunk_id not in seen_chunk_ids:
                response.results.append(r)
                seen_chunk_ids.add(r.chunk_id)
                response.deterministic_count += 1
        
        # If deterministic found results, we're done (perfect coverage)
        if response.deterministic_count > 0:
            response.intent = "AYAH_REF"
            # Still apply source diversity
            response.results = self._apply_source_diversity(response.results, top_k, min_per_source)
            return response
        
        # 2. BM25 retrieval - get large candidate pool
        bm25_candidates = []
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k=200)  # Large pool
            for idx, score in bm25_results:
                chunk = self.index.chunks[idx]
                bm25_candidates.append(RetrievalResult(
                    chunk_id=chunk['chunk_id'],
                    verse_key=chunk['verse_key'],
                    source=chunk['source'],
                    text=chunk['text_clean'],
                    score=score,
                    retrieval_method='bm25',
                    surah=chunk['surah'],
                    ayah=chunk['ayah'],
                ))
        
        # 3. Phase 5.5: Verse-first retrieval strategy
        # Find the most likely verse(s) from BM25 results, then get all sources for those verses
        verse_scores = defaultdict(float)
        for r in bm25_candidates:
            verse_scores[r.verse_key] += r.score
        
        # Get top candidate verses
        top_verses = sorted(verse_scores.items(), key=lambda x: -x[1])[:5]
        
        # For each top verse, get all chunks from all core sources
        verse_based_results = []
        for verse_key, verse_score in top_verses:
            chunks = self.index.get_by_verse(verse_key)
            for chunk in chunks:
                if chunk['source'] in CORE_SOURCES:
                    verse_based_results.append(RetrievalResult(
                        chunk_id=chunk['chunk_id'],
                        verse_key=chunk['verse_key'],
                        source=chunk['source'],
                        text=chunk['text_clean'],
                        score=verse_score,  # Use verse-level score
                        retrieval_method='verse_first',
                        surah=chunk['surah'],
                        ayah=chunk['ayah'],
                    ))
        
        # 4. Source-aware bucketed selection from verse-based results
        source_buckets = {s: [] for s in CORE_SOURCES}
        
        for r in verse_based_results:
            if r.source in CORE_SOURCES:
                source_buckets[r.source].append(r)
        
        # 5. Select min_per_source from each bucket first
        final_results = []
        for source in CORE_SOURCES:
            bucket = source_buckets[source]
            for r in bucket[:min_per_source]:
                if r.chunk_id not in seen_chunk_ids:
                    final_results.append(r)
                    seen_chunk_ids.add(r.chunk_id)
                    response.bm25_count += 1
        
        # 6. Fill remaining slots from BM25 candidates (fallback)
        remaining_slots = top_k - len(final_results)
        if remaining_slots > 0:
            for r in bm25_candidates:
                if r.chunk_id not in seen_chunk_ids and len(final_results) < top_k:
                    final_results.append(r)
                    seen_chunk_ids.add(r.chunk_id)
                    response.bm25_count += 1
        
        response.results = final_results
        
        # Log if no results found (but do NOT fabricate)
        if len(response.results) == 0:
            logger.warning(f"[HYBRID] No results found for query: {query[:50]}...")
            response.fallback_used = False
            response.fallback_reason = "no_results_found"
        
        return response
    
    def _apply_source_diversity(
        self,
        results: List[RetrievalResult],
        top_k: int,
        min_per_source: int,
    ) -> List[RetrievalResult]:
        """Apply source diversity to deterministic results."""
        source_counts = defaultdict(int)
        diverse_results = []
        
        # First pass: ensure min_per_source from each core source
        for r in results:
            if r.source in CORE_SOURCES:
                if source_counts[r.source] < min_per_source:
                    diverse_results.append(r)
                    source_counts[r.source] += 1
        
        # Second pass: fill remaining slots
        for r in results:
            if r not in diverse_results and len(diverse_results) < top_k:
                diverse_results.append(r)
        
        return diverse_results[:top_k]
    
    def _rerank_bucket_with_dense(
        self,
        query: str,
        bucket: List[RetrievalResult],
        top_n: int,
    ) -> List[RetrievalResult]:
        """Rerank a source bucket using dense embeddings."""
        if not bucket or not self.dense or not self.dense._built:
            return bucket
        
        # Get query embedding
        query_emb = self.dense.model.encode(query, normalize_embeddings=True)
        
        # Score each chunk in bucket
        scored = []
        for r in bucket:
            # Get chunk embedding from index
            chunk_idx = self.index.by_chunk_id.get(r.chunk_id)
            if chunk_idx is not None and chunk_idx < len(self.dense.embeddings):
                chunk_emb = self.dense.embeddings[chunk_idx]
                score = float(np.dot(query_emb, chunk_emb))
                scored.append((r, score))
            else:
                scored.append((r, r.score))  # Keep original score
        
        # Sort by dense score
        scored.sort(key=lambda x: -x[1])
        
        # Update scores and return
        reranked = []
        for r, new_score in scored[:top_n]:
            r.score = new_score
            r.retrieval_method = 'bm25+dense_rerank'
            reranked.append(r)
        
        return reranked


def get_hybrid_retriever(
    dense_model: str = "sentence-transformers/LaBSE",
    use_bm25: bool = True,
    use_dense: bool = False,  # Disabled by default for faster init
) -> HybridEvidenceRetriever:
    """Factory function to get a configured hybrid retriever."""
    return HybridEvidenceRetriever(
        dense_model=dense_model,
        use_bm25=use_bm25,
        use_dense=use_dense,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the retriever
    retriever = get_hybrid_retriever(use_bm25=True, use_dense=False)
    
    # Test deterministic retrieval
    print("\n" + "=" * 60)
    print("Testing deterministic retrieval (verse reference)")
    print("=" * 60)
    
    response = retriever.search("2:255")
    print(f"Query: 2:255")
    print(f"Results: {len(response.results)}")
    print(f"Deterministic: {response.deterministic_count}")
    for r in response.results[:5]:
        print(f"  - {r.source}: {r.text[:50]}...")
    
    # Test BM25 retrieval
    print("\n" + "=" * 60)
    print("Testing BM25 retrieval (keyword search)")
    print("=" * 60)
    
    response = retriever.search("الصبر على البلاء")
    print(f"Query: الصبر على البلاء")
    print(f"Results: {len(response.results)}")
    print(f"BM25: {response.bm25_count}")
    for r in response.results[:5]:
        print(f"  - {r.source} ({r.verse_key}): {r.text[:50]}...")
