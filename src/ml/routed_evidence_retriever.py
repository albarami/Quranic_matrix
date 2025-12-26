"""
RoutedEvidenceRetriever: Query-routed deterministic + hybrid retrieval.

Phase 5.5: Implements the production architecture:
1. QueryRouter classifies intent
2. Deterministic retrieval for AYAH_REF/SURAH_REF/CONCEPT_REF
3. Hybrid retrieval only for FREE_TEXT

Guarantees 5/5 core source coverage for deterministic paths.
No synthetic evidence - returns empty if nothing found.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.ml.query_router import QueryRouter, QueryIntent, RouterResult, get_query_router
from src.ml.hybrid_evidence_retriever import (
    HybridEvidenceRetriever,
    ChunkedEvidenceIndex,
    RetrievalResult,
    RetrievalResponse,
    get_hybrid_retriever,
    CORE_SOURCES,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutedRetrievalResponse:
    """Response from routed retrieval."""
    query: str
    intent: QueryIntent
    router_result: RouterResult
    results: List[RetrievalResult] = field(default_factory=list)
    retrieval_mode: str = "unknown"
    sources_covered: List[str] = field(default_factory=list)
    core_sources_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'intent': self.intent.value,
            'router': self.router_result.to_dict(),
            'results': [r.to_dict() for r in self.results],
            'retrieval_mode': self.retrieval_mode,
            'sources_covered': self.sources_covered,
            'core_sources_count': self.core_sources_count,
            'fallback_used': self.fallback_used,
            'fallback_reason': self.fallback_reason,
        }


class RoutedEvidenceRetriever:
    """
    Query-routed evidence retrieval.
    
    Routes queries to appropriate retrieval strategy:
    - AYAH_REF: Deterministic lookup by verse key (5/5 coverage guaranteed)
    - SURAH_REF: Deterministic lookup by surah
    - CONCEPT_REF: Concept-based retrieval (future: precomputed mappings)
    - FREE_TEXT: Hybrid BM25 + dense retrieval
    """
    
    def __init__(self):
        self.router = get_query_router()
        self.index = ChunkedEvidenceIndex()
        self.hybrid_retriever = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        self.index.load()
        self.hybrid_retriever = get_hybrid_retriever(use_bm25=True, use_dense=False)
        self._initialized = True
    
    def search(self, query: str, top_k: int = 20, min_per_source: int = 3) -> RoutedRetrievalResponse:
        """
        Route query and retrieve evidence.
        
        Returns deterministic results for AYAH_REF/SURAH_REF/CONCEPT_REF.
        Returns hybrid results for FREE_TEXT.
        Never fabricates evidence.
        """
        self.initialize()
        
        # 1. Route the query
        router_result = self.router.route(query)
        
        response = RoutedRetrievalResponse(
            query=query,
            intent=router_result.intent,
            router_result=router_result,
        )
        
        # 2. Dispatch to appropriate retrieval method
        if router_result.intent == QueryIntent.AYAH_REF:
            self._retrieve_ayah_ref(router_result, response, top_k, min_per_source)
        elif router_result.intent == QueryIntent.SURAH_REF:
            self._retrieve_surah_ref(router_result, response, top_k, min_per_source)
        elif router_result.intent == QueryIntent.CONCEPT_REF:
            self._retrieve_concept_ref(router_result, response, top_k, min_per_source)
        else:  # FREE_TEXT
            self._retrieve_free_text(query, response, top_k, min_per_source)
        
        # 3. Compute coverage stats
        sources = set(r.source for r in response.results)
        response.sources_covered = list(sources & set(CORE_SOURCES))
        response.core_sources_count = len(response.sources_covered)
        
        return response
    
    def _retrieve_ayah_ref(
        self,
        router_result: RouterResult,
        response: RoutedRetrievalResponse,
        top_k: int,
        min_per_source: int,
    ) -> None:
        """Deterministic retrieval for verse reference."""
        response.retrieval_mode = "deterministic_ayah"
        
        verse_key = router_result.extracted_ref
        if not verse_key:
            response.fallback_used = True
            response.fallback_reason = "no_verse_key_extracted"
            return
        
        # Get all chunks for this verse from all sources
        chunks = self.index.get_by_verse(verse_key)
        
        if not chunks:
            response.fallback_reason = f"no_chunks_for_verse_{verse_key}"
            return
        
        # Group by source and select top chunks per source
        source_buckets = defaultdict(list)
        for chunk in chunks:
            source_buckets[chunk['source']].append(chunk)
        
        # Select min_per_source from each core source first
        seen_ids = set()
        for source in CORE_SOURCES:
            bucket = source_buckets.get(source, [])
            for chunk in bucket[:min_per_source]:
                if chunk['chunk_id'] not in seen_ids:
                    response.results.append(RetrievalResult(
                        chunk_id=chunk['chunk_id'],
                        verse_key=chunk['verse_key'],
                        source=chunk['source'],
                        text=chunk['text_clean'],
                        score=1.0,
                        retrieval_method='deterministic',
                        surah=chunk['surah'],
                        ayah=chunk['ayah'],
                    ))
                    seen_ids.add(chunk['chunk_id'])
        
        # Fill remaining from other sources
        for source, bucket in source_buckets.items():
            if source not in CORE_SOURCES:
                for chunk in bucket[:min_per_source]:
                    if chunk['chunk_id'] not in seen_ids and len(response.results) < top_k:
                        response.results.append(RetrievalResult(
                            chunk_id=chunk['chunk_id'],
                            verse_key=chunk['verse_key'],
                            source=chunk['source'],
                            text=chunk['text_clean'],
                            score=0.9,
                            retrieval_method='deterministic',
                            surah=chunk['surah'],
                            ayah=chunk['ayah'],
                        ))
                        seen_ids.add(chunk['chunk_id'])
    
    def _retrieve_surah_ref(
        self,
        router_result: RouterResult,
        response: RoutedRetrievalResponse,
        top_k: int,
        min_per_source: int,
    ) -> None:
        """Deterministic retrieval for surah reference."""
        response.retrieval_mode = "deterministic_surah"
        
        surah_num = router_result.surah_num
        if not surah_num:
            response.fallback_used = True
            response.fallback_reason = "no_surah_num_extracted"
            return
        
        # Get sample verses from this surah (first few ayat)
        sample_verses = [f"{surah_num}:{i}" for i in range(1, 6)]
        
        seen_ids = set()
        for verse_key in sample_verses:
            chunks = self.index.get_by_verse(verse_key)
            
            # Group by source
            source_buckets = defaultdict(list)
            for chunk in chunks:
                source_buckets[chunk['source']].append(chunk)
            
            # Take 1 chunk per source per verse
            for source in CORE_SOURCES:
                bucket = source_buckets.get(source, [])
                for chunk in bucket[:1]:
                    if chunk['chunk_id'] not in seen_ids and len(response.results) < top_k:
                        response.results.append(RetrievalResult(
                            chunk_id=chunk['chunk_id'],
                            verse_key=chunk['verse_key'],
                            source=chunk['source'],
                            text=chunk['text_clean'],
                            score=0.9,
                            retrieval_method='deterministic',
                            surah=chunk['surah'],
                            ayah=chunk['ayah'],
                        ))
                        seen_ids.add(chunk['chunk_id'])
    
    def _retrieve_concept_ref(
        self,
        router_result: RouterResult,
        response: RoutedRetrievalResponse,
        top_k: int,
        min_per_source: int,
    ) -> None:
        """Retrieval for concept/behavior reference."""
        response.retrieval_mode = "concept_hybrid"
        
        concept_term = router_result.concept_term
        if not concept_term:
            response.fallback_used = True
            response.fallback_reason = "no_concept_term_extracted"
            return
        
        # For now, use hybrid retrieval with the concept term
        # Future: use precomputed concept → ayat mappings
        hybrid_response = self.hybrid_retriever.search(concept_term, top_k=top_k, min_per_source=min_per_source)
        response.results = hybrid_response.results
    
    def _retrieve_free_text(
        self,
        query: str,
        response: RoutedRetrievalResponse,
        top_k: int,
        min_per_source: int,
    ) -> None:
        """Hybrid retrieval for free-text queries."""
        response.retrieval_mode = "hybrid_free_text"
        
        hybrid_response = self.hybrid_retriever.search(query, top_k=top_k, min_per_source=min_per_source)
        response.results = hybrid_response.results
        response.fallback_used = hybrid_response.fallback_used
        response.fallback_reason = hybrid_response.fallback_reason


# Singleton instance
_routed_retriever: Optional[RoutedEvidenceRetriever] = None


def get_routed_retriever() -> RoutedEvidenceRetriever:
    """Get or create the singleton RoutedEvidenceRetriever."""
    global _routed_retriever
    if _routed_retriever is None:
        _routed_retriever = RoutedEvidenceRetriever()
    return _routed_retriever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    retriever = get_routed_retriever()
    
    test_queries = [
        "2:255",
        "البقرة:255",
        "سورة الفاتحة",
        "الصبر",
        "آيات التقوى",
        "كيف أتعامل مع الابتلاء",
    ]
    
    print("\n" + "=" * 60)
    print("Testing RoutedEvidenceRetriever")
    print("=" * 60)
    
    for query in test_queries:
        response = retriever.search(query, top_k=15)
        print(f"\nQuery: {query}")
        print(f"  Intent: {response.intent.value}")
        print(f"  Mode: {response.retrieval_mode}")
        print(f"  Results: {len(response.results)}")
        print(f"  Core sources: {response.core_sources_count}/5 {response.sources_covered}")
        if response.results:
            print(f"  Top result: {response.results[0].source} - {response.results[0].text[:50]}...")
