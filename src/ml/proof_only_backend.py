"""
Lightweight Proof-Only Backend - Phase 9.10E

This module provides a minimal proof pipeline that does NOT initialize:
- GPU embedding pipeline
- Cross-encoder reranker
- Vector index (FAISS)
- FullPower system

It uses only:
- evidence_index_v2_chunked.jsonl (deterministic verse-key lookup)
- semantic_graph_v2.json (JSON graph)
- concept_index_v2.jsonl (concept-to-verse mapping)

Target: <5 seconds for structured intent queries (SURAH_REF, AYAH_REF, CONCEPT_REF)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Core tafsir sources (7 sources)
CORE_TAFSIR_SOURCES = [
    "ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"
]


@dataclass
class LightweightProofDebug:
    """Debug info for lightweight proof-only backend."""
    intent: str = "FREE_TEXT"
    retrieval_mode: str = "lightweight_chunked"
    fullpower_used: bool = False  # Always False for this backend
    index_source: str = "json_chunked"
    core_sources_count: int = 7
    sources_covered: List[str] = field(default_factory=list)
    tafsir_fallbacks: Dict[str, bool] = field(default_factory=dict)
    primary_path_latency_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "retrieval_mode": self.retrieval_mode,
            "fullpower_used": self.fullpower_used,
            "index_source": self.index_source,
            "core_sources_count": self.core_sources_count,
            "sources_covered": self.sources_covered,
            "tafsir_fallbacks": self.tafsir_fallbacks,
            "primary_path_latency_ms": self.primary_path_latency_ms,
        }


class LightweightProofBackend:
    """
    Minimal proof backend for Tier-A tests.
    
    Does NOT initialize GPU components. Uses only JSON files.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._evidence_index: Optional[Dict[str, List[Dict]]] = None
        self._semantic_graph: Optional[Dict] = None
        self._concept_index: Optional[Dict] = None
        self._quran_data: Optional[Dict] = None
        
    def _load_evidence_index(self) -> Dict[str, List[Dict]]:
        """Load chunked evidence index (verse_key -> chunks)."""
        if self._evidence_index is not None:
            return self._evidence_index
            
        index_path = self.data_dir / "evidence" / "evidence_index_v2_chunked.jsonl"
        if not index_path.exists():
            logging.warning(f"Evidence index not found: {index_path}")
            return {}
            
        self._evidence_index = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                verse_key = entry.get("verse_key", "")
                if verse_key:
                    if verse_key not in self._evidence_index:
                        self._evidence_index[verse_key] = []
                    self._evidence_index[verse_key].append(entry)
        
        logging.info(f"[LightweightProof] Loaded {len(self._evidence_index)} verse keys from chunked index")
        return self._evidence_index
    
    def _load_semantic_graph(self) -> Dict:
        """Load semantic graph."""
        if self._semantic_graph is not None:
            return self._semantic_graph
            
        graph_path = self.data_dir / "graph" / "semantic_graph_v2.json"
        if not graph_path.exists():
            logging.warning(f"Semantic graph not found: {graph_path}")
            return {"nodes": [], "edges": []}
            
        with open(graph_path, "r", encoding="utf-8") as f:
            self._semantic_graph = json.load(f)
        
        logging.info(f"[LightweightProof] Loaded semantic graph with {len(self._semantic_graph.get('nodes', []))} nodes")
        return self._semantic_graph
    
    def _load_quran_data(self) -> Dict:
        """Load Quran text data from XML or build from evidence index."""
        if self._quran_data is not None:
            return self._quran_data
        
        # Try to load from evidence index (extract unique verse texts)
        # This is a fallback since we have verse_key in the evidence index
        self._quran_data = {"surahs": {}}
        
        # Build verse data from evidence index
        evidence_index = self._load_evidence_index()
        for verse_key, chunks in evidence_index.items():
            if ":" in verse_key:
                surah_str, ayah_str = verse_key.split(":", 1)
                surah_num = surah_str
                if surah_num not in self._quran_data["surahs"]:
                    self._quran_data["surahs"][surah_num] = {"verses": {}}
                # We don't have verse text in evidence index, but we have the verse_key
                # For proof_only mode, we just need to know the verse exists
                self._quran_data["surahs"][surah_num]["verses"][ayah_str] = f"[Verse {verse_key}]"
        
        return self._quran_data
    
    def _route_query(self, question: str) -> Dict[str, Any]:
        """Route query to determine intent."""
        import re
        
        # SURAH_REF: "سورة الفاتحة", "surah 1", etc.
        surah_patterns = [
            r"سورة\s+([\u0600-\u06FF]+)",
            r"surah\s+(\d+)",
            r"سورة\s+(\d+)",
        ]
        
        for pattern in surah_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return {"intent": "SURAH_REF", "surah": match.group(1)}
        
        # AYAH_REF: "2:255", "البقرة:255", etc.
        ayah_patterns = [
            r"(\d+):(\d+)",
            r"([\u0600-\u06FF]+):(\d+)",
        ]
        
        for pattern in ayah_patterns:
            match = re.search(pattern, question)
            if match:
                return {"intent": "AYAH_REF", "surah": match.group(1), "ayah": match.group(2)}
        
        # Default to FREE_TEXT
        return {"intent": "FREE_TEXT"}
    
    def _get_surah_number(self, surah_ref: str) -> Optional[int]:
        """Convert surah reference to number."""
        surah_names = {
            "الفاتحة": 1, "البقرة": 2, "آل عمران": 3, "النساء": 4, "المائدة": 5,
            "الأنعام": 6, "الأعراف": 7, "الأنفال": 8, "التوبة": 9, "يونس": 10,
            "هود": 11, "يوسف": 12, "الرعد": 13, "إبراهيم": 14, "الحجر": 15,
            "النحل": 16, "الإسراء": 17, "الكهف": 18, "مريم": 19, "طه": 20,
        }
        
        if surah_ref.isdigit():
            return int(surah_ref)
        return surah_names.get(surah_ref)
    
    def get_deterministic_evidence(self, question: str) -> Dict[str, Any]:
        """
        Get deterministic evidence for structured queries.
        
        Returns proof structure with 7 tafsir sources from chunked index.
        """
        start_time = time.time()
        
        # Initialize debug
        debug = LightweightProofDebug()
        
        # Route query
        route_result = self._route_query(question)
        intent = route_result.get("intent", "FREE_TEXT")
        debug.intent = intent
        
        # Load data
        evidence_index = self._load_evidence_index()
        quran_data = self._load_quran_data()
        
        # Initialize tafsir results
        tafsir_results = {src: [] for src in CORE_TAFSIR_SOURCES}
        quran_verses = []
        
        if intent == "SURAH_REF":
            surah_num = self._get_surah_number(route_result.get("surah", ""))
            if surah_num:
                # Get all verses for this surah
                surah_data = quran_data.get("surahs", {}).get(str(surah_num), {})
                verses = surah_data.get("verses", {})
                
                for ayah_num, verse_text in verses.items():
                    verse_key = f"{surah_num}:{ayah_num}"
                    quran_verses.append({
                        "surah": surah_num,
                        "ayah": int(ayah_num),
                        "text": verse_text,
                        "relevance": 1.0,
                    })
                    
                    # Get tafsir chunks for this verse
                    chunks = evidence_index.get(verse_key, [])
                    for chunk in chunks:
                        source = chunk.get("source", "")
                        if source in tafsir_results:
                            tafsir_results[source].append({
                                "surah": surah_num,
                                "ayah": int(ayah_num),
                                "text": chunk.get("text", ""),
                                "score": chunk.get("score", 1.0),
                                "chunk_id": chunk.get("chunk_id", ""),
                            })
                
                debug.retrieval_mode = "deterministic_chunked"
                
        elif intent == "AYAH_REF":
            surah_ref = route_result.get("surah", "")
            ayah_num = route_result.get("ayah", "")
            surah_num = self._get_surah_number(surah_ref)
            
            if surah_num and ayah_num:
                verse_key = f"{surah_num}:{ayah_num}"
                
                # Get verse text
                surah_data = quran_data.get("surahs", {}).get(str(surah_num), {})
                verse_text = surah_data.get("verses", {}).get(str(ayah_num), "")
                
                if verse_text:
                    quran_verses.append({
                        "surah": surah_num,
                        "ayah": int(ayah_num),
                        "text": verse_text,
                        "relevance": 1.0,
                    })
                
                # Get tafsir chunks
                chunks = evidence_index.get(verse_key, [])
                for chunk in chunks:
                    source = chunk.get("source", "")
                    if source in tafsir_results:
                        tafsir_results[source].append({
                            "surah": surah_num,
                            "ayah": int(ayah_num),
                            "text": chunk.get("text", ""),
                            "score": chunk.get("score", 1.0),
                            "chunk_id": chunk.get("chunk_id", ""),
                        })
                
                debug.retrieval_mode = "deterministic_chunked"
        
        # Track sources covered
        debug.sources_covered = [src for src in CORE_TAFSIR_SOURCES if tafsir_results[src]]
        debug.tafsir_fallbacks = {src: len(tafsir_results[src]) == 0 for src in CORE_TAFSIR_SOURCES}
        
        elapsed = time.time() - start_time
        debug.primary_path_latency_ms = round(elapsed * 1000)
        
        return {
            "question": question,
            "answer": "[proof_only mode - LLM answer skipped]",
            "quran": quran_verses,
            "tafsir": tafsir_results,
            "debug": debug.to_dict(),
            "processing_time_ms": round(elapsed * 1000, 2),
        }


# Singleton instance
_lightweight_backend: Optional[LightweightProofBackend] = None


def get_lightweight_backend() -> LightweightProofBackend:
    """Get or create lightweight proof backend singleton."""
    global _lightweight_backend
    if _lightweight_backend is None:
        _lightweight_backend = LightweightProofBackend()
    return _lightweight_backend
