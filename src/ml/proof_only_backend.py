"""
Lightweight Proof-Only Backend - Phase 9.10E

This module provides a minimal proof pipeline that does NOT initialize:
- GPU embedding pipeline
- Cross-encoder reranker
- Vector index (FAISS)
- FullPower system

It uses only:
- evidence_index_v2_chunked.jsonl (deterministic verse-key lookup)

Supported intents:
- SURAH_REF: Full surah tafsir retrieval by surah number/name
- AYAH_REF: Single verse tafsir retrieval by verse reference (e.g., 2:255)

Not yet implemented:
- CONCEPT_REF: Would require concept_index_v2.jsonl integration
- FREE_TEXT: Falls back to empty results (use FullPower for semantic search)

Target: <5 seconds for structured intent queries (SURAH_REF, AYAH_REF)
"""

import json
import logging
import os
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
    """
    Debug info for lightweight proof-only backend.
    
    IMPORTANT: This schema MUST match ProofDebug.to_dict() from mandatory_proof_system.py
    to ensure API contract parity between full and lightweight backends.
    """
    # Core fields (match ProofDebug)
    fallback_used: bool = False
    fallback_reasons: List[str] = field(default_factory=list)
    retrieval_distribution: Dict[str, int] = field(default_factory=dict)
    primary_path_latency_ms: int = 0
    index_source: str = "json_chunked"  # "disk" | "runtime_build" | "json_chunked"
    intent: str = "FREE_TEXT"
    retrieval_mode: str = "deterministic_chunked"  # "hybrid" | "stratified" | "rag_only" | "deterministic_chunked"
    sources_covered: List[str] = field(default_factory=list)
    core_sources_count: int = 7
    
    # Component fallbacks (match ProofDebug.component_fallbacks structure)
    quran_fallback: bool = False
    graph_fallback: bool = False
    taxonomy_fallback: bool = False
    tafsir_fallbacks: Dict[str, bool] = field(default_factory=dict)
    
    # Lightweight-specific fields
    fullpower_used: bool = False  # Always False for this backend
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary matching ProofDebug.to_dict() schema exactly.
        This ensures API contract parity between backends.
        """
        return {
            "fallback_used": self.fallback_used,
            "fallback_reasons": self.fallback_reasons,
            "retrieval_distribution": self.retrieval_distribution,
            "primary_path_latency_ms": self.primary_path_latency_ms,
            "index_source": self.index_source,
            "intent": self.intent,
            "retrieval_mode": self.retrieval_mode,
            "sources_covered": self.sources_covered,
            "core_sources_count": self.core_sources_count,
            "component_fallbacks": {
                "quran": self.quran_fallback,
                "graph": self.graph_fallback,
                "taxonomy": self.taxonomy_fallback,
                "tafsir": self.tafsir_fallbacks,
            },
            # Lightweight-specific (additive, not breaking)
            "fullpower_used": self.fullpower_used,
        }


def _get_data_dir() -> Path:
    """Get data directory path, robust to different working directories.
    
    Resolution order:
    1. QBM_DATA_DIR environment variable (absolute path)
    2. Relative to this module's location (../../data from src/ml/)
    3. Fallback to 'data' relative to CWD
    """
    # Check environment variable first
    env_path = os.environ.get("QBM_DATA_DIR")
    if env_path:
        return Path(env_path)
    
    # Resolve relative to module location
    module_dir = Path(__file__).resolve().parent  # src/ml/
    repo_root = module_dir.parent.parent  # repo root
    data_path = repo_root / "data"
    
    if data_path.exists():
        return data_path
    
    # Fallback to CWD-relative (for tests)
    return Path("data")


# Complete surah name mapping (all 114 surahs)
SURAH_NAMES = {
    "الفاتحة": 1, "البقرة": 2, "آل عمران": 3, "ال عمران": 3, "النساء": 4, "المائدة": 5,
    "الأنعام": 6, "الانعام": 6, "الأعراف": 7, "الاعراف": 7, "الأنفال": 8, "الانفال": 8,
    "التوبة": 9, "يونس": 10, "هود": 11, "يوسف": 12, "الرعد": 13, "إبراهيم": 14, "ابراهيم": 14,
    "الحجر": 15, "النحل": 16, "الإسراء": 17, "الاسراء": 17, "الكهف": 18, "مريم": 19, "طه": 20,
    "الأنبياء": 21, "الانبياء": 21, "الحج": 22, "المؤمنون": 23, "المؤمنين": 23, "النور": 24,
    "الفرقان": 25, "الشعراء": 26, "النمل": 27, "القصص": 28, "العنكبوت": 29, "الروم": 30,
    "لقمان": 31, "السجدة": 32, "الأحزاب": 33, "الاحزاب": 33, "سبأ": 34, "سبا": 34,
    "فاطر": 35, "يس": 36, "الصافات": 37, "ص": 38, "الزمر": 39, "غافر": 40,
    "فصلت": 41, "الشورى": 42, "الزخرف": 43, "الدخان": 44, "الجاثية": 45, "الأحقاف": 46, "الاحقاف": 46,
    "محمد": 47, "الفتح": 48, "الحجرات": 49, "ق": 50, "الذاريات": 51, "الطور": 52,
    "النجم": 53, "القمر": 54, "الرحمن": 55, "الواقعة": 56, "الحديد": 57, "المجادلة": 58,
    "الحشر": 59, "الممتحنة": 60, "الصف": 61, "الجمعة": 62, "المنافقون": 63, "التغابن": 64,
    "الطلاق": 65, "التحريم": 66, "الملك": 67, "القلم": 68, "الحاقة": 69, "المعارج": 70,
    "نوح": 71, "الجن": 72, "المزمل": 73, "المدثر": 74, "القيامة": 75, "الإنسان": 76, "الانسان": 76,
    "المرسلات": 77, "النبأ": 78, "النازعات": 79, "عبس": 80, "التكوير": 81, "الانفطار": 82,
    "المطففين": 83, "الانشقاق": 84, "البروج": 85, "الطارق": 86, "الأعلى": 87, "الاعلى": 87,
    "الغاشية": 88, "الفجر": 89, "البلد": 90, "الشمس": 91, "الليل": 92, "الضحى": 93,
    "الشرح": 94, "الانشراح": 94, "التين": 95, "العلق": 96, "القدر": 97, "البينة": 98,
    "الزلزلة": 99, "العاديات": 100, "القارعة": 101, "التكاثر": 102, "العصر": 103,
    "الهمزة": 104, "الفيل": 105, "قريش": 106, "الماعون": 107, "الكوثر": 108,
    "الكافرون": 109, "النصر": 110, "المسد": 111, "اللهب": 111, "الإخلاص": 112, "الاخلاص": 112,
    "الفلق": 113, "الناس": 114,
}


class LightweightProofBackend:
    """
    Minimal proof backend for Tier-A tests.
    
    Does NOT initialize GPU components. Uses only JSON files.
    Paths are resolved robustly via QBM_DATA_DIR env var or module-relative paths.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or _get_data_dir()
        self._evidence_index: Optional[Dict[str, List[Dict]]] = None
        self._quran_verses: Optional[Dict[str, str]] = None  # verse_key -> text
        
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
    
    def _load_quran_verses(self) -> Dict[str, str]:
        """Load Quran verse texts from quran_index.source.json.
        
        Returns dict mapping verse_key (e.g., '2:255') to Arabic text.
        If source file is not found, returns empty dict and verses will show as placeholders.
        """
        if self._quran_verses is not None:
            return self._quran_verses
        
        self._quran_verses = {}
        
        # Try to load from quran_index.source.json
        quran_path = self.data_dir / "quran" / "_incoming" / "quran_index.source.json"
        if quran_path.exists():
            try:
                with open(quran_path, "r", encoding="utf-8") as f:
                    quran_data = json.load(f)
                
                # Extract verse texts from the structure
                for surah in quran_data.get("surahs", []):
                    surah_num = surah.get("number", 0)
                    for ayah in surah.get("ayahs", []):
                        ayah_num = ayah.get("number", 0)
                        text = ayah.get("text", "")
                        if surah_num and ayah_num and text:
                            verse_key = f"{surah_num}:{ayah_num}"
                            self._quran_verses[verse_key] = text
                
                logging.info(f"[LightweightProof] Loaded {len(self._quran_verses)} Quran verses")
            except Exception as e:
                logging.warning(f"[LightweightProof] Failed to load Quran verses: {e}")
        else:
            logging.info(f"[LightweightProof] Quran source not found at {quran_path}, verse text will be placeholders")
        
        return self._quran_verses
    
    def _get_verse_text(self, verse_key: str) -> str:
        """Get verse text, with placeholder fallback."""
        verses = self._load_quran_verses()
        return verses.get(verse_key, f"[Verse {verse_key} - text not loaded in proof_only mode]")
    
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
        """Convert surah reference (name or number) to surah number."""
        if surah_ref.isdigit():
            num = int(surah_ref)
            return num if 1 <= num <= 114 else None
        return SURAH_NAMES.get(surah_ref)
    
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
        
        # Load evidence index (cached singleton)
        evidence_index = self._load_evidence_index()
        
        # Initialize tafsir results
        tafsir_results = {src: [] for src in CORE_TAFSIR_SOURCES}
        quran_verses = []
        
        if intent == "SURAH_REF":
            surah_num = self._get_surah_number(route_result.get("surah", ""))
            if surah_num:
                # Get all verse_keys for this surah from evidence index
                surah_prefix = f"{surah_num}:"
                surah_verse_keys = sorted(
                    [k for k in evidence_index.keys() if k.startswith(surah_prefix)],
                    key=lambda k: int(k.split(":")[1])
                )
                
                for verse_key in surah_verse_keys:
                    ayah_num = int(verse_key.split(":")[1])
                    verse_text = self._get_verse_text(verse_key)
                    
                    quran_verses.append({
                        "surah": surah_num,
                        "ayah": ayah_num,
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
                                "ayah": ayah_num,
                                "text": chunk.get("text_clean", chunk.get("text", "")),
                                "score": chunk.get("score", 1.0),
                                "chunk_id": chunk.get("chunk_id", ""),
                            })
                
                debug.retrieval_mode = "deterministic_chunked"
                
        elif intent == "AYAH_REF":
            surah_ref = route_result.get("surah", "")
            ayah_str = route_result.get("ayah", "")
            surah_num = self._get_surah_number(surah_ref)
            
            if surah_num and ayah_str:
                ayah_num = int(ayah_str)
                verse_key = f"{surah_num}:{ayah_num}"
                verse_text = self._get_verse_text(verse_key)
                
                quran_verses.append({
                    "surah": surah_num,
                    "ayah": ayah_num,
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
                            "ayah": ayah_num,
                            "text": chunk.get("text_clean", chunk.get("text", "")),
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
