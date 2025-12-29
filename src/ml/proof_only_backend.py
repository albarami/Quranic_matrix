"""
Lightweight Proof-Only Backend - Phase 9.10E + Benchmark Planners

This module provides a minimal proof pipeline that does NOT initialize:
- GPU embedding pipeline
- Cross-encoder reranker
- Vector index (FAISS)
- FullPower system

It uses only:
- evidence_index_v2_chunked.jsonl (deterministic verse-key lookup)
- concept_index_v2.jsonl (behavior -> verse mapping)
- semantic_graph_v2.json (causal graph for path queries)
- canonical_entities.json (behavior term resolution)

Supported intents (Benchmark Sections A-J):
- GRAPH_CAUSAL (A): Causal chain analysis using semantic_graph_v2
- CROSS_TAFSIR_ANALYSIS (B): Multi-source tafsir comparison
- PROFILE_11D (C): 11-dimensional behavior profiles
- GRAPH_METRICS (D): Graph statistics (node count, centrality, etc.)
- HEART_STATE (E): Heart state analysis
- AGENT_ANALYSIS (F): Agent type analysis
- TEMPORAL_SPATIAL (G): Temporal/spatial context
- CONSEQUENCE_ANALYSIS (H): Consequence/punishment analysis
- EMBEDDINGS_ANALYSIS (I): Embedding space analysis (partial)
- INTEGRATION_E2E (J): End-to-end integration queries
- SURAH_REF: Full surah tafsir retrieval
- AYAH_REF: Single verse tafsir retrieval
- CONCEPT_REF: Behavior/concept queries

Target: <5 seconds for structured intent queries
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

# Core tafsir sources (7 sources)
CORE_TAFSIR_SOURCES = CANONICAL_TAFSIR_SOURCES


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
    fail_closed_reason: Optional[str] = None
    intent: str = "FREE_TEXT"
    retrieval_mode: str = "deterministic_chunked"  # "hybrid" | "stratified" | "rag_only" | "deterministic_chunked"
    sources_covered: List[str] = field(default_factory=list)
    core_sources_count: int = 0
    
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
            "fail_closed_reason": self.fail_closed_reason,
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
        self._concept_index: Optional[Dict[str, Dict]] = None  # concept_id -> data
        self._semantic_graph: Optional[Dict[str, Any]] = None
        self._canonical_entities: Optional[Dict[str, Any]] = None
        self._behavior_term_map: Optional[Dict[str, str]] = None  # ar/en term -> behavior_id
        
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
    
    def _load_concept_index(self) -> Dict[str, Dict]:
        """Load concept index (behavior_id -> verse_keys + tafsir_chunks)."""
        if self._concept_index is not None:
            return self._concept_index
        
        self._concept_index = {}
        index_path = self.data_dir / "evidence" / "concept_index_v2.jsonl"
        if not index_path.exists():
            logging.warning(f"Concept index not found: {index_path}")
            return {}
        
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                concept_id = entry.get("concept_id", "")
                if concept_id:
                    self._concept_index[concept_id] = entry
        
        logging.info(f"[LightweightProof] Loaded {len(self._concept_index)} concepts from concept index")
        return self._concept_index
    
    def _load_semantic_graph(self) -> Dict[str, Any]:
        """Load semantic graph v2 for causal chain analysis."""
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
    
    def _load_canonical_entities(self) -> Dict[str, Any]:
        """Load canonical entities for behavior term resolution."""
        if self._canonical_entities is not None:
            return self._canonical_entities
        
        # Try vocab directory first
        entities_path = self.data_dir.parent / "vocab" / "canonical_entities.json"
        if not entities_path.exists():
            entities_path = Path("vocab") / "canonical_entities.json"
        
        if not entities_path.exists():
            logging.warning(f"Canonical entities not found: {entities_path}")
            return {"behaviors": []}
        
        with open(entities_path, "r", encoding="utf-8") as f:
            self._canonical_entities = json.load(f)
        
        # Build term -> behavior_id map
        self._behavior_term_map = {}
        for beh in self._canonical_entities.get("behaviors", []):
            beh_id = beh.get("id", "")
            ar_term = beh.get("ar", "")
            en_term = beh.get("en", "").lower()
            if ar_term:
                self._behavior_term_map[ar_term] = beh_id
            if en_term:
                self._behavior_term_map[en_term] = beh_id
        
        logging.info(f"[LightweightProof] Loaded {len(self._canonical_entities.get('behaviors', []))} behaviors")
        return self._canonical_entities
    
    def _resolve_behavior_term(self, term: str) -> Optional[str]:
        """Resolve Arabic/English behavior term to canonical behavior ID."""
        self._load_canonical_entities()
        if not self._behavior_term_map:
            return None
        
        # Try exact match first
        if term in self._behavior_term_map:
            return self._behavior_term_map[term]
        
        # Try lowercase
        term_lower = term.lower()
        if term_lower in self._behavior_term_map:
            return self._behavior_term_map[term_lower]
        
        # Try partial match for Arabic terms
        for key, beh_id in self._behavior_term_map.items():
            if term in key or key in term:
                return beh_id
        
        return None
    
    def _find_causal_paths(self, source_id: str, target_id: str, min_hops: int = 1, max_hops: int = 5) -> List[Dict]:
        """Find causal paths between two behaviors using BFS."""
        graph = self._load_semantic_graph()
        
        # Build adjacency list for causal edges
        causal_types = graph.get("causal_edge_types", ["CAUSES", "LEADS_TO", "STRENGTHENS"])
        edges = graph.get("edges", [])
        
        adj: Dict[str, List[Dict]] = {}
        for edge in edges:
            if edge.get("type") in causal_types:
                src = edge.get("source", "")
                if src not in adj:
                    adj[src] = []
                adj[src].append(edge)
        
        # BFS to find all paths
        from collections import deque
        paths = []
        queue = deque([(source_id, [source_id], [])])  # (current_node, path_nodes, path_edges)
        visited_paths = set()
        
        while queue and len(paths) < 20:  # Limit to 20 paths
            current, path_nodes, path_edges = queue.popleft()
            
            if current == target_id and len(path_nodes) > min_hops:
                path_key = tuple(path_nodes)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append({
                        "nodes": path_nodes,
                        "edges": path_edges,
                        "hops": len(path_nodes) - 1,
                    })
                continue
            
            if len(path_nodes) > max_hops:
                continue
            
            for edge in adj.get(current, []):
                next_node = edge.get("target", "")
                if next_node and next_node not in path_nodes:  # Avoid cycles
                    queue.append((
                        next_node,
                        path_nodes + [next_node],
                        path_edges + [edge]
                    ))
        
        return paths
    
    def _get_tafsir_for_verses(self, verse_keys: List[str], max_per_source: int = 3) -> Dict[str, List[Dict]]:
        """Get tafsir chunks for a list of verse keys."""
        evidence_index = self._load_evidence_index()
        tafsir_results = {src: [] for src in CORE_TAFSIR_SOURCES}
        
        for verse_key in verse_keys:
            chunks = evidence_index.get(verse_key, [])
            for chunk in chunks:
                source = chunk.get("source", "")
                if source in tafsir_results and len(tafsir_results[source]) < max_per_source * len(verse_keys):
                    parts = verse_key.split(":")
                    surah_num = int(parts[0]) if len(parts) > 0 else 0
                    ayah_num = int(parts[1]) if len(parts) > 1 else 0
                    tafsir_results[source].append({
                        "source": source,
                        "verse_key": verse_key,
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "chunk_id": chunk.get("chunk_id", ""),
                        "char_start": chunk.get("char_start", 0),
                        "char_end": chunk.get("char_end", len(chunk.get("text_clean", chunk.get("text", "")))),
                        "text": chunk.get("text_clean", chunk.get("text", "")),
                        "score": chunk.get("score", 1.0),
                    })
        
        return tafsir_results
    
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
        
        Routes to appropriate planner based on intent classification.
        Returns proof structure with 7 tafsir sources from chunked index.
        """
        start_time = time.time()
        
        # Initialize debug
        debug = LightweightProofDebug()
        
        # Use the new intent classifier
        from src.ml.intent_classifier import classify_intent, IntentType
        intent_result = classify_intent(question)
        intent = intent_result.intent.value
        debug.intent = intent
        
        # Load evidence index (cached singleton)
        evidence_index = self._load_evidence_index()
        
        # Initialize results
        tafsir_results = {src: [] for src in CORE_TAFSIR_SOURCES}
        quran_verses = []
        graph_data = {"nodes": [], "edges": [], "paths": []}
        extra_data = {}
        
        # Route to appropriate planner based on intent
        # Use existing LegendaryPlanner for complex benchmark intents
        benchmark_intents = {
            IntentType.GRAPH_CAUSAL,
            IntentType.CROSS_TAFSIR_ANALYSIS,
            IntentType.PROFILE_11D,
            IntentType.GRAPH_METRICS,
            IntentType.HEART_STATE,
            IntentType.AGENT_ANALYSIS,
            IntentType.TEMPORAL_SPATIAL,
            IntentType.CONSEQUENCE_ANALYSIS,
            IntentType.EMBEDDINGS_ANALYSIS,
            IntentType.INTEGRATION_E2E,
        }
        
        if intent_result.intent in benchmark_intents:
            # Use existing LegendaryPlanner for benchmark questions
            from src.ml.legendary_planner import get_legendary_planner
            planner = get_legendary_planner()
            planner_results, planner_debug = planner.query(question)

            # Extract evidence from planner results
            debug.retrieval_mode = "legendary_planner"

            # PHASE 4A: Get graph data first (always needed for benchmark intents)
            graph_data_result = planner_results.get("graph_data", {})

            # PHASE 4B: For GRAPH_METRICS, the graph data IS the evidence.
            # Do NOT extract verse evidence - it would be irrelevant and cause generic_opening_verses_default
            if intent_result.intent == IntentType.GRAPH_METRICS:
                # Graph metrics queries don't need verse evidence
                # The graph data (nodes, edges, centrality, etc.) IS the proof
                graph_data = graph_data_result if graph_data_result else graph_data
                extra_data["planner_results"] = planner_results
                # Skip verse extraction - continue to the final return
            else:
                # For other benchmark intents, extract verse evidence normally
                # Get verse keys from evidence (entity-specific queries)
                verse_keys = set()
                for ev in planner_results.get("evidence", []):
                    verse_keys.update(ev.get("verse_keys", []))

                # Extract from cycles (entity-free)
                for cycle in graph_data_result.get("cycles", []):
                    for edge in cycle.get("edges", []):
                        if isinstance(edge, dict):
                            for ev in edge.get("evidence", []):
                                vk = ev.get("verse_key", "")
                                if vk:
                                    verse_keys.add(vk)

                # Extract from causal_density top nodes (entity-free) - ONLY for causal queries
                if intent_result.intent == IntentType.GRAPH_CAUSAL:
                    causal_density = graph_data_result.get("causal_density", {})
                    for node_info in causal_density.get("outgoing_top10", []) + causal_density.get("incoming_top10", []):
                        node_id = node_info.get("id", "")
                        if node_id:
                            # Get evidence for this node from concept index
                            node_evidence = planner.get_concept_evidence(node_id)
                            verse_keys.update(node_evidence.get("verse_keys", [])[:5])

                # Extract from paths (entity-specific)
                for path in graph_data_result.get("paths", []):
                    if isinstance(path, list):
                        for edge in path:
                            if isinstance(edge, dict):
                                for ev in edge.get("evidence", []):
                                    vk = ev.get("verse_key", "")
                                    if vk:
                                        verse_keys.add(vk)

                # Build quran_verses from verse_keys
                for vk in sorted(verse_keys)[:20]:  # Limit to 20 verses
                    parts = vk.split(":")
                    if len(parts) == 2:
                        try:
                            surah_num = int(parts[0])
                            ayah_num = int(parts[1])
                            quran_verses.append({
                                "surah": surah_num,
                                "ayah": ayah_num,
                                "verse_key": vk,
                                "text": self._get_verse_text(vk),
                                "relevance": 1.0,
                            })
                        except ValueError:
                            pass

                # Get tafsir for these verses
                if verse_keys:
                    tafsir_results = self._get_tafsir_for_verses(list(verse_keys)[:20])

                # Include graph data if available
                graph_data = graph_data_result if graph_data_result else graph_data
                extra_data["planner_results"] = planner_results
            
        elif intent_result.intent == IntentType.SURAH_REF:
            surah_ref = intent_result.extracted_entities.get("surah", "")
            surah_num = self._get_surah_number(surah_ref)
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
                    
                    quran_verses.append(
                        {
                            "surah": surah_num,
                            "ayah": ayah_num,
                            "verse_key": verse_key,
                            "text": verse_text,
                            "relevance": 1.0,
                        }
                    )
                    
                    # Get tafsir chunks for this verse
                    chunks = evidence_index.get(verse_key, [])
                    for chunk in chunks:
                        source = chunk.get("source", "")
                        if source in tafsir_results:
                            tafsir_results[source].append(
                                {
                                    "source": source,
                                    "verse_key": verse_key,
                                    "surah": surah_num,
                                    "ayah": ayah_num,
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "char_start": chunk.get("char_start"),
                                    "char_end": chunk.get("char_end"),
                                    "text": chunk.get("text_clean", chunk.get("text", "")),
                                    "score": chunk.get("score", 1.0),
                                }
                            )
                
                debug.retrieval_mode = "deterministic_chunked"
                
        elif intent_result.intent == IntentType.AYAH_REF:
            surah_ref = intent_result.extracted_entities.get("surah", "")
            ayah_str = intent_result.extracted_entities.get("ayah", "")
            surah_num = self._get_surah_number(surah_ref)
            
            if surah_num and ayah_str:
                ayah_num = int(ayah_str)
                verse_key = f"{surah_num}:{ayah_num}"
                verse_text = self._get_verse_text(verse_key)
                
                quran_verses.append(
                    {
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "verse_key": verse_key,
                        "text": verse_text,
                        "relevance": 1.0,
                    }
                )
                
                # Get tafsir chunks
                chunks = evidence_index.get(verse_key, [])
                for chunk in chunks:
                    source = chunk.get("source", "")
                    if source in tafsir_results:
                        tafsir_results[source].append(
                            {
                                "source": source,
                                "verse_key": verse_key,
                                "surah": surah_num,
                                "ayah": ayah_num,
                                "chunk_id": chunk.get("chunk_id", ""),
                                "char_start": chunk.get("char_start"),
                                "char_end": chunk.get("char_end"),
                                "text": chunk.get("text_clean", chunk.get("text", "")),
                                "score": chunk.get("score", 1.0),
                            }
                        )
                
                debug.retrieval_mode = "deterministic_chunked"
        
        # Track sources covered + retrieval distribution
        debug.sources_covered = [src for src in CORE_TAFSIR_SOURCES if tafsir_results[src]]
        debug.core_sources_count = len(debug.sources_covered)
        debug.retrieval_distribution = {src: len(tafsir_results[src]) for src in CORE_TAFSIR_SOURCES}
        debug.tafsir_fallbacks = {src: len(tafsir_results[src]) == 0 for src in CORE_TAFSIR_SOURCES}
        
        elapsed = time.time() - start_time
        debug.primary_path_latency_ms = round(elapsed * 1000)

        # Phase 0: Fail-closed status (no evidence)
        # PHASE 4: Graph-only queries (GRAPH_METRICS) can have valid graph data without quran/tafsir
        tafsir_chunks_total = sum(len(v or []) for v in tafsir_results.values())
        has_quran_tafsir = len(quran_verses) > 0 or tafsir_chunks_total > 0
        has_graph_data = bool(
            graph_data.get("nodes") or
            graph_data.get("edges") or
            graph_data.get("paths") or
            graph_data.get("centrality") or
            graph_data.get("causal_density") or
            graph_data.get("cycles")
        )

        status = "ok"
        if not has_quran_tafsir and not has_graph_data:
            status = "no_evidence"
            if debug.fail_closed_reason is None:
                debug.fail_closed_reason = "no_evidence"

        return {
            "question": question,
            "answer": "[proof_only mode - LLM answer skipped]",
            "status": status,
            "quran": quran_verses,
            "tafsir": tafsir_results,
            "graph": graph_data,  # PHASE 4: Include graph data in proof response
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
