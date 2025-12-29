"""
CROSS_CONTEXT_BEHAVIOR Handler - Deterministic cross-context behavioral analysis.

Phase 11: Implements first-class intent handling for queries like:
"أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة"

This handler:
1. Detects CROSS_CONTEXT_BEHAVIOR intent via trigger patterns
2. Fails closed if no behavior is specified (returns need_behavior)
3. Computes context clustering deterministically from real data
4. Returns verse-locked evidence with proper confidence scores

Non-negotiables:
- NO generic verse retrieval
- NO "opening verses" as default
- NO 100% fake confidence
- NO cross-verse tafsir citations
- NO BHV_* placeholder graph nodes
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

logger = logging.getLogger(__name__)

# Canonical data paths
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
EVIDENCE_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
QURAN_DATA_FILE = Path("data/quran/quran_tokenized_v1.json")

# Tafsir sources (7 sources)
TAFSIR_SOURCES = CANONICAL_TAFSIR_SOURCES


@dataclass
class ContextSignature:
    """Context signature for a verse."""
    surah: int = 0
    agent_type: str = "unknown"
    systemic: str = "unknown"
    surah_phase: str = "unknown"  # makki/madani/unknown
    co_entities: List[str] = field(default_factory=list)
    
    def to_tuple(self) -> Tuple[int, str, str, str]:
        """Return grouping tuple for clustering."""
        return (self.surah, self.surah_phase, self.agent_type, self.systemic)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah": self.surah,
            "agent_type": self.agent_type,
            "systemic": self.systemic,
            "surah_phase": self.surah_phase,
            "co_entities": self.co_entities[:3],
        }


@dataclass
class VerseEvidence:
    """Evidence for a single verse."""
    verse_key: str
    surah: int
    ayah: int
    text: str
    context: ContextSignature
    tafsir: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    evidence_count: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_key": self.verse_key,
            "surah": self.surah,
            "ayah": self.ayah,
            "text": self.text,
            "context": self.context.to_dict(),
            "tafsir": self.tafsir,
            "evidence_count": self.evidence_count,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CrossContextResult:
    """Result from cross-context behavior analysis."""
    status: str  # "success", "partial", "no_evidence", "need_behavior", "error"
    behavior_id: Optional[str] = None
    behavior_term: Optional[str] = None
    behavior_term_en: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    context_groups: List[Dict[str, Any]] = field(default_factory=list)
    selected_verses: List[VerseEvidence] = field(default_factory=list)
    candidate_behaviors: List[Dict[str, Any]] = field(default_factory=list)
    message_ar: str = ""
    message_en: str = ""
    verification_pct: float = 0.0
    graph: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "behavior_id": self.behavior_id,
            "behavior_term": self.behavior_term,
            "behavior_term_en": self.behavior_term_en,
            "reasons": self.reasons,
            "context_groups": self.context_groups,
            "selected_verses": [v.to_dict() for v in self.selected_verses],
            "candidate_behaviors": self.candidate_behaviors,
            "message_ar": self.message_ar,
            "message_en": self.message_en,
            "verification_pct": round(self.verification_pct, 2),
            "graph": self.graph,
        }


class CrossContextBehaviorHandler:
    """
    Handler for CROSS_CONTEXT_BEHAVIOR queries.
    
    Implements deterministic cross-context behavioral analysis.
    """
    
    # Trigger patterns for CROSS_CONTEXT_BEHAVIOR intent
    TRIGGER_PATTERNS_AR = [
        r"نفس\s+السلوك",
        r"سياقات\s+مختلفة",
        r"في\s+سياقات",
        r"في\s+سياقات\s+متعددة",
        r"اختلاف\s+السياق(?:ات)?",
        r"نفس\s+(السلوك|الفعل|الصفة)\s*.*(مختلف|متعدد)",
        r"السلوك\s+في\s+سياقات",
        r"تكرار\s+السلوك",
        r"السلوك\s+المتكرر",
    ]
    
    TRIGGER_PATTERNS_EN = [
        r"same\s+behavior",
        r"different\s+contexts",
        r"across\s+contexts",
        r"behavior\s+in\s+(different|multiple|various)\s+contexts",
        r"repeated\s+behavior",
        r"recurring\s+behavior",
    ]
    
    def __init__(self):
        self._canonical_entities = None
        self._concept_index = None
        self._evidence_index = None
        self._quran_data = None
        self._behavior_term_to_id = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Load all required data."""
        if self._initialized:
            return
        
        # Load canonical entities
        if CANONICAL_ENTITIES_FILE.exists():
            with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
                self._canonical_entities = json.load(f)
            
            # Build term -> ID mapping
            for behavior in self._canonical_entities.get("behaviors", []):
                term_ar = behavior.get("ar", "").strip()
                term_ar_no_al = term_ar[2:] if term_ar.startswith("ال") else term_ar
                self._behavior_term_to_id[term_ar] = behavior["id"]
                self._behavior_term_to_id[term_ar_no_al] = behavior["id"]
                self._behavior_term_to_id[behavior["id"]] = behavior["id"]
                if behavior.get("en"):
                    self._behavior_term_to_id[behavior["en"].lower()] = behavior["id"]
        
        # Load concept index (behavior -> verses mapping)
        self._concept_index = {}
        if CONCEPT_INDEX_FILE.exists():
            with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._concept_index[entry["concept_id"]] = entry
        
        # Load evidence index (verse -> tafsir chunks)
        self._evidence_index = defaultdict(list)
        if EVIDENCE_INDEX_FILE.exists():
            with open(EVIDENCE_INDEX_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        verse_key = chunk.get("verse_key")
                        if verse_key:
                            self._evidence_index[verse_key].append(chunk)
        
        # Load Quran data
        if QURAN_DATA_FILE.exists():
            with open(QURAN_DATA_FILE, "r", encoding="utf-8") as f:
                self._quran_data = json.load(f)
        else:
            # Try alternative path
            alt_path = Path("data/quran_tokenized_v1.json")
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    self._quran_data = json.load(f)
        
        self._initialized = True
        logger.info(f"CrossContextBehaviorHandler initialized: {len(self._concept_index)} concepts, {len(self._evidence_index)} verse evidence entries")
    
    def is_cross_context_query(self, query: str) -> bool:
        """Check if query matches CROSS_CONTEXT_BEHAVIOR intent."""
        query_lower = query.lower()
        
        # Check Arabic patterns
        for pattern in self.TRIGGER_PATTERNS_AR:
            if re.search(pattern, query):
                return True
        
        # Check English patterns
        for pattern in self.TRIGGER_PATTERNS_EN:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def extract_behavior_from_query(self, query: str) -> Optional[str]:
        """
        Extract behavior term/ID from query.
        
        Returns behavior_id if found, None otherwise.
        """
        self.initialize()
        candidates = self.extract_behavior_candidates(query)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def extract_behavior_candidates(self, query: str) -> List[str]:
        """
        Extract 0..N behavior IDs from query, deterministically (in appearance order).

        If multiple behaviors are detected, the caller must fail-closed and request disambiguation.
        """
        self.initialize()

        found: List[str] = []
        seen: set[str] = set()

        # Explicit behavior IDs (BEH_*)
        for m in re.finditer(r"BEH_\w+", query):
            behavior_id = m.group(0)
            if behavior_id in self._concept_index and behavior_id not in seen:
                found.append(behavior_id)
                seen.add(behavior_id)

        # Arabic behavior terms (token order)
        for m in re.finditer(r"[\u0600-\u06FF]+", query):
            token = m.group(0)
            candidates = [token]
            if token.startswith("ال") and len(token) > 2:
                candidates.append(token[2:])
            for cand in candidates:
                behavior_id = self._behavior_term_to_id.get(cand)
                if behavior_id and behavior_id in self._concept_index and behavior_id not in seen:
                    found.append(behavior_id)
                    seen.add(behavior_id)

        # English terms (deterministic by longest match first)
        query_lower = query.lower()
        english_terms = [
            (term.lower(), behavior_id)
            for term, behavior_id in self._behavior_term_to_id.items()
            if isinstance(term, str) and term.isascii() and term
        ]
        english_terms.sort(key=lambda t: (-len(t[0]), t[0]))
        for term, behavior_id in english_terms:
            if term in query_lower and behavior_id in self._concept_index and behavior_id not in seen:
                found.append(behavior_id)
                seen.add(behavior_id)

        return found
    
    def get_candidate_behaviors(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top candidate behaviors for suggestion.
        
        Returns behaviors sorted by occurrence count.
        """
        self.initialize()
        
        candidates = []
        for concept_id, entry in self._concept_index.items():
            if not concept_id.startswith("BEH_"):
                continue
            
            verses = entry.get("verses", [])
            if len(verses) < 2:
                continue

            unique_surahs = {v.get("surah") for v in verses if isinstance(v, dict) and v.get("surah")}
            if len(unique_surahs) < 2:
                continue
            
            # Get behavior info from canonical entities
            behavior_info = None
            for b in self._canonical_entities.get("behaviors", []):
                if b["id"] == concept_id:
                    behavior_info = b
                    break
            
            if not behavior_info:
                continue
            
            # Get example verse keys
            example_verses = [v["verse_key"] for v in verses[:3]]
            
            candidates.append({
                "behavior_id": concept_id,
                "label_ar": behavior_info.get("ar", ""),
                "label_en": behavior_info.get("en", ""),
                "occurrence_count": len(verses),
                "example_verse_keys": example_verses,
            })
        
        # Sort by occurrence count descending
        candidates.sort(key=lambda x: x["occurrence_count"], reverse=True)
        return candidates[:top_n]
    
    def handle(self, query: str) -> CrossContextResult:
        """
        Handle a CROSS_CONTEXT_BEHAVIOR query.
        
        Returns:
        - need_behavior if no behavior specified
        - partial if only one context cluster can be formed
        - no_evidence if concept index has no verses
        - success with context-grouped verses otherwise
        """
        self.initialize()
        
        # 1. Extract behavior from query (fail-closed on ambiguity)
        matched_candidates = self.extract_behavior_candidates(query)
        behavior_id = matched_candidates[0] if len(matched_candidates) == 1 else None
        
        # 2. If no behavior specified, return need_behavior with candidates
        if not behavior_id:
            if len(matched_candidates) > 1:
                # Ambiguous: only suggest detected candidates
                candidates = []
                for beh_id in matched_candidates[:10]:
                    beh_info = next(
                        (b for b in self._canonical_entities.get("behaviors", []) if b.get("id") == beh_id),
                        None,
                    )
                    verse_count = len((self._concept_index.get(beh_id) or {}).get("verses", []))
                    candidates.append(
                        {
                            "behavior_id": beh_id,
                            "label_ar": (beh_info or {}).get("ar", ""),
                            "label_en": (beh_info or {}).get("en", ""),
                            "occurrence_count": verse_count,
                            "example_verse_keys": [],
                        }
                    )
            else:
                candidates = self.get_candidate_behaviors(10)
            return CrossContextResult(
                status="need_behavior",
                candidate_behaviors=candidates,
                reasons=["missing_or_ambiguous_behavior"],
                message_ar="حدد السلوك المطلوب (مثال: الصبر/الصدق/الحسد)",
                message_en="Please specify the behavior (e.g., patience/truthfulness/envy)",
                graph={"nodes": [], "edges": [], "status": "awaiting_behavior"},
            )
        
        # 3. Get behavior info
        concept_entry = self._concept_index.get(behavior_id)
        if not concept_entry:
            return CrossContextResult(
                status="no_evidence",
                behavior_id=behavior_id,
                reasons=["behavior_not_in_concept_index"],
                message_ar=f"لا يوجد دليل في فهرس المفاهيم للسلوك: {behavior_id}",
                message_en=f"No evidence in concept index for behavior: {behavior_id}",
            )
        
        behavior_info = None
        for b in self._canonical_entities.get("behaviors", []):
            if b["id"] == behavior_id:
                behavior_info = b
                break
        
        # 4. Get verses for this behavior (hard-gated by concept index)
        verses = concept_entry.get("verses", [])
        if not verses:
            return CrossContextResult(
                status="no_evidence",
                behavior_id=behavior_id,
                behavior_term=behavior_info.get("ar") if behavior_info else behavior_id,
                behavior_term_en=behavior_info.get("en") if behavior_info else "",
                reasons=["no_verses_for_behavior"],
                message_ar=f"لا توجد آيات مفهرسة للسلوك '{behavior_info.get('ar', behavior_id)}'",
                message_en=f"No indexed verses for behavior '{behavior_info.get('en', behavior_id)}'",
            )
        
        # 5. Cluster deterministically by tracked context dimensions (minimum: surah + Makki/Madani)
        group_stats: Dict[Tuple[int, str, str, str], Dict[str, Any]] = {}
        for v in verses:
            verse_key = v.get("verse_key")
            surah = v.get("surah")
            ayah = v.get("ayah")
            if not verse_key or not surah or not ayah:
                continue

            context = self._get_context_signature(verse_key, int(surah))
            group_key = context.to_tuple()

            evidence_count = len(self._evidence_index.get(verse_key, []))
            stats = group_stats.setdefault(
                group_key,
                {"verse_count": 0, "best": None, "best_evidence_count": -1, "best_surah": 0, "best_ayah": 0},
            )
            stats["verse_count"] += 1

            best_evidence = int(stats["best_evidence_count"])
            best_surah = int(stats["best_surah"])
            best_ayah = int(stats["best_ayah"])

            if (
                evidence_count > best_evidence
                or (evidence_count == best_evidence and (int(surah), int(ayah)) < (best_surah, best_ayah))
            ):
                stats["best"] = {"verse_key": verse_key, "surah": int(surah), "ayah": int(ayah)}
                stats["best_evidence_count"] = evidence_count
                stats["best_surah"] = int(surah)
                stats["best_ayah"] = int(ayah)

        if not group_stats:
            return CrossContextResult(
                status="no_evidence",
                behavior_id=behavior_id,
                behavior_term=behavior_info.get("ar") if behavior_info else behavior_id,
                behavior_term_en=behavior_info.get("en") if behavior_info else "",
                reasons=["no_context_groups"],
                message_ar=f"تعذر تشكيل مجموعات سياقية للسلوك '{behavior_info.get('ar', behavior_id)}' بسبب نقص بيانات السياق",
                message_en=f"Could not form context clusters for '{behavior_info.get('en', behavior_id)}' due to missing context data",
            )

        # 6. Select representative verses from up to N groups (deterministic ordering)
        max_groups = 6
        groups_ranked = sorted(
            group_stats.items(),
            key=lambda kv: (
                -int(kv[1]["verse_count"]),
                -int(kv[1]["best_evidence_count"]),
                int(kv[0][0]),  # surah
                str(kv[0][1]),  # surah_phase
                str(kv[0][2]),  # agent_type
                str(kv[0][3]),  # systemic
            ),
        )

        selected_verses: List[VerseEvidence] = []
        context_group_info: List[Dict[str, Any]] = []

        for group_key, stats in groups_ranked[:max_groups]:
            best = stats.get("best") or {}
            verse_key = best.get("verse_key")
            surah = best.get("surah")
            ayah = best.get("ayah")
            if not verse_key or not surah or not ayah:
                continue

            context = self._get_context_signature(verse_key, int(surah))
            verse_text = self._get_verse_text(int(surah), int(ayah))
            tafsir = self._get_verse_locked_tafsir(verse_key)
            evidence_count = sum(len(chunks) for chunks in tafsir.values())
            confidence = self._compute_confidence(tafsir, context)

            selected_verses.append(
                VerseEvidence(
                    verse_key=verse_key,
                    surah=int(surah),
                    ayah=int(ayah),
                    text=verse_text,
                    context=context,
                    tafsir=tafsir,
                    evidence_count=evidence_count,
                    confidence=confidence,
                )
            )
            context_group_info.append(
                {
                    "context_signature": {
                        "surah": group_key[0],
                        "surah_phase": group_key[1],
                        "agent_type": group_key[2],
                        "systemic": group_key[3],
                    },
                    "verse_count": int(stats.get("verse_count", 0)),
                    "representative_verse": verse_key,
                }
            )

        # 7. Compute verification percentage (deterministic)
        verified_count = sum(1 for v in selected_verses if len([s for s, chunks in v.tafsir.items() if chunks]) >= 3)
        verification_pct = (verified_count / len(selected_verses)) * 100 if selected_verses else 0

        graph = self._build_canonical_graph(behavior_id, selected_verses)

        # Hard gate: must produce >=2 distinct clusters, else partial (still evidence-backed)
        if len(group_stats) < 2:
            return CrossContextResult(
                status="partial",
                behavior_id=behavior_id,
                behavior_term=behavior_info.get("ar") if behavior_info else behavior_id,
                behavior_term_en=behavior_info.get("en") if behavior_info else "",
                reasons=["only_one_context_cluster"],
                context_groups=context_group_info,
                selected_verses=selected_verses,
                verification_pct=verification_pct,
                graph=graph,
                message_ar=f"السلوك '{behavior_info.get('ar', behavior_id)}' يظهر في سياق واحد فقط حسب أبعاد السياق المتاحة (المطلوب ≥ 2 سياقات)",
                message_en=f"Behavior '{behavior_info.get('en', behavior_id)}' appears in only one context cluster (need ≥ 2)",
            )
        
        return CrossContextResult(
            status="success",
            behavior_id=behavior_id,
            behavior_term=behavior_info.get("ar") if behavior_info else behavior_id,
            behavior_term_en=behavior_info.get("en") if behavior_info else "",
            context_groups=context_group_info,
            selected_verses=selected_verses,
            verification_pct=verification_pct,
            graph=graph,
            message_ar=f"تم العثور على السلوك '{behavior_info.get('ar', behavior_id)}' في {len(group_stats)} سياقات مختلفة",
            message_en=f"Found behavior '{behavior_info.get('en', behavior_id)}' in {len(group_stats)} different contexts",
        )
    
    def _get_context_signature(self, verse_key: str, surah: int) -> ContextSignature:
        """Get context signature for a verse."""
        # Determine Makki/Madani (simplified - based on surah number)
        # Surahs 1-86 are mostly Makki, 87-114 are mostly Madani (simplified)
        # In reality, this should come from a proper Makki/Madani dataset
        makki_surahs = {1, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 56, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114}
        surah_phase = "makki" if surah in makki_surahs else "madani"
        
        # Get co-entities from evidence index
        co_entities = []
        evidence_chunks = self._evidence_index.get(verse_key, [])
        for chunk in evidence_chunks[:5]:
            # Extract entity mentions from chunk if available
            if "entities" in chunk:
                co_entities.extend(chunk["entities"][:3])
        
        # Agent type and systemic would come from span annotations
        # For now, use "unknown" - this should be enhanced with real data
        agent_type = "unknown"
        systemic = "unknown"
        
        return ContextSignature(
            surah=surah,
            agent_type=agent_type,
            systemic=systemic,
            surah_phase=surah_phase,
            co_entities=list(set(co_entities))[:3],
        )
    
    def _get_verse_text(self, surah: int, ayah: int) -> str:
        """Get verse text from Quran data."""
        if not self._quran_data:
            return ""
        
        # Try different data structures
        if "verses" in self._quran_data:
            for verse in self._quran_data["verses"]:
                if verse.get("surah") == surah and verse.get("ayah") == ayah:
                    return verse.get("text", "")
        
        # Try surah-based structure
        if "surahs" in self._quran_data:
            for s in self._quran_data["surahs"]:
                if s.get("number") == surah:
                    for v in s.get("ayat", []):
                        if v.get("number") == ayah:
                            return v.get("text", "")
        
        return ""
    
    def _get_verse_locked_tafsir(self, verse_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get tafsir evidence locked to a specific verse.
        
        CRITICAL: Only returns tafsir chunks where chunk.verse_key == verse_key.
        Never returns cross-verse citations.
        """
        tafsir = {source: [] for source in TAFSIR_SOURCES}
        
        chunks = self._evidence_index.get(verse_key, [])
        for chunk in chunks:
            source = chunk.get("source", "")
            chunk_verse_key = chunk.get("verse_key", "")
            
            # HARD RULE: Only include if verse_key matches exactly
            if chunk_verse_key != verse_key:
                logger.warning(f"Skipping cross-verse citation: chunk {chunk.get('chunk_id')} has verse_key {chunk_verse_key}, expected {verse_key}")
                continue
            
            if source in tafsir:
                tafsir[source].append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "verse_key": chunk_verse_key,
                    "source": source,
                    "surah": chunk.get("surah"),
                    "ayah": chunk.get("ayah"),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                    "text": chunk.get("text_clean", chunk.get("text", ""))[:500],  # Limit text length
                })
        
        return tafsir
    
    def _compute_confidence(self, tafsir: Dict[str, List[Dict[str, Any]]], context: ContextSignature) -> float:
        """
        Compute deterministic confidence score.
        
        Formula:
        confidence = min(1.0, (sources_present/7)*0.6 + (has_mentions ? 0.2 : 0) + (context_fields_ratio*0.2))
        """
        # Count sources with evidence
        sources_present = sum(1 for source, chunks in tafsir.items() if chunks)
        source_score = (sources_present / 7) * 0.6
        
        # Check if any mentions exist
        has_mentions = sources_present > 0
        mention_score = 0.2 if has_mentions else 0.0
        
        # Context fields ratio
        context_fields = 0
        if context.agent_type != "unknown":
            context_fields += 1
        if context.systemic != "unknown":
            context_fields += 1
        if context.surah_phase != "unknown":
            context_fields += 1
        context_score = (context_fields / 3) * 0.2
        
        confidence = min(1.0, source_score + mention_score + context_score)
        return confidence
    
    def _build_canonical_graph(self, behavior_id: str, selected_verses: List[VerseEvidence]) -> Dict[str, Any]:
        """
        Build graph with canonical IDs only.
        
        HARD RULE: No BHV_* placeholders. Only canonical prefixes:
        BEH_*, AGT_*, ORG_*, HRT_*, CSQ_*
        """
        nodes = []
        edges = []
        
        # Add behavior node
        nodes.append({
            "id": behavior_id,
            "type": "BEHAVIOR",
            "label": behavior_id,
        })
        
        # Add verse nodes and edges
        for ve in selected_verses:
            verse_node_id = f"VERSE_{ve.verse_key.replace(':', '_')}"
            nodes.append({
                "id": verse_node_id,
                "type": "VERSE",
                "label": ve.verse_key,
                "context": ve.context.to_dict(),
            })
            
            edges.append({
                "source": behavior_id,
                "target": verse_node_id,
                "type": "APPEARS_IN",
            })
            
            # Add co-entity nodes if they have canonical IDs
            for co_entity in ve.context.co_entities:
                if any(co_entity.startswith(prefix) for prefix in ["BEH_", "AGT_", "ORG_", "HRT_", "CSQ_"]):
                    if not any(n["id"] == co_entity for n in nodes):
                        nodes.append({
                            "id": co_entity,
                            "type": co_entity.split("_")[0],
                            "label": co_entity,
                        })
                    edges.append({
                        "source": verse_node_id,
                        "target": co_entity,
                        "type": "CO_OCCURS",
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "status": "evidence_backed" if nodes else "no_evidence",
        }


# Singleton instance
_handler: Optional[CrossContextBehaviorHandler] = None


def get_cross_context_handler() -> CrossContextBehaviorHandler:
    """Get or create the singleton handler."""
    global _handler
    if _handler is None:
        _handler = CrossContextBehaviorHandler()
    return _handler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    handler = get_cross_context_handler()
    
    # Test queries
    test_queries = [
        # Should return need_behavior (no behavior specified)
        "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة",
        
        # Should process with behavior
        "الصبر في سياقات مختلفة",
        "BEH_EMO_PATIENCE في سياقات مختلفة",
        "same behavior patience in different contexts",
    ]
    
    print("\n" + "=" * 60)
    print("Testing CrossContextBehaviorHandler")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Is cross-context query: {handler.is_cross_context_query(query)}")
        
        if handler.is_cross_context_query(query):
            result = handler.handle(query)
            print(f"Status: {result.status}")
            print(f"Message (AR): {result.message_ar}")
            if result.status == "need_behavior":
                print(f"Candidate behaviors: {len(result.candidate_behaviors)}")
                for c in result.candidate_behaviors[:3]:
                    print(f"  - {c['label_ar']} ({c['behavior_id']}): {c['occurrence_count']} occurrences")
            elif result.status == "success":
                print(f"Context groups: {len(result.context_groups)}")
                print(f"Selected verses: {len(result.selected_verses)}")
                print(f"Verification %: {result.verification_pct:.1f}%")
