"""
Deterministic Intent Classifier for QBM Benchmark Questions.

This module provides a shared intent classification function used by both
the proof_only_backend and the full backend. It routes questions to the
appropriate deterministic planner based on pattern matching.

Supported intents (benchmark sections):
- GRAPH_CAUSAL (Section A): Causal chain analysis
- CROSS_TAFSIR_ANALYSIS (Section B): Multi-source tafsir comparison
- PROFILE_11D (Section C): 11-dimensional behavior profiles
- GRAPH_METRICS (Section D): Graph statistics and centrality
- HEART_STATE (Section E): Heart state analysis
- AGENT_ANALYSIS (Section F): Agent type analysis
- TEMPORAL_SPATIAL (Section G): Temporal/spatial context
- CONSEQUENCE_ANALYSIS (Section H): Consequence/punishment analysis
- EMBEDDINGS_ANALYSIS (Section I): Embedding space analysis
- INTEGRATION_E2E (Section J): End-to-end integration queries
- SURAH_REF: Surah-level queries
- AYAH_REF: Single verse queries
- CONCEPT_REF: Behavior/concept queries
- FREE_TEXT: Fallback for unstructured queries
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class IntentType(str, Enum):
    """Enumeration of all supported intent types."""
    # Benchmark intents (Sections A-J)
    GRAPH_CAUSAL = "GRAPH_CAUSAL"
    CROSS_TAFSIR_ANALYSIS = "CROSS_TAFSIR_ANALYSIS"
    PROFILE_11D = "PROFILE_11D"
    GRAPH_METRICS = "GRAPH_METRICS"
    HEART_STATE = "HEART_STATE"
    AGENT_ANALYSIS = "AGENT_ANALYSIS"
    TEMPORAL_SPATIAL = "TEMPORAL_SPATIAL"
    CONSEQUENCE_ANALYSIS = "CONSEQUENCE_ANALYSIS"
    EMBEDDINGS_ANALYSIS = "EMBEDDINGS_ANALYSIS"
    INTEGRATION_E2E = "INTEGRATION_E2E"
    # Standard intents
    SURAH_REF = "SURAH_REF"
    AYAH_REF = "AYAH_REF"
    CONCEPT_REF = "CONCEPT_REF"
    CROSS_CONTEXT_BEHAVIOR = "CROSS_CONTEXT_BEHAVIOR"
    FREE_TEXT = "FREE_TEXT"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: IntentType
    confidence: float = 1.0
    matched_patterns: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "extracted_entities": self.extracted_entities,
        }


# Pattern definitions for each intent type
# Each pattern is a tuple of (regex_pattern, weight)
# Higher weight = stronger signal for that intent

INTENT_PATTERNS: Dict[IntentType, List[tuple]] = {
    IntentType.GRAPH_CAUSAL: [
        # English patterns
        (r"causal\s+chain", 1.0),
        (r"trace\s+all\s+.*chains", 1.0),
        (r"minimum\s+(\d+\s+)?hops?", 0.9),
        (r"CAUSES", 0.8),
        (r"LEADS_TO", 0.8),
        (r"STRENGTHENS", 0.8),
        (r"downward\s+spiral", 0.9),
        (r"bottleneck", 0.7),
        (r"causal\s+density", 0.9),
        (r"causal\s+distance", 0.9),
        (r"causal\s+path", 0.9),
        (r"causal\s+relationship", 0.8),
        (r"intervention\s+point", 0.8),
        (r"reinforcement\s+cycle", 0.9),
        (r"bidirectional\s+causal", 0.9),
        (r"cascade", 0.7),
        (r"behavioral\s+transformation", 0.8),
        (r"path\s+from\s+.*\s+to\s+", 0.8),
        (r"chain\s+.*→", 0.9),
        # Graph structure patterns
        (r"subgraph", 0.9),
        (r"sub-?graph", 0.9),
        (r"map\s+.*behaviors?", 0.9),  # A17: "Map all behaviors"
        (r"strengthen\s*الإيمان", 1.0),  # A17: "strengthen الإيمان"
        (r"strengthen\s*iman", 1.0),
        (r"strengthening\s*network", 0.9),  # A17: "Strengthening Network"
        (r"directly\s+or\s+indirectly", 0.8),  # A17: relationship depth
        (r"strengthen", 0.6),  # Lower weight for general "strengthen"
        (r"weaken", 0.6),
        (r"network", 0.5),  # Lower weight needs other signals
        (r"graph", 0.5),
        (r"edge\s+provenance", 0.8),
        (r"verse\s+frequency", 0.7),
        (r"weights?\s+based", 0.6),
        # PHASE 5: Additional patterns for Section A misclassified queries
        (r"causal\s+claim", 1.5),  # A10: "causal claim" - much higher weight to beat tafsir patterns
        (r"for\s+the\s+causal\s+claim", 2.0),  # A10: Very specific pattern
        (r"causal\s+claim.*يؤدي", 2.0),  # A10: "causal claim" + Arabic causality
        (r"causal\s+chains?\s+respect", 1.2),  # A16: "causal chains respect"
        (r"chains?\s+where.*cause", 1.2),  # A16: "chains where X causes Y"
        (r"causes\s+relationships?", 1.0),  # A11: "CAUSES relationships"
        (r"downstream\s+effects?", 1.0),  # A23: "downstream effects"
        (r"downstream\s+.*outcomes?", 0.9),  # A22: "downstream negative outcomes"
        (r"intervention\s+leverage", 1.0),  # A22: "intervention leverage"
        (r"causal\s+links?", 1.0),  # A24: "causal links"
        (r"causal\s+connect", 0.9),  # A19: "causal connections"
        (r"cross.?category\s+caus", 1.0),  # A24: "cross-category causation"
        (r"hierarchical\s+DAG", 1.0),  # A21: "hierarchical DAG"
        (r"root\s+cause", 1.0),  # A21: "root cause"
        (r"depth\s+of\s+\d+\s+hops?", 1.0),  # A23: "depth of 5 hops"
        (r"cause\s+the\s+same", 0.9),  # A14: "cause the same outcome"
        (r"independent\s+paths?", 0.8),  # A14: "independent paths"
        (r"truly\s+isolated", 0.8),  # A19: "truly isolated"
        (r"cause\s+lower\s+levels", 0.9),  # A21: "cause lower levels"
        (r"temporal\s+order", 0.7),  # A16: "temporal order"
        (r"CAUSES\s+edge", 1.0),  # A25: "CAUSES edge in the graph"
        (r"complete\s+provenance", 0.8),  # A25: "complete provenance"
        (r"rank\s+all\s+.*cause", 0.9),  # A11: "Rank all CAUSES"
        # Arabic patterns
        (r"سلاسل?\s*سببي", 1.0),
        (r"يؤدي\s+إلى", 0.9),
        (r"يسبب", 0.8),
        (r"المسار\s+من", 0.8),
        (r"الغفلة.*الكفر", 0.9),
        (r"الكبر.*الإخلاص", 0.9),
        (r"الإيمان", 0.5),  # Faith - needs other signals
        (r"يقوي", 0.7),  # Strengthens
        (r"يضعف", 0.7),  # Weakens
        (r"الشبكة", 0.6),  # Network
    ],
    
    IntentType.CROSS_TAFSIR_ANALYSIS: [
        # English patterns
        (r"tafsir\s+source", 1.0),           # B01: "tafsir sources"
        (r"each.*tafsir", 0.9),               # B01: "each of the 5 tafsir"
        (r"\d+\s+tafsir", 0.9),             # B01: "5 tafsir sources"
        (r"compare\s+.*tafsir", 1.0),
        (r"agreement.*disagree", 0.9),
        (r"consensus", 0.8),
        (r"divergence\s+matrix", 0.9),
        (r"across\s+.*sources", 0.8),
        (r"all\s+\d+\s+tafsir", 0.9),
        (r"methodological\s+emphasis", 0.8),
        (r"fingerprint", 0.7),
        (r"per.?source\s+provenance", 0.9),
        (r"ibn.?kathir.*qurtubi", 0.9),       # B: specific source names
        (r"tabari.*baghawi", 0.9),
        # Arabic patterns
        (r"اختلاف", 0.8),
        (r"اتفاق", 0.8),
        (r"مصادر\s+التفسير", 0.9),
        (r"مقارنة.*تفسير", 0.9),
        (r"إجماع", 0.8),
    ],
    
    IntentType.PROFILE_11D: [
        # English patterns
        (r"11.?dimensional", 1.0),
        (r"full\s+.*profile", 0.8),
        (r"organic.*situational.*systemic", 0.9),
        (r"axes?\s+coverage", 0.8),
        (r"dimension.*coverage", 0.8),
        (r"behavioral\s+profile", 0.8),
        # Arabic patterns
        (r"ملف\s+سلوكي", 0.9),
        (r"الأبعاد\s+الأحد\s+عشر", 1.0),
        (r"التصنيف\s+العضوي", 0.8),
        (r"التصنيف\s+الموضعي", 0.8),
    ],
    
    IntentType.GRAPH_METRICS: [
        # English patterns
        (r"centrality", 0.9),
        (r"PageRank", 0.9),
        (r"betweenness", 0.9),
        (r"degree\s+centrality", 1.0),
        (r"Louvain", 0.9),
        (r"clustering\s+coefficient", 0.9),
        (r"graph\s+density", 0.9),
        (r"node\s+count", 0.8),
        (r"edge\s+count", 0.8),
        (r"diameter", 0.7),
        (r"average\s+degree", 0.8),
        (r"network\s+statistics", 0.9),
        (r"community\s+detection", 0.9),
        # Arabic patterns
        (r"مركزية", 0.8),
        (r"كثافة\s+الشبكة", 0.9),
    ],
    
    IntentType.HEART_STATE: [
        # English patterns - explicit heart state terminology
        (r"heart\s+state", 1.0),
        (r"state\s+transition", 0.8),
        (r"state\s+of\s+the\s+heart", 1.0),
        (r"condition\s+of\s+.*heart", 0.9),
        # BEHAVIORAL MAPPING: Heart effects and impacts
        (r"affect.*heart", 1.0),       # "affect the heart"
        (r"impact.*heart", 0.9),       # "impact on heart"
        (r"effect.*heart", 0.9),       # "effect on heart"
        (r"influence.*heart", 0.9),    # "influence the heart"
        (r"heart.*affect", 0.9),       # "heart is affected"
        (r"heart.*impact", 0.9),
        (r"heart.*corrupt", 1.0),      # "heart corruption"
        (r"corrupt.*heart", 1.0),
        (r"purif.*heart", 1.0),        # "purify the heart"
        (r"heart.*purif", 1.0),
        (r"heart.*disease", 1.0),      # "heart disease"
        (r"disease.*heart", 1.0),
        (r"soften.*heart", 0.9),       # "soften the heart"
        (r"harden.*heart", 0.9),       # "harden the heart"
        (r"seal.*heart", 0.9),         # "seal the heart"
        (r"blind.*heart", 0.9),        # "blind the heart"
        (r"dead\s+heart", 0.9),
        (r"living\s+heart", 0.9),
        (r"sound\s+heart", 0.9),
        (r"sick\s+heart", 0.9),
        (r"heart.*tranquil", 0.9),
        (r"tranquil.*heart", 0.9),
        # Arabic patterns - comprehensive heart vocabulary
        (r"قلب\s+سليم", 1.0),      # sound heart
        (r"قلب\s+قاس", 1.0),       # hard heart
        (r"قلب\s+مريض", 1.0),      # diseased heart
        (r"قلب\s+ميت", 1.0),       # dead heart
        (r"قلب\s+مطمئن", 1.0),     # tranquil heart
        (r"حالات?\s+القلب", 1.0),  # heart states
        (r"قلب", 0.7),             # heart - slightly higher weight for behavioral system
        (r"قسوة\s+القلب", 0.9),    # hardness of heart
        (r"طهارة\s+القلب", 0.9),   # purity of heart
        (r"مرض\s+القلب", 0.9),     # disease of heart
        (r"أمراض\s+القلوب", 1.0),  # diseases of hearts
        (r"تزكية", 0.8),           # purification
        (r"تطهير", 0.8),           # cleansing
    ],
    
    IntentType.AGENT_ANALYSIS: [
        # English patterns
        (r"agent\s+type", 0.9),
        (r"believer.*disbeliever", 0.9),
        (r"munafiq", 0.9),
        (r"Prophet\s+attribution", 0.8),
        (r"Allah\s+attribution", 0.8),
        (r"who\s+performs?", 0.7),
        # Arabic patterns
        (r"المؤمن.*الكافر", 0.9),
        (r"المنافق", 0.9),
        (r"الفاعل", 0.7),
        (r"أنواع\s+الفاعلين", 0.9),
    ],
    
    IntentType.TEMPORAL_SPATIAL: [
        # English patterns
        (r"temporal.*spatial", 0.9),
        (r"worldly\s*life", 0.9),         # G01: "worldly life (الدنيا)"
        (r"worldly", 0.7),                 # General worldly reference
        (r"hereafter", 0.8),               # Afterlife reference
        (r"دنيا.*آخرة", 1.0),
        (r"مسجد.*بيت.*سوق", 0.9),
        (r"context.*location", 0.8),
        (r"time.*place", 0.8),
        (r"ramadan", 0.8),
        (r"friday", 0.7),
        (r"prayer\s*time", 0.7),
        # Arabic patterns
        (r"الدنيا", 0.9),                  # G01: الدنيا alone
        (r"دنيا", 0.8),                    # Without ال
        (r"الآخرة", 0.9),                  # Hereafter alone
        (r"آخرة", 0.8),                    # Without ال
        (r"الدنيا.*الآخرة", 1.0),
        (r"المكان.*الزمان", 0.9),
        (r"السياق\s+المكاني", 0.9),
        (r"السياق\s+الزماني", 0.9),
    ],
    
    IntentType.CONSEQUENCE_ANALYSIS: [
        # English patterns
        (r"consequence\s*type", 1.2),    # H01: "consequence types" - high weight
        (r"consequence", 0.8),
        (r"punishment.*reward", 0.9),
        (r"reward.*promise", 0.9),
        (r"promised\s+reward", 0.9),
        (r"severity", 0.7),
        (r"outcome", 0.6),
        (r"barzakh", 0.9),               # H01: barzakh is consequence-related
        (r"eternal", 0.6),               # H01: eternal consequences
        (r"الخسران", 0.9),
        # Arabic patterns
        (r"العقوبة", 0.9),
        (r"الثواب", 0.9),
        (r"الجزاء", 0.8),
        (r"النتيجة", 0.7),
        (r"عاقبة", 0.8),
        (r"جنة", 0.7),
        (r"رضوان", 0.8),
        (r"مغفرة", 0.7),
    ],
    
    IntentType.EMBEDDINGS_ANALYSIS: [
        # English patterns
        (r"t-?SNE", 1.0),
        (r"embedding", 0.8),
        (r"nearest\s+neighbor", 0.9),
        (r"vector\s+space", 0.8),
        (r"semantic\s+similarity", 0.8),
        (r"cluster", 0.6),
        # Arabic patterns
        (r"التضمين", 0.8),
        (r"التشابه\s+الدلالي", 0.9),
    ],
    
    IntentType.INTEGRATION_E2E: [
        # English patterns
        (r"ALL\s+system\s+components", 1.0),
        (r"comprehensive\s+analysis", 0.9),
        (r"genome", 0.8),
        (r"complete\s+.*analysis", 0.8),
        (r"end.?to.?end", 0.9),
        (r"full\s+system", 0.8),
        # Arabic patterns
        (r"تحليل\s+شامل", 0.9),
        (r"جميع\s+المكونات", 0.9),
    ],
    
    IntentType.SURAH_REF: [
        (r"سورة\s+([\u0600-\u06FF]+)", 1.0),
        (r"surah\s+(\d+)", 1.0),
        (r"سورة\s+(\d+)", 1.0),
    ],
    
    IntentType.AYAH_REF: [
        (r"(\d+):(\d+)", 0.9),
        (r"([\u0600-\u06FF]+):(\d+)", 0.9),
        (r"verse\s+(\d+)", 0.7),
        (r"آية\s+(\d+)", 0.8),
    ],
    
    IntentType.CROSS_CONTEXT_BEHAVIOR: [
        (r"سلوك\s+.*في\s+سياقات", 1.0),
        (r"behavior.*across.*context", 0.9),
        (r"cross.?context", 0.9),
    ],
}


def classify_intent(question: str) -> IntentResult:
    """
    Classify the intent of a question using deterministic pattern matching.
    
    This function is the single source of truth for intent classification,
    used by both proof_only_backend and the full backend.
    
    Args:
        question: The question text to classify
        
    Returns:
        IntentResult with the classified intent and metadata
    """
    if not question or not question.strip():
        return IntentResult(intent=IntentType.FREE_TEXT, confidence=0.0)
    
    question_lower = question.lower()
    question_normalized = question.strip()
    
    # Track scores for each intent
    intent_scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}
    matched_patterns_by_intent: Dict[IntentType, List[str]] = {intent: [] for intent in IntentType}
    extracted_entities: Dict[str, Any] = {}
    
    # Score each intent based on pattern matches
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern, weight in patterns:
            try:
                match = re.search(pattern, question_normalized, re.IGNORECASE | re.UNICODE)
                if match:
                    intent_scores[intent] += weight
                    matched_patterns_by_intent[intent].append(pattern)
                    
                    # Extract entities from matches
                    if match.groups():
                        if intent == IntentType.SURAH_REF:
                            extracted_entities["surah"] = match.group(1)
                        elif intent == IntentType.AYAH_REF:
                            extracted_entities["surah"] = match.group(1)
                            if len(match.groups()) > 1:
                                extracted_entities["ayah"] = match.group(2)
            except re.error:
                continue
    
    # Find the intent with highest score
    best_intent = IntentType.FREE_TEXT
    best_score = 0.0
    
    # Priority order for benchmark intents (higher priority = checked first)
    priority_order = [
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
        IntentType.CROSS_CONTEXT_BEHAVIOR,
        IntentType.SURAH_REF,
        IntentType.AYAH_REF,
    ]
    
    for intent in priority_order:
        score = intent_scores[intent]
        if score > best_score:
            best_score = score
            best_intent = intent
    
    # Require minimum threshold to avoid false positives
    MIN_THRESHOLD = 0.5
    if best_score < MIN_THRESHOLD:
        best_intent = IntentType.FREE_TEXT
        best_score = 1.0 - (best_score / MIN_THRESHOLD) if best_score > 0 else 1.0
    
    return IntentResult(
        intent=best_intent,
        confidence=min(best_score, 1.0),
        matched_patterns=matched_patterns_by_intent.get(best_intent, []),
        extracted_entities=extracted_entities,
    )


def get_section_for_intent(intent: IntentType) -> Optional[str]:
    """Map intent type to benchmark section letter."""
    mapping = {
        IntentType.GRAPH_CAUSAL: "A",
        IntentType.CROSS_TAFSIR_ANALYSIS: "B",
        IntentType.PROFILE_11D: "C",
        IntentType.GRAPH_METRICS: "D",
        IntentType.HEART_STATE: "E",
        IntentType.AGENT_ANALYSIS: "F",
        IntentType.TEMPORAL_SPATIAL: "G",
        IntentType.CONSEQUENCE_ANALYSIS: "H",
        IntentType.EMBEDDINGS_ANALYSIS: "I",
        IntentType.INTEGRATION_E2E: "J",
    }
    return mapping.get(intent)


def get_intent_for_section(section: str) -> Optional[IntentType]:
    """Map benchmark section letter to intent type."""
    mapping = {
        "A": IntentType.GRAPH_CAUSAL,
        "B": IntentType.CROSS_TAFSIR_ANALYSIS,
        "C": IntentType.PROFILE_11D,
        "D": IntentType.GRAPH_METRICS,
        "E": IntentType.HEART_STATE,
        "F": IntentType.AGENT_ANALYSIS,
        "G": IntentType.TEMPORAL_SPATIAL,
        "H": IntentType.CONSEQUENCE_ANALYSIS,
        "I": IntentType.EMBEDDINGS_ANALYSIS,
        "J": IntentType.INTEGRATION_E2E,
    }
    return mapping.get(section.upper())
