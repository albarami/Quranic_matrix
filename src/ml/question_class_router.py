"""
Canonical Question Class Router for QBM

This module provides the SINGLE source of truth for question classification.
It wraps and unifies:
- intent_classifier.py (10 benchmark intents + standard intents)
- legendary_planner.py (25 question classes)

RULES:
- No benchmark-ID matching. Only semantic patterns + canonical entity extraction.
- Must work for similar phrasing in Arabic/English.
- Analytical questions must NOT route to FREE_TEXT.
- Returns QuestionClass for planner routing.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set

from src.ml.intent_classifier import classify_intent, IntentType, IntentResult
from src.ml.legendary_planner import QuestionClass, QUESTION_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result of question classification routing."""
    question_class: QuestionClass
    intent_type: IntentType
    confidence: float = 1.0
    matched_patterns: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    routing_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_class": self.question_class.value,
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "extracted_entities": self.extracted_entities,
            "routing_reason": self.routing_reason,
        }


# Mapping from IntentType to QuestionClass
INTENT_TO_QUESTION_CLASS: Dict[IntentType, QuestionClass] = {
    IntentType.GRAPH_CAUSAL: QuestionClass.CAUSAL_CHAIN,
    IntentType.CROSS_TAFSIR_ANALYSIS: QuestionClass.CROSS_TAFSIR_COMPARATIVE,
    IntentType.PROFILE_11D: QuestionClass.BEHAVIOR_PROFILE_11AXIS,
    IntentType.GRAPH_METRICS: QuestionClass.NETWORK_CENTRALITY,
    IntentType.HEART_STATE: QuestionClass.STATE_TRANSITION,
    IntentType.AGENT_ANALYSIS: QuestionClass.AGENT_ATTRIBUTION,
    IntentType.TEMPORAL_SPATIAL: QuestionClass.TEMPORAL_MAPPING,
    IntentType.CONSEQUENCE_ANALYSIS: QuestionClass.COMPLETE_ANALYSIS,  # Maps to complete for now
    IntentType.EMBEDDINGS_ANALYSIS: QuestionClass.SEMANTIC_LANDSCAPE,
    IntentType.INTEGRATION_E2E: QuestionClass.COMPLETE_ANALYSIS,
    IntentType.SURAH_REF: QuestionClass.FREE_TEXT,  # Handled by deterministic retrieval
    IntentType.AYAH_REF: QuestionClass.FREE_TEXT,   # Handled by deterministic retrieval
    IntentType.CONCEPT_REF: QuestionClass.BEHAVIOR_PROFILE_11AXIS,  # Concept queries get profile
    IntentType.CROSS_CONTEXT_BEHAVIOR: QuestionClass.COMPLETE_ANALYSIS,
    IntentType.FREE_TEXT: QuestionClass.FREE_TEXT,
}

# Analytical intents that should NEVER route to FREE_TEXT
ANALYTICAL_INTENTS: Set[IntentType] = {
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

# Additional Arabic patterns for better routing (extends intent_classifier patterns)
ADDITIONAL_ARABIC_PATTERNS: Dict[QuestionClass, List[str]] = {
    QuestionClass.CAUSAL_CHAIN: [
        r"ما\s+الذي\s+يؤدي",
        r"كيف\s+يؤدي",
        r"سلسلة.*سلوك",
        r"طريق.*الهلاك",
        r"طريق.*النجاة",
    ],
    QuestionClass.CROSS_TAFSIR_COMPARATIVE: [
        r"ما\s+رأي\s+المفسرين",
        r"كيف\s+فسر",
        r"اختلف.*المفسرون",
        r"اتفق.*المفسرون",
    ],
    QuestionClass.BEHAVIOR_PROFILE_11AXIS: [
        r"حلل\s+سلوك",
        r"ما\s+هو\s+سلوك",
        r"صف\s+سلوك",
        r"اشرح\s+سلوك",
        r"الكبر|التواضع|الحسد|الغيبة|الصبر|الشكر|التوكل|الإخلاص",
    ],
    QuestionClass.STATE_TRANSITION: [
        r"حالة\s+القلب",
        r"قلب.*سليم|قاسي|مريض|ميت",
        r"تحول.*القلب",
    ],
    QuestionClass.AGENT_ATTRIBUTION: [
        r"من\s+يفعل",
        r"المؤمن.*الكافر",
        r"المنافق",
        r"الفاعل",
    ],
    QuestionClass.NETWORK_CENTRALITY: [
        r"أهم\s+سلوك",
        r"السلوك\s+المركزي",
        r"ترتيب.*السلوكيات",
    ],
    QuestionClass.TEMPORAL_MAPPING: [
        r"متى",
        r"الدنيا.*الآخرة",
        r"الزمان",
    ],
    QuestionClass.SPATIAL_MAPPING: [
        r"أين",
        r"المكان",
        r"المسجد|البيت|السوق",
    ],
    QuestionClass.COMPLETE_ANALYSIS: [
        r"تحليل\s+شامل",
        r"كل\s+ما\s+يتعلق",
        r"جميع\s+جوانب",
    ],
}


def route_question(question: str) -> RouterResult:
    """
    Route a question to the appropriate QuestionClass for planner execution.
    
    This is the SINGLE entry point for question classification.
    It combines intent_classifier patterns with legendary_planner patterns.
    
    Args:
        question: The question text (Arabic or English)
        
    Returns:
        RouterResult with question_class, intent_type, and metadata
    """
    if not question or not question.strip():
        return RouterResult(
            question_class=QuestionClass.FREE_TEXT,
            intent_type=IntentType.FREE_TEXT,
            confidence=0.0,
            routing_reason="empty_question",
        )
    
    # Step 1: Use intent_classifier for initial classification
    intent_result = classify_intent(question)
    
    # Step 2: Check additional Arabic patterns for better coverage
    additional_matches = _check_additional_patterns(question)
    
    # Step 3: Determine final question class
    question_class = QuestionClass.FREE_TEXT
    routing_reason = "default_free_text"
    matched_patterns = intent_result.matched_patterns.copy()
    
    # If intent_classifier found an analytical intent, use it
    if intent_result.intent in ANALYTICAL_INTENTS:
        question_class = INTENT_TO_QUESTION_CLASS.get(
            intent_result.intent, QuestionClass.FREE_TEXT
        )
        routing_reason = f"intent_classifier:{intent_result.intent.value}"
    
    # If additional patterns found a stronger match, override
    elif additional_matches:
        best_match = max(additional_matches, key=lambda x: x[1])
        question_class = best_match[0]
        matched_patterns.extend(best_match[2])
        routing_reason = f"additional_patterns:{question_class.value}"
    
    # If intent_classifier found SURAH_REF or AYAH_REF, keep as FREE_TEXT
    # but mark for deterministic retrieval
    elif intent_result.intent in {IntentType.SURAH_REF, IntentType.AYAH_REF}:
        question_class = QuestionClass.FREE_TEXT
        routing_reason = f"deterministic_retrieval:{intent_result.intent.value}"
    
    # If intent_classifier found CONCEPT_REF, route to profile
    elif intent_result.intent == IntentType.CONCEPT_REF:
        question_class = QuestionClass.BEHAVIOR_PROFILE_11AXIS
        routing_reason = "concept_ref_to_profile"
    
    # Final fallback: use intent mapping
    else:
        question_class = INTENT_TO_QUESTION_CLASS.get(
            intent_result.intent, QuestionClass.FREE_TEXT
        )
        routing_reason = f"intent_mapping:{intent_result.intent.value}"
    
    # Log routing decision
    logger.info(
        f"[ROUTER] Question routed: class={question_class.value}, "
        f"intent={intent_result.intent.value}, reason={routing_reason}"
    )
    
    return RouterResult(
        question_class=question_class,
        intent_type=intent_result.intent,
        confidence=intent_result.confidence,
        matched_patterns=matched_patterns,
        extracted_entities=intent_result.extracted_entities,
        routing_reason=routing_reason,
    )


def _check_additional_patterns(question: str) -> List[tuple]:
    """
    Check additional Arabic patterns not covered by intent_classifier.
    
    Returns:
        List of (QuestionClass, score, matched_patterns) tuples
    """
    matches = []
    
    for qclass, patterns in ADDITIONAL_ARABIC_PATTERNS.items():
        score = 0.0
        matched = []
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE | re.UNICODE):
                score += 0.5
                matched.append(pattern)
        
        if score > 0:
            matches.append((qclass, score, matched))
    
    return matches


def is_analytical_question(question: str) -> bool:
    """
    Check if a question is analytical (should NOT route to FREE_TEXT + vector search).
    
    Args:
        question: The question text
        
    Returns:
        True if the question requires analytical processing
    """
    result = route_question(question)
    return result.question_class != QuestionClass.FREE_TEXT


def get_planner_for_question(question: str) -> str:
    """
    Get the planner name for a question.
    
    Args:
        question: The question text
        
    Returns:
        Planner name string (e.g., "causal_chain", "cross_tafsir_comparative")
    """
    result = route_question(question)
    return result.question_class.value
