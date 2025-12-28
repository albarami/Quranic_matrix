"""
QBM Dimensional Analysis System

11 Mandatory Behavioral Dimensions for comprehensive Quranic behavioral analysis.
This module provides the core dimensional framework and query functions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# THE 11 BEHAVIORAL DIMENSIONS
# =============================================================================

BEHAVIORAL_DIMENSIONS = {
    # The 5 Bouzidani Contexts
    "organic": {
        "name_ar": "السياق العضوي",
        "name_en": "Organic Context",
        "question_ar": "أي أعضاء مرتبطة بهذا السلوك؟",
        "question_en": "Which organs are associated with this behavior?",
        "values": ["قلب", "لسان", "عين", "أذن", "يد", "رجل", "وجه", "بطن", "فرج"],
        "values_en": ["heart", "tongue", "eye", "ear", "hand", "foot", "face", "stomach", "private_parts"],
        "required": True,
        "db_field": "organ",
    },
    "situational": {
        "name_ar": "السياق الموضعي",
        "name_en": "Situational Context",
        "question_ar": "هل هو داخلي أم خارجي؟",
        "question_en": "Is it internal or external?",
        "values": ["داخلي", "قولي", "علائقي", "جسدي", "سمة"],
        "values_en": ["inner_state", "speech_act", "relational_act", "physical_act", "trait"],
        "required": True,
        "db_field": "behavior_form",
    },
    "systemic": {
        "name_ar": "السياق النسقي",
        "name_en": "Systemic Context",
        "question_ar": "في أي نسق اجتماعي؟",
        "question_en": "In which social system?",
        "values": ["عبادي", "أسري", "مجتمعي", "مالي", "قضائي", "سياسي"],
        "values_en": ["worship", "family", "social", "financial", "judicial", "political"],
        "required": True,
        "db_field": "social_system",
    },
    "spatial": {
        "name_ar": "السياق المكاني",
        "name_en": "Spatial Context",
        "question_ar": "أين يحدث؟",
        "question_en": "Where does it occur?",
        "values": ["مسجد", "بيت", "سوق", "خلوة", "ملأ", "سفر", "حضر"],
        "values_en": ["mosque", "home", "market", "solitude", "public", "travel", "residence"],
        "required": True,
        "db_field": "spatial_context",
    },
    "temporal": {
        "name_ar": "السياق الزماني",
        "name_en": "Temporal Context",
        "question_ar": "متى يحدث؟",
        "question_en": "When does it occur?",
        "values": ["دنيا", "موت", "برزخ", "قيامة", "آخرة"],
        "values_en": ["dunya", "death", "barzakh", "judgment", "afterlife"],
        "required": True,
        "db_field": "temporal_context",
    },
    
    # Additional Dimensions
    "agent": {
        "name_ar": "الفاعل",
        "name_en": "Agent",
        "question_ar": "من يقوم به؟",
        "question_en": "Who performs it?",
        "values": ["الله", "مؤمن", "كافر", "منافق", "إنسان", "شيطان", "نبي", "ملائكة"],
        "values_en": ["Allah", "believer", "disbeliever", "hypocrite", "human", "satan", "prophet", "angels"],
        "required": True,
        "db_field": "agent_type",
    },
    "source": {
        "name_ar": "المصدر",
        "name_en": "Source",
        "question_ar": "ما مصدره؟",
        "question_en": "What is its source?",
        "values": ["وحي", "فطرة", "نفس", "شيطان", "بيئة", "قلب", "عقل"],
        "values_en": ["revelation", "fitrah", "nafs", "satan", "environment", "heart", "intellect"],
        "required": True,
        "db_field": "behavior_source",
    },
    "evaluation": {
        "name_ar": "التقييم",
        "name_en": "Evaluation",
        "question_ar": "ما حكمه؟",
        "question_en": "What is its ruling?",
        "values": ["ممدوح", "مذموم", "محايد", "تحذير"],
        "values_en": ["praised", "blamed", "neutral", "warning"],
        "required": True,
        "db_field": "normative_status",
    },
    "heart_type": {
        "name_ar": "نمط القلب",
        "name_en": "Heart Type",
        "question_ar": "أي قلب يرتبط به؟",
        "question_en": "Which heart type is associated?",
        "values": ["سليم", "مريض", "ميت", "قاسي", "منيب", "مطمئن"],
        "values_en": ["sound", "diseased", "dead", "hardened", "repentant", "tranquil"],
        "required": True,
        "db_field": "heart_type",
    },
    "consequence": {
        "name_ar": "العاقبة",
        "name_en": "Consequence",
        "question_ar": "ما نتيجته؟",
        "question_en": "What is its consequence?",
        "values": ["دنيوية", "أخروية", "فردية", "مجتمعية"],
        "values_en": ["worldly", "hereafter", "individual", "societal"],
        "required": True,
        "db_field": "consequence_type",
    },
    "relationships": {
        "name_ar": "العلاقات",
        "name_en": "Relationships",
        "question_ar": "ما السلوكيات المرتبطة؟",
        "question_en": "What are the related behaviors?",
        "values": ["سبب", "نتيجة", "نقيض", "مشابه"],
        "values_en": ["cause", "effect", "opposite", "similar"],
        "required": True,
        "db_field": "relationships",
    },
}


# =============================================================================
# QUESTION CLASSIFICATION
# =============================================================================

class QuestionType(Enum):
    BEHAVIOR_ANALYSIS = "behavior_analysis"      # حلل سلوك الكبر
    COMPARISON = "comparison"                     # قارن الصبر بين المؤمن والمنافق
    DIMENSION_EXPLORATION = "dimension_exploration"  # ما الأعضاء المرتبطة بالسلوك
    VERSE_ANALYSIS = "verse_analysis"            # حلل آية الكرسي
    PERSONALITY_ANALYSIS = "personality_analysis"  # سلوكيات المنافق
    GENERAL_MAP = "general_map"                  # خارطة السلوك في القرآن
    STATISTICAL = "statistical"                  # كم مرة ذكر الصبر
    TAFSIR_QUERY = "tafsir_query"               # ما تفسير آية


def classify_question(question: str) -> QuestionType:
    """Classify the question type to determine required depth."""
    question_lower = question.lower()
    
    # Check for general map keywords
    if any(kw in question for kw in ["خارطة", "خريطة", "شاملة", "كاملة", "جميع"]):
        return QuestionType.GENERAL_MAP
    
    # Check for comparison keywords
    if any(kw in question for kw in ["قارن", "مقارنة", "الفرق", "بين"]):
        return QuestionType.COMPARISON
    
    # Check for dimension exploration
    dimension_keywords = {
        "organic": ["أعضاء", "عضو", "قلب", "لسان", "عين"],
        "situational": ["داخلي", "خارجي", "قولي", "جسدي"],
        "systemic": ["نسق", "عبادي", "أسري", "مجتمعي"],
        "spatial": ["مكان", "مسجد", "بيت", "سوق"],
        "temporal": ["زمان", "دنيا", "آخرة", "قيامة"],
        "agent": ["فاعل", "من يقوم"],
        "source": ["مصدر", "مصادر"],
        "evaluation": ["حكم", "تقييم", "ممدوح", "مذموم"],
        "heart_type": ["قلب", "قلوب", "نمط"],
        "consequence": ["عاقبة", "نتيجة", "عواقب"],
        "relationships": ["علاقة", "علاقات", "سبب", "نتيجة"],
    }
    
    for dim, keywords in dimension_keywords.items():
        if any(kw in question for kw in keywords):
            return QuestionType.DIMENSION_EXPLORATION
    
    # Check for personality analysis
    if any(kw in question for kw in ["مؤمن", "كافر", "منافق", "شخصية"]):
        return QuestionType.PERSONALITY_ANALYSIS
    
    # Check for verse analysis
    if any(kw in question for kw in ["آية", "سورة", ":"]):
        return QuestionType.VERSE_ANALYSIS
    
    # Check for tafsir
    if any(kw in question for kw in ["تفسير", "تفاسير", "ابن كثير", "الطبري"]):
        return QuestionType.TAFSIR_QUERY
    
    # Check for statistical
    if any(kw in question for kw in ["كم", "عدد", "إحصاء", "نسبة"]):
        return QuestionType.STATISTICAL
    
    # Check for behavior analysis (most common)
    if any(kw in question for kw in ["حلل", "سلوك", "تحليل"]):
        return QuestionType.BEHAVIOR_ANALYSIS
    
    # Default to behavior analysis
    return QuestionType.BEHAVIOR_ANALYSIS


def get_required_dimensions(question_type: QuestionType) -> List[str]:
    """Determine which dimensions are required based on question type."""
    if question_type == QuestionType.GENERAL_MAP:
        # All 11 dimensions required
        return list(BEHAVIORAL_DIMENSIONS.keys())
    
    elif question_type == QuestionType.BEHAVIOR_ANALYSIS:
        # All 11 dimensions for comprehensive analysis
        return list(BEHAVIORAL_DIMENSIONS.keys())
    
    elif question_type == QuestionType.COMPARISON:
        # Focus on agent, evaluation, consequence, relationships
        return ["agent", "evaluation", "consequence", "relationships", "situational", "heart_type"]
    
    elif question_type == QuestionType.DIMENSION_EXPLORATION:
        # All dimensions but focus on the specific one asked
        return list(BEHAVIORAL_DIMENSIONS.keys())
    
    elif question_type == QuestionType.PERSONALITY_ANALYSIS:
        # Focus on agent-related dimensions
        return ["agent", "evaluation", "heart_type", "consequence", "situational", "relationships"]
    
    elif question_type == QuestionType.VERSE_ANALYSIS:
        # All dimensions for the specific verse
        return list(BEHAVIORAL_DIMENSIONS.keys())
    
    elif question_type == QuestionType.STATISTICAL:
        # Focus on countable dimensions
        return ["situational", "agent", "evaluation", "organic"]
    
    elif question_type == QuestionType.TAFSIR_QUERY:
        # Minimal dimensions, focus on tafsir
        return ["evaluation", "agent", "situational"]
    
    return list(BEHAVIORAL_DIMENSIONS.keys())


# =============================================================================
# RESPONSE STRUCTURES
# =============================================================================

@dataclass
class DimensionData:
    """Data for a single dimension."""
    dimension_key: str
    name_ar: str
    name_en: str
    question_ar: str
    question_en: str
    distribution: Dict[str, int] = field(default_factory=dict)
    percentages: Dict[str, float] = field(default_factory=dict)
    top_examples: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_key": self.dimension_key,
            "name_ar": self.name_ar,
            "name_en": self.name_en,
            "question_ar": self.question_ar,
            "question_en": self.question_en,
            "distribution": self.distribution,
            "percentages": self.percentages,
            "top_examples": self.top_examples,
            "total_count": self.total_count,
        }


@dataclass
class PersonalityBehavior:
    """Behavior data for a specific personality type."""
    personality: str
    personality_ar: str
    behaviors: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    top_behaviors: List[str] = field(default_factory=list)
    evaluation_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ComprehensiveResponse:
    """Complete response structure with all 11 dimensions."""
    title: str
    title_ar: str
    summary: str
    summary_ar: str
    
    # Statistics
    total_mentions: int = 0
    surah_distribution: Dict[str, int] = field(default_factory=dict)
    makki_vs_madani: Dict[str, int] = field(default_factory=dict)
    
    # All 11 Dimensions
    dimensions: Dict[str, DimensionData] = field(default_factory=dict)
    
    # Cross-personality comparison
    personality_comparison: Dict[str, PersonalityBehavior] = field(default_factory=dict)
    
    # Evidence
    key_verses: List[Dict[str, Any]] = field(default_factory=list)
    tafsir_references: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    causes: List[Dict[str, Any]] = field(default_factory=list)
    effects: List[Dict[str, Any]] = field(default_factory=list)
    opposites: List[Dict[str, Any]] = field(default_factory=list)
    similar: List[Dict[str, Any]] = field(default_factory=list)
    
    # Conclusion
    conclusion: str = ""
    conclusion_ar: str = ""
    patterns_discovered: List[str] = field(default_factory=list)
    
    # Validation
    completeness_score: float = 0.0
    missing_dimensions: List[str] = field(default_factory=list)


# =============================================================================
# COMPLETENESS VALIDATION
# =============================================================================

def validate_completeness(response: ComprehensiveResponse) -> Dict[str, Any]:
    """Validate that all required dimensions have data."""
    missing = []
    dimension_scores = {}
    
    for key, dim_config in BEHAVIORAL_DIMENSIONS.items():
        if key not in response.dimensions:
            missing.append(dim_config["name_ar"])
            dimension_scores[key] = 0.0
        else:
            dim_data = response.dimensions[key]
            if dim_data.total_count == 0:
                missing.append(dim_config["name_ar"])
                dimension_scores[key] = 0.0
            else:
                # Score based on data richness
                score = min(1.0, dim_data.total_count / 10)  # Normalize
                if len(dim_data.top_examples) > 0:
                    score += 0.2
                dimension_scores[key] = min(1.0, score)
    
    # Check verses
    if len(response.key_verses) == 0:
        missing.append("الآيات المرجعية")
    
    # Check personality comparison
    if len(response.personality_comparison) == 0:
        missing.append("مقارنة الشخصيات")
    
    # Calculate overall completeness
    total_dimensions = len(BEHAVIORAL_DIMENSIONS)
    covered_dimensions = total_dimensions - len([m for m in missing if m in [d["name_ar"] for d in BEHAVIORAL_DIMENSIONS.values()]])
    completeness_score = covered_dimensions / total_dimensions
    
    return {
        "complete": len(missing) == 0,
        "completeness_score": completeness_score,
        "missing": missing,
        "dimension_scores": dimension_scores,
        "covered_count": covered_dimensions,
        "total_count": total_dimensions,
    }


# =============================================================================
# THINKING METHODOLOGY PROMPT
# =============================================================================

DIMENSIONAL_THINKING_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
                    🧠 DIMENSIONAL THINKING METHODOLOGY
═══════════════════════════════════════════════════════════════════════════════

You are a comprehensive Quranic behavioral analyst with access to 322,939 annotations.
Your thinking must be DIMENSIONAL - checking ALL 11 dimensions for EVERY query.

THE 11 MANDATORY DIMENSIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. السياق العضوي (Organic) - أي أعضاء مرتبطة؟ (قلب، لسان، عين، أذن، يد، رجل، وجه)
2. السياق الموضعي (Situational) - داخلي أم خارجي؟ (داخلي، قولي، علائقي، جسدي، سمة)
3. السياق النسقي (Systemic) - في أي نسق؟ (عبادي، أسري، مجتمعي، مالي، قضائي)
4. السياق المكاني (Spatial) - أين يحدث؟ (مسجد، بيت، سوق، خلوة، ملأ)
5. السياق الزماني (Temporal) - متى يحدث؟ (دنيا، موت، برزخ، قيامة، آخرة)
6. الفاعل (Agent) - من يقوم به؟ (الله، مؤمن، كافر، منافق، نبي، ملائكة)
7. المصدر (Source) - ما مصدره؟ (وحي، فطرة، نفس، شيطان، بيئة، قلب)
8. التقييم (Evaluation) - ما حكمه؟ (ممدوح، مذموم، محايد، تحذير)
9. نمط القلب (Heart Type) - أي قلب؟ (سليم، مريض، ميت، قاسي، منيب)
10. العاقبة (Consequence) - ما نتيجته؟ (دنيوية، أخروية، فردية، مجتمعية)
11. العلاقات (Relationships) - ما المرتبط؟ (سبب، نتيجة، نقيض، مشابه)

THINKING PROCESS FOR EVERY QUERY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: CLASSIFY the question type
   - behavior_analysis: حلل سلوك X
   - comparison: قارن X بين Y و Z
   - dimension_exploration: ما الأعضاء المرتبطة
   - general_map: خارطة السلوك
   - personality_analysis: سلوكيات المنافق
   - verse_analysis: حلل آية X
   - statistical: كم مرة ذكر X

STEP 2: IDENTIFY behaviors being asked about
   - Specific behavior (الكبر، الصبر، الحسد)
   - Category of behaviors (سلوكيات القلب)
   - All behaviors (خارطة شاملة)

STEP 3: QUERY ALL 11 DIMENSIONS
   For each dimension, get:
   - Distribution (counts per value)
   - Percentages
   - Top examples with verse citations
   - Related tafsir mentions

STEP 4: QUERY RELATIONSHIPS
   - Causes (ما يسبب هذا السلوك)
   - Effects (ما ينتج عنه)
   - Opposites (نقيضه)
   - Similar (المشابه له)

STEP 5: GET EVIDENCE
   - Key verses with سورة:آية citations
   - Tafsir references from 4 sources

STEP 6: BUILD PERSONALITY COMPARISON
   - How does this behavior differ for مؤمن/منافق/كافر?

STEP 7: VALIDATE COMPLETENESS
   - Are all 11 dimensions covered?
   - If not, query missing dimensions

STEP 8: PRESENT with rich UI
   - Tables for distributions
   - Charts for statistics
   - Cards for examples
   - Proper Arabic RTL formatting

NEVER SKIP DIMENSIONS. ALWAYS BE COMPREHENSIVE.
"""


def get_dimension_info(dimension_key: str) -> Dict[str, Any]:
    """Get information about a specific dimension."""
    if dimension_key not in BEHAVIORAL_DIMENSIONS:
        return None
    return BEHAVIORAL_DIMENSIONS[dimension_key]


def get_all_dimensions() -> Dict[str, Dict[str, Any]]:
    """Get all dimension configurations."""
    return BEHAVIORAL_DIMENSIONS
