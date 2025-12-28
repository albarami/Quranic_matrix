"""
QBM Question-Agnostic Analysis System

This module implements generic query functions that work for ANY question with equal depth.
The intelligence is in the METHODOLOGY, not in pre-built answers.

Core Principle: Every function accepts generic parameters, no hardcoded values.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import json


# =============================================================================
# TEXT NORMALIZATION + VOCAB HELPERS
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VOCAB_CACHE: Dict[str, Any] = {}

# Arabic diacritics + Quranic annotation marks
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_TATWEEL_RE = re.compile(r"\u0640")
_ARABIC_ALIF_VARIANTS_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _ARABIC_TATWEEL_RE.sub("", text)
    text = _ARABIC_ALIF_VARIANTS_RE.sub("ا", text)
    # Common normalizations to improve plain-text matching
    text = text.replace("ى", "ي")
    return text.lower()


def _load_vocab_json(vocab_name: str) -> Dict[str, Any]:
    if vocab_name in _VOCAB_CACHE:
        return _VOCAB_CACHE[vocab_name]

    vocab_path = _REPO_ROOT / "vocab" / f"{vocab_name}.json"
    if not vocab_path.exists():
        _VOCAB_CACHE[vocab_name] = {}
        return _VOCAB_CACHE[vocab_name]

    try:
        _VOCAB_CACHE[vocab_name] = json.loads(vocab_path.read_text(encoding="utf-8"))
    except Exception:
        _VOCAB_CACHE[vocab_name] = {}
    return _VOCAB_CACHE[vocab_name]


def _load_vocab_items(vocab_name: str) -> List[Dict[str, Any]]:
    data = _load_vocab_json(vocab_name)
    items = data.get("items")
    return items if isinstance(items, list) else []


def _vocab_items_by_id(vocab_name: str) -> Dict[str, Dict[str, Any]]:
    cache_key = f"{vocab_name}::by_id"
    if cache_key in _VOCAB_CACHE:
        return _VOCAB_CACHE[cache_key]

    by_id: Dict[str, Dict[str, Any]] = {}
    for item in _load_vocab_items(vocab_name):
        item_id = item.get("id")
        if item_id:
            by_id[item_id] = item
    _VOCAB_CACHE[cache_key] = by_id
    return by_id


def _infer_spatial_from_text(text_ar: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer spatial label from verse text using controlled vocab quranic_terms.
    Returns (spatial_id, matched_term).
    """
    text_norm = normalize_arabic(text_ar)
    if not text_norm:
        return None, None

    items = _load_vocab_items("spatial")
    # Prefer more specific contexts first
    priority = {"LOC_MASJID": 0, "LOC_HOME": 1, "LOC_MARKET": 2, "LOC_BATTLEFIELD": 3, "LOC_TRAVEL": 4}
    items = sorted(items, key=lambda it: priority.get(it.get("id", ""), 99))

    for item in items:
        item_id = item.get("id")
        terms = []
        if isinstance(item.get("quranic_terms"), list):
            terms.extend([t for t in item["quranic_terms"] if isinstance(t, str)])
        if isinstance(item.get("ar"), str) and item["ar"]:
            terms.append(item["ar"])

        for term in terms:
            term_norm = normalize_arabic(term)
            if term_norm and term_norm in text_norm:
                return item_id, term
    return None, None


_TEMPORAL_KEYWORDS = {
    "دنيا": ["الدنيا", "حياة الدنيا", "الحياة الدنيا"],
    "عند_الموت": ["موت", "الموت", "يموت", "توفى", "يتوفى", "توفاكم", "وفاة"],
    "برزخ": ["برزخ", "البرزخ"],
    "قيامة": ["القيامة", "يوم القيامة", "يوم الدين", "يوم الحساب", "الساعة"],
    "آخرة": ["الاخرة", "الآخرة", "يوم الاخر", "يوم الآخر", "الدار الآخرة", "الدار الاخرة"],
}


def _infer_temporal_from_text(text_ar: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer temporal stage from verse text.
    Returns (temporal_key, matched_term).
    """
    text_norm = normalize_arabic(text_ar)
    if not text_norm:
        return None, None

    for temporal_key, raw_terms in _TEMPORAL_KEYWORDS.items():
        for term in raw_terms:
            term_norm = normalize_arabic(term)
            if term_norm and term_norm in text_norm:
                return temporal_key, term
    return None, None


# =============================================================================
# QUESTION TYPES
# =============================================================================

class QuestionType(Enum):
    BEHAVIOR_ANALYSIS = "behavior_analysis"           # حلل سلوك الكبر
    DIMENSION_EXPLORATION = "dimension_exploration"   # ما الأعضاء في القرآن؟
    COMPARISON = "comparison"                         # قارن بين X و Y
    JOURNEY_CHAIN = "journey_chain"                   # رحلة من X إلى Y
    VERSE_ANALYSIS = "verse_analysis"                 # حلل البقرة 10
    SURAH_ANALYSIS = "surah_analysis"                 # السلوكيات في سورة البقرة
    STATISTICAL = "statistical"                       # كم مرة ذُكر الصبر؟
    RELATIONSHIP = "relationship"                     # ما علاقة X بـ Y؟
    GENERAL_MAP = "general_map"                       # خارطة السلوك في القرآن


# =============================================================================
# THE 11 DIMENSIONS - QUERYABLE FOR ANYTHING
# =============================================================================

DIMENSIONS = {
    "organic": {
        "name_ar": "السياق العضوي",
        "name_en": "Organic Context",
        "db_column": "organ",
        "question_ar": "أي أعضاء مرتبطة؟",
        "values": None,  # Query from DB, don't hardcode
    },
    "situational": {
        "name_ar": "السياق الموضعي",
        "name_en": "Situational Context",
        "db_column": "behavior_form",
        "question_ar": "داخلي أم خارجي؟",
        "values": None,
    },
    "systemic": {
        "name_ar": "السياق النسقي",
        "name_en": "Systemic Context",
        "db_column": "social_system",
        "question_ar": "في أي نسق اجتماعي؟",
        "values": None,
    },
    "spatial": {
        "name_ar": "السياق المكاني",
        "name_en": "Spatial Context",
        "db_column": "spatial_context",
        "question_ar": "أين يحدث؟",
        "values": None,
    },
    "temporal": {
        "name_ar": "السياق الزماني",
        "name_en": "Temporal Context",
        "db_column": "temporal_context",
        "question_ar": "متى يحدث؟",
        "values": None,
    },
    "agent": {
        "name_ar": "الفاعل",
        "name_en": "Agent",
        "db_column": "agent_type",
        "question_ar": "من يقوم به؟",
        "values": None,
    },
    "source": {
        "name_ar": "المصدر",
        "name_en": "Source",
        "db_column": "behavior_source",
        "question_ar": "ما مصدره؟",
        "values": None,
    },
    "evaluation": {
        "name_ar": "التقييم",
        "name_en": "Evaluation",
        "db_column": "normative_status",
        "question_ar": "ما حكمه؟",
        "values": None,
    },
    "heart_type": {
        "name_ar": "نمط القلب",
        "name_en": "Heart Type",
        "db_column": "heart_type",
        "question_ar": "أي قلب يرتبط به؟",
        "values": None,
    },
    "consequence": {
        "name_ar": "العاقبة",
        "name_en": "Consequence",
        "db_column": "consequence_type",
        "question_ar": "ما نتيجته؟",
        "values": None,
    },
    "relationships": {
        "name_ar": "العلاقات",
        "name_en": "Relationships",
        "db_column": None,  # From graph/relationships, not table column
        "question_ar": "ما السلوكيات المرتبطة؟",
        "values": None,
    },
}

ALL_DIMENSIONS = list(DIMENSIONS.keys())


# =============================================================================
# PARSED QUESTION STRUCTURE
# =============================================================================

@dataclass
class ParsedQuestion:
    """Structured representation of a parsed question."""
    type: QuestionType
    entities: List[str] = field(default_factory=list)           # What's being asked about
    dimensions_requested: List[str] = field(default_factory=list)  # Which dimensions to focus on
    comparison_targets: List[str] = field(default_factory=list)  # For comparisons
    filter: Dict[str, Any] = field(default_factory=dict)         # Any constraints
    journey_start: Optional[str] = None
    journey_end: Optional[str] = None
    surah: Optional[int] = None
    ayah: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "entities": self.entities,
            "dimensions_requested": self.dimensions_requested,
            "comparison_targets": self.comparison_targets,
            "filter": self.filter,
            "journey_start": self.journey_start,
            "journey_end": self.journey_end,
            "surah": self.surah,
            "ayah": self.ayah,
        }


# =============================================================================
# QUESTION PARSER - UNDERSTANDS ANY QUESTION TYPE
# =============================================================================

def extract_behavior(question: str) -> Optional[str]:
    """Extract behavior name from question."""
    # Common behavior patterns
    patterns = [
        r"سلوك\s+(\w+)",
        r"حلل\s+(\w+)",
        r"تحليل\s+(\w+)",
        r"(\w+)\s+في\s+القرآن",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1)
    return None


def extract_dimension(question: str) -> Optional[str]:
    """Extract dimension being asked about."""
    dimension_keywords = {
        "organic": ["أعضاء", "عضو", "الأعضاء"],
        "situational": ["داخلي", "خارجي", "موضعي"],
        "systemic": ["نسق", "نسقي", "اجتماعي"],
        "spatial": ["مكان", "مكاني", "أين"],
        "temporal": ["زمان", "زماني", "متى"],
        "agent": ["فاعل", "فاعلون", "من يقوم"],
        "source": ["مصدر", "مصادر"],
        "evaluation": ["حكم", "تقييم", "أحكام"],
        "heart_type": ["قلب", "قلوب", "نمط القلب"],
        "consequence": ["عاقبة", "عواقب", "نتيجة", "نتائج"],
        "relationships": ["علاقة", "علاقات", "مرتبط"],
    }
    
    for dim, keywords in dimension_keywords.items():
        if any(kw in question for kw in keywords):
            return dim
    return None


def extract_comparison_targets(question: str) -> List[str]:
    """Extract entities being compared."""
    targets = []
    
    # Pattern: "قارن بين X و Y"
    match = re.search(r"بين\s+(\w+)\s+و\s*(\w+)", question)
    if match:
        targets.extend([match.group(1), match.group(2)])
        return targets
    
    # Pattern: "X و Y و Z"
    match = re.search(r"(\w+)\s+و\s*(\w+)(?:\s+و\s*(\w+))?", question)
    if match:
        targets = [g for g in match.groups() if g]
        return targets
    
    # Pattern: "المؤمن والكافر"
    personalities = ["المؤمن", "الكافر", "المنافق", "مؤمن", "كافر", "منافق"]
    for p in personalities:
        if p in question:
            targets.append(p.replace("ال", ""))
    
    return targets


def extract_journey_endpoints(question: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract start and end points of a journey."""
    # Pattern: "من X إلى Y"
    match = re.search(r"من\s+(\w+(?:\s+\w+)?)\s+إلى\s+(\w+(?:\s+\w+)?)", question)
    if match:
        return match.group(1), match.group(2)
    
    # Pattern: "رحلة X"
    match = re.search(r"رحلة\s+(\w+)", question)
    if match:
        return match.group(1), None
    
    return None, None


def extract_surah_ayah(question: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract surah and ayah numbers from question."""
    # Pattern: "سورة البقرة" or "البقرة"
    surah_names = {
        "الفاتحة": 1, "البقرة": 2, "آل عمران": 3, "النساء": 4, "المائدة": 5,
        "الأنعام": 6, "الأعراف": 7, "الأنفال": 8, "التوبة": 9, "يونس": 10,
        "هود": 11, "يوسف": 12, "الرعد": 13, "إبراهيم": 14, "الحجر": 15,
        "النحل": 16, "الإسراء": 17, "الكهف": 18, "مريم": 19, "طه": 20,
        "المنافقون": 63, "الملك": 67, "القلم": 68, "الحاقة": 69,
    }
    
    surah = None
    ayah = None
    
    # Check surah names
    for name, num in surah_names.items():
        if name in question:
            surah = num
            break
    
    # Pattern: "2:10" or "البقرة 10"
    match = re.search(r"(\d+):(\d+)", question)
    if match:
        surah = int(match.group(1))
        ayah = int(match.group(2))
    else:
        match = re.search(r"آية\s*(\d+)", question)
        if match:
            ayah = int(match.group(1))
        match = re.search(r"الآية\s*(\d+)", question)
        if match:
            ayah = int(match.group(1))
    
    return surah, ayah


def parse_question(question: str) -> ParsedQuestion:
    """
    Parse ANY question and return structured representation.
    This is the core of the question-agnostic system.
    """
    parsed = ParsedQuestion(
        type=QuestionType.GENERAL_MAP,
        dimensions_requested=ALL_DIMENSIONS,
    )
    
    # Pattern: "حلل سلوك X" or "تحليل X"
    if re.search(r"حلل.*سلوك|تحليل.*سلوك|سلوك.*حلل", question):
        behavior = extract_behavior(question)
        parsed.type = QuestionType.BEHAVIOR_ANALYSIS
        if behavior:
            parsed.entities = [behavior]
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "ما الأعضاء" / "ما المصادر" / "ما الفاعلون"
    if re.search(r"ما\s+(ال)?أعضاء|ما\s+(ال)?مصادر|ما\s+(ال)?فاعل|ما\s+هي\s+ال", question):
        dimension = extract_dimension(question)
        parsed.type = QuestionType.DIMENSION_EXPLORATION
        if dimension:
            parsed.dimensions_requested = [dimension]
        else:
            parsed.dimensions_requested = ALL_DIMENSIONS
        # Check if filtered by behavior
        behavior = extract_behavior(question)
        if behavior:
            parsed.entities = [behavior]
            parsed.filter["behavior"] = behavior
        return parsed
    
    # Pattern: "قارن بين X و Y" or "الفرق بين" or "مقارنة"
    if re.search(r"قارن|الفرق\s+بين|مقارنة", question):
        targets = extract_comparison_targets(question)
        parsed.type = QuestionType.COMPARISON
        parsed.comparison_targets = targets
        parsed.entities = targets
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "رحلة X" or "من X إلى Y" or "مراحل"
    if re.search(r"رحلة|من.*إلى|مراحل|تحول|سلسلة", question):
        start, end = extract_journey_endpoints(question)
        parsed.type = QuestionType.JOURNEY_CHAIN
        parsed.journey_start = start
        parsed.journey_end = end
        if start:
            parsed.entities.append(start)
        if end:
            parsed.entities.append(end)
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "سورة X" or "السلوكيات في سورة"
    if re.search(r"سورة|السور", question) and not re.search(r"آية", question):
        surah, _ = extract_surah_ayah(question)
        parsed.type = QuestionType.SURAH_ANALYSIS
        parsed.surah = surah
        if surah:
            parsed.filter["surah"] = surah
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "آية X" or verse reference
    if re.search(r"آية|الآية|\d+:\d+", question):
        surah, ayah = extract_surah_ayah(question)
        parsed.type = QuestionType.VERSE_ANALYSIS
        parsed.surah = surah
        parsed.ayah = ayah
        if surah:
            parsed.filter["surah"] = surah
        if ayah:
            parsed.filter["ayah"] = ayah
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "كم مرة" / "عدد" / "إحصاء"
    if re.search(r"كم\s+مرة|عدد|إحصاء|نسبة|توزيع", question):
        parsed.type = QuestionType.STATISTICAL
        behavior = extract_behavior(question)
        if behavior:
            parsed.entities = [behavior]
            parsed.filter["behavior"] = behavior
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Pattern: "علاقة X بـ Y" or "يرتبط"
    if re.search(r"علاقة|يرتبط|صلة|مسبب|نتيجة", question):
        parsed.type = QuestionType.RELATIONSHIP
        behavior = extract_behavior(question)
        if behavior:
            parsed.entities = [behavior]
        # Try to extract related entity
        match = re.search(r"علاقة\s+(\w+)\s+ب[ـ]?\s*(\w+)", question)
        if match:
            parsed.entities = [match.group(1), match.group(2)]
        parsed.dimensions_requested = ["relationships"] + ALL_DIMENSIONS
        return parsed
    
    # Pattern: "خارطة" / "خريطة" / "شاملة"
    if re.search(r"خارطة|خريطة|شاملة|كاملة|جميع", question):
        parsed.type = QuestionType.GENERAL_MAP
        parsed.dimensions_requested = ALL_DIMENSIONS
        return parsed
    
    # Default: Try to extract behavior and do full analysis
    behavior = extract_behavior(question)
    if behavior:
        parsed.type = QuestionType.BEHAVIOR_ANALYSIS
        parsed.entities = [behavior]
    else:
        parsed.type = QuestionType.GENERAL_MAP
    
    parsed.dimensions_requested = ALL_DIMENSIONS
    return parsed


# =============================================================================
# GENERIC QUERY FUNCTIONS - WORK FOR ANY INPUT
# =============================================================================

class QuestionAgnosticAnalyzer:
    """
    Generic analyzer that works for ANY question with equal depth.
    No hardcoded values - all data comes from the database.
    """
    
    def __init__(self, spans_data: List[Dict[str, Any]]):
        """Initialize with spans data from the database."""
        self.spans = spans_data
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for fast querying."""
        self.by_behavior = {}
        self.by_agent = {}
        self.by_surah = {}
        self.by_evaluation = {}
        self.by_organ = {}
        self.by_heart_type = {}
        
        for span in self.spans:
            # Index by text (for behavior search)
            text = span.get("text_ar", "")
            
            # Index by agent
            agent = span.get("agent", {}).get("type", "")
            if agent:
                if agent not in self.by_agent:
                    self.by_agent[agent] = []
                self.by_agent[agent].append(span)
            
            # Index by surah
            surah = span.get("reference", {}).get("surah")
            if surah:
                if surah not in self.by_surah:
                    self.by_surah[surah] = []
                self.by_surah[surah].append(span)
            
            # Index by evaluation
            evaluation = span.get("normative", {}).get("evaluation", "")
            if evaluation:
                if evaluation not in self.by_evaluation:
                    self.by_evaluation[evaluation] = []
                self.by_evaluation[evaluation].append(span)
    
    def get_dimension_data(
        self,
        dimension: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generic dimension query for ANY input.
        Works for any dimension with any filter.
        """
        if dimension not in DIMENSIONS:
            return {"error": f"Unknown dimension: {dimension}"}
        
        dim_config = DIMENSIONS[dimension]
        db_column = dim_config.get("db_column")
        
        # Filter spans
        filtered_spans = self._apply_filter(self.spans, filter)
        
        # Build distribution
        distribution = {}
        examples = {}
        
        for span in filtered_spans:
            # Get value for this dimension
            value = self._get_dimension_value(span, dimension, db_column)
            if value:
                if value not in distribution:
                    distribution[value] = 0
                    examples[value] = []
                distribution[value] += 1
                if len(examples[value]) < 3:  # Keep top 3 examples
                    examples[value].append({
                        "surah": span.get("reference", {}).get("surah"),
                        "ayah": span.get("reference", {}).get("ayah"),
                        "text": span.get("text_ar", "")[:100],
                    })
        
        # Calculate percentages
        total = sum(distribution.values())
        percentages = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in distribution.items()}
        
        return {
            "dimension": dimension,
            "name_ar": dim_config["name_ar"],
            "name_en": dim_config["name_en"],
            "question_ar": dim_config["question_ar"],
            "distribution": distribution,
            "percentages": percentages,
            "total": total,
            "top_values": sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10],
            "examples": examples,
        }
    
    def _get_dimension_value(self, span: Dict, dimension: str, db_column: Optional[str]) -> Optional[str]:
        """Extract dimension value from a span."""
        if dimension == "organic":
            return span.get("organ")
        elif dimension == "situational":
            return span.get("behavior_form")
        elif dimension == "systemic":
            return span.get("axes", {}).get("systemic")
        elif dimension == "spatial":
            spatial = span.get("axes", {}).get("spatial")
            if spatial:
                return spatial
            inferred, _ = _infer_spatial_from_text(span.get("text_ar", ""))
            return inferred
        elif dimension == "temporal":
            temporal = span.get("axes", {}).get("temporal")
            if temporal:
                return temporal
            inferred, _ = _infer_temporal_from_text(span.get("text_ar", ""))
            return inferred
        elif dimension == "agent":
            return span.get("agent", {}).get("type")
        elif dimension == "source":
            return span.get("behavior_source")
        elif dimension == "evaluation":
            return span.get("normative", {}).get("evaluation")
        elif dimension == "heart_type":
            return span.get("heart_type")
        elif dimension == "consequence":
            return span.get("consequence_type")
        elif dimension == "relationships":
            return None  # Handled separately
        return None
    
    def _apply_filter(self, spans: List[Dict], filter: Optional[Dict[str, Any]]) -> List[Dict]:
        """Apply filter to spans."""
        if not filter:
            return spans
        
        filtered = spans
        
        if "behavior" in filter:
            behavior = normalize_arabic(str(filter["behavior"]))
            if behavior:
                # Search in multiple fields: text_ar, behavior_label, and any behavior-related fields
                filtered = [
                    s for s in filtered
                    if (
                        behavior in normalize_arabic(s.get("text_ar", ""))
                        or behavior in normalize_arabic(s.get("behavior_label", ""))
                        or behavior in normalize_arabic(str(s.get("behavior_concept", "")))
                        or behavior in normalize_arabic(str(s.get("behavior_form", "")))
                    )
                ]
        
        if "surah" in filter:
            surah = filter["surah"]
            filtered = [s for s in filtered if s.get("reference", {}).get("surah") == surah]
        
        if "ayah" in filter:
            ayah = filter["ayah"]
            filtered = [s for s in filtered if s.get("reference", {}).get("ayah") == ayah]
        
        if "agent" in filter:
            agent = filter["agent"].lower()
            filtered = [s for s in filtered if agent in s.get("agent", {}).get("type", "").lower()]
        
        if "evaluation" in filter:
            evaluation = filter["evaluation"].lower()
            filtered = [s for s in filtered if evaluation in s.get("normative", {}).get("evaluation", "").lower()]
        
        return filtered
    
    def analyze_behavior(self, behavior: str) -> Dict[str, Any]:
        """
        Generic behavior analysis - works for ANY behavior.
        Returns full 11-dimensional analysis.
        """
        filter = {"behavior": behavior}
        
        # Get all 11 dimensions
        dimensions_data = {}
        for dim in ALL_DIMENSIONS:
            if dim != "relationships":  # Handle separately
                dimensions_data[dim] = self.get_dimension_data(dim, filter)
        
        # Get relationships
        relationships = self.get_relationships(behavior)
        dimensions_data["relationships"] = relationships
        
        # Get statistics
        filtered_spans = self._apply_filter(self.spans, filter)
        statistics = self._build_statistics(filtered_spans)
        
        # Get verses
        verses = self._get_key_verses(filtered_spans, limit=10)
        
        # Get personality comparison
        personality_comparison = self.compare_across_agents(behavior)
        
        return {
            "behavior": behavior,
            "total_mentions": len(filtered_spans),
            "dimensions": dimensions_data,
            "statistics": statistics,
            "key_verses": verses,
            "personality_comparison": personality_comparison,
            "completeness": self._validate_completeness(dimensions_data),
        }
    
    def analyze_dimension(self, dimension: str, value: Optional[str] = None) -> Dict[str, Any]:
        """
        Generic dimension analysis - works for ANY dimension.
        """
        filter = {dimension: value} if value else None
        
        # Get data for this dimension
        dim_data = self.get_dimension_data(dimension, filter)
        
        # Get behaviors associated with this dimension value
        behaviors = self._get_behaviors_by_dimension(dimension, value)
        
        # Get verses
        filtered_spans = self._apply_filter(self.spans, filter)
        verses = self._get_key_verses(filtered_spans, limit=10)
        
        return {
            "dimension": dimension,
            "value": value,
            "dimension_data": dim_data,
            "associated_behaviors": behaviors,
            "key_verses": verses,
            "total_spans": len(filtered_spans),
        }
    
    def compare_entities(
        self,
        entity_type: str,  # "behavior", "agent", "organ", "heart_type"
        entities: List[str],
        compare_across: str = "all"
    ) -> Dict[str, Any]:
        """
        Generic comparison - works for ANY entities.
        """
        results = {}
        
        for entity in entities:
            filter = {entity_type: entity}
            filtered_spans = self._apply_filter(self.spans, filter)
            
            # Get all dimensions for this entity
            dimensions = {}
            for dim in ALL_DIMENSIONS:
                if dim != "relationships":
                    dimensions[dim] = self.get_dimension_data(dim, filter)
            
            results[entity] = {
                "total": len(filtered_spans),
                "dimensions": dimensions,
                "key_verses": self._get_key_verses(filtered_spans, limit=5),
            }
        
        # Build comparison summary
        comparison_summary = self._build_comparison_summary(results, entities)
        
        return {
            "entity_type": entity_type,
            "entities": entities,
            "individual_data": results,
            "comparison_summary": comparison_summary,
        }
    
    def find_chain(
        self,
        start: str,
        end: Optional[str] = None,
        edge_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generic chain/journey finder - works for ANY start/end points.
        """
        if edge_types is None:
            edge_types = ["causes", "leads_to", "results_in"]
        
        # Find spans mentioning start
        start_spans = self._apply_filter(self.spans, {"behavior": start})
        
        # Find spans mentioning end (if provided)
        end_spans = []
        if end:
            end_spans = self._apply_filter(self.spans, {"behavior": end})
        
        # Build chain analysis
        chain_data = {
            "start": {
                "entity": start,
                "mentions": len(start_spans),
                "dimensions": {},
            },
            "stages": [],
            "end": None,
        }
        
        # Analyze start point
        for dim in ALL_DIMENSIONS:
            if dim != "relationships":
                chain_data["start"]["dimensions"][dim] = self.get_dimension_data(dim, {"behavior": start})
        
        # If end provided, analyze it too
        if end:
            chain_data["end"] = {
                "entity": end,
                "mentions": len(end_spans),
                "dimensions": {},
            }
            for dim in ALL_DIMENSIONS:
                if dim != "relationships":
                    chain_data["end"]["dimensions"][dim] = self.get_dimension_data(dim, {"behavior": end})
        
        # Get relationships to find intermediate stages
        relationships = self.get_relationships(start)
        chain_data["relationships"] = relationships
        
        return chain_data
    
    def get_relationships(self, entity: str) -> Dict[str, Any]:
        """
        Get ALL relationships for ANY entity.
        """
        filtered_spans = self._apply_filter(self.spans, {"behavior": entity})
        
        # Extract relationship patterns from text
        causes = []
        effects = []
        opposites = []
        similar = []
        
        # Analyze co-occurrence patterns
        for span in filtered_spans:
            text = span.get("text_ar", "")
            # This would be enhanced with NLP/graph analysis
            # For now, return structure
        
        return {
            "entity": entity,
            "causes": causes,
            "effects": effects,
            "opposites": opposites,
            "similar": similar,
            "total_mentions": len(filtered_spans),
        }
    
    def compare_across_agents(self, behavior: str) -> Dict[str, Any]:
        """
        Compare behavior across different agent types.
        Works for ANY behavior.
        """
        agents = ["مؤمن", "كافر", "منافق"]
        comparison = {}
        
        for agent in agents:
            filter = {"behavior": behavior}
            filtered = self._apply_filter(self.spans, filter)
            
            # Further filter by agent
            agent_filtered = [s for s in filtered if agent in s.get("agent", {}).get("type", "")]
            
            comparison[agent] = {
                "count": len(agent_filtered),
                "evaluation_distribution": self._get_evaluation_distribution(agent_filtered),
                "key_verses": self._get_key_verses(agent_filtered, limit=3),
            }
        
        return comparison
    
    def analyze_surah(self, surah: int) -> Dict[str, Any]:
        """
        Analyze ALL behaviors in a surah.
        Works for ANY surah.
        """
        filter = {"surah": surah}
        filtered_spans = self._apply_filter(self.spans, filter)
        
        # Get all dimensions
        dimensions = {}
        for dim in ALL_DIMENSIONS:
            if dim != "relationships":
                dimensions[dim] = self.get_dimension_data(dim, filter)
        
        return {
            "surah": surah,
            "total_spans": len(filtered_spans),
            "dimensions": dimensions,
            "key_verses": self._get_key_verses(filtered_spans, limit=20),
        }
    
    def analyze_verse(self, surah: int, ayah: int) -> Dict[str, Any]:
        """
        Analyze a specific verse.
        Works for ANY verse.
        """
        filter = {"surah": surah, "ayah": ayah}
        filtered_spans = self._apply_filter(self.spans, filter)
        
        # Get all dimensions
        dimensions = {}
        for dim in ALL_DIMENSIONS:
            if dim != "relationships":
                dimensions[dim] = self.get_dimension_data(dim, filter)
        
        return {
            "surah": surah,
            "ayah": ayah,
            "total_spans": len(filtered_spans),
            "dimensions": dimensions,
            "spans": filtered_spans[:10],
        }
    
    def get_statistics(self, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get statistics for ANY filter.
        """
        filtered_spans = self._apply_filter(self.spans, filter)
        return self._build_statistics(filtered_spans)
    
    def discover_patterns(self, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Discover patterns for ANY subset of data.
        """
        filtered_spans = self._apply_filter(self.spans, filter)
        
        patterns = {
            "co_occurrence": self._analyze_co_occurrence(filtered_spans),
            "distribution_anomalies": self._find_distribution_anomalies(filtered_spans),
            "temporal_patterns": self._analyze_temporal_patterns(filtered_spans),
        }
        
        return patterns
    
    def build_general_map(self) -> Dict[str, Any]:
        """
        Build complete behavioral map across ALL dimensions.
        """
        map_data = {
            "total_spans": len(self.spans),
            "dimensions": {},
        }
        
        for dim in ALL_DIMENSIONS:
            if dim != "relationships":
                map_data["dimensions"][dim] = self.get_dimension_data(dim)
        
        # Add statistics
        map_data["statistics"] = self._build_statistics(self.spans)
        
        # Add personality comparison
        map_data["personality_overview"] = {
            agent: len(spans) for agent, spans in self.by_agent.items()
        }
        
        return map_data
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _build_statistics(self, spans: List[Dict]) -> Dict[str, Any]:
        """Build statistics from spans."""
        stats = {
            "total": len(spans),
            "by_surah": {},
            "by_agent": {},
            "by_evaluation": {},
        }
        
        for span in spans:
            surah = span.get("reference", {}).get("surah")
            if surah:
                stats["by_surah"][surah] = stats["by_surah"].get(surah, 0) + 1
            
            agent = span.get("agent", {}).get("type", "")
            if agent:
                stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
            
            evaluation = span.get("normative", {}).get("evaluation", "")
            if evaluation:
                stats["by_evaluation"][evaluation] = stats["by_evaluation"].get(evaluation, 0) + 1
        
        return stats
    
    def _get_key_verses(self, spans: List[Dict], limit: int = 10) -> List[Dict]:
        """Get key verses from spans."""
        verses = []
        seen = set()
        
        for span in spans[:limit * 2]:  # Get more to dedupe
            ref = span.get("reference", {})
            key = (ref.get("surah"), ref.get("ayah"))
            if key not in seen and key[0] and key[1]:
                seen.add(key)
                verses.append({
                    "surah": ref.get("surah"),
                    "ayah": ref.get("ayah"),
                    "surah_name": ref.get("surah_name", ""),
                    "text": span.get("text_ar", "")[:200],
                })
                if len(verses) >= limit:
                    break
        
        return verses
    
    def _get_evaluation_distribution(self, spans: List[Dict]) -> Dict[str, int]:
        """Get evaluation distribution."""
        dist = {}
        for span in spans:
            evaluation = span.get("normative", {}).get("evaluation", "unknown")
            dist[evaluation] = dist.get(evaluation, 0) + 1
        return dist
    
    def _get_behaviors_by_dimension(self, dimension: str, value: Optional[str]) -> List[str]:
        """Get behaviors associated with a dimension value."""
        # This would be enhanced with actual behavior extraction
        return []
    
    def _build_comparison_summary(self, results: Dict, entities: List[str]) -> Dict[str, Any]:
        """Build comparison summary."""
        summary = {
            "totals": {e: results[e]["total"] for e in entities},
            "differences": [],
            "similarities": [],
        }
        return summary
    
    def _validate_completeness(self, dimensions_data: Dict) -> Dict[str, Any]:
        """Validate completeness of analysis."""
        missing = []
        covered = 0
        
        for dim in ALL_DIMENSIONS:
            if dim in dimensions_data:
                data = dimensions_data[dim]
                if isinstance(data, dict) and data.get("total", 0) > 0:
                    covered += 1
                else:
                    missing.append(dim)
            else:
                missing.append(dim)
        
        return {
            "complete": len(missing) == 0,
            "score": covered / len(ALL_DIMENSIONS),
            "covered": covered,
            "total": len(ALL_DIMENSIONS),
            "missing": missing,
        }
    
    def _analyze_co_occurrence(self, spans: List[Dict]) -> List[Dict]:
        """Analyze co-occurrence patterns."""
        return []
    
    def _find_distribution_anomalies(self, spans: List[Dict]) -> List[Dict]:
        """Find distribution anomalies."""
        return []
    
    def _analyze_temporal_patterns(self, spans: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns (Makki vs Madani)."""
        return {}


# =============================================================================
# GENERIC RESPONSE BUILDER
# =============================================================================

def build_response(parsed: ParsedQuestion, analyzer: QuestionAgnosticAnalyzer) -> Dict[str, Any]:
    """
    Build response for ANY question type.
    The depth is EQUAL across all question types.
    """
    response = {
        "question_type": parsed.type.value,
        "entities": parsed.entities,
        "dimensions_requested": parsed.dimensions_requested,
    }
    
    if parsed.type == QuestionType.BEHAVIOR_ANALYSIS:
        if parsed.entities:
            response["analysis"] = analyzer.analyze_behavior(parsed.entities[0])
        else:
            response["analysis"] = analyzer.build_general_map()
    
    elif parsed.type == QuestionType.DIMENSION_EXPLORATION:
        dimension = parsed.dimensions_requested[0] if parsed.dimensions_requested else "organic"
        value = parsed.filter.get(dimension)
        response["analysis"] = analyzer.analyze_dimension(dimension, value)
    
    elif parsed.type == QuestionType.COMPARISON:
        response["analysis"] = analyzer.compare_entities(
            "behavior" if parsed.entities else "agent",
            parsed.comparison_targets or parsed.entities,
        )
    
    elif parsed.type == QuestionType.JOURNEY_CHAIN:
        response["analysis"] = analyzer.find_chain(
            parsed.journey_start or (parsed.entities[0] if parsed.entities else ""),
            parsed.journey_end,
        )
    
    elif parsed.type == QuestionType.VERSE_ANALYSIS:
        if parsed.surah and parsed.ayah:
            response["analysis"] = analyzer.analyze_verse(parsed.surah, parsed.ayah)
        else:
            response["analysis"] = {"error": "Surah and ayah required"}
    
    elif parsed.type == QuestionType.SURAH_ANALYSIS:
        if parsed.surah:
            response["analysis"] = analyzer.analyze_surah(parsed.surah)
        else:
            response["analysis"] = {"error": "Surah number required"}
    
    elif parsed.type == QuestionType.STATISTICAL:
        response["analysis"] = analyzer.get_statistics(parsed.filter)
    
    elif parsed.type == QuestionType.RELATIONSHIP:
        if parsed.entities:
            response["analysis"] = analyzer.get_relationships(parsed.entities[0])
        else:
            response["analysis"] = {"error": "Entity required for relationship analysis"}
    
    elif parsed.type == QuestionType.GENERAL_MAP:
        response["analysis"] = analyzer.build_general_map()
    
    return response


# =============================================================================
# ENHANCED CONTEXT METHODS - Spatial, Temporal, Query Evidence, Graph
# =============================================================================

def get_spatial_context(spans: List[Dict], behavior: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed spatial context showing WHERE behaviors occur.
    Generic - works for ANY behavior or all behaviors.
    """
    import time
    start_time = time.time()
    
    behavior_norm = normalize_arabic(behavior) if behavior else ""

    spatial_vocab = _vocab_items_by_id("spatial")
    spatial_data: Dict[str, Dict[str, Any]] = {}
    inference_counts = {"axes": 0, "text_inference": 0, "missing": 0}

    for span in spans:
        text_ar = span.get("text_ar", "")
        if behavior_norm and behavior_norm not in normalize_arabic(text_ar):
            continue

        spatial = span.get("axes", {}).get("spatial") or span.get("spatial_context")
        matched_term = None
        method = "axes"

        if not spatial:
            spatial, matched_term = _infer_spatial_from_text(text_ar)
            method = "text_inference" if spatial else "missing"

        if not spatial:
            inference_counts["missing"] += 1
            continue

        inference_counts[method] += 1

        item = spatial_vocab.get(spatial, {})
        location_ar = item.get("ar") if item else None
        location_en = item.get("en") if item else None

        if spatial not in spatial_data:
            spatial_data[spatial] = {
                "location_id": spatial,
                "location": f"في {location_ar}" if location_ar else spatial,
                "location_en": location_en,
                "method": method,
                "matched_terms": [],
                "behaviors": [],
                "verse_count": 0,
                "example_verses": [],
            }

        entry = spatial_data[spatial]
        entry["verse_count"] += 1

        if matched_term and len(entry["matched_terms"]) < 5 and matched_term not in entry["matched_terms"]:
            entry["matched_terms"].append(matched_term)

        excerpt = text_ar[:50]
        if excerpt and excerpt not in entry["behaviors"] and len(entry["behaviors"]) < 5:
            entry["behaviors"].append(excerpt)

        ref = span.get("reference", {})
        verse_ref = f"{ref.get('surah', '?')}:{ref.get('ayah', '?')}"
        if verse_ref and verse_ref not in entry["example_verses"] and len(entry["example_verses"]) < 5:
            entry["example_verses"].append(verse_ref)
    
    execution_time = round((time.time() - start_time) * 1000, 2)
    behavior_matches = sum(inference_counts.values())
    spatial_labeled = behavior_matches - inference_counts["missing"]
    
    return {
        "spatial_distribution": sorted(
            spatial_data.values(), key=lambda d: d.get("verse_count", 0), reverse=True
        ),
        "total_with_spatial": sum(d.get("verse_count", 0) for d in spatial_data.values()),
        "query_evidence": {
            "filter_applied": f"behavior={behavior}" if behavior else "none",
            "spans_searched": len(spans),
            "matches_found": behavior_matches,
            "execution_time_ms": execution_time,
            "inference": inference_counts,
            "spatial_labeled": spatial_labeled,
            "vocab_loaded": bool(spatial_vocab),
        }
    }


def get_temporal_context(spans: List[Dict], behavior: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed temporal context showing WHEN behaviors occur and consequences manifest.
    Generic - works for ANY behavior or all behaviors.
    """
    import time
    start_time = time.time()
    
    behavior_norm = normalize_arabic(behavior) if behavior else ""
    
    # Temporal ordering
    TEMPORAL_ORDER = ["دنيا", "عند_الموت", "برزخ", "قيامة", "آخرة"]
    TEMPORAL_LABELS = {
        "دنيا": "في الدنيا (الممارسة والاختيار)",
        "عند_الموت": "عند الموت (ظهور الحقيقة)",
        "برزخ": "في البرزخ (الانتظار)",
        "قيامة": "في القيامة (الحساب والشهادة)",
        "آخرة": "في الآخرة (الجزاء النهائي)",
    }
    
    temporal_data = {t: {"timeframe": TEMPORAL_LABELS.get(t, t), "behavior_count": 0, 
                         "consequences": [], "example_verses": []} for t in TEMPORAL_ORDER}
    
    inference_counts = {"axes": 0, "text_inference": 0, "missing": 0, "out_of_vocab": 0}

    for span in spans:
        text_ar = span.get("text_ar", "")
        if behavior_norm and behavior_norm not in normalize_arabic(text_ar):
            continue

        temporal = span.get("axes", {}).get("temporal") or span.get("temporal_context")
        method = "axes"
        matched_term = None

        if not temporal:
            temporal, matched_term = _infer_temporal_from_text(text_ar)
            method = "text_inference" if temporal else "missing"

        if not temporal:
            inference_counts["missing"] += 1
            continue

        if temporal not in temporal_data:
            inference_counts["out_of_vocab"] += 1
            continue

        inference_counts[method] += 1
        temporal_data[temporal]["behavior_count"] += 1

        # Add consequence
        consequence = span.get("consequence_type", "")
        if consequence and consequence not in temporal_data[temporal]["consequences"]:
            temporal_data[temporal]["consequences"].append(consequence)

        # Add verse reference
        ref = span.get("reference", {})
        verse_ref = f"{ref.get('surah', '?')}:{ref.get('ayah', '?')}"
        if len(temporal_data[temporal]["example_verses"]) < 5:
            temporal_data[temporal]["example_verses"].append(verse_ref)
    
    execution_time = round((time.time() - start_time) * 1000, 2)

    # Filter out empty timeframes and maintain order
    result = [temporal_data[t] for t in TEMPORAL_ORDER if temporal_data[t]["behavior_count"] > 0]
    behavior_matches = sum(inference_counts.values())
    temporal_labeled = sum(d["behavior_count"] for d in result)
    
    return {
        "temporal_distribution": result,
        "total_with_temporal": sum(d["behavior_count"] for d in result),
        "query_evidence": {
            "filter_applied": f"behavior={behavior}" if behavior else "none",
            "spans_searched": len(spans),
            "matches_found": behavior_matches,
            "execution_time_ms": execution_time,
            "inference": inference_counts,
            "temporal_labeled": temporal_labeled,
        }
    }


def build_query_evidence(claim: str, filter_desc: str, results: List[Dict], 
                         execution_time_ms: float) -> Dict[str, Any]:
    """
    Build query evidence to prove data comes from real database queries.
    """
    sample_results = []
    for r in results[:3]:
        ref = r.get("reference", {})
        sample_results.append({
            "verse": f"{ref.get('surah', '?')}:{ref.get('ayah', '?')}",
            "text": r.get("text_ar", "")[:80],
        })
    
    return {
        "claim": claim,
        "query_description": filter_desc,
        "result_count": len(results),
        "execution_time_ms": execution_time_ms,
        "sample_results": sample_results,
    }


def build_journey_graph(stages: List[Dict], transitions: List[Dict] = None) -> Dict[str, Any]:
    """
    Build graph visualization data for journey/chain analysis.
    Returns nodes and edges for frontend rendering.
    """
    # Color mapping for different states
    STATE_COLORS = {
        "سليم": "#22c55e",      # Green - healthy
        "مريض": "#eab308",      # Yellow - sick
        "قاسي": "#f97316",      # Orange - hard
        "مختوم": "#ef4444",     # Red - sealed
        "ميت": "#1f2937",       # Dark - dead
        "منيب": "#3b82f6",      # Blue - repentant
    }
    
    nodes = []
    edges = []
    
    for i, stage in enumerate(stages):
        label = stage.get("name", stage.get("entity", f"Stage {i+1}"))
        
        # Determine color based on label
        color = "#6b7280"  # Default gray
        for key, c in STATE_COLORS.items():
            if key in label:
                color = c
                break
        
        nodes.append({
            "id": f"stage_{i}",
            "label": label,
            "type": stage.get("type", "heart_state"),
            "color": color,
            "data": {
                "mentions": stage.get("mentions", 0),
                "verses": stage.get("verses", []),
            }
        })
    
    # Build edges between consecutive stages
    if transitions:
        for t in transitions:
            edges.append({
                "source": t.get("from", ""),
                "target": t.get("to", ""),
                "label": t.get("behavior", "يؤدي إلى"),
            })
    else:
        # Default: connect consecutive stages
        for i in range(len(nodes) - 1):
            edges.append({
                "source": f"stage_{i}",
                "target": f"stage_{i + 1}",
                "label": "يؤدي إلى",
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "horizontal",  # Suggest horizontal layout for journey
    }


def build_context_graph(
    spatial_context: Dict[str, Any],
    temporal_context: Dict[str, Any],
    query_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build lightweight graph data for frontend visualization of context distributions.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    root_id = "query"
    nodes.append(
        {
            "id": root_id,
            "label": query_label or "Query",
            "type": "query",
            "color": "#111827",
        }
    )

    # Spatial nodes
    for i, entry in enumerate((spatial_context.get("spatial_distribution") or [])[:10]):
        node_id = f"spatial_{i}"
        nodes.append(
            {
                "id": node_id,
                "label": entry.get("location") or entry.get("location_id") or "Spatial",
                "type": "spatial",
                "color": "#10b981",
                "data": {"count": entry.get("verse_count", 0), "examples": entry.get("example_verses", [])},
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "spatial"})

        for j, verse_ref in enumerate((entry.get("example_verses") or [])[:3]):
            verse_node_id = f"{node_id}_verse_{j}"
            nodes.append({"id": verse_node_id, "label": verse_ref, "type": "verse", "color": "#6b7280"})
            edges.append({"source": node_id, "target": verse_node_id, "label": "example"})

    # Temporal nodes
    for i, entry in enumerate((temporal_context.get("temporal_distribution") or [])[:10]):
        node_id = f"temporal_{i}"
        nodes.append(
            {
                "id": node_id,
                "label": entry.get("timeframe") or "Temporal",
                "type": "temporal",
                "color": "#3b82f6",
                "data": {"count": entry.get("behavior_count", 0), "examples": entry.get("example_verses", [])},
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "temporal"})

        for j, verse_ref in enumerate((entry.get("example_verses") or [])[:3]):
            verse_node_id = f"{node_id}_verse_{j}"
            nodes.append({"id": verse_node_id, "label": verse_ref, "type": "verse", "color": "#6b7280"})
            edges.append({"source": node_id, "target": verse_node_id, "label": "example"})

    return {"nodes": nodes, "edges": edges, "layout": "radial"}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_question(question: str, spans_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process ANY question with equal depth.
    This is the main entry point for the question-agnostic system.
    """
    import time
    start_time = time.time()
    
    # Parse the question
    parsed = parse_question(question)
    
    # Create analyzer
    analyzer = QuestionAgnosticAnalyzer(spans_data)
    
    # Build response
    response = build_response(parsed, analyzer)
    
    # Add enhanced context for all responses
    response["spatial_context"] = get_spatial_context(
        spans_data, 
        parsed.entities[0] if parsed.entities else None
    )
    response["temporal_context"] = get_temporal_context(
        spans_data,
        parsed.entities[0] if parsed.entities else None
    )

    # Always provide a small context graph to support visualization
    response["context_graph"] = build_context_graph(
        response.get("spatial_context", {}),
        response.get("temporal_context", {}),
        query_label=parsed.entities[0] if parsed.entities else question,
    )
    
    # Add graph visualization for journey questions
    if parsed.type == QuestionType.JOURNEY_CHAIN and "analysis" in response:
        analysis = response["analysis"]
        if "stages" in analysis:
            response["graph"] = build_journey_graph(analysis["stages"])
    
    # Add metadata
    response["parsed_question"] = parsed.to_dict()
    response["methodology"] = "question_agnostic_11_dimensions"
    response["total_execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
    response["data_source"] = {
        "total_spans": len(spans_data),
        "database": "QBM Gold Dataset",
        "version": "1.0.0",
    }
    
    return response
