"""
QueryRouter: Deterministic query classification for QBM.

Phase 5.5: Routes queries to appropriate retrieval strategy.

Query Intents:
- AYAH_REF: Explicit verse reference (2:255, البقرة:255)
- SURAH_REF: Explicit surah reference (سورة البقرة, Al-Baqarah)
- CONCEPT_REF: Behavior/vocab term or ID (صبر, BEH_*, AGT_*)
- FREE_TEXT: Open questions, multi-concept queries

For AYAH_REF/SURAH_REF/CONCEPT_REF: deterministic retrieval
For FREE_TEXT: hybrid semantic retrieval
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification."""
    AYAH_REF = "ayah_ref"      # Explicit verse reference
    SURAH_REF = "surah_ref"    # Explicit surah reference
    CONCEPT_REF = "concept_ref"  # Behavior/vocab term or ID
    FREE_TEXT = "free_text"    # Open question


# Surah name mappings (Arabic and English)
SURAH_NAMES = {
    # Arabic names
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

# Common behavior/concept terms (Arabic)
BEHAVIOR_TERMS = {
    # Core behaviors
    'إيمان', 'كفر', 'نفاق', 'شرك', 'توبة', 'استغفار',
    'صبر', 'شكر', 'تقوى', 'خشية', 'رجاء', 'خوف',
    'صدق', 'كذب', 'أمانة', 'خيانة', 'عدل', 'ظلم',
    'تواضع', 'كبر', 'رياء', 'إخلاص', 'حسد', 'غيبة',
    'صلاة', 'زكاة', 'صيام', 'حج', 'جهاد', 'دعاء',
    'ذكر', 'تلاوة', 'تدبر', 'تفكر', 'توكل', 'رضا',
    # Agent types
    'مؤمن', 'كافر', 'منافق', 'مشرك', 'نبي', 'رسول',
    'ملك', 'شيطان', 'جن', 'إنسان',
    # Organs
    'قلب', 'عين', 'أذن', 'لسان', 'يد', 'رجل',
}

# Concept ID patterns
CONCEPT_ID_PATTERNS = [
    r'BEH_\w+',  # Behavior IDs
    r'AGT_\w+',  # Agent IDs
    r'ORG_\w+',  # Organ IDs
    r'ACT_\w+',  # Action IDs
]


@dataclass
class RouterResult:
    """Result from query routing."""
    query: str
    intent: QueryIntent
    confidence: float
    extracted_ref: Optional[str] = None  # e.g., "2:255" or "البقرة"
    surah_num: Optional[int] = None
    ayah_num: Optional[int] = None
    concept_term: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'intent': self.intent.value,
            'confidence': round(self.confidence, 4),
            'extracted_ref': self.extracted_ref,
            'surah_num': self.surah_num,
            'ayah_num': self.ayah_num,
            'concept_term': self.concept_term,
            'debug_info': self.debug_info,
        }


class QueryRouter:
    """
    Deterministic query router for QBM.
    
    Classifies queries into intents and extracts references.
    """
    
    def __init__(self):
        self.surah_names = SURAH_NAMES
        self.behavior_terms = BEHAVIOR_TERMS
        self.concept_id_patterns = [re.compile(p) for p in CONCEPT_ID_PATTERNS]
    
    def route(self, query: str) -> RouterResult:
        """
        Route a query to the appropriate intent.
        
        Priority order:
        1. AYAH_REF (explicit verse reference - highest priority)
        2. CONCEPT_REF (behavior/vocab term or ID - before surah to avoid false matches)
        3. SURAH_REF (explicit surah reference - only if "سورة" keyword present or exact match)
        4. FREE_TEXT (fallback)
        """
        query = query.strip()
        
        # 1. Check for AYAH_REF (verse reference) - highest priority
        ayah_result = self._detect_ayah_ref(query)
        if ayah_result:
            return ayah_result
        
        # 2. Check for CONCEPT_REF (behavior/vocab) - before surah to avoid false matches
        concept_result = self._detect_concept_ref(query)
        if concept_result:
            return concept_result
        
        # 3. Check for SURAH_REF (surah reference) - only explicit references
        surah_result = self._detect_surah_ref(query)
        if surah_result:
            return surah_result
        
        # 4. Fallback to FREE_TEXT
        return RouterResult(
            query=query,
            intent=QueryIntent.FREE_TEXT,
            confidence=1.0,
            debug_info={'reason': 'no_specific_reference_detected'},
        )
    
    def _detect_ayah_ref(self, query: str) -> Optional[RouterResult]:
        """Detect explicit verse reference."""
        # Pattern 1: Numeric format (2:255, 2:255-260)
        match = re.search(r'(\d{1,3}):(\d{1,3})(?:-(\d{1,3}))?', query)
        if match:
            surah = int(match.group(1))
            ayah = int(match.group(2))
            if 1 <= surah <= 114:
                return RouterResult(
                    query=query,
                    intent=QueryIntent.AYAH_REF,
                    confidence=1.0,
                    extracted_ref=f"{surah}:{ayah}",
                    surah_num=surah,
                    ayah_num=ayah,
                    debug_info={'pattern': 'numeric', 'match': match.group(0)},
                )
        
        # Pattern 2: Arabic surah name + ayah (البقرة:255, البقرة 255)
        for name, num in self.surah_names.items():
            if name in query:
                # Look for ayah number after surah name
                pattern = rf'{re.escape(name)}[\s:،,]+(\d+)'
                match = re.search(pattern, query)
                if match:
                    ayah = int(match.group(1))
                    return RouterResult(
                        query=query,
                        intent=QueryIntent.AYAH_REF,
                        confidence=0.95,
                        extracted_ref=f"{num}:{ayah}",
                        surah_num=num,
                        ayah_num=ayah,
                        debug_info={'pattern': 'arabic_name', 'surah_name': name},
                    )
        
        # Pattern 3: "آية" or "الآية" followed by number
        match = re.search(r'(?:آية|الآية)\s*(\d+)', query)
        if match:
            # Need context to determine surah - return as partial
            return None  # Can't determine surah
        
        return None
    
    def _detect_surah_ref(self, query: str) -> Optional[RouterResult]:
        """Detect explicit surah reference."""
        # Pattern 1: "سورة" + name (highest confidence)
        match = re.search(r'سورة\s+([\u0600-\u06FF\s]+)', query)
        if match:
            surah_name = match.group(1).strip()
            for name, num in self.surah_names.items():
                if name in surah_name or surah_name in name:
                    return RouterResult(
                        query=query,
                        intent=QueryIntent.SURAH_REF,
                        confidence=0.95,
                        extracted_ref=name,
                        surah_num=num,
                        debug_info={'pattern': 'surah_keyword', 'surah_name': name},
                    )
        
        # Pattern 2: Exact surah name match (query IS the surah name)
        query_clean = query.strip()
        for name, num in self.surah_names.items():
            if query_clean == name:
                return RouterResult(
                    query=query,
                    intent=QueryIntent.SURAH_REF,
                    confidence=0.9,
                    extracted_ref=name,
                    surah_num=num,
                    debug_info={'pattern': 'exact_match', 'surah_name': name},
                )
        
        return None
    
    def _detect_concept_ref(self, query: str) -> Optional[RouterResult]:
        """Detect behavior/vocab term or ID."""
        # Pattern 1: Concept IDs (BEH_*, AGT_*, etc.)
        for pattern in self.concept_id_patterns:
            match = pattern.search(query)
            if match:
                return RouterResult(
                    query=query,
                    intent=QueryIntent.CONCEPT_REF,
                    confidence=1.0,
                    concept_term=match.group(0),
                    debug_info={'pattern': 'concept_id', 'id': match.group(0)},
                )
        
        # Pattern 2: Known behavior terms (with or without ال prefix)
        query_words = set(re.findall(r'[\u0600-\u06FF]+', query))
        
        # Also check words without ال prefix
        query_words_normalized = set()
        for word in query_words:
            query_words_normalized.add(word)
            if word.startswith('ال'):
                query_words_normalized.add(word[2:])  # Remove ال
        
        matched_terms = query_words_normalized & self.behavior_terms
        
        if matched_terms:
            term = list(matched_terms)[0]  # Take first match
            
            # Check if it's a focused query about the concept
            concept_keywords = ['معنى', 'تعريف', 'ما هو', 'ما هي', 'آيات', 'أحاديث', 'فضل', 'أنواع']
            is_concept_query = (
                len(query_words) <= 3 or  # Short query with behavior term
                any(kw in query for kw in concept_keywords)  # Has concept keyword
            )
            
            if is_concept_query:
                return RouterResult(
                    query=query,
                    intent=QueryIntent.CONCEPT_REF,
                    confidence=0.85,
                    concept_term=term,
                    debug_info={'pattern': 'behavior_term', 'term': term, 'all_matches': list(matched_terms)},
                )
        
        return None


# Singleton instance
_query_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """Get or create the singleton QueryRouter."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    router = get_query_router()
    
    test_queries = [
        # AYAH_REF
        "2:255",
        "ما تفسير الآية 2:255",
        "البقرة:255",
        "تفسير آية الكرسي البقرة 255",
        
        # SURAH_REF
        "سورة البقرة",
        "الفاتحة",
        "ما هي سورة الكهف",
        
        # CONCEPT_REF
        "الصبر",
        "ما معنى التقوى",
        "BEH_SABR",
        "آيات الصبر",
        
        # FREE_TEXT
        "كيف أتعامل مع الابتلاء",
        "ما هي صفات المؤمنين",
        "العلاقة بين الإيمان والعمل الصالح",
    ]
    
    print("\n" + "=" * 60)
    print("Testing QueryRouter")
    print("=" * 60)
    
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.extracted_ref:
            print(f"  Ref: {result.extracted_ref}")
        if result.concept_term:
            print(f"  Concept: {result.concept_term}")
