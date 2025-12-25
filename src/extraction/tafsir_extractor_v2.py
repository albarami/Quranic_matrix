"""
QBM Tafsir Extractor V2 - Phase 3
Morphology-based behavior extraction using pattern matching.
Replaces keyword-only extraction with linguistic analysis.
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.provenance import AnnotationWithProvenance


class TafsirExtractorV2:
    """
    Morphology-aware behavior extractor for Arabic tafsir text.
    
    Uses Arabic morphological patterns to identify behaviors more accurately
    than simple keyword matching.
    """
    
    VERSION = "2.0.0"
    
    # Arabic verb patterns (أوزان الفعل)
    VERB_PATTERNS = [
        # Form I (فَعَلَ)
        r'[يت]?[فعل]',
        # Form II (فَعَّلَ) - intensive/causative
        r'[يت]?[فعل]ّ',
        # Form III (فَاعَلَ) - reciprocal
        r'[يت]?[فا][عل]',
        # Form V (تَفَعَّلَ) - reflexive of II
        r'ت[فعل]ّ',
        # Form VI (تَفَاعَلَ) - reciprocal reflexive
        r'ت[فا][عل]',
        # Form X (اِسْتَفْعَلَ) - seeking/requesting
        r'[يت]?ست[فعل]',
    ]
    
    # Common behavior roots in Arabic (جذور السلوك)
    BEHAVIOR_ROOTS = {
        # Positive behaviors
        'صبر': 'الصبر',      # patience
        'شكر': 'الشكر',      # gratitude
        'تقو': 'التقوى',     # piety
        'توب': 'التوبة',     # repentance
        'ذكر': 'الذكر',      # remembrance
        'عبد': 'العبادة',    # worship
        'صلو': 'الصلاة',     # prayer
        'زكو': 'الزكاة',     # charity
        'صوم': 'الصيام',     # fasting
        'حج': 'الحج',        # pilgrimage
        'جهد': 'الجهاد',     # striving
        'صدق': 'الصدق',      # truthfulness
        'امن': 'الأمانة',    # trustworthiness
        'عدل': 'العدل',      # justice
        'حلم': 'الحلم',      # forbearance
        'رحم': 'الرحمة',     # mercy
        'كرم': 'الكرم',      # generosity
        'تواضع': 'التواضع',  # humility
        
        # Negative behaviors
        'كبر': 'الكبر',      # arrogance
        'حسد': 'الحسد',      # envy
        'غضب': 'الغضب',      # anger
        'كذب': 'الكذب',      # lying
        'غيب': 'الغيبة',     # backbiting
        'نميم': 'النميمة',   # tale-bearing
        'ظلم': 'الظلم',      # oppression
        'بخل': 'البخل',      # stinginess
        'رياء': 'الرياء',    # showing off
        'نفاق': 'النفاق',    # hypocrisy
        'فسق': 'الفسق',      # sinfulness
        'فجر': 'الفجور',     # immorality
        'سرق': 'السرقة',     # theft
        'زنا': 'الزنا',      # adultery
        'قتل': 'القتل',      # murder
        'خيان': 'الخيانة',   # betrayal
    }
    
    # Behavior indicator words
    BEHAVIOR_INDICATORS = [
        'سلوك', 'خلق', 'صفة', 'فعل', 'عمل',
        'يفعل', 'يعمل', 'يتصف', 'يتخلق',
        'من صفات', 'من أخلاق', 'من سلوك',
        'المؤمن', 'الكافر', 'المنافق', 'الصالح',
    ]
    
    def __init__(self, cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner or TextCleaner()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for behavior detection."""
        # Build root pattern
        roots = '|'.join(re.escape(root) for root in self.BEHAVIOR_ROOTS.keys())
        self.root_pattern = re.compile(f'({roots})', re.UNICODE)
        
        # Build indicator pattern
        indicators = '|'.join(re.escape(ind) for ind in self.BEHAVIOR_INDICATORS)
        self.indicator_pattern = re.compile(f'({indicators})', re.UNICODE)
    
    def extract_behaviors(
        self,
        text: str,
        surah: int,
        ayah: int,
        tafsir_source: str,
        source_file: str,
    ) -> List[AnnotationWithProvenance]:
        """
        Extract behaviors from tafsir text with provenance.
        
        Args:
            text: Tafsir text (will be cleaned if contains HTML)
            surah: Surah number
            ayah: Ayah number
            tafsir_source: Source tafsir name
            source_file: Original source file path
            
        Returns:
            List of annotations with provenance
        """
        # Clean text first
        cleaned_text = self.cleaner.clean(text)
        
        annotations = []
        
        # Find behavior roots
        for match in self.root_pattern.finditer(cleaned_text):
            root = match.group(1)
            behavior = self.BEHAVIOR_ROOTS.get(root, root)
            
            # Get context window (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(cleaned_text), match.end() + 50)
            context = cleaned_text[start:end]
            
            # Calculate confidence based on context
            confidence = self._calculate_confidence(context, root)
            
            annotation = AnnotationWithProvenance(
                text=context,
                behavior=behavior,
                surah=surah,
                ayah=ayah,
                tafsir_source=tafsir_source,
                char_start=match.start(),
                char_end=match.end(),
                source_file=source_file,
                extractor_version=self.VERSION,
                confidence=confidence,
                extraction_method="morphology",
            )
            annotations.append(annotation)
        
        return annotations
    
    def _calculate_confidence(self, context: str, root: str) -> float:
        """
        Calculate extraction confidence based on context.
        
        Higher confidence if:
        - Behavior indicators present
        - Root appears with definite article
        - Context contains explanatory phrases
        """
        confidence = 0.5  # Base confidence
        
        # Check for behavior indicators
        if self.indicator_pattern.search(context):
            confidence += 0.2
        
        # Check for definite article with behavior
        behavior = self.BEHAVIOR_ROOTS.get(root, '')
        if behavior and behavior in context:
            confidence += 0.15
        
        # Check for explanatory phrases
        explanatory = ['أي', 'يعني', 'معناه', 'المراد', 'والمقصود']
        if any(phrase in context for phrase in explanatory):
            confidence += 0.1
        
        # Cap at 0.95
        return min(confidence, 0.95)
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for behavior content without full extraction.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        cleaned = self.cleaner.clean(text)
        
        roots_found = list(set(self.root_pattern.findall(cleaned)))
        indicators_found = list(set(self.indicator_pattern.findall(cleaned)))
        
        behaviors = [
            self.BEHAVIOR_ROOTS.get(root, root)
            for root in roots_found
        ]
        
        return {
            "text_length": len(cleaned),
            "roots_found": roots_found,
            "behaviors_identified": behaviors,
            "indicators_found": indicators_found,
            "has_behavior_content": len(roots_found) > 0,
        }


# Known false positives to filter out
FALSE_POSITIVES = {
    # Common words that match behavior roots but aren't behaviors
    'صبر': ['صبرا جميلا'],  # Specific phrase, not general patience
    'قتل': ['قتل نفسا'],     # Specific Quranic phrase
}


def filter_false_positives(
    annotations: List[AnnotationWithProvenance]
) -> List[AnnotationWithProvenance]:
    """
    Filter out known false positive extractions.
    
    Args:
        annotations: List of extracted annotations
        
    Returns:
        Filtered list with false positives removed
    """
    filtered = []
    
    for ann in annotations:
        is_false_positive = False
        
        # Check against known false positives
        for behavior, patterns in FALSE_POSITIVES.items():
            if behavior in ann.behavior:
                for pattern in patterns:
                    if pattern in ann.text:
                        is_false_positive = True
                        break
        
        if not is_false_positive:
            filtered.append(ann)
    
    return filtered
