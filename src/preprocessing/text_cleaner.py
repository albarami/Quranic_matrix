"""
QBM Text Cleaner - Phase 3
HTML stripping, Arabic normalization, and text cleaning for tafsir data.
"""

import re
import html
from typing import Optional
import unicodedata


class TextCleaner:
    """
    Text cleaner for Arabic tafsir content.
    Removes HTML, normalizes Arabic text, and cleans whitespace.
    """
    
    # HTML tag pattern
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    
    # HTML entity pattern
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')
    
    # Multiple whitespace pattern
    MULTI_WHITESPACE = re.compile(r'\s+')
    
    # Arabic diacritics (tashkeel) - optional removal
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')
    
    # Arabic letter normalization mappings
    ARABIC_NORMALIZATIONS = {
        'أ': 'ا',  # Alef with hamza above -> Alef
        'إ': 'ا',  # Alef with hamza below -> Alef
        'آ': 'ا',  # Alef with madda -> Alef
        'ٱ': 'ا',  # Alef wasla -> Alef
        'ة': 'ه',  # Teh marbuta -> Heh
        'ى': 'ي',  # Alef maksura -> Yeh
        'ؤ': 'و',  # Waw with hamza -> Waw
        'ئ': 'ي',  # Yeh with hamza -> Yeh
    }
    
    def __init__(
        self,
        strip_html: bool = True,
        normalize_arabic: bool = True,
        remove_diacritics: bool = False,
        normalize_whitespace: bool = True,
    ):
        self.strip_html = strip_html
        self.normalize_arabic = normalize_arabic
        self.remove_diacritics = remove_diacritics
        self.normalize_whitespace = normalize_whitespace
    
    def clean(self, text: str) -> str:
        """
        Clean text with all configured options.
        
        Args:
            text: Raw text potentially containing HTML
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Strip HTML tags
        if self.strip_html:
            text = self._strip_html(text)
        
        # Step 2: Decode HTML entities
        text = html.unescape(text)
        
        # Step 3: Remove any remaining HTML entities
        text = self.HTML_ENTITY_PATTERN.sub('', text)
        
        # Step 4: Normalize Arabic characters
        if self.normalize_arabic:
            text = self._normalize_arabic(text)
        
        # Step 5: Remove diacritics if requested
        if self.remove_diacritics:
            text = self.ARABIC_DIACRITICS.sub('', text)
        
        # Step 6: Normalize whitespace
        if self.normalize_whitespace:
            text = self.MULTI_WHITESPACE.sub(' ', text)
            text = text.strip()
        
        return text
    
    def _strip_html(self, text: str) -> str:
        """Remove all HTML tags from text."""
        return self.HTML_TAG_PATTERN.sub('', text)
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic characters for consistent matching."""
        for old, new in self.ARABIC_NORMALIZATIONS.items():
            text = text.replace(old, new)
        return text
    
    def get_html_contamination_rate(self, text: str) -> float:
        """
        Calculate the percentage of text that is HTML.
        
        Args:
            text: Raw text to analyze
            
        Returns:
            Float between 0 and 1 representing HTML contamination rate
        """
        if not text:
            return 0.0
        
        html_matches = self.HTML_TAG_PATTERN.findall(text)
        html_chars = sum(len(m) for m in html_matches)
        
        return html_chars / len(text) if len(text) > 0 else 0.0
    
    def has_html(self, text: str) -> bool:
        """Check if text contains any HTML tags."""
        return bool(self.HTML_TAG_PATTERN.search(text))


# Convenience functions
_default_cleaner = TextCleaner()


def clean_arabic_text(text: str, remove_diacritics: bool = False) -> str:
    """
    Clean Arabic text with default settings.
    
    Args:
        text: Raw text to clean
        remove_diacritics: Whether to remove Arabic diacritics
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(remove_diacritics=remove_diacritics)
    return cleaner.clean(text)


def strip_html(text: str) -> str:
    """
    Strip HTML tags from text.
    
    Args:
        text: Text with potential HTML
        
    Returns:
        Text with HTML removed
    """
    return _default_cleaner._strip_html(text)


def analyze_html_contamination(texts: list) -> dict:
    """
    Analyze HTML contamination across a list of texts.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        Dictionary with contamination statistics
    """
    cleaner = TextCleaner()
    
    total = len(texts)
    contaminated = 0
    total_contamination = 0.0
    
    for text in texts:
        if text and cleaner.has_html(text):
            contaminated += 1
            total_contamination += cleaner.get_html_contamination_rate(text)
    
    return {
        "total_texts": total,
        "contaminated_count": contaminated,
        "contamination_rate": contaminated / total if total > 0 else 0.0,
        "avg_contamination_per_text": total_contamination / contaminated if contaminated > 0 else 0.0,
    }
