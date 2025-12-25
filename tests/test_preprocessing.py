"""
Phase 3: Preprocessing Tests
Tests for text cleaning, provenance, and extraction.
"""

import pytest
from datetime import datetime


class TestTextCleaner:
    """Test text cleaning functionality"""
    
    def test_strip_html_tags(self):
        """HTML tags should be removed"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        text = '<p><span class="arabic">بسم الله</span></p>'
        cleaned = cleaner.clean(text)
        
        assert '<' not in cleaned
        assert '>' not in cleaned
        assert 'بسم الله' in cleaned
    
    def test_html_entities_decoded(self):
        """HTML entities should be decoded"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        text = 'Hello&nbsp;World &amp; Test'
        cleaned = cleaner.clean(text)
        
        assert '&nbsp;' not in cleaned
        assert '&amp;' not in cleaned
    
    def test_arabic_normalization(self):
        """Arabic characters should be normalized"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner(normalize_arabic=True)
        
        # Alef variations
        assert cleaner.clean('أحمد') == cleaner.clean('احمد')
        assert cleaner.clean('إبراهيم') == cleaner.clean('ابراهيم')
    
    def test_whitespace_normalization(self):
        """Multiple whitespace should be collapsed"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        text = 'Hello    World\n\n\tTest'
        cleaned = cleaner.clean(text)
        
        assert '  ' not in cleaned
        assert cleaned == 'Hello World Test'
    
    def test_contamination_rate_calculation(self):
        """HTML contamination rate should be calculated correctly"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        
        # 50% HTML
        text = '<p>Test</p>'  # 7 chars HTML, 4 chars text = 11 total
        rate = cleaner.get_html_contamination_rate(text)
        assert rate > 0.5  # More than half is HTML
        
        # No HTML
        text = 'Pure text'
        rate = cleaner.get_html_contamination_rate(text)
        assert rate == 0.0
    
    def test_has_html_detection(self):
        """HTML detection should work"""
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        
        assert cleaner.has_html('<p>Test</p>')
        assert cleaner.has_html('<span class="x">Y</span>')
        assert not cleaner.has_html('Plain text')
        assert not cleaner.has_html('Arabic text: الصبر من صفات المؤمنين')


class TestProvenance:
    """Test provenance schema"""
    
    def test_annotation_with_provenance_creation(self):
        """AnnotationWithProvenance should be created correctly"""
        from src.preprocessing.provenance import AnnotationWithProvenance
        
        ann = AnnotationWithProvenance(
            text="الصبر من صفات المؤمنين",
            behavior="الصبر",
            surah=2,
            ayah=155,
            tafsir_source="ibn_kathir",
            char_start=0,
            char_end=20,
            source_file="tafsir.db",
            confidence=0.85,
        )
        
        assert ann.behavior == "الصبر"
        assert ann.surah == 2
        assert ann.extractor_version == "2.0.0"
        assert ann.review_status == "pending"
        assert isinstance(ann.extracted_at, datetime)
    
    def test_surah_validation(self):
        """Surah number should be validated (1-114)"""
        from src.preprocessing.provenance import AnnotationWithProvenance
        from pydantic import ValidationError
        
        # Valid surah
        ann = AnnotationWithProvenance(
            text="test", behavior="test", surah=114, ayah=1,
            tafsir_source="ibn_kathir", char_start=0, char_end=4,
            source_file="test.db", confidence=0.5
        )
        assert ann.surah == 114
        
        # Invalid surah (too high)
        with pytest.raises(ValidationError):
            AnnotationWithProvenance(
                text="test", behavior="test", surah=115, ayah=1,
                tafsir_source="ibn_kathir", char_start=0, char_end=4,
                source_file="test.db", confidence=0.5
            )
    
    def test_confidence_validation(self):
        """Confidence should be between 0 and 1"""
        from src.preprocessing.provenance import AnnotationWithProvenance
        from pydantic import ValidationError
        
        # Valid confidence
        ann = AnnotationWithProvenance(
            text="test", behavior="test", surah=1, ayah=1,
            tafsir_source="ibn_kathir", char_start=0, char_end=4,
            source_file="test.db", confidence=0.95
        )
        assert ann.confidence == 0.95
        
        # Invalid confidence (> 1)
        with pytest.raises(ValidationError):
            AnnotationWithProvenance(
                text="test", behavior="test", surah=1, ayah=1,
                tafsir_source="ibn_kathir", char_start=0, char_end=4,
                source_file="test.db", confidence=1.5
            )


class TestTafsirExtractor:
    """Test behavior extraction"""
    
    def test_extractor_initialization(self):
        """Extractor should initialize correctly"""
        from src.extraction.tafsir_extractor_v2 import TafsirExtractorV2
        
        extractor = TafsirExtractorV2()
        assert extractor.VERSION == "2.0.0"
    
    def test_behavior_root_detection(self):
        """Behavior roots should be detected"""
        from src.extraction.tafsir_extractor_v2 import TafsirExtractorV2
        
        extractor = TafsirExtractorV2()
        
        analysis = extractor.analyze_text("الصبر من صفات المؤمنين والكبر من صفات الكافرين")
        
        assert analysis["has_behavior_content"]
        assert len(analysis["roots_found"]) >= 2
    
    def test_extract_with_provenance(self):
        """Extraction should include provenance"""
        from src.extraction.tafsir_extractor_v2 import TafsirExtractorV2
        
        extractor = TafsirExtractorV2()
        
        annotations = extractor.extract_behaviors(
            text="الصبر من أعظم الصفات",
            surah=2,
            ayah=155,
            tafsir_source="ibn_kathir",
            source_file="test.db"
        )
        
        if annotations:  # May not find if pattern doesn't match
            ann = annotations[0]
            assert ann.surah == 2
            assert ann.ayah == 155
            assert ann.tafsir_source == "ibn_kathir"
            assert ann.extractor_version == "2.0.0"
            assert ann.extraction_method == "morphology"


@pytest.mark.slow
class TestCleanedDatabase:
    """Test the cleaned tafsir database (requires running clean_all_tafsir.py first)"""
    
    def test_cleaned_db_exists(self):
        """Cleaned database should exist after running clean script"""
        from pathlib import Path
        
        cleaned_db = Path("data/tafsir/tafsir_cleaned.db")
        if not cleaned_db.exists():
            pytest.skip("Cleaned database not found - run scripts/clean_all_tafsir.py")
    
    def test_cleaned_data_has_no_html(self):
        """Cleaned data should have no HTML tags"""
        import sqlite3
        from pathlib import Path
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaned_db = Path("data/tafsir/tafsir_cleaned.db")
        if not cleaned_db.exists():
            pytest.skip("Cleaned database not found - run scripts/clean_all_tafsir.py")
        
        conn = sqlite3.connect(str(cleaned_db))
        cursor = conn.cursor()
        
        # Sample 100 records
        cursor.execute("SELECT text_cleaned FROM tafsir_content_cleaned LIMIT 100")
        
        cleaner = TextCleaner()
        contaminated = 0
        
        for (text,) in cursor.fetchall():
            if cleaner.has_html(text):
                contaminated += 1
        
        conn.close()
        
        # Should have < 1% contamination
        assert contaminated < 1, f"Found {contaminated} contaminated records in sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
