"""
VerseResolver: Two-stage retrieval for verse text → verse key resolution.

Phase 5.5: Implements deterministic verse resolution for Scenario 1.

Pipeline:
1. Query = Arabic verse text (normalized)
2. BM25 search over Quran verse index (6,236 verses)
3. Return top N verse keys with confidence scores
4. If confident, use resolved verse key for tafsir retrieval

CI Fixture Mode (QBM_USE_FIXTURE=1):
- Uses QuranStore which loads from fixture verse index
- No XML dependency in CI
"""

import json
import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from src.ml.quran_store import QuranStore, get_quran_store

# Paths
QURAN_XML_FILE = Path("data/quran/quran-uthmani.xml")
QURAN_INDEX_FILE = Path("data/evidence/quran_verse_index.jsonl")

# Check for fixture mode
USE_FIXTURE = os.getenv("QBM_USE_FIXTURE", "0") == "1"

logger = logging.getLogger(__name__)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for matching.
    
    Removes diacritics, normalizes letter forms.
    """
    if not text:
        return text
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Normalize Alef forms (أ إ آ ا ٱ → ا)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    
    # Normalize Yaa forms (ى → ي)
    text = re.sub(r'ى', 'ي', text)
    
    # Normalize Taa Marbuta (ة → ه)
    text = re.sub(r'ة', 'ه', text)
    
    # Remove tatweel (ـ)
    text = re.sub(r'ـ', '', text)
    
    # Normalize Hamza forms
    text = re.sub(r'[ؤئء]', 'ء', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


@dataclass
class VerseMatch:
    """A matched verse from the resolver."""
    verse_key: str
    surah: int
    ayah: int
    text: str
    text_normalized: str
    score: float
    rank: int


@dataclass
class ResolverResult:
    """Result from verse resolution."""
    query: str
    query_normalized: str
    matches: List[VerseMatch] = field(default_factory=list)
    top1_confident: bool = False
    confidence_margin: float = 0.0
    
    def get_verse_keys(self, top_n: int = 5) -> List[str]:
        """Get top N verse keys."""
        return [m.verse_key for m in self.matches[:top_n]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'query_normalized': self.query_normalized,
            'matches': [
                {
                    'verse_key': m.verse_key,
                    'surah': m.surah,
                    'ayah': m.ayah,
                    'score': round(m.score, 4),
                    'rank': m.rank,
                }
                for m in self.matches
            ],
            'top1_confident': self.top1_confident,
            'confidence_margin': round(self.confidence_margin, 4),
        }


class QuranVerseIndex:
    """In-memory index of Quran verses for resolution."""
    
    def __init__(self, xml_file: Path = QURAN_XML_FILE):
        self.xml_file = xml_file
        self.verses: List[Dict[str, Any]] = []
        self.by_verse_key: Dict[str, int] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load Quran verses from best available source."""
        if self._loaded:
            return
        
        # Use QuranStore in fixture mode (no XML dependency)
        if USE_FIXTURE:
            self._load_from_quran_store()
        else:
            self._load_from_xml()
    
    def _load_from_quran_store(self) -> None:
        """Load from QuranStore (fixture-aware)."""
        logger.info("Loading Quran verses from QuranStore (fixture mode)...")
        
        store = get_quran_store()
        store.load()
        
        for verse_data in store.get_all_verses():
            verse_key = verse_data.get('verse_key', f"{verse_data['surah']}:{verse_data['ayah']}")
            text = verse_data.get('text', '')
            text_normalized = normalize_arabic(text)
            
            self.verses.append({
                'verse_key': verse_key,
                'surah': verse_data.get('surah'),
                'ayah': verse_data.get('ayah'),
                'surah_name': '',
                'text': text,
                'text_normalized': text_normalized,
            })
            self.by_verse_key[verse_key] = len(self.verses) - 1
        
        self._loaded = True
        logger.info(f"Loaded {len(self.verses)} Quran verses from QuranStore")
    
    def _load_from_xml(self) -> None:
        """Load from XML file (original behavior)."""
        if not self.xml_file.exists():
            raise FileNotFoundError(f"Quran XML not found: {self.xml_file}")
        
        logger.info(f"Loading Quran verses from {self.xml_file}...")
        
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        for sura in root.findall('sura'):
            surah_num = int(sura.get('index'))
            surah_name = sura.get('name', '')
            
            for aya in sura.findall('aya'):
                ayah_num = int(aya.get('index'))
                text = aya.get('text', '')
                
                verse_key = f"{surah_num}:{ayah_num}"
                text_normalized = normalize_arabic(text)
                
                self.verses.append({
                    'verse_key': verse_key,
                    'surah': surah_num,
                    'ayah': ayah_num,
                    'surah_name': surah_name,
                    'text': text,
                    'text_normalized': text_normalized,
                })
                self.by_verse_key[verse_key] = len(self.verses) - 1
        
        self._loaded = True
        logger.info(f"Loaded {len(self.verses)} Quran verses")
    
    def get_all_texts(self) -> List[str]:
        """Get all normalized verse texts for BM25 indexing."""
        self.load()
        return [v['text_normalized'] for v in self.verses]
    
    def get_verse(self, verse_key: str) -> Optional[Dict[str, Any]]:
        """Get verse by key."""
        self.load()
        idx = self.by_verse_key.get(verse_key)
        if idx is not None:
            return self.verses[idx]
        return None


class VerseResolver:
    """
    Resolves Arabic verse text to verse key(s).
    
    Uses BM25 over Quran verses (6,236) for deterministic resolution.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.index = QuranVerseIndex()
        self.bm25 = None
        self.confidence_threshold = confidence_threshold
        self._built = False
    
    def build(self) -> None:
        """Build BM25 index over Quran verses."""
        if self._built:
            return
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 required for VerseResolver")
        
        self.index.load()
        
        logger.info("Building VerseResolver BM25 index...")
        
        texts = self.index.get_all_texts()
        tokenized = [self._tokenize(t) for t in texts]
        
        self.bm25 = BM25Okapi(tokenized)
        self._built = True
        
        logger.info(f"VerseResolver built with {len(texts)} verses")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize normalized Arabic text."""
        tokens = re.findall(r'[\u0600-\u06FF]+', text)
        return tokens
    
    def resolve(self, query: str, top_n: int = 5) -> ResolverResult:
        """
        Resolve verse text to verse key(s).
        
        Args:
            query: Arabic verse text
            top_n: Number of candidates to return
            
        Returns:
            ResolverResult with matched verses and confidence
        """
        if not self._built:
            self.build()
        
        query_normalized = normalize_arabic(query)
        query_tokens = self._tokenize(query_normalized)
        
        result = ResolverResult(
            query=query,
            query_normalized=query_normalized,
        )
        
        if not query_tokens:
            return result
        
        # BM25 search
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        for rank, idx in enumerate(top_indices, 1):
            verse = self.index.verses[idx]
            result.matches.append(VerseMatch(
                verse_key=verse['verse_key'],
                surah=verse['surah'],
                ayah=verse['ayah'],
                text=verse['text'],
                text_normalized=verse['text_normalized'],
                score=float(scores[idx]),
                rank=rank,
            ))
        
        # Compute confidence margin (top1 vs top2)
        if len(result.matches) >= 2:
            top1_score = result.matches[0].score
            top2_score = result.matches[1].score
            if top2_score > 0:
                result.confidence_margin = (top1_score - top2_score) / top2_score
            else:
                result.confidence_margin = 1.0
            
            result.top1_confident = result.confidence_margin >= self.confidence_threshold
        elif len(result.matches) == 1:
            result.top1_confident = True
            result.confidence_margin = 1.0
        
        return result


# Singleton instance
_verse_resolver: Optional[VerseResolver] = None


def get_verse_resolver() -> VerseResolver:
    """Get or create the singleton VerseResolver."""
    global _verse_resolver
    if _verse_resolver is None:
        _verse_resolver = VerseResolver()
        _verse_resolver.build()
    return _verse_resolver


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    resolver = get_verse_resolver()
    
    # Test with a known verse (Ayat al-Kursi)
    test_verses = [
        "ٱللَّهُ لَآ إِلَـٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ",  # 2:255
        "قُلْ هُوَ ٱللَّهُ أَحَدٌ",  # 112:1
        "إِنَّآ أَعْطَيْنَـٰكَ ٱلْكَوْثَرَ",  # 108:1
    ]
    
    print("\n" + "=" * 60)
    print("Testing VerseResolver")
    print("=" * 60)
    
    for verse_text in test_verses:
        result = resolver.resolve(verse_text)
        print(f"\nQuery: {verse_text[:50]}...")
        print(f"Top match: {result.matches[0].verse_key} (score={result.matches[0].score:.2f})")
        print(f"Confident: {result.top1_confident} (margin={result.confidence_margin:.2f})")
        for m in result.matches[:3]:
            print(f"  {m.rank}. {m.verse_key}: {m.text[:40]}... (score={m.score:.2f})")
