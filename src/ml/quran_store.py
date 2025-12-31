"""
Quran Store: Unified Quran text access with fixture mode support.

Resolution order:
1. If QBM_USE_FIXTURE=1: Load from fixture verse index (CI mode)
2. Else if tokenized JSON exists: Load from JSON SSOT
3. Else: Fall back to XML (last resort)

This eliminates the hard XML dependency in CI environments.
"""

import json
import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FIXTURE_VERSE_INDEX = PROJECT_ROOT / "data" / "indexes" / "tafsir" / "verse_index.json"
QURAN_JSON_FILE = PROJECT_ROOT / "data" / "quran" / "uthmani_hafs_v1.tok_v1.json"
QURAN_XML_FILE = PROJECT_ROOT / "data" / "quran" / "quran-uthmani.xml"

# Check for fixture mode
USE_FIXTURE = os.getenv("QBM_USE_FIXTURE", "0") == "1"


class QuranStore:
    """
    Unified Quran text store with multiple backends.
    
    Supports:
    - Fixture mode (CI): Uses pre-built verse index
    - JSON mode (preferred): Uses tokenized JSON SSOT
    - XML mode (fallback): Uses XML file
    """
    
    _instance = None
    
    def __init__(self):
        self.verses: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._source = None
    
    @classmethod
    def get_instance(cls) -> "QuranStore":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load(self) -> None:
        """Load Quran verses from the best available source."""
        if self._loaded:
            return
        
        # Try sources in order of preference for CI
        if USE_FIXTURE and FIXTURE_VERSE_INDEX.exists():
            self._load_from_fixture()
        elif QURAN_JSON_FILE.exists():
            self._load_from_json()
        elif QURAN_XML_FILE.exists():
            self._load_from_xml()
        else:
            if USE_FIXTURE:
                logger.warning("QBM_USE_FIXTURE=1 but no fixture verse index found")
                # In fixture mode, create empty store rather than fail
                self._source = "empty_fixture"
                self._loaded = True
            else:
                raise FileNotFoundError(
                    f"No Quran source found. Tried:\n"
                    f"  - Fixture: {FIXTURE_VERSE_INDEX}\n"
                    f"  - JSON: {QURAN_JSON_FILE}\n"
                    f"  - XML: {QURAN_XML_FILE}"
                )
    
    def _load_from_fixture(self) -> None:
        """Load from CI fixture verse index."""
        logger.info(f"Loading Quran from fixture: {FIXTURE_VERSE_INDEX}")
        
        with open(FIXTURE_VERSE_INDEX, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Fixture format: {"1:1": {"surah": 1, "ayah": 1, "text": "...", "key": "1:1"}, ...}
        for key, verse in data.items():
            self.verses[key] = {
                'verse_key': key,
                'surah': verse.get('surah'),
                'ayah': verse.get('ayah'),
                'text': verse.get('text', ''),
            }
        
        self._source = "fixture"
        self._loaded = True
        logger.info(f"Loaded {len(self.verses)} verses from fixture")
    
    def _load_from_json(self) -> None:
        """Load from tokenized JSON SSOT."""
        logger.info(f"Loading Quran from JSON: {QURAN_JSON_FILE}")
        
        with open(QURAN_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for surah in data.get('surahs', []):
            surah_num = surah.get('surah')
            for ayah in surah.get('ayat', []):
                ayah_num = ayah.get('ayah')
                text = ayah.get('text', '')
                key = f"{surah_num}:{ayah_num}"
                
                self.verses[key] = {
                    'verse_key': key,
                    'surah': surah_num,
                    'ayah': ayah_num,
                    'text': text,
                }
        
        self._source = "json"
        self._loaded = True
        logger.info(f"Loaded {len(self.verses)} verses from JSON")
    
    def _load_from_xml(self) -> None:
        """Load from XML file (fallback)."""
        logger.info(f"Loading Quran from XML: {QURAN_XML_FILE}")
        
        tree = ET.parse(QURAN_XML_FILE)
        root = tree.getroot()
        
        for sura in root.findall('sura'):
            surah_num = int(sura.get('index'))
            
            for aya in sura.findall('aya'):
                ayah_num = int(aya.get('index'))
                text = aya.get('text', '')
                key = f"{surah_num}:{ayah_num}"
                
                self.verses[key] = {
                    'verse_key': key,
                    'surah': surah_num,
                    'ayah': ayah_num,
                    'text': text,
                }
        
        self._source = "xml"
        self._loaded = True
        logger.info(f"Loaded {len(self.verses)} verses from XML")
    
    def get_verse_text(self, verse_key: str) -> Optional[str]:
        """Get verse text by key (e.g., '2:255')."""
        self.load()
        verse = self.verses.get(verse_key)
        return verse['text'] if verse else None
    
    def get_verse(self, verse_key: str) -> Optional[Dict[str, Any]]:
        """Get full verse data by key."""
        self.load()
        return self.verses.get(verse_key)
    
    def get_all_verses(self) -> List[Dict[str, Any]]:
        """Get all verses."""
        self.load()
        return list(self.verses.values())
    
    def get_verse_count(self) -> int:
        """Get total verse count."""
        self.load()
        return len(self.verses)
    
    def get_source(self) -> str:
        """Get the data source used."""
        self.load()
        return self._source
    
    def has_verse(self, verse_key: str) -> bool:
        """Check if verse exists."""
        self.load()
        return verse_key in self.verses


# Convenience functions
def get_quran_text(verse_key: str) -> Optional[str]:
    """Get Quran verse text by key."""
    return QuranStore.get_instance().get_verse_text(verse_key)


def get_quran_verse(verse_key: str) -> Optional[Dict[str, Any]]:
    """Get full Quran verse data by key."""
    return QuranStore.get_instance().get_verse(verse_key)


def get_quran_store() -> QuranStore:
    """Get the QuranStore singleton."""
    return QuranStore.get_instance()
