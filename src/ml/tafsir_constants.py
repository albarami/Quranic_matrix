"""
Canonical Tafsir Sources - Single Source of Truth

All modules MUST import tafsir source lists from here.
DO NOT hardcode tafsir source lists anywhere else.

Sources are loaded from vocab/tafsir_sources.json (7 sources total):
- 5 core: ibn_kathir, tabari, qurtubi, saadi, jalalayn
- 2 supplementary: baghawi, muyassar
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

TAFSIR_SOURCES_FILE = Path("vocab/tafsir_sources.json")

def _load_canonical_tafsir_sources() -> List[str]:
    """
    Load all 7 tafsir source IDs from vocab/tafsir_sources.json.
    
    Returns:
        List of 7 tafsir source IDs (5 core + 2 supplementary)
    """
    if TAFSIR_SOURCES_FILE.exists():
        try:
            with open(TAFSIR_SOURCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            core = [s["id"] for s in data.get("core_sources", [])]
            supplementary = [s["id"] for s in data.get("supplementary_sources", [])]
            sources = core + supplementary
            if len(sources) == 7:
                return sources
            logger.warning(f"Expected 7 tafsir sources, got {len(sources)}")
        except Exception as e:
            logger.warning(f"Failed to load tafsir sources from {TAFSIR_SOURCES_FILE}: {e}")
    
    # Fallback to hardcoded 7 if file unavailable
    return ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]


# Canonical constants - import these, don't hardcode
CANONICAL_TAFSIR_SOURCES: List[str] = _load_canonical_tafsir_sources()
TAFSIR_SOURCE_COUNT: int = len(CANONICAL_TAFSIR_SOURCES)

# For modules that need quran + tafsir
ALL_SOURCES_WITH_QURAN: List[str] = CANONICAL_TAFSIR_SOURCES + ["quran"]
