#!/usr/bin/env python3
"""
Tafsir Store - Single Source of Truth loader.

Loads tafsir JSONL files into an in-memory index keyed by verse_key.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

DEFAULT_TAFSIR_DIR = Path("data/tafsir")


@dataclass
class TafsirEntry:
    """Single tafsir entry bound to a verse key."""

    source: str
    surah: int
    ayah: int
    text_ar: str
    entry_type: str = "verse"

    @property
    def verse_key(self) -> str:
        return f"{self.surah}:{self.ayah}"

    def __post_init__(self) -> None:
        if self.ayah == 0 and self.entry_type == "verse":
            self.entry_type = "surah_intro"


@dataclass
class TafsirStore:
    """
    In-memory tafsir store keyed by source and verse_key.
    """

    entries_by_source: Dict[str, Dict[str, TafsirEntry]] = field(default_factory=dict)
    total_entries: int = 0
    source_path: str = ""
    missing_sources: List[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        sources: Optional[List[str]] = None,
        tafsir_dir: Optional[Path] = None,
    ) -> "TafsirStore":
        if sources is None:
            sources = list(CANONICAL_TAFSIR_SOURCES)
        if tafsir_dir is None:
            tafsir_dir = DEFAULT_TAFSIR_DIR

        store = cls()
        store.source_path = str(tafsir_dir)

        for source in sources:
            path = tafsir_dir / f"{source}.ar.jsonl"
            if not path.exists():
                store.missing_sources.append(source)
                continue

            entries: Dict[str, TafsirEntry] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ref = record.get("reference", {})
                    surah = ref.get("surah")
                    ayah = ref.get("ayah")
                    if surah is None or ayah is None:
                        continue

                    text_ar = record.get("text_ar") or record.get("text") or ""
                    entry_type = "surah_intro" if ayah == 0 else "verse"
                    entry = TafsirEntry(
                        source=source,
                        surah=int(surah),
                        ayah=int(ayah),
                        text_ar=text_ar,
                        entry_type=entry_type,
                    )
                    entries[entry.verse_key] = entry

            store.entries_by_source[source] = entries
            store.total_entries += len(entries)

        return store

    def get(self, source: str, verse_key: str) -> Optional[TafsirEntry]:
        return self.entries_by_source.get(source, {}).get(verse_key)

    def get_text(self, source: str, verse_key: str, default: str = "") -> str:
        entry = self.get(source, verse_key)
        return entry.text_ar if entry else default

    def get_verse(self, verse_key: str) -> Dict[str, TafsirEntry]:
        return {
            source: entries[verse_key]
            for source, entries in self.entries_by_source.items()
            if verse_key in entries
        }

    def get_sources(self) -> List[str]:
        return list(self.entries_by_source.keys())

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_entries": self.total_entries,
            "sources_loaded": len(self.entries_by_source),
            "missing_sources": len(self.missing_sources),
        }


_store: Optional[TafsirStore] = None


def get_tafsir_store(
    sources: Optional[List[str]] = None,
    tafsir_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> TafsirStore:
    global _store

    if _store is None or force_reload:
        _store = TafsirStore.load(sources=sources, tafsir_dir=tafsir_dir)

    return _store


def clear_store() -> None:
    global _store
    _store = None


if __name__ == "__main__":
    store = get_tafsir_store()
    stats = store.get_statistics()
    print(f"Tafsir sources loaded: {stats['sources_loaded']}")
    print(f"Total entries: {stats['total_entries']}")
    if store.missing_sources:
        print(f"Missing sources: {', '.join(store.missing_sources)}")
