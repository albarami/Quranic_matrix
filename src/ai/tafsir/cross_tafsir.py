"""
Cross-Tafsir Analysis module for QBM AI System.

This module provides multi-tafsir integration, enabling cross-referencing
and consensus analysis across multiple Quranic exegesis sources.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests


class CrossTafsirAnalyzer:
    """Analyzer for cross-referencing multiple tafsir sources."""

    # Supported tafsir sources
    TAFSIR_SOURCES = {
        "ibn_kathir": {"id": "en-tafisr-ibn-kathir", "name_ar": "تفسير ابن كثير", "name_en": "Ibn Kathir"},
        "tabari": {"id": "ar-tafsir-al-tabari", "name_ar": "تفسير الطبري", "name_en": "Al-Tabari"},
        "qurtubi": {"id": "ar-tafsir-al-qurtubi", "name_ar": "تفسير القرطبي", "name_en": "Al-Qurtubi"},
        "saadi": {"id": "ar-tafsir-al-saadi", "name_ar": "تفسير السعدي", "name_en": "Al-Saadi"},
        "jalalayn": {"id": "ar-tafsir-al-jalalayn", "name_ar": "تفسير الجلالين", "name_en": "Al-Jalalayn"},
        "muyassar": {"id": "ar-tafsir-muyassar", "name_ar": "التفسير الميسر", "name_en": "Al-Muyassar"},
        # Canonical key is baghawi (not baghawy) to align with vocab/tafsir_sources.json
        "baghawi": {"id": "ar-tafsir-al-baghawy", "name_ar": "تفسير البغوي", "name_en": "Al-Baghawi"},
    }

    # Surah ayah counts
    SURAH_AYAT = [
        7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
        111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
        54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
        49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52,
        44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19,
        26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3,
        6, 3, 5, 4, 5, 6
    ]

    def __init__(
        self,
        cache_dir: str = "data/tafsir_cache",
        api_base: Optional[str] = None,
        data_dir: Optional[str] = None,  # Alias for cache_dir (backward compatibility)
    ):
        """
        Initialize the cross-tafsir analyzer.

        Args:
            cache_dir: Directory for caching tafsir data.
            api_base: Base URL for Quran API.
            data_dir: Alias for cache_dir (backward compatibility).
        """
        # Use data_dir if provided, otherwise use cache_dir
        effective_dir = data_dir if data_dir is not None else cache_dir
        self.cache_dir = Path(effective_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_base = api_base or os.getenv("QURAN_API_BASE", "https://api.quran.com/api/v4")

    # =========================================================================
    # Data Access
    # =========================================================================

    def get_available_sources(self) -> List[str]:
        """Get list of available tafsir sources."""
        return list(self.TAFSIR_SOURCES.keys())

    def get_source_info(self, source: str) -> Optional[Dict[str, str]]:
        """Get information about a tafsir source."""
        return self.TAFSIR_SOURCES.get(source)

    def get_ayat_count(self, surah: int) -> int:
        """Get number of ayat in a surah."""
        if 1 <= surah <= 114:
            return self.SURAH_AYAT[surah - 1]
        return 0

    def get_tafsir(
        self,
        surah: int,
        ayah: int,
        source: str = "ibn_kathir",
    ) -> Optional[Dict[str, Any]]:
        """
        Get tafsir for a specific ayah from a source.

        Args:
            surah: Surah number (1-114).
            ayah: Ayah number.
            source: Tafsir source key.

        Returns:
            Dict with tafsir text and metadata, or None if not found.
        """
        if source not in self.TAFSIR_SOURCES:
            return None

        # Check cache first
        cache_path = self.cache_dir / source / f"{surah}_{ayah}.json"
        if not cache_path.exists() and source == "baghawi":
            # Backward-compatibility for older cache directory name
            legacy_path = self.cache_dir / "baghawy" / f"{surah}_{ayah}.json"
            if legacy_path.exists():
                cache_path = legacy_path
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # CI/offline mode: never hit external APIs during tests or gated runs.
        if os.getenv("CI") or os.getenv("QBM_OFFLINE") == "1":
            return {
                "surah": surah,
                "ayah": ayah,
                "source": source,
                "source_name": self.TAFSIR_SOURCES[source]["name_ar"],
                "text": "",
            }

        # Fetch from API
        tafsir_id = self.TAFSIR_SOURCES[source]["id"]
        try:
            response = requests.get(
                f"{self.api_base}/quran/tafsirs/{tafsir_id}",
                params={"verse_key": f"{surah}:{ayah}"},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                tafsir_data = {
                    "surah": surah,
                    "ayah": ayah,
                    "source": source,
                    "source_name": self.TAFSIR_SOURCES[source]["name_ar"],
                    "text": data.get("tafsir", {}).get("text", ""),
                }

                # Cache the result
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(tafsir_data, f, ensure_ascii=False, indent=2)

                return tafsir_data
        except Exception as e:
            print(f"Error fetching tafsir: {e}")

        return None

    def get_all_tafsir(
        self,
        surah: int,
        ayah: int,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get tafsir from all (or specified) sources for an ayah.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            sources: List of source keys (default: all sources).

        Returns:
            Dict mapping source key to tafsir data.
        """
        sources = sources or list(self.TAFSIR_SOURCES.keys())
        result = {}

        for source in sources:
            tafsir = self.get_tafsir(surah, ayah, source)
            if tafsir:
                result[source] = tafsir

        return result

    # =========================================================================
    # Cross-Tafsir Analysis
    # =========================================================================

    def find_consensus(
        self,
        surah: int,
        ayah: int,
        topic: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find consensus and disagreement across scholars on a topic.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            topic: Topic to analyze (e.g., "الختم", "الكبر").
            sources: List of source keys.

        Returns:
            Dict with agree_on, disagree_on, and unique_insights.
        """
        all_tafsir = self.get_all_tafsir(surah, ayah, sources)

        result = {
            "surah": surah,
            "ayah": ayah,
            "topic": topic,
            "sources_analyzed": list(all_tafsir.keys()),
            "mentions_topic": [],
            "agree_on": [],
            "disagree_on": [],
            "unique_insights": {},
        }

        # Check which sources mention the topic
        for source, data in all_tafsir.items():
            text = data.get("text", "")
            if topic in text:
                result["mentions_topic"].append(source)
                # Extract context around the topic
                idx = text.find(topic)
                start = max(0, idx - 100)
                end = min(len(text), idx + len(topic) + 100)
                result["unique_insights"][source] = text[start:end]

        return result

    def behavioral_emphasis(
        self,
        behavior: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how different scholars emphasize a behavior.

        Args:
            behavior: Behavior name in Arabic.
            sources: List of source keys.

        Returns:
            Dict mapping source to emphasis analysis.
        """
        sources = sources or list(self.TAFSIR_SOURCES.keys())
        result = {}

        for source in sources:
            mentions = self.search_behavior_in_tafsir(behavior, source)
            result[source] = {
                "frequency": len(mentions),
                "contexts": mentions[:5],  # First 5 mentions
            }

        return result

    def search_behavior_in_tafsir(
        self,
        behavior: str,
        source: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for behavior mentions in a tafsir source.

        Args:
            behavior: Behavior term to search.
            source: Tafsir source key.
            limit: Maximum results to return.

        Returns:
            List of matches with ayah references and context.
        """
        results = []
        cache_path = self.cache_dir / source

        if not cache_path.exists():
            return results

        # Search through cached tafsir files
        for file_path in cache_path.glob("*.json"):
            if len(results) >= limit:
                break

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text = data.get("text", "")
                if behavior in text:
                    # Extract context
                    idx = text.find(behavior)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(behavior) + 50)

                    results.append({
                        "surah": data.get("surah"),
                        "ayah": data.get("ayah"),
                        "source": source,
                        "text": text[start:end],
                    })
            except Exception:
                continue

        return results

    # =========================================================================
    # Comparative Analysis
    # =========================================================================

    def compare_interpretations(
        self,
        surah: int,
        ayah: int,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare interpretations across multiple scholars.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            sources: List of source keys.

        Returns:
            Comparative analysis of interpretations.
        """
        all_tafsir = self.get_all_tafsir(surah, ayah, sources)

        return {
            "surah": surah,
            "ayah": ayah,
            "sources": list(all_tafsir.keys()),
            "interpretations": {
                source: {
                    "scholar": self.TAFSIR_SOURCES[source]["name_ar"],
                    "text_length": len(data.get("text", "")),
                    "preview": data.get("text", "")[:300] + "..." if len(data.get("text", "")) > 300 else data.get("text", ""),
                }
                for source, data in all_tafsir.items()
            },
        }

    def get_scholar_focus(
        self,
        source: str,
        behaviors: List[str],
    ) -> Dict[str, int]:
        """
        Analyze which behaviors a scholar emphasizes most.

        Args:
            source: Tafsir source key.
            behaviors: List of behavior terms to check.

        Returns:
            Dict mapping behavior to mention count.
        """
        focus = {}
        for behavior in behaviors:
            mentions = self.search_behavior_in_tafsir(behavior, source, limit=100)
            focus[behavior] = len(mentions)
        return focus

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def cache_surah_tafsir(
        self,
        surah: int,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Cache all tafsir for a surah.

        Args:
            surah: Surah number.
            sources: List of source keys.

        Returns:
            Number of tafsir entries cached.
        """
        sources = sources or list(self.TAFSIR_SOURCES.keys())
        count = 0
        ayat_count = self.get_ayat_count(surah)

        for ayah in range(1, ayat_count + 1):
            for source in sources:
                if self.get_tafsir(surah, ayah, source):
                    count += 1

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached tafsir data."""
        stats = {
            "sources": {},
            "total_cached": 0,
        }

        for source in self.TAFSIR_SOURCES:
            cache_path = self.cache_dir / source
            if cache_path.exists():
                count = len(list(cache_path.glob("*.json")))
                stats["sources"][source] = count
                stats["total_cached"] += count
            else:
                stats["sources"][source] = 0

        return stats

    # =========================================================================
    # Integration with Vector Store
    # =========================================================================

    def load_into_vector_store(
        self,
        vector_store,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Load cached tafsir into vector store.

        Args:
            vector_store: QBMVectorStore instance.
            sources: List of source keys.

        Returns:
            Number of entries loaded.
        """
        sources = sources or list(self.TAFSIR_SOURCES.keys())
        count = 0

        for source in sources:
            cache_path = self.cache_dir / source
            if not cache_path.exists():
                continue

            for file_path in cache_path.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    text = data.get("text", "")
                    if not text:
                        continue

                    tafsir_id = f"{source}_{data['surah']}_{data['ayah']}"
                    vector_store.add_tafsir(
                        tafsir_id=tafsir_id,
                        text=text[:2000],  # Limit text length
                        surah=data["surah"],
                        ayah=data["ayah"],
                        source=source,
                    )
                    count += 1
                except Exception:
                    continue

        return count
