"""
Pattern Discovery for QBM Discovery System.

Discovers patterns in behavioral annotations across the Quran,
including co-occurrence, sequences, and thematic patterns.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class PatternDiscovery:
    """
    Discover patterns in Quranic behavioral annotations.
    
    Features:
    - Behavior co-occurrence analysis
    - Surah-level behavior patterns
    - Temporal/sequential patterns
    - Cause-effect relationship discovery
    """

    def __init__(
        self,
        annotations_path: str = "data/annotations/tafsir_annotations.jsonl",
        graph_path: str = "data/graph/qbm_graph.db",
    ):
        """
        Initialize pattern discovery.

        Args:
            annotations_path: Path to annotations file.
            graph_path: Path to knowledge graph database.
        """
        self.annotations_path = Path(annotations_path)
        self.graph_path = Path(graph_path)

        self.annotations: List[Dict] = []
        self.behavior_index: Dict[str, List[int]] = defaultdict(list)
        self.type_index: Dict[str, List[int]] = defaultdict(list)
        self.ayah_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.surah_index: Dict[int, List[int]] = defaultdict(list)

        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load and index annotations."""
        if not self.annotations_path.exists():
            print(f"Warning: Annotations not found at {self.annotations_path}")
            return

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ann = json.loads(line)
                self.annotations.append(ann)

                # Index by behavior value and Arabic name
                ann_type = ann.get("type", "")
                value = ann.get("value", "")
                metadata = ann.get("metadata", {})
                name_ar = metadata.get("name_ar", "") if isinstance(metadata, dict) else ""
                
                if value:
                    self.behavior_index[value].append(i)
                if name_ar:
                    self.behavior_index[name_ar].append(i)
                if ann_type:
                    self.type_index[ann_type].append(i)

                # Index by ayah
                surah = ann.get("surah", 0)
                ayah = ann.get("ayah", 0)
                if surah and ayah:
                    self.ayah_index[(surah, ayah)].append(i)
                    self.surah_index[surah].append(i)

        print(f"Loaded {len(self.annotations)} annotations")
        print(f"  Unique behaviors: {len(self.behavior_index)}")
        print(f"  Unique ayat: {len(self.ayah_index)}")
        print(f"  Surahs covered: {len(self.surah_index)}")

    def find_cooccurring_behaviors(
        self,
        min_cooccurrence: int = 5,
        same_ayah: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find behaviors that frequently co-occur.

        Args:
            min_cooccurrence: Minimum co-occurrence count.
            same_ayah: If True, count co-occurrence in same ayah only.

        Returns:
            List of co-occurring behavior pairs with counts.
        """
        cooccurrence = Counter()

        if same_ayah:
            # Count behaviors appearing in same ayah
            for (surah, ayah), indices in self.ayah_index.items():
                behaviors = set()
                for idx in indices:
                    # Use 'value' field which contains behavior ID
                    behavior = self.annotations[idx].get("value", "")
                    if behavior:
                        behaviors.add(behavior)

                # Count pairs
                behaviors = sorted(behaviors)
                for i, b1 in enumerate(behaviors):
                    for b2 in behaviors[i + 1:]:
                        cooccurrence[(b1, b2)] += 1
        else:
            # Count behaviors appearing in same surah
            for surah, indices in self.surah_index.items():
                behaviors = set()
                for idx in indices:
                    # Use 'value' field which contains behavior ID
                    behavior = self.annotations[idx].get("value", "")
                    if behavior:
                        behaviors.add(behavior)

                behaviors = sorted(behaviors)
                for i, b1 in enumerate(behaviors):
                    for b2 in behaviors[i + 1:]:
                        cooccurrence[(b1, b2)] += 1

        # Filter and format results
        results = []
        for (b1, b2), count in cooccurrence.most_common():
            if count >= min_cooccurrence:
                results.append({
                    "behavior_1": b1,
                    "behavior_2": b2,
                    "cooccurrence_count": count,
                    "scope": "ayah" if same_ayah else "surah",
                })

        return results

    def find_behavior_sequences(
        self,
        window_size: int = 3,
        min_frequency: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find sequential patterns of behaviors across ayat.

        Args:
            window_size: Number of consecutive ayat to consider.
            min_frequency: Minimum frequency for a pattern.

        Returns:
            List of behavior sequences with frequencies.
        """
        sequences = Counter()

        for surah in range(1, 115):
            if surah not in self.surah_index:
                continue

            # Get behaviors by ayah order
            ayah_behaviors = defaultdict(set)
            for idx in self.surah_index[surah]:
                ann = self.annotations[idx]
                ayah = ann.get("ayah", 0)
                # Use 'value' field which contains behavior ID
                behavior = ann.get("value", "")
                if behavior:
                    ayah_behaviors[ayah].add(behavior)

            # Find sequences
            ayahs = sorted(ayah_behaviors.keys())
            for i in range(len(ayahs) - window_size + 1):
                window_ayahs = ayahs[i:i + window_size]

                # Check if consecutive
                if window_ayahs[-1] - window_ayahs[0] == window_size - 1:
                    seq = []
                    for a in window_ayahs:
                        seq.extend(sorted(ayah_behaviors[a]))
                    if len(seq) >= 2:
                        sequences[tuple(seq)] += 1

        # Filter and format
        results = []
        for seq, count in sequences.most_common():
            if count >= min_frequency:
                results.append({
                    "sequence": list(seq),
                    "length": len(seq),
                    "frequency": count,
                    "window_size": window_size,
                })

        return results

    def find_surah_themes(
        self,
        top_n: int = 5,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Find dominant behavioral themes for each surah.

        Args:
            top_n: Number of top behaviors per surah.

        Returns:
            Dict mapping surah number to top behaviors.
        """
        surah_themes = {}

        for surah in range(1, 115):
            if surah not in self.surah_index:
                continue

            behavior_counts = Counter()
            for idx in self.surah_index[surah]:
                # Use 'value' field which contains behavior ID
                behavior = self.annotations[idx].get("value", "")
                if behavior:
                    behavior_counts[behavior] += 1

            total = sum(behavior_counts.values())
            themes = []
            for behavior, count in behavior_counts.most_common(top_n):
                themes.append({
                    "behavior": behavior,
                    "count": count,
                    "percentage": round(count / total * 100, 1) if total > 0 else 0,
                })

            surah_themes[surah] = themes

        return surah_themes

    def find_opposite_behaviors(
        self,
        min_distance: int = 1,
        max_distance: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find behaviors that appear near their opposites.

        Args:
            min_distance: Minimum ayah distance.
            max_distance: Maximum ayah distance.

        Returns:
            List of opposite behavior pairs found near each other.
        """
        # Known opposite pairs (from behavior taxonomy)
        opposite_pairs = [
            ("الإيمان", "الكفر"),  # faith vs disbelief
            ("التوبة", "الإصرار"),  # repentance vs persistence in sin
            ("الصبر", "الجزع"),  # patience vs panic
            ("الشكر", "الكفران"),  # gratitude vs ingratitude
            ("التواضع", "الكبر"),  # humility vs arrogance
            ("الصدق", "الكذب"),  # truthfulness vs lying
            ("الأمانة", "الخيانة"),  # trustworthiness vs betrayal
            ("العدل", "الظلم"),  # justice vs oppression
            ("الرحمة", "القسوة"),  # mercy vs cruelty
            ("الحب", "البغض"),  # love vs hatred
        ]

        results = []

        for surah in range(1, 115):
            if surah not in self.surah_index:
                continue

            # Get behaviors by ayah
            ayah_behaviors = defaultdict(set)
            for idx in self.surah_index[surah]:
                ann = self.annotations[idx]
                ayah = ann.get("ayah", 0)
                # Use 'value' field which contains behavior ID
                behavior = ann.get("value", "")
                if behavior:
                    ayah_behaviors[ayah].add(behavior)

            # Check for opposite pairs
            ayahs = sorted(ayah_behaviors.keys())
            for i, ayah1 in enumerate(ayahs):
                for ayah2 in ayahs[i + 1:]:
                    distance = ayah2 - ayah1
                    if distance < min_distance or distance > max_distance:
                        continue

                    for b1, b2 in opposite_pairs:
                        if (b1 in ayah_behaviors[ayah1] and b2 in ayah_behaviors[ayah2]) or \
                           (b2 in ayah_behaviors[ayah1] and b1 in ayah_behaviors[ayah2]):
                            results.append({
                                "behavior_1": b1,
                                "behavior_2": b2,
                                "surah": surah,
                                "ayah_1": ayah1,
                                "ayah_2": ayah2,
                                "distance": distance,
                            })

        return results

    def find_cause_effect_patterns(self) -> List[Dict[str, Any]]:
        """
        Find cause-effect patterns from relationship annotations.

        Returns:
            List of cause-effect patterns with evidence.
        """
        cause_effect = defaultdict(list)

        for ann in self.annotations:
            if ann.get("type") == "relationship":
                rel_type = ann.get("relationship_type", "")
                if rel_type in ["causes", "leads_to", "results_in"]:
                    cause = ann.get("source_behavior", "")
                    effect = ann.get("target_behavior", "")
                    if cause and effect:
                        cause_effect[(cause, effect)].append({
                            "surah": ann.get("surah"),
                            "ayah": ann.get("ayah"),
                            "source": ann.get("source"),
                            "context": ann.get("context", "")[:200],
                        })

        results = []
        for (cause, effect), evidence in cause_effect.items():
            results.append({
                "cause": cause,
                "effect": effect,
                "evidence_count": len(evidence),
                "evidence": evidence[:5],  # Limit to 5 examples
            })

        results.sort(key=lambda x: x["evidence_count"], reverse=True)
        return results

    def get_behavior_distribution(
        self,
        behavior_only: bool = False,
    ) -> Dict[str, int]:
        """
        Get distribution of values across all annotations.

        Args:
            behavior_only: If True, only include annotations with type='behavior'.

        Returns:
            Dict mapping value to count.
        """
        distribution = Counter()
        for ann in self.annotations:
            if behavior_only and ann.get("type") != "behavior":
                continue
            # Use 'value' field which contains behavior ID
            value = ann.get("value", "")
            if value:
                distribution[value] += 1
        return dict(distribution.most_common())

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern discovery statistics."""
        return {
            "total_annotations": len(self.annotations),
            "unique_behaviors": len(self.behavior_index),
            "unique_ayat": len(self.ayah_index),
            "surahs_covered": len(self.surah_index),
            "behavior_distribution": self.get_behavior_distribution(),
        }
