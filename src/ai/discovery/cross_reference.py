"""
Cross-Reference Finder for QBM Discovery System.

Finds related ayat, behaviors, and concepts across the Quran
using semantic similarity and knowledge graph relationships.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

import torch


class CrossReferenceFinder:
    """
    Find cross-references across Quranic annotations.
    
    Features:
    - Find ayat discussing same behavior
    - Find semantically similar passages
    - Find related concepts via knowledge graph
    - Build reference networks
    """

    def __init__(
        self,
        annotations_path: str = "data/annotations/tafsir_annotations.jsonl",
        embeddings_path: str = "data/embeddings/annotations_embeddings.npy",
        metadata_path: str = "data/embeddings/annotations_embeddings_metadata.json",
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize cross-reference finder.

        Args:
            annotations_path: Path to annotations file.
            embeddings_path: Path to embeddings file.
            metadata_path: Path to metadata file.
            similarity_threshold: Minimum similarity for cross-reference.
        """
        self.annotations_path = Path(annotations_path)
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.similarity_threshold = similarity_threshold

        self.annotations: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []

        # Indexes
        self.behavior_to_ayat: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
        self.ayah_to_behaviors: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self.ayah_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        self._load_data()

    def _load_data(self) -> None:
        """Load annotations and embeddings."""
        # Load annotations
        if self.annotations_path.exists():
            with open(self.annotations_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    ann = json.loads(line)
                    self.annotations.append(ann)

                    surah = ann.get("surah", 0)
                    ayah = ann.get("ayah", 0)
                    behavior = ann.get("behavior", "")

                    # Use 'value' and 'metadata.name_ar' for behavior names
                    value = ann.get("value", "")
                    metadata = ann.get("metadata", {})
                    name_ar = metadata.get("name_ar", "") if isinstance(metadata, dict) else ""
                    
                    if surah and ayah:
                        key = (surah, ayah)
                        self.ayah_to_indices[key].append(i)
                        if value:
                            self.behavior_to_ayat[value].add(key)
                            self.ayah_to_behaviors[key].add(value)
                        if name_ar:
                            self.behavior_to_ayat[name_ar].add(key)
                            self.ayah_to_behaviors[key].add(name_ar)

            print(f"Loaded {len(self.annotations)} annotations")

        # Load embeddings
        if self.embeddings_path.exists():
            self.embeddings = np.load(str(self.embeddings_path))
            print(f"Loaded embeddings: {self.embeddings.shape}")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def find_ayat_by_behavior(
        self,
        behavior: str,
        include_similar: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find all ayat discussing a specific behavior.

        Args:
            behavior: Behavior name (Arabic or English).
            include_similar: Include semantically similar behaviors.

        Returns:
            List of ayat with the behavior.
        """
        results = []
        seen_ayat = set()

        # Direct matches
        if behavior in self.behavior_to_ayat:
            for surah, ayah in self.behavior_to_ayat[behavior]:
                if (surah, ayah) not in seen_ayat:
                    seen_ayat.add((surah, ayah))
                    results.append({
                        "surah": surah,
                        "ayah": ayah,
                        "behavior": behavior,
                        "match_type": "exact",
                        "all_behaviors": list(self.ayah_to_behaviors[(surah, ayah)]),
                    })

        # Similar behaviors (partial match)
        if include_similar:
            for b in self.behavior_to_ayat:
                if behavior.lower() in b.lower() or b.lower() in behavior.lower():
                    if b != behavior:
                        for surah, ayah in self.behavior_to_ayat[b]:
                            if (surah, ayah) not in seen_ayat:
                                seen_ayat.add((surah, ayah))
                                results.append({
                                    "surah": surah,
                                    "ayah": ayah,
                                    "behavior": b,
                                    "match_type": "similar",
                                    "all_behaviors": list(self.ayah_to_behaviors[(surah, ayah)]),
                                })

        # Sort by surah and ayah
        results.sort(key=lambda x: (x["surah"], x["ayah"]))
        return results

    def find_similar_ayat(
        self,
        surah: int,
        ayah: int,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find ayat semantically similar to a given ayah.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            top_k: Number of similar ayat to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of similar ayat with similarity scores.
        """
        if self.embeddings is None:
            return []

        if min_similarity is None:
            min_similarity = self.similarity_threshold

        key = (surah, ayah)
        if key not in self.ayah_to_indices:
            return []

        # Get embeddings for this ayah
        indices = self.ayah_to_indices[key]
        ayah_embeddings = self.embeddings[indices]
        ayah_embedding = ayah_embeddings.mean(axis=0)  # Average if multiple

        # Normalize
        ayah_embedding = ayah_embedding / np.linalg.norm(ayah_embedding)
        
        # Compute similarities
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = embeddings_norm @ ayah_embedding

        # Get top-k
        top_indices = np.argsort(similarities)[::-1]

        results = []
        seen_ayat = {key}  # Exclude self

        for idx in top_indices:
            if len(results) >= top_k:
                break

            sim = similarities[idx]
            if sim < min_similarity:
                break

            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            s = meta.get("surah", 0)
            a = meta.get("ayah", 0)

            if (s, a) not in seen_ayat and s and a:
                seen_ayat.add((s, a))
                results.append({
                    "surah": s,
                    "ayah": a,
                    "similarity": float(sim),
                    "source": meta.get("source", ""),
                    "type": meta.get("type", ""),
                    "behaviors": list(self.ayah_to_behaviors.get((s, a), [])),
                })

        return results

    def find_cross_references(
        self,
        surah: int,
        ayah: int,
        method: str = "both",
        top_k: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find cross-references for an ayah using multiple methods.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            method: 'semantic', 'behavior', or 'both'.
            top_k: Number of references per method.

        Returns:
            Dict with cross-references by method.
        """
        results = {}

        if method in ["semantic", "both"]:
            results["semantic"] = self.find_similar_ayat(surah, ayah, top_k)

        if method in ["behavior", "both"]:
            key = (surah, ayah)
            behaviors = self.ayah_to_behaviors.get(key, set())
            
            behavior_refs = []
            seen = {key}
            
            for behavior in behaviors:
                for s, a in self.behavior_to_ayat.get(behavior, []):
                    if (s, a) not in seen:
                        seen.add((s, a))
                        behavior_refs.append({
                            "surah": s,
                            "ayah": a,
                            "shared_behavior": behavior,
                            "all_behaviors": list(self.ayah_to_behaviors[(s, a)]),
                        })

            behavior_refs.sort(key=lambda x: (x["surah"], x["ayah"]))
            results["behavior"] = behavior_refs[:top_k]

        return results

    def build_reference_network(
        self,
        surah: Optional[int] = None,
        min_similarity: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Build a network of cross-references.

        Args:
            surah: Limit to specific surah (None for all).
            min_similarity: Minimum similarity for edge.

        Returns:
            Network with nodes (ayat) and edges (references).
        """
        nodes = []
        edges = []

        # Get ayat to process
        if surah:
            ayat = [(surah, a) for (s, a) in self.ayah_to_indices.keys() if s == surah]
        else:
            ayat = list(self.ayah_to_indices.keys())[:100]  # Limit for performance

        # Build nodes
        for s, a in ayat:
            nodes.append({
                "id": f"{s}:{a}",
                "surah": s,
                "ayah": a,
                "behaviors": list(self.ayah_to_behaviors.get((s, a), [])),
            })

        # Build edges (behavior-based)
        for behavior, ayat_set in self.behavior_to_ayat.items():
            ayat_list = list(ayat_set)
            for i, (s1, a1) in enumerate(ayat_list):
                for s2, a2 in ayat_list[i + 1:]:
                    if surah is None or (s1 == surah or s2 == surah):
                        edges.append({
                            "source": f"{s1}:{a1}",
                            "target": f"{s2}:{a2}",
                            "type": "shared_behavior",
                            "behavior": behavior,
                        })

        return {
            "nodes": nodes,
            "edges": edges[:1000],  # Limit edges
            "stats": {
                "node_count": len(nodes),
                "edge_count": min(len(edges), 1000),
            },
        }

    def get_behavior_network(self) -> Dict[str, Any]:
        """
        Build a network of behavior co-occurrences.

        Returns:
            Network with behaviors as nodes and co-occurrences as edges.
        """
        nodes = []
        edges = []
        edge_weights = defaultdict(int)

        # Build nodes
        for behavior, ayat in self.behavior_to_ayat.items():
            nodes.append({
                "id": behavior,
                "count": len(ayat),
            })

        # Build edges (co-occurrence in same ayah)
        for key, behaviors in self.ayah_to_behaviors.items():
            behaviors = sorted(behaviors)
            for i, b1 in enumerate(behaviors):
                for b2 in behaviors[i + 1:]:
                    edge_weights[(b1, b2)] += 1

        for (b1, b2), weight in edge_weights.items():
            if weight >= 3:  # Minimum co-occurrence
                edges.append({
                    "source": b1,
                    "target": b2,
                    "weight": weight,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "behavior_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cross-reference statistics."""
        return {
            "total_annotations": len(self.annotations),
            "unique_behaviors": len(self.behavior_to_ayat),
            "unique_ayat": len(self.ayah_to_indices),
            "embeddings_loaded": self.embeddings is not None,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
        }
