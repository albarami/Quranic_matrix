"""
Thematic Clustering for QBM Discovery System.

Clusters Quranic annotations by theme using embeddings,
enabling discovery of thematic groups and topic modeling.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ThematicClustering:
    """
    Cluster Quranic annotations by theme.
    
    Features:
    - K-means and hierarchical clustering
    - Automatic cluster labeling
    - Theme extraction per cluster
    - Visualization support (PCA reduction)
    """

    def __init__(
        self,
        embeddings_path: str = "data/embeddings/annotations_embeddings.npy",
        metadata_path: str = "data/embeddings/annotations_embeddings_metadata.json",
        n_clusters: int = 20,
    ):
        """
        Initialize thematic clustering.

        Args:
            embeddings_path: Path to embeddings file.
            metadata_path: Path to metadata file.
            n_clusters: Default number of clusters.
        """
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.n_clusters = n_clusters

        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.pca_embeddings: Optional[np.ndarray] = None

        self._load_data()

    def _load_data(self) -> None:
        """Load embeddings and metadata."""
        if self.embeddings_path.exists():
            self.embeddings = np.load(str(self.embeddings_path))
            print(f"Loaded embeddings: {self.embeddings.shape}")

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {len(self.metadata)} items")

    def cluster_kmeans(
        self,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Cluster annotations using K-means.

        Args:
            n_clusters: Number of clusters (None for default).
            random_state: Random seed for reproducibility.

        Returns:
            Clustering results with labels and statistics.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        n_clusters = n_clusters or self.n_clusters

        print(f"Running K-means clustering with {n_clusters} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=3,
            batch_size=1024,
        )
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        # Calculate silhouette score (sample for speed)
        sample_size = min(10000, len(self.embeddings))
        indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        silhouette = silhouette_score(
            self.embeddings[indices],
            self.cluster_labels[indices],
        )

        # Get cluster sizes
        cluster_sizes = Counter(self.cluster_labels)

        return {
            "method": "kmeans",
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_sizes": dict(cluster_sizes),
            "inertia": float(kmeans.inertia_),
        }

    def cluster_hierarchical(
        self,
        n_clusters: Optional[int] = None,
        linkage: str = "ward",
        sample_size: int = 10000,
    ) -> Dict[str, Any]:
        """
        Cluster annotations using hierarchical clustering.

        Args:
            n_clusters: Number of clusters.
            linkage: Linkage method ('ward', 'complete', 'average').
            sample_size: Sample size for large datasets.

        Returns:
            Clustering results.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        n_clusters = n_clusters or self.n_clusters

        # Sample for large datasets
        if len(self.embeddings) > sample_size:
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            embeddings_sample = self.embeddings[indices]
        else:
            indices = np.arange(len(self.embeddings))
            embeddings_sample = self.embeddings

        print(f"Running hierarchical clustering with {n_clusters} clusters...")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        labels = clustering.fit_predict(embeddings_sample)

        # Calculate silhouette score
        silhouette = silhouette_score(embeddings_sample, labels)

        # Store labels (only for sampled data)
        self.cluster_labels = np.full(len(self.embeddings), -1)
        self.cluster_labels[indices] = labels

        cluster_sizes = Counter(labels)

        return {
            "method": "hierarchical",
            "linkage": linkage,
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_sizes": dict(cluster_sizes),
            "sample_size": len(embeddings_sample),
        }

    def get_cluster_themes(
        self,
        top_n: int = 5,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract themes for each cluster.

        Args:
            top_n: Number of top items per cluster.

        Returns:
            Dict mapping cluster ID to theme info.
        """
        if self.cluster_labels is None:
            raise ValueError("Run clustering first")

        themes = {}

        for cluster_id in range(max(self.cluster_labels) + 1):
            indices = np.where(self.cluster_labels == cluster_id)[0]

            # Count behaviors, sources, types
            behaviors = Counter()
            sources = Counter()
            types = Counter()
            surahs = Counter()

            for idx in indices:
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    behavior = meta.get("behavior", meta.get("type", ""))
                    if behavior:
                        behaviors[behavior] += 1
                    sources[meta.get("source", "")] += 1
                    types[meta.get("type", "")] += 1
                    surahs[meta.get("surah", 0)] += 1

            themes[cluster_id] = {
                "size": len(indices),
                "top_behaviors": behaviors.most_common(top_n),
                "top_sources": sources.most_common(top_n),
                "top_types": types.most_common(top_n),
                "top_surahs": surahs.most_common(top_n),
                "label": self._generate_cluster_label(behaviors, types),
            }

        return themes

    def _generate_cluster_label(
        self,
        behaviors: Counter,
        types: Counter,
    ) -> str:
        """Generate a human-readable label for a cluster."""
        top_behavior = behaviors.most_common(1)
        top_type = types.most_common(1)

        if top_behavior:
            return f"{top_behavior[0][0]}"
        elif top_type:
            return f"{top_type[0][0]}"
        else:
            return "Unknown"

    def get_cluster_samples(
        self,
        cluster_id: int,
        n_samples: int = 10,
        method: str = "random",
    ) -> List[Dict[str, Any]]:
        """
        Get sample annotations from a cluster.

        Args:
            cluster_id: Cluster ID.
            n_samples: Number of samples.
            method: 'random' or 'centroid' (closest to center).

        Returns:
            List of sample annotations.
        """
        if self.cluster_labels is None:
            raise ValueError("Run clustering first")

        indices = np.where(self.cluster_labels == cluster_id)[0]

        if len(indices) == 0:
            return []

        if method == "centroid" and self.cluster_centers is not None:
            # Get samples closest to centroid
            center = self.cluster_centers[cluster_id]
            cluster_embeddings = self.embeddings[indices]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            sorted_indices = indices[np.argsort(distances)]
            sample_indices = sorted_indices[:n_samples]
        else:
            # Random samples
            sample_indices = np.random.choice(
                indices,
                min(n_samples, len(indices)),
                replace=False,
            )

        samples = []
        for idx in sample_indices:
            if idx < len(self.metadata):
                samples.append(self.metadata[idx])

        return samples

    def reduce_dimensions(
        self,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.

        Args:
            n_components: Number of dimensions (2 or 3).

        Returns:
            Reduced embeddings.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        print(f"Reducing dimensions to {n_components}D...")
        pca = PCA(n_components=n_components)
        self.pca_embeddings = pca.fit_transform(self.embeddings)

        print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")

        return self.pca_embeddings

    def get_visualization_data(
        self,
        sample_size: int = 5000,
    ) -> Dict[str, Any]:
        """
        Get data for cluster visualization.

        Args:
            sample_size: Number of points to include.

        Returns:
            Visualization data with coordinates and labels.
        """
        if self.cluster_labels is None:
            raise ValueError("Run clustering first")

        if self.pca_embeddings is None:
            self.reduce_dimensions(n_components=2)

        # Sample for visualization
        indices = np.random.choice(
            len(self.embeddings),
            min(sample_size, len(self.embeddings)),
            replace=False,
        )

        points = []
        for idx in indices:
            point = {
                "x": float(self.pca_embeddings[idx, 0]),
                "y": float(self.pca_embeddings[idx, 1]),
                "cluster": int(self.cluster_labels[idx]),
            }
            if idx < len(self.metadata):
                point["metadata"] = self.metadata[idx]
            points.append(point)

        return {
            "points": points,
            "n_clusters": int(max(self.cluster_labels) + 1),
            "sample_size": len(points),
        }

    def find_optimal_clusters(
        self,
        min_k: int = 5,
        max_k: int = 50,
        step: int = 5,
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            min_k: Minimum number of clusters.
            max_k: Maximum number of clusters.
            step: Step size for k values.

        Returns:
            Analysis with scores for each k.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        # Sample for speed
        sample_size = min(10000, len(self.embeddings))
        indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        embeddings_sample = self.embeddings[indices]

        results = []
        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1, step):
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_sample)
            score = silhouette_score(embeddings_sample, labels)

            results.append({
                "k": k,
                "silhouette_score": float(score),
                "inertia": float(kmeans.inertia_),
            })

            if score > best_score:
                best_score = score
                best_k = k

        return {
            "results": results,
            "optimal_k": best_k,
            "best_silhouette": best_score,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        stats = {
            "embeddings_loaded": self.embeddings is not None,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "metadata_count": len(self.metadata),
            "clustered": self.cluster_labels is not None,
        }

        if self.cluster_labels is not None:
            stats["n_clusters"] = int(max(self.cluster_labels) + 1)
            stats["cluster_sizes"] = dict(Counter(self.cluster_labels))

        return stats
