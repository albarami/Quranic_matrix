"""
Phase 3: EMBEDDINGS Planner (3.9)

Thin wrapper around LegendaryPlanner for embedding-based semantic search.
MUST be truthful about model limitations - no accuracy claims without proof.

REUSES:
- Existing embedding index if available
- data/models/registry.json for model evaluation results

ADDS:
- Model limitation disclosure
- Truthful accuracy reporting from evaluation results

RULE: No accuracy claims without evaluation proof.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an embedding model."""
    model_name: str
    accuracy: float
    high_sim_accuracy: float
    low_sim_accuracy: float
    opposite_separation: float
    passes_threshold: bool
    evaluated_at: str
    is_active: bool = False


@dataclass
class EmbeddingSearchResult:
    """A single embedding search result."""
    entity_id: str
    label: str
    similarity_score: float
    confidence: str  # "high", "medium", "low" based on model accuracy


@dataclass
class EmbeddingsResult:
    """Result of embeddings analysis."""
    query_type: str  # "search", "model_info", "limitation_disclosure"
    model_info: Optional[ModelInfo]
    search_results: List[EmbeddingSearchResult]
    total_results: int
    limitations: List[str]
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "model_info": {
                "model_name": self.model_info.model_name,
                "accuracy": self.model_info.accuracy,
                "high_sim_accuracy": self.model_info.high_sim_accuracy,
                "low_sim_accuracy": self.model_info.low_sim_accuracy,
                "passes_threshold": self.model_info.passes_threshold,
                "is_active": self.model_info.is_active,
            } if self.model_info else None,
            "search_results": [
                {
                    "entity_id": r.entity_id,
                    "label": r.label,
                    "similarity_score": r.similarity_score,
                    "confidence": r.confidence,
                }
                for r in self.search_results
            ],
            "total_results": self.total_results,
            "limitations": self.limitations,
            "gaps": self.gaps,
        }


class EmbeddingsPlanner:
    """
    Planner for SEMANTIC_LANDSCAPE and embedding-based search.

    Wraps LegendaryPlanner to provide:
    - Truthful model limitation disclosure
    - Embedding search with confidence based on evaluated accuracy
    - No accuracy claims without proof

    CRITICAL: This planner must be truthful about model limitations.
    """

    REGISTRY_PATH = Path("data/models/registry.json")

    # Accuracy thresholds for confidence levels
    HIGH_CONFIDENCE_THRESHOLD = 0.80
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner
        self._registry = None

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def _load_registry(self) -> Dict:
        """Load model registry from file."""
        if self._registry is not None:
            return self._registry

        if self.REGISTRY_PATH.exists():
            with open(self.REGISTRY_PATH, "r", encoding="utf-8") as f:
                self._registry = json.load(f)
        else:
            self._registry = {}

        return self._registry

    def get_model_info(self) -> EmbeddingsResult:
        """
        Get information about available embedding models with truthful limitations.

        Returns:
            EmbeddingsResult with model information and limitations
        """
        registry = self._load_registry()

        gaps = []
        limitations = []
        model_info = None

        active_model = registry.get("active_model", "")
        models = registry.get("models", {})

        if not active_model:
            gaps.append("no_active_model_configured")
        elif active_model not in models:
            gaps.append(f"active_model_not_evaluated:{active_model}")
        else:
            model_data = models[active_model]
            accuracy = model_data.get("accuracy", 0)
            passes = model_data.get("passes_threshold", False)

            model_info = ModelInfo(
                model_name=active_model,
                accuracy=accuracy,
                high_sim_accuracy=model_data.get("high_sim_accuracy", 0),
                low_sim_accuracy=model_data.get("low_sim_accuracy", 0),
                opposite_separation=model_data.get("opposite_separation", 0),
                passes_threshold=passes,
                evaluated_at=model_data.get("evaluated_at", ""),
                is_active=True,
            )

            # Add truthful limitations
            if not passes:
                limitations.append(
                    f"Model {active_model} does NOT pass accuracy threshold (accuracy: {accuracy:.1%})"
                )
            if accuracy < 0.60:
                limitations.append(
                    "Model accuracy is below 60% - results should be treated as indicative only"
                )
            if model_data.get("low_sim_accuracy", 0) < 0.5:
                limitations.append(
                    "Model struggles with low-similarity pairs - may miss semantic distinctions"
                )
            if model_data.get("opposite_separation", 0) < 0:
                limitations.append(
                    "Model has negative opposite separation - may confuse antonyms with synonyms"
                )

        return EmbeddingsResult(
            query_type="model_info",
            model_info=model_info,
            search_results=[],
            total_results=0,
            limitations=limitations,
            gaps=gaps,
        )

    def get_limitation_disclosure(self) -> EmbeddingsResult:
        """
        Get full limitation disclosure for embedding-based analysis.

        This method should be called before any embedding-based claims.

        Returns:
            EmbeddingsResult with comprehensive limitations
        """
        model_result = self.get_model_info()

        # Add general limitations
        general_limitations = [
            "Embedding models may not capture Quranic Arabic semantics accurately",
            "Similarity scores do not imply theological equivalence",
            "Results should be verified against traditional tafsir sources",
            "No accuracy claims are made without evaluation proof from registry",
        ]

        all_limitations = model_result.limitations + general_limitations

        return EmbeddingsResult(
            query_type="limitation_disclosure",
            model_info=model_result.model_info,
            search_results=[],
            total_results=0,
            limitations=all_limitations,
            gaps=model_result.gaps,
        )

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        include_limitations: bool = True
    ) -> EmbeddingsResult:
        """
        Perform semantic search with truthful confidence levels.

        Args:
            query: The search query
            top_k: Number of results to return
            include_limitations: Whether to include model limitations in result

        Returns:
            EmbeddingsResult with search results and confidence levels
        """
        self._ensure_planner()

        gaps = []
        limitations = []
        search_results = []

        # Get model info for confidence calculation
        model_result = self.get_model_info()
        if include_limitations:
            limitations = model_result.limitations

        if model_result.model_info is None:
            gaps.append("cannot_perform_search:no_model_info")
            return EmbeddingsResult(
                query_type="search",
                model_info=None,
                search_results=[],
                total_results=0,
                limitations=limitations,
                gaps=gaps,
            )

        accuracy = model_result.model_info.accuracy

        # Determine confidence level based on model accuracy
        if accuracy >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif accuracy >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Try to get search results from planner if available
        if hasattr(self._planner, 'semantic_search'):
            try:
                raw_results = self._planner.semantic_search(query, top_k)
                for result in raw_results:
                    search_results.append(EmbeddingSearchResult(
                        entity_id=result.get("entity_id", ""),
                        label=result.get("label", ""),
                        similarity_score=result.get("score", 0.0),
                        confidence=confidence_level,
                    ))
            except Exception as e:
                gaps.append(f"search_error:{str(e)}")
        else:
            gaps.append("semantic_search_not_available")

        return EmbeddingsResult(
            query_type="search",
            model_info=model_result.model_info,
            search_results=search_results,
            total_results=len(search_results),
            limitations=limitations,
            gaps=gaps,
        )

    def get_all_evaluated_models(self) -> List[ModelInfo]:
        """
        Get information about all evaluated models.

        Returns:
            List of ModelInfo for all models in registry
        """
        registry = self._load_registry()
        active_model = registry.get("active_model", "")
        models = registry.get("models", {})

        result = []
        for model_name, model_data in models.items():
            result.append(ModelInfo(
                model_name=model_name,
                accuracy=model_data.get("accuracy", 0),
                high_sim_accuracy=model_data.get("high_sim_accuracy", 0),
                low_sim_accuracy=model_data.get("low_sim_accuracy", 0),
                opposite_separation=model_data.get("opposite_separation", 0),
                passes_threshold=model_data.get("passes_threshold", False),
                evaluated_at=model_data.get("evaluated_at", ""),
                is_active=(model_name == active_model),
            ))

        return result
