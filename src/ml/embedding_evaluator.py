"""
QBM Embedding Model Evaluator - Phase 5
Evaluates embedding models on semantic similarity for Classical Arabic.

Metrics:
- Accuracy: % of pairs where high similarity > 0.5 and low similarity < 0.5
- Opposite behavior separation: opposite pairs should have similarity < 0.5
- Same concept clustering: same_concept/synonym pairs should have similarity > 0.7
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = DATA_DIR / "evaluation"
MODELS_DIR = DATA_DIR / "models"
GOLD_FILE = EVALUATION_DIR / "semantic_similarity_gold.jsonl"
REGISTRY_FILE = MODELS_DIR / "registry.json"


@dataclass
class EvaluationResult:
    """Results from evaluating an embedding model."""
    model_name: str
    accuracy: float
    high_sim_accuracy: float  # % of high pairs with sim > 0.5
    low_sim_accuracy: float   # % of low pairs with sim < 0.5
    avg_high_similarity: float
    avg_low_similarity: float
    opposite_separation: float  # avg(high) - avg(low)
    details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "high_sim_accuracy": round(self.high_sim_accuracy, 4),
            "low_sim_accuracy": round(self.low_sim_accuracy, 4),
            "avg_high_similarity": round(self.avg_high_similarity, 4),
            "avg_low_similarity": round(self.avg_low_similarity, 4),
            "opposite_separation": round(self.opposite_separation, 4),
            "passes_threshold": self.accuracy >= 0.75,
        }


class EmbeddingEvaluator:
    """Evaluates embedding models on semantic similarity benchmark."""
    
    def __init__(self):
        self.gold_pairs = self._load_gold_pairs()
        self.model = None
        self.model_name = None
    
    def _load_gold_pairs(self) -> List[Dict[str, Any]]:
        """Load gold standard similarity pairs."""
        if not GOLD_FILE.exists():
            raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")
        
        pairs = []
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        
        logger.info(f"Loaded {len(pairs)} gold pairs")
        return pairs
    
    def load_model(self, model_name: str):
        """Load a sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Model loaded: {model_name}")
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = self.model.encode([text_a, text_b])
        
        # Cosine similarity
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        similarity = np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b + 1e-8)
        
        return float(similarity)
    
    def evaluate(self) -> EvaluationResult:
        """Evaluate the loaded model on all gold pairs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        details = []
        high_sims = []
        low_sims = []
        correct = 0
        total = 0
        
        for pair in self.gold_pairs:
            sim = self.compute_similarity(pair["text_a"], pair["text_b"])
            expected = pair["expected_similarity"]
            
            # Determine if prediction is correct
            if expected == "high":
                is_correct = sim > 0.5
                high_sims.append(sim)
            else:  # low
                is_correct = sim < 0.5
                low_sims.append(sim)
            
            if is_correct:
                correct += 1
            total += 1
            
            details.append({
                "id": pair["id"],
                "text_a": pair["text_a"],
                "text_b": pair["text_b"],
                "expected": expected,
                "similarity": round(sim, 4),
                "correct": is_correct,
                "category": pair.get("category", "unknown"),
            })
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        high_correct = sum(1 for d in details if d["expected"] == "high" and d["correct"])
        low_correct = sum(1 for d in details if d["expected"] == "low" and d["correct"])
        high_total = sum(1 for d in details if d["expected"] == "high")
        low_total = sum(1 for d in details if d["expected"] == "low")
        
        avg_high = np.mean(high_sims) if high_sims else 0
        avg_low = np.mean(low_sims) if low_sims else 0
        
        return EvaluationResult(
            model_name=self.model_name,
            accuracy=accuracy,
            high_sim_accuracy=high_correct / high_total if high_total > 0 else 0,
            low_sim_accuracy=low_correct / low_total if low_total > 0 else 0,
            avg_high_similarity=avg_high,
            avg_low_similarity=avg_low,
            opposite_separation=avg_high - avg_low,
            details=details,
        )


class ModelRegistry:
    """Registry for managing embedding models."""
    
    def __init__(self):
        self.registry_path = REGISTRY_FILE
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load or create registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "active_model": None,
            "models": {},
            "evaluation_history": [],
        }
    
    def save(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def register_model(self, model_name: str, evaluation: EvaluationResult):
        """Register a model with its evaluation results."""
        self.registry["models"][model_name] = {
            "name": model_name,
            "accuracy": evaluation.accuracy,
            "high_sim_accuracy": evaluation.high_sim_accuracy,
            "low_sim_accuracy": evaluation.low_sim_accuracy,
            "opposite_separation": evaluation.opposite_separation,
            "passes_threshold": evaluation.accuracy >= 0.75,
            "evaluated_at": str(np.datetime64('now')),
        }
        
        self.registry["evaluation_history"].append({
            "model": model_name,
            "accuracy": evaluation.accuracy,
            "timestamp": str(np.datetime64('now')),
        })
        
        self.save()
        logger.info(f"Registered model: {model_name} (accuracy: {evaluation.accuracy:.2%})")
    
    def set_active_model(self, model_name: str):
        """Set the active model for production use."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not registered: {model_name}")
        
        self.registry["active_model"] = model_name
        self.save()
        logger.info(f"Active model set to: {model_name}")
    
    def get_active_model(self) -> Optional[str]:
        """Get the currently active model."""
        return self.registry.get("active_model")
    
    def get_best_model(self) -> Optional[str]:
        """Get the model with highest accuracy."""
        if not self.registry["models"]:
            return None
        
        best = max(
            self.registry["models"].items(),
            key=lambda x: x[1]["accuracy"]
        )
        return best[0]


def evaluate_models(model_names: List[str]) -> Dict[str, EvaluationResult]:
    """Evaluate multiple models and return results."""
    evaluator = EmbeddingEvaluator()
    results = {}
    
    for model_name in model_names:
        try:
            evaluator.load_model(model_name)
            result = evaluator.evaluate()
            results[model_name] = result
            
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            print(f"Accuracy: {result.accuracy:.2%}")
            print(f"High similarity accuracy: {result.high_sim_accuracy:.2%}")
            print(f"Low similarity accuracy: {result.low_sim_accuracy:.2%}")
            print(f"Avg high similarity: {result.avg_high_similarity:.4f}")
            print(f"Avg low similarity: {result.avg_low_similarity:.4f}")
            print(f"Opposite separation: {result.opposite_separation:.4f}")
            print(f"Passes 75% threshold: {'✅ YES' if result.accuracy >= 0.75 else '❌ NO'}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            print(f"❌ Failed to evaluate {model_name}: {e}")
    
    return results


def run_phase5_evaluation():
    """
    Run Phase 5 embedding model evaluation.
    
    Compares multiple Arabic embedding models.
    """
    print("="*60)
    print("QBM Phase 5: Embedding Model Evaluation")
    print("="*60)
    
    models_to_evaluate = [
        "aubmindlab/bert-base-arabertv2",  # Current model
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
        "sentence-transformers/distiluse-base-multilingual-cased-v2",  # Distil multilingual
        "Alibaba-NLP/gte-multilingual-base",  # GTE multilingual
    ]
    
    results = evaluate_models(models_to_evaluate)
    
    # Register results
    registry = ModelRegistry()
    for model_name, result in results.items():
        registry.register_model(model_name, result)
    
    # Set best model as active
    best_model = registry.get_best_model()
    if best_model:
        registry.set_active_model(best_model)
        print(f"\n✅ Best model: {best_model}")
        print(f"   Set as active model in registry")
    
    # Save detailed results
    results_file = EVALUATION_DIR / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(
            {name: r.to_dict() for name, r in results.items()},
            f, indent=2, ensure_ascii=False
        )
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase5_evaluation()
