"""
QBM Embedding Model Evaluator - Phase 5 (Revised)

Supports two benchmark types:
1. RELATEDNESS: Graded similarity (0.0, 0.5, 1.0) - for retrieval tasks
   Metrics: Spearman correlation, separation by category
   
2. EQUIVALENCE: Binary similarity (0.0, 1.0) - for paraphrase detection
   Metrics: AUC, accuracy

See data/evaluation/BENCHMARK_DEFINITION.md for details.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = DATA_DIR / "evaluation"
MODELS_DIR = DATA_DIR / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"

# Benchmark files
GOLD_RELATEDNESS = EVALUATION_DIR / "gold_relatedness_v2.jsonl"
GOLD_EQUIVALENCE = EVALUATION_DIR / "gold_equivalence_v2.jsonl"


@dataclass
class RelatednessResult:
    """Results for relatedness benchmark (graded: 0.0, 0.5, 1.0)."""
    model_name: str
    spearman: float  # Primary metric for graded similarity
    pearson: float
    avg_equivalent_sim: float  # avg for 1.0 pairs
    avg_related_sim: float  # avg for 0.5 pairs
    avg_unrelated_sim: float  # avg for 0.0 pairs
    separation_equiv_related: float  # avg(equiv) - avg(related)
    separation_related_unrelated: float  # avg(related) - avg(unrelated)
    ranking_correct: float  # % where equiv > related > unrelated
    winrate_equiv_vs_related: float  # % equiv beats related
    winrate_related_vs_unrelated: float  # % related beats unrelated
    category_order_holds: bool  # mean(equiv) > mean(related) > mean(unrelated)
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def passes_threshold(self) -> bool:
        """Check if model meets Phase 5 acceptance criteria."""
        # 1. Category-order invariant (HARD GATE)
        if not self.category_order_holds:
            return False
        # 2. Each gap >= 0.15
        if self.separation_equiv_related < 0.15 or self.separation_related_unrelated < 0.15:
            return False
        # 3. Pairwise win-rate >= 70%
        if self.winrate_equiv_vs_related < 0.70 or self.winrate_related_vs_unrelated < 0.70:
            return False
        # 4. Spearman >= 0.35 (smoke benchmark threshold)
        if self.spearman < 0.35:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "relatedness",
            "model_name": self.model_name,
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
            "avg_equivalent_sim": round(self.avg_equivalent_sim, 4),
            "avg_related_sim": round(self.avg_related_sim, 4),
            "avg_unrelated_sim": round(self.avg_unrelated_sim, 4),
            "separation_equiv_related": round(self.separation_equiv_related, 4),
            "separation_related_unrelated": round(self.separation_related_unrelated, 4),
            "ranking_correct": round(self.ranking_correct, 4),
            "winrate_equiv_vs_related": round(self.winrate_equiv_vs_related, 4),
            "winrate_related_vs_unrelated": round(self.winrate_related_vs_unrelated, 4),
            "category_order_holds": self.category_order_holds,
            "passes_threshold": self.passes_threshold(),
        }


@dataclass
class EquivalenceResult:
    """Results for equivalence benchmark (binary: 0.0, 1.0)."""
    model_name: str
    auc: float  # Primary metric for binary classification
    accuracy: float  # % correct at threshold 0.5
    precision: float
    recall: float
    avg_positive_sim: float  # avg for 1.0 pairs
    avg_negative_sim: float  # avg for 0.0 pairs
    separation_gap: float  # avg(positive) - avg(negative)
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def passes_threshold(self) -> bool:
        """Check if model meets minimum quality bar."""
        return self.auc >= 0.7 and self.separation_gap >= 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "equivalence",
            "model_name": self.model_name,
            "auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "avg_positive_sim": round(self.avg_positive_sim, 4),
            "avg_negative_sim": round(self.avg_negative_sim, 4),
            "separation_gap": round(self.separation_gap, 4),
            "passes_threshold": self.passes_threshold(),
        }


class EmbeddingEvaluator:
    """Evaluates embedding models on semantic benchmarks."""
    
    def __init__(self, benchmark: Literal["relatedness", "equivalence"] = "relatedness"):
        """
        Initialize evaluator with specified benchmark type.
        
        Args:
            benchmark: "relatedness" for retrieval tasks, "equivalence" for paraphrase
        """
        self.benchmark_type = benchmark
        self.gold_file = GOLD_RELATEDNESS if benchmark == "relatedness" else GOLD_EQUIVALENCE
        self.gold_pairs = self._load_gold_pairs()
        self.model = None
        self.model_name = None
    
    def _load_gold_pairs(self) -> List[Dict[str, Any]]:
        """Load gold standard pairs from benchmark file."""
        if not self.gold_file.exists():
            raise FileNotFoundError(f"Gold file not found: {self.gold_file}")
        
        pairs = []
        with open(self.gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        
        logger.info(f"Loaded {len(pairs)} gold pairs from {self.gold_file.name}")
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
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        similarity = np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b + 1e-8)
        return float(similarity)
    
    def evaluate_relatedness(self) -> RelatednessResult:
        """Evaluate on relatedness benchmark (graded similarity)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        details = []
        predicted_sims = []
        gold_scores = []
        
        # Group by similarity level
        equiv_sims = []  # 1.0
        related_sims = []  # 0.5
        unrelated_sims = []  # 0.0
        
        for pair in self.gold_pairs:
            sim = self.compute_similarity(pair["text_a"], pair["text_b"])
            gold = pair["similarity"]
            
            predicted_sims.append(sim)
            gold_scores.append(gold)
            
            if gold >= 0.9:
                equiv_sims.append(sim)
            elif gold >= 0.4:
                related_sims.append(sim)
            else:
                unrelated_sims.append(sim)
            
            details.append({
                "id": pair["id"],
                "text_a": pair["text_a"],
                "text_b": pair["text_b"],
                "gold": gold,
                "predicted": round(sim, 4),
                "category": pair.get("category", "unknown"),
            })
        
        # Spearman correlation (primary metric for graded)
        spearman_r, _ = spearmanr(predicted_sims, gold_scores)
        pearson_r, _ = pearsonr(predicted_sims, gold_scores)
        
        # Averages per category
        avg_equiv = np.mean(equiv_sims) if equiv_sims else 0
        avg_related = np.mean(related_sims) if related_sims else 0
        avg_unrelated = np.mean(unrelated_sims) if unrelated_sims else 0
        
        # Pairwise win-rates (more robust than single Spearman)
        # Win-rate: equiv vs related
        equiv_vs_related_wins = 0
        equiv_vs_related_total = 0
        for e in equiv_sims:
            for r in related_sims:
                if e > r:
                    equiv_vs_related_wins += 1
                equiv_vs_related_total += 1
        winrate_equiv_related = equiv_vs_related_wins / equiv_vs_related_total if equiv_vs_related_total > 0 else 0
        
        # Win-rate: related vs unrelated
        related_vs_unrelated_wins = 0
        related_vs_unrelated_total = 0
        for r in related_sims:
            for u in unrelated_sims:
                if r > u:
                    related_vs_unrelated_wins += 1
                related_vs_unrelated_total += 1
        winrate_related_unrelated = related_vs_unrelated_wins / related_vs_unrelated_total if related_vs_unrelated_total > 0 else 0
        
        # Overall ranking correctness (all pairwise comparisons)
        ranking_correct = equiv_vs_related_wins + related_vs_unrelated_wins
        total_comparisons = equiv_vs_related_total + related_vs_unrelated_total
        # Also add equiv vs unrelated
        for e in equiv_sims:
            for u in unrelated_sims:
                if e > u:
                    ranking_correct += 1
                total_comparisons += 1
        ranking_pct = ranking_correct / total_comparisons if total_comparisons > 0 else 0
        
        # Category-order invariant: mean(equiv) > mean(related) > mean(unrelated)
        category_order_holds = (avg_equiv > avg_related > avg_unrelated)
        
        return RelatednessResult(
            model_name=self.model_name,
            spearman=spearman_r,
            pearson=pearson_r,
            avg_equivalent_sim=avg_equiv,
            avg_related_sim=avg_related,
            avg_unrelated_sim=avg_unrelated,
            separation_equiv_related=avg_equiv - avg_related,
            separation_related_unrelated=avg_related - avg_unrelated,
            ranking_correct=ranking_pct,
            winrate_equiv_vs_related=winrate_equiv_related,
            winrate_related_vs_unrelated=winrate_related_unrelated,
            category_order_holds=category_order_holds,
            details=details,
        )
    
    def evaluate_equivalence(self) -> EquivalenceResult:
        """Evaluate on equivalence benchmark (binary classification)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        details = []
        predicted_sims = []
        gold_labels = []
        
        pos_sims = []  # 1.0
        neg_sims = []  # 0.0
        
        for pair in self.gold_pairs:
            sim = self.compute_similarity(pair["text_a"], pair["text_b"])
            gold = pair["similarity"]
            
            predicted_sims.append(sim)
            gold_labels.append(int(gold >= 0.9))
            
            if gold >= 0.9:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)
            
            details.append({
                "id": pair["id"],
                "text_a": pair["text_a"],
                "text_b": pair["text_b"],
                "gold": gold,
                "predicted": round(sim, 4),
                "category": pair.get("category", "unknown"),
            })
        
        # AUC (primary metric for binary)
        try:
            auc = roc_auc_score(gold_labels, predicted_sims)
        except ValueError:
            auc = 0.5
        
        # Accuracy at threshold 0.5
        predictions = [1 if s > 0.5 else 0 for s in predicted_sims]
        correct = sum(p == g for p, g in zip(predictions, gold_labels))
        accuracy = correct / len(gold_labels) if gold_labels else 0
        
        # Precision/Recall
        tp = sum(p == 1 and g == 1 for p, g in zip(predictions, gold_labels))
        fp = sum(p == 1 and g == 0 for p, g in zip(predictions, gold_labels))
        fn = sum(p == 0 and g == 1 for p, g in zip(predictions, gold_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        avg_pos = np.mean(pos_sims) if pos_sims else 0
        avg_neg = np.mean(neg_sims) if neg_sims else 0
        
        return EquivalenceResult(
            model_name=self.model_name,
            auc=auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            avg_positive_sim=avg_pos,
            avg_negative_sim=avg_neg,
            separation_gap=avg_pos - avg_neg,
            details=details,
        )
    
    def evaluate(self):
        """Evaluate using the configured benchmark type."""
        if self.benchmark_type == "relatedness":
            return self.evaluate_relatedness()
        else:
            return self.evaluate_equivalence()


class ModelRegistry:
    """Registry for managing embedding models."""
    
    def __init__(self):
        self.registry_path = REGISTRY_FILE
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"active_model": None, "models": {}, "evaluation_history": []}
    
    def save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def register_model(self, model_name: str, result):
        """Register a model with its evaluation results."""
        self.registry["models"][model_name] = result.to_dict()
        self.registry["evaluation_history"].append({
            "model": model_name,
            "benchmark": result.to_dict().get("benchmark", "unknown"),
            "timestamp": str(np.datetime64('now')),
        })
        self.save()
        logger.info(f"Registered model: {model_name}")
    
    def set_active_model(self, model_name: str):
        self.registry["active_model"] = model_name
        self.save()
    
    def get_active_model(self) -> Optional[str]:
        return self.registry.get("active_model")


def evaluate_models_on_both_benchmarks(model_names: List[str]) -> Dict[str, Any]:
    """Evaluate multiple models on both benchmarks."""
    results = {"relatedness": [], "equivalence": []}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print('='*60)
        
        # Relatedness benchmark
        eval_rel = EmbeddingEvaluator(benchmark="relatedness")
        eval_rel.load_model(model_name)
        rel_result = eval_rel.evaluate_relatedness()
        results["relatedness"].append(rel_result.to_dict())
        
        print(f"\nRELATEDNESS (for retrieval) - Smoke Benchmark:")
        print(f"  Spearman: {rel_result.spearman:.4f} (threshold: ≥0.35)")
        print(f"  Category order holds: {'✅' if rel_result.category_order_holds else '❌'}")
        print(f"  Avg equiv: {rel_result.avg_equivalent_sim:.4f}")
        print(f"  Avg related: {rel_result.avg_related_sim:.4f}")
        print(f"  Avg unrelated: {rel_result.avg_unrelated_sim:.4f}")
        print(f"  Gap equiv-related: {rel_result.separation_equiv_related:.4f} (threshold: ≥0.15)")
        print(f"  Gap related-unrelated: {rel_result.separation_related_unrelated:.4f} (threshold: ≥0.15)")
        print(f"  Win-rate equiv>related: {rel_result.winrate_equiv_vs_related:.2%} (threshold: ≥70%)")
        print(f"  Win-rate related>unrelated: {rel_result.winrate_related_vs_unrelated:.2%} (threshold: ≥70%)")
        print(f"  Overall ranking correct: {rel_result.ranking_correct:.2%}")
        print(f"  PASSES PHASE 5: {'✅ YES' if rel_result.passes_threshold() else '❌ NO'}")
        
        # Equivalence benchmark
        eval_eq = EmbeddingEvaluator(benchmark="equivalence")
        eval_eq.load_model(model_name)
        eq_result = eval_eq.evaluate_equivalence()
        results["equivalence"].append(eq_result.to_dict())
        
        print(f"\nEQUIVALENCE (for paraphrase):")
        print(f"  AUC: {eq_result.auc:.4f}")
        print(f"  Accuracy: {eq_result.accuracy:.2%}")
        print(f"  Separation: {eq_result.separation_gap:.4f}")
        print(f"  Passes: {'✅' if eq_result.passes_threshold() else '❌'}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "aubmindlab/bert-base-arabertv2",
    ]
    
    results = evaluate_models_on_both_benchmarks(models)
    
    # Save results
    output_file = EVALUATION_DIR / "benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
