"""
QBM Embedding Model Evaluator - Phase 5 (v3)

Fixes from v2:
1. Caches embeddings (encode unique texts once)
2. Supports production pipeline adapter (GPUEmbeddingPipeline)
3. Proper pairwise win-rate metrics
4. Removes arbitrary threshold accuracy for equivalence
5. NaN guards for correlations
6. Single model load for both benchmarks

See data/evaluation/BENCHMARK_DEFINITION.md for benchmark details.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Union, Protocol
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

GOLD_RELATEDNESS = EVALUATION_DIR / "gold_relatedness_v2.jsonl"
GOLD_EQUIVALENCE = EVALUATION_DIR / "gold_equivalence_v2.jsonl"


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        ...


class SentenceTransformerBackend:
    """Backend using SentenceTransformers library."""
    
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings


class TransformersBackend:
    """Backend using transformers directly (for models without SentenceTransformer support)."""
    
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try loading with safetensors first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        except Exception:
            # Fallback to regular loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded {model_name} with TransformersBackend on {self.device}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        import torch
        
        all_embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**encoded)
                embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


class GPUPipelineBackend:
    """Backend using production GPUEmbeddingPipeline."""
    
    def __init__(self, pipeline=None):
        if pipeline is None:
            from src.ai.gpu import GPUEmbeddingPipeline
            self.pipeline = GPUEmbeddingPipeline(
                model_name="aubmindlab/bert-base-arabertv2",
                batch_size=64,
                use_multi_gpu=True,
            )
        else:
            self.pipeline = pipeline
        self.model_name = "GPUEmbeddingPipeline (production)"
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.embed_texts(texts, normalize=True, show_progress=False)


@dataclass
class RelatednessResult:
    """Results for relatedness benchmark (graded: 0.0, 0.5, 1.0)."""
    model_name: str
    spearman: float
    pearson: float
    avg_equivalent_sim: float
    avg_related_sim: float
    avg_unrelated_sim: float
    separation_equiv_related: float
    separation_related_unrelated: float
    pairwise_win_rate: float  # Overall win-rate across all comparisons
    winrate_equiv_vs_related: float
    winrate_related_vs_unrelated: float
    winrate_equiv_vs_unrelated: float
    category_order_holds: bool
    num_pairs: int
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def passes_threshold(self) -> bool:
        """Phase 5 acceptance: category order + win-rates, not just Spearman."""
        # 1. Category-order invariant (HARD GATE)
        if not self.category_order_holds:
            return False
        # 2. Each gap >= 0.10 (relaxed from 0.15 for smoke benchmark)
        if self.separation_equiv_related < 0.10 or self.separation_related_unrelated < 0.10:
            return False
        # 3. Pairwise win-rates >= 70%
        if self.winrate_equiv_vs_related < 0.70 or self.winrate_related_vs_unrelated < 0.70:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "relatedness",
            "model_name": self.model_name,
            "num_pairs": self.num_pairs,
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
            "avg_equivalent_sim": round(self.avg_equivalent_sim, 4),
            "avg_related_sim": round(self.avg_related_sim, 4),
            "avg_unrelated_sim": round(self.avg_unrelated_sim, 4),
            "separation_equiv_related": round(self.separation_equiv_related, 4),
            "separation_related_unrelated": round(self.separation_related_unrelated, 4),
            "pairwise_win_rate": round(self.pairwise_win_rate, 4),
            "winrate_equiv_vs_related": round(self.winrate_equiv_vs_related, 4),
            "winrate_related_vs_unrelated": round(self.winrate_related_vs_unrelated, 4),
            "winrate_equiv_vs_unrelated": round(self.winrate_equiv_vs_unrelated, 4),
            "category_order_holds": bool(self.category_order_holds),
            "passes_threshold": bool(self.passes_threshold()),
        }


@dataclass
class EquivalenceResult:
    """Results for equivalence benchmark (binary: 0.0, 1.0)."""
    model_name: str
    auc: float  # Primary metric
    avg_positive_sim: float
    avg_negative_sim: float
    separation_gap: float
    num_pairs: int
    num_positive: int
    num_negative: int
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def passes_threshold(self) -> bool:
        """AUC >= 0.7 and positive separation."""
        return self.auc >= 0.70 and self.separation_gap >= 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "equivalence",
            "model_name": self.model_name,
            "num_pairs": self.num_pairs,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
            "auc": round(self.auc, 4),
            "avg_positive_sim": round(self.avg_positive_sim, 4),
            "avg_negative_sim": round(self.avg_negative_sim, 4),
            "separation_gap": round(self.separation_gap, 4),
            "passes_threshold": bool(self.passes_threshold()),
        }


def load_gold_pairs(gold_file: Path) -> List[Dict[str, Any]]:
    """Load gold pairs from JSONL file."""
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_file}")
    
    pairs = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    pair = json.loads(line)
                    if "id" not in pair:
                        pair["id"] = f"auto_{line_num}"
                    pairs.append(pair)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    logger.info(f"Loaded {len(pairs)} pairs from {gold_file.name}")
    return pairs


def compute_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    # Embeddings should already be normalized, but be safe
    return float(np.dot(emb_a, emb_b))


class EmbeddingEvaluator:
    """Evaluates embedding models on semantic benchmarks with caching."""
    
    def __init__(self, backend: Optional[EmbeddingBackend] = None):
        """
        Initialize evaluator.
        
        Args:
            backend: Embedding backend (SentenceTransformerBackend or GPUPipelineBackend)
        """
        self.backend = backend
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    def set_backend(self, backend: EmbeddingBackend):
        """Set or change the embedding backend."""
        self.backend = backend
        self.embedding_cache.clear()  # Clear cache when backend changes
    
    def load_sentence_transformer(self, model_name: str):
        """Convenience method to load a SentenceTransformer model."""
        self.set_backend(SentenceTransformerBackend(model_name))
    
    def load_transformers_model(self, model_name: str):
        """Load model using transformers directly (for models with torch.load issues)."""
        self.set_backend(TransformersBackend(model_name))
    
    def load_production_pipeline(self, pipeline=None):
        """Load the production GPUEmbeddingPipeline."""
        self.set_backend(GPUPipelineBackend(pipeline))
    
    def _cache_embeddings(self, texts: List[str]):
        """Encode and cache embeddings for all unique texts."""
        if self.backend is None:
            raise RuntimeError("No backend set. Call load_sentence_transformer() or load_production_pipeline() first.")
        
        # Find texts not in cache
        new_texts = [t for t in texts if t not in self.embedding_cache]
        
        if new_texts:
            logger.info(f"Encoding {len(new_texts)} new texts...")
            embeddings = self.backend.encode(new_texts)
            for text, emb in zip(new_texts, embeddings):
                self.embedding_cache[text] = emb
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text (from cache or compute)."""
        if text not in self.embedding_cache:
            self._cache_embeddings([text])
        return self.embedding_cache[text]
    
    def evaluate_relatedness(self, gold_file: Path = GOLD_RELATEDNESS) -> RelatednessResult:
        """Evaluate on relatedness benchmark (graded similarity)."""
        pairs = load_gold_pairs(gold_file)
        
        # Collect all unique texts and cache embeddings
        all_texts = set()
        for pair in pairs:
            all_texts.add(pair["text_a"])
            all_texts.add(pair["text_b"])
        self._cache_embeddings(list(all_texts))
        
        # Compute similarities
        details = []
        predicted_sims = []
        gold_scores = []
        equiv_sims = []
        related_sims = []
        unrelated_sims = []
        
        for pair in pairs:
            emb_a = self.get_embedding(pair["text_a"])
            emb_b = self.get_embedding(pair["text_b"])
            sim = compute_cosine_similarity(emb_a, emb_b)
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
        
        # Spearman/Pearson with NaN guards
        try:
            spearman_r, _ = spearmanr(predicted_sims, gold_scores)
            if math.isnan(spearman_r):
                logger.warning("Spearman returned NaN (no variance?), setting to 0.0")
                spearman_r = 0.0
        except Exception as e:
            logger.warning(f"Spearman failed: {e}, setting to 0.0")
            spearman_r = 0.0
        
        try:
            pearson_r, _ = pearsonr(predicted_sims, gold_scores)
            if math.isnan(pearson_r):
                logger.warning("Pearson returned NaN, setting to 0.0")
                pearson_r = 0.0
        except Exception as e:
            logger.warning(f"Pearson failed: {e}, setting to 0.0")
            pearson_r = 0.0
        
        # Category averages
        avg_equiv = np.mean(equiv_sims) if equiv_sims else 0.0
        avg_related = np.mean(related_sims) if related_sims else 0.0
        avg_unrelated = np.mean(unrelated_sims) if unrelated_sims else 0.0
        
        # Pairwise win-rates
        def compute_winrate(higher_sims: List[float], lower_sims: List[float]) -> float:
            if not higher_sims or not lower_sims:
                return 0.0
            wins = sum(1 for h in higher_sims for l in lower_sims if h > l)
            total = len(higher_sims) * len(lower_sims)
            return wins / total if total > 0 else 0.0
        
        wr_equiv_related = compute_winrate(equiv_sims, related_sims)
        wr_related_unrelated = compute_winrate(related_sims, unrelated_sims)
        wr_equiv_unrelated = compute_winrate(equiv_sims, unrelated_sims)
        
        # Overall pairwise win-rate
        total_wins = (
            sum(1 for e in equiv_sims for r in related_sims if e > r) +
            sum(1 for r in related_sims for u in unrelated_sims if r > u) +
            sum(1 for e in equiv_sims for u in unrelated_sims if e > u)
        )
        total_comparisons = (
            len(equiv_sims) * len(related_sims) +
            len(related_sims) * len(unrelated_sims) +
            len(equiv_sims) * len(unrelated_sims)
        )
        overall_winrate = total_wins / total_comparisons if total_comparisons > 0 else 0.0
        
        # Category order check
        category_order_holds = (avg_equiv > avg_related > avg_unrelated)
        
        return RelatednessResult(
            model_name=getattr(self.backend, 'model_name', 'unknown'),
            spearman=spearman_r,
            pearson=pearson_r,
            avg_equivalent_sim=avg_equiv,
            avg_related_sim=avg_related,
            avg_unrelated_sim=avg_unrelated,
            separation_equiv_related=avg_equiv - avg_related,
            separation_related_unrelated=avg_related - avg_unrelated,
            pairwise_win_rate=overall_winrate,
            winrate_equiv_vs_related=wr_equiv_related,
            winrate_related_vs_unrelated=wr_related_unrelated,
            winrate_equiv_vs_unrelated=wr_equiv_unrelated,
            category_order_holds=category_order_holds,
            num_pairs=len(pairs),
            details=details,
        )
    
    def evaluate_equivalence(self, gold_file: Path = GOLD_EQUIVALENCE) -> EquivalenceResult:
        """Evaluate on equivalence benchmark (binary classification)."""
        pairs = load_gold_pairs(gold_file)
        
        # Collect all unique texts and cache embeddings
        all_texts = set()
        for pair in pairs:
            all_texts.add(pair["text_a"])
            all_texts.add(pair["text_b"])
        self._cache_embeddings(list(all_texts))
        
        # Compute similarities
        details = []
        predicted_sims = []
        gold_labels = []
        pos_sims = []
        neg_sims = []
        
        for pair in pairs:
            emb_a = self.get_embedding(pair["text_a"])
            emb_b = self.get_embedding(pair["text_b"])
            sim = compute_cosine_similarity(emb_a, emb_b)
            gold = pair["similarity"]
            is_positive = gold >= 0.9
            
            predicted_sims.append(sim)
            gold_labels.append(1 if is_positive else 0)
            
            if is_positive:
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
        
        # AUC (primary metric - threshold-free)
        try:
            if len(set(gold_labels)) < 2:
                logger.warning("Only one class in gold labels, AUC undefined")
                auc = 0.5
            else:
                auc = roc_auc_score(gold_labels, predicted_sims)
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}, setting to 0.5")
            auc = 0.5
        
        avg_pos = np.mean(pos_sims) if pos_sims else 0.0
        avg_neg = np.mean(neg_sims) if neg_sims else 0.0
        
        return EquivalenceResult(
            model_name=getattr(self.backend, 'model_name', 'unknown'),
            auc=auc,
            avg_positive_sim=avg_pos,
            avg_negative_sim=avg_neg,
            separation_gap=avg_pos - avg_neg,
            num_pairs=len(pairs),
            num_positive=len(pos_sims),
            num_negative=len(neg_sims),
            details=details,
        )
    
    def evaluate_both(self) -> Dict[str, Any]:
        """Evaluate on both benchmarks (single model load)."""
        rel_result = self.evaluate_relatedness()
        eq_result = self.evaluate_equivalence()
        return {
            "relatedness": rel_result.to_dict(),
            "equivalence": eq_result.to_dict(),
        }


def run_evaluation(model_names: List[str], use_production_pipeline: bool = False):
    """Run evaluation on multiple models."""
    results = []
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print('='*70)
        
        evaluator = EmbeddingEvaluator()
        
        if use_production_pipeline and model_name == "production":
            evaluator.load_production_pipeline()
        else:
            evaluator.load_sentence_transformer(model_name)
        
        # Evaluate both benchmarks with single model load
        rel_result = evaluator.evaluate_relatedness()
        eq_result = evaluator.evaluate_equivalence()
        
        print(f"\nRELATEDNESS (smoke benchmark, {rel_result.num_pairs} pairs):")
        print(f"  Category order: {'✅' if rel_result.category_order_holds else '❌'} (equiv > related > unrelated)")
        print(f"  Avg equiv: {rel_result.avg_equivalent_sim:.4f}")
        print(f"  Avg related: {rel_result.avg_related_sim:.4f}")
        print(f"  Avg unrelated: {rel_result.avg_unrelated_sim:.4f}")
        print(f"  Gap equiv-related: {rel_result.separation_equiv_related:.4f} (threshold: ≥0.10)")
        print(f"  Gap related-unrelated: {rel_result.separation_related_unrelated:.4f} (threshold: ≥0.10)")
        print(f"  Win-rate equiv>related: {rel_result.winrate_equiv_vs_related:.1%} (threshold: ≥70%)")
        print(f"  Win-rate related>unrelated: {rel_result.winrate_related_vs_unrelated:.1%} (threshold: ≥70%)")
        print(f"  Spearman: {rel_result.spearman:.4f}")
        print(f"  PASSES: {'✅ YES' if rel_result.passes_threshold() else '❌ NO'}")
        
        print(f"\nEQUIVALENCE ({eq_result.num_pairs} pairs, {eq_result.num_positive} pos / {eq_result.num_negative} neg):")
        print(f"  AUC: {eq_result.auc:.4f} (threshold: ≥0.70)")
        print(f"  Avg positive: {eq_result.avg_positive_sim:.4f}")
        print(f"  Avg negative: {eq_result.avg_negative_sim:.4f}")
        print(f"  Separation: {eq_result.separation_gap:.4f} (threshold: ≥0.15)")
        print(f"  PASSES: {'✅ YES' if eq_result.passes_threshold() else '❌ NO'}")
        
        results.append({
            "model": model_name,
            "relatedness": rel_result.to_dict(),
            "equivalence": eq_result.to_dict(),
        })
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]
    
    results = run_evaluation(models)
    
    # Save results
    output_file = EVALUATION_DIR / "benchmark_results_v3.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
