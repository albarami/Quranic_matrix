"""
Layer 6: Domain-Specific Reranker

Train a cross-encoder reranker on QBM domain.
Generic rerankers don't know what's relevant in YOUR domain.

This is the final retrieval layer before sending to frontier model.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Base cross-encoder (multilingual)
BASE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


# =============================================================================
# TRAINING DATA
# =============================================================================

def create_reranker_training_data(spans: List[Dict], tafsir_data: Dict) -> List[Dict]:
    """
    Create training data for domain-specific reranker.
    
    Format: (query, passage, label)
    - label=1.0: highly relevant
    - label=0.0: not relevant
    """
    training_data = []
    
    # Create QA-style pairs from behavioral spans
    for span in spans:
        text = span.get("text_ar", "")
        behavior = span.get("behavior_label", "")
        
        if not text or not behavior:
            continue
        
        # Positive: Question about behavior → span text is relevant
        questions = [
            f"ما هو {behavior} في القرآن؟",
            f"أين ذُكر {behavior}؟",
            f"ما الآيات التي تتحدث عن {behavior}؟",
        ]
        
        for q in questions:
            training_data.append({
                "query": q,
                "passage": text,
                "label": 1.0,
            })
        
        # Hard negatives: similar but wrong passages
        # (spans with different behaviors)
        
    # Create pairs from tafsir
    for source, verses in tafsir_data.items():
        for verse_key, tafsir_text in list(verses.items())[:100]:
            if len(tafsir_text) > 50:
                surah, ayah = verse_key.split(":")
                
                # Positive: verse reference → tafsir is relevant
                training_data.append({
                    "query": f"ما تفسير سورة {surah} آية {ayah}؟",
                    "passage": tafsir_text[:500],
                    "label": 1.0,
                })
    
    logger.info(f"Created {len(training_data)} reranker training examples")
    return training_data


# =============================================================================
# DOMAIN RERANKER
# =============================================================================

class DomainReranker:
    """
    Domain-specific cross-encoder reranker.
    
    Trained on QBM data to understand what's relevant
    for behavioral analysis questions.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = model_path or (MODELS_DIR / "qbm-domain-reranker")
        
        if CE_AVAILABLE:
            self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load trained model or initialize from base."""
        if self.model_path.exists():
            logger.info(f"Loading trained reranker from {self.model_path}")
            self.model = CrossEncoder(str(self.model_path))
        else:
            logger.info(f"Initializing from base: {BASE_RERANKER}")
            self.model = CrossEncoder(BASE_RERANKER)
    
    def rank(self, query: str, passages: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Rerank passages by relevance to query.
        
        Returns: List of (original_index, score, passage)
        """
        if not self.model or not passages:
            return [(i, 0.0, p) for i, p in enumerate(passages[:top_k])]
        
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        indexed_scores = [(i, float(scores[i]), passages[i]) for i in range(len(passages))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]
    
    def rank_with_metadata(self, query: str, 
                           candidates: List[Dict[str, Any]], 
                           text_key: str = "text",
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank candidates (dicts with text and metadata).
        
        Returns candidates with added 'rerank_score' field.
        """
        if not self.model or not candidates:
            return candidates[:top_k]
        
        passages = [c.get(text_key, "") for c in candidates]
        pairs = [[query, p] for p in passages]
        
        scores = self.model.predict(pairs)
        
        # Add scores to candidates
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return candidates[:top_k]
    
    def train(self, training_data: List[Dict], epochs: int = 5, batch_size: int = 16):
        """
        Fine-tune the reranker on domain-specific data.
        """
        if not CE_AVAILABLE:
            logger.error("CrossEncoder not available")
            return
        
        logger.info(f"Training reranker on {len(training_data)} examples")
        
        # Convert to InputExample format
        train_examples = []
        for item in training_data:
            train_examples.append(InputExample(
                texts=[item["query"], item["passage"]],
                label=item["label"],
            ))
        
        # Train
        self.model.fit(
            train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=batch_size),
            epochs=epochs,
            warmup_steps=100,
            output_path=str(self.model_path),
        )
        
        logger.info(f"Reranker saved to {self.model_path}")
    
    def save(self):
        """Save the model."""
        if self.model:
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.model_path))
    
    def load(self):
        """Load the model."""
        if self.model_path.exists():
            self.model = CrossEncoder(str(self.model_path))
            return True
        return False


# =============================================================================
# TESTS
# =============================================================================

def test_reranker(reranker: DomainReranker) -> Dict[str, Any]:
    """Test reranker quality."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: Relevant passage should rank higher
    query = "ما هو الكبر في القرآن؟"
    passages = [
        "الكبر هو التعالي على الناس واحتقارهم",  # Relevant
        "الطقس اليوم مشمس وجميل",  # Irrelevant
        "استكبر فرعون على موسى فأهلكه الله",  # Relevant
    ]
    
    ranked = reranker.rank(query, passages)
    
    # Check that relevant passages rank higher
    test1_passed = ranked[0][2] != passages[1]  # Irrelevant shouldn't be first
    results["tests"].append({
        "name": "Relevant > Irrelevant",
        "passed": test1_passed,
        "top_result": ranked[0][2][:50] if ranked else None,
    })
    results["passed" if test1_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def train_domain_reranker(spans: List[Dict] = None, 
                          tafsir_data: Dict = None,
                          epochs: int = 5) -> Dict[str, Any]:
    """Train the domain-specific reranker."""
    logger.info("=" * 60)
    logger.info("TRAINING DOMAIN RERANKER")
    logger.info("=" * 60)
    
    # Create training data
    if spans is None:
        spans = []
    if tafsir_data is None:
        tafsir_data = {}
    
    training_data = create_reranker_training_data(spans, tafsir_data)
    
    if not training_data:
        # Use sample data
        training_data = [
            {"query": "ما هو الكبر؟", "passage": "الكبر التعالي على الناس", "label": 1.0},
            {"query": "ما هو الكبر؟", "passage": "الطقس جميل اليوم", "label": 0.0},
            {"query": "ما تفسير آية الكرسي؟", "passage": "الله لا إله إلا هو الحي القيوم", "label": 1.0},
        ]
    
    reranker = DomainReranker()
    reranker.train(training_data, epochs=epochs)
    
    test_results = test_reranker(reranker)
    
    return {
        "status": "complete",
        "training_examples": len(training_data),
        "model_path": str(reranker.model_path),
        "test_results": test_results,
    }


_reranker_instance = None

def get_domain_reranker() -> DomainReranker:
    """Get the domain reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = DomainReranker()
    return _reranker_instance


if __name__ == "__main__":
    results = train_domain_reranker(epochs=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))
