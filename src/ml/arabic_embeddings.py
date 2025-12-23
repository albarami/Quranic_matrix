"""
Layer 2: Arabic-First Embeddings

Fine-tune sentence-transformers on QBM behavioral spans.
Replace generic multilingual embeddings with domain-specific Arabic embeddings.

This is TRAINING, not just inference.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML imports
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from sentence_transformers import (
        SentenceTransformer, 
        InputExample, 
        losses,
        evaluation
    )
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
SPANS_FILE = DATA_DIR / "annotations" / "gold_spans.jsonl"

# Arabic-specific base models (NOT generic multilingual)
ARABIC_BASE_MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv2",           # Best for MSA
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",  # Classical + MSA
    "marbert": "UBC-NLP/MARBERT",                          # Social media Arabic
    "arabert-large": "aubmindlab/bert-large-arabertv2",    # Larger, more accurate
}

# Behavior definitions for contrastive learning
BEHAVIOR_DEFINITIONS = {
    "الكبر": "التعالي على الناس واحتقارهم والإعجاب بالنفس",
    "النفاق": "إظهار الإيمان وإبطان الكفر والخداع",
    "الصبر": "حبس النفس على طاعة الله وعن معصيته وعلى أقداره",
    "الشكر": "الاعتراف بنعمة المنعم والثناء عليه بها",
    "التوبة": "الرجوع إلى الله والندم على المعصية والعزم على عدم العودة",
    "التقوى": "اتخاذ وقاية من عذاب الله بفعل أوامره واجتناب نواهيه",
    "الإيمان": "التصديق الجازم بالله وملائكته وكتبه ورسله واليوم الآخر",
    "الكفر": "جحود الحق وستره وإنكار ما يجب الإيمان به",
    "الظلم": "وضع الشيء في غير موضعه والتعدي على حقوق الآخرين",
    "الإحسان": "أن تعبد الله كأنك تراه فإن لم تكن تراه فإنه يراك",
    "الرياء": "إظهار العبادة لقصد رؤية الناس لها والثناء عليها",
    "الحسد": "تمني زوال النعمة عن الغير",
    "الغيبة": "ذكر أخيك بما يكره في غيبته",
    "الكذب": "الإخبار بخلاف الواقع عمداً",
    "الصدق": "مطابقة القول للواقع والإخبار بالحق",
    "الأمانة": "أداء الحقوق والوفاء بالعهود",
    "الخيانة": "الغدر ونقض العهد وعدم أداء الحقوق",
    "العدل": "إعطاء كل ذي حق حقه",
    "الرحمة": "رقة القلب وانعطافه نحو الخلق",
    "القسوة": "غلظة القلب وعدم تأثره بالمواعظ",
    "التواضع": "خفض الجناح للمؤمنين ولين الجانب",
    "الغفلة": "ذهول القلب عن ذكر الله والآخرة",
    "الذكر": "استحضار عظمة الله في القلب والثناء عليه باللسان",
    "الدعاء": "طلب العبد من ربه ما ينفعه ودفع ما يضره",
    "التوكل": "صدق الاعتماد على الله في جلب المنافع ودفع المضار",
    "الخوف": "توقع مكروه عن أمارة مظنونة أو معلومة",
    "الرجاء": "ارتياح القلب لانتظار ما هو محبوب",
    "المحبة": "ميل القلب إلى الله وإيثار ما يحبه",
    "الزهد": "ترك ما لا ينفع في الآخرة",
    "الورع": "ترك ما يريبك إلى ما لا يريبك",
    "الإخلاص": "تصفية العمل من كل شائبة",
    "اليقين": "العلم الجازم الذي لا يتطرق إليه شك",
    "الفسق": "الخروج عن طاعة الله",
    "الشرك": "جعل شريك لله في ربوبيته أو ألوهيته أو أسمائه وصفاته",
}


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

@dataclass
class TrainingPair:
    """A training pair for contrastive learning."""
    anchor: str           # Span text
    positive: str         # Correct behavior definition
    negative: str         # Wrong behavior definition
    behavior: str         # Behavior label


def load_spans(filepath: Path = SPANS_FILE) -> List[Dict[str, Any]]:
    """Load behavioral spans from JSONL file."""
    spans = []
    
    # Try multiple possible locations
    possible_paths = [
        filepath,
        DATA_DIR / "gold_spans.jsonl",
        DATA_DIR / "spans" / "gold_spans.jsonl",
        DATA_DIR / "behavioral_spans.jsonl",
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        spans.append(json.loads(line))
            logger.info(f"Loaded {len(spans)} spans from {path}")
            return spans
    
    logger.warning(f"No spans file found. Tried: {possible_paths}")
    return spans


def create_training_pairs(spans: List[Dict[str, Any]], 
                          num_negatives: int = 3) -> List[InputExample]:
    """
    Create contrastive learning pairs from behavioral spans.
    
    For each span:
    - Positive pair: span text ↔ correct behavior definition (label=1.0)
    - Negative pairs: span text ↔ wrong behavior definitions (label=0.0)
    """
    if not ST_AVAILABLE:
        logger.error("sentence-transformers not available")
        return []
    
    examples = []
    all_behaviors = list(BEHAVIOR_DEFINITIONS.keys())
    
    for span in spans:
        text = span.get("text_ar", "")
        if not text:
            continue
        
        # Get behavior from span (try multiple field names)
        behavior = (
            span.get("behavior_label") or 
            span.get("behavior") or 
            span.get("behavior_concept") or
            ""
        )
        
        # Skip if behavior not in our definitions
        if behavior not in BEHAVIOR_DEFINITIONS:
            continue
        
        # Positive pair: span text ↔ correct behavior definition
        positive_def = BEHAVIOR_DEFINITIONS[behavior]
        examples.append(InputExample(
            texts=[text, positive_def],
            label=1.0
        ))
        
        # Negative pairs: span text ↔ wrong behavior definitions
        wrong_behaviors = [b for b in all_behaviors if b != behavior]
        for wrong_behavior in random.sample(wrong_behaviors, min(num_negatives, len(wrong_behaviors))):
            negative_def = BEHAVIOR_DEFINITIONS[wrong_behavior]
            examples.append(InputExample(
                texts=[text, negative_def],
                label=0.0
            ))
    
    logger.info(f"Created {len(examples)} training pairs")
    return examples


def create_triplet_examples(spans: List[Dict[str, Any]]) -> List[InputExample]:
    """
    Create triplet examples for triplet loss training.
    
    (anchor, positive, negative) where:
    - anchor: span text
    - positive: another span with SAME behavior
    - negative: span with DIFFERENT behavior
    """
    if not ST_AVAILABLE:
        return []
    
    # Group spans by behavior
    by_behavior = {}
    for span in spans:
        behavior = span.get("behavior_label") or span.get("behavior", "")
        if behavior:
            if behavior not in by_behavior:
                by_behavior[behavior] = []
            by_behavior[behavior].append(span.get("text_ar", ""))
    
    examples = []
    behaviors = list(by_behavior.keys())
    
    for behavior, texts in by_behavior.items():
        if len(texts) < 2:
            continue
        
        other_behaviors = [b for b in behaviors if b != behavior]
        if not other_behaviors:
            continue
        
        for i, anchor in enumerate(texts):
            # Positive: another span with same behavior
            positives = [t for j, t in enumerate(texts) if j != i]
            if not positives:
                continue
            positive = random.choice(positives)
            
            # Negative: span with different behavior
            neg_behavior = random.choice(other_behaviors)
            negative = random.choice(by_behavior[neg_behavior])
            
            examples.append(InputExample(
                texts=[anchor, positive, negative]
            ))
    
    logger.info(f"Created {len(examples)} triplet examples")
    return examples


# =============================================================================
# FINE-TUNING
# =============================================================================

class ArabicEmbeddingTrainer:
    """
    Fine-tune Arabic embeddings on QBM behavioral spans.
    
    This is TRAINING, not just inference.
    Uses the GPU to LEARN from YOUR data.
    """
    
    def __init__(self, base_model: str = "arabert"):
        self.base_model_name = ARABIC_BASE_MODELS.get(base_model, base_model)
        self.model = None
        self.output_dir = MODELS_DIR / "qbm-arabic-embeddings"
        
        if ST_AVAILABLE:
            logger.info(f"Loading base model: {self.base_model_name}")
            logger.info(f"Device: {DEVICE}")
            self.model = SentenceTransformer(self.base_model_name, device=DEVICE)
        else:
            logger.error("sentence-transformers not available")
    
    def train_contrastive(self, 
                          train_examples: List[InputExample],
                          epochs: int = 10,
                          batch_size: int = 16,
                          warmup_steps: int = 100,
                          evaluation_steps: int = 500):
        """
        Train using contrastive loss (CosineSimilarityLoss).
        
        This teaches the model that:
        - Span text should be CLOSE to correct behavior definition
        - Span text should be FAR from wrong behavior definitions
        """
        if not self.model or not train_examples:
            logger.error("Model or training data not available")
            return
        
        logger.info(f"Training with {len(train_examples)} examples")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # Use cosine similarity loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir),
            show_progress_bar=True,
        )
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def train_triplet(self,
                      train_examples: List[InputExample],
                      epochs: int = 10,
                      batch_size: int = 16):
        """
        Train using triplet loss.
        
        This teaches the model that:
        - Spans with SAME behavior should be close
        - Spans with DIFFERENT behaviors should be far
        """
        if not self.model or not train_examples:
            logger.error("Model or training data not available")
            return
        
        logger.info(f"Training triplet loss with {len(train_examples)} examples")
        
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        train_loss = losses.TripletLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            output_path=str(self.output_dir),
            show_progress_bar=True,
        )
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def evaluate(self, test_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Evaluate embedding quality.
        
        Test that:
        - الكبر is closer to التكبر than to أكبر
        - Same behavior spans are closer than different behavior spans
        """
        if not self.model:
            return {}
        
        correct = 0
        total = 0
        
        for text1, text2, expected_sim in test_pairs:
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            
            # Cosine similarity
            sim = float(emb1 @ emb2 / (sum(emb1**2)**0.5 * sum(emb2**2)**0.5))
            
            # Check if similarity matches expectation
            if (expected_sim > 0.5 and sim > 0.5) or (expected_sim < 0.5 and sim < 0.5):
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Evaluation accuracy: {accuracy:.2%}")
        
        return {"accuracy": accuracy, "total": total}
    
    def save(self, path: Path = None):
        """Save the fine-tuned model."""
        if path is None:
            path = self.output_dir
        
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save(str(path))
            logger.info(f"Model saved to {path}")
    
    def load(self, path: Path = None):
        """Load a fine-tuned model."""
        if path is None:
            path = self.output_dir
        
        if path.exists() and ST_AVAILABLE:
            self.model = SentenceTransformer(str(path), device=DEVICE)
            logger.info(f"Model loaded from {path}")
            return True
        return False


# =============================================================================
# QUALITY TESTS
# =============================================================================

def test_embedding_quality(model: SentenceTransformer) -> Dict[str, Any]:
    """
    Test that embeddings understand Arabic behavioral nuances.
    
    Key tests:
    1. الكبر (arrogance) should be closer to التكبر than to أكبر (greater)
    2. Same behavior spans should cluster together
    3. Opposite behaviors should be far apart
    """
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: الكبر vs أكبر disambiguation
    kibr = model.encode("الكبر")  # Arrogance
    takabbur = model.encode("التكبر")  # Being arrogant
    akbar = model.encode("أكبر")  # Greater (as in الله أكبر)
    
    sim_correct = float(kibr @ takabbur / (sum(kibr**2)**0.5 * sum(takabbur**2)**0.5))
    sim_wrong = float(kibr @ akbar / (sum(kibr**2)**0.5 * sum(akbar**2)**0.5))
    
    test1_passed = sim_correct > sim_wrong
    results["tests"].append({
        "name": "الكبر disambiguation",
        "passed": test1_passed,
        "sim_correct": sim_correct,
        "sim_wrong": sim_wrong,
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: Opposite behaviors should be far
    sabr = model.encode("الصبر")  # Patience
    jaza = model.encode("الجزع")  # Impatience
    shukr = model.encode("الشكر")  # Gratitude
    
    sim_opposite = float(sabr @ jaza / (sum(sabr**2)**0.5 * sum(jaza**2)**0.5))
    sim_related = float(sabr @ shukr / (sum(sabr**2)**0.5 * sum(shukr**2)**0.5))
    
    test2_passed = sim_related > sim_opposite
    results["tests"].append({
        "name": "Opposite behaviors separation",
        "passed": test2_passed,
        "sim_related": sim_related,
        "sim_opposite": sim_opposite,
    })
    results["passed" if test2_passed else "failed"] += 1
    
    # Test 3: Context matters
    allah_akbar = model.encode("الله أكبر")  # Allah is greater
    istakbara = model.encode("استكبر فرعون")  # Pharaoh was arrogant
    
    # These should be DIFFERENT despite sharing root ك-ب-ر
    sim_context = float(allah_akbar @ istakbara / (sum(allah_akbar**2)**0.5 * sum(istakbara**2)**0.5))
    
    test3_passed = sim_context < 0.7  # Should not be too similar
    results["tests"].append({
        "name": "Context awareness",
        "passed": test3_passed,
        "similarity": sim_context,
    })
    results["passed" if test3_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_arabic_embeddings(
    base_model: str = "arabert",
    epochs: int = 10,
    batch_size: int = 16,
    use_triplet: bool = False
) -> Dict[str, Any]:
    """
    Main function to train Arabic embeddings on QBM data.
    
    This is what makes the system INTELLIGENT instead of MECHANICAL.
    """
    logger.info("=" * 60)
    logger.info("TRAINING ARABIC EMBEDDINGS ON QBM DATA")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"GPU available: {TORCH_AVAILABLE and torch.cuda.is_available()}")
    
    # Load spans
    spans = load_spans()
    if not spans:
        logger.error("No spans loaded. Cannot train.")
        return {"error": "No training data"}
    
    # Create training examples
    if use_triplet:
        train_examples = create_triplet_examples(spans)
    else:
        train_examples = create_training_pairs(spans)
    
    if not train_examples:
        logger.error("No training examples created")
        return {"error": "No training examples"}
    
    # Initialize trainer
    trainer = ArabicEmbeddingTrainer(base_model)
    
    # Train
    if use_triplet:
        trainer.train_triplet(train_examples, epochs=epochs, batch_size=batch_size)
    else:
        trainer.train_contrastive(train_examples, epochs=epochs, batch_size=batch_size)
    
    # Test quality
    if trainer.model:
        test_results = test_embedding_quality(trainer.model)
    else:
        test_results = {}
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return {
        "status": "complete",
        "spans_used": len(spans),
        "examples_created": len(train_examples),
        "model_path": str(trainer.output_dir),
        "test_results": test_results,
    }


# =============================================================================
# SINGLETON
# =============================================================================

_trained_model = None

def get_qbm_embeddings() -> Optional[SentenceTransformer]:
    """Get the trained QBM Arabic embeddings model."""
    global _trained_model
    
    if _trained_model is not None:
        return _trained_model
    
    model_path = MODELS_DIR / "qbm-arabic-embeddings"
    
    if model_path.exists() and ST_AVAILABLE:
        _trained_model = SentenceTransformer(str(model_path), device=DEVICE)
        logger.info(f"Loaded trained model from {model_path}")
        return _trained_model
    
    # Fall back to base Arabic model
    if ST_AVAILABLE:
        _trained_model = SentenceTransformer(
            ARABIC_BASE_MODELS["arabert"], 
            device=DEVICE
        )
        logger.info("Using base AraBERT model (not fine-tuned)")
        return _trained_model
    
    return None


if __name__ == "__main__":
    # Run training
    results = train_arabic_embeddings(
        base_model="arabert",
        epochs=5,
        batch_size=16,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
