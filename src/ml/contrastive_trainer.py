"""
QBM Contrastive Fine-Tuning for Arabic Semantic Similarity - Phase 5
Fine-tunes embedding model on Quranic behavioral data using contrastive learning.

Target: >= 75% accuracy on semantic similarity benchmark
Method: Triplet loss with hard negative mining
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_MODEL_DIR = MODELS_DIR / "qbm-arabic-finetuned"
TRAINING_DATA_FILE = DATA_DIR / "evaluation" / "contrastive_training.json"

# Opposite behavior pairs for hard negatives
OPPOSITE_BEHAVIORS = {
    "BEH_SABR": "BEH_GHADAB",      # patience vs anger
    "BEH_IMAN": "BEH_KUFR",        # faith vs disbelief
    "BEH_SHUKR": "BEH_KUFR",       # gratitude vs ingratitude
    "BEH_TAWBA": "BEH_FISQ",       # repentance vs sin
    "BEH_SIDQ": "BEH_KIDHB",       # truthfulness vs lying
    "BEH_AMANA": "BEH_KHIYANA",    # trustworthiness vs betrayal
    "BEH_TAWADU": "BEH_KIBR",      # humility vs arrogance
    "BEH_ADL": "BEH_DHULM",        # justice vs oppression
    "BEH_RAHMA": "BEH_GHADAB",     # mercy vs anger
    "BEH_KHUSHU": "BEH_GHAFLA",    # humility vs heedlessness
    "BEH_TAQWA": "BEH_FISQ",       # piety vs sin
    "BEH_IHSAN": "BEH_DHULM",      # excellence vs oppression
    "BEH_ZUHD": "BEH_BUKHL",       # asceticism vs miserliness
    "BEH_TAWAKKUL": "BEH_SHIRK",   # trust in God vs polytheism
    "BEH_DHIKR": "BEH_GHAFLA",     # remembrance vs heedlessness
    "BEH_RIDA": "BEH_GHADAB",      # contentment vs anger
    "BEH_HAYA": "BEH_FUJUR",       # modesty vs immorality
}

# Add reverse mappings
OPPOSITE_BEHAVIORS.update({v: k for k, v in OPPOSITE_BEHAVIORS.items()})


@dataclass
class TrainingExample:
    """A training example for contrastive learning."""
    anchor: str
    positive: str
    negative: str
    anchor_behavior: str
    positive_behavior: str
    negative_behavior: str


def load_behavioral_data() -> Dict[str, List[Dict[str, Any]]]:
    """Load behavioral annotations grouped by behavior_id."""
    data = {}
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                beh_id = entry.get("behavior_id", "")
                context = entry.get("context", "")
                
                # Clean context (remove HTML if present)
                if "<" in context:
                    import re
                    context = re.sub(r'<[^>]+>', '', context)
                
                if beh_id and context and len(context) > 30:
                    if beh_id not in data:
                        data[beh_id] = []
                    data[beh_id].append({
                        "context": context,
                        "surah": entry.get("surah"),
                        "ayah": entry.get("ayah"),
                        "behavior_ar": entry.get("behavior_ar", ""),
                    })
    
    logger.info(f"Loaded {sum(len(v) for v in data.values())} examples across {len(data)} behaviors")
    return data


# Definition-anchored training pairs
# Format: (term, definition) - these are KNOWN to be semantically equivalent
DEFINITION_PAIRS = [
    # Positive behaviors with definitions
    ("الصبر", "حبس النفس عن الجزع والتسخط"),
    ("الصبر", "الثبات عند البلاء"),
    ("الصبر", "احتمال المكروه دون شكوى"),
    ("الإيمان", "التصديق بالله ورسوله"),
    ("الإيمان", "الإقرار باللسان والتصديق بالقلب"),
    ("الشكر", "الاعتراف بالنعمة والثناء على المنعم"),
    ("الشكر", "حمد الله على نعمه"),
    ("التوبة", "الرجوع إلى الله من الذنب"),
    ("التوبة", "الندم على المعصية والعزم على عدم العودة"),
    ("الذكر", "استحضار الله في القلب واللسان"),
    ("الذكر", "تسبيح الله وتحميده"),
    ("التقوى", "اتقاء غضب الله بالطاعة"),
    ("التقوى", "الخوف من الله واجتناب معاصيه"),
    ("الصدق", "مطابقة القول للواقع"),
    ("الصدق", "الإخبار بالحقيقة"),
    ("العدل", "إعطاء كل ذي حق حقه"),
    ("العدل", "الإنصاف بين الناس"),
    ("الرحمة", "الرأفة واللين بالخلق"),
    ("الرحمة", "الشفقة على الضعفاء"),
    ("التواضع", "خفض الجناح للناس"),
    ("التواضع", "عدم التكبر على الخلق"),
    ("الخشوع", "خضوع القلب لله"),
    ("الخشوع", "سكون الجوارح في العبادة"),
    ("التوكل", "الاعتماد على الله مع الأخذ بالأسباب"),
    ("التوكل", "تفويض الأمر إلى الله"),
    ("الإخلاص", "إرادة وجه الله بالعمل"),
    ("الإخلاص", "تجريد القصد لله وحده"),
    # Negative behaviors with definitions
    ("الكبر", "رد الحق واحتقار الناس"),
    ("الكبر", "التعالي على الخلق"),
    ("الحسد", "تمني زوال النعمة عن الغير"),
    ("الحسد", "كراهية نعمة الله على غيره"),
    ("النفاق", "إظهار الإيمان وإخفاء الكفر"),
    ("النفاق", "مخالفة الظاهر للباطن"),
    ("الظلم", "وضع الشيء في غير موضعه"),
    ("الظلم", "التعدي على حقوق الآخرين"),
    ("الكفر", "الجحود بالله ونعمه"),
    ("الكفر", "ستر الحق وإنكاره"),
    ("الغضب", "ثوران النفس لدفع المكروه"),
    ("الغضب", "الحنق والسخط"),
    ("البخل", "منع الواجب من المال"),
    ("البخل", "الشح بالمعروف"),
    ("الكذب", "الإخبار بخلاف الواقع"),
    ("الكذب", "مخالفة القول للحقيقة"),
    ("الغفلة", "السهو عن ذكر الله"),
    ("الغفلة", "عدم التنبه للآخرة"),
    ("الرياء", "العمل لأجل الناس لا لله"),
    ("الرياء", "طلب المدح من الخلق"),
]

# Unrelated pairs for hard negatives (different semantic field entirely)
UNRELATED_TERMS = [
    "الأكل والشرب",
    "السفر والترحال",
    "البيع والشراء",
    "النوم والاستيقاظ",
    "الزراعة والحصاد",
    "الطبخ والطعام",
    "اللباس والثياب",
    "البناء والعمارة",
    "الصيد والقنص",
    "الكتابة والقراءة",
    "الركوب والمشي",
    "السباحة في الماء",
    "الحرف اليدوية",
    "التجارة والصناعة",
]

# Opposite pairs (related but not equivalent) - should have similarity ~0.5
OPPOSITE_PAIRS = [
    ("الصبر", "الجزع والهلع"),
    ("الإيمان", "الكفر والجحود"),
    ("الشكر", "الكفران وجحود النعمة"),
    ("التوبة", "الإصرار على المعصية"),
    ("الذكر", "الغفلة عن الله"),
    ("التقوى", "الفسق والعصيان"),
    ("الصدق", "الكذب والافتراء"),
    ("العدل", "الظلم والجور"),
    ("الرحمة", "القسوة والغلظة"),
    ("التواضع", "الكبر والتعالي"),
    ("الكبر", "التواضع والخشوع"),
    ("الحسد", "الرضا بقسمة الله"),
    ("النفاق", "الصدق والإخلاص"),
    ("الظلم", "العدل والإنصاف"),
    ("الكفر", "الإيمان والتصديق"),
    ("الغضب", "الحلم والأناة"),
    ("البخل", "الكرم والجود"),
    ("الكذب", "الصدق والأمانة"),
]


def create_training_examples(
    data: Dict[str, List[Dict[str, Any]]],
    num_examples: int = 5000,
) -> List[TrainingExample]:
    """
    Create definition-anchored training examples.
    
    Positives: (term, definition) - known semantic equivalence
    Negatives: unrelated terms from different semantic fields
    
    This ensures valid triplets where positive is truly closer than negative.
    """
    examples = []
    
    for _ in range(num_examples):
        # Pick a random definition pair for anchor-positive
        anchor, positive = random.choice(DEFINITION_PAIRS)
        
        # Pick an unrelated term for negative
        negative = random.choice(UNRELATED_TERMS)
        
        examples.append(TrainingExample(
            anchor=anchor,
            positive=positive,
            negative=negative,
            anchor_behavior="TERM",
            positive_behavior="DEFINITION",
            negative_behavior="UNRELATED",
        ))
    
    logger.info(f"Created {len(examples)} definition-anchored training examples")
    return examples


def create_sentence_transformer_examples(
    examples: List[TrainingExample]
) -> List[InputExample]:
    """
    Convert to SentenceTransformer InputExample format for CosineSimilarityLoss.
    
    Creates pairs with graded similarity scores:
    - (term, definition) -> 1.0 (equivalent)
    - (term, opposite) -> 0.5 (related but not equivalent)
    - (term, unrelated) -> 0.0 (unrelated)
    """
    st_examples = []
    
    for ex in examples:
        # Positive pair: term and definition are equivalent (1.0)
        st_examples.append(InputExample(
            texts=[ex.anchor, ex.positive],
            label=1.0,
        ))
        # Negative pair: term and unrelated are dissimilar (0.0)
        st_examples.append(InputExample(
            texts=[ex.anchor, ex.negative],
            label=0.0,
        ))
    
    # Add opposite pairs with 0.5 label
    for term, opposite in OPPOSITE_PAIRS:
        st_examples.append(InputExample(
            texts=[term, opposite],
            label=0.5,
        ))
    
    return st_examples


def create_evaluation_data() -> Tuple[List[str], List[str], List[float]]:
    """Create evaluation data from gold benchmark (graded labels)."""
    from src.ml.embedding_evaluator import GOLD_FILE
    
    sentences1 = []
    sentences2 = []
    scores = []
    
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                sentences1.append(entry["text_a"])
                sentences2.append(entry["text_b"])
                # Use graded similarity score (1.0, 0.5, 0.0)
                # Fall back to old format if needed
                if "similarity" in entry:
                    scores.append(entry["similarity"])
                else:
                    scores.append(1.0 if entry.get("expected_similarity") == "high" else 0.0)
    
    return sentences1, sentences2, scores


class ContrastiveTrainer:
    """Fine-tunes embedding model using contrastive learning."""
    
    def __init__(
        self,
        base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        output_dir: Path = OUTPUT_MODEL_DIR,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model = None
        
    def load_model(self):
        """Load base model."""
        logger.info(f"Loading base model: {self.base_model}")
        self.model = SentenceTransformer(self.base_model)
        logger.info("Model loaded")
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        num_examples: int = 5000,
    ):
        """
        Train the model using contrastive learning.
        
        Uses MultipleNegativesRankingLoss which is effective for
        learning semantic similarity.
        """
        if self.model is None:
            self.load_model()
        
        # Load data
        logger.info("Loading behavioral data...")
        behavioral_data = load_behavioral_data()
        
        # Create training examples
        logger.info("Creating training examples...")
        training_examples = create_training_examples(behavioral_data, num_examples)
        st_examples = create_sentence_transformer_examples(training_examples)
        
        # Save training data for reproducibility
        TRAINING_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump([
                {"anchor": ex.anchor, "positive": ex.positive, "negative": ex.negative}
                for ex in training_examples[:100]  # Save sample
            ], f, ensure_ascii=False, indent=2)
        
        # Create DataLoader
        train_dataloader = DataLoader(
            st_examples,
            shuffle=True,
            batch_size=batch_size,
        )
        
        # Use CosineSimilarityLoss with graded labels
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator
        sentences1, sentences2, scores = create_evaluation_data()
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores,
            name="qbm-semantic-similarity",
        )
        
        # Train
        logger.info(f"Training for {num_epochs} epochs...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=str(self.output_dir),
            show_progress_bar=True,
        )
        
        logger.info(f"Training complete. Model saved to {self.output_dir}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the trained model."""
        from src.ml.embedding_evaluator import EmbeddingEvaluator, ModelRegistry
        
        evaluator = EmbeddingEvaluator()
        evaluator.load_model(str(self.output_dir))
        result = evaluator.evaluate()
        
        # Register in model registry
        registry = ModelRegistry()
        registry.register_model(f"qbm-arabic-finetuned", result)
        
        if result.accuracy >= 0.75:
            registry.set_active_model("qbm-arabic-finetuned")
            logger.info(f"✅ Model passes 75% threshold! Accuracy: {result.accuracy:.2%}")
        else:
            logger.warning(f"⚠️ Model below 75% threshold. Accuracy: {result.accuracy:.2%}")
        
        return result.to_dict()


def run_gating_test(model) -> bool:
    """
    Run gating test before training to verify model can learn correctly.
    
    Fails if:
    - Equivalent pairs avg <= Unrelated pairs avg (inverted semantics)
    - AUC < 0.60 on mini benchmark
    """
    print("\n[GATING TEST] Verifying base model semantics...")
    
    # Test pairs
    equivalent_pairs = [
        ("الصبر", "الثبات عند البلاء"),
        ("الإيمان", "التصديق بالله"),
        ("الشكر", "حمد الله على نعمه"),
        ("التوبة", "الرجوع إلى الله"),
        ("الذكر", "تسبيح الله وحمده"),
    ]
    
    unrelated_pairs = [
        ("الصبر", "الأكل والشرب"),
        ("الإيمان", "السفر والترحال"),
        ("الشكر", "البيع والشراء"),
        ("التوبة", "النوم والاستيقاظ"),
        ("الذكر", "الزراعة والحصاد"),
    ]
    
    equivalent_sims = []
    unrelated_sims = []
    
    for a, b in equivalent_pairs:
        emb = model.encode([a, b])
        sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        equivalent_sims.append(sim)
    
    for a, b in unrelated_pairs:
        emb = model.encode([a, b])
        sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        unrelated_sims.append(sim)
    
    avg_equivalent = np.mean(equivalent_sims)
    avg_unrelated = np.mean(unrelated_sims)
    gap = avg_equivalent - avg_unrelated
    
    print(f"  Avg equivalent similarity: {avg_equivalent:.4f}")
    print(f"  Avg unrelated similarity: {avg_unrelated:.4f}")
    print(f"  Separation gap: {gap:.4f}")
    
    if gap <= 0:
        print("  ❌ GATING FAILED: Equivalent <= Unrelated (inverted semantics)")
        return False
    
    # Simple AUC check
    from sklearn.metrics import roc_auc_score
    labels = [1] * len(equivalent_sims) + [0] * len(unrelated_sims)
    scores = equivalent_sims + unrelated_sims
    auc = roc_auc_score(labels, scores)
    
    print(f"  AUC: {auc:.4f}")
    
    if auc < 0.60:
        print("  ❌ GATING FAILED: AUC < 0.60")
        return False
    
    print("  ✅ GATING PASSED: Model semantics are correct")
    return True


def run_phase5_training():
    """Run the complete Phase 5 training pipeline with gating tests."""
    print("="*60)
    print("QBM Phase 5: Contrastive Fine-Tuning")
    print("="*60)
    
    trainer = ContrastiveTrainer()
    trainer.load_model()
    
    # Gating test before training
    if not run_gating_test(trainer.model):
        print("\n❌ Training aborted: Base model failed gating test")
        print("   Fix the model or data before proceeding.")
        return None
    
    # Train
    print("\n[1/3] Training model...")
    trainer.train(
        num_epochs=5,
        batch_size=16,
        num_examples=8000,
    )
    
    # Post-training gating test
    print("\n[2/3] Post-training gating test...")
    trainer.model = SentenceTransformer(str(trainer.output_dir))
    if not run_gating_test(trainer.model):
        print("\n❌ Training DEGRADED the model!")
        print("   Review training data and loss function.")
    
    # Evaluate
    print("\n[3/3] Evaluating model...")
    results = trainer.evaluate()
    
    print("\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Pearson: {results['pearson']:.4f}")
    print(f"  Spearman: {results['spearman']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Separation gap: {results['separation_gap']:.4f}")
    print(f"  Passes 75% threshold: {'✅ YES' if results['passes_threshold'] else '❌ NO'}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase5_training()
