"""
Layer 2: Train Arabic Embeddings on QBM Behavioral Spans

Fine-tune AraBERT on contrastive pairs from behavioral annotations.
This creates domain-specific embeddings that understand Quranic Arabic.

Usage:
    python scripts/train_layer2_embeddings.py --epochs 10 --batch_size 32

Expected Training Time: ~2-4 hours on GPU
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training_splits"
MODELS_DIR = DATA_DIR / "models"
VOCAB_DIR = PROJECT_ROOT / "vocab"

# Configuration
BASE_MODEL = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = MODELS_DIR / "qbm-arabic-embeddings"

# Behavior definitions for contrastive learning
BEHAVIOR_DEFINITIONS = {
    "صبر": "حبس النفس على طاعة الله وعن معصيته وعلى أقداره المؤلمة",
    "شكر": "الاعتراف بنعمة المنعم والثناء عليه بها وصرفها في طاعته",
    "توبة": "الرجوع إلى الله والندم على المعصية والعزم على عدم العودة",
    "تقوى": "اتخاذ وقاية من عذاب الله بفعل أوامره واجتناب نواهيه",
    "إيمان": "التصديق الجازم بالله وملائكته وكتبه ورسله واليوم الآخر والقدر",
    "كفر": "جحود الحق وستره وإنكار ما يجب الإيمان به من أركان الإيمان",
    "نفاق": "إظهار الإيمان وإبطان الكفر والخداع في الدين",
    "كبر": "التعالي على الناس واحتقارهم والإعجاب بالنفس ورد الحق",
    "حسد": "تمني زوال النعمة عن الغير وكراهية ما أنعم الله به عليهم",
    "ظلم": "وضع الشيء في غير موضعه والتعدي على حقوق الآخرين",
    "صدق": "مطابقة القول للواقع والإخبار بالحق في جميع الأحوال",
    "كذب": "الإخبار بخلاف الواقع عمداً سواء في القول أو الفعل",
    "ذكر": "استحضار عظمة الله في القلب والثناء عليه باللسان",
    "دعاء": "طلب العبد من ربه ما ينفعه ودفع ما يضره بتضرع وخشوع",
    "توكل": "صدق الاعتماد على الله في جلب المنافع ودفع المضار مع الأخذ بالأسباب",
    "خوف": "توقع مكروه عن أمارة مظنونة أو معلومة مع انزعاج القلب",
    "رجاء": "ارتياح القلب لانتظار ما هو محبوب مع الأخذ بأسبابه",
    "محبة": "ميل القلب إلى الله وإيثار ما يحبه على ما تحبه النفس",
    "رحمة": "رقة القلب وانعطافه نحو الخلق بالإحسان إليهم",
    "عدل": "إعطاء كل ذي حق حقه من غير إفراط ولا تفريط",
    "إحسان": "أن تعبد الله كأنك تراه فإن لم تكن تراه فإنه يراك",
    "تواضع": "خفض الجناح للمؤمنين ولين الجانب وعدم التكبر",
    "غفلة": "ذهول القلب عن ذكر الله والآخرة والانشغال بالدنيا",
    "شرك": "جعل شريك لله في ربوبيته أو ألوهيته أو أسمائه وصفاته",
    "رياء": "إظهار العبادة لقصد رؤية الناس لها والثناء عليها",
    "غيبة": "ذكر أخيك بما يكره في غيبته سواء في خلقه أو دينه",
    "نميمة": "نقل الكلام بين الناس على وجه الإفساد بينهم",
    "مؤمن": "من آمن بالله ورسوله وعمل صالحاً واتقى الله",
    "كافر": "من جحد الحق وأنكر ما يجب الإيمان به",
    "منافق": "من أظهر الإيمان وأبطن الكفر",
    "نبي": "من أوحى الله إليه بشرع ولم يؤمر بتبليغه",
    "رسول": "من أوحى الله إليه بشرع وأمره بتبليغه للناس",
}


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def create_contrastive_pairs(spans: List[Dict], num_negatives: int = 3) -> List[Tuple[str, str, float]]:
    """
    Create contrastive learning pairs from behavioral spans.
    
    Returns: List of (text1, text2, label) tuples
    - label=1.0: positive pair (same behavior)
    - label=0.0: negative pair (different behavior)
    """
    logger.info(f"Creating contrastive pairs from {len(spans)} spans...")
    
    pairs = []
    behaviors = list(BEHAVIOR_DEFINITIONS.keys())
    
    # Group spans by behavior
    by_behavior = {}
    for span in spans:
        behavior = span.get("behavior_ar", "")
        if behavior and behavior in BEHAVIOR_DEFINITIONS:
            if behavior not in by_behavior:
                by_behavior[behavior] = []
            
            # Get text from span
            text = span.get("context", "") or span.get("text_ar", "")
            if text and len(text) > 20:
                by_behavior[behavior].append(text[:512])  # Truncate long texts
    
    logger.info(f"Found {len(by_behavior)} behaviors with examples")
    
    for behavior, texts in by_behavior.items():
        if len(texts) < 2:
            continue
        
        definition = BEHAVIOR_DEFINITIONS[behavior]
        
        for text in texts[:500]:  # Limit per behavior
            # Positive pair: text ↔ behavior definition
            pairs.append((text, definition, 1.0))
            
            # Positive pair: text ↔ another text with same behavior
            if len(texts) > 1:
                other_text = random.choice([t for t in texts if t != text][:10])
                pairs.append((text, other_text, 0.9))
            
            # Negative pairs: text ↔ wrong behavior definitions
            wrong_behaviors = [b for b in behaviors if b != behavior]
            for wrong in random.sample(wrong_behaviors, min(num_negatives, len(wrong_behaviors))):
                wrong_def = BEHAVIOR_DEFINITIONS[wrong]
                pairs.append((text, wrong_def, 0.0))
    
    random.shuffle(pairs)
    logger.info(f"Created {len(pairs)} contrastive pairs")
    
    return pairs


def train_embeddings(
    train_pairs: List[Tuple[str, str, float]],
    val_pairs: List[Tuple[str, str, float]],
    epochs: int = 10,
    batch_size: int = 32,
    warmup_steps: int = 500,
):
    """
    Fine-tune AraBERT on contrastive pairs.
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
        from torch.utils.data import DataLoader
        import torch
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")
    
    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device=device)
    
    # Create training examples
    train_examples = [
        InputExample(texts=[t1, t2], label=label)
        for t1, t2, label in train_pairs
    ]
    
    # Create validation evaluator
    val_sentences1 = [p[0] for p in val_pairs[:1000]]
    val_sentences2 = [p[1] for p in val_pairs[:1000]]
    val_labels = [p[2] for p in val_pairs[:1000]]
    
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        val_sentences1, val_sentences2, val_labels,
        name="qbm-val"
    )
    
    # Training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    
    logger.info(f"Starting training...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Training examples: {len(train_examples)}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_DIR),
        evaluation_steps=1000,
        save_best_model=True,
        show_progress_bar=True,
    )
    
    logger.info(f"Model saved to {OUTPUT_DIR}")
    return model


def evaluate_embeddings(model) -> Dict[str, Any]:
    """
    Evaluate embedding quality with specific tests.
    """
    import numpy as np
    
    logger.info("Evaluating embedding quality...")
    
    results = {"tests": [], "passed": 0, "failed": 0}
    
    # Test 1: الكبر vs أكبر disambiguation
    kibr = model.encode("الكبر")  # Arrogance
    akbar = model.encode("أكبر")  # Greater
    takabbur = model.encode("التكبر")  # Being arrogant
    
    dist_kibr_akbar = 1 - np.dot(kibr, akbar) / (np.linalg.norm(kibr) * np.linalg.norm(akbar))
    dist_kibr_takabbur = 1 - np.dot(kibr, takabbur) / (np.linalg.norm(kibr) * np.linalg.norm(takabbur))
    
    test1_passed = dist_kibr_akbar > dist_kibr_takabbur
    results["tests"].append({
        "name": "الكبر/أكبر disambiguation",
        "passed": test1_passed,
        "dist_kibr_akbar": float(dist_kibr_akbar),
        "dist_kibr_takabbur": float(dist_kibr_takabbur),
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: Same behavior clustering
    sabr1 = model.encode("الصبر على البلاء")
    sabr2 = model.encode("صبر المؤمن على المصائب")
    jaza = model.encode("الجزع والهلع")
    
    sim_sabr = np.dot(sabr1, sabr2) / (np.linalg.norm(sabr1) * np.linalg.norm(sabr2))
    sim_sabr_jaza = np.dot(sabr1, jaza) / (np.linalg.norm(sabr1) * np.linalg.norm(jaza))
    
    test2_passed = sim_sabr > sim_sabr_jaza
    results["tests"].append({
        "name": "Same behavior clustering",
        "passed": test2_passed,
        "sim_same_behavior": float(sim_sabr),
        "sim_different_behavior": float(sim_sabr_jaza),
    })
    results["passed" if test2_passed else "failed"] += 1
    
    # Test 3: Opposite behaviors should be far
    iman = model.encode("الإيمان بالله")
    kufr = model.encode("الكفر بالله")
    taqwa = model.encode("التقوى")
    
    sim_iman_kufr = np.dot(iman, kufr) / (np.linalg.norm(iman) * np.linalg.norm(kufr))
    sim_iman_taqwa = np.dot(iman, taqwa) / (np.linalg.norm(iman) * np.linalg.norm(taqwa))
    
    test3_passed = sim_iman_taqwa > sim_iman_kufr
    results["tests"].append({
        "name": "Opposite behaviors separation",
        "passed": test3_passed,
        "sim_iman_taqwa": float(sim_iman_taqwa),
        "sim_iman_kufr": float(sim_iman_kufr),
    })
    results["passed" if test3_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    
    for test in results["tests"]:
        status = "✅" if test["passed"] else "❌"
        logger.info(f"  {status} {test['name']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Arabic embeddings on QBM data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--num_negatives", type=int, default=3, help="Negative samples per positive")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LAYER 2: TRAINING ARABIC EMBEDDINGS")
    logger.info("=" * 60)
    
    # Load training data
    train_file = TRAINING_DIR / "train_spans.jsonl"
    val_file = TRAINING_DIR / "val_spans.jsonl"
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.error("Run scripts/validate_foundation_data.py first")
        return
    
    train_spans = load_jsonl(train_file)
    val_spans = load_jsonl(val_file)
    
    logger.info(f"Loaded {len(train_spans)} training spans")
    logger.info(f"Loaded {len(val_spans)} validation spans")
    
    # Create contrastive pairs
    train_pairs = create_contrastive_pairs(train_spans, num_negatives=args.num_negatives)
    val_pairs = create_contrastive_pairs(val_spans, num_negatives=1)
    
    # Train
    model = train_embeddings(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
    )
    
    if model is None:
        return
    
    # Evaluate
    eval_results = evaluate_embeddings(model)
    
    # Save evaluation results
    eval_file = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_file}")
    
    logger.info("=" * 60)
    logger.info("LAYER 2 TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"Tests passed: {eval_results['passed']}/{len(eval_results['tests'])}")


if __name__ == "__main__":
    main()
