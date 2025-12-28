"""
Layer 2: Train Arabic Embeddings on 8x A100 GPUs

Uses DistributedDataParallel (DDP) for efficient multi-GPU training.
With 8x A100 GPUs, training should complete in ~30 minutes instead of hours.

Usage:
    # Single command to launch on all 8 GPUs
    torchrun --nproc_per_node=8 scripts/train_layer2_multi_gpu.py --epochs 10 --batch_size 64

    # Or with accelerate
    accelerate launch --num_processes=8 scripts/train_layer2_multi_gpu.py --epochs 10
"""

import json
import random
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
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

# Configuration
BASE_MODEL = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = MODELS_DIR / "qbm-arabic-embeddings"

# Behavior definitions
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
    "توكل": "صدق الاعتماد على الله في جلب المنافع ودفع المضار",
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
    """Create contrastive learning pairs."""
    pairs = []
    behaviors = list(BEHAVIOR_DEFINITIONS.keys())
    
    by_behavior = {}
    for span in spans:
        behavior = span.get("behavior_ar", "")
        if behavior and behavior in BEHAVIOR_DEFINITIONS:
            if behavior not in by_behavior:
                by_behavior[behavior] = []
            text = span.get("context", "") or span.get("text_ar", "")
            if text and len(text) > 20:
                by_behavior[behavior].append(text[:512])
    
    for behavior, texts in by_behavior.items():
        if len(texts) < 2:
            continue
        
        definition = BEHAVIOR_DEFINITIONS[behavior]
        
        for text in texts[:500]:
            pairs.append((text, definition, 1.0))
            
            if len(texts) > 1:
                other_text = random.choice([t for t in texts if t != text][:10])
                pairs.append((text, other_text, 0.9))
            
            wrong_behaviors = [b for b in behaviors if b != behavior]
            for wrong in random.sample(wrong_behaviors, min(num_negatives, len(wrong_behaviors))):
                pairs.append((text, BEHAVIOR_DEFINITIONS[wrong], 0.0))
    
    random.shuffle(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)  # Larger batch for 8 GPUs
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=-1)  # For DDP
    args = parser.parse_args()
    
    # Check for distributed training
    is_distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        logger.info("=" * 60)
        logger.info("LAYER 2: MULTI-GPU ARABIC EMBEDDINGS TRAINING")
        logger.info("=" * 60)
        logger.info(f"Distributed: {is_distributed}")
        logger.info(f"World size: {world_size} GPUs")
    
    try:
        import torch
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader, DistributedSampler
        
        if is_distributed:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        if local_rank == 0:
            logger.info(f"Device: {device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"Total GPUs: {torch.cuda.device_count()}")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return
    
    # Load data (only on rank 0, then broadcast)
    train_file = TRAINING_DIR / "train_spans.jsonl"
    val_file = TRAINING_DIR / "val_spans.jsonl"
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        return
    
    train_spans = load_jsonl(train_file)
    val_spans = load_jsonl(val_file)
    
    if local_rank == 0:
        logger.info(f"Loaded {len(train_spans)} training spans")
    
    # Create pairs
    train_pairs = create_contrastive_pairs(train_spans, num_negatives=3)
    
    if local_rank == 0:
        logger.info(f"Created {len(train_pairs)} training pairs")
    
    # Load model
    model = SentenceTransformer(BASE_MODEL, device=str(device))
    
    # Create training examples
    train_examples = [
        InputExample(texts=[t1, t2], label=label)
        for t1, t2, label in train_pairs
    ]
    
    # Create dataloader with distributed sampler
    if is_distributed:
        sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=local_rank)
        train_dataloader = DataLoader(
            train_examples, 
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
        )
    else:
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=4,
        )
    
    train_loss = losses.CosineSimilarityLoss(model)
    
    if local_rank == 0:
        logger.info(f"Starting training...")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(f"  Effective batch size: {args.batch_size * world_size}")
        logger.info(f"  Training examples: {len(train_examples)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(OUTPUT_DIR) if local_rank == 0 else None,
        show_progress_bar=(local_rank == 0),
        use_amp=True,  # Mixed precision for faster training
    )
    
    if local_rank == 0:
        logger.info(f"Model saved to {OUTPUT_DIR}")
        
        # Evaluate
        import numpy as np
        
        logger.info("Evaluating...")
        
        kibr = model.encode("الكبر")
        akbar = model.encode("أكبر")
        takabbur = model.encode("التكبر")
        
        dist_kibr_akbar = 1 - np.dot(kibr, akbar) / (np.linalg.norm(kibr) * np.linalg.norm(akbar))
        dist_kibr_takabbur = 1 - np.dot(kibr, takabbur) / (np.linalg.norm(kibr) * np.linalg.norm(takabbur))
        
        test_passed = dist_kibr_akbar > dist_kibr_takabbur
        logger.info(f"الكبر/أكبر disambiguation: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        logger.info(f"  Distance الكبر-أكبر: {dist_kibr_akbar:.4f}")
        logger.info(f"  Distance الكبر-التكبر: {dist_kibr_takabbur:.4f}")
        
        logger.info("=" * 60)
        logger.info("LAYER 2 TRAINING COMPLETE")
        logger.info("=" * 60)
    
    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
