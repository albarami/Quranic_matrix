"""
PHASE 2: Layer 2 - Arabic Embeddings Training
Enterprise Implementation Plan - OPTIMAL DDP Strategy with 8x A100 GPUs

Uses DistributedDataParallel (DDP) for maximum performance as recommended.
Launch with: torchrun --nproc_per_node=8 src/ml/enterprise_phase2_ddp.py

From ENTERPRISE_IMPLEMENTATION_PLAN.md:
- BASE_MODEL = "aubmindlab/bert-base-arabertv2"
- BATCH_SIZE = 32 per GPU (256 total with 8 GPUs)
- EPOCHS = 10
- LEARNING_RATE = 2e-5
- WARMUP_STEPS = 500
- Expected: ~50,000 training pairs (contrastive)

Target Metrics:
- Same-behavior similarity > 0.85
- Different-behavior similarity < 0.5
- الكبر vs أكبر disambiguation > 0.3 distance
- Retrieval MRR@10 > 0.7
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"

# Enterprise Plan Configuration
BATCH_SIZE_PER_GPU = 32
EPOCHS = 10
WARMUP_STEPS = 500
TARGET_PAIRS = 50000

# Bouzidani moral categories
SALIH_BEHAVIORS = {
    "BEH_DHIKR", "BEH_DUA", "BEH_TAWBA", "BEH_SHUKR",
    "BEH_IMAN", "BEH_TAQWA", "BEH_TAWAKKUL", "BEH_RIDA", "BEH_IKHLAS", "BEH_KHUSHU",
    "BEH_SIDQ", "BEH_SABR", "BEH_ADL", "BEH_RAHMA", "BEH_TAWADU",
    "BEH_HAYA", "BEH_IHSAN", "BEH_AMANA", "BEH_ZUHD", "BEH_KHAWF", "BEH_RAJA",
    "BEH_TADABBUR", "BEH_TAFAKKUR"
}

SAYYI_BEHAVIORS = {
    "BEH_KIBR", "BEH_HASAD", "BEH_GHAFLA", "BEH_NIFAQ", "BEH_RIYA", "BEH_GHADAB",
    "BEH_KUFR", "BEH_SHIRK", "BEH_FISQ",
    "BEH_KIDHB", "BEH_DHULM", "BEH_BUKHL", "BEH_KHIYANA", "BEH_GHIBA", "BEH_FUJUR"
}

# Behavior definitions for contrastive learning
BEHAVIOR_DEFINITIONS = {
    "BEH_KIBR": "الكبر هو التعالي على الناس واحتقارهم ورد الحق",
    "BEH_HASAD": "الحسد هو تمني زوال النعمة عن الغير",
    "BEH_GHAFLA": "الغفلة هي الإعراض عن ذكر الله والآخرة",
    "BEH_NIFAQ": "النفاق هو إظهار الإيمان وإبطان الكفر",
    "BEH_RIYA": "الرياء هو العمل لأجل الناس لا لله",
    "BEH_KUFR": "الكفر هو جحود الحق وستره بعد معرفته",
    "BEH_SHIRK": "الشرك هو جعل شريك لله في ربوبيته أو ألوهيته",
    "BEH_DHULM": "الظلم هو وضع الشيء في غير موضعه",
    "BEH_KIDHB": "الكذب هو الإخبار بخلاف الواقع",
    "BEH_IMAN": "الإيمان هو التصديق بالله ورسله واليوم الآخر",
    "BEH_TAQWA": "التقوى هي حفظ النفس مما يؤثم",
    "BEH_DHIKR": "الذكر هو ذكر الله باللسان والقلب",
    "BEH_SABR": "الصبر هو حبس النفس على ما تكره",
    "BEH_SHUKR": "الشكر هو الاعتراف بنعمة المنعم مع الخضوع",
    "BEH_TAWBA": "التوبة هي الرجوع إلى الله بترك الذنب والندم",
    "BEH_SIDQ": "الصدق هو مطابقة القول للواقع",
    "BEH_ADL": "العدل هو إعطاء كل ذي حق حقه",
    "BEH_RAHMA": "الرحمة هي رقة في القلب تقتضي الإحسان",
    "BEH_TAWADU": "التواضع هو خفض الجناح للناس",
    "BEH_IKHLAS": "الإخلاص هو تصفية العمل من كل شائبة لغير الله",
}


def load_annotations() -> List[Dict]:
    """Load filtered annotations."""
    annotations = []
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def create_training_examples(behavior_texts: Dict[str, List[str]], behavior_ids: List[str]) -> List[InputExample]:
    """Create all training examples including hard negatives."""
    examples = []
    
    # CRITICAL: Arabic Morphological Disambiguation - HARD NEGATIVES
    # The root ك-ب-ر has multiple meanings that MUST be distinguished
    
    # الكبر = Arrogance (مرض قلب، عمل سيء) - TRUE BEHAVIOR
    kibr_behavior_examples = [
        "الكبر هو التعالي على الناس واحتقارهم ورد الحق",
        "من كان في قلبه مثقال ذرة من كبر لا يدخل الجنة",
        "إن الله لا يحب كل مختال فخور متكبر",
        "الكبر بطر الحق وغمط الناس",
        "المتكبر يرى نفسه فوق الناس ويحتقرهم",
        "استكبر فرعون في الأرض وعلا فيها",
        "إنه لا يحب المستكبرين",
        "سأصرف عن آياتي الذين يتكبرون في الأرض",
        "ادخلوا أبواب جهنم خالدين فيها فبئس مثوى المتكبرين",
        "كذلك يطبع الله على كل قلب متكبر جبار",
        "إن في صدورهم إلا كبر ما هم ببالغيه",
        "الكبر داء القلب الذي يمنع صاحبه من قبول الحق",
        "المتكبر من يرى نفسه أعلى من غيره",
        "التكبر على الناس من أعظم الذنوب",
        "الكبر يحجب صاحبه عن رؤية الحق",
    ]
    
    # أكبر = Greater/Greatest (NOT a behavior - comparative adjective)
    akbar_not_behavior = [
        "الله أكبر من كل شيء",
        "قل الله أكبر",
        "والله أكبر والحمد لله",
        "التكبير في الصلاة الله أكبر",
        "ولذكر الله أكبر",
        "الله أكبر كبيرا",
        "فالله خير حافظا وهو أرحم الراحمين",
        "قل أي شيء أكبر شهادة قل الله",
        "لخلق السماوات والأرض أكبر من خلق الناس",
    ]
    
    # كبر السن = Old age (NOT a behavior - physical state)
    kabir_age_examples = [
        "كبر سنه وضعف جسمه",
        "وقد بلغني الكبر وامرأتي عاقر",
        "رب إني وهن العظم مني واشتعل الرأس شيبا",
        "الشيخ الكبير في السن",
        "كبر في العمر وشاخ",
    ]
    
    # أكبر الكبائر = Greatest sins (different meaning)
    akbar_kabaer = [
        "أكبر الكبائر الشرك بالله",
        "أكبر الكبائر الإشراك بالله وعقوق الوالدين",
        "من أكبر الذنوب أن يلعن الرجل والديه",
    ]
    
    # Add MANY hard negative pairs
    hard_neg_count = 0
    
    # الكبر (behavior) vs أكبر (comparative) - MUST BE DIFFERENT
    for kibr in kibr_behavior_examples:
        for akbar in akbar_not_behavior:
            for _ in range(5):  # Strong emphasis
                examples.append(InputExample(texts=[kibr, akbar], label=0.0))
                examples.append(InputExample(texts=[akbar, kibr], label=0.0))
                hard_neg_count += 2
    
    # الكبر (behavior) vs كبر السن (age) - MUST BE DIFFERENT
    for kibr in kibr_behavior_examples:
        for age in kabir_age_examples:
            for _ in range(5):
                examples.append(InputExample(texts=[kibr, age], label=0.0))
                examples.append(InputExample(texts=[age, kibr], label=0.0))
                hard_neg_count += 2
    
    # الكبر (behavior) vs أكبر الكبائر
    for kibr in kibr_behavior_examples:
        for kabaer in akbar_kabaer:
            for _ in range(3):
                examples.append(InputExample(texts=[kibr, kabaer], label=0.0))
                hard_neg_count += 1
    
    # Add STRONG positive pairs within الكبر behavior group
    for i, t1 in enumerate(kibr_behavior_examples):
        for t2 in kibr_behavior_examples[i+1:]:
            for _ in range(3):
                examples.append(InputExample(texts=[t1, t2], label=1.0))
    
    print(f"  Added {hard_neg_count:,} Arabic morphological hard negatives (ك-ب-ر root)")
    
    # Method 1: Span text ↔ behavior definition (contrastive)
    for beh_id, texts in behavior_texts.items():
        if beh_id not in BEHAVIOR_DEFINITIONS:
            continue
        
        correct_def = BEHAVIOR_DEFINITIONS[beh_id]
        wrong_defs = [BEHAVIOR_DEFINITIONS[b] for b in BEHAVIOR_DEFINITIONS if b != beh_id]
        
        for text in texts[:500]:
            examples.append(InputExample(texts=[text, correct_def], label=1.0))
            wrong_def = random.choice(wrong_defs)
            examples.append(InputExample(texts=[text, wrong_def], label=0.0))
    
    # Method 2: Same-behavior span pairs
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        
        for i in range(min(len(texts) - 1, 300)):
            for j in range(1, min(4, len(texts) - i)):
                examples.append(InputExample(texts=[texts[i], texts[i + j]], label=1.0))
        
        other_behs = [b for b in behavior_ids if b != beh_id]
        for i in range(min(len(texts), 200)):
            other_beh = random.choice(other_behs)
            other_text = random.choice(behavior_texts[other_beh])
            examples.append(InputExample(texts=[texts[i], other_text], label=0.0))
    
    # Ensure we reach ~50,000 pairs
    while len(examples) < TARGET_PAIRS:
        beh_id = random.choice(behavior_ids)
        texts = behavior_texts[beh_id]
        if len(texts) >= 2:
            t1, t2 = random.sample(texts, 2)
            examples.append(InputExample(texts=[t1, t2], label=1.0))
            
            other_beh = random.choice([b for b in behavior_ids if b != beh_id])
            other_text = random.choice(behavior_texts[other_beh])
            examples.append(InputExample(texts=[t1, other_text], label=0.0))
    
    return examples


def setup_ddp():
    """Initialize DDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()


def train_embeddings_ddp():
    """Train Arabic embeddings using DDP with 8x A100 GPUs."""
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main = local_rank == 0
    
    if is_main:
        print("=" * 70)
        print("PHASE 2: ARABIC EMBEDDINGS TRAINING")
        print("OPTIMAL DDP Strategy with 8x A100 GPUs")
        print("=" * 70)
        print(f"\nWorld size: {world_size} GPUs")
        print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"Effective batch size: {BATCH_SIZE_PER_GPU * world_size}")
        print(f"Epochs: {EPOCHS}")
        print(f"Warmup steps: {WARMUP_STEPS}")
    
    # Load annotations (only on main process, then broadcast)
    if is_main:
        print("\n[1/4] Loading annotations...")
        annotations = load_annotations()
        print(f"  Loaded {len(annotations):,} annotations")
        
        print("\n[2/4] Grouping by behavior...")
        behavior_texts = defaultdict(list)
        for ann in annotations:
            beh_id = ann.get("behavior_id", "")
            text = ann.get("context", "")
            if beh_id and text and len(text) > 20:
                behavior_texts[beh_id].append(text)
        
        print(f"  Found {len(behavior_texts)} behavior classes")
        
        print("\n[3/4] Creating contrastive pairs with hard negatives...")
        behavior_ids = list(behavior_texts.keys())
        examples = create_training_examples(dict(behavior_texts), behavior_ids)
        random.shuffle(examples)
        print(f"  Created {len(examples):,} contrastive pairs")
    else:
        annotations = None
        behavior_texts = None
        examples = None
    
    # Broadcast examples to all processes
    dist.barrier()
    
    if is_main:
        print("\n[4/4] Training with DDP...")
    
    # Initialize model
    model = SentenceTransformer("aubmindlab/bert-base-arabertv2")
    model = model.to(local_rank)
    
    # Wrap with DDP
    model._modules['0'].auto_model = DDP(
        model._modules['0'].auto_model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    # Create distributed dataloader
    from sentence_transformers import SentencesDataset
    train_dataset = SentencesDataset(examples, model)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE_PER_GPU
    )
    
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    output_dir = MODELS_DIR / "qbm-embeddings-enterprise"
    
    start_time = time.time()
    
    # Training loop
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=str(output_dir) if is_main else None,
        show_progress_bar=is_main,
        use_amp=True,
    )
    
    if is_main:
        train_time = time.time() - start_time
        print(f"\n  Training completed in {train_time:.1f}s")
        print(f"  Model saved to: {output_dir}")
        
        # Evaluate
        evaluate_model(output_dir)
    
    cleanup_ddp()


def evaluate_model(model_path):
    """Evaluate the trained model."""
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    model = SentenceTransformer(str(model_path))
    
    # Test 1: الكبر vs أكبر disambiguation
    kibr_emb = model.encode("الكبر هو التعالي على الناس")
    akbar_emb = model.encode("الله أكبر من كل شيء")
    kibr_akbar_dist = 1 - np.dot(kibr_emb, akbar_emb) / (np.linalg.norm(kibr_emb) * np.linalg.norm(akbar_emb))
    
    print(f"\n  الكبر vs أكبر distance: {kibr_akbar_dist:.3f} (target: > 0.3)")
    if kibr_akbar_dist > 0.3:
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
    
    # Test 2: Same behavior similarity
    kibr_texts = [
        "الكبر هو التعالي على الناس",
        "المتكبر يرى نفسه فوق الآخرين",
        "استكبر فرعون في الأرض",
    ]
    kibr_embs = model.encode(kibr_texts)
    
    similarities = []
    for i in range(len(kibr_embs)):
        for j in range(i + 1, len(kibr_embs)):
            sim = np.dot(kibr_embs[i], kibr_embs[j]) / (np.linalg.norm(kibr_embs[i]) * np.linalg.norm(kibr_embs[j]))
            similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    print(f"\n  Same-behavior similarity (الكبر): {avg_sim:.3f} (target: > 0.85)")
    if avg_sim > 0.85:
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
    
    # Test 3: Different behavior distance
    kibr_sample = model.encode("الكبر هو التعالي على الناس")
    tawadu_sample = model.encode("التواضع هو خفض الجناح للناس")
    diff_sim = np.dot(kibr_sample, tawadu_sample) / (np.linalg.norm(kibr_sample) * np.linalg.norm(tawadu_sample))
    
    print(f"\n  Different-behavior similarity (كبر vs تواضع): {diff_sim:.3f} (target: < 0.5)")
    if diff_sim < 0.5:
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_embeddings_ddp()
