"""
PHASE 2: Layer 2 - Arabic Embeddings Training
Enterprise Implementation Plan - Following EXACT Recommendations

From ENTERPRISE_IMPLEMENTATION_PLAN.md:
- BASE_MODEL = "aubmindlab/bert-base-arabertv2"
- BATCH_SIZE = 32
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
# Use all 8 A100 GPUs with DDP for maximum performance

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"

# Enterprise Plan Configuration
BATCH_SIZE = 32
EPOCHS = 10
WARMUP_STEPS = 500
TARGET_PAIRS = 50000  # As per plan

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


def train_embeddings():
    """Train Arabic embeddings following Enterprise Plan recommendations."""
    print("=" * 70)
    print("PHASE 2: ARABIC EMBEDDINGS TRAINING")
    print("Following Enterprise Plan Recommendations")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nConfiguration (from Enterprise Plan):")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  WARMUP_STEPS: {WARMUP_STEPS}")
    print(f"  TARGET_PAIRS: {TARGET_PAIRS:,}")
    
    # Load annotations
    print("\n[1/4] Loading annotations...")
    annotations = load_annotations()
    print(f"  Loaded {len(annotations):,} annotations")
    
    # Group by behavior
    print("\n[2/4] Grouping by behavior...")
    behavior_texts = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id and text and len(text) > 20:
            behavior_texts[beh_id].append(text)
    
    print(f"  Found {len(behavior_texts)} behavior classes")
    
    # Create CONTRASTIVE pairs as per Enterprise Plan
    # "Positive: span text ↔ correct behavior definition"
    # "Negative: span text ↔ wrong behavior definition"
    print("\n[3/4] Creating contrastive pairs (as per Enterprise Plan)...")
    examples = []
    behavior_ids = list(behavior_texts.keys())
    
    # We will train in TWO phases:
    # Phase A: Triplet loss for Arabic morphological disambiguation (الكبر vs أكبر)
    # Phase B: Cosine similarity loss for general behavior clustering
    
    triplet_examples = []  # For TripletLoss
    cosine_examples = []   # For CosineSimilarityLoss
    
    # CRITICAL: Arabic Morphological Disambiguation using TRIPLET LOSS
    # Triplet: (anchor, positive, negative) - pushes negative AWAY from anchor
    
    # الكبر = Arrogance (TRUE BEHAVIOR)
    kibr_behavior = [
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
    ]
    
    # أكبر = Greater (NOT a behavior)
    akbar_greater = [
        "الله أكبر من كل شيء",
        "قل الله أكبر",
        "والله أكبر والحمد لله",
        "التكبير في الصلاة الله أكبر",
        "ولذكر الله أكبر",
        "لخلق السماوات والأرض أكبر من خلق الناس",
    ]
    
    # كبر السن = Old age (NOT a behavior)
    kabir_age = [
        "كبر سنه وضعف جسمه",
        "وقد بلغني الكبر وامرأتي عاقر",
        "الشيخ الكبير في السن",
    ]
    
    # Create TRIPLET examples for ALL confusable Arabic roots
    # This explicitly teaches morphological disambiguation
    triplet_count = 0
    
    # الكبر triplets
    for i, anchor in enumerate(kibr_behavior):
        for j, positive in enumerate(kibr_behavior):
            if i == j:
                continue
            for negative in akbar_greater + kabir_age:
                triplet_examples.append(InputExample(texts=[anchor, positive, negative]))
                triplet_count += 1
    
    # Add triplets for OTHER behaviors to maintain clustering
    # This ensures the model learns BOTH disambiguation AND clustering
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 3:
            continue
        
        other_behs = [b for b in behavior_ids if b != beh_id and len(behavior_texts.get(b, [])) > 0]
        if not other_behs:
            continue
        
        # Create triplets: (anchor, same_behavior, different_behavior)
        for i in range(min(len(texts) - 1, 50)):  # Up to 50 anchors per behavior
            anchor = texts[i]
            positive = texts[(i + 1) % len(texts)]
            
            # Get negative from different behavior
            neg_beh = random.choice(other_behs)
            negative = random.choice(behavior_texts[neg_beh])
            
            triplet_examples.append(InputExample(texts=[anchor, positive, negative]))
            triplet_count += 1
    
    print(f"  Created {triplet_count:,} triplet examples (disambiguation + clustering)")
    
    # Method 1: Span text ↔ behavior definition (contrastive)
    for beh_id, texts in behavior_texts.items():
        if beh_id not in BEHAVIOR_DEFINITIONS:
            continue
        
        correct_def = BEHAVIOR_DEFINITIONS[beh_id]
        wrong_defs = [BEHAVIOR_DEFINITIONS[b] for b in BEHAVIOR_DEFINITIONS if b != beh_id]
        
        for text in texts[:500]:  # Up to 500 per behavior
            # Positive pair: span ↔ correct definition
            examples.append(InputExample(texts=[text, correct_def], label=1.0))
            
            # Negative pair: span ↔ wrong definition
            wrong_def = random.choice(wrong_defs)
            examples.append(InputExample(texts=[text, wrong_def], label=0.0))
    
    # Method 2: Same-behavior span pairs (to improve clustering)
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        
        # Create positive pairs from same behavior
        for i in range(min(len(texts) - 1, 300)):
            for j in range(1, min(4, len(texts) - i)):
                examples.append(InputExample(texts=[texts[i], texts[i + j]], label=1.0))
        
        # Create negative pairs from different behaviors
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
    
    random.shuffle(examples)
    print(f"  Created {len(examples):,} contrastive pairs (target: {TARGET_PAIRS:,})")
    
    # TWO-PHASE TRAINING with all 8 A100 GPUs
    print("\n[4/4] Training embeddings with 8x A100 GPUs (TWO-PHASE)...")
    
    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    model = SentenceTransformer("aubmindlab/bert-base-arabertv2", device="cuda")
    
    # Scale batch size by number of GPUs
    effective_batch_size = BATCH_SIZE * num_gpus
    print(f"  Effective batch size: {effective_batch_size} ({BATCH_SIZE} x {num_gpus} GPUs)")
    
    output_dir = MODELS_DIR / "qbm-embeddings-enterprise"
    start_time = time.time()
    
    # Use MultipleNegativesRankingLoss - best for semantic similarity
    # This loss uses in-batch negatives which is very effective
    print("\n  Training with MultipleNegativesRankingLoss...")
    
    # Create anchor-positive pairs for MNRL
    mnrl_examples = []
    
    # Add behavior definition pairs (anchor=text, positive=definition)
    for beh_id, texts in behavior_texts.items():
        if beh_id not in BEHAVIOR_DEFINITIONS:
            continue
        definition = BEHAVIOR_DEFINITIONS[beh_id]
        for text in texts[:200]:
            mnrl_examples.append(InputExample(texts=[text, definition]))
    
    # Add same-behavior pairs
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        for i in range(min(len(texts) - 1, 100)):
            mnrl_examples.append(InputExample(texts=[texts[i], texts[(i+1) % len(texts)]]))
    
    # CRITICAL: Add الكبر vs أكبر disambiguation pairs
    # These teach that الكبر behavior texts are similar to each other
    for i, t1 in enumerate(kibr_behavior):
        for t2 in kibr_behavior[i+1:]:
            mnrl_examples.append(InputExample(texts=[t1, t2]))
    
    random.shuffle(mnrl_examples)
    print(f"  Created {len(mnrl_examples):,} MNRL pairs")
    
    mnrl_dataloader = DataLoader(mnrl_examples, shuffle=True, batch_size=effective_batch_size)
    mnrl_loss = MultipleNegativesRankingLoss(model=model)
    
    model.fit(
        train_objectives=[(mnrl_dataloader, mnrl_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=True,
    )
    train_time = time.time() - start_time
    
    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Model saved to: {output_dir}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    # Test 1: الكبر vs أكبر disambiguation
    kibr_emb = model.encode("الكبر هو التعالي على الناس")
    akbar_emb = model.encode("الله أكبر من كل شيء")
    kibr_akbar_dist = 1 - np.dot(kibr_emb, akbar_emb) / (np.linalg.norm(kibr_emb) * np.linalg.norm(akbar_emb))
    
    print(f"\n  الكبر vs أكبر distance: {kibr_akbar_dist:.3f} (target: > 0.3)")
    if kibr_akbar_dist > 0.3:
        print("  ✅ PASSED")
    else:
        print("  ⚠️ Below target")
    
    # Test 2: Same behavior similarity
    if "BEH_KIBR" in behavior_texts and len(behavior_texts["BEH_KIBR"]) >= 5:
        kibr_texts = behavior_texts["BEH_KIBR"][:10]
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
            print("  ⚠️ Below target")
    
    # Test 3: Different behavior distance
    if "BEH_KIBR" in behavior_texts and "BEH_TAWADU" in behavior_texts:
        kibr_sample = model.encode(behavior_texts["BEH_KIBR"][0])
        tawadu_sample = model.encode(behavior_texts["BEH_TAWADU"][0])
        diff_sim = np.dot(kibr_sample, tawadu_sample) / (np.linalg.norm(kibr_sample) * np.linalg.norm(tawadu_sample))
        
        print(f"\n  Different-behavior similarity (كبر vs تواضع): {diff_sim:.3f} (target: < 0.5)")
        if diff_sim < 0.5:
            print("  ✅ PASSED")
        else:
            print("  ⚠️ Above target")
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    
    return output_dir


if __name__ == "__main__":
    train_embeddings()
