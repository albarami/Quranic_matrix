"""
QBM Training with Official Schema

Uses schema-compliant annotations with proper BEH_* controlled IDs:
- 34 TRUE behavior classes (BEH_SPI_*, BEH_SOC_*, BEH_COG_*, etc.)
- Proper behavior forms (inner_state, speech_act, relational_act, etc.)
- Textual evaluations (EVAL_SALIH, EVAL_SAYYI, EVAL_NEUTRAL)
- Systemic contexts (SYS_SELF, SYS_GOD, SYS_CREATION, etc.)

Based on:
- vocab/behavior_concepts.json
- Bouzidani's 5-context framework
- behavioral_map_research.md
"""

import os
import sys
import json
import time
import logging
import warnings
import random
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
VOCAB_DIR = PROJECT_ROOT / "vocab"
SCHEMA_ANNOTATIONS = DATA_DIR / "annotations" / "tafsir_behavioral_annotations_schema.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Import mapping
from src.ml.qbm_vocab_mapping import (
    ARABIC_TO_BEH_ID,
    BEH_ID_TO_FORM,
    BEH_ID_TO_EVAL,
    BEH_ID_TO_SYSTEMIC,
)


def setup_device():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device

DEVICE = setup_device()


def load_schema_annotations() -> List[Dict]:
    """Load schema-compliant annotations."""
    annotations = []
    with open(SCHEMA_ANNOTATIONS, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    logger.info(f"Loaded {len(annotations):,} schema-compliant annotations")
    return annotations


def load_behavior_vocab() -> Dict:
    """Load official behavior vocabulary."""
    with open(VOCAB_DIR / "behavior_concepts.json", 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# LAYER 2: ARABIC EMBEDDINGS (Schema-aligned)
# =============================================================================

def train_layer_2():
    """Train Arabic embeddings using schema-aligned behaviors."""
    logger.info("=" * 60)
    logger.info("LAYER 2: Arabic Embeddings (Schema-aligned)")
    logger.info("=" * 60)
    
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import numpy as np
    
    annotations = load_schema_annotations()
    
    # Group by BEH_* ID
    behavior_texts = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id and text and len(text) > 20:
            behavior_texts[beh_id].append(text)
    
    logger.info(f"Found {len(behavior_texts)} behavior classes")
    
    # Group behaviors by evaluation for hard negative mining
    SALIH_BEHAVIORS = {beh_id for beh_id in behavior_texts if BEH_ID_TO_EVAL.get(beh_id) == "EVAL_SALIH"}
    SAYYI_BEHAVIORS = {beh_id for beh_id in behavior_texts if BEH_ID_TO_EVAL.get(beh_id) == "EVAL_SAYYI"}
    
    logger.info(f"SALIH behaviors: {len(SALIH_BEHAVIORS)}")
    logger.info(f"SAYYI behaviors: {len(SAYYI_BEHAVIORS)}")
    
    # Create triplets with hard negatives
    examples = []
    behavior_ids = list(behavior_texts.keys())
    
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        
        is_salih = beh_id in SALIH_BEHAVIORS
        
        # Hard negatives: same moral category
        if is_salih:
            hard_neg_pool = [b for b in SALIH_BEHAVIORS if b != beh_id and b in behavior_texts]
        else:
            hard_neg_pool = [b for b in SAYYI_BEHAVIORS if b != beh_id and b in behavior_texts]
        
        # Easy negatives: opposite moral category
        if is_salih:
            easy_neg_pool = [b for b in SAYYI_BEHAVIORS if b in behavior_texts]
        else:
            easy_neg_pool = [b for b in SALIH_BEHAVIORS if b in behavior_texts]
        
        for i in range(min(len(texts) - 1, 150)):
            anchor = texts[i]
            positive = texts[(i + 1) % len(texts)]
            
            # 60% hard negatives, 40% easy negatives
            if hard_neg_pool and random.random() < 0.6:
                neg_beh = random.choice(hard_neg_pool)
            elif easy_neg_pool:
                neg_beh = random.choice(easy_neg_pool)
            else:
                neg_beh = random.choice([b for b in behavior_ids if b != beh_id])
            
            negative = random.choice(behavior_texts[neg_beh])
            examples.append(InputExample(texts=[anchor, positive, negative]))
    
    logger.info(f"Created {len(examples):,} triplet examples")
    
    # Load model
    model = SentenceTransformer("aubmindlab/bert-base-arabertv2", device=DEVICE)
    
    # Train
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)
    train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE)
    
    output_dir = MODELS_DIR / "qbm-arabic-embeddings"
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=8,
        warmup_steps=300,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    
    # Evaluate
    test_pairs = [
        ("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION", True),
        ("BEH_SPI_FAITH", "BEH_SPI_TAQWA", True),
        ("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY", False),
        ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF", False),
    ]
    
    # Get Arabic terms for testing
    id_to_arabic = {v: k for k, v in ARABIC_TO_BEH_ID.items()}
    
    correct = 0
    for beh1, beh2, should_similar in test_pairs:
        ar1 = id_to_arabic.get(beh1, beh1)
        ar2 = id_to_arabic.get(beh2, beh2)
        emb1 = model.encode(ar1)
        emb2 = model.encode(ar2)
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        is_similar = sim > 0.6
        if is_similar == should_similar:
            correct += 1
        logger.info(f"  {ar1} <-> {ar2}: sim={sim:.3f}")
    
    accuracy = correct / len(test_pairs)
    logger.info(f"Embedding test accuracy: {accuracy:.2%}")
    
    return {"accuracy": accuracy, "examples": len(examples)}


# =============================================================================
# LAYER 3: BEHAVIORAL CLASSIFIER (34 BEH_* classes)
# =============================================================================

def train_layer_3():
    """Train 34-class behavioral classifier with BEH_* IDs."""
    logger.info("=" * 60)
    logger.info("LAYER 3: Behavioral Classifier (34 BEH_* classes)")
    logger.info("=" * 60)
    
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import f1_score, accuracy_score
    
    annotations = load_schema_annotations()
    
    # Get unique BEH_* IDs
    behavior_ids = sorted(set(ann.get("behavior_id", "") for ann in annotations if ann.get("behavior_id")))
    id_to_label = {beh_id: i for i, beh_id in enumerate(behavior_ids)}
    label_to_id = {i: beh_id for beh_id, i in id_to_label.items()}
    
    logger.info(f"Training on {len(behavior_ids)} BEH_* classes")
    
    # Create examples
    examples = []
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id in id_to_label and text and len(text) > 10:
            examples.append({
                "text": text[:512],
                "label": id_to_label[beh_id],
                "behavior_id": beh_id,
                "behavior_form": ann.get("behavior_form", "unknown"),
                "textual_eval": ann.get("textual_eval", "EVAL_UNKNOWN"),
            })
    
    logger.info(f"Created {len(examples):,} training examples")
    
    # Shuffle and split
    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]
    
    logger.info(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv2",
        num_labels=len(behavior_ids),
        ignore_mismatched_sizes=True
    )
    
    # Dataset
    class BehaviorDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(
                item["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(item["label"], dtype=torch.long)
            }
    
    train_dataset = BehaviorDataset(train_data, tokenizer)
    val_dataset = BehaviorDataset(val_data, tokenizer)
    
    output_dir = MODELS_DIR / "qbm-behavioral-classifier"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"f1": f1, "accuracy": acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save label mapping with full metadata
    label_map = {
        "behavior_ids": behavior_ids,
        "id_to_label": id_to_label,
        "label_to_id": {str(k): v for k, v in label_to_id.items()},
        "behavior_forms": {beh_id: BEH_ID_TO_FORM.get(beh_id, "unknown") for beh_id in behavior_ids},
        "textual_evals": {beh_id: BEH_ID_TO_EVAL.get(beh_id, "EVAL_UNKNOWN") for beh_id in behavior_ids},
    }
    with open(output_dir / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    logger.info(f"F1: {results.get('eval_f1', 0):.4f}, Accuracy: {results.get('eval_accuracy', 0):.4f}")
    
    return results


# =============================================================================
# LAYER 4: RELATION EXTRACTOR (Schema-aligned)
# =============================================================================

def train_layer_4():
    """Train relation extractor with Quranic causal relations."""
    logger.info("=" * 60)
    logger.info("LAYER 4: Relation Extractor (Schema-aligned)")
    logger.info("=" * 60)
    
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import f1_score, accuracy_score
    
    RELATIONS = ["CAUSES", "PREVENTS", "OPPOSITE", "INTENSIFIES", "LEADS_TO", "NONE"]
    rel_to_id = {r: i for i, r in enumerate(RELATIONS)}
    
    # Known relations using BEH_* IDs
    KNOWN_RELATIONS = [
        # CAUSES
        ("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES"),
        ("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION", "CAUSES"),
        ("BEH_COG_ARROGANCE", "BEH_SPI_DISBELIEF", "CAUSES"),
        ("BEH_SPI_DISBELIEF", "BEH_SOC_OPPRESSION", "CAUSES"),
        ("BEH_SPI_SHIRK", "BEH_SOC_OPPRESSION", "CAUSES"),
        ("BEH_SPEECH_LYING", "BEH_SPI_HYPOCRISY", "CAUSES"),
        ("BEH_EMO_ENVY", "BEH_SOC_OPPRESSION", "CAUSES"),
        
        # PREVENTS
        ("BEH_SPI_TAQWA", "BEH_SOC_TRANSGRESSION", "PREVENTS"),
        ("BEH_SPI_REMEMBRANCE", "BEH_COG_HEEDLESSNESS", "PREVENTS"),
        ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF", "PREVENTS"),
        ("BEH_EMO_PATIENCE", "BEH_EMO_ANGER", "PREVENTS"),
        ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING", "PREVENTS"),
        ("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE", "PREVENTS"),
        ("BEH_SOC_MERCY", "BEH_SOC_OPPRESSION", "PREVENTS"),
        ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION", "PREVENTS"),
        ("BEH_EMO_GRATITUDE", "BEH_SPI_DISBELIEF", "PREVENTS"),
        
        # OPPOSITE
        ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING", "OPPOSITE"),
        ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF", "OPPOSITE"),
        ("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE", "OPPOSITE"),
        ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION", "OPPOSITE"),
        ("BEH_SPI_REMEMBRANCE", "BEH_COG_HEEDLESSNESS", "OPPOSITE"),
        ("BEH_SOC_TRUSTWORTHINESS", "BEH_SOC_BETRAYAL", "OPPOSITE"),
        ("BEH_SPI_SINCERITY", "BEH_SPI_SHOWING_OFF", "OPPOSITE"),
        ("BEH_SPI_TAQWA", "BEH_SOC_TRANSGRESSION", "OPPOSITE"),
        
        # INTENSIFIES
        ("BEH_COG_ARROGANCE", "BEH_SPI_DISBELIEF", "INTENSIFIES"),
        ("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "INTENSIFIES"),
        ("BEH_SOC_TRANSGRESSION", "BEH_SPI_DISBELIEF", "INTENSIFIES"),
        
        # LEADS_TO
        ("BEH_SPI_DISBELIEF", "BEH_SOC_OPPRESSION", "LEADS_TO"),
        ("BEH_SPI_FAITH", "BEH_SPI_TAQWA", "LEADS_TO"),
        ("BEH_EMO_PATIENCE", "BEH_EMO_CONTENTMENT", "LEADS_TO"),
        ("BEH_SPI_TAQWA", "BEH_SOC_EXCELLENCE", "LEADS_TO"),
        ("BEH_SPI_REPENTANCE", "BEH_SPI_FAITH", "LEADS_TO"),
    ]
    
    annotations = load_schema_annotations()
    
    # Get context for behaviors
    behavior_context = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        ctx = ann.get("context", "")
        if beh_id and ctx:
            behavior_context[beh_id].append(ctx[:200])
    
    # Get Arabic terms
    id_to_arabic = {v: k for k, v in ARABIC_TO_BEH_ID.items()}
    
    # Create training examples
    examples = []
    
    for beh1, beh2, rel in KNOWN_RELATIONS:
        ar1 = id_to_arabic.get(beh1, beh1)
        ar2 = id_to_arabic.get(beh2, beh2)
        
        ctx1_list = behavior_context.get(beh1, [])[:5]
        ctx2_list = behavior_context.get(beh2, [])[:5]
        
        for ctx1 in ctx1_list if ctx1_list else [ar1]:
            for ctx2 in ctx2_list if ctx2_list else [ar2]:
                text = f"{ar1} [SEP] {ar2} [SEP] {ctx1} {ctx2}"
                examples.append({"text": text, "label": rel_to_id[rel]})
    
    logger.info(f"Created {len(examples)} positive relation examples")
    
    # Negative examples
    behavior_ids = list(behavior_context.keys())
    known_pairs = set((r[0], r[1]) for r in KNOWN_RELATIONS)
    known_pairs.update((r[1], r[0]) for r in KNOWN_RELATIONS)
    
    none_count = 0
    for _ in range(len(examples)):
        b1, b2 = random.sample(behavior_ids, 2)
        if (b1, b2) not in known_pairs:
            ar1 = id_to_arabic.get(b1, b1)
            ar2 = id_to_arabic.get(b2, b2)
            ctx1 = random.choice(behavior_context.get(b1, [ar1]))
            ctx2 = random.choice(behavior_context.get(b2, [ar2]))
            text = f"{ar1} [SEP] {ar2} [SEP] {ctx1} {ctx2}"
            examples.append({"text": text, "label": rel_to_id["NONE"]})
            none_count += 1
    
    logger.info(f"Added {none_count} NONE examples, Total: {len(examples)}")
    
    # Shuffle and split
    random.shuffle(examples)
    split = int(len(examples) * 0.85)
    train_data = examples[:split]
    val_data = examples[split:]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv2",
        num_labels=len(RELATIONS),
        ignore_mismatched_sizes=True
    )
    
    class RelDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(
                item["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(item["label"], dtype=torch.long)
            }
    
    train_dataset = RelDataset(train_data, tokenizer)
    val_dataset = RelDataset(val_data, tokenizer)
    
    output_dir = MODELS_DIR / "qbm-relation-extractor"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"f1": f1, "accuracy": acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    with open(output_dir / "relations.json", 'w', encoding='utf-8') as f:
        json.dump({"relations": RELATIONS, "rel_to_id": rel_to_id}, f, ensure_ascii=False)
    
    logger.info(f"F1: {results.get('eval_f1', 0):.4f}, Accuracy: {results.get('eval_accuracy', 0):.4f}")
    
    return results


# =============================================================================
# LAYER 5: GNN GRAPH REASONER
# =============================================================================

def train_layer_5():
    """Train GNN on behavior co-occurrence graph."""
    logger.info("=" * 60)
    logger.info("LAYER 5: GNN Graph Reasoner")
    logger.info("=" * 60)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    
    annotations = load_schema_annotations()
    
    # Build co-occurrence graph
    verse_behaviors = defaultdict(set)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
        if beh_id:
            verse_behaviors[verse].add(beh_id)
    
    all_behavior_ids = set()
    for behaviors in verse_behaviors.values():
        all_behavior_ids.update(behaviors)
    
    behavior_ids = sorted(all_behavior_ids)
    node_to_idx = {beh_id: i for i, beh_id in enumerate(behavior_ids)}
    
    logger.info(f"Graph nodes: {len(behavior_ids)} BEH_* behaviors")
    
    # Build weighted edges
    edge_weights = defaultdict(int)
    for verse, behaviors in verse_behaviors.items():
        behavior_list = list(behaviors)
        for i, b1 in enumerate(behavior_list):
            for b2 in behavior_list[i+1:]:
                idx1, idx2 = node_to_idx[b1], node_to_idx[b2]
                edge_weights[(idx1, idx2)] += 1
                edge_weights[(idx2, idx1)] += 1
    
    edges = list(edge_weights.keys())
    weights = [edge_weights[e] for e in edges]
    
    logger.info(f"Graph edges: {len(edges)}")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Initialize node features
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(str(MODELS_DIR / "qbm-arabic-embeddings"), device=DEVICE)
    
    id_to_arabic = {v: k for k, v in ARABIC_TO_BEH_ID.items()}
    
    node_features = []
    for beh_id in behavior_ids:
        arabic = id_to_arabic.get(beh_id, beh_id)
        emb = embedder.encode(arabic)
        node_features.append(emb)
    
    x = torch.tensor(node_features, dtype=torch.float)
    logger.info(f"Node features: {x.shape}")
    
    data = Data(x=x, edge_index=edge_index)
    
    # GAT model
    class BehaviorGAT(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=8, concat=True, dropout=0.2)
            self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=4, concat=True, dropout=0.2)
            self.conv3 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False, dropout=0.2)
            self.bn1 = nn.BatchNorm1d(hidden_dim * 8)
            self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        
        def forward(self, x, edge_index):
            x = F.elu(self.bn1(self.conv1(x, edge_index)))
            x = F.elu(self.bn2(self.conv2(x, edge_index)))
            x = self.conv3(x, edge_index)
            return x
    
    model = BehaviorGAT(x.shape[1], 128, 128)
    if DEVICE == "cuda":
        model = model.cuda()
        data = data.to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    model.train()
    best_auc = 0
    
    for epoch in range(150):
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        src, dst = data.edge_index
        pos_score = (out[src] * out[dst]).sum(dim=1)
        
        neg_dst = torch.randint(0, len(behavior_ids), (src.size(0),), device=src.device)
        neg_score = (out[src] * out[neg_dst]).sum(dim=1)
        
        loss = F.margin_ranking_loss(
            pos_score, neg_score,
            torch.ones_like(pos_score),
            margin=0.5
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 30 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pos_score = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
                neg_score = torch.sigmoid((out[src] * out[neg_dst]).sum(dim=1))
                auc = (pos_score > neg_score).float().mean().item()
                if auc > best_auc:
                    best_auc = auc
            model.train()
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        src, dst = data.edge_index
        pos_score = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
        neg_dst = torch.randint(0, len(behavior_ids), (src.size(0),), device=src.device)
        neg_score = torch.sigmoid((out[src] * out[neg_dst]).sum(dim=1))
        final_auc = (pos_score > neg_score).float().mean().item()
    
    logger.info(f"Final AUC: {final_auc:.4f}")
    
    # Save
    output_dir = MODELS_DIR / "qbm-graph-reasoner"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "node_to_idx": node_to_idx,
        "behavior_ids": behavior_ids,
    }, output_dir / "model.pt")
    
    return {"auc": final_auc, "best_auc": best_auc, "nodes": len(behavior_ids), "edges": len(edges)}


# =============================================================================
# LAYER 6: DOMAIN RERANKER
# =============================================================================

def train_layer_6():
    """Train domain reranker."""
    logger.info("=" * 60)
    logger.info("LAYER 6: Domain Reranker")
    logger.info("=" * 60)
    
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import accuracy_score, f1_score
    
    annotations = load_schema_annotations()
    
    # Group by behavior
    behavior_docs = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id and text and len(text) > 30:
            behavior_docs[beh_id].append(text)
    
    logger.info(f"Found {len(behavior_docs)} behaviors with documents")
    
    id_to_arabic = {v: k for k, v in ARABIC_TO_BEH_ID.items()}
    
    QUERY_TEMPLATES = [
        "ما هو {behavior}؟",
        "ما معنى {behavior} في القرآن؟",
        "اشرح {behavior}",
        "ما علاقة {behavior} بالإيمان؟",
        "كيف يؤثر {behavior} على القلب؟",
    ]
    
    examples = []
    behavior_ids = list(behavior_docs.keys())
    
    for beh_id, docs in behavior_docs.items():
        arabic = id_to_arabic.get(beh_id, beh_id)
        
        for template in QUERY_TEMPLATES:
            query = template.format(behavior=arabic)
            
            for doc in docs[:20]:
                examples.append({"query": query, "doc": doc[:300], "label": 1})
                
                neg_beh = random.choice([b for b in behavior_ids if b != beh_id])
                neg_doc = random.choice(behavior_docs[neg_beh])
                examples.append({"query": query, "doc": neg_doc[:300], "label": 0})
    
    logger.info(f"Created {len(examples):,} reranker examples")
    
    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]
    
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv2",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    class RerankerDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(
                item["query"],
                item["doc"],
                truncation=True,
                max_length=384,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(item["label"], dtype=torch.long)
            }
    
    train_dataset = RerankerDataset(train_data, tokenizer)
    val_dataset = RerankerDataset(val_data, tokenizer)
    
    output_dir = MODELS_DIR / "qbm-domain-reranker"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=300,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        return {"accuracy": acc, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info(f"Accuracy: {results.get('eval_accuracy', 0):.4f}, F1: {results.get('eval_f1', 0):.4f}")
    
    return results


# =============================================================================
# LAYER 7: INTEGRATION
# =============================================================================

def train_layer_7():
    """Verify all components integrate."""
    logger.info("=" * 60)
    logger.info("LAYER 7: Integration Test")
    logger.info("=" * 60)
    
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    models_to_check = [
        ("qbm-arabic-embeddings", "Layer 2"),
        ("qbm-behavioral-classifier", "Layer 3"),
        ("qbm-relation-extractor", "Layer 4"),
        ("qbm-graph-reasoner", "Layer 5"),
        ("qbm-domain-reranker", "Layer 6"),
    ]
    
    results = {}
    
    for model_name, layer in models_to_check:
        path = MODELS_DIR / model_name
        if path.exists():
            try:
                if "embeddings" in model_name:
                    model = SentenceTransformer(str(path))
                    test = model.encode("الكبر")
                    results[layer] = f"OK (dim={len(test)})"
                elif "graph" in model_name:
                    data = torch.load(path / "model.pt", weights_only=False)
                    results[layer] = f"OK ({len(data['behavior_ids'])} behaviors)"
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(path))
                    results[layer] = "OK"
                logger.info(f"  {layer}: {results[layer]}")
            except Exception as e:
                results[layer] = f"ERROR: {e}"
                logger.error(f"  {layer}: {results[layer]}")
        else:
            results[layer] = "MISSING"
            logger.warning(f"  {layer}: MISSING")
    
    success = all("OK" in v for v in results.values())
    
    return {"success": success, "models": results}


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("QBM SCHEMA-ALIGNED TRAINING")
    logger.info("=" * 70)
    
    results = {}
    total_start = time.time()
    
    # Layer 2
    start = time.time()
    results["layer_2"] = train_layer_2()
    logger.info(f"Layer 2 time: {time.time() - start:.1f}s")
    
    # Layer 3
    start = time.time()
    results["layer_3"] = train_layer_3()
    logger.info(f"Layer 3 time: {time.time() - start:.1f}s")
    
    # Layer 4
    start = time.time()
    results["layer_4"] = train_layer_4()
    logger.info(f"Layer 4 time: {time.time() - start:.1f}s")
    
    # Layer 5
    start = time.time()
    results["layer_5"] = train_layer_5()
    logger.info(f"Layer 5 time: {time.time() - start:.1f}s")
    
    # Layer 6
    start = time.time()
    results["layer_6"] = train_layer_6()
    logger.info(f"Layer 6 time: {time.time() - start:.1f}s")
    
    # Layer 7
    start = time.time()
    results["layer_7"] = train_layer_7()
    logger.info(f"Layer 7 time: {time.time() - start:.1f}s")
    
    total_time = time.time() - total_start
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SCHEMA-ALIGNED TRAINING SUMMARY")
    logger.info("=" * 70)
    
    for layer, res in results.items():
        logger.info(f"{layer}: {res}")
    
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save results
    with open(MODELS_DIR / "schema_training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
