"""
QBM Improved Layer Training
Fixes for underperforming layers (2, 4, 5, 6)

Issues identified:
- Layer 2: 50% accuracy (random) - needs more epochs, harder negatives
- Layer 4: 50% accuracy - needs more relation training pairs
- Layer 5: 54% AUC - needs richer graph structure
- Layer 6: 60% accuracy - needs more QBM-specific data
"""

import os
import sys
import json
import time
import logging
import warnings
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Suppress warnings
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

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_annotations.jsonl"


def setup_device():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device

DEVICE = setup_device()


def load_annotations() -> List[Dict]:
    """Load all annotations."""
    annotations = []
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


# =============================================================================
# LAYER 2: IMPROVED ARABIC EMBEDDINGS
# =============================================================================

def train_layer_2_improved():
    """
    Improved Arabic Embeddings with:
    - 10 epochs instead of 3
    - Hard negative mining
    - Multiple loss functions
    """
    logger.info("=" * 50)
    logger.info("LAYER 2: Improved Arabic Embeddings")
    logger.info("=" * 50)
    
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    
    annotations = load_annotations()
    
    # Group by behavior with more context
    behavior_texts = defaultdict(list)
    for ann in annotations:
        behavior = ann.get("behavior_ar", "")
        text = ann.get("context", "")
        if behavior and text and len(text) > 20:
            behavior_texts[behavior].append(text)
    
    logger.info(f"Found {len(behavior_texts)} behaviors")
    
    # Create HARD triplets (similar behaviors as negatives)
    # Define behavior similarity groups for hard negatives
    SIMILAR_GROUPS = [
        ["كبر", "ظلم", "كفر", "شرك"],  # Negative behaviors
        ["إيمان", "صدق", "صبر", "رضا"],  # Positive behaviors
        ["ذكر", "قلب", "سليم"],  # Heart-related
        ["نبي", "رسول", "ملائكة"],  # Divine messengers
    ]
    
    behavior_to_group = {}
    for i, group in enumerate(SIMILAR_GROUPS):
        for b in group:
            behavior_to_group[b] = i
    
    examples = []
    behaviors = list(behavior_texts.keys())
    
    for behavior, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        
        # Get hard negatives (similar behaviors)
        group_id = behavior_to_group.get(behavior, -1)
        hard_negatives = [b for b in behaviors if b != behavior and behavior_to_group.get(b, -2) == group_id]
        easy_negatives = [b for b in behaviors if b != behavior and behavior_to_group.get(b, -2) != group_id]
        
        for i in range(min(len(texts) - 1, 200)):
            anchor = texts[i]
            positive = texts[(i + 1) % len(texts)]
            
            # 70% hard negatives, 30% easy negatives
            if hard_negatives and random.random() < 0.7:
                neg_behavior = random.choice(hard_negatives)
            elif easy_negatives:
                neg_behavior = random.choice(easy_negatives)
            else:
                neg_behavior = random.choice([b for b in behaviors if b != behavior])
            
            negative = random.choice(behavior_texts[neg_behavior])
            examples.append(InputExample(texts=[anchor, positive, negative]))
    
    logger.info(f"Created {len(examples)} triplet examples (with hard negatives)")
    
    # Load model
    model = SentenceTransformer("aubmindlab/bert-base-arabertv2", device=DEVICE)
    
    # Train with more epochs
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)
    train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE)
    
    output_dir = MODELS_DIR / "qbm-arabic-embeddings"
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,  # Increased from 3
        warmup_steps=500,
        output_path=str(output_dir),
        show_progress_bar=True,
        checkpoint_save_steps=1000,
    )
    
    # Evaluate with proper test pairs
    test_pairs = [
        ("الكبر", "التكبر", True),   # Same concept
        ("الكبر", "الظلم", True),    # Related negative
        ("الكبر", "التواضع", False), # Opposite
        ("الإيمان", "الصدق", True),  # Related positive
        ("الإيمان", "الكفر", False), # Opposite
        ("الذكر", "الغفلة", False),  # Opposite
    ]
    
    correct = 0
    for t1, t2, should_be_similar in test_pairs:
        emb1 = model.encode(t1)
        emb2 = model.encode(t2)
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        is_similar = sim > 0.7
        if is_similar == should_be_similar:
            correct += 1
        logger.info(f"  {t1} <-> {t2}: sim={sim:.3f}, expected={should_be_similar}, got={is_similar}")
    
    accuracy = correct / len(test_pairs)
    logger.info(f"Test accuracy: {accuracy:.2%}")
    
    return {"accuracy": accuracy, "examples": len(examples)}


# =============================================================================
# LAYER 4: IMPROVED RELATION EXTRACTOR
# =============================================================================

def train_layer_4_improved():
    """
    Improved Relation Extractor with:
    - More training pairs from Quranic knowledge
    - Better negative sampling
    - Class balancing
    """
    logger.info("=" * 50)
    logger.info("LAYER 4: Improved Relation Extractor")
    logger.info("=" * 50)
    
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    RELATIONS = ["CAUSES", "PREVENTS", "OPPOSITE", "INTENSIFIES", "LEADS_TO", "NONE"]
    rel_to_id = {r: i for i, r in enumerate(RELATIONS)}
    
    # EXPANDED known relations from Quranic behavioral knowledge
    KNOWN_RELATIONS = [
        # CAUSES relationships
        ("الغفلة", "الكبر", "CAUSES"),
        ("الكبر", "الظلم", "CAUSES"),
        ("الكبر", "قسوة القلب", "CAUSES"),
        ("الكفر", "الظلم", "CAUSES"),
        ("الشرك", "الظلم", "CAUSES"),
        ("الكذب", "الكفر", "CAUSES"),
        ("الغفلة", "قسوة القلب", "CAUSES"),
        ("الكبر", "الكفر", "CAUSES"),
        ("الظلم", "العذاب", "CAUSES"),
        ("الشرك", "الضلال", "CAUSES"),
        
        # PREVENTS relationships
        ("التقوى", "المعصية", "PREVENTS"),
        ("الذكر", "الغفلة", "PREVENTS"),
        ("الإيمان", "الكفر", "PREVENTS"),
        ("الصبر", "الجزع", "PREVENTS"),
        ("الصدق", "الكذب", "PREVENTS"),
        ("التواضع", "الكبر", "PREVENTS"),
        ("الرحمة", "القسوة", "PREVENTS"),
        ("العدل", "الظلم", "PREVENTS"),
        ("الشكر", "الكفران", "PREVENTS"),
        ("الخشوع", "الغفلة", "PREVENTS"),
        
        # OPPOSITE relationships
        ("الصدق", "الكذب", "OPPOSITE"),
        ("الإيمان", "الكفر", "OPPOSITE"),
        ("التواضع", "الكبر", "OPPOSITE"),
        ("العدل", "الظلم", "OPPOSITE"),
        ("الرحمة", "القسوة", "OPPOSITE"),
        ("الصبر", "الجزع", "OPPOSITE"),
        ("الذكر", "الغفلة", "OPPOSITE"),
        ("الشكر", "الكفران", "OPPOSITE"),
        ("الهدى", "الضلال", "OPPOSITE"),
        ("النور", "الظلمات", "OPPOSITE"),
        
        # INTENSIFIES relationships
        ("الإصرار", "قسوة القلب", "INTENSIFIES"),
        ("التكرار", "الكفر", "INTENSIFIES"),
        ("الاستكبار", "الظلم", "INTENSIFIES"),
        ("الجحود", "الكفر", "INTENSIFIES"),
        ("العناد", "الضلال", "INTENSIFIES"),
        
        # LEADS_TO relationships
        ("الكفر", "الظلم", "LEADS_TO"),
        ("الإيمان", "الهدى", "LEADS_TO"),
        ("الصبر", "الفلاح", "LEADS_TO"),
        ("التقوى", "الجنة", "LEADS_TO"),
        ("الكبر", "النار", "LEADS_TO"),
        ("الظلم", "الهلاك", "LEADS_TO"),
        ("الشكر", "الزيادة", "LEADS_TO"),
        ("الكفران", "العذاب", "LEADS_TO"),
    ]
    
    # Load context from annotations
    annotations = load_annotations()
    behavior_context = {}
    for ann in annotations:
        b = ann.get("behavior_ar", "")
        ctx = ann.get("context", "")
        if b and ctx:
            if b not in behavior_context:
                behavior_context[b] = []
            behavior_context[b].append(ctx[:200])
    
    # Create training examples with augmentation
    examples = []
    
    # Positive examples (with context variations)
    for e1, e2, rel in KNOWN_RELATIONS:
        ctx1_list = behavior_context.get(e1, [e1])[:5]
        ctx2_list = behavior_context.get(e2, [e2])[:5]
        
        for ctx1 in ctx1_list:
            for ctx2 in ctx2_list:
                text = f"{e1} [SEP] {e2} [SEP] {ctx1} {ctx2}"
                examples.append({"text": text, "label": rel_to_id[rel]})
    
    logger.info(f"Created {len(examples)} positive examples")
    
    # Balanced negative examples (NONE relation)
    behaviors = list(behavior_context.keys())
    known_pairs = set((r[0], r[1]) for r in KNOWN_RELATIONS)
    known_pairs.update((r[1], r[0]) for r in KNOWN_RELATIONS)
    
    none_examples = []
    for _ in range(len(examples)):
        b1, b2 = random.sample(behaviors, 2)
        if (b1, b2) not in known_pairs:
            ctx1 = random.choice(behavior_context.get(b1, [b1]))
            ctx2 = random.choice(behavior_context.get(b2, [b2]))
            text = f"{b1} [SEP] {b2} [SEP] {ctx1} {ctx2}"
            none_examples.append({"text": text, "label": rel_to_id["NONE"]})
    
    examples.extend(none_examples)
    logger.info(f"Total examples: {len(examples)}")
    
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
        num_train_epochs=10,  # Increased
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_steps=100,
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
    
    # Save relation mapping
    with open(output_dir / "relations.json", 'w', encoding='utf-8') as f:
        json.dump({"relations": RELATIONS, "rel_to_id": rel_to_id}, f, ensure_ascii=False)
    
    logger.info(f"F1: {results.get('eval_f1', 0):.4f}, Accuracy: {results.get('eval_accuracy', 0):.4f}")
    
    return results


# =============================================================================
# LAYER 5: IMPROVED GNN
# =============================================================================

def train_layer_5_improved():
    """
    Improved GNN with:
    - Verse nodes for richer structure
    - Edge weights from co-occurrence frequency
    - More training epochs
    """
    logger.info("=" * 50)
    logger.info("LAYER 5: Improved GNN Graph Reasoner")
    logger.info("=" * 50)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    
    annotations = load_annotations()
    
    # Build richer graph with verse co-occurrence weights
    verse_behaviors = defaultdict(set)
    for ann in annotations:
        behavior = ann.get("behavior_ar", "")
        verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
        if behavior:
            verse_behaviors[verse].add(behavior)
    
    # Get behaviors and create node mapping
    all_behaviors = set()
    for behaviors in verse_behaviors.values():
        all_behaviors.update(behaviors)
    
    behaviors = sorted(all_behaviors)
    node_to_idx = {b: i for i, b in enumerate(behaviors)}
    
    # Build weighted edges
    edge_weights = defaultdict(int)
    for verse, verse_behaviors_set in verse_behaviors.items():
        behavior_list = list(verse_behaviors_set)
        for i, b1 in enumerate(behavior_list):
            for b2 in behavior_list[i+1:]:
                idx1, idx2 = node_to_idx[b1], node_to_idx[b2]
                edge_weights[(idx1, idx2)] += 1
                edge_weights[(idx2, idx1)] += 1
    
    # Create edge index and weights
    edges = list(edge_weights.keys())
    weights = [edge_weights[e] for e in edges]
    
    logger.info(f"Graph: {len(behaviors)} nodes, {len(edges)} unique edges")
    logger.info(f"Max edge weight: {max(weights)}, Mean: {sum(weights)/len(weights):.1f}")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # Initialize node features with behavior embeddings
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(str(MODELS_DIR / "qbm-arabic-embeddings"), device=DEVICE)
    
    node_features = []
    for b in behaviors:
        emb = embedder.encode(b)
        node_features.append(emb)
    
    x = torch.tensor(node_features, dtype=torch.float)
    logger.info(f"Node features shape: {x.shape}")
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    # Improved GAT model
    class ImprovedGAT(nn.Module):
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
    
    model = ImprovedGAT(x.shape[1], 128, 128)
    if DEVICE == "cuda":
        model = model.cuda()
        data = data.to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Train
    model.train()
    best_auc = 0
    
    for epoch in range(200):  # More epochs
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        # Weighted positive edges
        src, dst = data.edge_index
        pos_score = (out[src] * out[dst]).sum(dim=1)
        
        # Hard negative sampling
        neg_dst = torch.randint(0, len(behaviors), (src.size(0),), device=src.device)
        neg_score = (out[src] * out[neg_dst]).sum(dim=1)
        
        # Margin ranking loss
        margin = 0.5
        loss = F.margin_ranking_loss(
            pos_score, neg_score,
            torch.ones_like(pos_score),
            margin=margin
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            # Evaluate
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
        neg_dst = torch.randint(0, len(behaviors), (src.size(0),), device=src.device)
        neg_score = torch.sigmoid((out[src] * out[neg_dst]).sum(dim=1))
        final_auc = (pos_score > neg_score).float().mean().item()
    
    logger.info(f"Final AUC: {final_auc:.4f}")
    
    # Save
    output_dir = MODELS_DIR / "qbm-graph-reasoner"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "node_to_idx": node_to_idx,
        "behaviors": behaviors,
    }, output_dir / "model.pt")
    
    return {"auc": final_auc, "best_auc": best_auc}


# =============================================================================
# LAYER 6: IMPROVED RERANKER
# =============================================================================

def train_layer_6_improved():
    """
    Improved Reranker with:
    - More diverse query templates
    - Hard negative mining
    - More training data
    """
    logger.info("=" * 50)
    logger.info("LAYER 6: Improved Domain Reranker")
    logger.info("=" * 50)
    
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import accuracy_score, f1_score
    
    annotations = load_annotations()
    
    # Group by behavior
    behavior_docs = defaultdict(list)
    for ann in annotations:
        behavior = ann.get("behavior_ar", "")
        text = ann.get("context", "")
        if behavior and text and len(text) > 30:
            behavior_docs[behavior].append(text)
    
    # Query templates for diversity
    QUERY_TEMPLATES = [
        "ما هو {behavior}؟",
        "ما معنى {behavior} في القرآن؟",
        "اشرح {behavior}",
        "ما علاقة {behavior} بالإيمان؟",
        "كيف يؤثر {behavior} على القلب؟",
        "ما حكم {behavior}؟",
        "ما عاقبة {behavior}؟",
        "كيف نتجنب {behavior}؟",
        "ما أسباب {behavior}؟",
        "ما علاج {behavior}؟",
    ]
    
    examples = []
    behaviors = list(behavior_docs.keys())
    
    for behavior, docs in behavior_docs.items():
        for template in QUERY_TEMPLATES:
            query = template.format(behavior=behavior)
            
            for doc in docs[:30]:
                # Positive
                examples.append({"query": query, "doc": doc[:300], "label": 1})
                
                # Hard negative (similar behavior)
                similar = [b for b in behaviors if b != behavior and any(
                    c in b for c in behavior
                )]
                if similar:
                    neg_behavior = random.choice(similar)
                else:
                    neg_behavior = random.choice([b for b in behaviors if b != behavior])
                
                neg_doc = random.choice(behavior_docs[neg_behavior])
                examples.append({"query": query, "doc": neg_doc[:300], "label": 0})
    
    logger.info(f"Created {len(examples)} reranker examples")
    
    # Shuffle and split
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
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
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
# MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("QBM IMPROVED LAYER TRAINING")
    logger.info("=" * 60)
    
    results = {}
    
    # Layer 2
    start = time.time()
    results["layer_2"] = train_layer_2_improved()
    logger.info(f"Layer 2 time: {time.time() - start:.1f}s")
    
    # Layer 4
    start = time.time()
    results["layer_4"] = train_layer_4_improved()
    logger.info(f"Layer 4 time: {time.time() - start:.1f}s")
    
    # Layer 5
    start = time.time()
    results["layer_5"] = train_layer_5_improved()
    logger.info(f"Layer 5 time: {time.time() - start:.1f}s")
    
    # Layer 6
    start = time.time()
    results["layer_6"] = train_layer_6_improved()
    logger.info(f"Layer 6 time: {time.time() - start:.1f}s")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("IMPROVED TRAINING SUMMARY")
    logger.info("=" * 60)
    
    for layer, res in results.items():
        logger.info(f"{layer}: {res}")
    
    # Save results
    with open(MODELS_DIR / "improved_training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
