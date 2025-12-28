"""
QBM Complete Rebuild - Based on Bouzidani's Framework

This script rebuilds ALL components from scratch:
1. Filter annotations (34 TRUE behaviors, exclude 13 non-behaviors)
2. Layer 2: Arabic Embeddings
3. Layer 3: Behavioral Classifier (34 classes)
4. Layer 4: Relation Extractor
5. Layer 5: GNN Graph Reasoner
6. Layer 6: Domain Reranker
7. Update graph nodes/edges
8. Integration verification

Based on:
- السلوك البشري في سياقه القرآني (Bouzidani)
- مصفوفة التصنيف القرآني لسلوك الإنسان (جدول 02)
- 5-axis classification system
"""

import os
import sys
import json
import time
import logging
import warnings
import random
from pathlib import Path
from typing import Dict, List, Any
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
ANNOTATIONS_DIR = DATA_DIR / "annotations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Import schema
from src.ml.qbm_5axis_schema import (
    ARABIC_TO_ID, is_true_behavior, NON_BEHAVIOR_IDS,
    BehaviorCategory, MoralEvaluation, IntentionStatus
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


# =============================================================================
# STEP 1: FILTER ANNOTATIONS
# =============================================================================

def filter_annotations():
    """Filter annotations to TRUE behaviors only."""
    logger.info("=" * 60)
    logger.info("STEP 1: FILTERING ANNOTATIONS")
    logger.info("=" * 60)
    
    input_file = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
    output_file = ANNOTATIONS_DIR / "tafsir_behavioral_5axis.jsonl"
    
    kept = 0
    removed = 0
    behavior_counts = Counter()
    removed_counts = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    ann = json.loads(line)
                    arabic = ann.get('behavior_ar', '')
                    
                    if is_true_behavior(arabic):
                        beh_id = ARABIC_TO_ID.get(arabic)
                        ann['behavior_id'] = beh_id
                        f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')
                        kept += 1
                        behavior_counts[beh_id] += 1
                    else:
                        removed += 1
                        removed_counts[arabic] += 1
    
    logger.info(f"Kept: {kept:,} TRUE behaviors")
    logger.info(f"Removed: {removed:,} non-behaviors")
    logger.info(f"Unique behavior classes: {len(behavior_counts)}")
    
    logger.info("\nTop behaviors:")
    for beh_id, count in behavior_counts.most_common(10):
        logger.info(f"  {beh_id}: {count:,}")
    
    logger.info("\nRemoved (non-behaviors):")
    for arabic, count in removed_counts.most_common():
        logger.info(f"  {arabic}: {count:,}")
    
    return output_file, behavior_counts


def load_filtered_annotations(filepath) -> List[Dict]:
    """Load filtered annotations."""
    annotations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


# =============================================================================
# STEP 2: ARABIC EMBEDDINGS
# =============================================================================

def train_embeddings(annotations: List[Dict]):
    """Train Arabic embeddings with behavior semantics."""
    logger.info("=" * 60)
    logger.info("STEP 2: ARABIC EMBEDDINGS")
    logger.info("=" * 60)
    
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import numpy as np
    
    # Group by behavior ID
    behavior_texts = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id and text and len(text) > 20:
            behavior_texts[beh_id].append(text)
    
    logger.info(f"Found {len(behavior_texts)} behavior classes")
    
    # Define moral categories for hard negative mining
    # Based on Bouzidani: عمل صالح vs عمل سيء
    SALIH_BEHAVIORS = {
        "BEH_DHIKR", "BEH_DUA", "BEH_TAWBA", "BEH_SHUKR",
        "BEH_IMAN", "BEH_TAQWA", "BEH_TAWAKKUL", "BEH_RIDA", "BEH_IKHLAS", "BEH_KHUSHU",
        "BEH_SIDQ", "BEH_SABR", "BEH_ADL", "BEH_RAHMA", "BEH_TAWADU", 
        "BEH_HAYA", "BEH_IHSAN", "BEH_AMANA", "BEH_ZUHD"
    }
    SAYYI_BEHAVIORS = {
        "BEH_KIBR", "BEH_HASAD", "BEH_GHAFLA", "BEH_NIFAQ", "BEH_RIYA",
        "BEH_KUFR", "BEH_SHIRK", "BEH_FISQ",
        "BEH_KIDHB", "BEH_DHULM", "BEH_BUKHL", "BEH_KHIYANA", "BEH_GHIBA", "BEH_FUJUR"
    }
    
    # Create triplets
    examples = []
    behavior_ids = list(behavior_texts.keys())
    
    for beh_id, texts in behavior_texts.items():
        if len(texts) < 2:
            continue
        
        is_salih = beh_id in SALIH_BEHAVIORS
        
        # Hard negatives: same moral category
        if is_salih:
            hard_neg_pool = [b for b in SALIH_BEHAVIORS if b != beh_id and b in behavior_texts]
            easy_neg_pool = [b for b in SAYYI_BEHAVIORS if b in behavior_texts]
        else:
            hard_neg_pool = [b for b in SAYYI_BEHAVIORS if b != beh_id and b in behavior_texts]
            easy_neg_pool = [b for b in SALIH_BEHAVIORS if b in behavior_texts]
        
        for i in range(min(len(texts) - 1, 100)):
            anchor = texts[i]
            positive = texts[(i + 1) % len(texts)]
            
            # 60% hard negatives
            if hard_neg_pool and random.random() < 0.6:
                neg_beh = random.choice(hard_neg_pool)
            elif easy_neg_pool:
                neg_beh = random.choice(easy_neg_pool)
            else:
                neg_beh = random.choice([b for b in behavior_ids if b != beh_id])
            
            negative = random.choice(behavior_texts[neg_beh])
            examples.append(InputExample(texts=[anchor, positive, negative]))
    
    logger.info(f"Created {len(examples):,} triplet examples")
    
    # Train
    model = SentenceTransformer("aubmindlab/bert-base-arabertv2", device=DEVICE)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)
    train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE)
    
    output_dir = MODELS_DIR / "qbm-embeddings-v2"
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=6,
        warmup_steps=200,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    
    logger.info(f"Saved embeddings to {output_dir}")
    return output_dir


# =============================================================================
# STEP 3: BEHAVIORAL CLASSIFIER
# =============================================================================

def train_classifier(annotations: List[Dict]):
    """Train 34-class behavioral classifier."""
    logger.info("=" * 60)
    logger.info("STEP 3: BEHAVIORAL CLASSIFIER (34 classes)")
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
    
    # Get unique behavior IDs
    behavior_ids = sorted(set(ann.get("behavior_id", "") for ann in annotations if ann.get("behavior_id")))
    id_to_label = {beh_id: i for i, beh_id in enumerate(behavior_ids)}
    
    logger.info(f"Training on {len(behavior_ids)} behavior classes")
    
    # Create examples
    examples = []
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id in id_to_label and text and len(text) > 10:
            examples.append({"text": text[:512], "label": id_to_label[beh_id]})
    
    logger.info(f"Created {len(examples):,} examples")
    
    # Shuffle and split
    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv2",
        num_labels=len(behavior_ids),
        ignore_mismatched_sizes=True
    )
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(item["text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(), "attention_mask": enc["attention_mask"].squeeze(), "labels": torch.tensor(item["label"], dtype=torch.long)}
    
    train_dataset = Dataset(train_data, tokenizer)
    val_dataset = Dataset(val_data, tokenizer)
    
    output_dir = MODELS_DIR / "qbm-classifier-v2"
    
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
        return {"f1": f1_score(labels, preds, average='macro', zero_division=0), "accuracy": accuracy_score(labels, preds)}
    
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save label mapping
    with open(output_dir / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump({"behavior_ids": behavior_ids, "id_to_label": id_to_label}, f, ensure_ascii=False, indent=2)
    
    logger.info(f"F1: {results.get('eval_f1', 0):.4f}, Accuracy: {results.get('eval_accuracy', 0):.4f}")
    return output_dir, results


# =============================================================================
# STEP 4: RELATION EXTRACTOR
# =============================================================================

def train_relation_extractor(annotations: List[Dict]):
    """Train relation extractor with Quranic causal relations."""
    logger.info("=" * 60)
    logger.info("STEP 4: RELATION EXTRACTOR")
    logger.info("=" * 60)
    
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from sklearn.metrics import f1_score, accuracy_score
    
    RELATIONS = ["CAUSES", "PREVENTS", "OPPOSITE", "LEADS_TO", "NONE"]
    rel_to_id = {r: i for i, r in enumerate(RELATIONS)}
    
    # Known Quranic relations
    KNOWN_RELATIONS = [
        ("BEH_GHAFLA", "BEH_KIBR", "CAUSES"),
        ("BEH_KIBR", "BEH_DHULM", "CAUSES"),
        ("BEH_KIBR", "BEH_KUFR", "CAUSES"),
        ("BEH_KIDHB", "BEH_NIFAQ", "CAUSES"),
        ("BEH_TAQWA", "BEH_FISQ", "PREVENTS"),
        ("BEH_DHIKR", "BEH_GHAFLA", "PREVENTS"),
        ("BEH_IMAN", "BEH_KUFR", "PREVENTS"),
        ("BEH_SABR", "BEH_GHADAB", "PREVENTS"),
        ("BEH_SIDQ", "BEH_KIDHB", "OPPOSITE"),
        ("BEH_IMAN", "BEH_KUFR", "OPPOSITE"),
        ("BEH_TAWADU", "BEH_KIBR", "OPPOSITE"),
        ("BEH_ADL", "BEH_DHULM", "OPPOSITE"),
        ("BEH_IKHLAS", "BEH_RIYA", "OPPOSITE"),
        ("BEH_IMAN", "BEH_TAQWA", "LEADS_TO"),
        ("BEH_TAQWA", "BEH_IHSAN", "LEADS_TO"),
        ("BEH_TAWBA", "BEH_IMAN", "LEADS_TO"),
    ]
    
    # Get Arabic terms
    id_to_arabic = {v: k for k, v in ARABIC_TO_ID.items() if v.startswith("BEH_")}
    
    # Get context for behaviors
    behavior_context = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        ctx = ann.get("context", "")
        if beh_id and ctx:
            behavior_context[beh_id].append(ctx[:200])
    
    # Create examples
    examples = []
    for beh1, beh2, rel in KNOWN_RELATIONS:
        ar1 = id_to_arabic.get(beh1, beh1)
        ar2 = id_to_arabic.get(beh2, beh2)
        ctx1_list = behavior_context.get(beh1, [ar1])[:3]
        ctx2_list = behavior_context.get(beh2, [ar2])[:3]
        
        for ctx1 in ctx1_list:
            for ctx2 in ctx2_list:
                text = f"{ar1} [SEP] {ar2} [SEP] {ctx1} {ctx2}"
                examples.append({"text": text, "label": rel_to_id[rel]})
    
    # Add NONE examples
    behavior_ids = list(behavior_context.keys())
    known_pairs = set((r[0], r[1]) for r in KNOWN_RELATIONS)
    for _ in range(len(examples)):
        b1, b2 = random.sample(behavior_ids, 2)
        if (b1, b2) not in known_pairs:
            ar1 = id_to_arabic.get(b1, b1)
            ar2 = id_to_arabic.get(b2, b2)
            ctx1 = random.choice(behavior_context.get(b1, [ar1]))
            ctx2 = random.choice(behavior_context.get(b2, [ar2]))
            examples.append({"text": f"{ar1} [SEP] {ar2} [SEP] {ctx1} {ctx2}", "label": rel_to_id["NONE"]})
    
    logger.info(f"Created {len(examples)} relation examples")
    
    random.shuffle(examples)
    split = int(len(examples) * 0.85)
    train_data, val_data = examples[:split], examples[split:]
    
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=len(RELATIONS), ignore_mismatched_sizes=True)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(item["text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(), "attention_mask": enc["attention_mask"].squeeze(), "labels": torch.tensor(item["label"], dtype=torch.long)}
    
    output_dir = MODELS_DIR / "qbm-relations-v2"
    training_args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=6, per_device_train_batch_size=16, warmup_steps=50, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, report_to="none")
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        return {"f1": f1_score(labels, preds, average='macro', zero_division=0), "accuracy": accuracy_score(labels, preds)}
    
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset(train_data, tokenizer), eval_dataset=Dataset(val_data, tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    with open(output_dir / "relations.json", 'w') as f:
        json.dump({"relations": RELATIONS}, f)
    
    logger.info(f"F1: {results.get('eval_f1', 0):.4f}")
    return output_dir, results


# =============================================================================
# STEP 5: GNN GRAPH REASONER
# =============================================================================

def train_gnn(annotations: List[Dict], embeddings_dir: Path):
    """Train GNN on behavior co-occurrence graph."""
    logger.info("=" * 60)
    logger.info("STEP 5: GNN GRAPH REASONER")
    logger.info("=" * 60)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    from sentence_transformers import SentenceTransformer
    
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
    
    logger.info(f"Graph nodes: {len(behavior_ids)}")
    
    # Build edges
    edge_weights = defaultdict(int)
    for verse, behaviors in verse_behaviors.items():
        behavior_list = list(behaviors)
        for i, b1 in enumerate(behavior_list):
            for b2 in behavior_list[i+1:]:
                idx1, idx2 = node_to_idx[b1], node_to_idx[b2]
                edge_weights[(idx1, idx2)] += 1
                edge_weights[(idx2, idx1)] += 1
    
    edges = list(edge_weights.keys())
    logger.info(f"Graph edges: {len(edges)}")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Node features from embeddings
    embedder = SentenceTransformer(str(embeddings_dir), device=DEVICE)
    id_to_arabic = {v: k for k, v in ARABIC_TO_ID.items()}
    
    node_features = []
    for beh_id in behavior_ids:
        arabic = id_to_arabic.get(beh_id, beh_id)
        emb = embedder.encode(arabic)
        node_features.append(emb)
    
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    # GAT model
    class BehaviorGAT(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True, dropout=0.2)
            self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False, dropout=0.2)
            self.bn = nn.BatchNorm1d(hidden_dim * 4)
        def forward(self, x, edge_index):
            x = F.elu(self.bn(self.conv1(x, edge_index)))
            return self.conv2(x, edge_index)
    
    model = BehaviorGAT(x.shape[1], 128, 128)
    if DEVICE == "cuda":
        model = model.cuda()
        data = data.to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        src, dst = data.edge_index
        pos_score = (out[src] * out[dst]).sum(dim=1)
        neg_dst = torch.randint(0, len(behavior_ids), (src.size(0),), device=src.device)
        neg_score = (out[src] * out[neg_dst]).sum(dim=1)
        loss = F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=0.5)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
    
    # Save
    output_dir = MODELS_DIR / "qbm-gnn-v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "node_to_idx": node_to_idx, "behavior_ids": behavior_ids}, output_dir / "model.pt")
    
    logger.info(f"Saved GNN to {output_dir}")
    return output_dir


# =============================================================================
# STEP 6: DOMAIN RERANKER
# =============================================================================

def train_reranker(annotations: List[Dict]):
    """Train domain reranker."""
    logger.info("=" * 60)
    logger.info("STEP 6: DOMAIN RERANKER")
    logger.info("=" * 60)
    
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score
    
    behavior_docs = defaultdict(list)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        text = ann.get("context", "")
        if beh_id and text and len(text) > 30:
            behavior_docs[beh_id].append(text)
    
    id_to_arabic = {v: k for k, v in ARABIC_TO_ID.items()}
    
    examples = []
    behavior_ids = list(behavior_docs.keys())
    
    for beh_id, docs in behavior_docs.items():
        arabic = id_to_arabic.get(beh_id, beh_id)
        query = f"ما هو {arabic}؟"
        
        for doc in docs[:15]:
            examples.append({"query": query, "doc": doc[:300], "label": 1})
            neg_beh = random.choice([b for b in behavior_ids if b != beh_id])
            neg_doc = random.choice(behavior_docs[neg_beh])
            examples.append({"query": query, "doc": neg_doc[:300], "label": 0})
    
    logger.info(f"Created {len(examples):,} reranker examples")
    
    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_data, val_data = examples[:split], examples[split:]
    
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=2, ignore_mismatched_sizes=True)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(item["query"], item["doc"], truncation=True, max_length=384, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(), "attention_mask": enc["attention_mask"].squeeze(), "labels": torch.tensor(item["label"], dtype=torch.long)}
    
    output_dir = MODELS_DIR / "qbm-reranker-v2"
    training_args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=16, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, report_to="none")
    
    def compute_metrics(pred):
        return {"accuracy": accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=-1))}
    
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset(train_data, tokenizer), eval_dataset=Dataset(val_data, tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info(f"Accuracy: {results.get('eval_accuracy', 0):.4f}")
    return output_dir, results


# =============================================================================
# STEP 7: BUILD BEHAVIOR GRAPH
# =============================================================================

def build_behavior_graph(annotations: List[Dict]):
    """Build behavior graph with proper nodes and edges."""
    logger.info("=" * 60)
    logger.info("STEP 7: BUILD BEHAVIOR GRAPH")
    logger.info("=" * 60)
    
    # Count co-occurrences
    verse_behaviors = defaultdict(set)
    for ann in annotations:
        beh_id = ann.get("behavior_id", "")
        verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
        if beh_id:
            verse_behaviors[verse].add(beh_id)
    
    # Build edges
    edges = defaultdict(int)
    for verse, behaviors in verse_behaviors.items():
        behavior_list = list(behaviors)
        for i, b1 in enumerate(behavior_list):
            for b2 in behavior_list[i+1:]:
                key = tuple(sorted([b1, b2]))
                edges[key] += 1
    
    # Get all nodes
    all_nodes = set()
    for b1, b2 in edges.keys():
        all_nodes.add(b1)
        all_nodes.add(b2)
    
    id_to_arabic = {v: k for k, v in ARABIC_TO_ID.items()}
    
    # Create graph data
    graph_data = {
        "nodes": [{"id": node, "arabic": id_to_arabic.get(node, node)} for node in sorted(all_nodes)],
        "edges": [{"source": e[0], "target": e[1], "weight": w} for e, w in sorted(edges.items(), key=lambda x: -x[1])[:500]]
    }
    
    output_file = DATA_DIR / "behavior_graph_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Graph: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    logger.info(f"Saved to {output_file}")
    
    return output_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("QBM COMPLETE REBUILD - BOUZIDANI FRAMEWORK")
    logger.info("=" * 70)
    
    total_start = time.time()
    results = {}
    
    # Step 1: Filter annotations
    annotations_file, behavior_counts = filter_annotations()
    annotations = load_filtered_annotations(annotations_file)
    results["annotations"] = {"total": len(annotations), "classes": len(behavior_counts)}
    
    # Step 2: Embeddings
    embeddings_dir = train_embeddings(annotations)
    results["embeddings"] = str(embeddings_dir)
    
    # Step 3: Classifier
    classifier_dir, classifier_results = train_classifier(annotations)
    results["classifier"] = classifier_results
    
    # Step 4: Relations
    relations_dir, relations_results = train_relation_extractor(annotations)
    results["relations"] = relations_results
    
    # Step 5: GNN
    gnn_dir = train_gnn(annotations, embeddings_dir)
    results["gnn"] = str(gnn_dir)
    
    # Step 6: Reranker
    reranker_dir, reranker_results = train_reranker(annotations)
    results["reranker"] = reranker_results
    
    # Step 7: Graph
    graph_file = build_behavior_graph(annotations)
    results["graph"] = str(graph_file)
    
    total_time = time.time() - total_start
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("REBUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    for key, val in results.items():
        logger.info(f"{key}: {val}")
    
    # Save results
    with open(MODELS_DIR / "rebuild_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
