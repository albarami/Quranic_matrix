"""
QBM Clean Layer Training Pipeline
Production-quality training with proper error handling and no warnings.

Layers:
2. Arabic Embeddings - Fine-tune on behavioral spans
3. Behavioral Classifier - 87-class classification
4. Relation Extractor - 7-class relation classification
5. GNN Graph Reasoner - Link prediction on behavioral graph
6. Domain Reranker - Cross-encoder for relevance scoring
7. Integration - Connect all components
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_annotations.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DEVICE SETUP
# =============================================================================

def setup_device() -> Tuple[str, int]:
    """Setup compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name} ({num_gpus} available)")
        else:
            device = "cpu"
            num_gpus = 0
            logger.info("Using CPU")
        return device, num_gpus
    except ImportError:
        logger.warning("PyTorch not available")
        return "cpu", 0

DEVICE, NUM_GPUS = setup_device()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_annotations() -> List[Dict[str, Any]]:
    """Load behavioral annotations from JSONL file."""
    annotations = []
    
    if not ANNOTATIONS_FILE.exists():
        logger.error(f"Annotations file not found: {ANNOTATIONS_FILE}")
        return annotations
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    ann = json.loads(line)
                    annotations.append(ann)
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(annotations)} annotations")
    return annotations


# =============================================================================
# TRAINING RESULT
# =============================================================================

@dataclass
class TrainingResult:
    """Result of training a layer."""
    layer: int
    name: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: str = ""
    time_seconds: float = 0.0
    error: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} Layer {self.layer}: {self.name} ({self.time_seconds:.1f}s)"


# =============================================================================
# LAYER 2: ARABIC EMBEDDINGS
# =============================================================================

def train_layer_2() -> TrainingResult:
    """Train Arabic embeddings using contrastive learning."""
    logger.info("=" * 50)
    logger.info("LAYER 2: Arabic Embeddings")
    logger.info("=" * 50)
    
    start = time.time()
    output_dir = MODELS_DIR / "qbm-arabic-embeddings"
    
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        import torch
        
        # Load annotations
        annotations = load_annotations()
        if len(annotations) < 100:
            return TrainingResult(2, "Arabic Embeddings", False, 
                                  error="Not enough training data")
        
        # Group by behavior
        behavior_texts = {}
        for ann in annotations:
            behavior = ann.get("behavior_ar", "")
            text = ann.get("context", "")
            if behavior and text and len(text) > 10:
                if behavior not in behavior_texts:
                    behavior_texts[behavior] = []
                if len(behavior_texts[behavior]) < 500:  # Limit per behavior
                    behavior_texts[behavior].append(text)
        
        logger.info(f"Found {len(behavior_texts)} unique behaviors")
        
        # Create triplet examples
        import random
        examples = []
        behaviors = list(behavior_texts.keys())
        
        for behavior, texts in behavior_texts.items():
            if len(texts) < 2:
                continue
            
            other_behaviors = [b for b in behaviors if b != behavior]
            if not other_behaviors:
                continue
            
            for i in range(min(len(texts) - 1, 100)):
                anchor = texts[i]
                positive = texts[(i + 1) % len(texts)]
                neg_behavior = random.choice(other_behaviors)
                negative = random.choice(behavior_texts[neg_behavior])
                
                examples.append(InputExample(texts=[anchor, positive, negative]))
        
        logger.info(f"Created {len(examples)} triplet examples")
        
        if len(examples) < 50:
            return TrainingResult(2, "Arabic Embeddings", False,
                                  error="Not enough triplet examples")
        
        # Load base model
        model = SentenceTransformer("aubmindlab/bert-base-arabertv2", device=DEVICE)
        
        # Train
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            output_path=str(output_dir),
            show_progress_bar=True,
        )
        
        # Evaluate
        test_pairs = [
            ("الكبر", "التكبر", 0.8),  # Should be similar
            ("الكبر", "أكبر", 0.3),    # Should be different
        ]
        
        correct = 0
        for t1, t2, expected in test_pairs:
            emb1 = model.encode(t1)
            emb2 = model.encode(t2)
            sim = float(emb1 @ emb2 / (sum(emb1**2)**0.5 * sum(emb2**2)**0.5))
            if (expected > 0.5 and sim > 0.5) or (expected < 0.5 and sim < 0.5):
                correct += 1
        
        accuracy = correct / len(test_pairs)
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=2,
            name="Arabic Embeddings",
            success=True,
            metrics={"accuracy": accuracy, "examples": len(examples)},
            model_path=str(output_dir),
            time_seconds=elapsed
        )
        
    except Exception as e:
        return TrainingResult(2, "Arabic Embeddings", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# LAYER 3: BEHAVIORAL CLASSIFIER
# =============================================================================

def train_layer_3() -> TrainingResult:
    """Train behavioral classifier."""
    logger.info("=" * 50)
    logger.info("LAYER 3: Behavioral Classifier")
    logger.info("=" * 50)
    
    start = time.time()
    output_dir = MODELS_DIR / "qbm-behavioral-classifier"
    
    try:
        import torch
        import numpy as np
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from sklearn.metrics import f1_score, accuracy_score
        
        # Load annotations
        annotations = load_annotations()
        
        # Get unique behaviors and create mapping
        behaviors = sorted(set(ann.get("behavior_ar", "") for ann in annotations if ann.get("behavior_ar")))
        behavior_to_id = {b: i for i, b in enumerate(behaviors)}
        
        logger.info(f"Found {len(behaviors)} behavior classes")
        
        # Create examples
        examples = []
        for ann in annotations:
            behavior = ann.get("behavior_ar", "")
            text = ann.get("context", "")
            if behavior in behavior_to_id and text and len(text) > 10:
                examples.append({
                    "text": text[:512],
                    "label": behavior_to_id[behavior]
                })
        
        logger.info(f"Created {len(examples)} examples")
        
        if len(examples) < 100:
            return TrainingResult(3, "Behavioral Classifier", False,
                                  error="Not enough training data")
        
        # Shuffle and split
        np.random.shuffle(examples)
        split = int(len(examples) * 0.9)
        train_data = examples[:split]
        val_data = examples[split:]
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        model = AutoModelForSequenceClassification.from_pretrained(
            "aubmindlab/bert-base-arabertv2",
            num_labels=len(behaviors),
            ignore_mismatched_sizes=True
        )
        
        # Dataset class
        class ClassifierDataset(torch.utils.data.Dataset):
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
        
        train_dataset = ClassifierDataset(train_data, tokenizer)
        val_dataset = ClassifierDataset(val_data, tokenizer)
        
        # Training args
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",  # Disable wandb etc
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
        
        # Save
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Save label mapping
        with open(output_dir / "label_map.json", 'w', encoding='utf-8') as f:
            json.dump({"behaviors": behaviors, "behavior_to_id": behavior_to_id}, f, ensure_ascii=False)
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=3,
            name="Behavioral Classifier",
            success=True,
            metrics={"f1": results.get("eval_f1", 0), "accuracy": results.get("eval_accuracy", 0)},
            model_path=str(output_dir),
            time_seconds=elapsed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(3, "Behavioral Classifier", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# LAYER 4: RELATION EXTRACTOR
# =============================================================================

def train_layer_4() -> TrainingResult:
    """Train relation extractor."""
    logger.info("=" * 50)
    logger.info("LAYER 4: Relation Extractor")
    logger.info("=" * 50)
    
    start = time.time()
    output_dir = MODELS_DIR / "qbm-relation-extractor"
    
    try:
        import torch
        import numpy as np
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from sklearn.metrics import f1_score, accuracy_score
        
        # Relation types
        RELATIONS = ["CAUSES", "PREVENTS", "OPPOSITE", "INTENSIFIES", "LEADS_TO", "NONE"]
        rel_to_id = {r: i for i, r in enumerate(RELATIONS)}
        
        # Known relations for training
        KNOWN = [
            ("الغفلة", "الكبر", "CAUSES"),
            ("الكبر", "الظلم", "CAUSES"),
            ("الكبر", "قسوة القلب", "CAUSES"),
            ("التقوى", "المعصية", "PREVENTS"),
            ("الذكر", "الغفلة", "PREVENTS"),
            ("الصدق", "الكذب", "OPPOSITE"),
            ("الإيمان", "الكفر", "OPPOSITE"),
            ("التواضع", "الكبر", "OPPOSITE"),
            ("الإصرار", "قسوة القلب", "INTENSIFIES"),
            ("الكفر", "الظلم", "LEADS_TO"),
        ]
        
        # Load annotations for context
        annotations = load_annotations()
        behavior_context = {}
        for ann in annotations:
            b = ann.get("behavior_ar", "")
            ctx = ann.get("context", "")
            if b and ctx and b not in behavior_context:
                behavior_context[b] = ctx[:200]
        
        # Create training examples
        examples = []
        
        # Positive examples from known relations
        for e1, e2, rel in KNOWN:
            ctx1 = behavior_context.get(e1, e1)
            ctx2 = behavior_context.get(e2, e2)
            text = f"{e1} [SEP] {e2} [SEP] {ctx1} {ctx2}"
            examples.append({"text": text, "label": rel_to_id[rel]})
        
        # Negative examples (NONE relation)
        import random
        behaviors = list(behavior_context.keys())
        for _ in range(len(KNOWN) * 2):
            b1, b2 = random.sample(behaviors, 2)
            # Check not in known
            if not any((b1 == k[0] and b2 == k[1]) or (b1 == k[1] and b2 == k[0]) for k in KNOWN):
                text = f"{b1} [SEP] {b2} [SEP] {behavior_context[b1]} {behavior_context[b2]}"
                examples.append({"text": text, "label": rel_to_id["NONE"]})
        
        logger.info(f"Created {len(examples)} relation examples")
        
        # Shuffle and split
        np.random.shuffle(examples)
        split = int(len(examples) * 0.8)
        train_data = examples[:split]
        val_data = examples[split:]
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        model = AutoModelForSequenceClassification.from_pretrained(
            "aubmindlab/bert-base-arabertv2",
            num_labels=len(RELATIONS),
            ignore_mismatched_sizes=True
        )
        
        # Dataset
        class RelationDataset(torch.utils.data.Dataset):
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
        
        train_dataset = RelationDataset(train_data, tokenizer)
        val_dataset = RelationDataset(val_data, tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
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
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=4,
            name="Relation Extractor",
            success=True,
            metrics={"f1": results.get("eval_f1", 0), "accuracy": results.get("eval_accuracy", 0)},
            model_path=str(output_dir),
            time_seconds=elapsed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(4, "Relation Extractor", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# LAYER 5: GNN GRAPH REASONER
# =============================================================================

def train_layer_5() -> TrainingResult:
    """Train GNN for link prediction."""
    logger.info("=" * 50)
    logger.info("LAYER 5: GNN Graph Reasoner")
    logger.info("=" * 50)
    
    start = time.time()
    output_dir = MODELS_DIR / "qbm-graph-reasoner"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Check for torch_geometric
        try:
            from torch_geometric.nn import GATConv
            from torch_geometric.data import Data
        except ImportError:
            return TrainingResult(5, "GNN Graph Reasoner", False,
                                  error="torch_geometric not installed")
        
        # Load annotations and build graph
        annotations = load_annotations()
        
        # Build co-occurrence graph
        verse_behaviors = {}
        for ann in annotations:
            behavior = ann.get("behavior_ar", "")
            verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
            if behavior:
                if verse not in verse_behaviors:
                    verse_behaviors[verse] = set()
                verse_behaviors[verse].add(behavior)
        
        # Get unique behaviors
        all_behaviors = set()
        for behaviors in verse_behaviors.values():
            all_behaviors.update(behaviors)
        
        behaviors = sorted(all_behaviors)
        node_to_idx = {b: i for i, b in enumerate(behaviors)}
        
        logger.info(f"Graph: {len(behaviors)} nodes")
        
        # Build edges (co-occurrence)
        edges = []
        for verse, verse_behaviors_set in verse_behaviors.items():
            behavior_list = list(verse_behaviors_set)
            for i, b1 in enumerate(behavior_list):
                for b2 in behavior_list[i+1:]:
                    idx1, idx2 = node_to_idx[b1], node_to_idx[b2]
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])  # Bidirectional
        
        logger.info(f"Graph: {len(edges)} edges")
        
        if len(edges) < 10:
            return TrainingResult(5, "GNN Graph Reasoner", False,
                                  error="Not enough edges in graph")
        
        # Create PyG data
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.randn(len(behaviors), 128)  # Random initial features
        
        data = Data(x=x, edge_index=edge_index)
        
        # Simple GAT model
        class SimpleGAT(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
                self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)
            
            def forward(self, x, edge_index):
                x = F.elu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x
        
        model = SimpleGAT(128, 64, 64)
        if DEVICE == "cuda":
            model = model.cuda()
            data = data.to("cuda")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for link prediction
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            
            # Positive edges
            src, dst = data.edge_index
            pos_score = (out[src] * out[dst]).sum(dim=1)
            
            # Negative edges
            neg_dst = torch.randint(0, len(behaviors), (src.size(0),), device=src.device)
            neg_score = (out[src] * out[neg_dst]).sum(dim=1)
            
            # Loss
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss = {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            src, dst = data.edge_index
            pos_score = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
            neg_dst = torch.randint(0, len(behaviors), (src.size(0),), device=src.device)
            neg_score = torch.sigmoid((out[src] * out[neg_dst]).sum(dim=1))
            auc = (pos_score > neg_score).float().mean().item()
        
        logger.info(f"Link prediction AUC: {auc:.4f}")
        
        # Save
        torch.save({
            "model_state": model.state_dict(),
            "node_to_idx": node_to_idx,
            "behaviors": behaviors,
        }, output_dir / "model.pt")
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=5,
            name="GNN Graph Reasoner",
            success=True,
            metrics={"auc": auc, "nodes": len(behaviors), "edges": len(edges)},
            model_path=str(output_dir),
            time_seconds=elapsed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(5, "GNN Graph Reasoner", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# LAYER 6: DOMAIN RERANKER
# =============================================================================

def train_layer_6() -> TrainingResult:
    """Train domain-specific reranker."""
    logger.info("=" * 50)
    logger.info("LAYER 6: Domain Reranker")
    logger.info("=" * 50)
    
    start = time.time()
    output_dir = MODELS_DIR / "qbm-domain-reranker"
    
    try:
        import torch
        import numpy as np
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from sklearn.metrics import accuracy_score
        
        # Load annotations
        annotations = load_annotations()
        
        # Group by behavior
        behavior_docs = {}
        for ann in annotations:
            behavior = ann.get("behavior_ar", "")
            text = ann.get("context", "")
            if behavior and text and len(text) > 20:
                if behavior not in behavior_docs:
                    behavior_docs[behavior] = []
                if len(behavior_docs[behavior]) < 100:
                    behavior_docs[behavior].append(text)
        
        logger.info(f"Found {len(behavior_docs)} behaviors with documents")
        
        if len(behavior_docs) < 5:
            return TrainingResult(6, "Domain Reranker", False,
                                  error="Not enough behaviors for reranker training")
        
        # Create query-document pairs
        examples = []
        behaviors = list(behavior_docs.keys())
        
        for behavior, docs in behavior_docs.items():
            query = f"ما هو {behavior}؟"
            
            for doc in docs[:20]:
                # Positive
                examples.append({"query": query, "doc": doc[:256], "label": 1})
                
                # Negative
                other = np.random.choice([b for b in behaviors if b != behavior])
                other_doc = np.random.choice(behavior_docs[other])
                examples.append({"query": query, "doc": other_doc[:256], "label": 0})
        
        logger.info(f"Created {len(examples)} reranker examples")
        
        # Shuffle and split
        np.random.shuffle(examples)
        split = int(len(examples) * 0.9)
        train_data = examples[:split]
        val_data = examples[split:]
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        model = AutoModelForSequenceClassification.from_pretrained(
            "aubmindlab/bert-base-arabertv2",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Dataset
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
                    max_length=256,
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
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
        )
        
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=-1)
            return {"accuracy": accuracy_score(labels, preds)}
        
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
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=6,
            name="Domain Reranker",
            success=True,
            metrics={"accuracy": results.get("eval_accuracy", 0)},
            model_path=str(output_dir),
            time_seconds=elapsed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(6, "Domain Reranker", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# LAYER 7: INTEGRATION
# =============================================================================

def train_layer_7() -> TrainingResult:
    """Verify all components integrate correctly."""
    logger.info("=" * 50)
    logger.info("LAYER 7: Integration Test")
    logger.info("=" * 50)
    
    start = time.time()
    
    try:
        # Check all models exist
        required_models = [
            ("qbm-arabic-embeddings", "Layer 2"),
            ("qbm-behavioral-classifier", "Layer 3"),
            ("qbm-relation-extractor", "Layer 4"),
            ("qbm-graph-reasoner", "Layer 5"),
            ("qbm-domain-reranker", "Layer 6"),
        ]
        
        missing = []
        for model_name, layer in required_models:
            path = MODELS_DIR / model_name
            if not path.exists():
                missing.append(layer)
        
        if missing:
            return TrainingResult(7, "Integration", False,
                                  error=f"Missing models: {missing}")
        
        # Try loading each model
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        # Load embeddings
        embedder = SentenceTransformer(str(MODELS_DIR / "qbm-arabic-embeddings"))
        test_emb = embedder.encode("الكبر")
        logger.info(f"Embeddings: OK (dim={len(test_emb)})")
        
        # Load classifier
        clf_path = MODELS_DIR / "qbm-behavioral-classifier"
        clf_tokenizer = AutoTokenizer.from_pretrained(str(clf_path))
        clf_model = AutoModelForSequenceClassification.from_pretrained(str(clf_path))
        logger.info("Classifier: OK")
        
        # Load relation extractor
        rel_path = MODELS_DIR / "qbm-relation-extractor"
        rel_tokenizer = AutoTokenizer.from_pretrained(str(rel_path))
        rel_model = AutoModelForSequenceClassification.from_pretrained(str(rel_path))
        logger.info("Relation Extractor: OK")
        
        # Load GNN
        gnn_path = MODELS_DIR / "qbm-graph-reasoner" / "model.pt"
        gnn_data = torch.load(gnn_path, weights_only=False)
        logger.info(f"GNN: OK ({len(gnn_data['behaviors'])} behaviors)")
        
        # Load reranker
        rer_path = MODELS_DIR / "qbm-domain-reranker"
        rer_tokenizer = AutoTokenizer.from_pretrained(str(rer_path))
        rer_model = AutoModelForSequenceClassification.from_pretrained(str(rer_path))
        logger.info("Reranker: OK")
        
        elapsed = time.time() - start
        return TrainingResult(
            layer=7,
            name="Integration",
            success=True,
            metrics={"models_loaded": 5},
            model_path="all",
            time_seconds=elapsed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(7, "Integration", False,
                              time_seconds=time.time() - start, error=str(e))


# =============================================================================
# MAIN
# =============================================================================

def train_all() -> List[TrainingResult]:
    """Train all layers."""
    logger.info("=" * 60)
    logger.info("QBM CLEAN TRAINING PIPELINE")
    logger.info("=" * 60)
    
    results = []
    
    # Train each layer
    results.append(train_layer_2())
    results.append(train_layer_3())
    results.append(train_layer_4())
    results.append(train_layer_5())
    results.append(train_layer_6())
    results.append(train_layer_7())
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    total_time = sum(r.time_seconds for r in results)
    success_count = sum(1 for r in results if r.success)
    
    for r in results:
        logger.info(str(r))
        if r.metrics:
            logger.info(f"   Metrics: {r.metrics}")
        if r.error:
            logger.info(f"   Error: {r.error}")
    
    logger.info("")
    logger.info(f"Total: {success_count}/{len(results)} layers trained successfully")
    logger.info(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save results
    results_file = MODELS_DIR / "training_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([{
            "layer": r.layer,
            "name": r.name,
            "success": r.success,
            "metrics": r.metrics,
            "model_path": r.model_path,
            "time_seconds": r.time_seconds,
            "error": r.error
        } for r in results], f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == "__main__":
    results = train_all()
    
    # Exit with error if any layer failed
    if not all(r.success for r in results):
        sys.exit(1)
