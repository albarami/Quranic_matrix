"""
QBM Layer Training Pipeline
Trains all 6 ML layers (2-7) per ENTERPRISE_IMPLEMENTATION_PLAN.md

Layer 2: Arabic Embeddings - Fine-tune AraBERT on behavioral spans
Layer 3: Behavioral Classifier - Train 87-class classifier
Layer 4: Relation Extractor - Train 7-class relation classifier
Layer 5: GNN Graph Reasoner - Train GAT on behavioral graph
Layer 6: Domain Reranker - Fine-tune cross-encoder on QBM pairs
Layer 7: Hybrid RAG Integration - Connect all trained components

Usage:
    python src/ml/train_all_layers.py --layer 2  # Train specific layer
    python src/ml/train_all_layers.py --all      # Train all layers
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU availability
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Device: {DEVICE}, GPUs: {NUM_GPUS}")
except ImportError:
    DEVICE = "cpu"
    NUM_GPUS = 0


@dataclass
class TrainingResult:
    """Result of training a layer."""
    layer: int
    name: str
    success: bool
    metrics: Dict[str, float]
    model_path: str
    training_time_seconds: float
    error: Optional[str] = None


# =============================================================================
# LAYER 2: ARABIC EMBEDDINGS
# =============================================================================

def train_layer_2_embeddings() -> TrainingResult:
    """
    Layer 2: Fine-tune AraBERT on behavioral spans.
    Target: same-behavior similarity > 0.85
    """
    logger.info("=" * 60)
    logger.info("LAYER 2: Arabic Embeddings Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from src.ml.arabic_embeddings import (
            ArabicEmbeddingTrainer,
            create_training_pairs,
            create_triplet_examples,
            load_spans,
            test_embedding_quality,
        )
        
        # Load training data
        spans = load_spans()
        logger.info(f"Loaded {len(spans)} behavioral spans")
        
        if len(spans) < 100:
            # Generate synthetic spans from annotations
            annotations_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
            if annotations_path.exists():
                spans = []
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            ann = json.loads(line)
                            spans.append({
                                "text_ar": ann.get("text", ""),
                                "behavior": ann.get("behavior_ar", ""),
                                "behavior_label": ann.get("behavior_ar", ""),
                            })
                logger.info(f"Loaded {len(spans)} spans from annotations")
        
        # Create training examples
        contrastive_pairs = create_training_pairs(spans[:10000])
        triplet_examples = create_triplet_examples(spans[:10000])
        
        logger.info(f"Created {len(contrastive_pairs)} contrastive pairs")
        logger.info(f"Created {len(triplet_examples)} triplet examples")
        
        # Initialize trainer
        trainer = ArabicEmbeddingTrainer(base_model="arabert")
        
        # Train with triplet loss (better for behavioral similarity)
        if triplet_examples:
            trainer.train_triplet(
                train_examples=triplet_examples,
                epochs=5,
                batch_size=32 if NUM_GPUS > 0 else 8,
            )
        
        # Evaluate
        metrics = {}
        if trainer.model:
            quality = test_embedding_quality(trainer.model)
            metrics = quality
            logger.info(f"Embedding quality metrics: {quality}")
        
        # Save model
        model_path = MODELS_DIR / "qbm-arabic-embeddings"
        trainer.save(model_path)
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=2,
            name="Arabic Embeddings",
            success=True,
            metrics=metrics,
            model_path=str(model_path),
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 2 training failed: {e}")
        return TrainingResult(
            layer=2,
            name="Arabic Embeddings",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# LAYER 3: BEHAVIORAL CLASSIFIER
# =============================================================================

def train_layer_3_classifier() -> TrainingResult:
    """
    Layer 3: Train 87-class behavioral classifier.
    Target: Macro F1 > 0.80
    """
    logger.info("=" * 60)
    logger.info("LAYER 3: Behavioral Classifier Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from src.ml.behavioral_classifier import (
            BEHAVIOR_CLASSES,
            BEHAVIOR_TO_ID,
            BehaviorDataset,
        )
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        
        # Load training data from annotations
        annotations_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
        examples = []
        
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        behavior = ann.get("behavior_ar", "")
                        if behavior in BEHAVIOR_TO_ID:
                            examples.append({
                                "text": ann.get("context", ann.get("text", "")),
                                "context": "",
                                "label": BEHAVIOR_TO_ID[behavior],
                            })
        
        logger.info(f"Loaded {len(examples)} training examples")
        
        if len(examples) < 100:
            logger.warning("Not enough training data, using synthetic examples")
            # Create synthetic examples from behavior definitions
            from src.ml.arabic_embeddings import BEHAVIOR_DEFINITIONS
            for behavior, definition in BEHAVIOR_DEFINITIONS.items():
                if behavior in BEHAVIOR_TO_ID:
                    examples.append({
                        "text": definition,
                        "context": "",
                        "label": BEHAVIOR_TO_ID[behavior],
                    })
        
        # Split train/val
        np.random.shuffle(examples)
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
        
        # Load model and tokenizer
        model_name = "aubmindlab/bert-base-arabertv2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(BEHAVIOR_CLASSES),
        )
        
        if DEVICE == "cuda":
            model = model.cuda()
        
        # Create simple single-label dataset (not multi-label)
        class SingleLabelDataset(torch.utils.data.Dataset):
            def __init__(self, examples, tokenizer, max_length=256):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                ex = self.examples[idx]
                encoding = self.tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(ex["label"], dtype=torch.long),
                }
        
        train_dataset = SingleLabelDataset(train_examples, tokenizer)
        val_dataset = SingleLabelDataset(val_examples, tokenizer)
        
        # Training arguments
        output_dir = MODELS_DIR / "qbm-behavioral-classifier"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16 if NUM_GPUS > 0 else 4,
            per_device_eval_batch_size=32 if NUM_GPUS > 0 else 8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            # Ensure labels are 1D integers
            if len(labels.shape) > 1:
                labels = np.argmax(labels, axis=-1)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            acc = accuracy_score(labels, predictions)
            return {"f1": f1, "accuracy": acc}
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=3,
            name="Behavioral Classifier",
            success=True,
            metrics=eval_results,
            model_path=str(output_dir),
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 3 training failed: {e}")
        import traceback
        traceback.print_exc()
        return TrainingResult(
            layer=3,
            name="Behavioral Classifier",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# LAYER 4: RELATION EXTRACTOR
# =============================================================================

def train_layer_4_relations() -> TrainingResult:
    """
    Layer 4: Train 7-class relation extractor.
    Target: Accuracy > 0.80, CAUSES F1 > 0.75
    """
    logger.info("=" * 60)
    logger.info("LAYER 4: Relation Extractor Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from src.ml.relation_extractor import (
            RELATION_TYPES,
            RELATION_TO_ID,
            KNOWN_RELATIONS,
        )
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        
        # Create training examples from known relations
        examples = []
        
        # Load behavioral annotations to get context
        annotations_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
        behavior_texts = {}
        
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        behavior = ann.get("behavior_ar", "")
                        if behavior and behavior not in behavior_texts:
                            behavior_texts[behavior] = ann.get("text", "")
        
        # Create relation examples
        for rel in KNOWN_RELATIONS:
            e1 = rel["entity1"]
            e2 = rel["entity2"]
            relation = rel["relation"]
            
            # Get text context for entities
            text1 = behavior_texts.get(e1, e1)
            text2 = behavior_texts.get(e2, e2)
            
            # Create input: [CLS] entity1 [SEP] entity2 [SEP] context
            input_text = f"{e1} [SEP] {e2} [SEP] {text1[:100]} {text2[:100]}"
            
            examples.append({
                "text": input_text,
                "label": RELATION_TO_ID.get(relation, RELATION_TO_ID["NONE"]),
            })
        
        # Add negative examples (NONE relation)
        behaviors = list(behavior_texts.keys())
        for _ in range(len(examples)):
            import random
            b1, b2 = random.sample(behaviors, 2) if len(behaviors) >= 2 else (behaviors[0], behaviors[0])
            # Check if this pair has a known relation
            has_relation = any(
                (r["entity1"] == b1 and r["entity2"] == b2) or
                (r["entity1"] == b2 and r["entity2"] == b1)
                for r in KNOWN_RELATIONS
            )
            if not has_relation:
                examples.append({
                    "text": f"{b1} [SEP] {b2} [SEP] {behavior_texts.get(b1, '')[:50]} {behavior_texts.get(b2, '')[:50]}",
                    "label": RELATION_TO_ID["NONE"],
                })
        
        logger.info(f"Created {len(examples)} relation examples")
        
        # Split train/val
        np.random.shuffle(examples)
        split_idx = int(len(examples) * 0.8)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Load model
        model_name = "aubmindlab/bert-base-arabertv2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(RELATION_TYPES),
        )
        
        if DEVICE == "cuda":
            model = model.cuda()
        
        # Create simple dataset
        class RelationDataset(torch.utils.data.Dataset):
            def __init__(self, examples, tokenizer, max_length=256):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                ex = self.examples[idx]
                encoding = self.tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(ex["label"]),
                }
        
        train_dataset = RelationDataset(train_examples, tokenizer)
        val_dataset = RelationDataset(val_examples, tokenizer)
        
        # Training
        output_dir = MODELS_DIR / "qbm-relation-extractor"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=5,
            per_device_train_batch_size=16 if NUM_GPUS > 0 else 4,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1 = f1_score(labels, predictions, average='macro')
            acc = accuracy_score(labels, predictions)
            return {"f1": f1, "accuracy": acc}
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=4,
            name="Relation Extractor",
            success=True,
            metrics=eval_results,
            model_path=str(output_dir),
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 4 training failed: {e}")
        import traceback
        traceback.print_exc()
        return TrainingResult(
            layer=4,
            name="Relation Extractor",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# LAYER 5: GNN GRAPH REASONER
# =============================================================================

def train_layer_5_gnn() -> TrainingResult:
    """
    Layer 5: Train GAT on behavioral graph.
    Target: Link prediction AUC > 0.85
    """
    logger.info("=" * 60)
    logger.info("LAYER 5: GNN Graph Reasoner Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from src.ml.graph_reasoner import (
            QBMGraphReasoner,
            GraphBuilder,
            ReasoningEngine,
            PYG_AVAILABLE,
        )
        
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric not available")
        
        from torch_geometric.data import Data
        import torch.nn.functional as F
        
        # Build graph from annotations
        annotations_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
        
        graph_builder = GraphBuilder()
        verse_behaviors = {}  # verse -> list of behaviors
        
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        behavior = ann.get("behavior_ar", "")
                        verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
                        
                        if behavior:
                            graph_builder.add_node(behavior, "behavior")
                            
                            if verse not in verse_behaviors:
                                verse_behaviors[verse] = []
                            verse_behaviors[verse].append(behavior)
        
        # Add co-occurrence edges
        for verse, behaviors in verse_behaviors.items():
            for i, b1 in enumerate(behaviors):
                for b2 in behaviors[i+1:]:
                    graph_builder.add_edge(b1, b2, "co_occurs")
                    graph_builder.add_edge(b2, b1, "co_occurs")
        
        # Build PyG data
        graph_data = graph_builder.build()
        
        if graph_data is None:
            raise ValueError("Failed to build graph data")
        
        logger.info(f"Graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        
        # Initialize GNN model
        model = QBMGraphReasoner(
            num_node_features=graph_data.x.shape[1],
            hidden_dim=256,
            num_relations=6,
            num_heads=8,
        )
        
        if DEVICE == "cuda":
            model = model.cuda()
            graph_data = graph_data.to("cuda")
        
        # Training for link prediction
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create train/test edge split
        num_edges = graph_data.edge_index.shape[1]
        perm = torch.randperm(num_edges)
        train_edges = graph_data.edge_index[:, perm[:int(0.8 * num_edges)]]
        test_edges = graph_data.edge_index[:, perm[int(0.8 * num_edges):]]
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = model.forward(graph_data.x, train_edges)
            
            # Link prediction loss
            # Positive edges
            src, dst = train_edges
            pos_scores = (node_embeddings[src] * node_embeddings[dst]).sum(dim=1)
            
            # Negative edges (random)
            neg_dst = torch.randint(0, graph_data.x.shape[0], (src.shape[0],), device=src.device)
            neg_scores = (node_embeddings[src] * node_embeddings[neg_dst]).sum(dim=1)
            
            # Binary cross entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            node_embeddings = model.forward(graph_data.x, graph_data.edge_index)
            
            # Test edge scores
            src, dst = test_edges
            pos_scores = torch.sigmoid((node_embeddings[src] * node_embeddings[dst]).sum(dim=1))
            
            neg_dst = torch.randint(0, graph_data.x.shape[0], (src.shape[0],), device=src.device)
            neg_scores = torch.sigmoid((node_embeddings[src] * node_embeddings[neg_dst]).sum(dim=1))
            
            # AUC approximation
            auc = ((pos_scores > neg_scores).float().mean()).item()
        
        logger.info(f"Link prediction AUC: {auc:.4f}")
        
        # Save model
        output_dir = MODELS_DIR / "qbm-graph-reasoner"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=5,
            name="GNN Graph Reasoner",
            success=True,
            metrics={"auc": auc},
            model_path=str(output_dir),
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 5 training failed: {e}")
        import traceback
        traceback.print_exc()
        return TrainingResult(
            layer=5,
            name="GNN Graph Reasoner",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# LAYER 6: DOMAIN RERANKER
# =============================================================================

def train_layer_6_reranker() -> TrainingResult:
    """
    Layer 6: Fine-tune cross-encoder on QBM pairs.
    Target: NDCG@10 > 0.80, MRR > 0.75
    """
    logger.info("=" * 60)
    logger.info("LAYER 6: Domain Reranker Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        # Load training data
        annotations_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
        
        # Create query-document pairs
        train_data = []
        behavior_docs = {}  # behavior -> list of texts
        
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        behavior = ann.get("behavior_ar", "")
                        text = ann.get("context", ann.get("text", ""))
                        if behavior and text:
                            if behavior not in behavior_docs:
                                behavior_docs[behavior] = []
                            behavior_docs[behavior].append(text)
        
        # Create training pairs
        behaviors = list(behavior_docs.keys())
        for behavior, docs in behavior_docs.items():
            query = f"ما هو {behavior}؟"  # What is [behavior]?
            
            for doc in docs[:10]:  # Limit per behavior
                # Positive example
                train_data.append({"query": query, "doc": doc, "label": 1})
                
                # Negative example (random other behavior's doc)
                if len(behaviors) > 1:
                    other_behavior = np.random.choice([b for b in behaviors if b != behavior])
                    other_doc = np.random.choice(behavior_docs[other_behavior])
                    train_data.append({"query": query, "doc": other_doc, "label": 0})
        
        logger.info(f"Created {len(train_data)} training pairs")
        
        if len(train_data) < 10:
            raise ValueError("Not enough training data for reranker")
        
        # Split train/val
        np.random.shuffle(train_data)
        split_idx = int(len(train_data) * 0.9)
        train_examples = train_data[:split_idx]
        val_examples = train_data[split_idx:]
        
        # Load model and tokenizer (use Arabic BERT for cross-encoding)
        model_name = "aubmindlab/bert-base-arabertv2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary: relevant/not relevant
        )
        
        if DEVICE == "cuda":
            model = model.cuda()
        
        # Create dataset
        class RerankerDataset(torch.utils.data.Dataset):
            def __init__(self, examples, tokenizer, max_length=256):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                ex = self.examples[idx]
                # Cross-encoder format: [CLS] query [SEP] doc [SEP]
                encoding = self.tokenizer(
                    ex["query"],
                    ex["doc"],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(ex["label"]),
                }
        
        train_dataset = RerankerDataset(train_examples, tokenizer)
        val_dataset = RerankerDataset(val_examples, tokenizer)
        
        # Training
        output_dir = MODELS_DIR / "qbm-domain-reranker"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16 if NUM_GPUS > 0 else 4,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            return {"accuracy": acc}
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"Reranker eval results: {eval_results}")
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=6,
            name="Domain Reranker",
            success=True,
            metrics=eval_results,
            model_path=str(output_dir),
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 6 training failed: {e}")
        import traceback
        traceback.print_exc()
        return TrainingResult(
            layer=6,
            name="Domain Reranker",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# LAYER 7: HYBRID RAG INTEGRATION
# =============================================================================

def train_layer_7_integration() -> TrainingResult:
    """
    Layer 7: Connect all trained components.
    This doesn't train a new model, but integrates all layers.
    """
    logger.info("=" * 60)
    logger.info("LAYER 7: Hybrid RAG Integration")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Check all trained models exist
        models_to_check = [
            ("qbm-arabic-embeddings", "Layer 2"),
            ("qbm-behavioral-classifier", "Layer 3"),
            ("qbm-relation-extractor", "Layer 4"),
            ("qbm-graph-reasoner", "Layer 5"),
            ("qbm-domain-reranker", "Layer 6"),
        ]
        
        missing = []
        for model_name, layer_name in models_to_check:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                missing.append(f"{layer_name} ({model_name})")
        
        if missing:
            logger.warning(f"Missing models: {missing}")
        
        # Update hybrid_rag_system.py to use trained models
        from src.ml.hybrid_rag_system import HybridRAGSystem
        
        # Initialize with trained models
        system = HybridRAGSystem()
        
        # Test integration
        test_query = "ما علاقة الكبر بقسوة القلب؟"
        
        # This should use all trained components
        result = system.answer(test_query)
        
        success = len(result.get("answer", "")) > 100
        
        elapsed = time.time() - start_time
        
        return TrainingResult(
            layer=7,
            name="Hybrid RAG Integration",
            success=success,
            metrics={"integration_test": "passed" if success else "failed"},
            model_path="integrated",
            training_time_seconds=elapsed,
        )
        
    except Exception as e:
        logger.error(f"Layer 7 integration failed: {e}")
        import traceback
        traceback.print_exc()
        return TrainingResult(
            layer=7,
            name="Hybrid RAG Integration",
            success=False,
            metrics={},
            model_path="",
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_all_layers() -> List[TrainingResult]:
    """Train all layers sequentially."""
    results = []
    
    logger.info("=" * 60)
    logger.info("QBM FULL TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}, GPUs: {NUM_GPUS}")
    
    # Layer 2: Arabic Embeddings
    results.append(train_layer_2_embeddings())
    
    # Layer 3: Behavioral Classifier
    results.append(train_layer_3_classifier())
    
    # Layer 4: Relation Extractor
    results.append(train_layer_4_relations())
    
    # Layer 5: GNN Graph Reasoner
    results.append(train_layer_5_gnn())
    
    # Layer 6: Domain Reranker
    results.append(train_layer_6_reranker())
    
    # Layer 7: Integration
    results.append(train_layer_7_integration())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    total_time = sum(r.training_time_seconds for r in results)
    successful = sum(1 for r in results if r.success)
    
    for r in results:
        status = "✓" if r.success else "✗"
        logger.info(f"{status} Layer {r.layer}: {r.name}")
        logger.info(f"   Time: {r.training_time_seconds:.1f}s")
        logger.info(f"   Metrics: {r.metrics}")
        if r.error:
            logger.info(f"   Error: {r.error}")
    
    logger.info(f"\nTotal: {successful}/{len(results)} layers trained")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save results
    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump([{
            "layer": r.layer,
            "name": r.name,
            "success": r.success,
            "metrics": r.metrics,
            "model_path": r.model_path,
            "training_time_seconds": r.training_time_seconds,
            "error": r.error,
        } for r in results], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


def train_single_layer(layer: int) -> TrainingResult:
    """Train a single layer."""
    layer_functions = {
        2: train_layer_2_embeddings,
        3: train_layer_3_classifier,
        4: train_layer_4_relations,
        5: train_layer_5_gnn,
        6: train_layer_6_reranker,
        7: train_layer_7_integration,
    }
    
    if layer not in layer_functions:
        raise ValueError(f"Invalid layer: {layer}. Must be 2-7.")
    
    return layer_functions[layer]()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QBM ML layers")
    parser.add_argument("--layer", type=int, help="Train specific layer (2-7)")
    parser.add_argument("--all", action="store_true", help="Train all layers")
    
    args = parser.parse_args()
    
    if args.all:
        results = train_all_layers()
    elif args.layer:
        result = train_single_layer(args.layer)
        print(f"\nLayer {result.layer}: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Metrics: {result.metrics}")
    else:
        # Default: train all
        results = train_all_layers()
