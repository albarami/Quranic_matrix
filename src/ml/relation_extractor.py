"""
Layer 4: Relation Extraction Model

Learn CAUSAL relationships between behaviors, not just co-occurrence.

Current (BAD):  if behavior_a in same_verse as behavior_b → "co_occurs"
New (GOOD):     model.predict(text, b1, b2) → {"relation": "CAUSES", "confidence": 0.87}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
BASE_MODEL = "aubmindlab/bert-base-arabertv2"

# Relationship types (causal, not just co-occurrence)
RELATION_TYPES = {
    "CAUSES": "يسبب",           # الغفلة → الكبر
    "RESULTS_IN": "ينتج عنه",    # الكبر → الظلم
    "PREVENTS": "يمنع",          # التقوى → المعصية
    "OPPOSITE_OF": "نقيض",       # الصدق ↔ الكذب
    "PREREQUISITE": "شرط لـ",    # الإيمان → قبول العمل
    "INTENSIFIES": "يزيد",       # الإصرار → القسوة
    "LEADS_TO": "يؤدي إلى",      # General causation
    "NONE": "لا علاقة",          # No relationship
}

RELATION_TO_ID = {r: i for i, r in enumerate(RELATION_TYPES.keys())}
ID_TO_RELATION = {i: r for i, r in enumerate(RELATION_TYPES.keys())}

# Known causal relationships for training
KNOWN_RELATIONS = [
    # CAUSES relationships
    {"entity1": "الغفلة", "entity2": "الكبر", "relation": "CAUSES"},
    {"entity1": "الكبر", "entity2": "الظلم", "relation": "CAUSES"},
    {"entity1": "الكبر", "entity2": "قسوة_القلب", "relation": "CAUSES"},
    {"entity1": "الذنوب", "entity2": "قسوة_القلب", "relation": "CAUSES"},
    {"entity1": "الإعراض", "entity2": "ختم_القلب", "relation": "CAUSES"},
    {"entity1": "النفاق", "entity2": "مرض_القلب", "relation": "CAUSES"},
    {"entity1": "الكفر", "entity2": "الظلم", "relation": "CAUSES"},
    {"entity1": "الحسد", "entity2": "البغضاء", "relation": "CAUSES"},
    {"entity1": "الكذب", "entity2": "النفاق", "relation": "CAUSES"},
    {"entity1": "الرياء", "entity2": "حبط_العمل", "relation": "CAUSES"},
    
    # PREVENTS relationships
    {"entity1": "التقوى", "entity2": "المعصية", "relation": "PREVENTS"},
    {"entity1": "الذكر", "entity2": "الغفلة", "relation": "PREVENTS"},
    {"entity1": "الصبر", "entity2": "الجزع", "relation": "PREVENTS"},
    {"entity1": "الإيمان", "entity2": "الكفر", "relation": "PREVENTS"},
    {"entity1": "التوبة", "entity2": "العقوبة", "relation": "PREVENTS"},
    {"entity1": "الشكر", "entity2": "زوال_النعمة", "relation": "PREVENTS"},
    
    # OPPOSITE_OF relationships
    {"entity1": "الصدق", "entity2": "الكذب", "relation": "OPPOSITE_OF"},
    {"entity1": "الإيمان", "entity2": "الكفر", "relation": "OPPOSITE_OF"},
    {"entity1": "التواضع", "entity2": "الكبر", "relation": "OPPOSITE_OF"},
    {"entity1": "الصبر", "entity2": "الجزع", "relation": "OPPOSITE_OF"},
    {"entity1": "الشكر", "entity2": "الكفران", "relation": "OPPOSITE_OF"},
    {"entity1": "العدل", "entity2": "الظلم", "relation": "OPPOSITE_OF"},
    {"entity1": "الأمانة", "entity2": "الخيانة", "relation": "OPPOSITE_OF"},
    
    # INTENSIFIES relationships
    {"entity1": "الإصرار", "entity2": "قسوة_القلب", "relation": "INTENSIFIES"},
    {"entity1": "التكرار", "entity2": "الذنب", "relation": "INTENSIFIES"},
    {"entity1": "الاستمرار", "entity2": "الغفلة", "relation": "INTENSIFIES"},
    
    # PREREQUISITE relationships
    {"entity1": "الإيمان", "entity2": "قبول_العمل", "relation": "PREREQUISITE"},
    {"entity1": "الإخلاص", "entity2": "قبول_العبادة", "relation": "PREREQUISITE"},
    {"entity1": "التوبة", "entity2": "المغفرة", "relation": "PREREQUISITE"},
    
    # RESULTS_IN relationships
    {"entity1": "الصبر", "entity2": "الفوز", "relation": "RESULTS_IN"},
    {"entity1": "التقوى", "entity2": "الجنة", "relation": "RESULTS_IN"},
    {"entity1": "الكفر", "entity2": "النار", "relation": "RESULTS_IN"},
    {"entity1": "الظلم", "entity2": "الهلاك", "relation": "RESULTS_IN"},
]


# =============================================================================
# DATASET
# =============================================================================

class RelationDataset(Dataset):
    """Dataset for relation extraction."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Format: [CLS] entity1 [SEP] entity2 [SEP] context [SEP]
        text = f"{ex['entity1']} [SEP] {ex['entity2']} [SEP] {ex.get('context', '')}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        label = RELATION_TO_ID.get(ex.get("relation", "NONE"), RELATION_TO_ID["NONE"])
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label),
        }


# =============================================================================
# RELATION EXTRACTOR
# =============================================================================

class RelationExtractor:
    """
    Extract causal relationships between behaviors.
    
    Learns patterns like:
    - الكبر CAUSES قسوة_القلب
    - التقوى PREVENTS المعصية
    - الصدق OPPOSITE_OF الكذب
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or (MODELS_DIR / "qbm-relation-extractor")
        
        if TRANSFORMERS_AVAILABLE:
            self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load or initialize model."""
        if self.model_path.exists():
            logger.info(f"Loading from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        else:
            logger.info(f"Initializing from {BASE_MODEL}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(RELATION_TYPES),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        if TORCH_AVAILABLE:
            self.model.to(DEVICE)
    
    def predict(self, entity1: str, entity2: str, context: str = "") -> Dict[str, Any]:
        """
        Predict relationship between two entities.
        
        Returns:
            {
                "entity1": "الكبر",
                "entity2": "قسوة_القلب", 
                "relation": "CAUSES",
                "relation_ar": "يسبب",
                "confidence": 0.87,
                "direction": "entity1 → entity2"
            }
        """
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        text = f"{entity1} [SEP] {entity2} [SEP] {context}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        
        if TORCH_AVAILABLE:
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        
        pred_id = int(probs.argmax())
        relation = ID_TO_RELATION[pred_id]
        confidence = float(probs[pred_id])
        
        # Determine direction
        direction = "entity1 → entity2"
        if relation == "OPPOSITE_OF":
            direction = "entity1 ↔ entity2"
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "relation": relation,
            "relation_ar": RELATION_TYPES[relation],
            "confidence": confidence,
            "direction": direction,
            "all_scores": {ID_TO_RELATION[i]: float(p) for i, p in enumerate(probs)},
        }
    
    def find_all_relations(self, behavior: str, candidates: List[str]) -> List[Dict[str, Any]]:
        """Find relationships between a behavior and all candidates."""
        results = []
        for candidate in candidates:
            if candidate != behavior:
                result = self.predict(behavior, candidate)
                if result.get("relation") != "NONE" and result.get("confidence", 0) > 0.5:
                    results.append(result)
        
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results
    
    def train(self, train_examples: List[Dict], eval_examples: List[Dict] = None,
              epochs: int = 10, batch_size: int = 16):
        """Train the relation extractor."""
        if not self.model or not self.tokenizer:
            logger.error("Model not initialized")
            return
        
        logger.info(f"Training on {len(train_examples)} examples")
        
        train_dataset = RelationDataset(train_examples, self.tokenizer)
        eval_dataset = RelationDataset(eval_examples, self.tokenizer) if eval_examples else None
        
        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            fp16=TORCH_AVAILABLE and torch.cuda.is_available(),
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        
        self.model.save_pretrained(str(self.model_path))
        self.tokenizer.save_pretrained(str(self.model_path))
        logger.info(f"Model saved to {self.model_path}")


# =============================================================================
# TRAINING DATA AUGMENTATION
# =============================================================================

def augment_training_data(base_relations: List[Dict]) -> List[Dict]:
    """Augment training data with variations."""
    augmented = []
    
    for rel in base_relations:
        # Original
        augmented.append(rel)
        
        # Add context variations
        contexts = [
            f"في القرآن الكريم {rel['entity1']} {RELATION_TYPES[rel['relation']]} {rel['entity2']}",
            f"العلاقة بين {rel['entity1']} و {rel['entity2']}",
            f"{rel['entity1']} و {rel['entity2']} في السلوك القرآني",
        ]
        
        for ctx in contexts:
            augmented.append({**rel, "context": ctx})
        
        # For OPPOSITE_OF, add reverse
        if rel["relation"] == "OPPOSITE_OF":
            augmented.append({
                "entity1": rel["entity2"],
                "entity2": rel["entity1"],
                "relation": "OPPOSITE_OF",
            })
        
        # Add NONE examples (negative sampling)
        import random
        other_entities = [r["entity1"] for r in base_relations if r["entity1"] != rel["entity1"]]
        if other_entities:
            augmented.append({
                "entity1": rel["entity1"],
                "entity2": random.choice(other_entities),
                "relation": "NONE",
            })
    
    return augmented


# =============================================================================
# TESTS
# =============================================================================

def test_relation_extraction(extractor: RelationExtractor) -> Dict[str, Any]:
    """Test relation extraction quality."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: الكبر CAUSES قسوة_القلب
    result = extractor.predict("الكبر", "قسوة_القلب")
    test1_passed = result.get("relation") == "CAUSES"
    results["tests"].append({
        "name": "الكبر → قسوة_القلب = CAUSES",
        "passed": test1_passed,
        "predicted": result.get("relation"),
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: الصدق OPPOSITE_OF الكذب
    result = extractor.predict("الصدق", "الكذب")
    test2_passed = result.get("relation") == "OPPOSITE_OF"
    results["tests"].append({
        "name": "الصدق ↔ الكذب = OPPOSITE_OF",
        "passed": test2_passed,
        "predicted": result.get("relation"),
    })
    results["passed" if test2_passed else "failed"] += 1
    
    # Test 3: التقوى PREVENTS المعصية
    result = extractor.predict("التقوى", "المعصية")
    test3_passed = result.get("relation") == "PREVENTS"
    results["tests"].append({
        "name": "التقوى → المعصية = PREVENTS",
        "passed": test3_passed,
        "predicted": result.get("relation"),
    })
    results["passed" if test3_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def train_relation_extractor(epochs: int = 10, batch_size: int = 16) -> Dict[str, Any]:
    """Train the relation extractor."""
    logger.info("=" * 60)
    logger.info("TRAINING RELATION EXTRACTOR")
    logger.info("=" * 60)
    
    # Augment training data
    train_data = augment_training_data(KNOWN_RELATIONS)
    
    # Split
    import random
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    train_examples = train_data[:split_idx]
    eval_examples = train_data[split_idx:]
    
    logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    
    extractor = RelationExtractor()
    extractor.train(train_examples, eval_examples, epochs, batch_size)
    
    test_results = test_relation_extraction(extractor)
    
    return {
        "status": "complete",
        "train_examples": len(train_examples),
        "model_path": str(extractor.model_path),
        "test_results": test_results,
    }


_extractor_instance = None

def get_relation_extractor() -> RelationExtractor:
    """Get the trained relation extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = RelationExtractor()
    return _extractor_instance


if __name__ == "__main__":
    results = train_relation_extractor(epochs=5, batch_size=16)
    print(json.dumps(results, indent=2, ensure_ascii=False))
