"""
Layer 3: Behavioral Classifier

Replace keyword matching with ML-based classification.
This model UNDERSTANDS context, not just matches strings.

Current (BAD):  if "كبر" in text → behavior = "الكبر"  # FALSE POSITIVES
New (GOOD):     classifier.predict(text, context) → {"behavior": "الكبر", "confidence": 0.94}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML imports
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
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EvalPrediction,
    )
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Arabic base model
BASE_MODEL = "aubmindlab/bert-base-arabertv2"

# 87 Behavior classes (from QBM taxonomy)
def _load_behavior_classes() -> List[str]:
    """Load behavior labels from canonical_entities.json (SSOT)."""
    entities_path = Path(__file__).parent.parent.parent / "vocab" / "canonical_entities.json"
    try:
        if entities_path.exists():
            with open(entities_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            behaviors = [b.get("ar", "") for b in data.get("behaviors", []) if b.get("ar")]
            if behaviors:
                return behaviors
    except Exception as exc:
        logger.warning(f"Failed to load canonical behaviors: {exc}")
    return list(_FALLBACK_BEHAVIOR_CLASSES)

_FALLBACK_BEHAVIOR_CLASSES = [
    # Positive behaviors (praised)
    "الإيمان", "الصبر", "الشكر", "التوبة", "التقوى", "الإحسان", "الصدق", "الأمانة",
    "العدل", "الرحمة", "التواضع", "الخشوع", "الذكر", "الدعاء", "التوكل", "الرضا",
    "الحياء", "الزهد", "الورع", "الإخلاص", "اليقين", "الخوف", "الرجاء", "المحبة",
    "الإنفاق", "الجهاد", "الأمر_بالمعروف", "النهي_عن_المنكر", "صلة_الرحم", "بر_الوالدين",
    "الوفاء", "الحلم", "العفو", "الكرم", "الشجاعة", "الصمت", "التفكر", "التدبر",
    
    # Negative behaviors (blamed)
    "الكفر", "النفاق", "الكبر", "الحسد", "الغيبة", "الكذب", "الظلم", "الفسق",
    "الرياء", "الغضب", "البخل", "الغفلة", "الشرك", "الفجور", "الخيانة", "الجهل",
    "العجب", "الحقد", "السخرية", "اللعن", "السرقة", "الزنا", "القتل", "الربا",
    "الإسراف", "التبذير", "الجبن", "الكسل", "اليأس", "القنوط", "الجزع", "السفه",
    "الإعراض", "التكذيب", "الاستهزاء", "المكر", "الخداع", "الغش", "النميمة", "البهتان",
    
    # Heart states
    "قسوة_القلب", "مرض_القلب", "ختم_القلب", "طبع_القلب", "إنابة_القلب",
    
    # Neutral/contextual
    "السؤال", "الاستفهام", "الإخبار", "الوصف",
]

BEHAVIOR_CLASSES = _load_behavior_classes()

BEHAVIOR_TO_ID = {b: i for i, b in enumerate(BEHAVIOR_CLASSES)}
ID_TO_BEHAVIOR = {i: b for i, b in enumerate(BEHAVIOR_CLASSES)}


# =============================================================================
# DATASET
# =============================================================================

class BehaviorDataset(Dataset):
    """Dataset for behavioral classification."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine text with context using [SEP]
        text = example.get("text", "")
        context = example.get("context", "")
        combined = f"{text} [SEP] {context}" if context else text
        
        # Tokenize
        encoding = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Multi-label: create binary vector for all behaviors
        labels = torch.zeros(len(BEHAVIOR_CLASSES))
        for behavior in example.get("behaviors", []):
            if behavior in BEHAVIOR_TO_ID:
                labels[BEHAVIOR_TO_ID[behavior]] = 1.0
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


# =============================================================================
# CLASSIFIER
# =============================================================================

class BehavioralClassifier:
    """
    Multi-label behavioral classifier.
    
    Replaces keyword matching with ML-based understanding.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or (MODELS_DIR / "qbm-behavior-classifier")
        
        if TRANSFORMERS_AVAILABLE:
            self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load trained model or initialize from base."""
        if self.model_path.exists():
            logger.info(f"Loading trained classifier from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        else:
            logger.info(f"Initializing from base model: {BASE_MODEL}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(BEHAVIOR_CLASSES),
                problem_type="multi_label_classification",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        if TORCH_AVAILABLE:
            self.model.to(DEVICE)
    
    def predict(self, text: str, context: str = "", threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict behaviors in text.
        
        Returns:
            {
                "behaviors": ["الكبر", "النفاق"],
                "confidences": {"الكبر": 0.94, "النفاق": 0.87},
            }
        """
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        # Combine text with context
        combined = f"{text} [SEP] {context}" if context else text
        
        # Tokenize
        encoding = self.tokenizer(
            combined,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        if TORCH_AVAILABLE:
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # Extract predictions above threshold
        behaviors = []
        confidences = {}
        
        for i, prob in enumerate(probs):
            behavior = ID_TO_BEHAVIOR[i]
            if prob >= threshold:
                behaviors.append(behavior)
                confidences[behavior] = float(prob)
        
        # Sort by confidence
        behaviors.sort(key=lambda b: confidences[b], reverse=True)
        
        return {
            "text": text[:100],
            "behaviors": behaviors,
            "confidences": confidences,
            "threshold": threshold,
        }
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Predict behaviors for multiple texts."""
        return [self.predict(text, threshold=threshold) for text in texts]
    
    def train(self, 
              train_examples: List[Dict[str, Any]],
              eval_examples: List[Dict[str, Any]] = None,
              epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 2e-5):
        """
        Train the classifier on annotated examples.
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not initialized")
            return
        
        logger.info(f"Training on {len(train_examples)} examples")
        logger.info(f"Device: {DEVICE}")
        
        # Create datasets
        train_dataset = BehaviorDataset(train_examples, self.tokenizer)
        eval_dataset = BehaviorDataset(eval_examples, self.tokenizer) if eval_examples else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            fp16=TORCH_AVAILABLE and torch.cuda.is_available(),
            logging_steps=100,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save
        self.model.save_pretrained(str(self.model_path))
        self.tokenizer.save_pretrained(str(self.model_path))
        
        logger.info(f"Model saved to {self.model_path}")
    
    def _compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).numpy()
        labels = pred.label_ids
        
        # Micro F1
        tp = (predictions & labels).sum()
        fp = (predictions & ~labels).sum()
        fn = (~predictions & labels).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

def prepare_training_data(spans_file: Path = None) -> Tuple[List[Dict], List[Dict]]:
    """Prepare training data from annotated spans."""
    if spans_file is None:
        spans_file = DATA_DIR / "annotations" / "gold_spans.jsonl"
    
    examples = []
    
    possible_paths = [
        spans_file,
        DATA_DIR / "gold_spans.jsonl",
        DATA_DIR / "spans" / "gold_spans.jsonl",
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        span = json.loads(line)
                        behaviors = []
                        behavior = span.get("behavior_label") or span.get("behavior", "")
                        if behavior and behavior in BEHAVIOR_TO_ID:
                            behaviors.append(behavior)
                        
                        if behaviors:
                            examples.append({
                                "text": span.get("text_ar", ""),
                                "context": span.get("verse_text", ""),
                                "behaviors": behaviors,
                            })
            
            logger.info(f"Loaded {len(examples)} examples from {path}")
            break
    
    if not examples:
        logger.warning("No training examples found")
        return [], []
    
    # Split 90/10 train/eval
    import random
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    
    return examples[:split_idx], examples[split_idx:]


# =============================================================================
# TESTS
# =============================================================================

def test_no_false_positives(classifier: BehavioralClassifier) -> Dict[str, Any]:
    """Test that أكبر is NOT classified as الكبر."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: الله أكبر should NOT be arrogance
    result = classifier.predict("الله أكبر")
    test1_passed = "الكبر" not in result["behaviors"]
    results["tests"].append({
        "name": "الله أكبر ≠ الكبر",
        "passed": test1_passed,
        "predicted": result["behaviors"],
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: استكبر فرعون SHOULD be arrogance
    result = classifier.predict("استكبر فرعون على موسى")
    test2_passed = "الكبر" in result["behaviors"]
    results["tests"].append({
        "name": "استكبر فرعون = الكبر",
        "passed": test2_passed,
        "predicted": result["behaviors"],
    })
    results["passed" if test2_passed else "failed"] += 1
    
    # Test 3: كبر سنه (grew old) should NOT be arrogance
    result = classifier.predict("كبر سنه وضعف بصره")
    test3_passed = "الكبر" not in result["behaviors"]
    results["tests"].append({
        "name": "كبر سنه ≠ الكبر",
        "passed": test3_passed,
        "predicted": result["behaviors"],
    })
    results["passed" if test3_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def train_behavioral_classifier(epochs: int = 10, batch_size: int = 16) -> Dict[str, Any]:
    """Train the behavioral classifier."""
    logger.info("=" * 60)
    logger.info("TRAINING BEHAVIORAL CLASSIFIER")
    logger.info("=" * 60)
    
    train_examples, eval_examples = prepare_training_data()
    
    if not train_examples:
        return {"error": "No training data available"}
    
    classifier = BehavioralClassifier()
    classifier.train(train_examples, eval_examples, epochs, batch_size)
    test_results = test_no_false_positives(classifier)
    
    return {
        "status": "complete",
        "train_examples": len(train_examples),
        "model_path": str(classifier.model_path),
        "test_results": test_results,
    }


_classifier_instance = None

def get_behavioral_classifier() -> BehavioralClassifier:
    """Get the trained behavioral classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = BehavioralClassifier()
    return _classifier_instance


if __name__ == "__main__":
    results = train_behavioral_classifier(epochs=5, batch_size=16)
    print(json.dumps(results, indent=2, ensure_ascii=False))
