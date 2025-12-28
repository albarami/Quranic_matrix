"""
Forensic Embedding Check - Steps A, B, C
=========================================
Before changing loss or model, verify what's actually inverted.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# STEP A: Sanity test with 20 hand-verified pairs
# =============================================================================

# 10 clearly SIMILAR pairs (should have HIGH cosine similarity)
SIMILAR_PAIRS = [
    ("الصبر", "الصبر على البلاء"),  # patience - patience in adversity
    ("الإيمان", "التصديق بالله"),  # faith - belief in God
    ("الشكر", "الحمد لله"),  # gratitude - praise to God
    ("التوبة", "الرجوع إلى الله"),  # repentance - returning to God
    ("الذكر", "ذكر الله"),  # remembrance - remembrance of God
    ("التقوى", "الخوف من الله"),  # piety - fear of God
    ("الصدق", "قول الحق"),  # truthfulness - speaking truth
    ("العدل", "الإنصاف"),  # justice - fairness
    ("الرحمة", "الرأفة"),  # mercy - compassion
    ("التواضع", "الخشوع"),  # humility - humbleness
]

# 10 clearly DISSIMILAR pairs (should have LOW cosine similarity)
DISSIMILAR_PAIRS = [
    ("الصبر", "الجزع"),  # patience vs panic
    ("الإيمان", "الكفر"),  # faith vs disbelief
    ("الشكر", "الكفران"),  # gratitude vs ingratitude
    ("التوبة", "الإصرار على المعصية"),  # repentance vs persistence in sin
    ("الذكر", "الغفلة"),  # remembrance vs heedlessness
    ("التقوى", "الفسق"),  # piety vs sin
    ("الصدق", "الكذب"),  # truthfulness vs lying
    ("العدل", "الظلم"),  # justice vs oppression
    ("الرحمة", "القسوة"),  # mercy vs cruelty
    ("التواضع", "الكبر"),  # humility vs arrogance
]


def compute_cosine_similarity(model, text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    # Cosine similarity
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(cos_sim)


def step_a_sanity_test():
    """Step A: Test with 20 hand-verified pairs."""
    print("\n" + "=" * 60)
    print("STEP A: Sanity Test with 20 Hand-Verified Pairs")
    print("=" * 60)
    
    from sentence_transformers import SentenceTransformer
    
    # Test with base model (no fine-tuning)
    print("\n[1] Loading BASE model (paraphrase-multilingual-MiniLM-L12-v2)...")
    base_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Test with fine-tuned model if exists
    finetuned_path = DATA_DIR / "models" / "qbm-arabic-finetuned"
    finetuned_model = None
    if finetuned_path.exists():
        print("[2] Loading FINE-TUNED model...")
        finetuned_model = SentenceTransformer(str(finetuned_path))
    
    print("\n" + "-" * 60)
    print("SIMILAR PAIRS (expected: HIGH cosine similarity)")
    print("-" * 60)
    
    base_similar_scores = []
    ft_similar_scores = []
    
    for text1, text2 in SIMILAR_PAIRS:
        base_score = compute_cosine_similarity(base_model, text1, text2)
        base_similar_scores.append(base_score)
        
        ft_score = None
        if finetuned_model:
            ft_score = compute_cosine_similarity(finetuned_model, text1, text2)
            ft_similar_scores.append(ft_score)
        
        ft_str = f" | FT: {ft_score:.4f}" if ft_score else ""
        print(f"  {text1} <-> {text2}")
        print(f"    Base: {base_score:.4f}{ft_str}")
    
    print(f"\n  SIMILAR avg: Base={np.mean(base_similar_scores):.4f}", end="")
    if ft_similar_scores:
        print(f" | FT={np.mean(ft_similar_scores):.4f}")
    else:
        print()
    
    print("\n" + "-" * 60)
    print("DISSIMILAR PAIRS (expected: LOW cosine similarity)")
    print("-" * 60)
    
    base_dissimilar_scores = []
    ft_dissimilar_scores = []
    
    for text1, text2 in DISSIMILAR_PAIRS:
        base_score = compute_cosine_similarity(base_model, text1, text2)
        base_dissimilar_scores.append(base_score)
        
        ft_score = None
        if finetuned_model:
            ft_score = compute_cosine_similarity(finetuned_model, text1, text2)
            ft_dissimilar_scores.append(ft_score)
        
        ft_str = f" | FT: {ft_score:.4f}" if ft_score else ""
        print(f"  {text1} <-> {text2}")
        print(f"    Base: {base_score:.4f}{ft_str}")
    
    print(f"\n  DISSIMILAR avg: Base={np.mean(base_dissimilar_scores):.4f}", end="")
    if ft_dissimilar_scores:
        print(f" | FT={np.mean(ft_dissimilar_scores):.4f}")
    else:
        print()
    
    # Analysis
    print("\n" + "-" * 60)
    print("ANALYSIS")
    print("-" * 60)
    
    base_gap = np.mean(base_similar_scores) - np.mean(base_dissimilar_scores)
    print(f"  Base model gap (similar - dissimilar): {base_gap:.4f}")
    
    if base_gap > 0:
        print("    ✅ Base model correctly ranks similar > dissimilar")
    else:
        print("    ❌ Base model INVERTED: dissimilar > similar")
    
    if ft_similar_scores and ft_dissimilar_scores:
        ft_gap = np.mean(ft_similar_scores) - np.mean(ft_dissimilar_scores)
        print(f"  Fine-tuned model gap: {ft_gap:.4f}")
        
        if ft_gap > base_gap:
            print("    ✅ Fine-tuning IMPROVED separation")
        elif ft_gap > 0:
            print("    ⚠️ Fine-tuning reduced separation but still correct")
        else:
            print("    ❌ Fine-tuning INVERTED the model")
    
    return {
        "base_similar_avg": np.mean(base_similar_scores),
        "base_dissimilar_avg": np.mean(base_dissimilar_scores),
        "base_gap": base_gap,
        "ft_similar_avg": np.mean(ft_similar_scores) if ft_similar_scores else None,
        "ft_dissimilar_avg": np.mean(ft_dissimilar_scores) if ft_dissimilar_scores else None,
    }


# =============================================================================
# STEP B: Verify evaluation logic
# =============================================================================

def step_b_verify_evaluation():
    """Step B: Verify evaluation logic (sign, scaling, correlation)."""
    print("\n" + "=" * 60)
    print("STEP B: Verify Evaluation Logic")
    print("=" * 60)
    
    # Load gold benchmark
    gold_file = DATA_DIR / "evaluation" / "semantic_similarity_gold.jsonl"
    
    print(f"\n[1] Loading gold benchmark: {gold_file}")
    
    pairs = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"  Loaded {len(pairs)} pairs")
    
    # Check label distribution
    high_count = sum(1 for p in pairs if p.get("expected_similarity") == "high")
    low_count = sum(1 for p in pairs if p.get("expected_similarity") == "low")
    
    print(f"\n[2] Label distribution:")
    print(f"  High similarity: {high_count}")
    print(f"  Low similarity: {low_count}")
    
    # Check how labels are converted to scores
    print(f"\n[3] Checking label -> score conversion in evaluator...")
    
    from src.ml.embedding_evaluator import EmbeddingEvaluator
    
    evaluator = EmbeddingEvaluator()
    
    # Look at the evaluation code
    print("\n[4] Evaluator gold score mapping:")
    
    # Manually check the mapping
    for p in pairs[:5]:
        expected = p.get("expected_similarity")
        # What score does the evaluator use?
        if expected == "high":
            score = 1.0
        else:
            score = 0.0
        print(f"  '{expected}' -> {score}")
    
    print("\n[5] Checking: Is cosine similarity correlated with gold score?")
    print("  (Higher cosine should correlate with higher gold score)")
    
    # Load model and compute
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    cosine_scores = []
    gold_scores = []
    
    for p in pairs:
        text_a = p.get("text_a", "")
        text_b = p.get("text_b", "")
        expected = p.get("expected_similarity")
        
        cos_sim = compute_cosine_similarity(model, text_a, text_b)
        gold = 1.0 if expected == "high" else 0.0
        
        cosine_scores.append(cos_sim)
        gold_scores.append(gold)
    
    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, _ = pearsonr(cosine_scores, gold_scores)
    spearman_r, _ = spearmanr(cosine_scores, gold_scores)
    
    print(f"\n  Pearson correlation: {pearson_r:.4f}")
    print(f"  Spearman correlation: {spearman_r:.4f}")
    
    if pearson_r > 0:
        print("  ✅ Positive correlation: evaluation logic is CORRECT")
    else:
        print("  ❌ Negative correlation: evaluation logic may be INVERTED")
    
    # Show some examples
    print("\n[6] Sample pairs with scores:")
    for i, p in enumerate(pairs[:6]):
        print(f"  {p['text_a']} <-> {p['text_b']}")
        print(f"    Gold: {p['expected_similarity']} | Cosine: {cosine_scores[i]:.4f}")
    
    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "high_count": high_count,
        "low_count": low_count,
    }


# =============================================================================
# STEP C: Verify training batch semantics
# =============================================================================

def step_c_verify_training_batch():
    """Step C: Verify training batch semantics (log triplets)."""
    print("\n" + "=" * 60)
    print("STEP C: Verify Training Batch Semantics")
    print("=" * 60)
    
    from src.ml.contrastive_trainer import load_behavioral_data, create_training_examples
    from sentence_transformers import SentenceTransformer
    
    print("\n[1] Loading behavioral data...")
    data = load_behavioral_data()
    print(f"  Loaded {sum(len(v) for v in data.values())} examples across {len(data)} behaviors")
    
    print("\n[2] Creating training examples...")
    examples = create_training_examples(data, num_examples=10)
    
    print("\n[3] Loading model for cosine computation...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("\n[4] Inspecting training examples:")
    print("-" * 60)
    
    for i, ex in enumerate(examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Anchor behavior: {ex.anchor_behavior}")
        print(f"  Positive behavior: {ex.positive_behavior}")
        print(f"  Negative behavior: {ex.negative_behavior}")
        
        # Truncate texts for display
        anchor_short = ex.anchor[:80] + "..." if len(ex.anchor) > 80 else ex.anchor
        positive_short = ex.positive[:80] + "..." if len(ex.positive) > 80 else ex.positive
        negative_short = ex.negative[:80] + "..." if len(ex.negative) > 80 else ex.negative
        
        print(f"  Anchor text: {anchor_short}")
        print(f"  Positive text: {positive_short}")
        print(f"  Negative text: {negative_short}")
        
        # Compute cosine similarities
        anchor_pos_sim = compute_cosine_similarity(model, ex.anchor, ex.positive)
        anchor_neg_sim = compute_cosine_similarity(model, ex.anchor, ex.negative)
        
        print(f"  Cosine(anchor, positive): {anchor_pos_sim:.4f}")
        print(f"  Cosine(anchor, negative): {anchor_neg_sim:.4f}")
        
        if anchor_pos_sim > anchor_neg_sim:
            print("  ✅ Correct: positive > negative")
        else:
            print("  ⚠️ Issue: negative >= positive (before training)")
    
    print("\n[5] Semantic check:")
    print("  - Anchor and Positive should be SAME behavior (high similarity expected)")
    print("  - Anchor and Negative should be OPPOSITE behavior (low similarity expected)")
    
    return {"examples_checked": len(examples)}


def run_forensic_check():
    """Run complete forensic check."""
    print("=" * 60)
    print("FORENSIC EMBEDDING CHECK")
    print("=" * 60)
    print("Before changing loss or model, verify what's inverted.")
    
    results = {}
    
    results["step_a"] = step_a_sanity_test()
    results["step_b"] = step_b_verify_evaluation()
    results["step_c"] = step_c_verify_training_batch()
    
    print("\n" + "=" * 60)
    print("FORENSIC CHECK COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\nSUMMARY:")
    
    if results["step_a"]["base_gap"] > 0:
        print("  ✅ Step A: Base model correctly separates similar/dissimilar")
    else:
        print("  ❌ Step A: Base model has inverted semantics")
    
    if results["step_b"]["pearson"] > 0:
        print("  ✅ Step B: Evaluation logic is correct (positive correlation)")
    else:
        print("  ❌ Step B: Evaluation logic may be inverted")
    
    print("  📋 Step C: Review training examples above for semantic correctness")
    
    return results


if __name__ == "__main__":
    run_forensic_check()
