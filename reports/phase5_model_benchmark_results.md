# Phase 5: Embedding Model Benchmark Results

**Date**: 2025-12-25  
**Evaluator**: embedding_evaluator_v3.py  
**Benchmark Files**: gold_relatedness_v2.jsonl (58 pairs), gold_equivalence_v2.jsonl (65 pairs)

---

## Executive Summary

**All three models fail the category-order invariant.** This indicates either:
1. The benchmark labels are misaligned with what embedding space measures, OR
2. All tested models are fundamentally unsuitable for Arabic Islamic vocabulary

Given that three different architectures (multilingual, AraBERT, CAMeL Classical Arabic) all fail in the same direction, **the benchmark design requires audit**.

---

## Consolidated Results Table

### Relatedness Benchmark (58 pairs)

| Model | Category Order | Avg Equiv | Avg Related | Avg Unrelated | WR E>R | WR R>U | Spearman |
|-------|---------------|-----------|-------------|---------------|--------|--------|----------|
| paraphrase-multilingual-MiniLM-L12-v2 | ❌ FAIL | 0.5522 | 0.6264 | 0.5256 | 39.8% | 65.8% | -0.0376 |
| aubmindlab/bert-base-arabertv2 | ❌ FAIL | 0.5661 | 0.7236 | 0.7522 | 16.1% | 49.6% | -0.5722 |
| CAMeL-Lab/bert-base-arabic-camelbert-ca | ❌ FAIL | 0.8221 | 0.8612 | 0.8894 | 36.1% | 39.6% | -0.3719 |

**Threshold Requirements**:
- Category order: avg(equiv) > avg(related) > avg(unrelated) ✅
- Gap ≥ 0.10 between categories
- Win-rate ≥ 70%
- Spearman ≥ 0.35

### Equivalence Benchmark (65 pairs)

| Model | AUC | Avg Positive | Avg Negative | Separation |
|-------|-----|--------------|--------------|------------|
| paraphrase-multilingual-MiniLM-L12-v2 | 0.3610 | 0.5672 | 0.6714 | -0.1042 |
| aubmindlab/bert-base-arabertv2 | 0.1600 | 0.5748 | 0.7386 | -0.1638 |
| CAMeL-Lab/bert-base-arabic-camelbert-ca | 0.2790 | 0.8195 | 0.8743 | -0.0548 |

**Threshold Requirements**:
- AUC ≥ 0.70
- Separation ≥ 0.15

---

## Critical Observations

### 1. All Models Show Inverted Semantics
- **Related pairs score HIGHER than equivalent pairs** in all models
- **Unrelated pairs score HIGHER than related pairs** in AraBERT and CAMeL
- Negative Spearman correlations indicate rankings are inverted

### 2. AraBERT Performs Worst Despite Being Arabic-Specific
- Spearman: -0.5722 (strongly inverted)
- AUC: 0.1600 (much worse than random 0.5)
- This suggests the model was not trained for semantic similarity tasks

### 3. CAMeL Classical Arabic Has Highest Overall Similarity
- All categories cluster around 0.82-0.89
- Poor discrimination between categories
- May be due to the model treating all Islamic vocabulary as highly related

---

## Top 10 Worst Inversions (Equivalent pairs scoring below unrelated)

From multilingual model (representative):

| ID | Category | Gold | Predicted | text_a | text_b |
|----|----------|------|-----------|--------|--------|
| rel_013 | same_concept | 1.0 | 0.2727 | النفاق | إظهار الإيمان وإخفاء الكفر |
| rel_020 | same_concept | 1.0 | 0.3011 | الظلم | وضع الشيء في غير موضعه |
| rel_026 | same_concept | 1.0 | 0.3045 | التوكل | الاعتماد على الله |
| rel_032 | same_concept | 1.0 | 0.3306 | الغضب | ثوران النفس لدفع المكروه |
| rel_034 | same_concept | 1.0 | 0.3649 | الرياء | العمل لأجل الناس |
| rel_018 | same_concept | 1.0 | 0.3881 | التقوى | الخوف من الله واجتناب المعاصي |
| rel_011 | same_concept | 1.0 | 0.4093 | التوبة | الرجوع إلى الله |
| rel_038 | same_concept | 1.0 | 0.4221 | الأمانة | حفظ الحقوق وأداؤها |
| rel_016 | same_concept | 1.0 | 0.4337 | الحسد | تمني زوال النعمة عن الغير |
| rel_024 | same_concept | 1.0 | 0.4593 | الخشوع | خضوع القلب لله |

**Pattern**: Long definitional phrases score LOW, while short term pairs (including opposites) score HIGH.

---

## Top 10 Worst Inversions (Opposite/Related pairs scoring above equivalent)

| ID | Category | Gold | Predicted | text_a | text_b |
|----|----------|------|-----------|--------|--------|
| rel_041 | opposite | 0.5 | 0.9646 | قسوة القلب | لين القلب |
| rel_047 | related | 0.5 | 0.9523 | الحسد | الغيرة |
| rel_025 | opposite | 0.5 | 0.8704 | الخشوع | الغفلة |
| rel_031 | opposite | 0.5 | 0.8417 | البخل | الكرم |
| rel_019 | opposite | 0.5 | 0.8378 | التقوى | الفسق |
| rel_023 | opposite | 0.5 | 0.7931 | الرحمة | القسوة |
| rel_021 | opposite | 0.5 | 0.7824 | الظلم | العدل |
| rel_037 | opposite | 0.5 | 0.7756 | الصدق | الكذب |
| rel_005 | opposite | 0.5 | 0.7693 | الإيمان | الكفر |
| rel_010 | opposite | 0.5 | 0.7647 | الشكر | الكفران |

**Pattern**: Single-word antonym pairs score VERY HIGH (0.76-0.96), because they share morphological roots and semantic field.

---

## Root Cause Analysis

### Hypothesis 1: Benchmark Mislabeling (LIKELY)
The benchmark treats:
- **Equivalent**: term + long definition → but models see these as "different length = different"
- **Related/Opposite**: term + term → models see these as "same structure = similar"

This is a **length bias** and **structural bias** in the benchmark design.

### Hypothesis 2: Models Not Trained for Similarity (POSSIBLE)
AraBERT and CAMeL are MLM models, not trained for semantic similarity. Mean pooling may not produce meaningful similarity scores.

### Hypothesis 3: Arabic Morphology Confounds Similarity (POSSIBLE)
Arabic antonyms often share roots (e.g., كفر/كفران, صدق/كذب share patterns). Models may conflate morphological similarity with semantic similarity.

---

## Recommendations

1. **Audit gold labels** (see phase5_gold_audit.md)
2. **Restructure equivalent pairs** to use term↔term format, not term↔definition
3. **Add verse↔tafsir pairs** which are the actual retrieval target
4. **Consider fine-tuning** on contrastive pairs before evaluation
5. **Test sentence-transformers Arabic models** (e.g., `sentence-transformers/LaBSE`) which are trained for similarity

---

## Sanity Checks Passed

- ✅ **Sanity A**: Identical pairs (text_a == text_b) return similarity 1.0
- ✅ **Sanity B**: Evaluator reads files correctly, computes similarities properly

---

## Files Generated

- `data/evaluation/phase5_all_models_results.json` - Full results with details
- `data/evaluation/gold_relatedness_v2.jsonl` - Relatedness benchmark (58 pairs)
- `data/evaluation/gold_equivalence_v2.jsonl` - Equivalence benchmark (65 pairs)
- `src/ml/embedding_evaluator_v3.py` - Evaluator with caching, win-rates, NaN guards
