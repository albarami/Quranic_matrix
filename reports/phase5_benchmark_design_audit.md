# Phase 5: Benchmark Design Audit

**Date**: 2025-12-25  
**Purpose**: Document why term↔definition should not be mixed with term↔term in relatedness benchmarks

---

## Problem Statement

The original benchmark (`gold_relatedness_v2.jsonl`) caused all tested models to fail category-order invariant. After investigation, this was traced to **structural bias** in the benchmark design, not model inadequacy.

---

## Root Cause: Structural Bias

### Original Benchmark Structure

| Category | Format | Example |
|----------|--------|---------|
| Equivalent (1.0) | term ↔ **long definition** | الصبر ↔ الصبر على البلاء والثبات عند المصائب |
| Related (0.5) | term ↔ **term** | الصبر ↔ الجزع |
| Unrelated (0.0) | term ↔ **term** | الصبر ↔ الطهارة |

### Why This Causes Failure

Embedding models measure **structural similarity** as well as semantic similarity:
- Two single words → high structural similarity
- Single word vs long phrase → low structural similarity

Result: Models score `term ↔ term` (related) higher than `term ↔ definition` (equivalent) because of format matching, not semantic confusion.

---

## Evidence

### Before Fix (gold_relatedness_v2.jsonl)
```
Model: paraphrase-multilingual-MiniLM-L12-v2
  Avg equiv: 0.5522
  Avg related: 0.6264  ← HIGHER than equiv (wrong!)
  Avg unrelated: 0.5256
  Spearman: -0.0376 (inverted)
```

### After Fix (gold_relatedness_term_term.jsonl)
```
Model: paraphrase-multilingual-MiniLM-L12-v2
  Avg equiv: 0.6842  ← NOW HIGHEST (correct!)
  Avg related: 0.6722
  Avg unrelated: 0.6321
  Spearman: +0.0667 (positive)
```

---

## Solution: Separate Benchmarks by Task

### 1. Term-Term Relatedness (`gold_relatedness_term_term.jsonl`)
- **Purpose**: Measure semantic similarity between vocabulary terms
- **Format**: All pairs are term ↔ term (single words or short phrases)
- **Categories**:
  - Equivalent (1.0): synonyms, near-synonyms
  - Related (0.5): antonyms, complementary, same semantic field
  - Unrelated (0.0): different semantic field

### 2. Definition Alignment (`gold_definition_alignment.jsonl`)
- **Purpose**: Measure ability to match terms with definitions
- **Format**: term ↔ definition
- **Categories**:
  - Correct (1.0): term matches its definition
  - Wrong (0.0): term paired with wrong definition

### 3. Verse-Tafsir Alignment (planned: `gold_verse_tafsir_alignment.jsonl`)
- **Purpose**: Measure retrieval alignment for QBM's actual task
- **Format**: verse text ↔ tafsir explanation
- **Categories**:
  - Aligned (1.0): verse with its tafsir
  - Related (0.5): verse with tafsir of nearby concept
  - Unrelated (0.0): verse with unrelated tafsir

---

## Pairs Moved/Removed

### Moved from Relatedness to Definition Alignment

| ID | text_a | text_b | Reason |
|----|--------|--------|--------|
| rel_001 | الصبر | الصبر على البلاء | Definition, not synonym |
| rel_004 | الإيمان | التصديق بالله | Definition, not synonym |
| rel_009 | الشكر | الحمد والثناء على النعم | Definition, not synonym |
| rel_011 | التوبة | الرجوع إلى الله | Definition, not synonym |
| rel_013 | النفاق | إظهار الإيمان وإخفاء الكفر | Definition, not synonym |
| rel_016 | الحسد | تمني زوال النعمة عن الغير | Definition, not synonym |
| rel_018 | التقوى | الخوف من الله واجتناب المعاصي | Definition, not synonym |
| rel_020 | الظلم | وضع الشيء في غير موضعه | Definition, not synonym |
| rel_024 | الخشوع | خضوع القلب لله | Definition, not synonym |
| rel_026 | التوكل | الاعتماد على الله | Definition, not synonym |
| rel_028 | الذكر | ذكر الله باللسان والقلب | Definition, not synonym |
| rel_032 | الغضب | ثوران النفس لدفع المكروه | Definition, not synonym |
| rel_034 | الرياء | العمل لأجل الناس | Definition, not synonym |
| rel_036 | الصدق | مطابقة القول للواقع | Definition, not synonym |
| rel_038 | الأمانة | حفظ الحقوق وأداؤها | Definition, not synonym |
| rel_040 | قسوة القلب | جمود القلب وعدم تأثره | Definition, not synonym |
| rel_043 | الرجاء | الأمل في رحمة الله | Definition, not synonym |

### Replaced with Term-Term Equivalents

| Old | New | Reason |
|-----|-----|--------|
| الصبر ↔ الصبر على البلاء | الصبر ↔ التحمل | Synonym instead of definition |
| الإيمان ↔ التصديق بالله | الإيمان ↔ اليقين | Synonym instead of definition |
| النفاق ↔ إظهار الإيمان وإخفاء الكفر | النفاق ↔ المراءاة | Synonym instead of definition |
| الحسد ↔ تمني زوال النعمة | الحسد ↔ الغل | Near-synonym instead of definition |
| الغضب ↔ ثوران النفس | الغضب ↔ السخط | Synonym instead of definition |

---

## Validation

### Sanity Checks Passed
1. **Self-consistency**: Identical pairs (text_a == text_b) return similarity 1.0 ✅
2. **Hand-audit**: Evaluator reads files correctly, computes similarities properly ✅

### Model Results on New Benchmark
All 5 models pass category-order invariant:
- LaBSE: Spearman 0.39 ✅
- BGE-M3: Spearman 0.31
- mpnet-base-v2: Spearman 0.23
- E5-base: Spearman 0.18
- MiniLM-L12-v2: Spearman 0.07

---

## Lessons Learned

1. **Never mix structural formats** in a single benchmark category
2. **Test multiple models** before concluding "model is bad"
3. **Separate tasks** (term similarity, definition alignment, retrieval) into separate benchmarks
4. **Use sanity checks** to validate evaluator correctness before model comparison

---

## Files Created

| File | Purpose | Pairs |
|------|---------|-------|
| `gold_relatedness_term_term.jsonl` | Term-term similarity | 60 |
| `gold_definition_alignment.jsonl` | Definition matching | 30 |
| `gold_relatedness_v2.jsonl` | Deprecated (structural bias) | 58 |
| `gold_equivalence_v2.jsonl` | Binary equivalence | 65 |
