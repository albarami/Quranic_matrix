# Phase 5: Gold Benchmark Audit

**Date**: 2025-12-25  
**Trigger**: All three tested models (multilingual, AraBERT, CAMeL-CA) fail category-order invariant

---

## Summary

Since all models fail in the same direction (related > equivalent > unrelated is inverted), the benchmark design is likely misaligned with what embedding space measures. This audit identifies 20 problematic pairs and proposes corrections.

---

## Identified Issues

### Issue 1: Length/Structure Bias in Equivalent Pairs

**Problem**: "Equivalent" pairs use term + long definition, while "related/opposite" pairs use term + term. Models interpret structural similarity (same length, same format) as semantic similarity.

**Evidence**:
- `النفاق` ↔ `إظهار الإيمان وإخفاء الكفر` (1 word vs 4 words) → predicted 0.27
- `قسوة القلب` ↔ `لين القلب` (2 words vs 2 words) → predicted 0.96

**Proposed Fix**: Restructure equivalent pairs to use term↔term or short phrase↔short phrase format.

---

### Issue 2: Morphological Root Sharing in Opposites

**Problem**: Arabic antonyms often share morphological patterns or roots, causing high embedding similarity.

**Examples**:
| Pair | Predicted | Issue |
|------|-----------|-------|
| الشكر ↔ الكفران | 0.76 | Both from ك-ف-ر / ش-ك-ر roots but share pattern |
| الصدق ↔ الكذب | 0.78 | Both are مصدر forms |
| الإيمان ↔ الكفر | 0.77 | Both are religious state nouns |

**Proposed Fix**: Accept that opposites may score high in embedding space. Consider moving opposites to "related" category (0.5) in relatedness benchmark, not treating them as low-similarity.

---

### Issue 3: Definition Quality Varies

**Problem**: Some definitions are canonical, others are explanatory phrases that don't match how the term is used in context.

**Problematic Pairs**:

| ID | text_a | text_b | Issue |
|----|--------|--------|-------|
| rel_013 | النفاق | إظهار الإيمان وإخفاء الكفر | Definition is a sentence, not a term |
| rel_020 | الظلم | وضع الشيء في غير موضعه | Classical definition, but not how term appears in Quran |
| rel_032 | الغضب | ثوران النفس لدفع المكروه | Technical definition, not natural language |
| rel_034 | الرياء | العمل لأجل الناس | Incomplete definition |

**Proposed Fix**: Use synonyms or short paraphrases instead of technical definitions.

---

## 20 Problematic Pairs (Proposed Changes)

### Category: same_concept → should be restructured

| ID | text_a | text_b | Current | Proposed | Reason |
|----|--------|--------|---------|----------|--------|
| rel_013 | النفاق | إظهار الإيمان وإخفاء الكفر | 1.0 | Replace with: النفاق ↔ المراءاة | Long definition causes length bias |
| rel_020 | الظلم | وضع الشيء في غير موضعه | 1.0 | Replace with: الظلم ↔ الجور | Technical definition |
| rel_026 | التوكل | الاعتماد على الله | 1.0 | Keep but verify | Reasonable pair |
| rel_032 | الغضب | ثوران النفس لدفع المكروه | 1.0 | Replace with: الغضب ↔ السخط | Technical definition |
| rel_034 | الرياء | العمل لأجل الناس | 1.0 | Replace with: الرياء ↔ المراءاة | Incomplete definition |
| rel_018 | التقوى | الخوف من الله واجتناب المعاصي | 1.0 | Replace with: التقوى ↔ الورع | Long compound definition |
| rel_038 | الأمانة | حفظ الحقوق وأداؤها | 1.0 | Replace with: الأمانة ↔ الوفاء | Long definition |
| rel_016 | الحسد | تمني زوال النعمة عن الغير | 1.0 | Replace with: الحسد ↔ الغل | Long definition |
| rel_024 | الخشوع | خضوع القلب لله | 1.0 | Keep | Reasonable length |
| rel_011 | التوبة | الرجوع إلى الله | 1.0 | Keep | Reasonable length |

### Category: opposite → may need reclassification

| ID | text_a | text_b | Current | Proposed | Reason |
|----|--------|--------|---------|----------|--------|
| rel_041 | قسوة القلب | لين القلب | 0.5 | Keep 0.5 | Correct as related, but models see as very similar |
| rel_047 | الحسد | الغيرة | 0.5 | Verify: are these really related or equivalent? | Often confused |
| rel_025 | الخشوع | الغفلة | 0.5 | Keep | Correct opposite |
| rel_031 | البخل | الكرم | 0.5 | Keep | Correct opposite |
| rel_019 | التقوى | الفسق | 0.5 | Keep | Correct opposite |
| rel_023 | الرحمة | القسوة | 0.5 | Keep | Correct opposite |
| rel_021 | الظلم | العدل | 0.5 | Keep | Correct opposite |
| rel_037 | الصدق | الكذب | 0.5 | Keep | Correct opposite |
| rel_005 | الإيمان | الكفر | 0.5 | Keep | Correct opposite |
| rel_010 | الشكر | الكفران | 0.5 | Keep | Correct opposite |

---

## Structural Recommendations

### 1. Restructure Equivalent Pairs
Replace term↔definition with term↔synonym:
```
OLD: الصبر ↔ الصبر على البلاء (1.0)
NEW: الصبر ↔ التحمل (1.0)
```

### 2. Add Verse↔Tafsir Pairs
The actual retrieval task is verse↔tafsir alignment. Add pairs like:
```
text_a: "إِنَّ اللَّهَ مَعَ الصَّابِرِينَ"
text_b: "يخبر تعالى أنه مع الصابرين بالنصر والتأييد"
similarity: 1.0
category: verse_tafsir
```

### 3. Normalize Text Length
Ensure equivalent pairs have similar text lengths to avoid structural bias.

### 4. Consider Separate Benchmarks
- **Term-term benchmark**: For vocabulary similarity
- **Verse-tafsir benchmark**: For retrieval alignment
- **Definition benchmark**: For concept understanding

---

## Action Items (Do NOT implement without approval)

1. [ ] Restructure 10 equivalent pairs to use synonyms instead of definitions
2. [ ] Add 30-50 verse↔tafsir aligned pairs
3. [ ] Re-run benchmark after restructuring
4. [ ] If still failing, consider fine-tuning on contrastive pairs

---

## Conclusion

The benchmark failure is primarily due to **structural bias** (length/format differences between categories) rather than model inadequacy. The opposite pairs scoring high is expected behavior for embedding models that capture semantic field proximity.

**Recommendation**: Do not conclude "models are unsuitable" until the benchmark is restructured to remove structural bias.
