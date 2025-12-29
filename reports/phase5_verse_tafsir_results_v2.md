# Phase 5: Verse-Tafsir Benchmark Results v2 (Hardened)

**Date**: 2025-12-25  
**Benchmark**: gold_verse_tafsir_alignment_v2.jsonl (387 pairs)  
**Corpus**: Full tafsir corpus (41,596 entries from 7 sources)  
**Purpose**: Production-aligned gating for embedding model selection

---

## Executive Summary

**Full-corpus retrieval is significantly harder than the v1 benchmark suggested.**

| Model | Recall@10 | nDCG@10 | MRR | Status |
|-------|-----------|---------|-----|--------|
| **LaBSE** | **27.2%** | **0.5349** | **0.4646** | Best overall |
| BGE-M3 | 22.8% | 0.4882 | 0.4347 | Second |
| E5-base | 4.0% | 0.1064 | 0.0785 | ❌ REJECT |

**No model meets the 70% Recall@10 threshold** on full-corpus retrieval.

---

## Benchmark Design v2 (Hardened)

### Candidate Pool
- **Full corpus**: 41,596 tafsir entries
- **Sources**: Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn, Baghawi, Muyassar
- **NOT sampled**: Every tafsir entry is a candidate for every query

### Query Set
- **50 verses** selected from verses with all 5 core sources available
- **Multi-source positives**: Each verse has 5 aligned tafsir (one per core source)

### Pair Categories
| Category | Count | Description |
|----------|-------|-------------|
| Aligned | 250 | Verse ↔ tafsir of same ayah (5 per query × 50 queries) |
| Related | 50 | Verse ↔ tafsir of nearby ayah (±5 ayat, same surah) |
| Hard unrelated (same surah) | 37 | Verse ↔ tafsir from same surah but ±50+ ayat away |
| Unrelated (diff surah) | 50 | Verse ↔ tafsir from different surah |

### Improvements over v1
1. **Full-corpus candidate pool** (41K vs 150 pairs)
2. **Multi-source positives** (5 aligned per query vs 1)
3. **Hard negatives** (same surah far away, not just different surah)
4. **Correct E5 formatting** (query: / passage: prefixes)

---

## Detailed Results

### LaBSE (sentence-transformers/LaBSE)

**Overall Metrics:**
- Recall@10: **27.2%**
- nDCG@10: 0.5349
- MRR: 0.4646

**Recall@10 per Source:**
| Source | Recall@10 |
|--------|-----------|
| Jalalayn | **62.0%** |
| Tabari | 26.0% |
| Saadi | 20.0% |
| Qurtubi | 20.0% |
| Ibn Kathir | 8.0% |

**Analysis**: Strong on Jalalayn (short, direct tafsir) but weak on Ibn Kathir (long, detailed tafsir). The model struggles with verbose explanations.

### BGE-M3 (BAAI/bge-m3)

**Overall Metrics:**
- Recall@10: 22.8%
- nDCG@10: 0.4882
- MRR: 0.4347

**Recall@10 per Source:**
| Source | Recall@10 |
|--------|-----------|
| Tabari | **48.0%** |
| Jalalayn | 24.0% |
| Saadi | 22.0% |
| Ibn Kathir | 12.0% |
| Qurtubi | 8.0% |

**Analysis**: Better on Tabari than LaBSE, but worse on Jalalayn. More balanced across sources but lower overall.

### E5-base (intfloat/multilingual-e5-base)

**Overall Metrics:**
- Recall@10: 4.0% ❌
- nDCG@10: 0.1064 ❌
- MRR: 0.0785 ❌

**Recall@10 per Source:**
| Source | Recall@10 |
|--------|-----------|
| Tabari | 12.0% |
| Saadi | 6.0% |
| Qurtubi | 2.0% |
| Jalalayn | 0.0% |
| Ibn Kathir | 0.0% |

**Analysis**: REJECT. Even with correct query/passage formatting, E5-base fails on Arabic verse-tafsir retrieval. The model cannot distinguish relevant from irrelevant tafsir.

---

## Failure Analysis

### Common Failure Patterns (LaBSE)

| Verse Ref | Verse Text (truncated) | Issue |
|-----------|------------------------|-------|
| 25:32 | وَقَالَ ٱلَّذِينَ كَفَرُوا۟ لَوْلَا نُزِّلَ عَلَيْ... | Long verse, tafsir buried in corpus |
| 5:38 | وَٱلسَّارِقُ وَٱلسَّارِقَةُ فَٱقْطَعُوٓا۟ أَيْدِيَ... | Legal verse, many similar tafsir |
| 4:88 | فَمَا لَكُمْ فِى ٱلْمُنَـٰفِقِينَ فِئَتَيْنِ... | Munafiqin topic appears in many surahs |

### Root Causes
1. **Corpus size**: 41K candidates makes retrieval hard
2. **Tafsir length variance**: Short tafsir (Jalalayn) easier to match than long (Ibn Kathir)
3. **Topic overlap**: Common themes (kufr, iman, munafiqin) appear across many ayat
4. **Lexical confounds**: Same words appear in unrelated tafsir

---

## Comparison: v1 vs v2 Benchmark

| Metric | v1 (Easy) | v2 (Hardened) | Delta |
|--------|-----------|---------------|-------|
| Corpus size | 150 pairs | 41,596 entries | +277× |
| Positives per query | 1 | 5 | +5× |
| Hard negatives | No | Yes | Added |
| LaBSE Recall@10 | 82.0% | 27.2% | -54.8% |
| BGE-M3 Recall@10 | 80.0% | 22.8% | -57.2% |

**Conclusion**: v1 benchmark was artificially easy. v2 reflects production difficulty.

---

## Recommendations

### Immediate (No model meets 70% threshold)

1. **Do NOT select "active" model yet** - neither LaBSE nor BGE-M3 meets production requirements
2. **Consider hybrid retrieval**: BM25 + embedding reranking
3. **Consider fine-tuning**: Contrastive learning on verse-tafsir pairs

### If forced to choose now

| Criterion | Winner |
|-----------|--------|
| Overall Recall@10 | LaBSE (27.2%) |
| Overall nDCG@10 | LaBSE (0.53) |
| Overall MRR | LaBSE (0.46) |
| Jalalayn retrieval | LaBSE (62%) |
| Tabari retrieval | BGE-M3 (48%) |
| Ibn Kathir retrieval | BGE-M3 (12%) |

**LaBSE is marginally better overall**, but both models need improvement for production use.

### Future Work

1. **Hybrid retrieval**: Combine BM25 lexical search with embedding reranking
2. **Fine-tuning**: Train on QBM-specific verse-tafsir pairs
3. **Chunking strategy**: Split long tafsir into smaller chunks
4. **Query expansion**: Add verse context (surah name, topic) to queries

---

## Files Generated

- `data/evaluation/gold_verse_tafsir_alignment_v2.jsonl` - Hardened benchmark (387 pairs)
- `data/evaluation/tafsir_corpus_index.json` - Full corpus index (41,596 entries)
- `data/evaluation/phase5_verse_tafsir_results_v2.json` - Full results with failures

---

## Acceptance Status

| Criterion | Threshold | LaBSE | BGE-M3 | E5-base |
|-----------|-----------|-------|--------|---------|
| Recall@10 | ≥ 70% | ❌ 27% | ❌ 23% | ❌ 4% |
| nDCG@10 | ≥ 0.60 | ❌ 0.53 | ❌ 0.49 | ❌ 0.11 |
| MRR | ≥ 0.50 | ❌ 0.46 | ❌ 0.43 | ❌ 0.08 |

**Phase 5 Gate: NOT PASSED** - No model meets production retrieval requirements on full-corpus benchmark.

**Next step**: Implement hybrid retrieval (BM25 + reranking) or fine-tune embedding model.
