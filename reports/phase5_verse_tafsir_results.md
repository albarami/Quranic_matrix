# Phase 5: Verse-Tafsir Benchmark Results (Canonical Gate)

**Date**: 2025-12-25  
**Benchmark**: gold_verse_tafsir_alignment.jsonl (150 pairs)  
**Purpose**: Production gating for embedding model selection

---

## Executive Summary

**LaBSE and BGE-M3 are both viable candidates** for QBM verse-tafsir retrieval.

| Model | Recall@10 | nDCG@10 | MRR | Spearman | Recommendation |
|-------|-----------|---------|-----|----------|----------------|
| **LaBSE** | **82.0%** | 0.6693 | 0.6216 | 0.5906 | ✅ **PRIMARY** |
| **BGE-M3** | 80.0% | **0.6725** | **0.6327** | **0.5912** | ✅ **BACKUP** |
| E5-base | 18.0% | 0.1442 | 0.1340 | 0.2057 | ❌ REJECT |

---

## Benchmark Design

### Data Sources
- **Quran**: 6,236 verses from quran-uthmani.xml
- **Tafsir**: 7 sources (Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn, Baghawi, Muyassar)
- **Total tafsir entries**: 41,596

### Pair Categories
| Category | Count | Description | Label |
|----------|-------|-------------|-------|
| Aligned | 50 | Verse ↔ tafsir of same ayah | 1.0 |
| Related | 50 | Verse ↔ tafsir of nearby ayah (same surah, ±10 ayat) | 0.5 |
| Unrelated | 50 | Verse ↔ tafsir from different surah | 0.0 |

### Tafsir Source Distribution
| Source | Count |
|--------|-------|
| Saadi | 27 |
| Muyassar | 24 |
| Tabari | 22 |
| Qurtubi | 21 |
| Baghawi | 20 |
| Jalalayn | 20 |
| Ibn Kathir | 16 |

---

## Detailed Results

### LaBSE (sentence-transformers/LaBSE)

**Relatedness Metrics:**
- Category order: ✅ PASS
- Avg aligned: 0.4812
- Avg related: 0.2737
- Avg unrelated: 0.2211
- Gap aligned-related: 0.2075 ✅
- Gap related-unrelated: 0.0526
- Spearman: 0.5906 ✅

**Retrieval Metrics:**
- Recall@10: **82.0%** ✅
- nDCG@10: 0.6693
- MRR: 0.6216

**Analysis**: Best recall, good discrimination between aligned and related/unrelated. Lower overall similarity scores but better separation.

### BGE-M3 (BAAI/bge-m3)

**Relatedness Metrics:**
- Category order: ✅ PASS
- Avg aligned: 0.5758
- Avg related: 0.4683
- Avg unrelated: 0.4367
- Gap aligned-related: 0.1075
- Gap related-unrelated: 0.0316
- Spearman: 0.5912 ✅

**Retrieval Metrics:**
- Recall@10: 80.0% ✅
- nDCG@10: **0.6725** ✅
- MRR: **0.6327** ✅

**Analysis**: Slightly lower recall but best nDCG and MRR. Higher overall similarities with good separation.

### E5-base (intfloat/multilingual-e5-base)

**Relatedness Metrics:**
- Category order: ✅ PASS (barely)
- Avg aligned: 0.8765
- Avg related: 0.8627
- Avg unrelated: 0.8617
- Gap aligned-related: 0.0138 ❌
- Gap related-unrelated: 0.0010 ❌
- Spearman: 0.2057 ❌

**Retrieval Metrics:**
- Recall@10: 18.0% ❌
- nDCG@10: 0.1442 ❌
- MRR: 0.1340 ❌

**Analysis**: REJECT. All similarities cluster around 0.86 with near-zero discrimination. The model cannot distinguish aligned from unrelated pairs.

---

## Acceptance Criteria

### Required for Phase 5 Pass
| Metric | Threshold | LaBSE | BGE-M3 | E5-base |
|--------|-----------|-------|--------|---------|
| Category order | PASS | ✅ | ✅ | ✅ |
| Recall@10 | ≥ 70% | ✅ 82% | ✅ 80% | ❌ 18% |
| Spearman | ≥ 0.35 | ✅ 0.59 | ✅ 0.59 | ❌ 0.21 |

### Recommended (not required)
| Metric | Target | LaBSE | BGE-M3 | E5-base |
|--------|--------|-------|--------|---------|
| nDCG@10 | ≥ 0.60 | ✅ 0.67 | ✅ 0.67 | ❌ 0.14 |
| MRR | ≥ 0.50 | ✅ 0.62 | ✅ 0.63 | ❌ 0.13 |

---

## Recommendation

### Primary: LaBSE
- Best Recall@10 (82%)
- Largest gap between aligned and related (0.21)
- Well-tested multilingual model with Arabic support
- Smaller model size than BGE-M3

### Backup: BGE-M3
- Best nDCG@10 and MRR
- Newer model with strong multilingual capabilities
- Slightly higher computational cost

### Reject: E5-base
- Poor discrimination (all similarities ~0.86)
- Recall@10 only 18%
- Query prefix may not be suitable for Arabic verse/tafsir

---

## Next Steps

1. **Integrate LaBSE** into Full Power system as primary embedding model
2. **Run production retrieval tests** on actual QBM queries
3. **Monitor latency** - if LaBSE is too slow, switch to BGE-M3
4. **Consider fine-tuning** if retrieval quality needs improvement

---

## Files Generated

- `data/evaluation/gold_verse_tafsir_alignment.jsonl` - Benchmark (150 pairs, 7 tafsirs)
- `data/evaluation/phase5_verse_tafsir_results.json` - Full results with details
