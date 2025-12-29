# Phase 5: Term-Term Benchmark Results (Sanity Layer)

**Date**: 2025-12-25  
**Benchmark**: gold_relatedness_term_term.jsonl (60 pairs)  
**Purpose**: Sanity check for embedding model semantic discrimination

---

## Status: SANITY LAYER ONLY

This benchmark validates that models can distinguish between:
- Equivalent terms (synonyms)
- Related terms (antonyms, complementary)
- Unrelated terms (different semantic field)

**This is NOT the gating benchmark.** See `phase5_verse_tafsir_results.md` for the canonical Phase 5 gate.

---

## Results Summary

| Model | Cat Order | Avg Equiv | Avg Related | Avg Unrel | Spearman | Sanity |
|-------|-----------|-----------|-------------|-----------|----------|--------|
| **LaBSE** | ✅ PASS | 0.5667 | 0.5318 | 0.4129 | 0.3898 | ✅ PASS |
| BGE-M3 | ✅ PASS | 0.5818 | 0.5476 | 0.5037 | 0.3056 | ✅ PASS |
| mpnet-base-v2 | ✅ PASS | 0.6686 | 0.6215 | 0.5537 | 0.2306 | ✅ PASS |
| E5-base | ✅ PASS | 0.8420 | 0.8339 | 0.8248 | 0.1781 | ✅ PASS |
| MiniLM-L12-v2 | ✅ PASS | 0.6842 | 0.6722 | 0.6321 | 0.0667 | ✅ PASS |

**All 5 models pass sanity check** (category order holds).

---

## Interpretation

1. **Category order passing** confirms the benchmark is structure-neutral and models behave correctly
2. **LaBSE has best discrimination** with largest gaps and highest Spearman
3. **E5-base has poor discrimination** - all categories cluster around 0.82-0.84
4. This benchmark does NOT measure retrieval quality - see verse-tafsir benchmark

---

## Conclusion

Term-term sanity layer: **ALL MODELS PASS**

Proceed to verse-tafsir benchmark for production gating decision.
