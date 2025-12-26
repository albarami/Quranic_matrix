# Phase 5.5: Query Router + Deterministic Evidence Retrieval

**Date**: 2025-12-26  
**Purpose**: Implement production architecture with query routing and deterministic retrieval

---

## Executive Summary

**Query Router + Deterministic Retrieval achieves 100% 5/5 core source coverage for AYAH_REF queries.**

| Query Type | Retrieval Mode | 5/5 Coverage |
|------------|----------------|--------------|
| **AYAH_REF** | Deterministic | **100%** |
| SURAH_REF | Deterministic | 100% |
| CONCEPT_REF | Hybrid | Variable |
| FREE_TEXT | Hybrid | Variable |

---

## Architecture

### Query Router

Classifies queries into 4 intents:

| Intent | Pattern | Example |
|--------|---------|---------|
| **AYAH_REF** | Verse reference | `2:255`, `البقرة:255` |
| **SURAH_REF** | Surah reference | `سورة البقرة`, `الفاتحة` |
| **CONCEPT_REF** | Behavior/vocab | `الصبر`, `BEH_SABR`, `آيات التقوى` |
| **FREE_TEXT** | Open question | `كيف أتعامل مع الابتلاء` |

### Retrieval Strategy

| Intent | Strategy | Coverage Guarantee |
|--------|----------|-------------------|
| AYAH_REF | Deterministic lookup by verse key | 5/5 by design |
| SURAH_REF | Deterministic lookup by surah | 5/5 by design |
| CONCEPT_REF | Hybrid (BM25 + source-aware) | Best effort |
| FREE_TEXT | Hybrid (BM25 + source-aware) | Best effort |

---

## Results

### AYAH_REF (Verse Reference Queries)

| Metric | Before (Phase 5) | After (Phase 5.5) |
|--------|------------------|-------------------|
| Normalized Recall@20 | 85.6% | **100%** |
| 5/5 Coverage Rate | 46% | **100%** |
| Per-source Hit Rate | 58-100% | **100% all sources** |

### Per-Source Coverage (AYAH_REF)

| Source | Hit Rate |
|--------|----------|
| Ibn Kathir | 100% |
| Tabari | 100% |
| Qurtubi | 100% |
| Saadi | 100% |
| Jalalayn | 100% |

---

## Why This Works

### 1. Deterministic Path Guarantees Coverage

For AYAH_REF queries:
1. Router extracts verse key (e.g., `2:255`)
2. Retriever fetches ALL chunks for that verse from evidence index
3. Source-aware selection ensures min_per_source from each core source
4. **5/5 coverage is guaranteed by design**

### 2. No Embedding Dependency for Structured Queries

- AYAH_REF and SURAH_REF use deterministic lookup
- No BM25 or dense retrieval needed
- Reproducible, explainable results

### 3. Hybrid Only for FREE_TEXT

- Open questions use BM25 + source-aware selection
- Embeddings available as optional reranker
- No pretense of determinism for semantic queries

---

## Files Created

| File | Description |
|------|-------------|
| `src/ml/query_router.py` | Query intent classification |
| `src/ml/routed_evidence_retriever.py` | Intent-based retrieval dispatch |
| `tests/test_query_router.py` | 18 tests for router and retrieval |
| `data/evaluation/phase5_5_routed_results.json` | Evaluation results |

---

## Tests Passing

| Test Suite | Tests | Status |
|------------|-------|--------|
| test_query_router.py | 18 | ✅ All passed |

### Key Tests

- `test_ayah_ref_full_coverage`: AYAH_REF achieves 5/5 coverage
- `test_no_false_surah_match`: Behavior terms don't match single-letter surahs
- `test_ayah_ref_coverage_gate`: ≥85% coverage on benchmark verses

---

## Comparison: Before vs After

| Metric | Phase 5 (Hybrid only) | Phase 5.5 (Routed) |
|--------|----------------------|-------------------|
| AYAH_REF 5/5 Coverage | 46% | **100%** |
| Retrieval Mode | Always hybrid | Intent-based |
| Determinism | No | Yes for structured queries |
| Explainability | Limited | Full provenance |

---

## Next Steps

1. **Integrate into Proof System**: Wire `RoutedEvidenceRetriever` into `/api/proof/query`
2. **Concept Mappings**: Build precomputed concept → ayat mappings for CONCEPT_REF
3. **Graph Rebuild**: Use evidence offsets for typed edge creation
