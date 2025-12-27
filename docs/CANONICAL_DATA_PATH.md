# QBM Canonical Data Path

## Overview

This document declares the **canonical data path** for QBM v2.0.0. All production queries MUST flow through this path. Legacy stores are deprecated and should not be used for auditable outputs.

## Canonical Truth Layer

### Primary Data Sources

| Component | Canonical Source | Format | Status |
|-----------|------------------|--------|--------|
| **Quran Text** | `data/quran/quran_uthmani.json` | JSON | ✅ Canonical |
| **Canonical Entities** | `vocab/canonical_entities.json` | JSON | ✅ Canonical |
| **Semantic Graph** | `data/graph/semantic_graph_v2.json` | JSON | ✅ Canonical |
| **Tafsir** | `data/tafsir/*.jsonl` | JSONL | ✅ Canonical |
| **Evidence Index** | `data/indices/chunked_evidence_index.json` | JSON | ✅ Canonical |

### Entity Counts (Canonical)

From `vocab/canonical_entities.json`:
- **73** behaviors
- **14** agents
- **11** organs
- **12** heart states
- **16** consequences
- **11** classification axes

### Tafsir Sources (Canonical)

7 sources from `data/tafsir/*.jsonl`:
1. Ibn Kathir (ibn_kathir.ar.jsonl)
2. Tabari (tabari.ar.jsonl)
3. Qurtubi (qurtubi.ar.jsonl)
4. Sa'di (saadi.ar.jsonl)
5. Jalalayn (jalalayn.ar.jsonl)
6. Baghawi (baghawi.ar.jsonl)
7. Muyassar (muyassar.ar.jsonl)

## Canonical API Path

```
User Query
    │
    ▼
POST /api/proof/query
    │
    ├─► Deterministic Retrieval (BM25 + Evidence Index)
    │   └─► chunk_id + char_start/char_end provenance
    │
    ├─► Semantic Graph (semantic_graph_v2.json)
    │   └─► Evidence-backed edges with confidence scores
    │
    └─► Canonical Entities (canonical_entities.json)
        └─► 73 behaviors, 14 agents, etc.
```

### Provenance Requirements

All evidence in the canonical path includes:
- `chunk_id`: Unique identifier (e.g., `ibn_kathir_2_155_001`)
- `verse_key`: Surah:Ayah reference (e.g., `2:155`)
- `char_start`: Character offset start in source
- `char_end`: Character offset end in source
- `quote`: Verbatim text from source

## Deprecated Stores

The following stores are **DEPRECATED** and should NOT be used for production or auditable outputs:

### 1. Chroma Vector Store

**Status:** DEPRECATED - Not used in canonical path

**Files:**
- `chroma_db/` directory
- Any `qbm_ayat`, `qbm_tafsir` collections

**Reason:** Empty collections, inconsistent with canonical entities, not evidence-grounded.

**Migration:** Use deterministic retrieval + BM25 + evidence index instead.

### 2. Legacy Behavior ID Systems

**Status:** DEPRECATED

**Files:**
- Any files using 46-behavior or 33-behavior ID systems
- `data/annotations/*` files with non-canonical IDs

**Reason:** Superseded by `vocab/canonical_entities.json` (73 behaviors).

**Migration:** Map to canonical entity IDs or regenerate from canonical source.

### 3. Legacy Silver Export

**Status:** DEPRECATED

**Files:**
- `data/exports/qbm_silver_20251221.json`

**Reason:** Lacks token/char offsets, uses non-canonical entity IDs.

**Migration:** Regenerate from canonical entities + evidence index with full provenance.

### 4. Legacy SQLite Tafsir DB

**Status:** DEPRECATED for production

**Files:**
- `data/tafsir/tafsir.db`
- `data/tafsir/tafsir_cleaned.db`

**Reason:** HTML encoding issues, inconsistent coverage.

**Canonical Alternative:** Use `data/tafsir/*.jsonl` files directly.

### 5. Legacy NPY Indexes

**Status:** DEPRECATED for auditable use

**Files:**
- `*.npy` embedding files without stable IDs

**Reason:** Positional indexing without stable provenance.

**Use:** Internal acceleration only, not for audit-grade outputs.

## Runtime Enforcement

### CI Tests

The following CI tests enforce the canonical path:

1. **No Chroma in Prod Path**
   ```python
   # tests/test_canonical_path.py
   def test_proof_query_does_not_use_chroma():
       # Asserts /api/proof/query does not instantiate Chroma
   ```

2. **Canonical Entity Counts**
   ```python
   # tests/test_genome_q25.py
   def test_behaviors_count_is_73():
       # Asserts exactly 73 behaviors from canonical source
   ```

3. **Evidence Provenance**
   ```python
   # tests/test_genome_q25.py
   def test_edges_have_required_fields():
       # Asserts all edges have chunk_id, confidence, evidence_count
   ```

### Runtime Flags

| Flag | Default | Description |
|------|---------|-------------|
| `USE_CANONICAL_PATH` | `true` | Force canonical data path |
| `ALLOW_LEGACY_STORES` | `false` | Allow deprecated stores (dev only) |
| `REQUIRE_PROVENANCE` | `true` | Require chunk_id on all evidence |

## Migration Guide

### For Developers

1. **Do NOT** import from deprecated stores in new code
2. **Do** use canonical sources for all new features
3. **Do** include provenance (chunk_id, char offsets) in all evidence

### For Data Scientists

1. **Do NOT** use legacy exports for analysis
2. **Do** regenerate datasets from canonical sources
3. **Do** validate entity IDs against `canonical_entities.json`

### For Auditors

1. **Verify** all evidence traces to canonical sources
2. **Check** chunk_id and char offsets are present
3. **Confirm** entity counts match canonical registry

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025-12-27 | Initial canonical path declaration |

## Contact

For questions about the canonical data path, contact the QBM development team.
