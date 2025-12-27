# Deprecated Artifacts

This document lists artifacts that are **DEPRECATED** as of QBM v2.0.0. These should not be used for production or auditable outputs.

## Status Legend

| Status | Meaning |
|--------|---------|
| 🔴 DEPRECATED | Do not use in production |
| 🟡 LEGACY | May exist for backward compatibility |
| 🟢 CANONICAL | Use this instead |

---

## Data Stores

### Chroma Vector Store

**Status:** 🔴 DEPRECATED

**Location:** `chroma_db/`

**Reason:** Empty collections, not evidence-grounded, inconsistent with canonical entities.

**Canonical Alternative:** 🟢 Deterministic retrieval + BM25 + `data/indices/chunked_evidence_index.json`

---

### Legacy Silver Export

**Status:** 🔴 DEPRECATED

**Location:** `data/exports/qbm_silver_20251221.json`

**Reason:** 
- Lacks token/char offsets for provenance
- Uses non-canonical entity IDs (46/33 systems)
- Not regenerated from canonical sources

**Canonical Alternative:** 🟢 Use `/api/genome/export` endpoint which uses `vocab/canonical_entities.json`

---

### Legacy SQLite Tafsir Databases

**Status:** 🟡 LEGACY (dev only)

**Location:** 
- `data/tafsir/tafsir.db`
- `data/tafsir/tafsir_cleaned.db`

**Reason:**
- HTML encoding issues
- Inconsistent coverage across sources
- Not the canonical tafsir substrate

**Canonical Alternative:** 🟢 `data/tafsir/*.jsonl` files (7 sources)

---

### Legacy NPY Embedding Indexes

**Status:** 🟡 LEGACY (internal acceleration only)

**Location:** `*.npy` files without stable IDs

**Reason:**
- Positional indexing without stable provenance
- Cannot trace back to source documents

**Use:** Internal acceleration layer only, NOT for audit-grade outputs.

**Canonical Alternative:** 🟢 Chunked evidence index with `chunk_id` provenance

---

## Entity ID Systems

### 46-Behavior System

**Status:** 🔴 DEPRECATED

**Location:** Various `data/annotations/*` files

**Reason:** Superseded by canonical 73-behavior registry.

**Canonical Alternative:** 🟢 `vocab/canonical_entities.json` (73 behaviors)

---

### 33-Behavior System

**Status:** 🔴 DEPRECATED

**Location:** Legacy annotation files

**Reason:** Incomplete, superseded by canonical registry.

**Canonical Alternative:** 🟢 `vocab/canonical_entities.json` (73 behaviors)

---

## Code Modules

### Chroma RAG Module

**Status:** 🔴 DEPRECATED

**Files:** Any code importing `chromadb` for production queries

**Reason:** 
- Empty collections in production
- Not evidence-grounded
- Bypassed by deterministic retrieval

**Canonical Alternative:** 🟢 Use `src/api/routers/proof.py` deterministic retrieval path

---

### FullPower len>5 Filter (FIXED)

**Status:** 🟢 FIXED in v2.0.0

**Issue:** Previously excluded muqattaʿāt verses (الم، طه، يس، etc.)

**Fix:** Removed `len(verse_text) > 5` filter in `src/ml/full_power_system.py`

**Verification:** `tests/test_canonical_path.py::TestQuranIndexCompleteness`

---

## Migration Guide

### For Developers

1. **Do NOT** import from deprecated stores
2. **Do NOT** use 46/33 behavior ID systems
3. **Do** use canonical sources from `vocab/canonical_entities.json`
4. **Do** include provenance (chunk_id, char offsets) in all evidence

### For Data Scientists

1. **Do NOT** use `qbm_silver_20251221.json` for analysis
2. **Do** regenerate datasets from `/api/genome/export`
3. **Do** validate entity IDs against canonical registry

### For Auditors

1. **Verify** evidence traces to canonical sources
2. **Check** for deprecated store references in code
3. **Confirm** entity counts match canonical registry (73 behaviors, 14 agents, etc.)

---

## Deprecation Timeline

| Version | Date | Action |
|---------|------|--------|
| v2.0.0 | 2025-12-27 | Deprecated stores documented |
| v2.1.0 | TBD | Runtime warnings for deprecated imports |
| v3.0.0 | TBD | Remove deprecated stores from repo |

---

## Questions?

Contact the QBM development team for migration assistance.
