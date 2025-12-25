# QBM Master Remediation Plan

**Date:** 2025-12-25  
**Owner:** AI Coder (Implementation)  
**Reviewer/Approver:** Salim Al-Barami  
**Scope:** Backend (FastAPI + ML/RAG/Graph), Data Pipeline, Tests/CI, Frontend  
**Target:** Transform QBM from 6.2/10 → 8.5+/10 production-ready system  
**Timeline:** 20 working days

---

## Approval Checklist (Salim)

Before implementation begins, confirm:

- [ ] The plan explicitly prevents "system works only because fallback patched it"
- [ ] Each phase has pass/fail gates + acceptance checks
- [ ] Retrieval redesign guarantees 5 tafsir sources via primary retrieval (not patch)
- [ ] Graph semantics are corrected (co-occurrence ≠ causation; no random UI weights)
- [ ] Data pipeline includes provenance (offsets, extractor version, confidence)
- [ ] Embedding model fix targets 75%+ accuracy (not 50% random chance)
- [ ] CI ensures regressions are blocked
- [ ] Every phase ends with commit + push
- [ ] Manual verification required alongside automated tests

**Signature:** __________________ **Date:** __________________

---

## Current State Analysis

After examining the codebase, I've confirmed the following issues align with the audit findings:

| Issue | Location | Current State |
|-------|----------|---------------|
| API Monolith | [`src/api/main.py`](src/api/main.py) | 2404 lines (vs 500 max) |
| CORS Permissive | [`src/api/main.py:103`](src/api/main.py) | `allow_origins=["*"]` |
| No Rate Limiting | [`src/api/main.py`](src/api/main.py) | None implemented |
| Random UI Weights | [`qbm-frontendv3/src/app/discovery/page.tsx:136`](qbm-frontendv3/src/app/discovery/page.tsx) | `Math.random() * 5 + 1` |
| No CI/CD | `.github/workflows/` | Directory doesn't exist |
| No pytest.ini | Project root | File doesn't exist |
| Unpinned deps | [`requirements.txt`](requirements.txt) | Uses `>=` not `==` |
| Embedding fallback | [`src/ai/vectors/qbm_vectors.py:33`](src/ai/vectors/qbm_vectors.py) | Falls back to multilingual model |

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph phase0_1 [Phase 0-1: Foundation]
        Instrumentation[Fallback Instrumentation]
        CI[CI/CD Setup]
        Tests[Test Hygiene]
    end
    
    subgraph phase2_3 [Phase 2-3: Security and Data]
        Security[SQL Params + Rate Limit + CORS]
        DataClean[Text Cleaning + Provenance]
    end
    
    subgraph phase4_5 [Phase 4-5: Core ML]
        Retrieval[Stratified Retrieval + Hybrid Search]
        Embeddings[Arabic Embedding Model 75%+ Accuracy]
    end
    
    subgraph phase6_7 [Phase 6-7: Architecture]
        Graph[Split Co-occurrence vs Semantic]
        API[Modular Routers]
    end
    
    subgraph phase8_9 [Phase 8-9: Integration]
        Frontend[Real Backend Data]
        Docs[Documentation + v2.0.0]
    end
    
    phase0_1 --> phase2_3 --> phase4_5 --> phase6_7 --> phase8_9
```

---

## Implementation Phases

### Phase 0: Baseline and Instrumentation (Day 1)

**Root Cause Addressed:** Cannot distinguish "main path works" from "fallback patched it"

Add `fallback_used` tracking to proof endpoint and create baseline report.

**Key Changes:**
- Add `ProofDebug` schema with `fallback_used`, `fallback_reasons`, `retrieval_distribution`
- Create `tests/conftest.py` with `assert_no_fallback()` helper
- Generate `reports/baseline_2025-12-25.md`

**Deliverables:**
```python
class ProofDebug(BaseModel):
    fallback_used: bool = False
    fallback_reasons: List[str] = []
    retrieval_distribution: Dict[str, int] = {}
    primary_path_latency_ms: int = 0
    index_source: Literal["disk", "runtime_build"] = "disk"
```

**Acceptance Criteria:**
- [x] Proof endpoint includes `debug` section
- [x] Can programmatically detect when fallback was used
- [x] Baseline report documents current fallback usage rate

**Git Commit:**
```bash
git add .
git commit -m "phase(0): add no-fallback instrumentation and baseline report"
git push origin main
```

---

### Phase 1: Test Hygiene and CI (Day 2)

**Root Cause Addressed:** Unreliable test collection, non-reproducible builds, no CI enforcement

**Key Changes:**
- Create `pytest.ini` with proper markers and test paths
- Pin all dependencies in `requirements.txt` to exact versions
- Create `.github/workflows/ci.yml` for automated testing
- Create mini-fixtures for CPU-only testing

**Acceptance Criteria:**
- [x] `pytest --collect-only` shows 0 errors
- [x] `pip install -r requirements.txt` succeeds on fresh environment
- [x] GitHub Actions CI passes
- [x] Mini-fixture allows CPU-only testing

**Git Commit:**
```bash
git add .
git commit -m "phase(1): stabilize pytest + CI workflow + pin deps"
git push origin main
```

---

### Phase 2: Security Hardening (Day 3)

**Root Cause Addressed:** SQL injection, no auth, permissive CORS, no input validation

**Key Changes in [`src/api/main.py`](src/api/main.py):**
- Fix CORS: Replace `["*"]` with `ALLOWED_ORIGINS` from env var
- Add rate limiting with slowapi (30/minute on `/api/proof`)
- Add input validation schemas with Pydantic
- Standardize error responses (no bare `except: return {}`)

**Note:** Graph module ([`src/ai/graph/qbm_graph.py`](src/ai/graph/qbm_graph.py)) already uses proper parameterized queries.

**Acceptance Criteria:**
- [x] Rate limit triggers after 30 requests/minute
- [x] Malformed inputs return 400, not 500
- [x] CORS does not allow arbitrary origins in prod

**Git Commit:**
```bash
git add .
git commit -m "phase(2): security hardening - rate limit + CORS + validation"
git push origin main
```

---

### Phase 3: Data Cleaning and Provenance (Days 4-5)

**Root Cause Addressed:** HTML in embeddings, false positive extraction, no provenance

**New Files:**
- `src/preprocessing/text_cleaner.py` - HTML stripping, Arabic normalization
- `scripts/clean_all_tafsir.py` - Re-process all tafsir data
- `src/extraction/tafsir_extractor_v2.py` - Morphology-based extraction using camel-tools

**Provenance Schema:**
```python
class AnnotationWithProvenance(BaseModel):
    text: str
    behavior: str
    surah: int
    ayah: int
    tafsir_source: str
    char_start: int
    char_end: int
    source_file: str
    extractor_version: str = "2.0.0"
    confidence: float
    extracted_at: datetime
```

**Acceptance Criteria:**
- [ ] HTML contamination rate < 1% (was ~50%)
- [ ] Known false positives eliminated
- [ ] All records have provenance fields

---

### Phase 4: Retrieval Redesign (Days 6-8)

**Root Cause Addressed:** Source collapse, runtime builds, no hybrid search

**Key Changes:**
- Split indexes by source: `data/indexes/tafsir/{ibn_kathir,tabari,qurtubi,saadi,jalalayn}.idx`
- Implement `StratifiedTafsirRetriever` guaranteeing 5 results per source
- Add `HybridRetriever` combining BM25 + semantic search with RRF fusion
- Remove runtime index builds - fail fast if indexes missing at startup

**Hard Rule:** If the system needs a "fill missing sources" mechanism during normal operation, the design is broken.

**Acceptance Criteria:**
- [ ] `fallback_used = false` for standard queries
- [ ] All 5 tafsir sources have >= 5 results in distribution
- [ ] No "need N more from source X" logs during normal operation
- [ ] Hybrid search finds exact Arabic terms

---

### Phase 5: Embedding Model Fix (Days 9-11)

**Root Cause Addressed:** Model at random chance (50%), not suited for Classical Arabic

**Key Changes:**
- Create evaluation benchmark at `data/evaluation/semantic_similarity_gold.jsonl`
- Evaluate: CAMeL-Lab/bert-base-arabic-camelbert-ca vs current model
- Fine-tune on Quranic corpus using contrastive learning
- Add model registry at `data/models/registry.json`

**Acceptance Criteria:**
- [ ] Embedding accuracy >= 75% (was 50%)
- [ ] "الصبر" query returns patience verses in top 5
- [ ] Opposite behaviors have similarity < 0.5
- [ ] Model registry determines which model is used

---

### Phase 6: Graph Semantics Fix (Days 12-14)

**Root Cause Addressed:** Co-occurrence treated as causation, random UI weights, inconsistent taxonomy

**Key Changes to [`src/ai/graph/qbm_graph.py`](src/ai/graph/qbm_graph.py):**
- Split into `CO_OCCURRENCE` vs `SEMANTIC` graph types
- `SemanticEdge` requires: relation type, evidence, verse_ref, confidence
- Causal queries must reject co-occurrence edges

**Key Changes to [`qbm-frontendv3/src/app/discovery/page.tsx`](qbm-frontendv3/src/app/discovery/page.tsx):**
- Remove `Math.random()` weights (lines 136-150)
- Fetch real graph data from `/api/graph/semantic`

**Acceptance Criteria:**
- [ ] Causal queries use semantic graph only
- [ ] Co-occurrence explicitly labeled as non-causal
- [ ] Taxonomy has exactly 73 behaviors (or documented count)
- [ ] Frontend shows real backend data, no random weights

---

### Phase 7: API Refactoring (Days 15-16)

**Root Cause Addressed:** Monolithic API, mixed concerns, inconsistent errors

**Split [`src/api/main.py`](src/api/main.py) (2404 lines) into:**

```
src/api/
├── main.py              # < 100 lines - app init only
├── routers/
│   ├── verses.py        # /api/verses/*
│   ├── tafsir.py        # /api/tafsir/*
│   ├── behaviors.py     # /api/behaviors/*
│   ├── analysis.py      # /api/analysis/*
│   ├── graph.py         # /api/graph/*
│   ├── proof.py         # /api/proof/*
│   └── health.py        # /health, /ready
├── middleware/
│   ├── rate_limit.py
│   └── logging.py
├── schemas/
│   ├── requests.py
│   ├── responses.py
│   └── debug.py
└── dependencies/
    └── common.py
```

**Acceptance Criteria:**
- [ ] `main.py` under 100 lines
- [ ] All routers under 300 lines
- [ ] Health and ready endpoints work
- [ ] Caching reduces repeated query times

---

### Phase 8: Frontend Integration (Days 17-18)

**Root Cause Addressed:** UI disconnected from backend reality, navigation confusion

**Key Changes:**
- Fix environment handling in frontend config
- Connect all visualizations to real backend data
- Add deep links between Proof, Explorer, Taxonomy
- Create E2E tests with Playwright

**Acceptance Criteria:**
- [ ] Frontend builds without errors
- [ ] Graph shows real backend data
- [ ] Taxonomy browser works
- [ ] Deep links function correctly
- [ ] E2E test passes

---

### Phase 9: Documentation and Release (Days 19-20)

**Root Cause Addressed:** Drift between docs and reality, deployment uncertainty

**Create:**
- `docs/ARCHITECTURE.md` - System diagram, data flows
- `docs/API.md` - All endpoints documented
- `docs/METHODOLOGY.md` - Bouzidani framework explanation
- `docs/DEPLOYMENT.md` - Production checklist
- `docs/KNOWN_LIMITATIONS.md` - Research vs production disclaimer

**Tag:** `v2.0.0`

**Acceptance Criteria:**
- [ ] All documentation files exist
- [ ] README reflects actual system capabilities
- [ ] Deployment guide tested on fresh environment
- [ ] Known limitations clearly stated

---

## Mandatory Invariant Checks

These checks must **always pass** after any phase:

### No-Fallback Invariant
```python
def test_no_fallback_invariant(strict_client):
    """Standard queries must not use fallback"""
    queries = ["ما هو الصبر؟", "ما هو الكبر؟", "ما هي التقوى؟"]
    for query in queries:
        response = strict_client.post("/api/proof", json={"query": query})
        assert response.json()["debug"]["fallback_used"] == False
```

### Source Distribution Invariant
```python
def test_source_distribution_invariant():
    """All 5 tafsir sources must have >= min_per_source"""
    response = client.post("/api/proof", json={"query": "test"})
    dist = response.json()["debug"]["retrieval_distribution"]
    
    for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]:
        assert dist.get(source, 0) >= 5
```

### Graph Integrity Invariant
```python
def test_graph_integrity_invariant():
    """Causal reasoning must use semantic graph only"""
    graph = QBMGraph("data/graph.db")
    
    for u, v, d in graph.semantic.edges(data=True):
        assert 'relation' in d
        assert 'evidence' in d
        assert d['relation'] in ['CAUSES', 'PREVENTS', 'OPPOSITE_OF', 'LEADS_TO']
```

### Data Cleanliness Invariant
```python
def test_data_cleanliness_invariant():
    """Indexed data must be clean"""
    with open("data/annotations/tafsir_annotations_clean.jsonl") as f:
        html_count = 0
        total = 0
        for line in f:
            total += 1
            record = json.loads(line)
            if '<' in record.get('context', ''):
                html_count += 1
        
        contamination_rate = html_count / total
        assert contamination_rate < 0.01  # < 1%
```

---

## Timeline Summary

| Phase | Days | Focus | Key Deliverable |
|-------|------|-------|-----------------|
| 0 | 1 | Baseline + Instrumentation | `fallback_used` tracking |
| 1 | 1 | Test Hygiene + CI | GitHub Actions, pinned deps |
| 2 | 1 | Security | SQL params, rate limits, CORS |
| 3 | 2 | Data Cleaning | HTML strip, morphology extraction |
| 4 | 3 | Retrieval Redesign | Stratified retrieval, hybrid search |
| 5 | 3 | Embedding Model | 75%+ accuracy, model registry |
| 6 | 3 | Graph Semantics | Split graphs, unified taxonomy |
| 7 | 2 | API Refactoring | Modular routers, caching |
| 8 | 2 | Frontend Integration | Real data, E2E tests |
| 9 | 2 | Documentation | Complete docs, v2.0.0 tag |
| **Total** | **20** | | **Production-ready system** |

---

## Success Criteria

After all phases complete, the system must:

1. **Score 8.5+/10** on re-audit
2. **Zero critical issues**
3. **`fallback_used = false`** for all standard queries
4. **75%+ embedding accuracy** (was 50%)
5. **All 5 tafsir sources** represented in every proof
6. **No random weights** in any visualization
7. **Full CI pipeline** passing
8. **Complete documentation** matching reality

---

## Phase Todos

- [x] **Phase 0:** Add fallback instrumentation to proof endpoint + create baseline report
- [x] **Phase 1:** Create pytest.ini, pin deps, add GitHub Actions CI workflow
- [x] **Phase 2:** Security: Fix CORS, add rate limiting, input validation
- [ ] **Phase 3:** Data cleaning pipeline + morphology extraction + provenance
- [ ] **Phase 4:** Stratified retrieval + hybrid search + prebuilt indexes
- [ ] **Phase 5:** Evaluate/fine-tune Arabic embedding model to 75%+ accuracy
- [ ] **Phase 6:** Split graph into co-occurrence vs semantic + fix frontend random weights
- [ ] **Phase 7:** Refactor API into modular routers (each < 300 lines)
- [ ] **Phase 8:** Frontend integration + E2E tests + deep links
- [ ] **Phase 9:** Documentation + v2.0.0 release tag

---

*Plan created: December 25, 2025*  
*Based on: 3 independent audit reports + codebase analysis*  
*Target: QBM v1.0 (6.2/10) → QBM v2.0 (8.5+/10)*

